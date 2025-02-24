# using MKL
using NeuralPDE, Lux, LuxCUDA, Optimization, OptimizationOptimisers
using LinearAlgebra, Random, Printf, ComponentArrays, JLD2
using ModelingToolkit: Interval, infimum, supremum

const c_dev = cpu_device()
const g_dev = gpu_device()

CUDA.reclaim()

# problem parameters
const R = 0.25f0
const v = 0.6f0
const ω_max = 1.1f0

const t_max = 0.0f0
const t_min = -1.0f0
const x_max = 1.0f0
const x_min = -1.0f0
const y_max = 1.0f0
const y_min = -1.0f0
const θ_max = Float32(pi)
const θ_min = -Float32(pi)

@views function in_norm(x)
    return   [        (x[1:1, :] .- t_min) / (t_max - t_min);
                2f0 * (x[2:2, :] .- x_min) / (x_max - x_min) .- 1f0;
                2f0 * (x[3:3, :] .- y_min) / (y_max - y_min) .- 1f0;
                2f0 * (x[4:4, :] .- θ_min) / (θ_max - θ_min) .- 1f0]
end

function model_construction(n_in, n_out; l_hidden=3, n_hidden=512)
    model = Lux.Chain(
        WrappedFunction(in_norm),
        Dense(n_in, n_hidden, sin),
        [Dense(n_hidden, n_hidden, sin) for i in 1:l_hidden],
        Dense(n_hidden, n_out)
        # WrappedFunction(Base.Fix1(broadcast, relu))
    )
    return model
end

function train_model!(model, ps, st, x_data, y_data; max_iter=5000, lr=1f-3)
    train_state = Lux.Training.TrainState(model, ps, st, OptimizationOptimisers.Adam(lr))

    for iter in 1:max_iter
        _, loss, _, train_state = Lux.Training.single_train_step!(
            AutoZygote(), MSELoss(),
            (x_data, y_data), train_state
        )
        if iter % 100 == 1
            @printf "Iteration: %04d \t Loss: %10.9g\n" iter loss
        end
        if loss < 1.0f-15
            break
        end
    end

    return model, ps, st
end

function model_prediction_CPU(in_pred; ps=nothing, seed=0, n_hidden=512)
    # seeding
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    # model construction
    model = model_construction(4, 1; n_hidden=n_hidden)

    # load learned parameters
    if isnothing(ps)
        ps_learned, st_learned = load("res_pretraining.jld2", "ps", "st")
    else
        ps_learned = ps
        _, st_learned = Lux.setup(rng, model)
        # const _st = st
    end

    # model evaluation
    # y_pred, st = Lux.apply(model, in_pred, ps, st)
    
    return model(in_pred, ps_learned, st_learned)[1]
end

function main_pretraining(; seed=0, n_hidden=512, n_sample=65000)
    # Seeding
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    # model construction
    model = model_construction(4, 1; n_hidden=n_hidden)

    # Parameter and State Variables
    ps, st = Lux.setup(rng, model) |> g_dev

    # Finite Sample Input Generation
    ts = repeat([0.0f0], 1, n_sample)
    xs = rand(rng, Float32, (1, n_sample)) * (x_max - x_min) .+ x_min
    ys = rand(rng, Float32, (1, n_sample)) * (y_max - y_min) .+ y_min
    θs = rand(rng, Float32, (1, n_sample)) * (θ_max - θ_min) .+ θ_min

    in_data = [ts; xs; ys; θs] |> g_dev
    out_data = zeros(Float32, 1, n_sample) |> g_dev

    train_model!(model, ps, st, in_data, out_data; max_iter=30000, lr=0.001f0)
    train_model!(model, ps, st, in_data, out_data; max_iter=10000, lr=0.0001f0)
    train_model!(model, ps, st, in_data, out_data; max_iter=1000, lr=0.00001f0)

    (ps, st) = (ps, st) |> c_dev
    save("res_pretraining.jld2", "ps", ps, "st", st)
    return ps
end


function main_DeepReach(; seed=0, max_iter=10^5, lr=1.0f-3, n_hidden=512, n_sample=65000, ps0=nothing)
    # seeding
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    # problem definition
    @parameters t, x, y, θ
    @variables U(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dy = Differential(y)
    Dθ = Differential(θ)

    l(x, y) = sqrt(x^2 + y^2) - R
    Value = l(x, y) + t * U(t, x, y, θ)
    Hamiltonian = Dx(Value) * v * cos(θ) + Dy(Value) * v * sin(θ) + abs(Dθ(Value)) * ω_max

    eqs = 0.0 ~ min(l(x, y) - Value, Dt(Value) + Hamiltonian)
    bcs = [0.0 ~ U(0.0, x, y, θ)]

    domains = [t ∈ Interval(t_min, t_max), x ∈ Interval(x_min, x_max), y ∈ Interval(y_min, y_max), θ ∈ Interval(θ_min, θ_max)]

    # model construction
    model = model_construction(4, 1; n_hidden=n_hidden)

    # model initialisation
    if isnothing(ps0)
        ps = Lux.setup(rng, model)[1]
    else
        ps = ps0
    end
    ps = ps |> ComponentArray |> g_dev  #.|> Float64

    # neural PDE problem setup
    # strategy = StochasticTraining(n_sample)
    strategy = QuasiRandomTraining(n_sample; resampling=false, minibatch=1)
    discretization = PhysicsInformedNN(model, strategy; init_params=ps)
    @named pdesystem = PDESystem(eqs, bcs, domains, [t, x, y, θ], [U(t, x, y, θ)])
    # num_prob = discretize(pdesystem, discretization)
    sym_prob = symbolic_discretize(pdesystem, discretization)

    # phi = sym_prob.phi
    # full_loss_function = sym_prob.loss_functions.full_loss_function
    pde_loss_functions = sym_prob.loss_functions.pde_loss_functions
    # bc_loss_functions = sym_prob.loss_functions.bc_loss_functions

    callback = function (opt_state, loss_val)
        # if opt_state.iter % 50 == 1 # || opt_state.iter == max_iter
            @printf "Iteration: %04d \t Loss: %10.9g\n" opt_state.iter loss_val
        # end

        if opt_state.iter % 2 == 1 # || opt_state.iter == max_iter
        # CUDA.pool_status()
            GC.gc(true)
        end

        # println("loss: ", loss_val)
        # println("pde_losses: ", map(l_ -> l_(opt_state.u), pde_loss_functions))
        # println("bcs_losses: ", map(l_ -> l_(opt_state.u), bc_loss_functions))
        # println("\n")
        return false
    end

    loss_function(u, p) = CUDA.sum(l_ -> l_(u), pde_loss_functions) |> g_dev
    f_ = OptimizationFunction(loss_function, AutoZygote())

    prob = OptimizationProblem(f_, ps)  # ps == sym_prob.flat_init_params
    res = solve(prob, OptimizationOptimisers.Adam(lr); maxiters=max_iter, callback)

    # prob = remake(prob, u0=res.u)
    # res = solve(prob, LBFGS(linesearch=BackTracking()); maxiters=100, callback)

    # save("res_DeepReach_NeuralPDE.jld2", "ps", res.u)
    save("res_DeepReach_NeuralPDE.jld2", "prob", prob, "res", res, "ps", res.u)
    return prob, res
end

if isfile("res_pretraining.jld2")
    ps0 = load("res_pretraining.jld2", "ps")
else
    ps0 = main_pretraining()
end

@time prob, res = main_DeepReach(;lr = 1f-4, max_iter=100000, n_sample=20000, seed = 2025)

##
using Plots

ts = t_min:1f-2:t_max
xs = x_min:1f-2:x_max 
ys = y_min:1f-2:y_max
θs = θ_min:1f-2:θ_max

l(x, y) = sqrt(x^2 + y^2) - R
ps = res.u |> c_dev

tq =-1f0
θq = 0f0

in_pred = stack([[tq, x, y, θq] for x in xs for y in ys])
L = map(x->l(x[2],x[3]), eachcol(in_pred))
U = vec(model_prediction_CPU(in_pred; ps=ps))
V_predict = L + tq * U
# V_predict = [l(x, y) + tq * model_prediction_CPU([tq, x, y, θq]; ps=ps)[] for x in xs for y in ys]

target = [l(x, y) for x in xs for y in ys]
pV = plot(xs, ys, V_predict, linetype=:contour, title="V prediction", xlabel="x", ylabel="y", levels=30, aspect_ratio=:equal, cbar=:right)
contour!(pV, xs, ys, V_predict, levels=[0.0], linecolor=:teal)
# pl = plot(xs, ys, target, linetype = :contourf, title = "target", xlabel = "x", ylabel = "y")
# pe = plot(xs, ys, V_predict-target, linetype = :contourf, title = "error", xlabel = "x", ylabel = "y")
# plot(pV, pl, pe)