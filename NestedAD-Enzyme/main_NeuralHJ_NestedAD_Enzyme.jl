# using MKL
using Lux, LuxCUDA, Optimisers, ADTypes, Reactant, Enzyme
using LinearAlgebra, Random, Statistics, Printf, JLD2, Plots

# Reactant.set_default_backend("cpu")
const x_dev = reactant_device(; force=true)
# const g_dev = gpu_device()
const c_dev = cpu_device()

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

target_constraint(indvars) = map(x -> sqrt(x[2]^2 + x[3]^2) - R, eachcol(indvars)) 

@views function in_norm(indvars)
    return [(indvars[1:1, :] .- t_min) / (t_max - t_min);
        2.0f0 * (indvars[2:2, :] .- x_min) / (x_max - x_min) .- 1.0f0;
        2.0f0 * (indvars[3:3, :] .- y_min) / (y_max - y_min) .- 1.0f0;
        2.0f0 * (indvars[4:4, :] .- θ_min) / (θ_max - θ_min) .- 1.0f0]
end

function model_construction(n_in, n_out; l_hidden=3, n_hidden=512)
    return Chain(
        WrappedFunction(in_norm),
        Dense(n_in => n_hidden, sin),
        [Dense(n_hidden => n_hidden, sin) for i in 1:l_hidden],
        Dense(n_hidden => n_out, softplus)
    )
end

@views function value_prediction_CPU(in_pred, ps, st; n_hidden=512)
    model = model_construction(4, 1; n_hidden=n_hidden)
    (in_pred, ps, st) = (in_pred, ps, st) |> c_dev

    l_pred = target_constraint(in_pred)
    t_pred = in_pred[1, :]
    U_pred = vec(model(in_pred, ps, st)[1])
    V_pred = l_pred .+ t_pred .* U_pred
    return V_pred, U_pred, l_pred
end

@views function physics_informed_loss_function(
    smodel::StatefulLuxLayer, indvars::AbstractArray, ∂l_∂indvars::AbstractArray
)
    t, θ = indvars[1:1, :], indvars[4:4, :]
    U = smodel(indvars)
    ∂U_∂indvars = Enzyme.gradient(Enzyme.Reverse, sum ∘ smodel, indvars)[1]

    ∂V_∂t = U .+ t .* ∂U_∂indvars[1:1, :]
    ∂V_∂x = ∂l_∂indvars[2:2, :] .+ t .* ∂U_∂indvars[2:2, :]
    ∂V_∂y = ∂l_∂indvars[3:3, :] .+ t .* ∂U_∂indvars[3:3, :]
    ∂V_∂θ = t .* ∂U_∂indvars[4:4, :]

    return mean(abs2, min.(-t .* U, ∂V_∂t .+ ∂V_∂x .* v .* cos.(θ) .+ ∂V_∂y .* v .* sin.(θ) .+ abs.(∂V_∂θ) .* ω_max))
end

function loss_function(model, ps, st, (indvars, ∂l_∂indvars))
    smodel = StatefulLuxLayer{true}(model, ps, st)
    loss = physics_informed_loss_function(smodel, indvars, ∂l_∂indvars)
    return (loss, smodel.st, (; loss))
end

function pretrain_model!(model, ps, st, x_data, y_data; max_iter=5000, lr=1f-3)
    train_state = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(lr))

    for iter in 1:max_iter
        _, loss, _, train_state = Lux.Training.single_train_step!(
            AutoEnzyme(), MSELoss(),
            (x_data, y_data), train_state
        )
        if iter % 100 == 1
            @printf "Iteration: %04d \t Loss: %10.9g\n" iter loss
        end
        if loss < 1.0f-6
            break
        end
    end

    return model, ps, st, train_state
end

function train_model!(model, ps, st, in_data, supp_data; max_iter=5000, lr0=1.0f-3)
    train_state = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(lr0))
    lr = i -> i < 50000 ? lr0 : (i < 90000 ? 1.0f-1 * lr0 : 1.0f-2 * lr0)

    for iter in 1:max_iter
        # GC.gc()
        Optimisers.adjust!(train_state, lr(iter))
        _, loss, stats, train_state = Training.single_train_step!(
            AutoEnzyme(), loss_function,
            (in_data, supp_data), train_state
        )

        if iter % 25 == 0 || iter == 1 || iter == max_iter
            @printf "Iteration: [%6d/%6d] \t Loss: %.9f \t stats.loss: %.9f\n" iter max_iter loss stats.loss
            # GC.gc(true)
        end
        
        if iter % 1000 == 0
            pVU = vis_DeepReach(train_state.parameters, train_state.states; tq=-1f0)
            display(pVU)
            savefig(pVU, "./fig_temp/Fig_$(iter).pdf")
        end
        if loss < 1.0f-6
            break
        end
    end

    return train_state
end

function main_DeepReach(; seed=0, max_iter=10^5, lr0=1.0f-3, n_hidden=512, n_grid_train=16, ps0=nothing, st0=nothing, mode_train=1)
    # seeding
    rng = Random.default_rng()
    Random.seed!(rng, seed)

    # model construction
    model = model_construction(4, 1; n_hidden=n_hidden)

    # model initialisation
    if isnothing(ps0)
        ps, st = Lux.setup(rng, model) |> x_dev
    else
        (ps, st) = (ps0, st0) |> x_dev
    end

    # training grid definition (input evaluation points) 
    if mode_train == 0
        t_grid = [t_max]
    else
        t_grid = range(t_min, t_max; length=n_grid_train)
    end
    x_grid = filter(!iszero, range(x_min, x_max; length=n_grid_train))
    y_grid = filter(!iszero, range(y_min, y_max; length=n_grid_train))
    θ_grid = range(θ_min, θ_max; length=n_grid_train)
    # in_train = stack([t, x, y, θ] for t in t_grid for x in x_grid for y in y_grid for θ in θ_grid) |> x_dev
    in_train = stack([t, x, y, θ] for θ in θ_grid for y in y_grid for x in x_grid for t in t_grid) |> x_dev
    #--> equal to in_train = stack([[elem...] for elem in vec(collect(Iterators.product(t_grid, x_grid, y_grid, θ_grid)))]) |> x_dev
    ∂l_∂in_train = only(Enzyme.gradient(Enzyme.Reverse, sum ∘ target_constraint, c_dev(in_train))) |> x_dev

    # training
    if mode_train == 0
        out_pretrain = repeat([0f0], 1, size(in_train,2)) |> x_dev
        pretrain_model!(model, ps, st, in_train, out_pretrain; max_iter=max_iter, lr=lr0)
    else
        train_state = train_model!(model, ps, st, in_train, ∂l_∂in_train; max_iter=max_iter, lr0=lr0)
    end

    (ps, st) = (train_state.parameters, train_state.states) |> c_dev
    if mode_train == 0
        save("res_pretraining.jld2", "ps", ps, "st", st)
    else
        save("res_DeepReach_NestedAD_avoid.jld2", "ps", ps, "st", st)
    end

    return StatefulLuxLayer{true}(model, ps, st)
end

function vis_DeepReach(ps, st; tq=0f0, θq=0f0, n_grid_pred=100)
    # t_grid = range(t_min, t_max; length=n_grid_pred)
    x_grid = filter(!iszero, range(x_min, x_max; length=n_grid_pred))
    y_grid = filter(!iszero, range(y_min, y_max; length=n_grid_pred))
    # θ_grid = range(θ_min, θ_max; length=n_grid_pred)

    in_pred = stack([[tq, x, y, θq] for y in y_grid for x in x_grid])
    V_pred, U_pred, l_pred = value_prediction_CPU(in_pred, ps, st)

    pV = plot(x_grid, y_grid, V_pred, linetype=:contourf, levels=30, aspect_ratio=:equal, cbar=:right, title="V prediction: t = $tq, θ = $θq", xlabel="x", ylabel="y")
    contour!(pV, x_grid, y_grid, V_pred, levels=[0.0], linecolor=:cyan, linewidth=1, colorbar_entry=false)
    contour!(pV, x_grid, y_grid, l_pred, levels=[0.0], linecolor=:yellow, linewidth=1, colorbar_entry=false)
    
    pU = plot(x_grid, y_grid, U_pred, linetype=:contour, title="U prediction: t = $tq, θ = $θq", xlabel="x", ylabel="y", levels=30, aspect_ratio=:equal, cbar=:right)

    pVU = plot(pV, pU, size=(1000, 500))
    # _, _, (_, dV_pred) = loss_function(trained_model.model, trained_model.ps, trained_model.st, (in_pred,))
    # pdV = plot(x_grid, y_grid, vec(dV_pred), linetype=:contour, title="dV prediction: t = $tq, θ = $θq", xlabel="x", ylabel="y", levels=30, aspect_ratio=:equal, cbar=:right)

    return pVU
end

# pretraining: mode_train = 0
# @time trained_model = main_DeepReach(; seed=2025, lr0=1.0f-4, max_iter=5000, n_grid_train=16, mode_train=0)

# main training: mode_train = 1
# ps_pre, st_pre = load("res_pretraining.jld2","ps","st")
@time trained_model = main_DeepReach(; seed=0, lr0=1.0f-4, max_iter=1000, n_grid_train=16)
# trained_model = Lux.testmode(trained_model)

## re-training: mode_train = 1
# @time trained_model = main_DeepReach(; seed=2025, lr0=1.0f-4, max_iter=5000, n_grid_train=5, ps0=trained_model.ps, st0=trained_model.st)

## visualisation
# ps, st = load("res_DeepReach_NestedAD_avoid.jld2","ps","st")
(ps, st) = (trained_model.ps, trained_model.st)

vis_DeepReach(ps, st; tq=-1f0)

anim = Animation("./fig_temp")
for tq in t_max:-5f-2:t_min
    frame(anim, vis_DeepReach(ps, st; tq=tq))
end
for θq in 0f0:5f-1:θ_max
    frame(anim, vis_DeepReach(ps, st; tq=t_min, θq=θq))
end
gif(anim, "./anim_fps15.gif", fps = 15)