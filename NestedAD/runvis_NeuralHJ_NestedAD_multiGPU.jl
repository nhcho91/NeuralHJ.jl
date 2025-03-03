using Lux, LuxCUDA, Plots, JLD2

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

const θ_max = Float32(5 / 4 * pi)
const θ_min = -Float32(5 / 4 * pi)

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

function vis_DeepReach(ps, st; tq=0.0f0, θq=0.0f0, n_grid_pred=100)
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

# run main file across multiple GPUs
run(`mpiexecjl -n 4 julia --project=. -t auto ./main_NeuralHJ_NestedAD_multiGPU.jl`)

# visualisation
ps, st = load("res_DeepReach_NestedAD.jld2","ps","st")
# (ps, st) = (trained_model.ps, trained_model.st)

vis_DeepReach(ps, st; tq=-1.0f0, θq=Float32(pi / 4))

# animation
anim = Animation("./fig_temp")
for tq in t_max:-5.0f-2:t_min
    frame(anim, vis_DeepReach(ps, st; tq=tq))
end
for θq in 0.0f0:5.0f-1:θ_max
    frame(anim, vis_DeepReach(ps, st; tq=t_min, θq=θq))
end
gif(anim, "./anim_fps15.gif", fps=15)