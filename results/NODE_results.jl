import Pkg
Pkg.activate("MMLDS_project")
using MMLDS_project
using OrdinaryDiffEq, NODEData, Plots, Flux, Hyperopt

### Create training Data ###

ode_sol = create_data()


### Get NODE ###

ho_rhs = Hyperoptimizer(1,
            N_weights=[32],#, 32, 64, 128],
            N_hidden_layers=1,#:4,
            activation=[leakyrelu],#,swish],
            τ_max=[20],#]:10:70,
            eta_decrease=[10],#,10,15],
            reg=[1e-2]
)

function rhs_sug(u, t)
    x, y, z = u
    σ, ρ, β = [10, 28, 8/3]#p
    dx = σ * (y - x) 
    dy = x * (ρ - z) #- y
    dz = x * y #- β * z
    return [dx, dy, dz]
end

#p_rhs = [10, 28, 8/3] 

x0=[1, 1, 1]
dt=0.01
best_model_rhs, best_neural_ode_rhs = hyperOpt(ho_rhs, ode_sol, x0, rhs_sug = rhs_sug)

### Evaluation ###

tstart = 0.
tend = 100.
tspan = (tstart, tend)
prob_rhs = ODEProblem(best_neural_ode_rhs, x0, tspan, best_model_rhs.p)
sol_node_rhs = solve(prob_rhs, Tsit5(), saveat=tstart:0.01:tend, )


p1 = plot(ode_sol, idxs=[1], tspan=(0, 5), label="lorenz", ylabel = "x(t)")
plot!(sol_node_rhs, idxs=[1], tspan=(0, 5), label="node_rhs")
p2 = plot(ode_sol, idxs=[2], tspan=(0, 5), label="lorenz", ylabel = "y(t)")
plot!(sol_node_rhs, idxs=[2], tspan=(0, 5), label="node_rhs")
p3 = plot(ode_sol, idxs=[3], tspan=(0, 5), label="lorenz", ylabel = "z(t)")
plot!(sol_node_rhs, idxs=[3], tspan=(0, 5), label="node_rhs")
plot(p1, p2, p3, layout = (3, 1), size = (800, 600))
png("NODE_rhs_lorenz-shortterm.png")

p1 = plot(ode_sol, idxs=[1], tspan=(50, 100), label="lorenz", ylabel = "x(t)")
plot!(sol_node_rhs, idxs=[1], tspan=(50, 100), label="node_rhs")
p2 = plot(ode_sol, idxs=[2], tspan=(50, 100), label="lorenz", ylabel = "y(t)")
plot!(sol_node_rhs, idxs=[2], tspan=(50, 100), label="node_rhs")
p3 = plot(ode_sol, idxs=[3], tspan=(50, 100), label="lorenz", ylabel = "z(t)")
plot!(sol_node_rhs, idxs=[3], tspan=(50, 100), label="node_rhs")
plot(p1, p2, p3, layout = (3, 1), size = (800, 600))
png("NODE_rhs_lorenz-longterm.png")



### without rhs ###
ho_rhs = Hyperoptimizer(1,
            N_weights=[128],#, 32, 64, 128],
            N_hidden_layers=2,#:4,
            activation=[leakyrelu],#,swish],
            τ_max=[40],#]:10:70,
            eta_decrease=[10],#,10,15],
            reg=[1e-2]
)

best_model, best_neural_ode = hyperOpt(ho_rhs, ode_sol, x0)
prob = ODEProblem(best_neural_ode, x0, tspan, best_model.p)
sol_node = solve(prob, Tsit5(), saveat=tstart:0.01:tend, )

p1 = plot(ode_sol, idxs=[1], tspan=(0, 5), label="lorenz", ylabel = "x(t)")
plot!(sol_node, idxs=[1], tspan=(0, 5), label="node")
p2 = plot(ode_sol, idxs=[2], tspan=(0, 5), label="lorenz", ylabel = "y(t)")
plot!(sol_node, idxs=[2], tspan=(0, 5), label="node")
p3 = plot(ode_sol, idxs=[3], tspan=(0, 5), label="lorenz", ylabel = "z(t)")
plot!(sol_node, idxs=[3], tspan=(0, 5), label="node")
plot(p1, p2, p3, layout = (3, 1), size = (800, 600))
png("NODE_lorenz-shortterm.png")

p1 = plot(ode_sol, idxs=[1], tspan=(50, 100), label="lorenz", ylabel = "x(t)")
plot!(sol_node, idxs=[1], tspan=(50, 100), label="node")
p2 = plot(ode_sol, idxs=[2], tspan=(50, 100), label="lorenz", ylabel = "y(t)")
plot!(sol_node, idxs=[2], tspan=(50, 100), label="node")
p3 = plot(ode_sol, idxs=[3], tspan=(50, 100), label="lorenz", ylabel = "z(t)")
plot!(sol_node, idxs=[3], tspan=(50, 100), label="node")
plot(p1, p2, p3, layout = (3, 1), size = (800, 600))
png("NODE_lorenz-longterm.png")