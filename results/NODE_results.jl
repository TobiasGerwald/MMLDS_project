import Pkg
Pkg.activate("MMLDS_project")
using MMLDS_project
using OrdinaryDiffEq, NODEData, Plots, Flux, Hyperopt

### Create training Data ###

ode_sol = create_data()


### Get NODE ###

ho = Hyperoptimizer(1,
            N_weights=[32],#, 32, 64, 128],
            N_hidden_layers=1,#:4,
            activation=[leakyrelu],#,swish],
            τ_max=[100],#]:10:70,
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
best_model, best_neural_ode = hyperOpt(ho, ode_sol, x0, dt, 50, rhs_sug)#, p_rhs)


### Evaluation ###

tstart = 0.
tend = 100.
tspan = (tstart, tend)
prob = ODEProblem(best_neural_ode, x0, tspan, best_model.p)
sol_node = solve(prob, Tsit5(), saveat=tstart:0.01:tend, )
            
plot(ode_sol, idxs=[1], tspan=(0, 5), label="lorenz", title="on training data")
plot!(sol_node, idxs=[1], tspan=(0, 5), label="node")
            
plot(ode_sol, idxs=[1], tspan=(50, 100), label="lorenz", title="on validation data")
plot!(sol_node, idxs=[1], tspan=(50, 100), label="node")