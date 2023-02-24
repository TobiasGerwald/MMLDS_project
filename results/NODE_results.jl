import Pkg
Pkg.activate("MMLDS_project")
using MMLDS_project
using OrdinaryDiffEq, NODEData, Plots, Flux, Hyperopt

### Create training Data ###

ode_sol = create_data()



### Get NODE ###

ho = Hyperoptimizer(100,
            N_weights=[16, 32, 64, 128],
            N_hidden_layers=1:4,
            activation=[leakyrelu,swish],
            τ_max=50:10:70,
#            eta=[1f-2,1f-3,1f-4],
            eta_decrease=[5,10,15],
            reg=[1e-1,1e-2,1e-3,1e-4])

x0=[1, 1, 1]
dt=0.01
best_model = hyperOpt(ho, ode_sol, x0, dt)


### Evaluation ###
            
sol_node = solve(best_model.prob, Tsit5())
            
plot(ode_sol, idxs=[1], tspan=(0, 5), label="lorenz", title="on training data")
plot!(sol_node, idxs=[1], tspan=(0, 5), label="node")
            
plot(sol, idxs=[1], tspan=(50, 100), label="lorenz", title="on validation data")
plot!(sol_node, idxs=[1], tspan=(50, 100), label="node")