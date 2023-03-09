import Pkg
Pkg.activate("MMLDS_project")
using MMLDS_project
using NetCDF, NODEData, Plots, Flux, Hyperopt, OrdinaryDiffEq


path = "/Users/tobiasgerwald/Library/CloudStorage/OneDrive-Persönlich/Desktop/Modelling and ML for DS in Julia/MMLDS_project/data/sst.mon.mean.nc"

x_data = ncread(path, "sst")
x_reduced = x_data[121:170, 86:95, :]
x = compress_data_matrix(x_reduced, 5)


train, valid = NODEDataloader(x, 1:2040, 100, valid_set=0.5)

ho = Hyperoptimizer(16,
            N_weights=[32, 64, 128],
            N_hidden_layers=1:2,#:4,
            activation=[leakyrelu],#,swish],
            τ_max=[100],#]:10:70,
            eta_decrease=[10],#,10,15],
            reg=[1e-2, 1e-3]
)

x0 = x[:,1]
best_model, best_neural_ode = hyperOpt(ho, x, x0, N_epochs = 100, mode = "real")#, rhs_sug)


### Evaluation ###

tstart = 0.
tend = 2040.
tspan = (tstart, tend)
prob = ODEProblem(best_neural_ode, x0, tspan, best_model.p)
sol_node = solve(prob, Tsit5(), saveat=tstart:tend)
 
teval = 100
#plot(ode_sol, idxs=[1], tspan=(0, 100), label="El Nino", title="on training data")
plot(1:teval, x[1,1:teval], label="El Nino", title="on training data")
plot!(sol_node, idxs=[1], tspan=(0, teval), label="node")
            
plot(1021:2040, x[1, 1021:2040], label="El Nino", title="on training data")
plot!(sol_node, idxs=[1], tspan=(1021, 2040), label="node")