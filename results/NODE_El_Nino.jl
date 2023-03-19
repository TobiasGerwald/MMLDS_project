import Pkg
Pkg.activate("MMLDS_project")
using MMLDS_project
using NetCDF, NODEData, Plots, Flux, Hyperopt, OrdinaryDiffEq, DynamicalSystems, CairoMakie


path = "/Users/tobiasgerwald/Library/CloudStorage/OneDrive-Persönlich/Desktop/Modelling and ML for DS in Julia/MMLDS_project/data/sst.mon.mean.nc"

x_data = ncread(path, "sst")
x_reduced = x_data[191:240, 86:95, :]
x = compress_data_matrix(x_reduced, 5)


train, valid = NODEDataloader(x, 1:2040, 100, valid_set=0.5)

ho = Hyperoptimizer(1,
            N_weights=256,#[32, 64, 128],
            N_hidden_layers=1,#1:4,
            activation=[leakyrelu],#,swish],
            τ_max=[20],#]:10:70,
            eta_decrease=[10],#,10,15],
            reg=1e-2#[1e-2, 1e-3]
)

x0 = x[:,1]
best_model, best_neural_ode = hyperOpt(ho, x, x0, N_epochs = 100, mode = "real")#, rhs_sug)
#N_EPOCHS= 100 - N_weights=128 - N_hidden_layers=2 - activation=leakyrelu - reg=0.01

#save_NODE(best_model, "model.jld2", best_neural_ode, "neural_ode.jld2")
#best_model, best_neural_ode = load_NODE("model.jld2","neural_ode.jld2")

### Evaluation ###

tstart = 0.
tend = 2040.
tspan = (tstart, tend)
prob = ODEProblem(best_neural_ode, x0, tspan, best_model.p)
sol_node = solve(prob, Tsit5(), saveat=tstart:tend)
 
teval = 100
#plot(ode_sol, idxs=[1], tspan=(0, 100), label="El Nino", title="on training data")
Plots.plot(1:teval, x[1,1:teval], label="El Nino", title="on training data")#, ylims =[25,30])
Plots.plot!(sol_node, idxs=[1], tspan=(1, teval), label="node")
            
Plots.plot(1021:2040, x[1, 1021:2040], label="El Nino", title="on validation data")
Plots.plot!(sol_node, idxs=[1], tspan=(1021, 2040), label="node")


plot_lyapunov_exp(sol_node, 15:5:30, 5:5:20, k_values = 0:10:100)
plot_lyapunov_exp(sol_node, 15:5:30, 5:5:20, k_values = 0:10:1000)

