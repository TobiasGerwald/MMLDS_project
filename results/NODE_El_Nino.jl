import Pkg
Pkg.activate("MMLDS_project")
using MMLDS_project
using NetCDF, NODEData, Plots, Flux, Hyperopt, OrdinaryDiffEq, DynamicalSystems, CairoMakie


path = "/Users/tobiasgerwald/Library/CloudStorage/OneDrive-Persönlich/Desktop/Modelling and ML for DS in Julia/MMLDS_project/data/sst.mon.mean.nc"


### Spacial ###

x_data = ncread(path, "sst")
x_reduced = x_data[191:240, 86:95, :]
x = compress_data_matrix(x_reduced, 5)


#train, valid = NODEDataloader(x, 1:2040, 100, valid_set=0.5)

ho = Hyperoptimizer(1,
            N_weights=128,#[32, 64, 128],
            N_hidden_layers=2,#1:4,
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

# Evaluation #

tstart = 0.
tend = 2040.
tspan = (tstart, tend)
prob = ODEProblem(best_neural_ode, x0, tspan, best_model.p)
sol_node = solve(prob, Tsit5(), saveat=tstart:tend)
 
teval = 100


Plots.plot(sol_node, tspan=(0, teval), label="El Nino", ylims =[25,30])

#plot(ode_sol, idxs=[1], tspan=(0, 100), label="El Nino", title="on training data")
Plots.plot(1:teval, x[1,1:teval], label="El Nino", title="on training data", ylims =[25,30])
Plots.plot!(sol_node, idxs=[1], tspan=(0, teval-1), label="node")
            
Plots.plot(1021:2040, x[1, 1021:2040], label="El Nino", title="on validation data")
Plots.plot!(sol_node, idxs=[1], tspan=(1021, 2040), label="node")


plot_lyapunov_exp(sol_node, 15:5:30, 5:5:20, k_values = 0:10:100)
plot_lyapunov_exp(sol_node, 15:5:30, 5:5:20, k_values = 0:10:500)


#Evaluate MSE after one lyapunov time
loss(x,y) = sum(abs2, x - y)


λ_max = 0.2
MSE = 0
n_restarts = 5
for i in 1:n_restarts
    tstart = 0.
    tend = trunc(Int, 1/λ_max)
    tspan = (tstart, tend)
    x0 = x[:,i]

    prob = ODEProblem(best_neural_ode, x0, tspan, best_model.p)
    sol_ode = solve(prob, Tsit5(), saveat=tstart:tend)
    #plot_SINDy_and_ElNino_trajectories(savePath, X_vector, sol_sindy)
    MSE += loss(sol_ode, x[:,i:i+tend])

end

MSE /= n_restarts




### Temporal ###

x_temp = x_data[191, 86, :] 
D, τ, E = optimal_traditional_de(x_temp, τs = 1:200)

#train_temp, valid_temp = NODEDataloader(Matrix{Float64}(D)', 1:length(D), 100, valid_set=0.5)

D0 = D[1]
best_temp_model, best_temp_neural_ode = hyperOpt(ho, Matrix{Float64}(D)', D0, N_epochs = 100, mode = "real")
#save_NODE(best_temp_model, "temp_model.jld2", best_temp_neural_ode, "temp_neural_ode.jld2")



tstart = 0.
tend = 2040.
tspan = (tstart, tend)
prob_temp = ODEProblem(best_temp_neural_ode, Vector{Float64}(D0), tspan, best_temp_model.p)
sol_node_temp = solve(prob_temp, Tsit5(), saveat=tstart:tend)
Plots.plot(sol_node_temp, tspan=(0, teval), title="El Nino Temporal Approach", ylims =[25,30])

Plots.plot(6:teval, x[1,6:teval], label="El Nino", title="on training data", ylims =[25,30])
Plots.plot!(sol_node_temp, idxs=[1], tspan=(1, teval), label="node")

plot_lyapunov_exp(sol_node_temp, 5:5:15, 1:3:10, k_values = 0:10:100)
plot_lyapunov_exp(sol_node_temp, 5:5:15, 1:3:10, k_values = 0:10:1000)


λ_max = 
MSE = 0
n_restarts = 5
for i in 1:n_restarts
    tstart = 0.
    tend = trunc(Int, 1/λ_max)
    tspan = (tstart, tend)
    x0 = x[:,i]

    prob = ODEProblem(best_neural_ode, x0, tspan, best_model.p)
    sol_ode = solve(prob, Tsit5(), saveat=tstart:tend)
    #plot_SINDy_and_ElNino_trajectories(savePath, X_vector, sol_sindy)
    MSE += loss(sol_ode, x[:,i:i+tend])

end

MSE /= n_restarts