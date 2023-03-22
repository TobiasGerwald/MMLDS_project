import Pkg
Pkg.activate("MMLDS_project")
using MMLDS_project
using ProgressMeter, DynamicalSystems, ReservoirComputing, Plots, Printf, DifferentialEquations, OrdinaryDiffEq, NetCDF, JLD2

#Loading and changing El Nino Data
path_to_data = "C:/Users/tomfi/Desktop/Uni/3M_semester/Project_MMLDS/MMLDS_4/MMLDS_project/src/sst.mon.mean.nc"

x_data = ncread(path_to_data, "sst") #sst = Sea Surface Temperature
x_reduced = x_data[191:240, 86:95, :] #only concerned with the important region for the El Nino happening
X_vector = compress_data_matrix(x_reduced, 5, false, true)

#generating train, val and test set
train_data, val_data, test_data = train_val_test_split(X_vector, val_seconds = 5, test_seconds = 5)

#Define grid:
param_grid = []

reservoir_sizes = [256, 512, 1024, 2048, 4096]
spectral_radii = [0.8, 0.9, 1.0,1.1, 1.2] 
sparsities = [0.01, 0.03, 0.05]
input_scales = [0.1,0.01, 0.05]
ridge_values = [1e-6, 1e-5, 1e-4, 1e-3,1e-2,1e-1]
# Take the Cartesian product of the possible values
for params in Iterators.product(reservoir_sizes, spectral_radii, sparsities, input_scales, ridge_values)
    push!(param_grid, ESNHyperparams(params...))
end
println(length(param_grid), " hyperparameter combinations.")

#train network:
#esn, W_out = cross_validate_esn(train_data, val_data, param_grid)

#save network
absolute_path = "C:/Users/tomfi/Desktop/Uni/3M_semester/Project_MMLDS/MMLDS_4/MMLDS_project/results/"
network_pathString = absolute_path*"esn_network.jld2"
W_out_pathString = absolute_path*"W_out_matrix.jld2"
#save_ESN(esn, network_pathString, W_out, W_out_pathString)

#load a saved network
esn_loaded, W_out_loaded = load_ESN(network_pathString, W_out_pathString)

#load a trained network:
predictions_loaded = esn_loaded(Generative(100), W_out_loaded)

first_field_predicted = predictions_loaded[1,1:100]

times = 1:1:100
label = ["El Nino" "ESN"]

savePath_init = "C:/Users/tomfi/Desktop/Uni/3M_semester/Project_MMLDS/MMLDS_4/MMLDS_project/plots/ESN/"
#savePath_init = string("/home/tom/Documents/University/3M_semester/Project_MMLDS/MMLDS_2/MMLDS_project/Figures/Loaded_El_Nino_Predictions/", saveEnding,".png" )

for i in 1:20
    i_data = X_vector[i, 1:100]
    p = Plots.plot(times, [i_data, predictions_loaded[i,1:100]], label = label, ylabel = "temperature", xlabel = "months")#
    saveEnding = string(i)
    savePath = savePath_init * saveEnding * ".png" 
    savefig(p, savePath)
end

# Estimate the optimal embedding
predictions_loaded = esn_loaded(Generative(2000), W_out_loaded)
h(u) = u[1, :]
s = h(predictions_loaded)
D, τ, E = optimal_traditional_de(s)

plot_lyapunov_exp(predictions_loaded, [20,30], [3, 5, 7, 15], k_values = 0:10:100)

#define maximal lyapunov exponent and calculate MSE
λ_max = 0.02
MSE = 0
n_restarts = 20
loss(x,y) = sum(abs2, x - y)
for i in 1:n_restarts
    tstart = 0.
    #tend = trunc(Int, 1/λ_max)
    tend = 10
    tspan = (tstart, tend)
    x0 = X_vector[:,i]
    
    #test if we can use recovered dynamics here
    predictions = esn_loaded(Generative(tend+1), W_out_loaded)
    MSE += loss(predictions, X_vector[:,i:i+tend])
end

MSE /= n_restarts


### Temporal ###

x_temp = x_data[191, 86, :] 
D, τ, E = optimal_traditional_de(x_temp, τs = 1:200)
train_data, val_data, test_data = train_val_test_split(Matrix{Float64}(D)', val_seconds = 5, test_seconds = 5)

 
#esn_temp, W_out_temp = cross_validate_esn(train_data, val_data, param_grid)
network_pathString_temp = absolute_path*"esn_network_temp.jld2"
W_out_pathString_temp = absolute_path*"W_out_matrix_temp.jld2"
#save_ESN(esn_temp, network_pathString_temp, W_out_temp, W_out_pathString_temp)

#load a saved network
esn_temp_loaded, W_out_temp_loaded = load_ESN(network_pathString_temp, W_out_pathString_temp)

#load a trained network:
predictions_temp = esn_temp_loaded(Generative(10000), W_out_temp_loaded)

times = 1:1:100
label = ["El Nino" "esn_temp "]
p = Plots.plot(times, [x_temp[1:100], predictions_temp[1,1:100]], label = label, ylabel = "temperature", xlabel = "months")#
saveEnding = string(1)
savePath = savePath_init * saveEnding * "_temporal.png" 
savefig(p, savePath)


plot_lyapunov_exp(predictions_temp,[5,10,20], [4,6,12], k_values = 0:10:100)