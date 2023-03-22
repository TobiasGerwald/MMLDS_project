import Pkg
Pkg.activate("MMLDS_project")
using MMLDS_project
using OrdinaryDiffEq, Plots

#COULD NOT BE TESTED YET

#generate data
sol = create_data()

#generate parameter grid
param_grid = []

reservoir_sizes = [512, 1024]
spectral_radii = [0.8, 1.0, 1.2]
sparsities = [0.03, 0.05]
input_scales = [0.1]
ridge_values = [0.0, 1e-6, 1e-5]
# Take the Cartesian product of the possible values
for params in Iterators.product(reservoir_sizes, spectral_radii, sparsities, input_scales, ridge_values)
    push!(param_grid, ESNHyperparams(params...))
end

#data: train, val, test split 
train_data, val_data, test_data = train_val_test_split(sol, val_seconds = 15, test_seconds = 15)

#cross-validation
esn, Wₒᵤₜ = cross_validate_esn(train_data, val_data, param_grid)

#plot prediction
times = 0:0.01:5
prediction_longterm = esn(Generative(10001), Wₒᵤₜ)
label = ["lorenz" "ESN"]
p1 = plot(times, [sol[1,1:501], prediction_longterm[1,1:501]], label = label, ylabel = "x(t)")
p2 = plot(times, [sol[1,1:501], prediction_longterm[1,1:501]], label = label, ylabel = "y(t)")
p3 = plot(times, [sol[1,1:501], prediction_longterm[1,1:501]], label = label, ylabel = "z(t)")
p4 = plot(p1, p2, p3, layout = (3, 1), size = (800, 600))
savefig(p4, "long_term_esn_lorenz")

