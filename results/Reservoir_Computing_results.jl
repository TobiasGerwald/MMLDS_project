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
function plot_prediction(esn, Wₒᵤₜ, test_data)
    steps_to_predict = size(test_data, 2)
    prediction = esn(Generative(steps_to_predict), Wₒᵤₜ)
    dt = 0.01
    label = ["actual" "predicted"]
    times = dt * collect(0:steps_to_predict)[1:end-1] 

    p1 = plot(times, [test_data[1, :], prediction[1, :]], label = label, ylabel = "x(t)")
    p2 = plot(times, [test_data[2, :], prediction[2, :]], label = label, ylabel = "y(t)")
    p3 = plot(times, [test_data[3, :], prediction[3, :]], label = label, ylabel = "z(t)", xlabel = "t")
    plot(p1, p2, p3, layout = (3, 1), size = (800, 600))
end
plot_prediction(esn, Wₒᵤₜ, test_data)