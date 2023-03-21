module MMLDS_project

# Write your package code here.
#github link for NODEData: https://github.com/maximilian-gelbrecht/NODEData.jl.git
    using DynamicalSystems, CairoMakie, DifferentialEquations, Plots, OrdinaryDiffEq, NODEData, Printf, Flux, DiffEqSensitivity, Parameters, Hyperopt, StatsBase, ReservoirComputing, DataDrivenDiffEq, DataDrivenSparse, NetCDF, ProgressMeter, JLD2, Random, Distributions
    include("data.jl")
    include("SINDy.jl")
    include("NODE.jl")
    include("Reservoir_Computing.jl")
    include("Lyapunov.jl")

    export create_data, train_SINDy, hyperOpt, generate_esn, train_val_test_split, ESNHyperparams, cross_validate_esn, compress_data_matrix, save_NODE, load_NODE, plot_lyapunov_exp, plot_SINDy_and_ElNino_trajectories, MSE_on_lyapunov_time, load_ESN, save_ESN

end
