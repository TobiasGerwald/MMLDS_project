module MMLDS_project

# Write your package code here.
#github link for NODEData: https://github.com/maximilian-gelbrecht/NODEData.jl.git
    using DynamicalSystems, CairoMakie, DifferentialEquations, Plots, OrdinaryDiffEq, NODEData, Printf, Flux, DiffEqSensitivity, Parameters, Hyperopt, StatsBase, ReservoirComputing, DataDrivenDiffEq, DataDrivenSparse, NetCDF, ProgressMeter, JLD2
    include("data.jl")
    include("SINDy.jl")
    include("NODE.jl")
    include("Reservoir_Computing.jl")

    export create_data, train_SINDy, hyperOpt, generate_esn, train_val_test_split, ESNHyperparams, cross_validate_esn, compress_data_matrix

end
