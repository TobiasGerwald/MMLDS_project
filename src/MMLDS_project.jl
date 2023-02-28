module MMLDS_project

# Write your package code here.
    using DynamicalSystems, CairoMakie, DifferentialEquations, Plots, OrdinaryDiffEq, NODEData, Printf, Flux, DiffEqSensitivity, Parameters, Hyperopt, StatsBase, ReservoirComputing, DataDrivenDiffEq, DataDrivenSparse, NetCDF
    include("data.jl")
    include("SINDy.jl")
    include("NODE.jl")
    include("Reservoir_Computing.jl")

    export create_data, train_SINDy, hyperOpt, generate_esn, train_val_test_split, ESNHyperparams, cross_validate_esn

end
