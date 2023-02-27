module MMLDS_project

# Write your package code here.
    using DynamicalSystems, CairoMakie, DifferentialEquations, Plots, OrdinaryDiffEq, NODEData, Printf, Flux, DiffEqSensitivity, Parameters, Hyperopt, StatsBase, ReservoirComputing, DataDrivenDiffEq, DataDrivenSparse
    include("data.jl")
    include("SINDy.jl")
    include("NODE.jl")
    include("Reservoir_Computing.jl")

    export create_data, train_SINDy, hyperOpt

end
