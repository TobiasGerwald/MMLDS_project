import Pkg
Pkg.activate("MMLDS_project")
using MMLDS_project
using DataDrivenDiffEq
using DataDrivenSparse
using DynamicalSystems
using DifferentialEquations
using OrdinaryDiffEq
using CairoMakie
using NearestNeighbors
using ChaosTools

#Loading and changing El Nino Data
path_to_data = "sst.mon.mean.nc"

x_data = ncread(path_to_data, "sst") #sst = Sea Surface Temperature
x_reduced = x_data[191:240, 86:95, :] #only concerned with the important region for the El Nino happening
X_vector = compress_data_matrix(x_reduced, 5, false, true)






### Temporal ###

x_temp = x_data[191, 86, :] 
D, τ, E = optimal_traditional_de(x_temp, τs = 1:200)

#lasso parameter lambda cannot be smaller than 1e-2, otherwise an error occurs
ddsol = train_SINDy(Matrix{Float64}(D)', 1e-1, 1e-1, basis = "fourier_basis", n = 10)

function recovered_dynamics(u,p,t)
    return ddsol(u,p,t)
end

function create_SINDy_trajectory(x0 = [1.,1.,1.],dt = 1, tstart = 1., tend = 20.)

    p_SINDy = ddsol.prob.p
    tspan = (tstart, tend)
    saveat = tstart:dt:tend
    
    prob = ODEProblem(recovered_dynamics, x0, tspan, p_SINDy) 
    sol = solve(prob, Tsit5(), saveat=saveat)
    return sol
end
D0 = D[1]
sol_sindy = create_SINDy_trajectory(D0,.005,1,20)
Plots.plot(sol_sindy)