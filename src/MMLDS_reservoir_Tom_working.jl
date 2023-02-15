import Pkg
Pkg.activate("MMLDS_project")
using MMLDS_project
using DifferentialEquations

function lorenz(u, p, t)
    x, y, z = u
    σ, ρ, β = p
    dx = σ * (y - x)
    dy = x * (ρ - z) - y
    dz = x * y - β * z
return [dx, dy, dz]
end

p_lorenz = [10, 28, 8/3] 
tspan = (0.,100.)
dt = 0.01

x0 = [1., 1., 1.]

prob = ODEProblem(lorenz, x0, tspan, p_lorenz) 
sol = solve(prob, Tsit5())