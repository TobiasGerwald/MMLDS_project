using OrdinaryDiffEq
### Create training Data ###

function lorenz(u, p, t)
    x, y, z = u
    σ, ρ, β = p
    dx = σ * (y - x)
    dy = x * (ρ - z) - y
    dz = x * y - β * z
return [dx, dy, dz]
end

function create_data(x0 = [1.,1.,1.],dt = 0.01)

    p_lorenz = [10, 28, 8/3] 
    tstart = 0.
    tend = 100.
    tspan = (tstart, tend)
    saveat = tstart:dt:tend

    prob = ODEProblem(lorenz, x0, tspan, p_lorenz) 
    sol = solve(prob, Tsit5(), saveat=saveat)

    return sol

end