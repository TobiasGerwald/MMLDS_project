import Pkg
Pkg.activate("MMLDS_project")
using MMLDS_project
using OrdinaryDiffEq, Plots, DataDrivenDiffEq, Random, Distributions

ode_sol = create_data()
ddsol = train_SINDy(ode_sol, 1e-1, 0, false, n = 4)

ddsol.basis
ddsol.prob.p  #perfect reconstruction if we set the ridge regression penalty to 0

### add noise to data ###
d = Normal(0, 1e-4)
ode_sol_noise = reduce(hcat,ode_sol.u) + rand(d, size(ode_sol))

ddsol_noise = train_SINDy(ode_sol_noise, 1e-2, 0, false, n = 2)

ddsol_noise.basis
ddsol_noise.prob.p 

#wrapper funtions
function f(u,p,t)
    return ddsol(u,p,t)
end

function g(u,p,t)
    return ddsol_noise(u,p,t)
end


tspan = (0., 100.)
prob = ODEProblem(f, [1,1,1], tspan, ddsol.prob.p)
sol = solve(prob, Tsit5(), saveat=0:0.01:100)

prob_noise = ODEProblem(g, [1,1,1], tspan, ddsol_noise.prob.p)
sol_noise = solve(prob_noise, Tsit5(), saveat=0:0.01:100)


plot(ode_sol, idxs=[1], tspan=(0, 5), label="lorenz", title="on training data")
plot!(sol, idxs=[1], tspan=(0, 5), label="sindy")
plot!(sol_noise, idxs=[1], tspan=(0, 5), label="sindy on noise", ylims = [-15,20])


plot(ode_sol, idxs=[1], tspan=(50, 100), label="lorenz", title="on validation data")
plot!(sol, idxs=[1], tspan=(50, 100), label="sindy")
plot!(sol_noise, idxs=[1], tspan=(50, 100), label="sindy on noise")#, ylims = [-1000,20])