import Pkg
Pkg.activate("MMLDS_project")
using MMLDS_project
using OrdinaryDiffEq, Plots, DataDrivenDiffEq, Random, Distributions

ode_sol = create_data()
ddsol = train_SINDy(ode_sol, 1e-1, 0, false, n = 4)

ddsol.basis
ddsol.prob.p  #perfect reconstruction if we set the penalty to 0

### add noise to data ###
d = Normal(0, 1e-4)
ode_sol_noise = reduce(hcat,ode_sol.u) + rand(d, size(ode_sol))

ddsol_noise = train_SINDy(ode_sol_noise, 1e-2, 0, false, n = 2)

println(get_basis(ddsol_noise))

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


p1 = plot(ode_sol, idxs=[1], tspan=(0, 5), label="lorenz", ylabel = "x(t)")
plot!(sol, idxs=[1], tspan=(0, 5), label="sindy")
plot!(sol_noise, idxs=[1], tspan=(0, 5), label="sindy on noise", ylims = [-15,20])
p2 = plot(ode_sol, idxs=[2], tspan=(0, 5), label="lorenz", ylabel = "y(t)")
plot!(sol, idxs=[2], tspan=(0, 5), label="sind")
plot!(sol_noise, idxs=[2], tspan=(0, 5), label="sindy on noise", ylims = [-15,20])
p3 = plot(ode_sol, idxs=[3], tspan=(0, 5), label="lorenz", ylabel = "z(t)")
plot!(sol, idxs=[3], tspan=(0, 5), label="sind")
plot!(sol_noise, idxs=[3], tspan=(0, 5), label="sindy on noise", ylims = [0,50])
plot(p1, p2, p3, layout = (3, 1), size = (800, 600))
png("SINDy_lorenz-shortterm.png")


p1 = plot(ode_sol, idxs=[1], tspan=(50, 100), label="lorenz", ylabel = "x(t)")
plot!(sol, idxs=[1], tspan=(50, 100), label="sindy")
plot!(sol_noise, idxs=[1], tspan=(50, 100), label="sindy on noise")#, ylims = [-15,20])
p2 = plot(ode_sol, idxs=[2], tspan=(50, 100), label="lorenz", ylabel = "y(t)")
plot!(sol, idxs=[2], tspan=(50, 100), label="sind")
plot!(sol_noise, idxs=[2], tspan=(50, 100), label="sindy on noise")#, ylims = [-15,20])
p3 = plot(ode_sol, idxs=[3], tspan=(50, 100), label="lorenz", ylabel = "z(t)")
plot!(sol, idxs=[3], tspan=(50, 100), label="sind")
plot!(sol_noise, idxs=[3], tspan=(50, 100), label="sindy on noise")#, ylims = [-15,20])
plot(p1, p2, p3, layout = (3, 1), size = (800, 600))
png("SINDy_lorenz-longterm.png")