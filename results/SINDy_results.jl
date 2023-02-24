import Pkg
Pkg.activate("MMLDS_project")
using MMLDS_project
using OrdinaryDiffEq, Plots

ode_sol = create_data()
ddsol = train_SINDy(ode_sol, 4, 1e-1, 0)

ddsol.basis
ddsol.prob.p  #perfect reconstruction if we set the ridge regression penalty to 0

ddsol.prob


sol = solve(sol.prob, Tsit5())

plot(ode_sol, idxs=[1], tspan=(0, 5), label="lorenz", title="on training data")
plot!(sol, idxs=[1], tspan=(0, 5), label="SINDy")

plot(ode_sol, idxs=[1], tspan=(50, 100), label="lorenz", title="on validation data")
plot!(sol, idxs=[1], tspan=(50, 100), label="SINDy")