import Pkg
Pkg.activate("MMLDS_project")
using MMLDS_project
using OrdinaryDiffEq, Plots, DataDrivenDiffEq

ode_sol = create_data()
ddsol = train_SINDy(ode_sol, 1e-1, 0, l1_reg = false, n = 4)

ddsol.basis
ddsol.prob.p  #perfect reconstruction if we set the ridge regression penalty to 0

