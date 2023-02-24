using DataDrivenDiffEq, DataDrivenSparse

function train_SINDy(ode_sol, n, threshold=1e-1, λ=1e-1)
    
    ddprob = DataDrivenProblem(ode_sol)
    
    @variables t x(t) y(t) z(t)  # Symbolic variables
    u = [x, y, z]
    basis = Basis(polynomial_basis(u, n), u, iv = t)
     
    optimiser = STLSQ(threshold, λ) #sparsity cut off threshold, Ridge regression parameter

    solve(ddprob, basis, optimiser, options = DataDrivenCommonOptions(digits = 2))

end