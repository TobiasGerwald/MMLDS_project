using DataDrivenDiffEq, DataDrivenSparse


function create_polynomial_basis(dim, n)
    @variables t (x(t))[1:dim]#x(t) y(t) z(t)  # Symbolic variables
    #u = x#[x, y, z]
    basis = Basis(polynomial_basis(x, n), x, iv = t)
    return basis
end


function train_SINDy(ode_sol, threshold=1e-1, λ=1e-1; basis = nothing, n = nothing)
    
    ddprob = DataDrivenProblem(ode_sol)
    dim = length(ode_sol.u[1])
    
    if basis === nothing
        basis = create_polynomial_basis(dim, n)
    end
     
    optimiser = STLSQ(threshold, λ) #sparsity cut off threshold, Ridge regression parameter

    ddsol = solve(ddprob, basis, optimiser, options = DataDrivenCommonOptions(digits = 2))
    println(get_basis(ddsol))

    return ddsol

end