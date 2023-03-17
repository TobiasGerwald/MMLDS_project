using DataDrivenDiffEq, DataDrivenSparse


function create_polynomial_basis(dim, n)
    @variables t (x(t))[1:dim]#x(t) y(t) z(t)  # Symbolic variables
    #u = x#[x, y, z]
    basis = Basis(polynomial_basis(x, n), x, iv = t)
    return basis
end

function create_basis(basis, dim, n)
    @variables t (x(t))[1:dim]

    if basis == "fourier_basis"
        basis = Basis(fourier_basis(x,n), x, iv=t)
    elseif basis == "sin_basis"
        basis = Basis(sin_basis(x,n), x, iv=t)
    elseif basis == "cos_basis"
        basis = Basis(cos_basis(x,n), x, iv=t)
    else
        basis = Basis(polynomial_basis(x, n), x, iv = t)
    end
    return basis
end


function train_SINDy(ode_sol, threshold=1e-1, λ=1e-1, l1_reg = true; basis = nothing, n = nothing)
    
    ddprob = DataDrivenProblem(ode_sol)
    if ode_sol isa Matrix
        dim, = size(ode_sol)
    else
        dim = length(ode_sol.u[1])
    end

    basis = create_basis(basis, dim, n)
    
    if l1_reg == true
        optimiser = ADMM(threshold, λ)
    else 
        optimiser = STLSQ(threshold, λ)
    end

    ddsol = solve(ddprob, basis, optimiser, options = DataDrivenCommonOptions(digits = 2))
    println(get_basis(ddsol))

    return ddsol
end
