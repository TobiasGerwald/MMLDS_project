using OrdinaryDiffEq, DynamicalSystems, CairoMakie

function plot_lyapunov_exp(ode_sol, dims=[3, 4], τs=[1, 15, 17, 19]; Δt = 0.1, k_values = 0:10:100)
    h(u) = u[1,:]
    s = h(ode_sol)

    # Estimate the optimal embedding
    D, τ, E = optimal_traditional_de(s)

    fig = CairoMakie.Figure(figsize = (500, 500))
    ax = CairoMakie.Axis(fig[1, 1], xlabel = L"k \times Δt", ylabel="E(k)")

    for dim in dims, τ in τs  # Try different embedding dimensions and time delays
       data_embedded = embed(s, dim, τ)
       E = lyapunov_from_data(data_embedded, k_values)  # Returns [E(k) for k ∈ k_values]
       λ = linear_region(k_values .* Δt, E)[2]          # Returns the slope of the linear region, i.e. the Lyapunov exponent
       lines!(
           ax, 
           k_values .* Δt, 
           E, 
           label = "dim=$dim,τ=$τ, λ=$(round(λ, digits = 3))",
       )
    end

    axislegend(ax, position = :rb)
    fig
end


loss(x,y) = sum(abs2, x - y)

function MSE_on_lyapunov_time(data, ode_func, t_lyapunov, n_restarts)

    MSE = 0

    for i in 1:n_restarts
        tstart = 0.
        tend = round(t_lyapunov)
        tspan = (tstart, tend)
        x0 = data[i]
        prob = ODEProblem(ode_func, x0, tspan)
        sol_ode = solve(prob, Tsit5(), saveat=tstart:tend)
        MSE += loss(sol_ode, data[i,i+tend])

    end

    MSE /= n_restarts

    return MSE
end