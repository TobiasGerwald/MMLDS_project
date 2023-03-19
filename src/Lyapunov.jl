using OrdinaryDiffEq, DynamicalSystems, CairoMakie

function plot_lyapunov_exp(ode_sol, dims=[3, 4], τs=[1, 15, 17, 19])
    s = ode_sol[1, :]

# Estimate the optimal embedding
    D, τ, E = optimal_traditional_de(s)



    k_values = 0:10:100  # Integer timesteps k * Δt

    fig = CairoMakie.Figure(figsize = (500, 500))
    x = CairoMakie.Axis(fig[1, 1], xlabel = L"k \times Δt", ylabel="E(k)")

    for dim in dims, τ in τs  # Try different embedding dimensions and time delays
       data_embedded = embed(s, dim, τ)
       E = lyapunov_from_data(data_embedded, k_values)  # Returns [E(k) for k ∈ k_values]
       λ = linear_region(k_values .* Δt, E)[2]          # Returns the slope of the linear region, i.e. the Lyapunov exponent
       lines!(
           ax, 
           k_values .* Δt, 
           E, 
           label = "dim=(dim),τ=(τ), λ=$(round(λ, digits = 3))",
       )
    end

    axislegend(ax, position = :rb)
    fig
end