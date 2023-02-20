import Pkg
Pkg.activate("MMLDS_project")
using MMLDS_project
using OrdinaryDiffEq, NODEData, Plots, Printf

### Create training Data ###

function lorenz(u, p, t)
    x, y, z = u
    σ, ρ, β = p
    dx = σ * (y - x)
    dy = x * (ρ - z) - y
    dz = x * y - β * z
return [dx, dy, dz]
end

p_lorenz = [10, 28, 8/3] 
tspan = (0.,100.)
dt = 0.01

x0 = [1., 1., 1.]

prob = ODEProblem(lorenz, x0, tspan, p_lorenz) 
sol = solve(prob, Tsit5())

train, valid = NODEDataloader(sol, 100; dt=dt, valid_set=0.5)


### Setup Model, Training and Hyperparameter Optimization ###

using Flux, DiffEqSensitivity, Parameters


abstract type AbstractChaoticNDEModel end 

struct ChaoticNDE{P,R,A,K} <: AbstractChaoticNDEModel
    p::P 
    prob::R 
    alg::A
    kwargs::K
end 

function ChaoticNDE(prob; alg=Tsit5(), kwargs...)
    p = prob.p 
    ChaoticNDE{typeof(p), typeof(prob), typeof(alg), typeof(kwargs)}(p, prob, alg, kwargs)
end 

Flux.@functor ChaoticNDE
Flux.trainable(m::ChaoticNDE) = (p=m.p,)

function (m::ChaoticNDE)(X,p=m.p)
    (t, x) = X 
    Array(solve(remake(m.prob; tspan=(t[1],t[end]),u0=x[:,1],p=p), m.alg; saveat=t, m.kwargs...))
end


using Hyperopt, StatsBase

ho = Hyperoptimizer(50,
            N_weights=8:8:64,
            N_hidden_layers=1:4,
            activation=[leakyrelu,swish],
            τ_max=2:4,
#            eta=[1f-2,1f-3,1f-4],
            eta_decrease=[5,10,15])


function hyperOpt(ho, train, valid, N_epochs = 20)

    best_model = []
    best_val_loss = Inf

    for (i,N_weights,N_hidden_layers,activation,τ_max,eta_decrease) in ho #,eta

        hidden_layers = [Flux.Dense(N_weights, N_weights, activation) for i=1:N_hidden_layers]
        nn = Chain(Flux.Dense(3, N_weights, activation), hidden_layers...,  Flux.Dense(N_weights, 3)) |> gpu

        p, re_nn = Flux.destructure(nn)
        neural_ode(u, p, t) = re_nn(p)(u)
        node_prob = ODEProblem(neural_ode, x0, (Float32(0.),Float32(dt)), p)
        model = ChaoticNDE(node_prob)

        loss = Flux.Losses.mse

        η = 1f-3#eta
        opt = Flux.AdamW(η)
        opt_state = Flux.setup(opt, model)

        println("starting training with N_EPOCHS= ",N_epochs, " - N_weights=",N_weights, " - N_hidden_layers=", N_hidden_layers, " - activation=",activation)
        for i_τ = 2:τ_max
    
            N_epochs_i = i_τ == 2 ? 2*Int(ceil(N_epochs/τ_max)) : ceil(N_epochs/τ_max) # N_epochs sets the total amount of epochs 
        
            train_i = NODEDataloader(train, i_τ)
            for i_e = 1:N_epochs_i

                Flux.train!(model, train_i, opt_state) do m, t, x
                    result = m((t,x))
                    loss(result, x)
                end 
            
                if (i_e % eta_decrease) == 0  # reduce the learning rate
                    η /= 2
                    Flux.adjust!(opt_state, η)
                end
            end 

        #GC.gc(true)
        end

        val_loss = 0
        for batch in valid
            _,x = batch
            val_loss += loss(model(batch), x)
        end
        val_loss /= length(valid)

        if val_loss < best_val_loss
            println("update best_model")
            best_model = model
            best_val_loss = val_loss
        end

    end

    return best_model

end

best_model = hyperOpt(ho, train, valid, 20)



### Evaluation

sol_node = solve(best_model.prob, Tsit5())

plot(sol, idxs=[1], tspan=(0, 50), label="lorenz", title="on training data")
plot!(sol_node, idxs=[1], tspan=(0, 50), label="node")

plot(sol, idxs=[1], tspan=(50, 100), label="lorenz", title="on validation data")
plot!(sol_node, idxs=[1], tspan=(50, 100), label="node")
