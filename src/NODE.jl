using Printf, Flux, DiffEqSensitivity, Parameters, Hyperopt, StatsBase, NODEData


### Setup Model, Training and Hyperparameter ###
#From the lecture
#############################################################
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
#############################################################

null(u,t) = 0

function hyperOpt(ho, ode_sol, x0, dt, N_epochs = 50, rhs_sug = null)#, p_rhs = nothing)

    train, valid = NODEDataloader(ode_sol, 100; dt=dt, valid_set=0.5)

    best_model = []
    best_neural_ode = []
    best_val_loss = Inf

    n_in_out = length(ode_sol.u[1])


    for (i,N_weights,N_hidden_layers,activation,τ_max,eta_decrease,reg) in ho #,eta

        hidden_layers = [Flux.Dense(N_weights, N_weights, activation) for i=1:N_hidden_layers]
        nn = Chain(Flux.Dense(n_in_out, N_weights, activation), hidden_layers...,  Flux.Dense(N_weights, n_in_out)) |> gpu

        p, re_nn = Flux.destructure(nn)
        node(u, p, t) = re_nn(p)(u)


        #p_n = [p_rhs, p]
        function neural_ode(u, p, t)
            #p_rhs, p_node = p
            return rhs_sug(u, t) .+ node(u, p, t) #, p_rhs,
        end
            
        node_prob = ODEProblem(neural_ode, x0, (Float32(0.),Float32(dt)), p)#_n)
        model = ChaoticNDE(node_prob)

        #loss = Flux.Losses.mse
        loss(x,y) = sum(abs2, x - y)

        η = 1f-4
        opt = Flux.AdamW(η)
        opt_state = Flux.setup(opt, model)
        

        println("starting training with N_EPOCHS= ",N_epochs, " - N_weights=",N_weights, " - N_hidden_layers=",N_hidden_layers, " - activation=",activation, " - reg=",reg)
        for i_τ = 20:10:τ_max
    
            N_epochs_i = i_τ == 2 ? 2*Int(ceil(N_epochs/τ_max)) : ceil(N_epochs/τ_max) # N_epochs sets the total amount of epochs 
        
            train_i = NODEDataloader(train, i_τ)
            for i_e = 1:N_epochs_i

                Flux.train!(model, train_i, opt_state) do m, t, x
                    result = m((t,x))
                    loss(result, x) + reg * sum(abs2, model.p)
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
            println("update best_model; new best_val_loss = ", val_loss)
            best_model = model
            best_neural_ode = neural_ode
            best_val_loss = val_loss
        end

    end

    return [best_model, best_neural_ode]

end