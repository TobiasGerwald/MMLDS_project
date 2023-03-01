using ReservoirComputing
using ProgressMeter

###
#generate_esn(input_signal, reservoir_size = 1000, spectral_radius = 1.0, sparsity = 0.1, input_scale = 0.1)

#Generate an Echo State Network consisting of the reservoir weights W and the input weights Wᵢₙ.
#Taken from the lecture
###
function generate_esn(input_signal, reservoir_size = 1000, spectral_radius = 1.0, sparsity = 0.1, input_scale = 0.1)
    W = RandSparseReservoir(reservoir_size, radius = spectral_radius, sparsity = sparsity)
    Wᵢₙ = WeightedLayer(scaling = input_scale)
    return ESN(input_signal, reservoir = W, input_layer = Wᵢₙ)
end


#training, val, test Split
function train_val_test_split(data; val_seconds, test_seconds, Δt = 0.1)
    N = size(data, 2)
    N_val = round(Int, val_seconds / Δt)
    N_test = round(Int, test_seconds / Δt)
    
    ind1 = N - N_test - N_val
    ind2 = N - N_test
    
    train_data = data[:, 1:ind1]
    val_data = data[:, ind1+1:ind2]
    test_data = data[:, ind2+1:end]
    
    return train_data, val_data, test_data
end

#train a given esn
function train_esn!(esn, y_target, ridge_param)
    training_method = StandardRidge(ridge_param)
    return train(esn, y_target, training_method)
end

#setup for hyperparameter search
struct ESNHyperparams
    reservoir_size
    spectral_radius
    sparsity
    input_scale
    ridge_param
end

#grid-search for best hyperparams
function cross_validate_esn(train_data, val_data, param_grid)
    best_loss = Inf
    best_params = nothing

    # We want to predict one step ahead, so the input signal is equal to the target signal from the previous step
    u_train = train_data[:, 1:end-1]
    y_train = train_data[:, 2:end]
    iterations = length(param_grid)
    p = Progress(iterations, 1)
    for hyperparams in param_grid        
        # Unpack the hyperparams struct
        (;reservoir_size, spectral_radius, sparsity, input_scale, ridge_param) = hyperparams

        # Generate and train an ESN
        esn = generate_esn(u_train, reservoir_size, spectral_radius, sparsity, input_scale)
        Wₒᵤₜ = train_esn!(esn, y_train, ridge_param)

        # Evaluate the loss on the validation set
        steps_to_predict = size(val_data, 2)
        prediction = esn(Generative(steps_to_predict), Wₒᵤₜ)
        loss = sum(abs2, prediction - val_data)
        
        # Keep track of the best hyperparameter values
        if loss < best_loss
            best_loss = loss
            best_params = hyperparams
            println(best_params)
            @printf "Validation loss = %.1e\n" best_loss
        end
        next!(p)
    end
    
    # Retrain the model using the optimal hyperparameters on both the training and validation data
    # This is necessary because we don't want errors incurred during validation to affect the test error
    (;reservoir_size, spectral_radius, sparsity, input_scale, ridge_param) = best_params
    data = hcat(train_data, val_data)
    u = data[:, 1:end-1]
    y = data[:, 2:end]
    esn = generate_esn(u, reservoir_size, spectral_radius, sparsity, input_scale)
    Wₒᵤₜ = train_esn!(esn, y, ridge_param)
    
    return esn, Wₒᵤₜ
end