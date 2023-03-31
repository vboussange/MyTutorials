using Flux
using Random
using Statistics
using JLD2

# Hyperparameters and configuration of AR process
@Base.kwdef mutable struct ArgsEco
    seed::Int            = 72                  # Random seed
    # Recurrent net parameters
    opt                  = ADAM                # Optimizer
    η::Float64           = 5e-3                # Learning rate
    epochs::Int          = 1000                 # Number of epochs
    seqlen::Int          = 50                  # Sequence length to use as input
    seqshift::Int        = 1                  # Shift between sequences (see utils.jl)
    verbose::Bool        = true                # Whether we log the results during training or not
end

# Creates training and testing samples according to hyperparameters `args`
function generate_rnn_data(data, args)
    # Create input X and output y (series shifted by 1)
    X, y = data[:, 1:end-1], data[:, 2:end]
    X = vcat(X, water_availability.(tsteps[1:end-1])')
    # Split data into training and testing sets    # Transform data to time series batches and return
    map(x -> batch_timeseries(x, args.seqlen, args.seqshift), (X, y))
end

# Create batches of a time series `X` by splitting the series into
# sequences of length `s`. Each new sequence is shifted by `r` steps.
# When s == r,  the series is split into non-overlapping batches.
function batch_timeseries(X, s::Int, r::Int)
    r > 0 || error("r must be positive")
    # If X is passed in format T×1, reshape it
    if isa(X, AbstractVector)       
        X = permutedims(X)
    end
    T = size(X, 2)
    s ≤ T || error("s cannot be longer than the total series")
    # Ensure uniform sequence lengths by dropping the first observations until
    # the total sequence length matches a multiple of the batchsize
    X = X[:, ((T - s) % r)+1:end]   
    [X[:, t:r:end-s+t] for t ∈ 1:s] # Output
end

function mse_loss(model, x, y)
    # Warm up recurrent model on first observation
    model(x[1])
    # Compute mean squared error loss on the rest of the sequence
    _l = 0f0
    for i in 2:length(x)
        _l += sum((log.(model(x[i])).- log.(y[i])) .^ 2)
    end
    return _l
end

# Trains and outputs the model according to the chosen hyperparameters `args`
function train_model!(model, data, args)

    Xtrain, ytrain = generate_rnn_data(data, args)

    Random.seed!(args.seed)
    # Get data

    opt = Flux.setup(args.opt(args.η), model)
    # Training loop
    for i ∈ 1:args.epochs
        Flux.reset!(model) # Reset hidden state of the recurrent model
        # Compute the gradients of the loss function
        (∇m,) = Flux.gradient(model) do m
            mse_loss(m, Xtrain, ytrain)
        end
        Flux.update!(opt, model, ∇m) # Update model parameters
        if args.verbose && i % 100 == 0 # Log results every 10 epochs
            # Compute loss on train and test set for logging (important: the model must be reset!)
            Flux.reset!(model)
            train_loss = mse_loss(model, Xtrain, ytrain)
            @info "Epoch $i / $(args.epochs), train loss: $(round(train_loss, digits=3))"
        end
    end
end

import ParametricModels.simulate
function simulate(rnn_model::Flux.Chain, init_states, water_availability_range)
    # concat u0 and w0 for warmup and initial state
    u0w0 = vcat(init_states,reshape(water_availability_range[1:2], 1, 2))
    
    Flux.reset!(rnn_model)
    rnn_model(u0w0[:,1]) # warmup
    pred_rnn = rnn_model(u0w0[:,2]) # first prediction
    for w = water_availability_range[3:end]
        uw = vcat(pred_rnn[:,end], w)
        pred_rnn = hcat(pred_rnn,rnn_model(uw))
    end
    return pred_rnn
end