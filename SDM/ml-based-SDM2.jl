cd(@__DIR__)
using Pkg
Pkg.activate(".")
using Rasters
using GBIF2
using Plots
using DataFrames
using JLD2


# download presence data from GBIF
function download_GBIF_data(species_name)
    country = :CH
    obs = GBIF2.occurrence_search(species_name, country=country, hasCoordinate=true, limit=4000)
    return obs
end

function generate_presence_data(obs, env_var)
    coords = collect((r.decimalLongitude, r.decimalLatitude) for r in obs)

    # retrieve environmtal variable at presence coordinates
    presence_data = DataFrame(extract(env_var, coords))
    y = ones(size(presence_data,1))

    return presence_data, y
end

using Distributions
function generate_absence_data(obs, env_var)
    # we need to generate pseudo absences. here is a python script to do that
    min_lat, max_lat = minimum(obs.decimalLatitude), maximum(obs.decimalLatitude)
    min_long, max_long = minimum(obs.decimalLongitude), maximum(obs.decimalLongitude)

    num_pseudo_absences = size(obs, 1)
    pseud_absence_coords = [(rand(Uniform(min_long, max_long)), rand(Uniform(min_lat, max_lat))) for i = 1:num_pseudo_absences]

    pseud_absence_data = DataFrame(extract(env_var, pseud_absence_coords))
    y = zeros(size(pseud_absence_data,1))
    return pseud_absence_data, y
end

using Random
function generate_training_data(species_name)
    obs = download_GBIF_data(species_name)
    # load environmental variable raster
    env_var = RasterStack("data/CH_CHELSA_BIOCLIM/")[Band(1)]
    presence_data, y_presence = generate_presence_data(obs, env_var)
    pseud_absence_data, y_absence = generate_absence_data(obs, env_var)
    # removing the coordinate column
    predictors = vcat(presence_data, pseud_absence_data)[:,2:end] |> Array{Float32,2} |> adjoint
    y = vcat(y_presence, y_absence) |> Vector{Float32} |> adjoint
    
    sidx = shuffle(1:length(y))

    return predictors[:,sidx], y[sidx]
end

species_name = "Passer montanus"
data = generate_training_data(species_name)

using Plots
using MLUtils

# building a data generator function
split_at = 0.7
train_idx, test_idx = splitobs(1:size(train_data_balanced,2), at=split_at)

X_train, y_train = train_data_balanced[:, train_idx], y[:, train_idx]
X_test, y_test = train_data_balanced[:, test_idx], y[:, test_idx]

# Scale features


# building a neural net
using Flux

# Define model architecture
model = Chain(
    Dense(size(train_data_balanced, 1), 10, relu),
    Dense(10, 1),
    softmax)


# testing model
model(X_train)
y
binarycrossentropy(model(X_train), y_train)


# Train model
import Flux.Data:DataLoader
# using cross entropy
using Flux.Losses

batch_size = 32
num_epochs = 50
opt_state = Flux.setup(Adam(), model)

# Training loop, using the whole data set 1000 times:
losses = []
for epoch in 1:num_epochs
    for batch in Flux.Data.DataLoader((X_train, y_train), batchsize=batch_size, shuffle=true)
        _l, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            binarycrossentropy(model(X_train), y_train)
        end
        Flux.update!(opt_state, model, grads[1])
        push!(losses, _l)  # logging, outside gradient context
        # Evaluate model on test set
    end
    acc = binarycrossentropy(model(X_test), y_test')
    println("Epoch $epoch, accuracy: $acc, loss: $(losses[end])")
end

