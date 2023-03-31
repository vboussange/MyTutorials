using Distributions
using OptimizationFlux
using Bijectors

const species_colors = ["tab:red", "tab:green", "tab:blue"]
const labels_sp = ["Resource", "Consumer", "Predator"]
const u0_bij = bijector(Uniform(5f-3, 1f2)) # seems like stacked is not working with AD

loss_u0_prior(u0_data, u0_pred) = sum((log.(u0_data) .- log.(u0_pred)).^2)

const inference_problem_args = (u0_bij = u0_bij, 
                                loss_u0_prior = loss_u0_prior)

const piecewise_MLE_args = (batchsizes = [10],
                        adtype = Optimization.AutoZygote(),
                        verbose_loss = true,
                        info_per_its = 50,
                        multi_threading = false,)

function plot_time_series(data)
    fig, ax = subplots(figsize=(7,4))
    N = size(data,1)
    for i in 1:size(data,1)
        ax.plot(data[i, :], label=labels_sp[i], color = species_colors[i])
    end
    # ax.set_yscale("log")
    ax.set_ylabel("Species abundance")
    ax.set_xlabel("Time (days)")
    fig.set_facecolor("None")
    ax.set_facecolor("None")
    fig.legend()
    display(fig)
    return fig, ax
end

# using Distributions to facilitate loglikelihood calculations
# Noise should be lognormal, noise_distrib should be of type MvNormal with zero mean
function loss_fn_lognormal_distrib(data, pred)
    if any(pred .<= 0.) # we do not tolerate non-positive ICs -
        return Inf
    elseif size(data) != size(pred) # preventing Zygote to crash
        return Inf
    end

    l = sum((log.(data) .- log.(pred)).^2)

    if l isa Number # preventing any other reason for Zygote to crash
        return l
    else 
        return Inf
    end
end

function plot_foodweb(foodweb, pos, labs)
    g_nx = nx.DiGraph(np.array(adjacency_matrix(foodweb)))

    # shifting indices
    pos = Dict(keys(pos) .- 1 .=> values(pos))
    labs = Dict(keys(labs) .- 1 .=> values(labs))

    fig, ax = subplots(1)
    nx.draw(g_nx, pos, ax=ax, node_color=species_colors, node_size=5000, labels=labs)
    ax.set_facecolor("none")
    fig.set_facecolor("none")
    display(fig)
    return fig, ax
end