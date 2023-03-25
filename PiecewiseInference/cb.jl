#=
Call back function for MiniBatchInference
=#
using PythonCall
plt = pyimport("matplotlib.pyplot")

_cmap = plt.cm.get_cmap("tab20", 20) # only for cb
color_palette = [_cmap(i) for i in 1:20]

function plotting_fit(losses, pred, ranges, data_set_caggreg, tsteps)
    N = size(data_set_caggreg,1)

    prod_to_plot = sortperm(sum(data_set_caggreg,dims=2)[:], rev=true)[1:min(N,20)] #plotting only 20 biggest sectors

    plt.close("all")

    fig, axs = plt.subplots(2, figsize = (5, 7)) # loss, params convergence and 2 time series (ensemble problems)


    # plotting loss
    ax = axs[0]
    ax.plot(1:length(losses), losses, c = "tab:blue", label = "Loss")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("Iterations"); ax.set_ylabel("Loss")
    ax.set_yticklabels([""])


    # plotting time series

    ax = axs[1]
    for g in 1:length(ranges)
        _tsteps = tsteps[ranges[g]]
        if isassigned(pred, g)
            _pred = pred[g]
            for (i,p) in enumerate(prod_to_plot)
                ax.plot(_tsteps, _pred[p,:,1], color = color_palette[i])
            end
        end
        for (i,p) in enumerate(prod_to_plot)
            ax.scatter(_tsteps, 
                        data_set_caggreg[p,ranges[g],1], 
                        # label = sitc_fulllabels[p], 
                        s=5., 
                        color = color_palette[i])
        end
    end

    ax.set_yscale("log")
    fig.tight_layout()
    display(fig)
    return fig, axs
end

function plotting_fit_with_params(losses, pred, ranges, data_set_caggreg, tsteps, θs; yscale = "linear")
    N = size(data_set_caggreg,1)

    prod_to_plot = sortperm(sum(data_set_caggreg,dims=2)[:], rev=true)[1:min(N,20)] #plotting only 20 biggest sectors

    plt.close("all")

    fig, axs = plt.subplots(3, figsize = (5, 7)) # loss, params convergence and 2 time series (ensemble problems)


    # plotting loss
    ax = axs[0]
    ax.plot(1:length(losses), losses, c = "tab:blue", 0)
    ax.set_yscale("log")
    ax.set_xlabel("Iterations"); ax.set_ylabel("Loss")
    ax.set_yticklabels([""])

    # plotting θs
    ax = axs[1]
    ax.plot(1:length(θs), θs, c = "tab:red", )
    ax.set_yscale("log")
    ax.set_xlabel("Iterations"); ax.set_ylabel("Param error")
    ax.set_yticklabels([""])

    # plotting time series

    ax = axs[2]
    for g in 1:length(ranges)
        _tsteps = tsteps[ranges[g]]
        if isassigned(pred, g)
            _pred = pred[g]
            for (i,p) in enumerate(prod_to_plot)
                ax.plot(_tsteps, _pred[p,:,1], color = color_palette[i])
            end
        end
        for (i,p) in enumerate(prod_to_plot)
            ax.scatter(_tsteps, 
                        data_set_caggreg[p,ranges[g],1], 
                        # label = sitc_fulllabels[p], 
                        s=5., 
                        color = color_palette[i])
        end
    end

    ax.set_yscale(yscale)
    fig.tight_layout()
    display(fig)
    return fig, axs
end

function print_param_values(p_trained::NamedTuple, p_true::NamedTuple)
    for k in keys(p_trained)
        println(string(k))
        println("trained value = "); display(p_trained[k])
        println("true value ="); display(p_true[k])
    end
end