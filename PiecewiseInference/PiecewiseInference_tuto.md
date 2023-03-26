---
featured: false
subtitle: ""
summary: "This blog post discusses the use of PiecewiseInference.jl, a Julia package that enables the use of machine learning to fit complex ecological models on ecological dataset."
date: "2023-03-25"
header-includes:
  - "\\newcommand{\\M}{\\mathcal{M}}"
draft: false
title: "Inverse ecosystem modeling made easy with PiecewiseInference.jl"
tags: []
categories: []
authors: []
lastmod: "2023-03-25"
---


# Inverse ecosystem modeling made easy with PiecewiseInference.jl 

The field of ecology has developed mechanistic ecosystem models embedding fundamental knowledge on how population, species and communities grow, interact and evolve. Yet calibrating
them to fit real-world data is a daunting task. That's why I'm excited to
introduce PiecewiseInference.jl, a new Julia package that provides a
user-friendly and efficient framework for inverse ecosystem modeling. In this
blog post, I will guide you through the main features of PiecewiseInference.jl
and provide a step-by-step tutorial on how to use it with a five-compartment
ecosystem model. Whether you're a seasoned ecologist or a curious data
scientist, I hope this post will inspire you to explore the fascinating world of
ecological modeling and inference.

## Preliminary steps
Let's import the necessary Julia packages for this tutorial. We'll be using the
`EcoEvoModelZoo` package which provides access to a collection of ecosystem
models. We'll also be using `ParametricModels`, which is a wrapper around
`OrdinaryDiffEq` to easily manipulate ODE models, similarly as you would
manipulate a deep learning model. `Graphs` is used to create a directed graph to
represent the food web, and `SparseArrays` is used to store the model parameters
as a sparse matrix. The `OrdinaryDiffEq` package
provides tools for solving ordinary differential equations, while the
`LinearAlgebra` package is used for linear algebraic computations. The `UnPack`
package provides a convenient way to extract fields from structures, and the
`ComponentArrays` package is used to store and manipulate the model parameters
conveniently. Finally, the `PythonCall` package is used to interface with
Python's Matplotlib library for visualization.

```julia
using Graphs
using EcoEvoModelZoo
using ParametricModels
using LinearAlgebra
using UnPack
using OrdinaryDiffEq
using Statistics
using SparseArrays
using ComponentArrays
using PythonPlot
```




## Definition of the forward model

### Defining hyperparameters for the forward simulation of the model.

Next, we define the algorithm used for solving the ODE model. We also define the
absolute tolerance (`abstol`) and relative tolerance (`reltol`) for the solver.
`tspan` is a tuple representing the time range we will simulate the system for,
and `tsteps` is a vector representing the times we want to output the simulated
data.

```julia
alg = BS3()
abstol = 1e-6
reltol = 1e-6
tspan = (0.0, 600)
tsteps = range(300, tspan[end], length=100)
```

```
300.0:3.0303030303030303:600.0
```





### Defining the foodweb structure
We'll define an ecosystem as presented in [Post et al. (2000)](https://www.jstor.org/stable/177129). We will use `SimpleEcosystemModel` from EcoEvoModeZoo.jl, which requires as input a foodweb structure. Let's use a `DiGraph` to represent it. 

```julia
N = 5 # number of compartment

foodweb = DiGraph(N)
add_edge!(foodweb, 2 => 1) # C1 to R1
add_edge!(foodweb, 5 => 4) # C2 to R2
add_edge!(foodweb, 3 => 2) # P to C1
add_edge!(foodweb, 3 => 5) # P to C2
```

```
true
```




The `N` variable specifies the number of
compartments in the model. The `add_edge!` function is used to add edges to the
graph, specifying the flow of resources between compartments.

For fun, let's just plot the foodweb. Here we use the PythonCall and PythonPlot
packages to visualize the food web as a directed graph using `networkx` and `numpy`.
We create a color list for the different species, and then create a directed
graph g_nx with networkx using the adjacency matrix of the food web. We also
specify the position of each node in the graph, and use nx.draw to draw the
graph with


```julia
using PythonCall, PythonPlot
nx = pyimport("networkx")
np = pyimport("numpy")
species_colors = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple"]

g_nx = nx.DiGraph(np.array(adjacency_matrix(foodweb)))
pos = Dict(0 => [0, 0], 1 => [1, 1], 2 => [2, 2], 3 => [4, 0], 4 => [3, 1])

fig, ax = subplots(1)
nx.draw(g_nx, pos, ax=ax, node_color=species_colors, node_size=1000)
display(fig)
```

![](figures/PiecewiseInference_tuto_4_1.png)



### Defining the ecosystem model

Now that we have defined the foodweb structure, we can build the ecosystem
model, which will be a `SimpleEcosystemModel` from `EcoEvoModelZoo`.

The next several functions are required by `SimpleEcosystemModel` and define the
specific dynamics of the model. The `intinsic_growth_rate` function specifies
the intrinsic growth rate of each compartment, while the `carrying_capacity`
function specifies the carrying capacity of each compartment. The `competition`
function specifies the competition between and within compartments, while the
`resource_conversion_efficiency` function specifies the efficiency with which
resources are converted into consumer biomass. The `feeding` function specifies
the feeding interactions between compartments.

```julia
intinsic_growth_rate(p, t) = p.r

function carrying_capacity(p, t)
    @unpack K₁₁ = p
    K = vcat(K₁₁, ones(N - 1))
    return K
end

function competition(u, p, t)
    @unpack A₁₁, A₄₄ = p
    A = spdiagm(vcat(A₁₁, 0, 0, A₄₄, 0))
    return A * u
end

resource_conversion_efficiency(p, t) = ones(N)
```

```
resource_conversion_efficiency (generic function with 1 method)
```




To define the feeding processes, we use `adjacency_matrix` to get the adjacency matrix of the food web. We then use `findnz` to get the row and column indices of the non-zero entries in the adjacency matrix, which we store in `I` and `J`. 

```julia
I, J, _ = findnz(adjacency_matrix(foodweb))
```

```
([2, 3, 5, 3], [1, 2, 4, 5], [1, 1, 1, 1])
```






`I`, `J` are used to define the functional responses of the species.

```julia
function feeding(u, p, t)
    @unpack ω, H₂₁, H₅₄, H₃₂, H₃₅, q₂₁, q₅₄, q₃₂, q₃₅ = p

    # creating foodweb
    W = sparse(I, J, vcat(1.0, ω, 1.0, 1 .- ω))

    # handling time
    H = sparse(I, J, vcat(H₂₁, H₃₂, H₅₄, H₃₅))

    # attack rates
    q = sparse(I, J, vcat(q₂₁, q₃₂, q₅₄, q₃₅))

    return q .* W ./ (one(eltype(u)) .+ q .* H .* (W * u))
end
```

```
feeding (generic function with 1 method)
```





We are done defining the ecological processes. 

#### Defining the ecosystem model parameters for generating a dataset

The parameters for the ecosystem model are defined using a `ComponentArray`. The
`u0_true` variable specifies the initial conditions for the simulation. The
`ModelParams` type from the ParametricModels package is used to specify the
model parameters and simulation settings. Finally, the `SimpleEcosystemModel`
type from the EcoEvoModelZoo package is used to define the ecosystem model.

```julia
p_true = ComponentArray(ω=[0.2],
                        H₂₁=[2.89855],
                        H₅₄=[2.89855],
                        H₃₂=[7.35294],
                        H₃₅=[7.35294],
                        q₂₁=[1.38],
                        q₅₄=[1.38],
                        q₃₂=[0.272],
                        q₃₅=[0.272],
                        r=[1.0, -0.15, -0.08, 1.0, -0.15],
                        K₁₁=[1.0],
                        A₁₁=[1.0],
                        A₄₄=[1.0])

u0_true = [0.77, 0.060, 0.945, 0.467, 0.18]

mp = ModelParams(; p=p_true,
    tspan,
    u0=u0_true,
    alg,
    reltol,
    abstol,
    saveat=tsteps,
    verbose=false, # suppresses warnings for maxiters
    maxiters=50_000
)
model = SimpleEcosystemModel(; mp, intinsic_growth_rate,
    carrying_capacity,
    competition,
    resource_conversion_efficiency,
    feeding)
```

```
`Model` SimpleEcosystemModel
```







Let's run the model to generate a dataset! There is nothing more simple than that. Let's also plot it,
to get a sense of what it looks like.

```julia
data = simulate(model, u0=u0_true) |> Array

# plotting
using PythonPlot;
function plot_time_series(data)
    fig, ax = subplots()
    for i in 1:N
        ax.plot(data[i, :], label="Species $i", color = species_colors[i])
    end
    # ax.set_yscale("log")
    ax.set_ylabel("Species abundance")
    ax.set_xlabel("Time (days)")
    fig.set_facecolor("None")
    ax.set_facecolor("None")
    fig.legend()
    return fig
end

display(plot_time_series(data))
```

![](figures/PiecewiseInference_tuto_9_1.png)




Let's add a bit of noise to the data to simulate experimental errors. We proceed by adding
log normally distributed noise, so that abundance are always positive (negative abundance would not make sense, but could happen when adding normally distributed noise!).


```julia
data = data .* exp.(0.1 * randn(size(data)))

display(plot_time_series(data))
```

![](figures/PiecewiseInference_tuto_10_1.png)



## Inversion with `PiecewiseInference.jl`

Now that we have set up our model and generated some data, we can proceed with
the inverse modelling using PiecewiseInference.jl.

PiecewiseInference.jl allows to perform inversion based on a segmentation method that partitions the data into short time series (segments), each treated independently and matched against simulations of the model considered. The segmentation approach helps to avoid the ill-behaved loss functions that arise from the strong nonlinearities of ecosystem models, when formulation the inference problem.


### Definition of the `InferenceProblem`
We first import the packages required for the inversion. `PiecewiseInference` is the
main package used, but we also need `OptimizationFlux` for the `Adam` optimizer,
and `SciMLSensitivity` to define the sensitivity method used to differentiate
the forward model.

```julia
using PiecewiseInference
using OptimizationFlux
using SciMLSensitivity
```





To initialize the inversion, we set the initial values for the parameters in `p_init`
to those of `p_true` but modify the `ω` parameter.


```julia
p_init = p_true
p_init.ω .= 0.1 #
```

```
1-element view(::Vector{Float64}, 1:1) with eltype Float64:
 0.1
```





Next, we define a loss function `loss_likelihood` that compares the observed data
with the predicted data. Here, we use a simple mean-squared error loss function while log transforming the abundance, since the noise is log-normally distributed.

```julia
loss_likelihood(data, pred, rg) = sum((log.(data) .- log.(pred)) .^ 2)# loss_fn_lognormal_distrib(data, pred, noise_distrib)
```

```
loss_likelihood (generic function with 1 method)
```






We then define the infprob as an `InferenceProblem`, which contains the forward
model, the initial parameter values, and the loss function.

```julia
infprob = InferenceProblem(model, p_init; loss_likelihood)
```

```
PiecewiseInference.InferenceProblem{EcoEvoModelZoo.SimpleEcosystemModel{Par
ametricModels.ModelParams{ComponentArrays.ComponentVector{Float64, Vector{F
loat64}, Tuple{ComponentArrays.Axis{(ω = 1:1, H₂₁ = 2:2, H₅₄ = 3:3, H₃₂ = 4
:4, H₃₅ = 5:5, q₂₁ = 6:6, q₅₄ = 7:7, q₃₂ = 8:8, q₃₅ = 9:9, r = 10:14, K₁₁ =
 15:15, A₁₁ = 16:16, A₄₄ = 17:17)}}}, Tuple{Float64, Int64}, Vector{Float64
}, OrdinaryDiffEq.BS3{typeof(OrdinaryDiffEq.trivial_limiter!), typeof(Ordin
aryDiffEq.trivial_limiter!), Static.False}, Base.Pairs{Symbol, Any, NTuple{
5, Symbol}, NamedTuple{(:reltol, :abstol, :saveat, :verbose, :maxiters), Tu
ple{Float64, Float64, StepRangeLen{Float64, Base.TwicePrecision{Float64}, B
ase.TwicePrecision{Float64}, Int64}, Bool, Int64}}}}, typeof(Main.var"##Wea
veSandBox#312".intinsic_growth_rate), typeof(Main.var"##WeaveSandBox#312".c
arrying_capacity), typeof(Main.var"##WeaveSandBox#312".competition), typeof
(Main.var"##WeaveSandBox#312".resource_conversion_efficiency), typeof(Main.
var"##WeaveSandBox#312".feeding)}, ComponentArrays.ComponentVector{Float64,
 Vector{Float64}, Tuple{ComponentArrays.Axis{(ω = 1:1, H₂₁ = 2:2, H₅₄ = 3:3
, H₃₂ = 4:4, H₃₅ = 5:5, q₂₁ = 6:6, q₅₄ = 7:7, q₃₂ = 8:8, q₃₅ = 9:9, r = 10:
14, K₁₁ = 15:15, A₁₁ = 16:16, A₄₄ = 17:17)}}}, typeof(PiecewiseInference._d
efault_param_prior), typeof(PiecewiseInference._default_loss_u0_prior), typ
eof(Main.var"##WeaveSandBox#312".loss_likelihood), NamedTuple{(:q₃₅, :q₂₁, 
:H₅₄, :q₅₄, :r, :A₄₄, :H₃₂, :q₃₂, :ω, :A₁₁, :K₁₁, :H₂₁, :H₃₅), NTuple{13, t
ypeof(identity)}}, typeof(identity)}(`Model` SimpleEcosystemModel
, (ω = [0.1], H₂₁ = [2.89855], H₅₄ = [2.89855], H₃₂ = [7.35294], H₃₅ = [7.3
5294], q₂₁ = [1.38], q₅₄ = [1.38], q₃₂ = [0.272], q₃₅ = [0.272], r = [1.0, 
-0.15, -0.08, 1.0, -0.15], K₁₁ = [1.0], A₁₁ = [1.0], A₄₄ = [1.0]), Piecewis
eInference._default_param_prior, PiecewiseInference._default_loss_u0_prior,
 Main.var"##WeaveSandBox#312".loss_likelihood, (q₃₅ = identity, q₂₁ = ident
ity, H₅₄ = identity, q₅₄ = identity, r = identity, A₄₄ = identity, H₃₂ = id
entity, q₃₂ = identity, ω = identity, A₁₁ = identity, K₁₁ = identity, H₂₁ =
 identity, H₃₅ = identity), identity)
```






We can now define a callback function `callback` that will be called after each
iteration of the optimization routine. This function is useful for visualizing
the progress of the inversion. It tracks the loss value and plots the
data and the model predictions if the `plotting` variable is set to true.

```julia
info_per_its = 50
include("cb.jl")
function callback(p_trained, losses, pred, ranges)
    # print_param_values(re(p_trained), p_true)
    if length(losses) % info_per_its == 0
        plotting_fit(losses, pred, ranges, data, tsteps)
    end
end
```

```
callback (generic function with 1 method)
```





### `piecewise_MLE` hyperparameters
To use `piecewise_MLE`, the main function of PiecewiseInference  to estimate the parameters that fit the observed data, we need to decide on two critical hyperparameters

- `group_size`: the number of data points that define an interval, or segment. This number is usually small, but should be decided upon the dynamics of the model: to more nonlinear is the model, the lower `group_size` should be.
- `batch_size`: the number of intervals, or segments, to consider on a single epoch. The higher the `batch_size`, the more computationally expensive a single iteration of `piecewise_MLE`, but the faster the convergence.


 It takes in the inference problem, the optimization
algorithm, the batch size, the data, and the callback function.

Another critical parameter to be decided upon is the automatic differentiation backend used to differentiate the ODE model. Two are supported, `Optimization.AutoForwardDiff()` and `Optimization.Autozygote()`.

Simply put, `Optimization.AutoForwardDiff()` is used for forward mode sensitivity analysis, while `Optimization.Autozygote()` is used for backward mode sensitivity analysis. For more information on those, please refer to the documentation of [`Optimization.jl`](https://docs.sciml.ai/Optimization/stable/), which is used by PiecewiseInference.jl under the hood.

Other parameters required by `piecewise_MLE` are

- `optimizers` specifies the optimization algorithm to be used for each batch. We use the `Adam` optimizer, which is the go-to optimizer for most ML projects. It has a learning rate parameter that controls the step size at each iteration. We have chosen a value of `1e-2` because it provides good convergence without causing numerical instability.

- `epochs` specifies the number of epochs to be used for each batch. We chose a value of `1000` because it is sufficient to achieve good convergence, without risking overfitting.

- `verbose_loss` prints the value of the loss function during training.

- `info_per_its` specifies how often the `callback` function should be called during training

```julia
stats = @timed piecewise_MLE(infprob;
                            adtype = Optimization.AutoZygote(),
                            group_size = 11,
                            batchsizes = [3],
                            data = data,
                            tsteps = tsteps,
                            optimizers = [Adam(1e-2)],
                            epochs = [1000],
                            verbose_loss = true,
                            info_per_its = info_per_its,
                            multi_threading = false,
                            cb = callback)
```

```
piecewise_MLE with 100 points and 10 groups.
Current loss after 50 iterations: 2.426858653109929
Current loss after 100 iterations: 1.697283074571371
Current loss after 150 iterations: 1.636661420297555
Current loss after 200 iterations: 1.8973961984383652
Current loss after 250 iterations: 4.114731774366781
Current loss after 300 iterations: 1.9144363789516046
Current loss after 350 iterations: 1.6178591471866994
Current loss after 400 iterations: 1.600272434728539
Current loss after 450 iterations: 1.736842979295901
Current loss after 500 iterations: 2.0445872880232314
Current loss after 550 iterations: 4.7423917166642635
Current loss after 600 iterations: 1.8786764155774613
Current loss after 650 iterations: 1.7829553601282577
Current loss after 700 iterations: 1.767086428230664
Current loss after 750 iterations: 2.081023786053446
Current loss after 800 iterations: 3.3854789538487613
Current loss after 850 iterations: 1.749346081331972
Current loss after 900 iterations: 1.6906955101196752
Current loss after 950 iterations: 1.708488390367206
Current loss after 1000 iterations: 1.648345687011716
(value = `InferenceResult` with model SimpleEcosystemModel
, time = 331.740342375, bytes = 461243841577, gctime = 37.818660081, gcstat
s = Base.GC_Diff(461243841577, 238, 32, 3912109389, 2208143, 2014, 37818660
081, 1639, 1))
```


![](figures/PiecewiseInference_tuto_16_1.png)
![](figures/PiecewiseInference_tuto_16_2.png)
![](figures/PiecewiseInference_tuto_16_3.png)
![](figures/PiecewiseInference_tuto_16_4.png)
![](figures/PiecewiseInference_tuto_16_5.png)
![](figures/PiecewiseInference_tuto_16_6.png)
![](figures/PiecewiseInference_tuto_16_7.png)
![](figures/PiecewiseInference_tuto_16_8.png)
![](figures/PiecewiseInference_tuto_16_9.png)
![](figures/PiecewiseInference_tuto_16_10.png)
![](figures/PiecewiseInference_tuto_16_11.png)
![](figures/PiecewiseInference_tuto_16_12.png)
![](figures/PiecewiseInference_tuto_16_13.png)
![](figures/PiecewiseInference_tuto_16_14.png)
![](figures/PiecewiseInference_tuto_16_15.png)
![](figures/PiecewiseInference_tuto_16_16.png)
![](figures/PiecewiseInference_tuto_16_17.png)
![](figures/PiecewiseInference_tuto_16_18.png)
![](figures/PiecewiseInference_tuto_16_19.png)
![](figures/PiecewiseInference_tuto_16_20.png)



## Conclusion

In this blog post, we have explored how to perform parameter inference in a
dynamical system model using the Julia programming language and the
PiecewiseInference.jl package. We have shown how to use the package to set up an
inference problem, define a loss function, and optimize the model parameters
using a piecewise optimization algorithm. We have also discussed some best
practices, such as setting the right learning rate and batch size and monitoring
the optimization process.

PiecewiseInference.jl provides an efficient and flexible way to perform inference
on complex ecological models, making use of automatic differentiation and
parallel computation. By dividing the time series into smaller pieces,
PiecewiseInference.jl enables the use of more advanced and computationally
intensive inference algorithms that would otherwise be infeasible on larger
datasets.

Furthermore, the integration of PiecewiseInference.jl with EcoEvoModelZoo.jl
offers a powerful toolkit for ecologists and evolutionary biologists to build,
test and refine models that can be fitted to real-world data using
state-of-the-art inference techniques. The combination of theoretical modelling
and machine learning can provide new insights into complex ecological systems,
helping us to better understand and predict the dynamics of biodiversity in a
changing world.

We invite users to explore these packages and contribute to their development,
by adding new models to the EcoEvoModelZoo.jl and improving the inference
methods in PiecewiseInference.jl. With these tools, we can continue to push the
boundaries of ecological modelling and make important strides towards a more
sustainable and biodiverse future.