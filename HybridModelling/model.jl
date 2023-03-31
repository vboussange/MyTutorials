using ParametricModels
using ComponentArrays
using LinearAlgebra
using UnPack
using OrdinaryDiffEq
using Graphs
using EcoEvoModelZoo
using Statistics
using SparseArrays
using PythonPlot
using Interpolations
using SciMLSensitivity

# Hyper parameters of the model
const tspan = (0f0, 600f0)
const dt = 3f0
const tsteps = range(300f0, tspan[end], step = dt)

solve_params = (alg = BS3(),
                abstol = 1e-6,
                reltol = 1e-6,
                tspan = (0.0, 600),
                saveat = tsteps,
                verbose=true, # suppresses warnings for maxiters
                maxiters=50_000,
                sensealg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)),
                )