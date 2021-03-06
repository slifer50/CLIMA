module DGmethods

using MPI
using ..MPIStateArrays
using ..Mesh.Grids
using ..Mesh.Topologies
using StaticArrays
using ..SpaceMethods
using ..VariableTemplates
using DocStringExtensions
using GPUifyLoops

export BalanceLaw, DGModel, init_ode_param, init_ode_state, node_apply_aux!

include("balancelaw.jl")
include("DGmodel.jl")
include("NumericalFluxes.jl")
include("DGmodel_kernels.jl")

end
