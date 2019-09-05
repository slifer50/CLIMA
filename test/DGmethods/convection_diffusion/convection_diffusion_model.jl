using StaticArrays
using CLIMA.VariableTemplates
import CLIMA.DGmethods: BalanceLaw,
                        vars_aux, vars_state, vars_gradient, vars_diffusive,
                        flux!, source!, gradvariables!, diffusive!,
                        init_aux!, init_state!, boundarycondition!,
                        wavespeed, LocalGeometry

abstract type ConvectionDiffusionProblem end
struct ConvectionDiffusion{dim, P} <: BalanceLaw
  problem::P
  function ConvectionDiffusion{dim}(problem::P) where {dim, P <: ConvectionDiffusionProblem}
    new{dim, P}(problem)
  end
end

# Stored in the aux state are:
#   `coord` coordinate points (needed for BCs)
#   `u` convection velocity
#   `D` Diffusion tensor
vars_aux(::ConvectionDiffusion, T) = @vars(coord::SVector{3, T},
                                           u::SVector{3, T},
                                           D::SMatrix{3, 3, T, 9})
#
# Density is only state
vars_state(::ConvectionDiffusion, T) = @vars(ρ::T)

# Take the gradient of density
vars_gradient(::ConvectionDiffusion, T) = @vars(ρ::T)

# The DG auxiliary variable: D ∇ρ
vars_diffusive(::ConvectionDiffusion, T) = @vars(σ::SVector{3,T})

"""
    flux!(m::ConvectionDiffusion, flux::Grad, state::Vars, auxDG::Vars,
          aux::Vars, t::Real)

Computes flux `F` in:

```
∂ρ
-- = - ∇ • (u ρ - σ) = - ∇ • F
∂t
```
Where

 - `u` is the advection velocity
 - `ρ` is the advected quantity
 - `σ` is DG auxiliary variable (`σ = D ∇ ρ` with D being the diffusion tensor)
"""
function flux!(m::ConvectionDiffusion, flux::Grad, state::Vars, auxDG::Vars,
               aux::Vars, t::Real)
  ρ = state.ρ
  u = aux.u
  σ = auxDG.σ
  flux.ρ = u * ρ - σ
end

"""
    gradvariables!(m::ConvectionDiffusion, transform::Vars, state::Vars,
                   aux::Vars, t::Real)

Set the variable to take the gradient of (`ρ` in this case)
"""
function gradvariables!(m::ConvectionDiffusion, transform::Vars, state::Vars,
                        aux::Vars, t::Real)
  transform.ρ = state.ρ
end

"""
    diffusive!(m::ConvectionDiffusion, transform::Vars, state::Vars, aux::Vars,
               t::Real)

Set the variable to take the gradient of (`ρ` in this case)
"""
function diffusive!(m::ConvectionDiffusion, auxDG::Vars, gradvars::Grad,
                    state::Vars, aux::Vars, t::Real)
  ∇ρ = gradvars.ρ
  D = aux.D
  auxDG.σ = D * ∇ρ
end

"""
    source!(m::ConvectionDiffusion, _...)

There is no source in the convection-diffusion model
"""
source!(m::ConvectionDiffusion, _...) = nothing

"""
    wavespeed(m::ConvectionDiffusion, nM, state::Vars, aux::Vars, t::Real)

Wavespeed with respect to vector `nM`
"""
function wavespeed(m::ConvectionDiffusion, nM, state::Vars, aux::Vars, t::Real)
  u = aux.u
  abs(dot(nM, u))
end

"""
    init_aux!(m::ConvectionDiffusion, aux::Vars, geom::LocalGeometry)

initialize the auxiliary state
"""
function init_aux!(m::ConvectionDiffusion, aux::Vars, geom::LocalGeometry)
  aux.coord = geom.coord
  init_velocity_diffusion!(m.problem, aux, geom)
end

function init_state!(m::ConvectionDiffusion, state::Vars, aux::Vars,
                     coords::NTuple, t::Real)
  initial_condition!(m.problem, state, aux, coords, t)
end

function boundarycondition!(m::ConvectionDiffusion,
                            stateP::Vars, diffP::Vars, auxP::Vars,
                            nM,
                            stateM::Vars, diffM::Vars, auxM::Vars,
                            bctype, t)
  # FILL ME!
end
