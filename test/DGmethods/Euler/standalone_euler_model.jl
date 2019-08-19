using CLIMA.VariableTemplates
using StaticArrays
using CLIMA.PlanetParameters: grav, cv_d, T_0
using CLIMA.MoistThermodynamics: PhaseDry, air_pressure, soundspeed_air

import CLIMA.DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                        vars_diffusive, flux!, source!, wavespeed,
                        boundarycondition!, gradvariables!, diffusive!,
                        init_aux!, init_state!, init_ode_param, init_ode_state

abstract type EulerProblem end

struct EulerModel{P, G} <: BalanceLaw
  problem::P
  gravity::G
end
function EulerModel(problem)
  gravity = gravitymodel(problem)
  EulerModel{typeof(problem), typeof(gravity)}(problem, gravity)
end

init_state!(m::EulerModel, x...) = initial_condition!(m, m.problem, x...)

function vars_state(::EulerModel, T)
  NamedTuple{(:ρ, :ρu⃗, :ρe), Tuple{T, SVector{3, T}, T}}
end

function vars_aux(m::EulerModel, T)
  @vars begin
    gravity::vars_aux(m.gravity, T)
  end
end
vars_gradient(::EulerModel, T) = Tuple{}
vars_diffusive(::EulerModel, T) = Tuple{}


function flux!(m::EulerModel, flux::Grad, state::Vars, _::Vars, aux::Vars,
               t::Real)

  (ρ, ρu⃗, ρe) = (state.ρ, state.ρu⃗, state.ρe)

  ρinv = 1 / ρ
  u⃗ = ρinv * ρu⃗
  ϕ = geopotential(m.gravity, aux)
  e = ρinv * ρe
  p = air_pressure(PhaseDry(e - u⃗' * u⃗ / 2 - ϕ, ρ))

  # compute the flux!
  flux.ρ  = ρu⃗
  flux.ρu⃗ = ρu⃗ .* u⃗' + p * I
  flux.ρe = u⃗ * (ρe + p)
end

gradvariables!(::EulerModel, _...) = nothing
diffusive!(::EulerModel, _...) = nothing

function source!(m::EulerModel, source::Vars, state::Vars, aux::Vars, t::Real)
  source.ρ = 0
  source.ρu⃗ = @SVector zeros(eltype(source.ρu⃗), 3)
  source.ρe = 0
  geopotential_source!(m.gravity, source, state, aux)
end

function init_aux!(m::EulerModel, aux::Vars, (x1, x2, x3))
  init_aux!(m.gravity, aux, (x1, x2, x3))
end

function wavespeed(m::EulerModel, nM, state::Vars, aux::Vars, t::Real)
  T = eltype(state)

  (ρ, ρu⃗, ρe) = (state.ρ, state.ρu⃗, state.ρe)

  ρinv = 1 / ρ
  u⃗ = ρinv * ρu⃗
  e = ρinv * ρe
  ϕ = geopotential(m.gravity, aux)
  @inbounds n⃗ = SVector{3, T}(nM[1], nM[2], nM[3])
  abs(n⃗' * u⃗) + soundspeed_air(PhaseDry(e - u⃗' * u⃗ / 2 - ϕ, ρ))
end

abstract type GravityModel end
vars_aux(m::GravityModel, T) = @vars(ϕ::T, ∇ϕ::SVector{3, T})
geopotential(::GravityModel, aux) = aux.gravity.ϕ
function geopotential_source!(::GravityModel, source, state, aux)
  source.ρu⃗ -= state.ρ * aux.gravity.∇ϕ
end

struct NoGravity <: GravityModel end
vars_aux(m::NoGravity, T) = @vars()
init_aux!(::NoGravity, _...) = nothing
geopotential(::NoGravity, _...) = 0
geopotential_source!(::NoGravity, _...) = nothing

struct SphereGravity{T} <: GravityModel
  h::T
end
function init_aux!(g::SphereGravity, aux, x⃗)
  x⃗ = SVector(x⃗)
  r = hypot(x⃗...)
  aux.gravity.ϕ = grav * (r-g.h)
  T = eltype(aux.gravity.∇ϕ)
  aux.gravity.∇ϕ = T(grav) * x⃗ / r
end

struct BoxGravity{dim} <: GravityModel end
function init_aux!(::BoxGravity{dim}, aux, x⃗) where dim
  @inbounds aux.gravity.ϕ = grav * x⃗[dim]

  T = eltype(aux.gravity.∇ϕ)
  if dim == 2
    aux.gravity.∇ϕ = SVector{3, T}(0, grav, 0)
  else
    aux.gravity.∇ϕ = SVector{3, T}(0, 0, grav)
  end
end

function boundarycondition!(::EulerModel, stateP::Vars, _, auxP::Vars, normalM,
                            stateM::Vars, _, auxM::Vars, bctype, t)
  if bctype == 1
    nofluxbc!(stateP, normalM, stateM, auxM)
  else
    error("unknown boundary condition type!")
  end
end
function nofluxbc!(stateP, nM, stateM, auxM)
  @inbounds begin
    ρM, ρu⃗M, ρeM = stateM.ρ, stateP.ρu⃗, stateM.ρe

    ## scalars are preserved
    stateP.ρ, stateP.ρe = ρM, ρeM

    ## reflect velocities
    n⃗ = SVector(nM)
    n⃗_ρu⃗M = n⃗' * ρu⃗M
    stateP.ρu⃗ = ρu⃗M - 2n⃗_ρu⃗M * n⃗
  end
end
