# TODO: figure out a better interface for this.
# at the moment we can just pass a function, but we should do something better
# need to figure out how subcomponents will interact.
function atmos_boundarycondition_state!(f::Function, m::AtmosModel,
                                        stateP::Vars, auxP::Vars,
                                        nM, stateM::Vars,
                                        auxM::Vars, bctype, t)
  f(stateP, auxP, nM, stateM, auxM, bctype, t)
end

function atmos_boundarycondition_diffusive!(f::Function, m::AtmosModel,
                                            stateP::Vars, diffP::Vars,
                                            auxP::Vars, nM, stateM::Vars,
                                            diffM::Vars, auxM::Vars, bctype, t)
  f(stateP, diffP, auxP, nM, stateM, diffM, auxM, bctype, t)
end

# lookup boundary condition by face
function atmos_boundarycondition_state!(bctup::Tuple, m::AtmosModel,
                                        stateP::Vars, auxP::Vars, nM,
                                        stateM::Vars, auxM::Vars, bctype, t)  
  atmos_boundarycondition_state!(bctup[bctype], m, stateP, auxP, nM, stateM,
                                 auxM, bctype, t)
end

function atmos_boundarycondition_diffusive!(bctup::Tuple, m::AtmosModel,
                                            stateP::Vars, diffP::Vars,
                                            auxP::Vars, nM, stateM::Vars,
                                            diffM::Vars, auxM::Vars, bctype, t)
  atmos_boundarycondition_diffusive!(bctup[bctype], m, stateP, diffP, auxP, nM,
                                     stateM, diffM, auxM, bctype, t)
end


abstract type BoundaryCondition
end

"""
    NoFluxBC <: BoundaryCondition

Set the momentum at the boundary to be zero.
"""
# TODO: This should be fixed later once BCs are figured out (likely want
# different things here?)
struct NoFluxBC <: BoundaryCondition
end

function atmos_boundarycondition_state!(bc::NoFluxBC, m::AtmosModel,
                                        stateP::Vars, auxP::Vars,
                                        nM, stateM::Vars,
                                        auxM::Vars, bctype, t) 
  DF = eltype(stateM)
  stateP.ρ = stateM.ρ
  stateP.ρu -= 2 * dot(stateM.ρu, nM) * SVector(nM)
end

function atmos_boundarycondition_diffusive!(bc::NoFluxBC, m::AtmosModel,
                                            stateP::Vars, diffP::Vars,
                                            auxP::Vars, nM, stateM::Vars,
                                            diffM::Vars, auxM::Vars, bctype, t) 
  DF = eltype(stateM)
  stateP.ρ = stateM.ρ
  stateP.ρu -= 2 * dot(stateM.ρu, nM) * SVector(nM)
  diffP.ρτ = SVector(DF(0), DF(0), DF(0), DF(0), DF(0), DF(0))
  diffP.moisture.ρd_h_tot = SVector(DF(0), DF(0), DF(0))
end

"""
    InitStateBC <: BoundaryCondition

Set the value at the boundary to match the `init_state!` function. This is
mainly useful for cases where the problem has an explicit solution.
"""
# TODO: This should be fixed later once BCs are figured out (likely want
# different things here?)
struct InitStateBC <: BoundaryCondition
end
function atmos_boundarycondition_state!(bc::InitStateBC, m::AtmosModel,
                                        stateP::Vars, auxP::Vars,
                                        nM, stateM::Vars, auxM::Vars, bctype, t) 
  init_state!(m, stateP, auxP, auxP.coord, t)
end
function atmos_boundarycondition_diffusive!(bc::InitStateBC, m::AtmosModel,
                                            stateP::Vars, diffP::Vars,
                                            auxP::Vars, nM, stateM::Vars,
                                            diffM::Vars, auxM::Vars, bctype, t) 
  init_state!(m, stateP, auxP, auxP.coord, t)
end
