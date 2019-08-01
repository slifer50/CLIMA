module InitialiseDryCases

using CLIMA
using CLIMA.Atmos
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.VariableTemplates
using LinearAlgebra
using StaticArrays
using CLIMA.PlanetParameters

export ProblemDefinition

export Setup

struct ProblemDefinition
  InitialCondition!::Function
  Source!::Function
  BoundaryCondition!::Union{Function, NoFluxBC, InitStateBC}
  TurbulenceModel::Union{Nothing,SmagorinskyLilly, ConstantViscosityWithDivergence}
end


"""
   Density_Current_Init!(state::Vars, aux::Vars, (x,y,z), t)

Initial Condition Function
Initial state definition for the case of the stable, falling thermal bubble. 
3D (2D bubble definition with periodic third dimension). Viscous flow case.
"""

  function Density_Current_Init!(state::Vars, aux::Vars, (x,y,z), t)
    x,y,z = aux.coord.x, aux.coord.y, aux.coord.z
    DF = eltype(state.ρ)
    R_gas::DF         = R_d
    c_p::DF           = cp_d
    c_v::DF           = cv_d
    p0::DF            = MSLP
    gravity::DF       = grav
    γ::DF             = cp_d / cv_d
    # initialise with dry domain 
    xc::DF            = 0
    zc::DF            = 3200
    rx::DF            = 4000
    rz::DF            = 2000
    # perturbation parameters for rising bubble
    r::DF             = sqrt((x - xc)^2/rx^2 + (z - zc)^2/rz^2)
    θ_ref::DF         = 300
    θ_c::DF           = -15.0
    Δθ::DF            = 0.0
    if r <= 1
      Δθ = θ_c * (1 + cospi(r)/2)
    end
    θ                     = θ_ref + Δθ # potential temperature
    π_exner               = 1.0 - gravity / (c_p * θ) * z # exner pressure
    ρ                     = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density
    P                     = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
    T                     = P / (ρ * R_gas) # temperature
    ρu, ρv, ρw            = 0.0 , 0.0 , 0.0  # momentum components
    # energy definitions
    e_kin                 = (ρu^2 + ρv^2 + ρw^2) / (2*ρ)/ ρ
    e_pot                 = gravity * z
    e_int                 = c_v * (T - T_0) #Dry Internal Energy
    ρe_tot                = ρ * (e_int + e_kin + e_pot)  #Total Energy
    state.ρ = ρ
    state.ρu = SVector(ρu,
                       ρv,
                       0)
    state.ρe = ρe_tot
  end

"""
   Density_Current_Source!(state::Vars, aux::Vars, t)

Source Function  for Density Current Bubble (Viscous) Problem
"""

  function Density_Current_Source!(source::Vars, state::Vars, aux::Vars, t::Real)
    x,y,z = aux.coord.x, aux.coord.y, aux.coord.z
    geopotential = -state.ρ * grav
    source.ρu = SVector(0,
                        0,
                        geopotential)
  end

"""
   Rising_Thermal_Bubble_Init!(state::Vars, aux::Vars, (x,y,z), t)

Initial Condition Function
Initial state definition for the case of the stable, rising thermal bubble. 
3D (2D bubble definition with periodic third dimension). Viscous flow case.
"""

  function Rising_Thermal_Bubble_Init!(state::Vars, aux::Vars, (x,y,z), t)
    x,y,z = aux.coord.x, aux.coord.y, aux.coord.z
    DF = eltype(state.ρ)
    R_gas::DF         = R_d
    c_p::DF           = cp_d
    c_v::DF           = cv_d
    p0::DF            = MSLP
    gravity::DF       = grav
    γ::DF             = cp_d / cv_d
    # initialise with dry domain 
    xc::DF            = 750
    zc::DF            = 300
    # perturbation parameters for rising bubble
    r                 = sqrt((x - xc)^2 + (z - zc)^2)
    rc::DF            = 300
    θ_ref::DF         = 300
    θ_c::DF           = 5.0
    Δθ::DF            = 0.0
    if r <= rc 
      Δθ = θ_c * (1 + cospi(r/rc))/2
    end
    θ                     = θ_ref + Δθ # potential temperature
    π_exner               = 1.0 - gravity / (c_p * θ) * z # exner pressure
    ρ                     = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density
    P                     = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
    T                     = P / (ρ * R_gas) # temperature
    ρu, ρv, ρw            = 0.0 , 0.0 , 0.0  # momentum components
    # energy definitions
    e_kin                 = (ρu^2 + ρv^2 + ρw^2) / (2*ρ)/ ρ
    e_pot                 = gravity * z
    e_int                 = c_v * (T - T_0) #Dry Internal Energy
    ρe_tot                = ρ * (e_int + e_kin + e_pot)  #Total Energy
    state.ρ = ρ
    state.ρu = SVector(ρu,
                       ρv,
                       0)
    state.ρe = ρe_tot
  end

"""
   Rising_Thermal_Bubble_Source!(state::Vars, aux::Vars, t)

Source Function  for Rising Thermal Bubble (Viscous) Problem
"""

  function Rising_Thermal_Bubble_Source!(source::Vars, state::Vars, aux::Vars, t::Real)
    x,y,z = aux.coord.x, aux.coord.y, aux.coord.z
    geopotential = -state.ρ * grav
    source.ρu = SVector(0,
                        0,
                        geopotential)
  end
  
  # List of currently available problem setups 
  DensityCurrent   = ProblemDefinition(Density_Current_Init!, 
                                       Density_Current_Source!, 
                                       NoFluxBC(), 
                                       SmagorinskyLilly(Float64(0.15), 50), 
                                       )

  RisingBubble     = ProblemDefinition(Rising_Thermal_Bubble_Init!, 
                                       Rising_Thermal_Bubble_Source!, 
                                       NoFluxBC(), 
                                       SmagorinskyLilly(Float64(0.15),50), 
                                       )

  # User Inputs Problem Name Here  
  Setup = DensityCurrent 
end

