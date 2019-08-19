using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.Vtk
using CLIMA.PlanetParameters: planet_radius, R_d, cv_d, MSLP
using CLIMA.MoistThermodynamics: air_temperature, internal_energy, air_density,
                                 gas_constant_air

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
end

include("standalone_euler_model.jl")

struct AcousticWave{T} <: EulerProblem
  domain_height::T
  T0::T
end

function initial_condition!(m::EulerModel, aw::AcousticWave, state::Vars,
                            aux::Vars, x⃗, t)
  @inbounds begin
    DFloat = eltype(x⃗)
    p0 = DFloat(MSLP)

    r = hypot(x⃗...)
    λ = atan(x⃗[2], x⃗[1])
    φ = asin(x⃗[3] / r)
    h = r - DFloat(planet_radius)

    ## Get the reference pressure from the previously defined reference state
    ϕ = geopotential(m.gravity, aux)
    T_ref = aw.T0
    P_ref = p0 * exp(-ϕ / (gas_constant_air(DFloat) * aw.T0))

    ## Define the initial pressure Perturbation
    α, nv, γ = 3, 1, 100
    β = min(DFloat(1), α * acos(cos(φ) * cos(λ)))
    f = (1 + cos(π * β)) / 2
    g = sin(nv * DFloat(π) * h / aw.domain_height)
    dP = γ * f * g

    ## Define the initial pressure and compute the density perturbation
    P = P_ref + dP
    ρ = air_density(T_ref, P)

    ## Define the initial total energy perturbation
    e_int = internal_energy(T_ref)
    ρe = e_int * ρ + ρ * ϕ
    state.ρ = ρ
    state.ρu⃗ = SVector{3, DFloat}(0,0,0)
    state.ρe = ρe
  end
end
gravitymodel(::AcousticWave) = SphereGravity(planet_radius)

function run(mpicomm, ArrayType, problem, topl, N, timeend, DFloat, dt)
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )

  dg = DGModel(EulerModel(problem),
               grid,
               Rusanov(),
               DefaultGradNumericalFlux())

  param = init_ode_param(dg)

  Q = init_ode_state(dg, param, DFloat(0))

  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0

  # Set up the information callback
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end
  step = [0]
  vtkdir = "vtk_acoustic_wave"
  mkpath(vtkdir)
  cbvtk = GenericCallbacks.EveryXSimulationSteps(10) do (init=false)
    outprefix = @sprintf("%s/acoustic_wave_mpirank%04d_step%04d", vtkdir,
                         MPI.Comm_rank(mpicomm), step[1])
    @debug "doing VTK output" outprefix
    writevtk(outprefix, Q, dg, ("ρ", "ρu1", "ρu2", "ρu3", "ρe"))
    pvtuprefix = @sprintf("acoustic_wave_step%04d", step[1])
    prefixes = ntuple(i->
                      @sprintf("%s/acoustic_wave_mpirank%04d_step%04d", vtkdir,
                               i-1, step[1]),
                      MPI.Comm_size(mpicomm))
    writepvtu(pvtuprefix, prefixes, ("ρ", "ρu1", "ρu2", "ρu3", "ρe"))
    step[1] += 1
    nothing
  end

  solve!(Q, lsrk, param; timeend=timeend, callbacks=(cbinfo, cbvtk))

  engf = norm(Q)

  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  """ engf engf/eng0 engf-eng0
  engf
end

using Test
let
  MPI.Initialized() || MPI.Init()
  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
  ll == "WARN"  ? Logging.Warn  :
  ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))
  @static if haspkg("CUDAnative")
    device!(MPI.Comm_rank(mpicomm) % length(devices()))
  end

  polynomialorder = 4
  base_num_elem_horz = 4
  base_num_elem_vert = 4

  timeend = 33 * 60 * 60

  @static if haspkg("CuArrays")
    ArrayType = CuArray
  else
    ArrayType = Array
  end

  lvls = 1
  @testset "$(@__FILE__)" for DFloat in (Float64, )
    aw = AcousticWave(DFloat(10e3), DFloat(300))
    for l = 1:lvls

      num_elem_horz = base_num_elem_horz * 2^(l-1)
      num_elem_vert = base_num_elem_vert * 2^(l-1)
      Rrange = range(DFloat(planet_radius), length = num_elem_vert + 1,
                     stop = planet_radius + aw.domain_height)

      topl = StackedCubedSphereTopology(mpicomm, num_elem_horz, Rrange)

      element_size = (aw.domain_height / num_elem_vert)
      acoustic_speed = 300
      dt = element_size / acoustic_speed / polynomialorder^2

      result = run(mpicomm, ArrayType, aw, topl, polynomialorder, timeend, DFloat, dt)
    end
  end

end
nothing
