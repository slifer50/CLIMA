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
using CLIMA.Atmos
using CLIMA.VariableTemplates
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.Vtk
using CLIMA.PlanetParameters
using CLIMA.InitialiseDryCases

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayType = CuArray
else
  const ArrayType = Array
end

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

function run(mpicomm,ArrayType, dim, N, brickrange, timeend, DF, dt)
  # ----------------------------------------------------------------
  topl = BrickTopology(mpicomm, brickrange, periodicity = (false, true, false))
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DF,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N
                                          )
  # ----------------------------------------------------------------
  model = AtmosModel(Setup.TurbulenceModel, DryModel(), NoRadiation(),
                     Setup.Source!, Setup.BoundaryCondition!, Setup.InitialCondition!)
  
  dg = DGModel(model,
               grid,
               Rusanov(),
               DefaultGradNumericalFlux())

  param = init_ode_param(dg)

  Q = init_ode_state(dg, param, DF(0))
  
  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)
  
  # ----------------------------------------------------------------
  # Set up the information callback
  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do (s=false)
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
  
  # ----------------------------------------------------------------
  # Output - VTK Format 
  #npoststates = 5
  # _ρ, _u, _v, _w, _E = 1:npoststates
  # postnames = ("ρ", "ρu", "ρv", "ρw", "ρe_tot")
  # postprocessarray = MPIStateArray(dg; nstate=npoststates)
  # step = [0]
  # mkpath("vtk")
  # cbvtk = GenericCallbacks.EveryXSimulationSteps(1000) do (init=false)
  # DGBalanceLawDiscretizations.dof_iteration!(postprocessarray, dg,
  #                                           Q) do R, Q, QV, aux
  #    @inbounds let
  #    end
  #  end
  #  outprefix = @sprintf("vtk/cns_%dD_mpirank%04d_step%04d", dim,
  #                       MPI.Comm_rank(mpicomm), step[1])
  #  @debug "doing VTK output" outprefix
  #  writevtk(outprefix, Q, dg, statenames, postprocessarray, postnames)
  #  step[1] += 1
  #  nothing
  # end
  
  # ----------------------------------------------------------------
  solve!(Q, lsrk, param; timeend=timeend, callbacks=(cbinfo,))
  # ----------------------------------------------------------------
  # Print some end of the simulation information
  engf = norm(Q)
  Qe = init_ode_state(dg, param, DF(timeend))

  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  norm(Q - Qe)            = %.16e
  norm(Q - Qe) / norm(Qe) = %.16e
  """ engf engf/eng0 engf-eng0 errf errf / engfe
  errf
end

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

  result = 0 
  polynomialorder = 4
  DF = Float64
  dim = 3
  Ne = (40,1,20)
  dt = 0.01
  timeend = 0.1
  brickrange = (range(DF(0); length=Ne[1]+1, stop=DF(25600)),
                range(DF(0); length=Ne[2]+1, stop=DF(100)),
                range(DF(0); length=Ne[3]+1, stop=DF(6400)))
  
  @info (ArrayType, DF, dim)
  result = run(mpicomm,ArrayType, dim, polynomialorder, brickrange, timeend, DF, dt)
end


#nothing
