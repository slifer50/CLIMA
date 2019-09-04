using MPI
using CLIMA
using Logging
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayTypes = (CuArray, )
else
  const ArrayTypes = (Array, )
end

include("convection_diffusion_model.jl")

struct Pseudo1D{n, α, β} <: ConvectionDiffusionProblem end

function init_velocity_diffusion!(::Pseudo1D{n, α, β}, aux::Vars,
                                  geom::LocalGeometry) where {n, α, β}
  # Direction of flow is n with magnitude α
  aux.u = α * n

  # diffusion of strength β in the n direction
  aux.D = β * n * n'
end

function run(mpicomm, ArrayType, dim, topl, N, timeend, DFloat, dt, n, α, β)

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )
  dg = DGModel(ConvectionDiffusion{dim}(Pseudo1D{n, α, β}()),
               grid,
               Rusanov(),
               DefaultGradNumericalFlux())

  param = init_ode_param(dg)

  #=
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

  solve!(Q, lsrk, param; timeend=timeend, callbacks=(cbinfo, ))
  # solve!(Q, lsrk, param; timeend=timeend, callbacks=(cbinfo, cbvtk))


  # Print some end of the simulation information
  engf = norm(Q)
  Qe = init_ode_state(dg, param, DFloat(timeend))

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
  =#
  0
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
  base_num_elem = 4

  lvls = 4

  @testset "$(@__FILE__)" for ArrayType in ArrayTypes
    for DFloat in (Float64,) #Float32)
      result = zeros(DFloat, lvls)
      for dim = 2:3
        n = SVector(1, 1, dim == 2 ? 0 : 1) / DFloat(sqrt(dim))
        α = DFloat(5 // 2)
        β = DFloat(1 // 1000)
        for l = 1:lvls
          Ne = 2^(l-1) * base_num_elem
          brickrange = ntuple(j->range(DFloat(-1); length=Ne+1, stop=1), dim)
          periodicity = ntuple(j->false, dim)
          topl = BrickTopology(mpicomm, brickrange; periodicity = periodicity)
          dt = 1e-2 / Ne

          timeend = 1

          nsteps = ceil(Int64, timeend / dt)
          dt = timeend / nsteps

          @info (ArrayType, DFloat, dim)
          result[l] = run(mpicomm, ArrayType, dim, topl, polynomialorder,
                          timeend, DFloat, dt, n, α, β)
          # @test result[l] ≈ DFloat(expected_result[dim-1, l])
        end
      end
    end
  end
end

nothing

