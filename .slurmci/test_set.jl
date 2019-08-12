# Test to run on CPU only
cpu_tests = Set(((3, "examples/DGmethods_old/ex_001_periodic_advection.jl"),
                 (3, "examples/DGmethods_old/ex_002_solid_body_rotation.jl"),
                 (3, "test/DGmethods_old/compressible_Navier_Stokes/dycoms.jl"),
                 (3, "test/DGmethods_old/compressible_Navier_Stokes/rtb_visc.jl"),
                ))

# Test to run on GPU only
gpu_tests = Set()

# Test to run on both the CPU and GPU
cpu_gpu_tests = Set(((3, "examples/DGmethods_old/ex_001_periodic_advection.jl"),
                     (3, "examples/DGmethods_old/ex_002_solid_body_rotation.jl"),
                     (3, "examples/DGmethods_old/ex_003_acoustic_wave.jl"),
                     (3, "examples/DGmethods_old/ex_004_nonnegative.jl"),
                     (3, "examples/Microphysics/ex_1_saturation_adjustment.jl"),
                     (3, "examples/Microphysics/ex_2_Kessler.jl"),
                     (3, "test/DGmethods/compressible_Navier_Stokes/mms_bc_atmos.jl"),
                     (3, "test/DGmethods/compressible_Navier_Stokes/mms_bc_dgmodel.jl"),
                     (3, "test/DGmethods_old/Euler/RTB_IMEX.jl"),
                     (3, "test/DGmethods_old/Euler/isentropic_vortex_standalone.jl"),
                     (3, "test/DGmethods_old/Euler/isentropic_vortex_standalone_IMEX.jl"),
                     (3, "test/DGmethods_old/Euler/isentropic_vortex_standalone_aux.jl"),
                     (3, "test/DGmethods_old/Euler/isentropic_vortex_standalone_bc.jl"),
                     (3, "test/DGmethods_old/Euler/isentropic_vortex_standalone_integral.jl"),
                     (3, "test/DGmethods_old/Euler/isentropic_vortex_standalone_source.jl"),
                     (3, "test/DGmethods_old/compressible_Navier_Stokes/mms_bc.jl"),
                     (3, "test/DGmethods_old/conservation/sphere.jl"),
                     (2, "test/DGmethods_old/sphere/advection_sphere_lsrk.jl"),
                     (2, "test/DGmethods_old/sphere/advection_sphere_ssp33.jl"),
                     (2, "test/DGmethods_old/sphere/advection_sphere_ssp34.jl"),
                     (2, "test/LinearSolvers/poisson.jl"),
                    ))

