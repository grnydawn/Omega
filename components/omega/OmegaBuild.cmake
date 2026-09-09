###########################
# Build Control Variables #
###########################

set(OMEGA_PROJECT_NAME            "OmegaOceanModel")
set(OMEGA_EXE_NAME                "omega.exe")
set(OMEGA_LIB_NAME                "OmegaLib")
set(OMEGA_SOURCE_DIR              ${CMAKE_CURRENT_LIST_DIR})

set(OMEGA_BUILD_MODES             "E3SM" "STANDALONE" "NOT_DEFINED")
set(OMEGA_BUILD_MODE              NOT_DEFINED CACHE STRING "Omega build mode")
set_property(CACHE OMEGA_BUILD_MODE PROPERTY STRINGS ${OMEGA_BUILD_MODES})
set(OMEGA_BUILD_DIR               ${CMAKE_CURRENT_BINARY_DIR})
set(OMEGA_DEFAULT_BUILD_TYPE      Release) # Debug or Release

set(E3SM_ROOT                     "${OMEGA_SOURCE_DIR}/../..")
set(E3SM_CIME_ROOT                "${E3SM_ROOT}/cime")
set(E3SM_CIMECONFIG_ROOT          "${E3SM_ROOT}/cime_config")
set(E3SM_EXTERNALS_ROOT           "${E3SM_ROOT}/externals")

###########################
# Macros                  #
###########################

macro(common)

  option(OMEGA_DEBUG "Turn on error message throwing (default OFF)." OFF)
  option(OMEGA_LOG_FLUSH "Turn on unbuffered logging (default OFF)." OFF)
  option(OMEGA_TEST_CDASH "Turn on CDash support (default ON)." ON)
  option(OMEGA_EXTERNAL_PROF "Integration of Omega timers with external profiling tools (default OFF)." OFF)

  if("${OMEGA_BUILD_TYPE}" STREQUAL "Debug" OR "${OMEGA_BUILD_TYPE}" STREQUAL "DEBUG")
    set(OMEGA_DEBUG ON)
  endif()

  if(NOT DEFINED OMEGA_CXX_FLAGS)
    set(OMEGA_CXX_FLAGS "")
  endif()

  if(NOT DEFINED OMEGA_LINK_OPTIONS)
    set(OMEGA_LINK_OPTIONS "")
  endif()

  set(OMEGA_VECTOR_LENGTH 1 CACHE STRING "Omega vector length")

endmacro()

macro(run_bash_command command outvar)

  execute_process(
	COMMAND bash -c "${command}"
	OUTPUT_VARIABLE ${outvar}
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

endmacro()

macro(cime_xmlquery query outvar)

  run_bash_command("cd ${CASEROOT} && ./xmlquery ${query} --value" ${outvar})

endmacro()

macro(read_cime_config)

  set(NEWCASE_COMMAND "${E3SM_ROOT}/cime/scripts/create_newcase \
    --res T62_oQU120 \
    --compset CMPASO-NYF \
    --handle-preexisting-dirs r \
    --case ${CASEROOT}")

  if(NOT "${OMEGA_CIME_MACHINE}" STREQUAL "")
    set(NEWCASE_COMMAND "${NEWCASE_COMMAND} --machine ${OMEGA_CIME_MACHINE}")
  endif()

  if(NOT "${OMEGA_CIME_COMPILER}" STREQUAL "")
    set(NEWCASE_COMMAND "${NEWCASE_COMMAND} --compiler ${OMEGA_CIME_COMPILER}")
  endif()

  if(NOT "${OMEGA_CIME_PROJECT}" STREQUAL "")
    set(NEWCASE_COMMAND "${NEWCASE_COMMAND} --project ${OMEGA_CIME_PROJECT}")
  endif()

  if(NOT EXISTS ${CASEROOT})
    run_bash_command("${NEWCASE_COMMAND}" NEWCASE_OUTPUT)
  else()
    message(WARNING "Reusing ${CASEROOT}")
  endif()

  run_bash_command("cd ${CASEROOT} && ./case.setup" CASESETUP_OUTPUT)
  run_bash_command("source ${CASEROOT}/.env_mach_specific.sh && env" ENV_OUTPUT)

  string(REPLACE "\n" ";" lines ${ENV_OUTPUT})

  # set env. variables
  foreach(line ${lines})
    string(REGEX MATCH "([A-Za-z_][A-Za-z0-9_]*)=(.*)" ENV_LINE ${line})
    set(ENV_VAR "${CMAKE_MATCH_1}")
    set(ENV_VAL "${CMAKE_MATCH_2}")

    if(NOT "${ENV_VAR}" STREQUAL "")
        set(ENV{${ENV_VAR}} "${ENV_VAL}")
		#message(STATUS "${ENV_VAR}: ${ENV_VAL}")
    endif()
  endforeach()

  # Read .case.run.sh script in case directory
  file(READ "${CASEROOT}/.case.run.sh" CASE_RUN)

  # Convert a string to a list
  string(REPLACE "\n" ";" lines ${CASE_RUN})

  # get mpi launch command-line arguments
  foreach(line ${lines})
    string(FIND ${line} "e3sm.exe" _LINE_FOUND)
    if(NOT _LINE_FOUND EQUAL -1)
        string(REPLACE " " ";" args ${line})
        set(SKIP_ARG FALSE)
        list(GET args 0 OMEGA_MPI_EXEC)
        list(REMOVE_AT args 0)
        set(OMEGA_MPI_ARGS)
        foreach(arg ${args})
            if("${SKIP_ARG}" STREQUAL "TRUE")
                set(SKIP_ARG FALSE)
                continue()
            endif()

            string(FIND "${arg}" "e3sm.exe" _ARG_FOUND)

            if(NOT _ARG_FOUND EQUAL -1)
                break()

            elseif("${arg}" STREQUAL "-n" OR "${arg}" STREQUAL "-N" OR
                   "${arg}" STREQUAL "-c" OR "${arg}" STREQUAL "-np")
                set(SKIP_ARG TRUE)

            else()
                list(APPEND OMEGA_MPI_ARGS "${arg}")
            endif()
        endforeach()
    endif()
  endforeach()

  cime_xmlquery("MPILIB" MPILIB_NAME)
  cime_xmlquery("GMAKE_J" GMAKE_J)
  cime_xmlquery("BUILD_THREADED" BUILD_THREADED)
  cime_xmlquery("THREAD_COUNT" THREAD_COUNT)
  cime_xmlquery("COMPILER" COMPILER)
  cime_xmlquery("MACH" MACH)

  if("${BUILD_THREADED}" STREQUAL "TRUE")
    option(compile_threaded "" ON)
  endif()

  set(SRCROOT "${E3SM_ROOT}")

  include("${CASEROOT}/Macros.cmake")

endmacro()

# Collect the per-machine compiler and linker flags that belong to Omega's own
# code. src/CMakeLists.txt puts these on the OmegaLibFlags interface target,
# which every Omega target links and nothing else does.
#
# Deliberately NOT CMAKE_CXX_FLAGS / CMAKE_EXE_LINKER_FLAGS. Unlike every other
# E3SM component, components/omega vendors third-party projects underneath
# itself -- spdlog, yaml-cpp, Kokkos via EKAT, scorpio, GPTL, pacer, cpptrace
# and GSW-C, see external/CMakeLists.txt -- so a flag set at this directory's
# scope reaches all of them, and every configure-time try_compile in the
# subtree. None of those probes sets CMAKE_TRY_COMPILE_TARGET_TYPE, so each one
# builds and links an executable. Two ways that has bitten, both on Aurora:
#
#   - a probe inherits -fsycl-targets=spir64_gen with no AOT device, and ocloc
#     fails with "Error: Device name missing." Kokkos's own
#     -Xsycl-target-backend cannot help: it goes to KOKKOS_COMPILE_OPTIONS,
#     a target property, which no try_compile ever sees.
#   - scorpio appends -std=c++14 to whatever CMAKE_CXX_FLAGS it inherits
#     (externals/scorpio/src/clib/CMakeLists.txt), so its sources compile as
#     "-fsycl -std=c++14" and fail the SYCL headers' C++17 static assert.
#
# The rule this encodes: a consequence of *which compiler* the subtree uses
# stays in CMAKE_CXX_FLAGS, because the vendored projects are compiled by it
# too; a flag that describes *Omega's code* goes on Omega's targets.
#
# Both build modes call this, so the standalone and E3SM paths cannot drift
# apart on what the machine configuration means. That drift is what left the
# coupled path with no architecture flag handling at all.
macro(omega_collect_machine_flags)

  # Omega's own C++ flags, whatever the architecture.
  if(OMEGA_CXX_FLAGS)
    string(APPEND OMEGA_MACHINE_CXX_FLAGS " ${OMEGA_CXX_FLAGS}")
  endif()

  # OMEGA_<ARCH>_FLAGS is Omega's own variable, from -DOMEGA_<ARCH>_FLAGS= or
  # from a machine file, and is always collected: it was written for Omega's C++
  # compiler by whoever set it.
  #
  # The bare <ARCH>_FLAGS spellings are CIME's, and only SYCL's is collected.
  # They are not one family with one meaning -- each is the flags for that
  # architecture's *device* compiler:
  #
  #   build_model.cmake:78  set(YAKL_CUDA_FLAGS "${CPPDEFS} ${CUDA_FLAGS}")
  #   build_model.cmake:83  set(YAKL_HIP_FLAGS  "${CPPDEFS} ${HIP_FLAGS}")
  #
  # For CUDA and HIP the device compiler is a different program from the C++
  # compiler -- nvcc, hipcc -- so their flags are not C++ flags. chicoma-gpu is
  # the proof: it puts "-ccbin CC -O2 -arch sm_80 --use_fast_math" in the bare
  # CUDA_FLAGS, character for character the same string pm-gpu_gnugpu puts in
  # CMAKE_CUDA_FLAGS, and an E3SM build hands Omega "CC" as CMAKE_CXX_COMPILER
  # (chicoma-gpu_gnugpu.cmake:15,18), which rejects -arch and --use_fast_math.
  # Collecting one spelling while excluding the other would admit the identical
  # content under two names.
  #
  # SYCL is the exception because for SYCL there is no separate device compiler:
  # the C++ compiler is the device compiler. aurora's SYCL_FLAGS is
  # "-fsycl -fsycl-targets=spir64_gen -mlong-double-64" -- C++ flags throughout,
  # with no YAKL consumer in build_model.cmake.
  #
  # CMAKE_<ARCH>_FLAGS is read for no architecture. It configures the CUDA and
  # HIP CMake languages, which Omega never enables (CMakeLists.txt declares
  # LANGUAGES C CXX).
  foreach(_OmegaArch CUDA HIP SYCL)
    if("${OMEGA_ARCH}" STREQUAL "${_OmegaArch}")

      if(${_OmegaArch}_FLAGS AND "${_OmegaArch}" STREQUAL "SYCL")
        set(OMEGA_${_OmegaArch}_FLAGS
            "${OMEGA_${_OmegaArch}_FLAGS} ${${_OmegaArch}_FLAGS}")
      endif()

      if(OMEGA_${_OmegaArch}_FLAGS)
        string(APPEND OMEGA_MACHINE_CXX_FLAGS " ${OMEGA_${_OmegaArch}_FLAGS}")
      endif()

      if(OMEGA_${_OmegaArch}_EXE_LINKER_FLAGS)
        string(APPEND OMEGA_MACHINE_LINK_FLAGS
               " ${OMEGA_${_OmegaArch}_EXE_LINKER_FLAGS}")
      endif()

    endif()
  endforeach()

endmacro()

# Collect machine and compiler info from CIME
# and detect OMEGA_ARCH and compilers
macro(init_standalone_build)

  # A standalone build has no E3SM case to read machine settings from, so it
  # creates a throwaway one (see read_cime_config) and points CASEROOT at it.
  set(CASEROOT "${OMEGA_BUILD_DIR}/e3smcase")

  # get cime configuration
  read_cime_config()

  # find compilers
  if(OMEGA_C_COMPILER)
    find_program(_OMEGA_C_COMPILER ${OMEGA_C_COMPILER})

  elseif("${MPILIB}" STREQUAL "mpi-serial")
    find_program(_OMEGA_C_COMPILER ${SCC})

  else()
    find_program(_OMEGA_C_COMPILER ${MPICC})
  endif()

  if(_OMEGA_C_COMPILER)
    set(OMEGA_C_COMPILER ${_OMEGA_C_COMPILER})

  else()
    message(FATAL_ERROR "C compiler, '${OMEGA_C_COMPILER}', is not found." )
  endif()

  if(OMEGA_CXX_COMPILER)
    find_program(_OMEGA_CXX_COMPILER ${OMEGA_CXX_COMPILER})

  elseif("${MPILIB}" STREQUAL "mpi-serial")
    find_program(_OMEGA_CXX_COMPILER ${SCXX})

  else()
    find_program(_OMEGA_CXX_COMPILER ${MPICXX})
  endif()

  if(_OMEGA_CXX_COMPILER)
    set(OMEGA_CXX_COMPILER ${_OMEGA_CXX_COMPILER})

  else()
    message(FATAL_ERROR "C++ compiler, '${OMEGA_CXX_COMPILER}', is not found." )
  endif()

  if(OMEGA_Fortran_COMPILER)
    find_program(_OMEGA_Fortran_COMPILER ${OMEGA_Fortran_COMPILER})

  elseif("${MPILIB}" STREQUAL "mpi-serial")
    find_program(_OMEGA_Fortran_COMPILER ${SFC})

  else()
    find_program(_OMEGA_Fortran_COMPILER ${MPIFC})
  endif()

  if(_OMEGA_Fortran_COMPILER)
    set(OMEGA_Fortran_COMPILER ${_OMEGA_Fortran_COMPILER})

  else()
    message(FATAL_ERROR "Fortran compiler, '${OMEGA_Fortran_COMPILER}', is not found." )
  endif()

  message(STATUS "OMEGA_C_COMPILER = ${OMEGA_C_COMPILER}")
  message(STATUS "OMEGA_CXX_COMPILER = ${OMEGA_CXX_COMPILER}")
  message(STATUS "OMEGA_Fortran_COMPILER = ${OMEGA_Fortran_COMPILER}")

  # detect OMEGA_ARCH if not provided
  if("${OMEGA_ARCH}" STREQUAL "")

    if(USE_CUDA)
      set(OMEGA_ARCH "CUDA")

    elseif(USE_HIP)
      set(OMEGA_ARCH "HIP")

    elseif(USE_SYCL)
      set(OMEGA_ARCH "SYCL")

    else()

      execute_process(
        COMMAND ${OMEGA_CXX_COMPILER} --version
        RESULT_VARIABLE _CXX_VER_RESULT
        OUTPUT_VARIABLE _CXX_VER_OUTPUT)

      if (_CXX_VER_RESULT EQUAL 0)

        string(REGEX MATCH "HIP|hip"       _HIP_CHECK "${_CXX_VER_OUTPUT}")
        string(REGEX MATCH "AMD|amd"       _AMD_CHECK "${_CXX_VER_OUTPUT}")
        string(REGEX MATCH "NVCC|nvcc"     _NVCC_CHECK "${_CXX_VER_OUTPUT}")
        string(REGEX MATCH "NVIDIA|nvidia" _NVIDIA_CHECK "${_CXX_VER_OUTPUT}")

        if(_HIP_CHECK AND _AMD_CHECK)
          set(OMEGA_ARCH "HIP")

        elseif(_NVCC_CHECK AND _NVIDIA_CHECK)
          set(OMEGA_ARCH "CUDA")

        elseif(compile_threaded)
          set(OMEGA_ARCH "OPENMP")

        else()
          set(OMEGA_ARCH "SERIAL")

        endif()

      elseif(compile_threaded)
        set(OMEGA_ARCH "OPENMP")

      else()
        set(OMEGA_ARCH "SERIAL")

      endif()
    endif()
  endif()

  message(STATUS "OMEGA_ARCH = ${OMEGA_ARCH}")

  # create a env script
  set(_EnvScript ${OMEGA_BUILD_DIR}/omega_env.sh)
  file(WRITE ${_EnvScript}  "#!/usr/bin/env bash\n\n")

  file(APPEND ${_EnvScript} "SCRIPT_DIR=$(cd $(dirname $BASH_SOURCE[0]) && pwd)\n\n")
  file(APPEND ${_EnvScript} "source $SCRIPT_DIR/e3smcase/.env_mach_specific.sh\n\n")
  if("${OMEGA_ARCH}" STREQUAL "OPENMP")
    file(APPEND ${_EnvScript} "export OMP_NUM_THREADS=${THREAD_COUNT}\n\n")
    if(DEFINED ENV{OMP_PROC_BIND})
      file(APPEND ${_EnvScript} "export OMP_PROC_BIND=$ENV{OMP_PROC_BIND}\n\n")
    else()
      file(APPEND ${_EnvScript} "export OMP_PROC_BIND=spread\n\n")
    endif()
    if(DEFINED ENV{OMP_PLACES})
      file(APPEND ${_EnvScript} "export OMP_PLACES=$ENV{OMP_PLACES}\n\n")
    else()
      file(APPEND ${_EnvScript} "export OMP_PLACES=threads\n\n")
    endif()

  endif()

  # create a build script
  set(_BuildScript ${OMEGA_BUILD_DIR}/omega_build.sh)
  file(WRITE ${_BuildScript}  "#!/usr/bin/env bash\n\n")
  file(APPEND ${_BuildScript} "source ./omega_env.sh\n\n")
  file(APPEND ${_BuildScript} "make -j ${GMAKE_J}\n\n")

  # create a run script
  set(_RunScript ${OMEGA_BUILD_DIR}/omega_run.sh)
  file(WRITE ${_RunScript}  "#!/usr/bin/env bash\n\n")
  file(APPEND ${_RunScript} "source ./omega_env.sh\n\n")
  list(JOIN OMEGA_MPI_ARGS " " OMEGA_MPI_ARGS_STR)
  file(APPEND ${_RunScript} "cd test; ${OMEGA_MPI_EXEC} ${OMEGA_MPI_ARGS_STR} -n 8 -- ../src/omega.exe\n\n")

  # create a ctest script
  set(_CtestScript ${OMEGA_BUILD_DIR}/omega_ctest.sh)
  file(WRITE ${_CtestScript}  "#!/usr/bin/env bash\n\n")
  file(APPEND ${_CtestScript} "source ./omega_env.sh\n\n")
  # each test truncates its own log on startup, so this only removes logs left
  # behind by tests that are no longer run
  file(APPEND ${_CtestScript} "rm -f test/logs/*.log\n\n")
  if(OMEGA_DEBUG)
    file(APPEND ${_CtestScript} "ctest --output-on-failure --verbose $* # --rerun-failed\n\n")
  else()
    file(APPEND ${_CtestScript} "ctest --output-on-failure $* # --rerun-failed\n\n")
  endif()

  # create a profile script
  set(_ProfileScript ${OMEGA_BUILD_DIR}/omega_profile.sh)
  file(WRITE ${_ProfileScript}  "#!/usr/bin/env bash\n\n")
  file(APPEND ${_ProfileScript} "source ./omega_env.sh\n\n")
  file(APPEND ${_ProfileScript} "# modify 'OUTFILE' with a path in that the profiler can\n")
  file(APPEND ${_ProfileScript} "# create files such as a path in a scratch file system.\n")

  # copy yaml configuration files
  file(MAKE_DIRECTORY "${OMEGA_BUILD_DIR}/configs")
  file(COPY "${OMEGA_SOURCE_DIR}/configs/Default.yml"
       DESTINATION "${OMEGA_BUILD_DIR}/configs")
  file(COPY "${OMEGA_SOURCE_DIR}/configs/Default.yml"
       DESTINATION "${OMEGA_BUILD_DIR}/test")
  file(RENAME "${OMEGA_BUILD_DIR}/test/Default.yml"
       "${OMEGA_BUILD_DIR}/test/omega.yml")

  # set C and Fortran compilers *before* calling CMake project()
  set(CMAKE_C_COMPILER ${OMEGA_C_COMPILER})
  set(CMAKE_Fortran_COMPILER ${OMEGA_Fortran_COMPILER})

  # Collect the machine's flags for Omega's own targets. See
  # omega_collect_machine_flags() for why these do not go into CMAKE_CXX_FLAGS.
  omega_collect_machine_flags()

  # set CXX compiler *before* calling CMake project()
  if("${OMEGA_ARCH}" STREQUAL "CUDA")

    if(NOT OMEGA_CUDA_COMPILER)
      find_program(OMEGA_CUDA_COMPILER
        "nvcc_wrapper"
        PATHS "${OMEGA_SOURCE_DIR}/../../externals/ekat/extern/kokkos/bin"
      )
    endif()

    if(OMEGA_CUDA_COMPILER)
      message(STATUS "OMEGA_CUDA_COMPILER = ${OMEGA_CUDA_COMPILER}")

    else()
      message(FATAL_ERROR "Cuda compiler is not found." )
    endif()

    set(CMAKE_CXX_COMPILER ${OMEGA_CUDA_COMPILER})
    set(CMAKE_CUDA_HOST_COMPILER ${OMEGA_CXX_COMPILER})

    # -ccbin and -Wno-deprecated-gpu-targets stay in CMAKE_CXX_FLAGS: they are
    # consequences of nvcc_wrapper being this directory's compiler, so the
    # vendored projects under external/ need them too. Guard on CMAKE_CXX_FLAGS
    # alone -- OMEGA_MACHINE_CXX_FLAGS is target-scoped and cannot stand in for
    # this one, and omega_collect_machine_flags() has already dropped any
    # -ccbin that came with the machine's flags. Match "-ccbin", not "--ccbin",
    # which never matched the single-dash spelling the machine files use.
    # nvcc_wrapper falls back to g++ when nobody names a host compiler, and
    # Omega standalone hands it to CMake as CMAKE_CXX_COMPILER directly, so it
    # gets none of the NVCC_WRAPPER_DEFAULT_COMPILER plumbing an E3SM build has
    # (share/build/buildlib.ekat, kokkos_launch_compiler). Name it explicitly.
    # Directory-scoped on purpose: the vendored projects under external/ are
    # compiled by the same wrapper.
    #
    # The probe read "--ccbin" until now, which never matches the one-dash flag,
    # so a -ccbin the user passed in CMAKE_CXX_FLAGS was silently doubled.
    string(FIND "${CMAKE_CXX_FLAGS}" "-ccbin" pos)
    if(${pos} EQUAL -1)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ccbin ${CMAKE_CUDA_HOST_COMPILER}")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-gpu-targets")

    message(STATUS "CMAKE_CUDA_HOST_COMPILER = ${CMAKE_CUDA_HOST_COMPILER}")

    file(APPEND ${_ProfileScript} "OUTFILE=${OMEGA_BUILD_DIR}/nsys_output\n\n")
    file(APPEND ${_ProfileScript} "# load Nsight Systems Profiler\n")
    file(APPEND ${_ProfileScript} "module load Nsight-Systems\n\n")
    file(APPEND ${_ProfileScript} "nsys profile -o \$OUTFILE \\\n")
    file(APPEND ${_ProfileScript} "    --cuda-memory-usage=true --force-overwrite=true \\\n")
    file(APPEND ${_ProfileScript} "    --trace=cuda,nvtx,osrt \\\n")
    file(APPEND ${_ProfileScript} "    ./src/omega.exe 1000")

  elseif("${OMEGA_ARCH}" STREQUAL "HIP")

    if(NOT OMEGA_HIP_COMPILER)
      find_program(OMEGA_HIP_COMPILER "hipcc")
    endif()

    if(OMEGA_HIP_COMPILER)
      message(STATUS "OMEGA_HIP_COMPILER = ${OMEGA_HIP_COMPILER}")

    else()
      message(FATAL_ERROR "hipcc is not found." )
    endif()

    set(CMAKE_HIP_COMPILER ${OMEGA_HIP_COMPILER})
    set(CMAKE_CXX_COMPILER ${OMEGA_CXX_COMPILER})

    if("${MPILIB_NAME}" STREQUAL "mpich")
      if(NOT $ENV{MPICH_CXX})
        set(ENV{MPICH_CXX} ${OMEGA_HIP_COMPILER})
      endif()

    elseif("${MPILIB_NAME}" STREQUAL "openmpi")
      if(NOT $ENV{OMPI_CXX})
        set(ENV{OMPI_CXX} ${OMEGA_HIP_COMPILER})
      endif()

    else()
      message(FATAL_ERROR "'$ENV{MPILIB_NAME}' is not supported yet.")

    endif()

    file(APPEND ${_ProfileScript} "OUTFILE=${OMEGA_BUILD_DIR}/rocprof_output.csv\n")
    file(APPEND ${_ProfileScript} "rocprof --hip-trace --hsa-trace --timestamp on \\\n")
    file(APPEND ${_ProfileScript} "    -o \$OUTFILE ./src/omega.exe 1000")

  elseif("${OMEGA_ARCH}" STREQUAL "SYCL")
    set(CMAKE_CXX_COMPILER ${OMEGA_CXX_COMPILER})

  else()
    set(CMAKE_CXX_COMPILER ${OMEGA_CXX_COMPILER})

  endif()

  execute_process(COMMAND chmod +x ${_EnvScript})
  execute_process(COMMAND chmod +x ${_BuildScript})
  execute_process(COMMAND chmod +x ${_RunScript})
  execute_process(COMMAND chmod +x ${_CtestScript})
  execute_process(COMMAND chmod +x ${_ProfileScript})

  if(KOKKOS_OPTIONS)

    string(REPLACE " " ";" opts ${KOKKOS_OPTIONS})
    foreach(opt ${opts})
      string(REGEX MATCH "-D[ \t]*([A-Za-z_][A-Za-z0-9_]*)=(.*)" KOPT ${opt})
      set(KOPT_VAR "${CMAKE_MATCH_1}")
      set(KOPT_VAL "${CMAKE_MATCH_2}")

      if(NOT "${KOPT_VAR}" STREQUAL "")
        option(${KOPT_VAR} "" ${KOPT_VAL})
      endif()
    endforeach()

    unset(KOKKOS_OPTIONS)

  endif()

  message(STATUS "CMAKE_CXX_COMPILER     = ${CMAKE_CXX_COMPILER}")
  message(STATUS "CMAKE_CXX_FLAGS        = ${CMAKE_CXX_FLAGS}")
  message(STATUS "CMAKE_EXE_LINKER_FLAGS = ${CMAKE_EXE_LINKER_FLAGS}")
  message(STATUS "OMEGA_MACHINE_CXX_FLAGS  = ${OMEGA_MACHINE_CXX_FLAGS}")
  message(STATUS "OMEGA_MACHINE_LINK_FLAGS = ${OMEGA_MACHINE_LINK_FLAGS}")

endmacro()

# set build-control-variables for standalone build
macro(setup_standalone_build)

  if(NOT DEFINED OMEGA_BUILD_TYPE)
    set(OMEGA_BUILD_TYPE ${OMEGA_DEFAULT_BUILD_TYPE})
  endif()

  if( EXISTS ${OMEGA_SOURCE_DIR}/../../components AND
      EXISTS ${OMEGA_SOURCE_DIR}/../../cime AND
      EXISTS ${OMEGA_SOURCE_DIR}/../../cime_config AND
      EXISTS ${OMEGA_SOURCE_DIR}/../../externals)

    set(E3SM_SOURCE_DIR ${OMEGA_SOURCE_DIR}/../../components)

  else()
    # so far, we assume that Omega exists inside of E3SM.
    # However, we leave this else part for later usage.

  endif()

  set(OMEGA_BUILD_MODE "STANDALONE")
  set(OMEGA_BUILD_EXECUTABLE ON)

endmacro()

# set build-control-variables for e3sm build
macro(setup_e3sm_build)

  set(OMEGA_BUILD_TYPE ${E3SM_DEFAULT_BUILD_TYPE})

  set(OMEGA_CXX_COMPILER ${CMAKE_CXX_COMPILER})

  # Detect OMEGA_ARCH from the E3SM/CIME build variables when not provided.
  # USE_CUDA/USE_HIP/USE_SYCL are set by the GPU machine cmake_macros
  if(NOT DEFINED OMEGA_ARCH OR "${OMEGA_ARCH}" STREQUAL "")
    if(USE_CUDA)
      set(OMEGA_ARCH "CUDA")

    elseif(USE_HIP)
      set(OMEGA_ARCH "HIP")

    elseif(USE_SYCL)
      set(OMEGA_ARCH "SYCL")

    elseif(compile_threaded)
      set(OMEGA_ARCH "OPENMP")

    else()
      set(OMEGA_ARCH "SERIAL")

    endif()
  endif()

  # Take the machine's flags for Omega's own targets. See
  # omega_collect_machine_flags() for why these do not go into CMAKE_CXX_FLAGS.
  omega_collect_machine_flags()

  set(OMEGA_BUILD_MODE "E3SM")

  message(STATUS "OMEGA_CXX_COMPILER = ${OMEGA_CXX_COMPILER}")
  message(STATUS "OMEGA_ARCH = ${OMEGA_ARCH}")
  message(STATUS "OMEGA_KOKKOS_OPTIONS = ${KOKKOS_OPTIONS}")
  message(STATUS "OMEGA_MACHINE_CXX_FLAGS = ${OMEGA_MACHINE_CXX_FLAGS}")
  message(STATUS "OMEGA_MACHINE_LINK_FLAGS = ${OMEGA_MACHINE_LINK_FLAGS}")

endmacro()

##################################
# Set Cmake and Kokkos variables #
##################################
macro(update_variables)

  # Set the build type
  set(CMAKE_BUILD_TYPE ${OMEGA_BUILD_TYPE})

  add_definitions(-DOMEGA_BUILD_MODE=${OMEGA_BUILD_MODE})

  if(NOT DEFINED OMEGA_LOG_LEVEL)
    set(OMEGA_LOG_LEVEL "INFO")
  endif()

  if(OMEGA_DEBUG)
    set(OMEGA_LOG_FLUSH ON)
    add_definitions(-DOMEGA_DEBUG -DOMEGA_LOG_LEVEL=1)
  else()
    string(TOUPPER "${OMEGA_LOG_LEVEL}" _LOG_LEVEL)
    if ("${_LOG_LEVEL}" STREQUAL "TRACE")
      add_definitions(-DOMEGA_LOG_LEVEL=0)
    elseif("${_LOG_LEVEL}" STREQUAL "DEBUG")
      add_definitions(-DOMEGA_LOG_LEVEL=1)
    elseif("${_LOG_LEVEL}" STREQUAL "INFO")
      add_definitions(-DOMEGA_LOG_LEVEL=2)
    elseif("${_LOG_LEVEL}" STREQUAL "WARN")
      add_definitions(-DOMEGA_LOG_LEVEL=3)
    elseif("${_LOG_LEVEL}" STREQUAL "ERROR")
      add_definitions(-DOMEGA_LOG_LEVEL=4)
    elseif("${_LOG_LEVEL}" STREQUAL "CRITICAL")
      add_definitions(-DOMEGA_LOG_LEVEL=5)
    elseif("${_LOG_LEVEL}" STREQUAL "OFF")
      add_definitions(-DOMEGA_LOG_LEVEL=6)
    else()
      message(FATAL_ERROR "Unknown log level: '${OMEGA_LOG_LEVEL}'" )
    endif()
  endif()

  if(OMEGA_LOG_FLUSH)
    add_definitions(-DOMEGA_LOG_FLUSH)
  endif()

  if(OMEGA_MEMORY_LAYOUT)
    string(TOUPPER "${OMEGA_MEMORY_LAYOUT}" _LAYOUT)
    add_definitions(-DOMEGA_LAYOUT_${_LAYOUT})
  else()
    add_definitions(-DOMEGA_LAYOUT_RIGHT)
  endif()

  if(OMEGA_TILE_LENGTH)
    add_definitions(-DOMEGA_TILE_LENGTH=${OMEGA_TILE_LENGTH})
  endif()

  message(STATUS "OMEGA_LINK_OPTIONS     = ${OMEGA_LINK_OPTIONS}")

  # check if MPI is supported
  string(CONCAT _TestMPISource
    "#include \"mpi.h\"\n"
    "int main(int argc, char* argv[])\n"
    "{MPI_Init(&argc, &argv)\; return 0\;}\n")
  set(_TestMPISrcFile ${CMAKE_CURRENT_BINARY_DIR}/_testMPI.cpp)
  set(_TestMPIObjFile ${CMAKE_CURRENT_BINARY_DIR}/_testMPI.o)
  file(WRITE ${_TestMPISrcFile}  ${_TestMPISource})

  execute_process(
    COMMAND ${OMEGA_CXX_COMPILER} -c ${_TestMPISrcFile} -o ${_TestMPIObjFile}
    OUTPUT_QUIET ERROR_QUIET
    RESULT_VARIABLE _MPI_TEST_RESULT
    OUTPUT_VARIABLE _MPI_TEST_OUTPUT
    ERROR_VARIABLE _MPI_TEST_ERROR)

  if(NOT OMEGA_DEBUG)
    file(REMOVE ${_TestMPISrcFile})
    file(REMOVE ${_TestMPIObjFile})
  endif()

  if (NOT _MPI_TEST_RESULT EQUAL 0)
    if (_MPI_TEST_RESULT MATCHES "^[-]?[0-9]+$")
      find_package(MPI)

      if(MPI_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${MPI_CXX_INCLUDE_DIRS}")

      else()
        message(FATAL_ERROR "MPI is not found" )
      endif()
    else()
      message(FATAL_ERROR "MPI test failure: ${_MPI_TEST_RESULT}" )
    endif()
  endif()

  if(OMEGA_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${OMEGA_INSTALL_PREFIX})
  endif()

  if(NOT DEFINED OMEGA_MPI_ON_DEVICE)
    option(OMEGA_MPI_ON_DEVICE "Allow device buffers in MPI communication (default ON)." ON)
  endif()

  option(OMEGA_CUDA_MALLOC_ASYNC "Enable CUDA async support (default OFF)." OFF)

  set(OMEGA_TARGET_DEVICE FALSE)
  if("${OMEGA_ARCH}" STREQUAL "CUDA" OR
     "${OMEGA_ARCH}" STREQUAL "HIP"  OR
     "${OMEGA_ARCH}" STREQUAL "SYCL")
    set(OMEGA_TARGET_DEVICE TRUE)
  endif()

  # In a coupled build that also includes EAMxx, EAMxx has already created the
  # Kokkos::kokkos target with the correct per-machine architecture, backend
  # and (for CUDA) compiler launcher. Omega then reuses that target as-is and
  # must not re-set any Kokkos_* options.
  if(NOT TARGET Kokkos::kokkos)

    # In E3SM mode, reuse the per-machine Kokkos settings that the CIME machine
    # configuration already provides through KOKKOS_OPTIONS (set by
    # cime_config/machines/cmake_macros/<machine>.cmake and consumed unchanged
    # by EAMxx/EKAT).
    set(_OMEGA_KOKKOS_ARCH_SET FALSE)
    if("${OMEGA_BUILD_MODE}" STREQUAL "E3SM" AND KOKKOS_OPTIONS)
      string(REPLACE " " ";" _OmegaKokkosOpts "${KOKKOS_OPTIONS}")
      foreach(_OmegaKopt ${_OmegaKokkosOpts})
        string(REGEX MATCH
               "(Kokkos_(ARCH|ENABLE)_[A-Za-z0-9_]+)=([A-Za-z0-9_]+)"
               _OmegaKmatch "${_OmegaKopt}")
        if(CMAKE_MATCH_1)
          set(_OmegaKvar  "${CMAKE_MATCH_1}")
          set(_OmegaKkind "${CMAKE_MATCH_2}")
          set(_OmegaKval  "${CMAKE_MATCH_3}")
          option(${_OmegaKvar} "" ${_OmegaKval})
          if("${_OmegaKkind}" STREQUAL "ARCH")
            if(_OmegaKval) # value form: On/ON/TRUE -> true, OFF -> false
              set(_OMEGA_KOKKOS_ARCH_SET TRUE)
            endif()
          endif()
        endif()
      endforeach()

      # Kokkos treats a variable literally named KOKKOS_OPTIONS as a DEPRECATED
      # option list and hard-errors on it (kokkos_functions.cmake
      # kokkos_deprecated_list, reached from kokkos_setup_build_environment).
      unset(KOKKOS_OPTIONS)
    endif()

    # Enable the Kokkos backend that matches OMEGA_ARCH. option() is a no-op
    # when the backend was already enabled by the KOKKOS_OPTIONS above.
    if("${OMEGA_ARCH}" STREQUAL "CUDA")
      option(Kokkos_ENABLE_CUDA "" ON)
      option(Kokkos_ENABLE_CUDA_LAMBDA "" ON)
      option(Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC "" OFF)
      set(Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC ${OMEGA_CUDA_MALLOC_ASYNC}
          CACHE BOOL "" FORCE)

    elseif("${OMEGA_ARCH}" STREQUAL "HIP")
      option(Kokkos_ENABLE_HIP "" ON)

    elseif("${OMEGA_ARCH}" STREQUAL "SYCL")
      option(Kokkos_ENABLE_SYCL "" ON)

    elseif("${OMEGA_ARCH}" STREQUAL "OPENMP")
      option(Kokkos_ENABLE_OPENMP "" ON)

    elseif("${OMEGA_ARCH}" STREQUAL "THREADS")
      option(Kokkos_ENABLE_THREADS "" ON)

    else()
      set(OMEGA_ARCH "SERIAL")
      option(Kokkos_ENABLE_SERIAL "" ON)

    endif()

    # Fail loudly if Omega must build its own Kokkos for a GPU but no Kokkos
    # architecture was selected (e.g. a machine whose cmake_macros do not carry
    # the arch in KOKKOS_OPTIONS).
    if("${OMEGA_BUILD_MODE}" STREQUAL "E3SM" AND OMEGA_TARGET_DEVICE AND
       NOT _OMEGA_KOKKOS_ARCH_SET)
      message(FATAL_ERROR
        "OMEGA_ARCH=${OMEGA_ARCH} requests a GPU build but no Kokkos_ARCH_* "
        "was provided. Omega is building its own Kokkos here because the "
        "Kokkos::kokkos target does not already exist (no EAMxx in this case). "
        "On machine '${MACH}' the GPU architecture is expected in KOKKOS_OPTIONS "
        "(cime_config/machines/cmake_macros/); add the appropriate Kokkos_ARCH_* "
        "there, or include EAMxx so Omega reuses its Kokkos.")
    endif()

  endif()

  # Belt and braces: drop KOKKOS_OPTIONS in E3SM mode even on the branch where
  # Omega reused an existing Kokkos::kokkos (EAMxx present) and so never entered
  # the parse above. Kokkos hard-errors on this deprecated variable name, and
  # build_eamxx()'s own unset is function-local and does not reach this scope.
  if("${OMEGA_BUILD_MODE}" STREQUAL "E3SM")
    unset(KOKKOS_OPTIONS)
  endif()

  add_definitions(-DOMEGA_ENABLE_${OMEGA_ARCH})

  if(OMEGA_TARGET_DEVICE)
    add_definitions(-DOMEGA_TARGET_DEVICE)
  endif()

  if(OMEGA_MPI_ON_DEVICE)
    add_definitions(-DOMEGA_MPI_ON_DEVICE)
  endif()

  # Include the findParmetis script
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
  find_package(Parmetis REQUIRED)

#  # prints generates all cmake variables
#  get_cmake_property(_variableNames VARIABLES)
#  list (SORT _variableNames)
#  foreach (_variableName ${_variableNames})
#      message(STATUS "${_variableName}=${${_variableName}}")
#  endforeach()

endmacro()



################################
# Verify variable integrity    #
################################
macro(check_setup)

  #message("OMEGA_BUILD_MODE = ${OMEGA_BUILD_MODE}")

  if("${OMEGA_BUILD_MODE}" STREQUAL "E3SM")
    message(STATUS "*** Omega E3SM-component Build ***")

  elseif("${OMEGA_BUILD_MODE}" STREQUAL "STANDALONE")
    message(STATUS "*** Omega Standalone Build ***")

  else()

    message(FATAL_ERROR "OMEGA_BUILD_MODE is neither E3SM nor STANDALONE.")

  endif()

#  if (NOT DEFINED YAKL_ARCH)
#    message(FATAL_ERROR "YAKL_ARCH is not defined.")
#  endif()

endmacro()


################################
# Prepare output               #
################################
macro(wrap_outputs)

  if(OMEGA_INSTALL_PREFIX)

    install(TARGETS ${OMEGA_LIB_NAME}
      LIBRARY DESTINATION "${OMEGA_INSTALL_PREFIX}/lib"
    )

    if(OMEGA_BUILD_EXECUTABLE)
      install(TARGETS ${OMEGA_EXE_NAME}
        RUNTIME DESTINATION "${OMEGA_INSTALL_PREFIX}/bin"
      )
    endif()

  endif()

endmacro()
