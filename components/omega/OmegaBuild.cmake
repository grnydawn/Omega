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

set(CASEROOT                      "${OMEGA_BUILD_DIR}/e3smcase")

###########################
# Macros                  #
###########################

macro(common)

  option(OMEGA_DEBUG "Turn on error message throwing (default OFF)." OFF)
  option(OMEGA_LOG_FLUSH "Turn on unbuffered logging (default OFF)." OFF)
  option(OMEGA_TEST_CDASH "Turn on CDash support (default ON)." ON)
  option(OMEGA_EXTERNAL_PROF "Integration of Omega timers with external profiling tools (default OFF)." OFF)
  option(OMEGA_MEMCHECK "Wrap unit tests with a per-rank memory-leak checker (standalone only, default OFF)." OFF)
  option(OMEGA_COVERAGE "Instrument the standalone build for host code coverage (standalone only, default OFF)." OFF)

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

# Collect machine and compiler info from CIME
# and detect OMEGA_ARCH and compilers
macro(init_standalone_build)

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

    if(OMEGA_MEMCHECK)
      # Busy-wait OpenMP runtimes can look like a hang under valgrind; make them
      # yield so the leak checker completes.
      file(APPEND ${_EnvScript} "export KMP_BLOCKTIME=0\n")
      file(APPEND ${_EnvScript} "export OMP_WAIT_POLICY=passive\n")
      file(APPEND ${_EnvScript} "export GOMP_SPINCOUNT=0\n\n")
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

# TODO: do we want to use these variables?
#  # Set compiler and linker flags
#  if (CXXFLAGS)
#    separate_arguments(_CXXFLAGS NATIVE_COMMAND ${CXXFLAGS})
#    list(APPEND OMEGA_CXX_FLAGS ${_CXXFLAGS})
#  endif()
#
#  if (LDFLAGS)
#    separate_arguments(_LDFLAGS NATIVE_COMMAND ${LDFLAGS})
#    list(APPEND OMEGA_LINK_OPTIONS ${_LDFLAGS})
#  endif()
#
#  if (SLIBS)
#    separate_arguments(_SLIBS NATIVE_COMMAND ${SLIBS})
#    list(APPEND OMEGA_LINK_OPTIONS ${_SLIBS})
#  endif()

  if(OMEGA_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMEGA_CXX_FLAGS}")
  endif()

#  if(OMEGA_EXE_LINKER_FLAGS)
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OMEGA_EXE_LINKER_FLAGS}")
#  endif()

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

    if(OMEGA_CUDA_FLAGS)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMEGA_CUDA_FLAGS}")
    endif()

    string(FIND "${CMAKE_CXX_FLAGS}" "--ccbin" pos)
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

    if(OMEGA_HIP_FLAGS)
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${OMEGA_HIP_FLAGS}")
    endif()

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

    # add flags from upstream-E3SM
    if(SYCL_FLAGS)
      set(OMEGA_SYCL_FLAGS "${OMEGA_SYCL_FLAGS} ${SYCL_FLAGS}")
    endif()
    if(OMEGA_SYCL_FLAGS)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMEGA_SYCL_FLAGS}")
    endif()
    if(OMEGA_SYCL_EXE_LINKER_FLAGS)
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OMEGA_SYCL_EXE_LINKER_FLAGS}")
    endif()

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

  #TODO: set OMEGA_ARCH according to E3SM variables
  set(OMEGA_ARCH "")
  set(OMEGA_BUILD_MODE "E3SM")

  message(STATUS "OMEGA_CXX_COMPILER = ${OMEGA_CXX_COMPILER}")

endmacro()

##################################
# Set Cmake and Kokkos variables #
##################################
# Analysis tooling. Real bodies land in later tasks; the empty launcher keeps
# add_omega_test() byte-identical while OMEGA_MEMCHECK is OFF or unwired.
# ---------------------------------------------------------------------------
# omega_capability_map(compiler_id arch  out_memtool out_memopts out_gcovtool out_covflags)
#
# Central, side-effect-free lookup for analysis tooling. Keyed on
# CMAKE_CXX_COMPILER_ID (valid only AFTER project()) x OMEGA_ARCH.
#   * Add a machine/arch  -> add one row in the "memcheck tool <- arch" block.
#   * Add a compiler      -> extend the "compiler family -> gcov" classifier.
#   out_memtool : leak-checker executable name, or "" if none wired for this arch
#   out_memopts : ;-list of leak-checker options (must carry the --error-exitcode contract)
#   out_gcovtool: gcov front-end, "gcov" or "llvm-cov gcov"
#   out_covflags: coverage instrumentation flag(s), "--coverage"
# ---------------------------------------------------------------------------
function(omega_capability_map compiler_id arch
         out_memtool out_memopts out_gcovtool out_covflags)

  # compiler family -> gcov front-end
  if(compiler_id MATCHES "^GNU")
    set(_gcov "gcov")                              # g++
  elseif(compiler_id MATCHES "Clang|IntelLLVM")    # clang++, icpx, amdclang++
    set(_gcov "llvm-cov gcov")
  else()
    set(_gcov "gcov")                              # host fallback (nvcc_wrapper/Cray/empty)
  endif()

  set(_covflags "--coverage")   # accepted by GNU, Clang and icpx alike
  set(_memtool "")
  set(_memopts "")

  # memcheck tool <- arch
  if(arch MATCHES "^(SERIAL|OPENMP|THREADS)$")
    set(_memtool "valgrind")
    set(_memopts
      "--tool=memcheck"
      "--leak-check=full"
      "--show-leak-kinds=definite,indirect"
      "--errors-for-leak-kinds=definite,indirect"
      "--error-exitcode=1"
      "--child-silent-after-fork=yes")
    if(NOT arch STREQUAL "SERIAL")
      # threaded runtimes busy-wait; --fair-sched avoids valgrind stalls
      list(APPEND _memopts "--fair-sched=yes")
    endif()
  elseif(arch STREQUAL "CUDA")
    # wired-but-manual (no GPU in Chrysalis CI)
    set(_memtool "compute-sanitizer")
    set(_memopts "--tool=memcheck" "--leak-check=full" "--error-exitcode=1")
  else()
    # HIP / SYCL: no drop-in launcher-based leak tool with an --error-exitcode
    # contract; wired-but-manual (rocgdb / Intel Inspector).
    set(_memtool "")
  endif()

  set(${out_memtool}  "${_memtool}"  PARENT_SCOPE)
  set(${out_memopts}  "${_memopts}"  PARENT_SCOPE)
  set(${out_gcovtool} "${_gcov}"     PARENT_SCOPE)
  set(${out_covflags} "${_covflags}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# setup_memcheck() — STANDALONE-only. Resolve a per-rank leak launcher for the
# active (OMEGA_ARCH x CMAKE_CXX_COMPILER_ID) and publish OMEGA_MEMCHECK_LAUNCHER
# (a ;-list) which add_omega_test() splices before ./exe in each test COMMAND,
# so the tool wraps each MPI RANK (not the launcher). valgrind --error-exitcode=1
# turns a leak into a test failure under a plain `ctest`.
#
# We deliberately do NOT set MEMORYCHECK_COMMAND: CTest's native MemCheck
# prepends it to the WHOLE command, which under MPI would valgrind srun/mpirun,
# not each rank. Missing tool -> WARNING + empty launcher, configure succeeds.
# Call AFTER project() and BEFORE add_subdirectory(test).
# ---------------------------------------------------------------------------
function(setup_memcheck)
  # Always define the launcher; empty expands to ZERO args in add_test().
  set(OMEGA_MEMCHECK_LAUNCHER "" CACHE INTERNAL "per-rank memory-check launcher" FORCE)

  if(NOT OMEGA_MEMCHECK)
    return()
  endif()
  if(NOT "${OMEGA_BUILD_MODE}" STREQUAL "STANDALONE")
    message(WARNING "OMEGA_MEMCHECK is supported only in standalone builds; ignoring.")
    return()
  endif()

  omega_capability_map("${CMAKE_CXX_COMPILER_ID}" "${OMEGA_ARCH}"
                       _mt _mo _ignore_gcov _ignore_cov)

  if("${_mt}" STREQUAL "")
    message(WARNING "OMEGA_MEMCHECK: no leak tool is wired for OMEGA_ARCH=${OMEGA_ARCH}; "
                    "disabling. Use a vendor tool (rocgdb, Intel Inspector) manually.")
    return()
  endif()

  # Resolve fresh each configure (arch may have changed since a prior run).
  unset(OMEGA_MEMCHECK_EXE CACHE)
  find_program(OMEGA_MEMCHECK_EXE NAMES ${_mt})
  if(NOT OMEGA_MEMCHECK_EXE)
    message(WARNING "OMEGA_MEMCHECK: '${_mt}' not found in PATH; disabling memory checking. "
                    "Configure continues; build and tests are unaffected.")
    return()
  endif()

  set(_launch "${OMEGA_MEMCHECK_EXE};${_mo}")
  set(_supp "${OMEGA_SOURCE_DIR}/test/omega.supp")
  if("${_mt}" STREQUAL "valgrind" AND EXISTS "${_supp}")
    list(APPEND _launch "--suppressions=${_supp}")
  endif()

  set(OMEGA_MEMCHECK_LAUNCHER "${_launch}" CACHE INTERNAL "per-rank memory-check launcher" FORCE)
  list(JOIN _mo " " _mo_str)
  message(STATUS "OMEGA_MEMCHECK enabled (${OMEGA_ARCH}/${CMAKE_CXX_COMPILER_ID}): "
                 "${OMEGA_MEMCHECK_EXE} ${_mo_str}")

  # Generate + chmod the helper script (mirrors omega_ctest.sh).
  set(_McScript ${OMEGA_BUILD_DIR}/omega_memcheck.sh)
  file(WRITE  ${_McScript} "#!/usr/bin/env bash\n\n")
  file(APPEND ${_McScript} "source ./omega_env.sh\n\n")
  file(APPEND ${_McScript} "# Each test rank runs under ${OMEGA_MEMCHECK_EXE} (configured at build time).\n")
  file(APPEND ${_McScript} "# A leaking test fails via --error-exitcode=1.\n\n")
  file(APPEND ${_McScript} "ctest --output-on-failure \"$@\"\n\n")
  execute_process(COMMAND chmod +x ${_McScript})
endfunction()

# ---------------------------------------------------------------------------
# setup_coverage() — STANDALONE-only. Instrument Omega src+test (NOT external/,
# already added) for host code coverage. -O0 -g is applied via directory
# COMPILE_OPTIONS, which are emitted AFTER CMAKE_CXX_FLAGS_<CONFIG> on the
# compile line, so -O0 reliably overrides a Release -O3 while preserving the
# named build type's other flags. A single generated 'gcov' wrapper (execs
# 'gcov' or 'llvm-cov gcov') backs both COVERAGE_COMMAND (CTest's parser keys on
# the basename 'gcov') and gcovr. Call AFTER project() + add_subdirectory(external)
# and BEFORE add_subdirectory(src)/include(CTest).
# ---------------------------------------------------------------------------
function(setup_coverage)
  if(NOT OMEGA_COVERAGE)
    return()
  endif()
  if(NOT "${OMEGA_BUILD_MODE}" STREQUAL "STANDALONE")
    message(WARNING "OMEGA_COVERAGE is supported only in standalone builds; ignoring.")
    return()
  endif()

  omega_capability_map("${CMAKE_CXX_COMPILER_ID}" "${OMEGA_ARCH}"
                       _ignore_mt _ignore_mo _gcovtool _covflags)

  message(STATUS "OMEGA_COVERAGE = ON (host-only instrumentation; gcov='${_gcovtool}')")

  # 1. Instrument compile + link for Omega (src + test) only. external/ was
  #    already added, so its targets do not inherit these options.
  separate_arguments(_cov_list NATIVE_COMMAND "${_covflags}")
  add_compile_options(${_cov_list} -O0 -g)
  add_link_options(${_cov_list})

  if(OMEGA_TARGET_DEVICE)
    message(WARNING "OMEGA_COVERAGE instruments HOST code only; device "
                    "(CUDA/HIP/SYCL) kernels are not measured.")
  endif()

  # 2. Uniform 'gcov' wrapper so CTest's basename parser and gcovr both work for
  #    GNU (gcov) and LLVM (llvm-cov gcov). Absolute path captured now.
  set(_GcovWrap ${OMEGA_BUILD_DIR}/gcov)
  file(WRITE  ${_GcovWrap} "#!/usr/bin/env bash\n")
  file(APPEND ${_GcovWrap} "exec ${_gcovtool} \"$@\"\n")
  execute_process(COMMAND chmod +x ${_GcovWrap})

  # 3. CDash / ctest_coverage() wiring (consumed by include(CTest) into
  #    DartConfiguration.tcl, read back by ctest_start() as CTEST_COVERAGE_COMMAND).
  set(COVERAGE_COMMAND "${_GcovWrap}" CACHE FILEPATH "gcov command for ctest_coverage" FORCE)

  # 4. gcovr HTML/text report (optional dependency; manual 'coverage' target).
  find_program(OMEGA_GCOVR_EXE NAMES gcovr)
  if(OMEGA_GCOVR_EXE)
    set(_CovDir ${OMEGA_BUILD_DIR}/coverage)
    add_custom_target(coverage
      COMMAND ${CMAKE_COMMAND} -E make_directory ${_CovDir}
      COMMAND ${OMEGA_GCOVR_EXE}
              --root ${OMEGA_SOURCE_DIR}
              --filter ${OMEGA_SOURCE_DIR}/src/
              --gcov-executable ${_GcovWrap}
              --exclude-unreachable-branches --exclude-throw-branches
              --print-summary
              --txt ${_CovDir}/omega_coverage.txt
              --html-details ${_CovDir}/omega_coverage.html
              ${OMEGA_BUILD_DIR}
      WORKING_DIRECTORY ${OMEGA_BUILD_DIR}
      COMMENT "gcovr: Omega coverage report -> ${_CovDir}"
      VERBATIM)

    # Helper (mirrors omega_ctest.sh): run tests, then build the report.
    set(_CovScript ${OMEGA_BUILD_DIR}/omega_coverage.sh)
    file(WRITE  ${_CovScript} "#!/usr/bin/env bash\n\n")
    file(APPEND ${_CovScript} "source ./omega_env.sh\n\n")
    file(APPEND ${_CovScript} "# NOTE: multi-rank tests write .gcda concurrently; libgcov file-locks and\n")
    file(APPEND ${_CovScript} "# merges per source on a shared node. For cross-node robustness set a\n")
    file(APPEND ${_CovScript} "# per-rank GCOV_PREFIX (see doc/devGuide/Testing.md).\n\n")
    file(APPEND ${_CovScript} "ctest --output-on-failure \"$@\"\n\n")
    file(APPEND ${_CovScript} "cmake --build . --target coverage\n\n")
    execute_process(COMMAND chmod +x ${_CovScript})
  else()
    message(WARNING "OMEGA_COVERAGE: 'gcovr' not found (add via dev-conda.txt); "
                    "skipping HTML report / 'coverage' target. Raw .gcda/.gcno and "
                    "'ctest -T coverage' still work.")
  endif()
endfunction()

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

  if(OMEGA_LOG_TASKS)
    string(TOUPPER "${OMEGA_LOG_TASKS}" _LOG_TASKS)
    add_definitions(-DOMEGA_LOG_TASKS=${_LOG_TASKS})
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

  if("${OMEGA_ARCH}" STREQUAL "CUDA")
    option(Kokkos_ENABLE_CUDA "" ON)
    option(Kokkos_ENABLE_CUDA_LAMBDA "" ON)
    set(OMEGA_TARGET_DEVICE TRUE)
    option(Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC "" OFF)
    set(Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC ${OMEGA_CUDA_MALLOC_ASYNC} CACHE BOOL "" FORCE)

  elseif("${OMEGA_ARCH}" STREQUAL "HIP")
    option(Kokkos_ENABLE_HIP "" ON)
    set(OMEGA_TARGET_DEVICE TRUE)

  elseif("${OMEGA_ARCH}" STREQUAL "SYCL")
    option(Kokkos_ENABLE_SYCL "" ON)
    set(OMEGA_TARGET_DEVICE TRUE)


  elseif("${OMEGA_ARCH}" STREQUAL "OPENMP")
    option(Kokkos_ENABLE_OPENMP "" ON)

  elseif("${OMEGA_ARCH}" STREQUAL "THREADS")
    option(Kokkos_ENABLE_THREADS "" ON)

  else()
    set(OMEGA_ARCH "SERIAL")
    option(Kokkos_ENABLE_SERIAL "" ON)

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
