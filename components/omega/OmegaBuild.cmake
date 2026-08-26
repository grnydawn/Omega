###############################################################################
# Build Control Variables                                                     #
###############################################################################

set(OMEGA_PROJECT_NAME            "OmegaOceanModel")
set(OMEGA_EXE_NAME                "omega.exe")
set(OMEGA_LIB_NAME                "OmegaLib")
set(OMEGA_SOURCE_DIR              ${CMAKE_CURRENT_LIST_DIR})

set(OMEGA_BUILD_MODES             "E3SM" "STANDALONE" "NOT_DEFINED")
set(OMEGA_BUILD_MODE              NOT_DEFINED CACHE STRING "Omega build mode")
set_property(CACHE OMEGA_BUILD_MODE PROPERTY STRINGS ${OMEGA_BUILD_MODES})

# Every architecture omega_update_variables() knows how to configure a Kokkos
# backend for. An empty OMEGA_ARCH means "detect it"; anything else must appear
# here.
set(OMEGA_ARCHS                   "CUDA" "HIP" "SYCL"
                                  "OPENMP" "THREADS" "SERIAL")
set(OMEGA_BUILD_DIR               ${CMAKE_CURRENT_BINARY_DIR})
set(OMEGA_DEFAULT_BUILD_TYPE      Release) # Debug or Release

set(E3SM_ROOT                     "${OMEGA_SOURCE_DIR}/../..")
set(E3SM_CIME_ROOT                "${E3SM_ROOT}/cime")
set(E3SM_CIMECONFIG_ROOT          "${E3SM_ROOT}/cime_config")
set(E3SM_EXTERNALS_ROOT           "${E3SM_ROOT}/externals")

# CMP0054 is required by omega_read_cime_config(), which compares environment
# variable names against string literals: without it, a name that happens to
# match an existing variable (LESS, MORE, ...) is dereferenced and if() fails
# with "Unknown arguments specified". Set it here rather than relying on the
# caller, so this file is correct no matter who includes it. A macro restores
# the policies of the scope that defined it, so this reaches every macro below.
cmake_policy(SET CMP0054 NEW)

# if(<item> IN_LIST <list>) is used by the validation below to check enumerated
# options against the lists that declare their legal values.
cmake_policy(SET CMP0057 NEW)

###############################################################################
# Build phases                                                                #
#                                                                             #
# The build runs as six ordered phases (see doc/devGuide/CMakeBuild.md). Each #
# phase macro below declares which phase it belongs to, and CMakeLists.txt    #
# opens each phase in turn. The point is that the phases are enforced rather  #
# than merely documented: calling a macro from the wrong phase, or reaching a #
# phase without the variables its predecessor was supposed to produce, is a   #
# configure-time error instead of a wrong build discovered hours later.       #
#                                                                             #
# 1 SETUP      declare options; discover machine/compiler settings; detect    #
# the target architecture. Reads the world, changes nothing.                  #
# 2 TOOLCHAIN  commit the compilers and flags, then call project(). This is   #
# the one phase CMake's own ordering rules force to exist, and                #
# it is a no-op in E3SM mode, where the parent already did it.                #
# 3 UPDATE     derive the CMake and Kokkos variables from the Omega           #
# build-controlling variables.                                                #
# 4 VALIDATE   assert that what phases 1-3 produced is self-consistent,       #
# before anything is built.                                                   #
# 5 BUILD      resolve external dependencies, verify they produced the        #
# targets Omega links, then build the library, executable and                 #
# tests.                                                                      #
# 6 OUTPUT     emit the developer scripts, copy configs, install.             #
###############################################################################

set(OMEGA_BUILD_PHASES "SETUP" "TOOLCHAIN" "UPDATE" "VALIDATE" "BUILD" "OUTPUT")

# Open a phase. Verifies that phases run in order and that the phase just
# finished produced the variables the next phases rely on.
macro(omega_begin_phase _OmegaPhaseNum _OmegaPhaseName)

  # OMEGA_BUILD_PHASES is 0-indexed; phase numbers are 1-based for readability.
  math(EXPR _OmegaPhaseIdx "${_OmegaPhaseNum} - 1")
  list(GET OMEGA_BUILD_PHASES ${_OmegaPhaseIdx} _OmegaExpectedName)
  if(NOT "${_OmegaExpectedName}" STREQUAL "${_OmegaPhaseName}")
    message(FATAL_ERROR
      "Omega build phase ${_OmegaPhaseNum} is '${_OmegaExpectedName}', not "
      "'${_OmegaPhaseName}'. OMEGA_BUILD_PHASES and CMakeLists.txt disagree.")
  endif()

  get_property(_OmegaPrevPhase GLOBAL PROPERTY OMEGA_CURRENT_PHASE)
  if(NOT DEFINED _OmegaPrevPhase OR "${_OmegaPrevPhase}" STREQUAL "")
    set(_OmegaPrevPhase 0)
  endif()
  math(EXPR _OmegaExpectedPrev "${_OmegaPhaseNum} - 1")
  if(NOT ${_OmegaPrevPhase} EQUAL ${_OmegaExpectedPrev})
    message(FATAL_ERROR
      "Omega build phases ran out of order: entering phase ${_OmegaPhaseNum} "
      "(${_OmegaPhaseName}) from phase ${_OmegaPrevPhase}.")
  endif()

  set_property(GLOBAL PROPERTY OMEGA_CURRENT_PHASE ${_OmegaPhaseNum})

  list(LENGTH OMEGA_BUILD_PHASES _OmegaNumPhases)
  message(STATUS
    "Omega build phase ${_OmegaPhaseNum}/${_OmegaNumPhases}: "
    "${_OmegaPhaseName}")

endmacro()

# Guard placed at the top of every phase macro. Turns "this macro was called
# from the wrong place" into an immediate, self-describing error.
macro(omega_require_phase _OmegaPhaseNum _OmegaWho)

  get_property(_OmegaCurPhase GLOBAL PROPERTY OMEGA_CURRENT_PHASE)
  if(NOT "${_OmegaCurPhase}" STREQUAL "${_OmegaPhaseNum}")
    math(EXPR _OmegaWantIdx "${_OmegaPhaseNum} - 1")
    list(GET OMEGA_BUILD_PHASES ${_OmegaWantIdx} _OmegaWantName)
    message(FATAL_ERROR
      "${_OmegaWho}() belongs to Omega build phase ${_OmegaPhaseNum} "
      "(${_OmegaWantName}) but was called from phase '${_OmegaCurPhase}'.")
  endif()

endmacro()

# Exit assertion for a phase: every listed variable must be defined and
# non-empty by now. Use it to make the hand-off between phases explicit.
macro(omega_phase_produces _OmegaWho)

  foreach(_OmegaVar ${ARGN})
    if(NOT DEFINED ${_OmegaVar} OR "${${_OmegaVar}}" STREQUAL "")
      message(FATAL_ERROR
        "${_OmegaWho}: ${_OmegaVar} is unset or empty at the end of this "
        "phase, but later phases require it.")
    endif()
  endforeach()

endmacro()

###############################################################################
# Macros                                                                      #
###############################################################################

macro(omega_common)

  omega_require_phase(1 omega_common)

  option(OMEGA_DEBUG "Turn on error message throwing (default OFF)." OFF)
  option(OMEGA_LOG_FLUSH "Turn on unbuffered logging (default OFF)." OFF)
  option(OMEGA_TEST_CDASH "Turn on CDash support (default ON)." ON)
  option(OMEGA_EXTERNAL_PROF
         "Integrate Omega timers with external profiling tools (default OFF)."
         OFF)

  if(NOT DEFINED OMEGA_CXX_FLAGS)
    set(OMEGA_CXX_FLAGS "")
  endif()

  if(NOT DEFINED OMEGA_LINK_OPTIONS)
    set(OMEGA_LINK_OPTIONS "")
  endif()

  set(OMEGA_VECTOR_LENGTH 1 CACHE STRING "Omega vector length")

  # Normalize a user-supplied OMEGA_ARCH here, at the very start, because every
  # later test on it is a case-sensitive STREQUAL.
  if(NOT "${OMEGA_ARCH}" STREQUAL "")
    string(TOUPPER "${OMEGA_ARCH}" OMEGA_ARCH)
    if(NOT "${OMEGA_ARCH}" IN_LIST OMEGA_ARCHS)
      string(REPLACE ";" ", " _OmegaArchList "${OMEGA_ARCHS}")
      message(FATAL_ERROR
        "Unknown OMEGA_ARCH '${OMEGA_ARCH}'. Expected one of: "
        "${_OmegaArchList}.")
    endif()
  endif()

endmacro()

macro(omega_run_bash_command command outvar)

  execute_process(
  COMMAND bash -c "${command}"
  OUTPUT_VARIABLE ${outvar}
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

endmacro()

macro(omega_cime_xmlquery query outvar)

  omega_run_bash_command(
    "cd ${CASEROOT} && ./xmlquery ${query} --value" ${outvar})

endmacro()

macro(omega_read_cime_config)

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
    omega_run_bash_command("${NEWCASE_COMMAND}" NEWCASE_OUTPUT)
  else()
    message(WARNING "Reusing ${CASEROOT}")
  endif()

  omega_run_bash_command("cd ${CASEROOT} && ./case.setup" CASESETUP_OUTPUT)
  omega_run_bash_command(
    "source ${CASEROOT}/.env_mach_specific.sh && env" ENV_OUTPUT)

  # Quote the input.
  string(REPLACE "\n" ";" lines "${ENV_OUTPUT}")

  # set env. variables
  foreach(line ${lines})
    string(REGEX MATCH "^([A-Za-z_][A-Za-z0-9_]*)=(.*)$" ENV_LINE "${line}")
    set(ENV_VAR "${CMAKE_MATCH_1}")
    set(ENV_VAL "${CMAKE_MATCH_2}")

    if(NOT "${ENV_VAR}" STREQUAL "")
        set(ENV{${ENV_VAR}} "${ENV_VAL}")
    #message(STATUS "${ENV_VAR}: ${ENV_VAL}")
    endif()
  endforeach()

  # Read .case.run.sh script in case directory
  file(READ "${CASEROOT}/.case.run.sh" CASE_RUN)

  # Convert a string to a list (quoted, for the reason given above)
  string(REPLACE "\n" ";" lines "${CASE_RUN}")

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

  omega_cime_xmlquery("MPILIB" MPILIB_NAME)
  omega_cime_xmlquery("GMAKE_J" GMAKE_J)
  omega_cime_xmlquery("BUILD_THREADED" BUILD_THREADED)
  omega_cime_xmlquery("THREAD_COUNT" THREAD_COUNT)
  omega_cime_xmlquery("COMPILER" COMPILER)
  omega_cime_xmlquery("MACH" MACH)

  if("${BUILD_THREADED}" STREQUAL "TRUE")
    option(compile_threaded "" ON)
  endif()

  set(SRCROOT "${E3SM_ROOT}")

  # Macros.cmake dispatches to the per-machine cmake_macros, several of which
  # branch on MPILIB (impi on chrysalis, mpi-serial elsewhere) and on DEBUG.
  # CIME normally supplies both with -D on the cmake command line; a standalone
  # build includes Macros.cmake directly, so without this those blocks never
  # fire and Omega silently resolves a different MPI stack than the case would
  # have used. Only set what the caller has not already provided.
  if(NOT DEFINED MPILIB)
    set(MPILIB "${MPILIB_NAME}")
  endif()
  if(NOT DEFINED DEBUG)
    if("${OMEGA_BUILD_TYPE}" STREQUAL "Debug" OR
       "${OMEGA_BUILD_TYPE}" STREQUAL "DEBUG")
      set(DEBUG TRUE)
    else()
      set(DEBUG FALSE)
    endif()
  endif()

  include("${CASEROOT}/Macros.cmake")

endmacro()

# Collect machine and compiler info from CIME and detect OMEGA_ARCH and the
# compilers. Discovery only.
macro(omega_init_standalone_build)

  omega_require_phase(1 omega_init_standalone_build)

  # A standalone build has no E3SM case to read machine settings from, so it
  # creates a throwaway one (see omega_read_cime_config) and points CASEROOT
  # at it.
  set(CASEROOT "${OMEGA_BUILD_DIR}/e3smcase")

  # get cime configuration
  omega_read_cime_config()

  # find compilers
  if(OMEGA_C_COMPILER)
    find_program(_OMEGA_C_COMPILER ${OMEGA_C_COMPILER})

  elseif("${MPILIB_NAME}" STREQUAL "mpi-serial")
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

  elseif("${MPILIB_NAME}" STREQUAL "mpi-serial")
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

  elseif("${MPILIB_NAME}" STREQUAL "mpi-serial")
    find_program(_OMEGA_Fortran_COMPILER ${SFC})

  else()
    find_program(_OMEGA_Fortran_COMPILER ${MPIFC})
  endif()

  if(_OMEGA_Fortran_COMPILER)
    set(OMEGA_Fortran_COMPILER ${_OMEGA_Fortran_COMPILER})

  else()
    message(FATAL_ERROR
      "Fortran compiler, '${OMEGA_Fortran_COMPILER}', is not found.")
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

  omega_phase_produces(omega_init_standalone_build
    OMEGA_C_COMPILER OMEGA_CXX_COMPILER OMEGA_Fortran_COMPILER OMEGA_ARCH)

endmacro()


###############################################################################
# PHASE 2: TOOLCHAIN                                                          #
#                                                                             #
# Commit the compilers and flags that phase 1 discovered. Everything here has #
# to happen before project(), which is the reason this phase exists as its own#
# step at all. It is standalone-only: in E3SM mode the parent already selected#
# the toolchain and called project() (components/CMakeLists.txt).             #
###############################################################################
macro(omega_commit_toolchain)

  omega_require_phase(2 omega_commit_toolchain)

  # set C and Fortran compilers *before* calling CMake project()
  set(CMAKE_C_COMPILER ${OMEGA_C_COMPILER})
  set(CMAKE_Fortran_COMPILER ${OMEGA_Fortran_COMPILER})

  if(OMEGA_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMEGA_CXX_FLAGS}")
  endif()

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
      set(CMAKE_CXX_FLAGS
          "${CMAKE_CXX_FLAGS} -ccbin ${CMAKE_CUDA_HOST_COMPILER}")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-gpu-targets")

    message(STATUS "CMAKE_CUDA_HOST_COMPILER = ${CMAKE_CUDA_HOST_COMPILER}")

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

    # Point the MPI compiler wrapper at hipcc, but only if the machine has not
    # already done so.
    if("${MPILIB_NAME}" STREQUAL "mpich")
      if(NOT DEFINED ENV{MPICH_CXX} OR "$ENV{MPICH_CXX}" STREQUAL "")
        set(ENV{MPICH_CXX} ${OMEGA_HIP_COMPILER})
      endif()

    elseif("${MPILIB_NAME}" STREQUAL "openmpi")
      if(NOT DEFINED ENV{OMPI_CXX} OR "$ENV{OMPI_CXX}" STREQUAL "")
        set(ENV{OMPI_CXX} ${OMEGA_HIP_COMPILER})
      endif()

    else()
      # ${MPILIB_NAME}, not $ENV{MPILIB_NAME}: this is the CMake variable set
      # by omega_cime_xmlquery above, and there is no environment variable of
      # that name, so the diagnostic used to name no MPI library at all.
      message(FATAL_ERROR
        "MPI library '${MPILIB_NAME}' is not supported yet for OMEGA_ARCH=HIP.")

    endif()

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
      set(CMAKE_EXE_LINKER_FLAGS
          "${CMAKE_EXE_LINKER_FLAGS} ${OMEGA_SYCL_EXE_LINKER_FLAGS}")
    endif()

  else()
    set(CMAKE_CXX_COMPILER ${OMEGA_CXX_COMPILER})

  endif()

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
macro(omega_setup_standalone_build)

  omega_require_phase(1 omega_setup_standalone_build)

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

# Recover per-machine settings from the case's Macros.cmake that E3SM does not
# make visible to Omega.
function(omega_read_e3sm_macros)

  include("${CASEROOT}/Macros.cmake")

  set(KOKKOS_OPTIONS "${KOKKOS_OPTIONS}" PARENT_SCOPE)

  # USE_SYCL must come from here too.
  set(USE_SYCL "${USE_SYCL}" PARENT_SCOPE)

endfunction()

# set build-control-variables for e3sm build
macro(omega_setup_e3sm_build)

  omega_require_phase(1 omega_setup_e3sm_build)

  set(OMEGA_BUILD_TYPE ${E3SM_DEFAULT_BUILD_TYPE})

  set(OMEGA_CXX_COMPILER ${CMAKE_CXX_COMPILER})

  # Recover the per-machine settings that E3SM does not propagate into this
  # scope (KOKKOS_OPTIONS, USE_SYCL). This must run BEFORE the arch detection
  # below, which reads USE_SYCL. CASEROOT here is the real case root CIME passed
  # in with -DCASEROOT=.
  if(CASEROOT AND EXISTS "${CASEROOT}/Macros.cmake")
    omega_read_e3sm_macros()
  endif()

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

  set(OMEGA_BUILD_MODE "E3SM")

  message(STATUS "OMEGA_CXX_COMPILER = ${OMEGA_CXX_COMPILER}")
  message(STATUS "OMEGA_ARCH = ${OMEGA_ARCH}")
  message(STATUS "OMEGA_KOKKOS_OPTIONS = ${KOKKOS_OPTIONS}")

endmacro()

###############################################################################
# PHASE 3: UPDATE - derive the CMake and Kokkos variables                     #
###############################################################################
macro(omega_update_variables)

  omega_require_phase(3 omega_update_variables)

  # Derive OMEGA_DEBUG now that both build modes have finalized
  # OMEGA_BUILD_TYPE (see the note in omega_common(), which is where this used
  # to live and where it could not work for a coupled DEBUG case).
  if("${OMEGA_BUILD_TYPE}" STREQUAL "Debug" OR
     "${OMEGA_BUILD_TYPE}" STREQUAL "DEBUG")
    set(OMEGA_DEBUG ON)
  endif()

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
    option(OMEGA_MPI_ON_DEVICE
           "Allow device buffers in MPI communication (default ON)." ON)
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
        "On machine '${MACH}' the GPU architecture is expected in "
        "KOKKOS_OPTIONS (cime_config/machines/cmake_macros/), add the "
        "appropriate Kokkos_ARCH_* "
        "there, or include EAMxx so Omega reuses its Kokkos.")
    endif()

  endif()

  # Drop KOKKOS_OPTIONS in E3SM mode even on the branch where
  # Omega reused an existing Kokkos::kokkos (EAMxx present) and so never entered
  # the parse above.
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
  # Not REQUIRED: FindParmetis.cmake now reports through
  # find_package_handle_standard_args, so a miss would abort right here with
  # only that one message. omega_check_setup() (phase 4) enforces it instead,
  # which lets a misconfigured build report every problem it has in one pass.
  find_package(Parmetis)

#  # prints generates all cmake variables
#  get_cmake_property(_variableNames VARIABLES)
#  list (SORT _variableNames)
#  foreach (_variableName ${_variableNames})
#      message(STATUS "${_variableName}=${${_variableName}}")
#  endforeach()

endmacro()



###############################################################################
# PHASE 4: VALIDATE                                                           #
#                                                                             #
# Everything phases 1-3 produced must be self-consistent before anything is   #
# built. Each check below exists because its absence has produced, or can     #
# produce, a build that succeeds and is wrong - which is far more expensive   #
# than a configure that stops here with a specific message.                   #
#                                                                             #
# This runs before add_subdirectory(external), so it can only assert about    #
# variables. Assertions about targets belong in omega_check_dependencies(),   #
# which runs after the externals have been added.                             #
###############################################################################

# Reset check.
macro(omega_check_reset)
  set(_OMEGA_CHECK_ERROR_COUNT 0)
  set(_OMEGA_CHECK_REPORT "")
  set(_OMEGA_CHECK_WARNINGS "")
endmacro()

macro(omega_check_failed)
  set(_OmegaMsg "")
  # foreach(... ${ARGV}), not foreach(... IN LISTS ARGV): inside a macro ARGV is
  # substituted textually and is not a real variable, so the IN LISTS form finds
  # nothing and every message comes out empty.
  foreach(_OmegaPart ${ARGV})
    string(APPEND _OmegaMsg "${_OmegaPart}")
  endforeach()
  math(EXPR _OMEGA_CHECK_ERROR_COUNT "${_OMEGA_CHECK_ERROR_COUNT} + 1")
  string(APPEND _OMEGA_CHECK_REPORT "\n  * ${_OmegaMsg}")
endmacro()

# Emit the collected failures, if any, as a single actionable error.
macro(omega_check_report _OmegaWhat)
  if(_OMEGA_CHECK_ERROR_COUNT GREATER 0)
    message(FATAL_ERROR
      "${_OmegaWhat} failed ${_OMEGA_CHECK_ERROR_COUNT} "
      "check(s):${_OMEGA_CHECK_REPORT}\n")
  endif()
endmacro()

macro(omega_check_setup)

  omega_require_phase(4 omega_check_setup)

  omega_check_reset()

  # --- build mode -----------------------------------------------------------
  if("${OMEGA_BUILD_MODE}" STREQUAL "E3SM")
    message(STATUS "*** Omega E3SM-component Build ***")

  elseif("${OMEGA_BUILD_MODE}" STREQUAL "STANDALONE")
    message(STATUS "*** Omega Standalone Build ***")

  else()
    omega_check_failed(
      "OMEGA_BUILD_MODE is '${OMEGA_BUILD_MODE}', neither E3SM nor STANDALONE.")
  endif()

  # --- architecture ---------------------------------------------------------
  # omega_update_variables() rewrites an empty OMEGA_ARCH to SERIAL, so by now
  # it must be one of the known values.
  if(NOT "${OMEGA_ARCH}" IN_LIST OMEGA_ARCHS)
    string(REPLACE ";" ", " _OmegaArchList "${OMEGA_ARCHS}")
    omega_check_failed(
      "OMEGA_ARCH is '${OMEGA_ARCH}', expected one of: ${_OmegaArchList}.")
  endif()

  # OMEGA_TARGET_DEVICE drives -DOMEGA_TARGET_DEVICE and the device-aware MPI
  # path in src/base/Halo.h, so a disagreement here is a wrong build, not a
  # cosmetic one.
  set(_OmegaDeviceArchs "CUDA" "HIP" "SYCL")
  if("${OMEGA_ARCH}" IN_LIST _OmegaDeviceArchs)
    if(NOT OMEGA_TARGET_DEVICE)
      omega_check_failed(
        "OMEGA_ARCH=${OMEGA_ARCH} is a device architecture but "
        "OMEGA_TARGET_DEVICE is false.")
    endif()
  elseif(OMEGA_TARGET_DEVICE)
    omega_check_failed(
      "OMEGA_TARGET_DEVICE is true but OMEGA_ARCH=${OMEGA_ARCH} is a host "
      "architecture.")
  endif()

  # The Kokkos backend actually enabled must be the one OMEGA_ARCH asked for.
  # This only applies when Omega configures Kokkos itself; in a coupled build
  # that includes EAMxx, Kokkos::kokkos already exists with its own settings and
  # Omega deliberately leaves them alone (see omega_update_variables).
  if(NOT TARGET Kokkos::kokkos)
    set(_OmegaWantBackend "")
    if("${OMEGA_ARCH}" STREQUAL "CUDA")
      set(_OmegaWantBackend Kokkos_ENABLE_CUDA)
    elseif("${OMEGA_ARCH}" STREQUAL "HIP")
      set(_OmegaWantBackend Kokkos_ENABLE_HIP)
    elseif("${OMEGA_ARCH}" STREQUAL "SYCL")
      set(_OmegaWantBackend Kokkos_ENABLE_SYCL)
    elseif("${OMEGA_ARCH}" STREQUAL "OPENMP")
      set(_OmegaWantBackend Kokkos_ENABLE_OPENMP)
    elseif("${OMEGA_ARCH}" STREQUAL "THREADS")
      set(_OmegaWantBackend Kokkos_ENABLE_THREADS)
    endif()
    if(_OmegaWantBackend AND NOT ${_OmegaWantBackend})
      omega_check_failed(
        "OMEGA_ARCH=${OMEGA_ARCH} requires ${_OmegaWantBackend}=ON, but it is "
        "'${${_OmegaWantBackend}}'. A stale CMakeCache.txt from a build with a "
        "different OMEGA_ARCH is the usual cause; configure into a clean "
        "directory.")
    endif()
  endif()

  # HIP support in CMake needs 3.21; a standalone build asserts that with
  # cmake_minimum_required, but a coupled build inherits E3SM's 3.18 floor.
  if("${OMEGA_ARCH}" STREQUAL "HIP" AND CMAKE_VERSION VERSION_LESS 3.21)
    omega_check_failed(
      "OMEGA_ARCH=HIP requires CMake 3.21 or later; this is ${CMAKE_VERSION}.")
  endif()

  # --- build type -----------------------------------------------------------
  set(_OmegaBuildTypes "DEBUG" "RELEASE" "RELWITHDEBINFO" "MINSIZEREL")
  string(TOUPPER "${OMEGA_BUILD_TYPE}" _OmegaBuildType)
  if("${_OmegaBuildType}" STREQUAL "")
    omega_check_failed("OMEGA_BUILD_TYPE is empty.")
  elseif(NOT "${_OmegaBuildType}" IN_LIST _OmegaBuildTypes)
    string(REPLACE ";" ", " _OmegaBuildTypeList "${_OmegaBuildTypes}")
    omega_check_failed(
      "OMEGA_BUILD_TYPE is '${OMEGA_BUILD_TYPE}', expected one of: "
      "${_OmegaBuildTypeList}.")
  endif()

  # OMEGA_DEBUG gates -DOMEGA_DEBUG, which is what makes OMEGA_ASSERT do
  # anything (src/infra/Error.h). A debug build without it is the silent
  # failure this check exists for.
  if("${_OmegaBuildType}" STREQUAL "DEBUG" AND NOT OMEGA_DEBUG)
    omega_check_failed(
      "OMEGA_BUILD_TYPE=${OMEGA_BUILD_TYPE} but OMEGA_DEBUG is OFF, so "
      "OMEGA_ASSERT would compile to nothing in a debug build.")
  endif()

  # --- external library roots ----------------------------------------------
  # find_package(Parmetis) cannot fail on its own (FindParmetis.cmake has no
  # find_package_handle_standard_args), and src/CMakeLists.txt links 'parmetis'
  # and 'metis' by bare name, so a miss degrades to raw -lparmetis/-lmetis and
  # surfaces as an unrelated-looking compile or link error much later.
  if(NOT Parmetis_FOUND)
    omega_check_failed(
      "ParMETIS was not found. Set -DOMEGA_PARMETIS_ROOT to the installation "
      "prefix containing include/parmetis.h and lib/libparmetis.a.")
  endif()
  if(NOT Metis_FOUND)
    omega_check_failed(
      "METIS was not found. Set -DOMEGA_METIS_ROOT (it defaults to "
      "OMEGA_PARMETIS_ROOT='${OMEGA_PARMETIS_ROOT}', which does not contain "
      "include/metis.h and lib/libmetis.a).")
  endif()

  # --- numeric and enumerated options --------------------------------------
  if(NOT "${OMEGA_VECTOR_LENGTH}" MATCHES "^[1-9][0-9]*$")
    omega_check_failed(
      "OMEGA_VECTOR_LENGTH must be a positive integer, not "
      "'${OMEGA_VECTOR_LENGTH}'. It is substituted into a constexpr int "
      "(src/base/MachEnv.h).")
  endif()

  if(DEFINED OMEGA_TILE_LENGTH AND NOT "${OMEGA_TILE_LENGTH}" STREQUAL "")
    if(NOT "${OMEGA_TILE_LENGTH}" MATCHES "^[1-9][0-9]*$")
      omega_check_failed(
        "OMEGA_TILE_LENGTH must be a positive integer, not "
        "'${OMEGA_TILE_LENGTH}'.")
    endif()
  endif()

  if(OMEGA_MEMORY_LAYOUT)
    string(TOUPPER "${OMEGA_MEMORY_LAYOUT}" _OmegaLayout)
    if(NOT "${_OmegaLayout}" STREQUAL "LEFT" AND
       NOT "${_OmegaLayout}" STREQUAL "RIGHT")
      omega_check_failed(
        "OMEGA_MEMORY_LAYOUT is '${OMEGA_MEMORY_LAYOUT}', expected LEFT or "
        "RIGHT.")
    endif()
  endif()

  # --- test harness ---------------------------------------------------------
  # test/CMakeLists.txt builds every MPI test command from OMEGA_MPI_EXEC. It
  # is produced by omega_read_cime_config() on the standalone path only, so
  # with tests enabled in any other configuration every add_test() would
  # silently produce a command line starting with the test's own -n argument.
  if(OMEGA_BUILD_TEST AND "${OMEGA_MPI_EXEC}" STREQUAL "")
    omega_check_failed(
      "OMEGA_BUILD_TEST is ON but OMEGA_MPI_EXEC is empty, so the ctest "
      "commands would have no MPI launcher.")
  endif()

  # --- source tree ----------------------------------------------------------
  foreach(_OmegaDir E3SM_ROOT E3SM_CIME_ROOT E3SM_EXTERNALS_ROOT)
    if(NOT EXISTS "${${_OmegaDir}}")
      omega_check_failed("${_OmegaDir}='${${_OmegaDir}}' does not exist.")
    endif()
  endforeach()

  if(NOT EXISTS "${E3SM_EXTERNALS_ROOT}/ekat/extern/kokkos/CMakeLists.txt")
    omega_check_failed(
      "Kokkos sources are missing at "
      "${E3SM_EXTERNALS_ROOT}/ekat/extern/kokkos. Run 'git submodule update "
      "--init --recursive externals/ekat'.")
  endif()

  # --- report ---------------------------------------------------------------
  omega_check_report("Omega build setup")

  message(STATUS "Omega build setup validated")

endmacro()


###############################################################################
# PHASE 5: BUILD - dependency verification                                    #
#                                                                             #
# Runs immediately after add_subdirectory(external), which is the first point #
# at which the third-party targets exist. src/CMakeLists.txt links most of    #
# them by bare name, and CMake does not object to linking a name that is not a#
# target: it emits a raw -lname flag and the build fails much later with a    #
# linker error that names neither Omega nor the dependency that went missing. #
###############################################################################
macro(omega_check_dependencies)

  omega_require_phase(5 omega_check_dependencies)

  omega_check_reset()

  # Targets that src/CMakeLists.txt links into OmegaLibFlags. stdc++fs is
  # deliberately excluded: it is a compiler-provided library, never a target.
  set(_OmegaRequiredTargets
      Kokkos::kokkos
      spdlog
      yaml-cpp
      pioc
      gptl
      pacer
      gswteos-10
      cpptrace::cpptrace
      parmetis
      metis)

  foreach(_OmegaTarget ${_OmegaRequiredTargets})
    if(NOT TARGET ${_OmegaTarget})
      omega_check_failed(
        "'${_OmegaTarget}' is linked by src/CMakeLists.txt but is not a CMake "
        "target, so it would be passed to the linker as a bare -l flag.")
    endif()
  endforeach()

  if(GKlib_FOUND AND NOT TARGET gklib)
    omega_check_failed(
      "GKlib_FOUND is true but the 'gklib' target does not exist.")
  endif()

  omega_check_report("Omega dependencies")

  message(STATUS "Omega dependency targets validated")

endmacro()


###############################################################################
# PHASE 6: OUTPUT                                                             #
#                                                                             #
# The developer scripts and the config copies used to be written in phase 1,  #
# from inside omega_init_standalone_build(). That put them 150 lines ahead of #
# the code that decided the architecture and the compiler environment, so     #
# anything derived later could not reach them - which is how the              #
# MPICH_GPU_SUPPORT_ENABLED export, added in e7f17cfe0a, ended up reading     #
# variables from a phase that had not run yet and was dropped in 338a3abf29.  #
# Generating them here, last, means every value they embed is final.          #
###############################################################################
macro(omega_write_dev_scripts)

  omega_require_phase(6 omega_write_dev_scripts)

  # create a env script
  set(_EnvScript ${OMEGA_BUILD_DIR}/omega_env.sh)
  file(WRITE ${_EnvScript}  "#!/usr/bin/env bash\n\n")

  file(APPEND ${_EnvScript}
       "SCRIPT_DIR=$(cd $(dirname $BASH_SOURCE[0]) && pwd)\n\n")
  file(APPEND ${_EnvScript}
       "source $SCRIPT_DIR/e3smcase/.env_mach_specific.sh\n\n")
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

  # Omega compiles with -DOMEGA_MPI_ON_DEVICE, so src/base/Halo.h hands device
  # pointers straight to MPI. Cray MPICH only accepts those when
  # MPICH_GPU_SUPPORT_ENABLED is set, and the machine environment does not
  # always set it (frontier defaults it to 0 for every compiler that is not
  # *hipcc). Restored from e7f17cfe0a; it can live here now because
  # OMEGA_TARGET_DEVICE is a phase 3 product.
  if(OMEGA_MPI_ON_DEVICE AND OMEGA_TARGET_DEVICE AND
     "${MPILIB_NAME}" STREQUAL "mpich")
    file(APPEND ${_EnvScript} "export MPICH_GPU_SUPPORT_ENABLED=1\n\n")
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
  file(APPEND ${_RunScript}
       "cd test; ${OMEGA_MPI_EXEC} ${OMEGA_MPI_ARGS_STR} -n 8 -- ../src/omega.exe\n\n")

  # create a ctest script
  set(_CtestScript ${OMEGA_BUILD_DIR}/omega_ctest.sh)
  file(WRITE ${_CtestScript}  "#!/usr/bin/env bash\n\n")
  file(APPEND ${_CtestScript} "source ./omega_env.sh\n\n")
  # each test truncates its own log on startup, so this only removes logs left
  # behind by tests that are no longer run
  file(APPEND ${_CtestScript} "rm -f test/logs/*.log\n\n")
  if(OMEGA_DEBUG)
    file(APPEND ${_CtestScript}
         "ctest --output-on-failure --verbose $* # --rerun-failed\n\n")
  else()
    file(APPEND ${_CtestScript}
         "ctest --output-on-failure $* # --rerun-failed\n\n")
  endif()

  # create a profile script
  set(_ProfileScript ${OMEGA_BUILD_DIR}/omega_profile.sh)
  file(WRITE ${_ProfileScript}  "#!/usr/bin/env bash\n\n")
  file(APPEND ${_ProfileScript} "source ./omega_env.sh\n\n")
  file(APPEND ${_ProfileScript}
       "# modify 'OUTFILE' with a path in that the profiler can\n")
  file(APPEND ${_ProfileScript}
       "# create files such as a path in a scratch file system.\n")

  if("${OMEGA_ARCH}" STREQUAL "CUDA")
    file(APPEND ${_ProfileScript} "OUTFILE=${OMEGA_BUILD_DIR}/nsys_output\n\n")
    file(APPEND ${_ProfileScript} "# load Nsight Systems Profiler\n")
    file(APPEND ${_ProfileScript} "module load Nsight-Systems\n\n")
    file(APPEND ${_ProfileScript} "nsys profile -o \$OUTFILE \\\n")
    file(APPEND ${_ProfileScript}
         "    --cuda-memory-usage=true --force-overwrite=true \\\n")
    file(APPEND ${_ProfileScript} "    --trace=cuda,nvtx,osrt \\\n")
    file(APPEND ${_ProfileScript} "    ./src/omega.exe 1000")

  elseif("${OMEGA_ARCH}" STREQUAL "HIP")
    file(APPEND ${_ProfileScript}
         "OUTFILE=${OMEGA_BUILD_DIR}/rocprof_output.csv\n")
    file(APPEND ${_ProfileScript}
         "rocprof --hip-trace --hsa-trace --timestamp on \\\n")
    file(APPEND ${_ProfileScript} "    -o \$OUTFILE ./src/omega.exe 1000")

  endif()

  execute_process(COMMAND chmod +x ${_EnvScript})
  execute_process(COMMAND chmod +x ${_BuildScript})
  execute_process(COMMAND chmod +x ${_RunScript})
  execute_process(COMMAND chmod +x ${_CtestScript})
  execute_process(COMMAND chmod +x ${_ProfileScript})

  # copy yaml configuration files
  file(MAKE_DIRECTORY "${OMEGA_BUILD_DIR}/configs")
  file(COPY "${OMEGA_SOURCE_DIR}/configs/Default.yml"
       DESTINATION "${OMEGA_BUILD_DIR}/configs")
  file(COPY "${OMEGA_SOURCE_DIR}/configs/Default.yml"
       DESTINATION "${OMEGA_BUILD_DIR}/test")
  file(RENAME "${OMEGA_BUILD_DIR}/test/Default.yml"
       "${OMEGA_BUILD_DIR}/test/omega.yml")

endmacro()


###############################################################################
# PHASE 6: OUTPUT - install rules                                             #
###############################################################################
macro(omega_wrap_outputs)

  omega_require_phase(6 omega_wrap_outputs)

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
