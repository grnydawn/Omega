# OmegaE3SMCase.cmake
# E3SM Case-Based Configuration Module
# Creates a minimal E3SM case to extract build settings

#------------------------------------------------------------------------------
# Execute a bash command and capture output
#------------------------------------------------------------------------------
macro(run_bash_command command outvar)
  execute_process(
    COMMAND bash -c "${command}"
    OUTPUT_VARIABLE ${outvar}
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endmacro()

#------------------------------------------------------------------------------
# Query CIME XML configuration
#------------------------------------------------------------------------------
macro(cime_xmlquery query outvar)
  run_bash_command("cd ${CASEROOT} && ./xmlquery ${query} --value" ${outvar})
endmacro()

#------------------------------------------------------------------------------
# Create E3SM case and extract configuration
# This reads CIME configuration and sets up environment variables
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# Detect compilers from E3SM case configuration
#------------------------------------------------------------------------------
macro(detect_compilers_from_e3sm)

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

endmacro()

#------------------------------------------------------------------------------
# Detect architecture from E3SM case or compiler
#------------------------------------------------------------------------------
macro(detect_omega_arch)

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

endmacro()

#------------------------------------------------------------------------------
# Configure CXX compiler based on architecture
#------------------------------------------------------------------------------
macro(configure_cxx_compiler_for_arch)

  # set C and Fortran compilers *before* calling CMake project()
  set(CMAKE_C_COMPILER ${OMEGA_C_COMPILER})
  set(CMAKE_Fortran_COMPILER ${OMEGA_Fortran_COMPILER})

  if(OMEGA_CXX_FLAGS)
    list(APPEND OMEGA_COMPILE_OPTIONS ${OMEGA_CXX_FLAGS})
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
      list(APPEND OMEGA_COMPILE_OPTIONS ${OMEGA_CUDA_FLAGS})
    endif()

    # Check if --ccbin is already set
    string(FIND "${OMEGA_CXX_FLAGS}" "--ccbin" pos)
    if(${pos} EQUAL -1)
      list(APPEND OMEGA_COMPILE_OPTIONS "-ccbin" "${CMAKE_CUDA_HOST_COMPILER}")
    endif()

    list(APPEND OMEGA_COMPILE_OPTIONS "-Wno-deprecated-gpu-targets")

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
      list(APPEND OMEGA_COMPILE_OPTIONS ${OMEGA_HIP_FLAGS})
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

  elseif("${OMEGA_ARCH}" STREQUAL "SYCL")
    set(CMAKE_CXX_COMPILER ${OMEGA_CXX_COMPILER})

    # add flags from upstream-E3SM
    if(SYCL_FLAGS)
      list(APPEND OMEGA_COMPILE_OPTIONS ${SYCL_FLAGS})
    endif()
    if(OMEGA_SYCL_FLAGS)
      list(APPEND OMEGA_COMPILE_OPTIONS ${OMEGA_SYCL_FLAGS})
    endif()
    if(OMEGA_SYCL_EXE_LINKER_FLAGS)
      list(APPEND OMEGA_LINK_OPTIONS ${OMEGA_SYCL_EXE_LINKER_FLAGS})
    endif()

  else()
    set(CMAKE_CXX_COMPILER ${OMEGA_CXX_COMPILER})

  endif()

  # Apply compile options using modern CMake
  if(OMEGA_COMPILE_OPTIONS)
    add_compile_options(${OMEGA_COMPILE_OPTIONS})
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
  message(STATUS "OMEGA_COMPILE_OPTIONS  = ${OMEGA_COMPILE_OPTIONS}")
  message(STATUS "CMAKE_EXE_LINKER_FLAGS = ${CMAKE_EXE_LINKER_FLAGS}")

endmacro()
