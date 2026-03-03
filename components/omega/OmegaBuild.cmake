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
# Include Modules         #
###########################

include(${OMEGA_SOURCE_DIR}/cmake/OmegaE3SMCase.cmake)
include(${OMEGA_SOURCE_DIR}/cmake/OmegaScripts.cmake)

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

  # Initialize compile options list (replaces CMAKE_CXX_FLAGS manipulation)
  set(OMEGA_COMPILE_OPTIONS "")

  # Initialize compile definitions list (replaces add_definitions)
  set(OMEGA_COMPILE_DEFINITIONS "")

endmacro()

# Collect machine and compiler info from CIME
# and detect OMEGA_ARCH and compilers
macro(init_standalone_build)

  # get cime configuration (from OmegaE3SMCase.cmake)
  read_cime_config()

  # detect compilers from E3SM case (from OmegaE3SMCase.cmake)
  detect_compilers_from_e3sm()

  # detect architecture (from OmegaE3SMCase.cmake)
  detect_omega_arch()

  # configure CXX compiler for architecture (from OmegaE3SMCase.cmake)
  configure_cxx_compiler_for_arch()

  # generate helper scripts (from OmegaScripts.cmake)
  generate_omega_scripts()

  # copy configuration files (from OmegaScripts.cmake)
  copy_omega_config_files()

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
macro(update_variables)

  # Set the build type
  set(CMAKE_BUILD_TYPE ${OMEGA_BUILD_TYPE})

  # Collect compile definitions in a list (to be applied to targets later)
  list(APPEND OMEGA_COMPILE_DEFINITIONS "OMEGA_BUILD_MODE=${OMEGA_BUILD_MODE}")

  if(NOT DEFINED OMEGA_LOG_LEVEL)
    set(OMEGA_LOG_LEVEL "INFO")
  endif()

  if(OMEGA_DEBUG)
    set(OMEGA_LOG_FLUSH ON)
    list(APPEND OMEGA_COMPILE_DEFINITIONS "OMEGA_DEBUG" "OMEGA_LOG_LEVEL=1")
  else()
    string(TOUPPER "${OMEGA_LOG_LEVEL}" _LOG_LEVEL)
    if ("${_LOG_LEVEL}" STREQUAL "TRACE")
      list(APPEND OMEGA_COMPILE_DEFINITIONS "OMEGA_LOG_LEVEL=0")
    elseif("${_LOG_LEVEL}" STREQUAL "DEBUG")
      list(APPEND OMEGA_COMPILE_DEFINITIONS "OMEGA_LOG_LEVEL=1")
    elseif("${_LOG_LEVEL}" STREQUAL "INFO")
      list(APPEND OMEGA_COMPILE_DEFINITIONS "OMEGA_LOG_LEVEL=2")
    elseif("${_LOG_LEVEL}" STREQUAL "WARN")
      list(APPEND OMEGA_COMPILE_DEFINITIONS "OMEGA_LOG_LEVEL=3")
    elseif("${_LOG_LEVEL}" STREQUAL "ERROR")
      list(APPEND OMEGA_COMPILE_DEFINITIONS "OMEGA_LOG_LEVEL=4")
    elseif("${_LOG_LEVEL}" STREQUAL "CRITICAL")
      list(APPEND OMEGA_COMPILE_DEFINITIONS "OMEGA_LOG_LEVEL=5")
    elseif("${_LOG_LEVEL}" STREQUAL "OFF")
      list(APPEND OMEGA_COMPILE_DEFINITIONS "OMEGA_LOG_LEVEL=6")
    else()
      message(FATAL_ERROR "Unknown log level: '${OMEGA_LOG_LEVEL}'" )
    endif()
  endif()

  if(OMEGA_LOG_FLUSH)
    list(APPEND OMEGA_COMPILE_DEFINITIONS "OMEGA_LOG_FLUSH")
  endif()

  if(OMEGA_LOG_TASKS)
    string(TOUPPER "${OMEGA_LOG_TASKS}" _LOG_TASKS)
    list(APPEND OMEGA_COMPILE_DEFINITIONS "OMEGA_LOG_TASKS=${_LOG_TASKS}")
  endif()

  if(OMEGA_MEMORY_LAYOUT)
    string(TOUPPER "${OMEGA_MEMORY_LAYOUT}" _LAYOUT)
    list(APPEND OMEGA_COMPILE_DEFINITIONS "OMEGA_LAYOUT_${_LAYOUT}")
  else()
    list(APPEND OMEGA_COMPILE_DEFINITIONS "OMEGA_LAYOUT_RIGHT")
  endif()

  if(OMEGA_TILE_LENGTH)
    list(APPEND OMEGA_COMPILE_DEFINITIONS "OMEGA_TILE_LENGTH=${OMEGA_TILE_LENGTH}")
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
        list(APPEND OMEGA_COMPILE_OPTIONS "-I${MPI_CXX_INCLUDE_DIRS}")

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

  list(APPEND OMEGA_COMPILE_DEFINITIONS "OMEGA_ENABLE_${OMEGA_ARCH}")

  if(OMEGA_TARGET_DEVICE)
    list(APPEND OMEGA_COMPILE_DEFINITIONS "OMEGA_TARGET_DEVICE")
  endif()

  if(OMEGA_MPI_ON_DEVICE)
    list(APPEND OMEGA_COMPILE_DEFINITIONS "OMEGA_MPI_ON_DEVICE")
  endif()

  # Include the findParmetis script
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
  find_package(Parmetis REQUIRED)

endmacro()



################################
# Verify variable integrity    #
################################
function(check_setup)

  # Verify build mode
  if("${OMEGA_BUILD_MODE}" STREQUAL "E3SM")
    message(STATUS "*** Omega E3SM-component Build ***")

  elseif("${OMEGA_BUILD_MODE}" STREQUAL "STANDALONE")
    message(STATUS "*** Omega Standalone Build ***")

  else()
    message(FATAL_ERROR "OMEGA_BUILD_MODE is neither E3SM nor STANDALONE.")

  endif()

  # Verify architecture is set
  if("${OMEGA_ARCH}" STREQUAL "")
    message(FATAL_ERROR "OMEGA_ARCH is not set. Valid values: CUDA, HIP, SYCL, OPENMP, SERIAL")
  endif()

  # Verify architecture is valid
  set(_VALID_ARCHS "CUDA" "HIP" "SYCL" "OPENMP" "THREADS" "SERIAL")
  list(FIND _VALID_ARCHS "${OMEGA_ARCH}" _ARCH_INDEX)
  if(_ARCH_INDEX EQUAL -1)
    message(FATAL_ERROR "Invalid OMEGA_ARCH: ${OMEGA_ARCH}. Valid values: ${_VALID_ARCHS}")
  endif()

  # Verify compilers are set
  if(NOT DEFINED CMAKE_CXX_COMPILER OR "${CMAKE_CXX_COMPILER}" STREQUAL "")
    message(FATAL_ERROR "CMAKE_CXX_COMPILER is not set.")
  endif()

  # Standalone-specific checks
  if("${OMEGA_BUILD_MODE}" STREQUAL "STANDALONE")

    # Verify E3SM root exists
    if(NOT EXISTS "${E3SM_ROOT}")
      message(FATAL_ERROR "E3SM_ROOT does not exist: ${E3SM_ROOT}")
    endif()

    # Verify E3SM case was created
    if(NOT EXISTS "${CASEROOT}")
      message(FATAL_ERROR "E3SM case directory does not exist: ${CASEROOT}")
    endif()

    # Verify MPI exec is set
    if(NOT DEFINED OMEGA_MPI_EXEC OR "${OMEGA_MPI_EXEC}" STREQUAL "")
      message(FATAL_ERROR "OMEGA_MPI_EXEC is not set. E3SM case may not have been configured correctly.")
    endif()

  endif()

  # Verify build type is valid
  if(NOT "${OMEGA_BUILD_TYPE}" STREQUAL "Release" AND NOT "${OMEGA_BUILD_TYPE}" STREQUAL "Debug")
    message(WARNING "OMEGA_BUILD_TYPE '${OMEGA_BUILD_TYPE}' is non-standard. Expected: Release or Debug")
  endif()

  # Print configuration summary
  message(STATUS "")
  message(STATUS "=== Omega Configuration Summary ===")
  message(STATUS "  Build Mode:        ${OMEGA_BUILD_MODE}")
  message(STATUS "  Build Type:        ${OMEGA_BUILD_TYPE}")
  message(STATUS "  Architecture:      ${OMEGA_ARCH}")
  message(STATUS "  Target Device:     ${OMEGA_TARGET_DEVICE}")
  message(STATUS "  CXX Compiler:      ${CMAKE_CXX_COMPILER}")
  message(STATUS "  Debug Mode:        ${OMEGA_DEBUG}")
  message(STATUS "  Build Tests:       ${OMEGA_BUILD_TEST}")
  message(STATUS "  Build Executable:  ${OMEGA_BUILD_EXECUTABLE}")
  if("${OMEGA_BUILD_MODE}" STREQUAL "STANDALONE")
    message(STATUS "  MPI Exec:          ${OMEGA_MPI_EXEC}")
    message(STATUS "  E3SM Case:         ${CASEROOT}")
  endif()
  message(STATUS "===================================")
  message(STATUS "")

endfunction()


################################
# Prepare output               #
################################
function(wrap_outputs)

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

endfunction()
