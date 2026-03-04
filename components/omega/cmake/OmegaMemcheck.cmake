# OmegaMemcheck.cmake
# Memory check test configuration using valgrind4hpc and GPU sanitizers
#
# This module provides functions to:
# - Detect memory check tools (valgrind4hpc, compute-sanitizer)
# - Register memcheck versions of unit tests
# - Generate convenience scripts for running all memcheck tests

#------------------------------------------------------------------------------
# Options
#------------------------------------------------------------------------------
option(OMEGA_BUILD_MEMCHECK "Build memory check tests (requires valgrind4hpc or GPU sanitizer)" OFF)

#------------------------------------------------------------------------------
# Detect valgrind4hpc availability
# Priority: 1) module avail 2) find executable 3) user-provided path
#------------------------------------------------------------------------------
function(detect_valgrind4hpc)
  set(OMEGA_VALGRIND4HPC_FOUND FALSE PARENT_SCOPE)
  set(OMEGA_VALGRIND4HPC_COMMAND "" PARENT_SCOPE)

  # Check if user provided path
  if(DEFINED OMEGA_VALGRIND4HPC_EXECUTABLE AND EXISTS "${OMEGA_VALGRIND4HPC_EXECUTABLE}")
    message(STATUS "Using user-provided valgrind4hpc: ${OMEGA_VALGRIND4HPC_EXECUTABLE}")
    set(OMEGA_VALGRIND4HPC_FOUND TRUE PARENT_SCOPE)
    set(OMEGA_VALGRIND4HPC_COMMAND "${OMEGA_VALGRIND4HPC_EXECUTABLE}" PARENT_SCOPE)
    return()
  endif()

  # Try module avail first
  execute_process(
    COMMAND bash -c "module avail valgrind4hpc 2>&1 | grep -q valgrind4hpc"
    RESULT_VARIABLE _module_result
    OUTPUT_QUIET ERROR_QUIET
  )

  if(_module_result EQUAL 0)
    message(STATUS "Found valgrind4hpc module available")
    # The command will need to load the module first
    set(OMEGA_VALGRIND4HPC_FOUND TRUE PARENT_SCOPE)
    set(OMEGA_VALGRIND4HPC_COMMAND "valgrind4hpc" PARENT_SCOPE)
    set(OMEGA_VALGRIND4HPC_NEEDS_MODULE TRUE PARENT_SCOPE)
    return()
  endif()

  # Try to find executable directly
  find_program(_valgrind4hpc_exe NAMES valgrind4hpc)
  if(_valgrind4hpc_exe)
    message(STATUS "Found valgrind4hpc executable: ${_valgrind4hpc_exe}")
    set(OMEGA_VALGRIND4HPC_FOUND TRUE PARENT_SCOPE)
    set(OMEGA_VALGRIND4HPC_COMMAND "${_valgrind4hpc_exe}" PARENT_SCOPE)
    return()
  endif()

  # Not found
  message(STATUS "valgrind4hpc not found")
  message(STATUS "  - Module 'valgrind4hpc' not available")
  message(STATUS "  - Executable 'valgrind4hpc' not in PATH")
  message(STATUS "  - OMEGA_VALGRIND4HPC_EXECUTABLE not set")

endfunction()

#------------------------------------------------------------------------------
# Detect NVIDIA compute-sanitizer availability
#------------------------------------------------------------------------------
function(detect_compute_sanitizer)
  set(OMEGA_COMPUTE_SANITIZER_FOUND FALSE PARENT_SCOPE)
  set(OMEGA_COMPUTE_SANITIZER_COMMAND "" PARENT_SCOPE)

  # Check if user provided path
  if(DEFINED OMEGA_COMPUTE_SANITIZER_EXECUTABLE AND EXISTS "${OMEGA_COMPUTE_SANITIZER_EXECUTABLE}")
    message(STATUS "Using user-provided compute-sanitizer: ${OMEGA_COMPUTE_SANITIZER_EXECUTABLE}")
    set(OMEGA_COMPUTE_SANITIZER_FOUND TRUE PARENT_SCOPE)
    set(OMEGA_COMPUTE_SANITIZER_COMMAND "${OMEGA_COMPUTE_SANITIZER_EXECUTABLE}" PARENT_SCOPE)
    return()
  endif()

  # Try to find executable (usually in CUDA toolkit)
  find_program(_compute_sanitizer_exe NAMES compute-sanitizer
    HINTS
      $ENV{CUDA_HOME}/bin
      $ENV{CUDA_PATH}/bin
      /usr/local/cuda/bin
  )

  if(_compute_sanitizer_exe)
    message(STATUS "Found compute-sanitizer: ${_compute_sanitizer_exe}")
    set(OMEGA_COMPUTE_SANITIZER_FOUND TRUE PARENT_SCOPE)
    set(OMEGA_COMPUTE_SANITIZER_COMMAND "${_compute_sanitizer_exe}" PARENT_SCOPE)
    return()
  endif()

  message(STATUS "compute-sanitizer not found (CUDA memory checker)")

endfunction()

#------------------------------------------------------------------------------
# Detect AMD ROCm memory checker (rocm-smi or rocprof with memory tracking)
#------------------------------------------------------------------------------
function(detect_rocm_sanitizer)
  set(OMEGA_ROCM_SANITIZER_FOUND FALSE PARENT_SCOPE)
  set(OMEGA_ROCM_SANITIZER_COMMAND "" PARENT_SCOPE)

  # ROCm doesn't have a direct equivalent to compute-sanitizer
  # Options: rocgdb, rocprof with memory tracking, or AddressSanitizer
  # For now, check if rocgdb is available for basic memory debugging

  find_program(_rocgdb_exe NAMES rocgdb
    HINTS
      $ENV{ROCM_PATH}/bin
      /opt/rocm/bin
  )

  if(_rocgdb_exe)
    message(STATUS "Found rocgdb: ${_rocgdb_exe}")
    message(STATUS "  Note: ROCm memory checking is limited compared to CUDA compute-sanitizer")
    set(OMEGA_ROCM_SANITIZER_FOUND TRUE PARENT_SCOPE)
    set(OMEGA_ROCM_SANITIZER_COMMAND "${_rocgdb_exe}" PARENT_SCOPE)
    return()
  endif()

  message(STATUS "ROCm memory debugging tools not found")

endfunction()

#------------------------------------------------------------------------------
# Initialize memory check tools based on architecture
#------------------------------------------------------------------------------
function(init_memcheck_tools)
  if(NOT OMEGA_BUILD_MEMCHECK)
    return()
  endif()

  message(STATUS "")
  message(STATUS "=== Memory Check Configuration ===")

  # Always try to detect valgrind4hpc for CPU memory checking
  detect_valgrind4hpc()

  # Detect GPU-specific tools based on architecture
  if("${OMEGA_ARCH}" STREQUAL "CUDA")
    detect_compute_sanitizer()
  elseif("${OMEGA_ARCH}" STREQUAL "HIP")
    detect_rocm_sanitizer()
  endif()

  # Determine what's available
  set(_has_cpu_memcheck FALSE)
  set(_has_gpu_memcheck FALSE)

  if(OMEGA_VALGRIND4HPC_FOUND)
    set(_has_cpu_memcheck TRUE)
  endif()

  if("${OMEGA_ARCH}" STREQUAL "CUDA" AND OMEGA_COMPUTE_SANITIZER_FOUND)
    set(_has_gpu_memcheck TRUE)
  elseif("${OMEGA_ARCH}" STREQUAL "HIP" AND OMEGA_ROCM_SANITIZER_FOUND)
    set(_has_gpu_memcheck TRUE)
  endif()

  # Report findings
  if(NOT _has_cpu_memcheck AND NOT _has_gpu_memcheck)
    message(WARNING "OMEGA_BUILD_MEMCHECK is ON but no memory check tools found!")
    message(STATUS "  To enable memory check tests, either:")
    message(STATUS "    - Load the valgrind4hpc module")
    message(STATUS "    - Set OMEGA_VALGRIND4HPC_EXECUTABLE=/path/to/valgrind4hpc")
    if("${OMEGA_ARCH}" STREQUAL "CUDA")
      message(STATUS "    - Set OMEGA_COMPUTE_SANITIZER_EXECUTABLE=/path/to/compute-sanitizer")
    endif()
    set(OMEGA_BUILD_MEMCHECK OFF PARENT_SCOPE)
  else()
    message(STATUS "Memory check tools available:")
    if(_has_cpu_memcheck)
      message(STATUS "  - valgrind4hpc: ${OMEGA_VALGRIND4HPC_COMMAND}")
    endif()
    if(_has_gpu_memcheck)
      if("${OMEGA_ARCH}" STREQUAL "CUDA")
        message(STATUS "  - compute-sanitizer: ${OMEGA_COMPUTE_SANITIZER_COMMAND}")
      elseif("${OMEGA_ARCH}" STREQUAL "HIP")
        message(STATUS "  - rocgdb: ${OMEGA_ROCM_SANITIZER_COMMAND}")
      endif()
    endif()
  endif()

  message(STATUS "==================================")
  message(STATUS "")

  # Export variables to parent scope
  set(OMEGA_VALGRIND4HPC_FOUND ${OMEGA_VALGRIND4HPC_FOUND} PARENT_SCOPE)
  set(OMEGA_VALGRIND4HPC_COMMAND ${OMEGA_VALGRIND4HPC_COMMAND} PARENT_SCOPE)
  set(OMEGA_VALGRIND4HPC_NEEDS_MODULE ${OMEGA_VALGRIND4HPC_NEEDS_MODULE} PARENT_SCOPE)
  set(OMEGA_COMPUTE_SANITIZER_FOUND ${OMEGA_COMPUTE_SANITIZER_FOUND} PARENT_SCOPE)
  set(OMEGA_COMPUTE_SANITIZER_COMMAND ${OMEGA_COMPUTE_SANITIZER_COMMAND} PARENT_SCOPE)
  set(OMEGA_ROCM_SANITIZER_FOUND ${OMEGA_ROCM_SANITIZER_FOUND} PARENT_SCOPE)
  set(OMEGA_ROCM_SANITIZER_COMMAND ${OMEGA_ROCM_SANITIZER_COMMAND} PARENT_SCOPE)

endfunction()

#------------------------------------------------------------------------------
# Register a memcheck test for a given test executable
# Usage: omega_register_memcheck_test(TEST_NAME EXE_NAME MPI_TASKS)
#------------------------------------------------------------------------------
function(omega_register_memcheck_test test_name exe_name mpi_tasks)
  if(NOT OMEGA_BUILD_MEMCHECK)
    return()
  endif()

  set(_memcheck_test_name "MEMCHECK_${test_name}")

  # For GPU architectures, use GPU-specific sanitizer
  if("${OMEGA_ARCH}" STREQUAL "CUDA" AND OMEGA_COMPUTE_SANITIZER_FOUND)
    # NVIDIA compute-sanitizer
    # MPI launcher runs compute-sanitizer on each rank
    add_test(
      NAME ${_memcheck_test_name}
      COMMAND ${OMEGA_MPI_EXEC} ${OMEGA_MPI_ARGS} -n ${mpi_tasks} --
              ${OMEGA_COMPUTE_SANITIZER_COMMAND}
              --tool memcheck
              --leak-check full
              ./${exe_name}
    )
    set_tests_properties(${_memcheck_test_name} PROPERTIES
      LABELS "memcheck;${OMEGA_ARCH};Omega-memcheck"
      TIMEOUT 600
    )

  elseif("${OMEGA_ARCH}" STREQUAL "HIP" AND OMEGA_ROCM_SANITIZER_FOUND)
    # ROCm - limited support, skip for now with informative message
    message(STATUS "Skipping memcheck test ${_memcheck_test_name} - ROCm memory checking not fully supported")

  elseif(OMEGA_VALGRIND4HPC_FOUND)
    # CPU architectures - use valgrind4hpc
    if(OMEGA_VALGRIND4HPC_NEEDS_MODULE)
      # Need to load module first
      add_test(
        NAME ${_memcheck_test_name}
        COMMAND bash -c "module load valgrind4hpc && valgrind4hpc -n ${mpi_tasks} --valgrind-args='--leak-check=full --show-leak-kinds=all --track-origins=yes' ./${exe_name}"
      )
    else()
      add_test(
        NAME ${_memcheck_test_name}
        COMMAND ${OMEGA_VALGRIND4HPC_COMMAND}
                -n ${mpi_tasks}
                --valgrind-args="--leak-check=full --show-leak-kinds=all --track-origins=yes"
                ./${exe_name}
      )
    endif()
    set_tests_properties(${_memcheck_test_name} PROPERTIES
      LABELS "memcheck;${OMEGA_ARCH};Omega-memcheck"
      TIMEOUT 1200  # Valgrind is slow, allow 20 minutes
    )

  else()
    message(STATUS "Skipping memcheck test ${_memcheck_test_name} - no memory check tool available")
  endif()

endfunction()

#------------------------------------------------------------------------------
# Generate convenience script to run all memcheck tests
#------------------------------------------------------------------------------
function(generate_memcheck_script)
  if(NOT OMEGA_BUILD_MEMCHECK)
    return()
  endif()

  set(_script_content "#!/usr/bin/env bash
# Generated by OMEGA CMake - Run all memory check tests
# Usage: ./omega_memcheck.sh [ctest options]

set -e

source ./omega_env.sh

echo \"Running OMEGA memory check tests...\"
echo \"Architecture: ${OMEGA_ARCH}\"
echo \"\"

# Run all tests with 'memcheck' label
ctest -L memcheck --output-on-failure \"\$@\"

echo \"\"
echo \"Memory check tests complete.\"
")

  file(WRITE ${OMEGA_BUILD_DIR}/omega_memcheck.sh "${_script_content}")
  execute_process(COMMAND chmod +x ${OMEGA_BUILD_DIR}/omega_memcheck.sh)
  message(STATUS "Generated omega_memcheck.sh script")

endfunction()
