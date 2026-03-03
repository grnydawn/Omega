# OmegaScripts.cmake
# Script generation module using templates
# Uses configure_file() instead of file(WRITE/APPEND)

set(OMEGA_CMAKE_TEMPLATES_DIR "${CMAKE_CURRENT_LIST_DIR}/templates")

#------------------------------------------------------------------------------
# Generate all helper scripts from templates
#------------------------------------------------------------------------------
function(generate_omega_scripts)

  # Prepare OpenMP settings for omega_env.sh
  if("${OMEGA_ARCH}" STREQUAL "OPENMP")
    set(OMEGA_ENV_OPENMP_SETTINGS "export OMP_NUM_THREADS=${THREAD_COUNT}

")
    if(DEFINED ENV{OMP_PROC_BIND})
      string(APPEND OMEGA_ENV_OPENMP_SETTINGS "export OMP_PROC_BIND=$ENV{OMP_PROC_BIND}

")
    else()
      string(APPEND OMEGA_ENV_OPENMP_SETTINGS "export OMP_PROC_BIND=spread

")
    endif()
    if(DEFINED ENV{OMP_PLACES})
      string(APPEND OMEGA_ENV_OPENMP_SETTINGS "export OMP_PLACES=$ENV{OMP_PLACES}

")
    else()
      string(APPEND OMEGA_ENV_OPENMP_SETTINGS "export OMP_PLACES=threads

")
    endif()
  else()
    set(OMEGA_ENV_OPENMP_SETTINGS "")
  endif()

  # Prepare MPI args string for omega_run.sh
  list(JOIN OMEGA_MPI_ARGS " " OMEGA_MPI_ARGS_STR)

  # Prepare ctest verbose flag
  if(OMEGA_DEBUG)
    set(OMEGA_CTEST_VERBOSE "--verbose")
  else()
    set(OMEGA_CTEST_VERBOSE "")
  endif()

  # Prepare profiler commands for omega_profile.sh
  if("${OMEGA_ARCH}" STREQUAL "CUDA")
    set(OMEGA_PROFILE_COMMANDS "OUTFILE=${OMEGA_BUILD_DIR}/nsys_output

# load Nsight Systems Profiler
module load Nsight-Systems

nsys profile -o \$OUTFILE \\
    --cuda-memory-usage=true --force-overwrite=true \\
    --trace=cuda,nvtx,osrt \\
    ./src/omega.exe 1000")

  elseif("${OMEGA_ARCH}" STREQUAL "HIP")
    set(OMEGA_PROFILE_COMMANDS "OUTFILE=${OMEGA_BUILD_DIR}/rocprof_output.csv
rocprof --hip-trace --hsa-trace --timestamp on \\
    -o \$OUTFILE ./src/omega.exe 1000")

  else()
    set(OMEGA_PROFILE_COMMANDS "# No profiler configured for ${OMEGA_ARCH}")
  endif()

  # Generate scripts using templates
  configure_file(
    ${OMEGA_CMAKE_TEMPLATES_DIR}/omega_env.sh.in
    ${OMEGA_BUILD_DIR}/omega_env.sh
    @ONLY
  )

  configure_file(
    ${OMEGA_CMAKE_TEMPLATES_DIR}/omega_build.sh.in
    ${OMEGA_BUILD_DIR}/omega_build.sh
    @ONLY
  )

  configure_file(
    ${OMEGA_CMAKE_TEMPLATES_DIR}/omega_run.sh.in
    ${OMEGA_BUILD_DIR}/omega_run.sh
    @ONLY
  )

  configure_file(
    ${OMEGA_CMAKE_TEMPLATES_DIR}/omega_ctest.sh.in
    ${OMEGA_BUILD_DIR}/omega_ctest.sh
    @ONLY
  )

  configure_file(
    ${OMEGA_CMAKE_TEMPLATES_DIR}/omega_profile.sh.in
    ${OMEGA_BUILD_DIR}/omega_profile.sh
    @ONLY
  )

  # Make scripts executable
  execute_process(COMMAND chmod +x ${OMEGA_BUILD_DIR}/omega_env.sh)
  execute_process(COMMAND chmod +x ${OMEGA_BUILD_DIR}/omega_build.sh)
  execute_process(COMMAND chmod +x ${OMEGA_BUILD_DIR}/omega_run.sh)
  execute_process(COMMAND chmod +x ${OMEGA_BUILD_DIR}/omega_ctest.sh)
  execute_process(COMMAND chmod +x ${OMEGA_BUILD_DIR}/omega_profile.sh)

endfunction()

#------------------------------------------------------------------------------
# Copy configuration files to build directory
#------------------------------------------------------------------------------
function(copy_omega_config_files)

  file(MAKE_DIRECTORY "${OMEGA_BUILD_DIR}/configs")
  file(COPY "${OMEGA_SOURCE_DIR}/configs/Default.yml"
       DESTINATION "${OMEGA_BUILD_DIR}/configs")
  file(COPY "${OMEGA_SOURCE_DIR}/configs/Default.yml"
       DESTINATION "${OMEGA_BUILD_DIR}/test")
  file(RENAME "${OMEGA_BUILD_DIR}/test/Default.yml"
       "${OMEGA_BUILD_DIR}/test/omega.yml")

endfunction()
