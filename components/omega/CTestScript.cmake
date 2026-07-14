cmake_minimum_required(VERSION 3.20)

ctest_start(Experimental)

ctest_build(
  RETURN_VALUE BuildRetval
  CAPTURE_CMAKE_ERROR BuildResult
)

ctest_test(
  RETURN_VALUE TestRetval
  CAPTURE_CMAKE_ERROR TestResult
)

# Memcheck (OMEGA_MEMCHECK=ON): the leak tool is embedded per-rank in each test
# command (see add_omega_test) and leaks already fail ctest_test() above via
# valgrind --error-exitcode=1. CTEST_MEMORYCHECK_COMMAND is intentionally left
# unset so ctest_memcheck() stays inert and never wraps the MPI launcher. This
# guard only fires if a user opts into the CTest-native (non-MPI) path.
if(CTEST_MEMORYCHECK_COMMAND)
  ctest_memcheck(
    RETURN_VALUE MemcheckRetval
    CAPTURE_CMAKE_ERROR MemcheckResult
  )
endif()

ctest_submit(
  RETURN_VALUE SubmitRetval
  CAPTURE_CMAKE_ERROR SubmitResult
)
