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

# Coverage (OMEGA_COVERAGE=ON): setup_coverage() writes COVERAGE_COMMAND into
# DartConfiguration.tcl, but ctest_start() does NOT load it back into this -S
# script's CTEST_COVERAGE_COMMAND. Set it explicitly from the generated gcov
# wrapper, which exists only in a coverage-instrumented build. When the -S
# invocation does not define CTEST_BINARY_DIRECTORY, or in a non-coverage build,
# this stays unset and ctest_coverage() is skipped (the gcovr 'coverage' target
# / omega_coverage.sh remains the primary, always-available report path).
if(DEFINED CTEST_BINARY_DIRECTORY AND EXISTS "${CTEST_BINARY_DIRECTORY}/gcov")
  set(CTEST_COVERAGE_COMMAND "${CTEST_BINARY_DIRECTORY}/gcov")
endif()

if(CTEST_COVERAGE_COMMAND)
  ctest_coverage(
    RETURN_VALUE CoverageRetval
    CAPTURE_CMAKE_ERROR CoverageResult
  )
endif()

# Memcheck (OMEGA_MEMCHECK=ON): the leak tool is embedded per-rank in each test
# command (see add_omega_test) and leaks already fail ctest_test() above via
# valgrind --error-exitcode=1. Note: include(CTest) auto-discovers valgrind into
# DartConfiguration.tcl's MemoryCheckCommand, but ctest_start() does NOT mirror
# it into this script's CTEST_MEMORYCHECK_COMMAND, and we deliberately never set
# that variable here -- so ctest_memcheck() stays inert and never wraps the MPI
# launcher (srun/mpirun). Do NOT set CTEST_MEMORYCHECK_COMMAND from the
# DartConfiguration value: that would valgrind-wrap the launcher, not each rank.
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
