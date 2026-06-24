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

# Collect coverage for CDash (decision 4: report to BOTH CDash and Codecov).
# CTEST_COVERAGE_COMMAND is set in CTestConfig.cmake; ctest_coverage() is a no-op
# when coverage was not enabled / no coverage tool is configured, so this is safe
# for ordinary (non-coverage) runs.
if(DEFINED CTEST_COVERAGE_COMMAND AND CTEST_COVERAGE_COMMAND)
  ctest_coverage(
    RETURN_VALUE CoverageRetval
    CAPTURE_CMAKE_ERROR CoverageResult
  )
endif()

ctest_submit(
  RETURN_VALUE SubmitRetval
  CAPTURE_CMAKE_ERROR SubmitResult
)
