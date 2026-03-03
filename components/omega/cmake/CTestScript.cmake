# CTestScript.cmake
# CDash integration script for Omega
#
# Usage (from build directory):
#   ctest -S ../cmake/CTestScript.cmake
#
# Or with explicit paths:
#   ctest -S /path/to/cmake/CTestScript.cmake \
#         -DCTEST_SOURCE_DIRECTORY=/path/to/omega \
#         -DCTEST_BINARY_DIRECTORY=/path/to/build

cmake_minimum_required(VERSION 3.20)

# Set source and binary directories if not provided
if(NOT DEFINED CTEST_SOURCE_DIRECTORY)
  # Assume script is in <source>/cmake/ and we're running from build dir
  get_filename_component(CTEST_SOURCE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

if(NOT DEFINED CTEST_BINARY_DIRECTORY)
  # Assume current working directory is the build directory
  set(CTEST_BINARY_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  if("${CTEST_BINARY_DIRECTORY}" STREQUAL "")
    # Fallback: use current working directory
    execute_process(
      COMMAND pwd
      OUTPUT_VARIABLE CTEST_BINARY_DIRECTORY
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()
endif()

# Set build name if not provided
if(NOT DEFINED CTEST_BUILD_NAME)
  set(CTEST_BUILD_NAME "Omega-Experimental")
endif()

# Set site name if not provided
if(NOT DEFINED CTEST_SITE)
  cmake_host_system_information(RESULT CTEST_SITE QUERY HOSTNAME)
endif()

message(STATUS "CTest Configuration:")
message(STATUS "  Source Directory: ${CTEST_SOURCE_DIRECTORY}")
message(STATUS "  Binary Directory: ${CTEST_BINARY_DIRECTORY}")
message(STATUS "  Build Name: ${CTEST_BUILD_NAME}")
message(STATUS "  Site: ${CTEST_SITE}")

# Start the test run
ctest_start(Experimental)

# Build the project
ctest_build(
  RETURN_VALUE BuildRetval
  CAPTURE_CMAKE_ERROR BuildResult
)

if(NOT BuildRetval EQUAL 0)
  message(WARNING "Build failed with return value: ${BuildRetval}")
endif()

# Run the tests
ctest_test(
  RETURN_VALUE TestRetval
  CAPTURE_CMAKE_ERROR TestResult
)

if(NOT TestRetval EQUAL 0)
  message(WARNING "Tests failed with return value: ${TestRetval}")
endif()

# Submit results to CDash (if configured)
ctest_submit(
  RETURN_VALUE SubmitRetval
  CAPTURE_CMAKE_ERROR SubmitResult
)

if(NOT SubmitRetval EQUAL 0)
  message(WARNING "Submit failed with return value: ${SubmitRetval}")
endif()

# Summary
message(STATUS "")
message(STATUS "CTest Summary:")
message(STATUS "  Build Result: ${BuildRetval}")
message(STATUS "  Test Result: ${TestRetval}")
message(STATUS "  Submit Result: ${SubmitRetval}")
