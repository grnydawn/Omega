# BUILD_FLAGS_TEST: checks where Omega's -O1 lands on the compile line.
#
#   cmake -DFLAGS_OMEGA=<OmegaLib's flags.make>
#         -DFLAGS_EXTERNAL=<an external library's flags.make>
#         -DOMEGA_ARCH=<arch> -DBUILD_CONFIG=<build type>
#         -P CheckBuildFlags.cmake
#
# The expectation is this script's own, so that a change which moves or drops
# the -O1 in src/CMakeLists.txt fails here: -O1 is expected on Omega's C++ if
# and only if OMEGA_ARCH is SYCL and the build type is Debug, in any case
# (Omega issue #666). Where it is expected, it must be the last -O on the
# line, the level the compiler uses; where it is not, it must be absent. An
# external library's C++ never has it. Each target's flags are the CXX_FLAGS
# line of the flags.make the Makefile generator writes for it.

cmake_minimum_required(VERSION 3.21)

foreach(_var FLAGS_OMEGA FLAGS_EXTERNAL OMEGA_ARCH BUILD_CONFIG)
  if(NOT DEFINED ${_var})
    message(FATAL_ERROR "CheckBuildFlags.cmake needs -D${_var}=<value>")
  endif()
endforeach()

string(TOUPPER "${BUILD_CONFIG}" _config)
if("${OMEGA_ARCH}" STREQUAL "SYCL" AND "${_config}" STREQUAL "DEBUG")
  set(_expected TRUE)
  message(STATUS "OMEGA_ARCH=${OMEGA_ARCH}, build type ${BUILD_CONFIG}: -O1 expected on Omega's C++, as its last -O")
else()
  set(_expected FALSE)
  message(STATUS "OMEGA_ARCH=${OMEGA_ARCH}, build type ${BUILD_CONFIG}: no -O1 expected on Omega's C++")
endif()

set(_errors "")

# check_flags(<label> <flags.make> <-O1 expected>): print the file's CXX_FLAGS,
# and append what is wrong with its -O1 to _errors
function(check_flags label file expected)
  if(NOT EXISTS "${file}")
    set(_errors "${_errors}\n  ${file}: not found" PARENT_SCOPE)
    return()
  endif()
  file(STRINGS "${file}" lines REGEX "^CXX_FLAGS =")
  list(LENGTH lines n)
  if(NOT n EQUAL 1)
    set(_errors "${_errors}\n  ${file}: ${n} CXX_FLAGS lines, not 1" PARENT_SCOPE)
    return()
  endif()
  string(REGEX REPLACE "^CXX_FLAGS =" "" flags "${lines}")
  string(STRIP "${flags}" flags)
  message(STATUS "${label} (${file}): CXX_FLAGS = ${flags}")

  separate_arguments(args UNIX_COMMAND "${flags}")
  set(last "")
  set(has_O1 FALSE)
  foreach(arg IN LISTS args)
    if(arg MATCHES "^-O")
      set(last "${arg}")
      if(arg STREQUAL "-O1")
        set(has_O1 TRUE)
      endif()
    endif()
  endforeach()

  set(err "")
  if(expected)
    if(NOT has_O1)
      set(err "-O1 is expected and missing")
    elseif(NOT last STREQUAL "-O1")
      set(err "-O1 is not the last -O: ${last} follows it")
    endif()
  elseif(has_O1)
    set(err "-O1 is present and not expected")
  endif()
  if(NOT err STREQUAL "")
    set(_errors "${_errors}\n  ${file}: ${err}" PARENT_SCOPE)
  endif()
endfunction()

check_flags("Omega C++" "${FLAGS_OMEGA}" ${_expected})
check_flags("external C++" "${FLAGS_EXTERNAL}" FALSE)

if(NOT _errors STREQUAL "")
  message(FATAL_ERROR "BUILD_FLAGS_TEST failed:${_errors}")
endif()
message(STATUS "BUILD_FLAGS_TEST passed")
