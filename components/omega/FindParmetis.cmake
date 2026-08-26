# Locate the ParMETIS, METIS and (optionally) GKlib installations Omega links
# against.
#
# Input variables:
#   OMEGA_PARMETIS_ROOT  install prefix for ParMETIS (required)
#   OMEGA_METIS_ROOT     install prefix for METIS, defaults to the ParMETIS one
#   OMEGA_GKLIB_ROOT     install prefix for GKlib, defaults to the ParMETIS one
#
# Output variables:
#   Parmetis_FOUND / Metis_FOUND / GKlib_FOUND
#   Parmetis_LIBRARY / Metis_LIBRARY / GKlib_LIBRARY
#   Parmetis_INCLUDE_DIRS  all of the include directories that were found

find_path(Parmetis_INCLUDE_DIR
          parmetis.h
          PATHS ${OMEGA_PARMETIS_ROOT}
          PATH_SUFFIXES include
          NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH)

# If not defined, assume the METIS path is the same as ParMETIS
if(NOT DEFINED OMEGA_METIS_ROOT)
  set(OMEGA_METIS_ROOT ${OMEGA_PARMETIS_ROOT})
endif()

find_path(Metis_INCLUDE_DIR
          metis.h
          PATHS ${OMEGA_METIS_ROOT}
          PATH_SUFFIXES include
          NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH)

# Assume the GKlib path is the same as METIS if it is not defined.
# This library is not mandatory and is therefore optional.
if(NOT DEFINED OMEGA_GKLIB_ROOT)
  set(OMEGA_GKLIB_ROOT ${OMEGA_PARMETIS_ROOT})
endif()

find_path(GKlib_INCLUDE_DIR
          GKlib.h
          PATHS ${OMEGA_GKLIB_ROOT}
          PATH_SUFFIXES include
          NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH)

# Currently using static libraries, but retain the following for
# potential future use with shared libraries.
#
# if(${OMEGA_PREFER_SHARED})
#   find_library(Parmetis_LIBRARY
#                NAMES parmetis
#                HINTS ${Parmetis_INCLUDE_DIR}/../lib
#                NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH)
#
#   find_library(Metis_LIBRARY
#                NAMES metis
#                HINTS ${Metis_INCLUDE_DIR}/../lib
#                NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH)
#
#   if(${GKlib_INCLUDE_DIR})
#     find_library(GKlib_LIBRARY
#                  NAMES GKlib
#                  HINTS ${GKlib_INCLUDE_DIR}/../lib
#                  NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH)
#   endif()
# else()

find_library(Parmetis_LIBRARY
             NAMES libparmetis.a
             HINTS ${Parmetis_INCLUDE_DIR}/../lib
             NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH)

find_library(Metis_LIBRARY
             NAMES libmetis.a
             HINTS ${Metis_INCLUDE_DIR}/../lib
             NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH)

# In some installations, GKlib is optional
if(DEFINED GKlib_INCLUDE_DIR AND NOT GKlib_INCLUDE_DIR STREQUAL "")
  find_library(GKlib_LIBRARY
               NAMES libGKlib.a
               HINTS ${GKlib_INCLUDE_DIR}/../lib
               NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH)
endif()

# endif()

set(Parmetis_INCLUDE_DIRS)

if(Parmetis_INCLUDE_DIR AND Parmetis_LIBRARY)

  set(Parmetis_FOUND TRUE)
  list(APPEND Parmetis_INCLUDE_DIRS ${Parmetis_INCLUDE_DIR})

  message(STATUS "Found Parmetis Library: ${Parmetis_LIBRARY}")
  message(STATUS "Found Parmetis Include: ${Parmetis_INCLUDE_DIR}")

else()
  set(Parmetis_FOUND FALSE)
endif()

if(Metis_INCLUDE_DIR AND Metis_LIBRARY)

  list(APPEND Parmetis_INCLUDE_DIRS ${Metis_INCLUDE_DIR})
  set(Metis_FOUND TRUE)

  message(STATUS "Found Metis Library: ${Metis_LIBRARY}")
  message(STATUS "Found Metis Include: ${Metis_INCLUDE_DIR}")

else()
  set(Metis_FOUND FALSE)
endif()

if(GKlib_INCLUDE_DIR AND GKlib_LIBRARY)

  list(APPEND Parmetis_INCLUDE_DIRS ${GKlib_INCLUDE_DIR})
  set(GKlib_FOUND TRUE)

  message(STATUS "Found GKlib Library: ${GKlib_LIBRARY}")
  message(STATUS "Found GKlib Include: ${GKlib_INCLUDE_DIR}")

else()
  set(GKlib_FOUND FALSE)
endif()

# Report through find_package_handle_standard_args so that Parmetis_FOUND is
# set by the mechanism the rest of CMake expects, and so that a caller passing
# REQUIRED actually gets an error.
# Metis is in REQUIRED_VARS because Omega links it unconditionally: a ParMETIS
# install without METIS is not usable here.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Parmetis
  REQUIRED_VARS
    Parmetis_LIBRARY Parmetis_INCLUDE_DIR
    Metis_LIBRARY Metis_INCLUDE_DIR
  # One argument: find_package_handle_standard_args treats adjacent quoted
  # strings as separate keywords, so this cannot be wrapped.
  FAIL_MESSAGE
    "Did not find the required libraries ParMETIS and METIS. Set their install prefixes with -DOMEGA_PARMETIS_ROOT and, if they differ, -DOMEGA_METIS_ROOT"
)
