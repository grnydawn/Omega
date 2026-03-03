# FindParmetis.cmake
# Modern CMake find module for ParMETIS, METIS, and GKlib
#
# Creates the following imported targets:
#   Parmetis::parmetis - ParMETIS library
#   Metis::metis       - METIS library
#   GKlib::gklib       - GKlib library (optional)
#
# Sets the following variables:
#   Parmetis_FOUND       - TRUE if ParMETIS was found
#   Metis_FOUND          - TRUE if METIS was found
#   GKlib_FOUND          - TRUE if GKlib was found
#   Parmetis_INCLUDE_DIRS - Include directories (for backward compatibility)
#
# User-configurable variables:
#   OMEGA_PARMETIS_ROOT  - Root directory of ParMETIS installation
#   OMEGA_METIS_ROOT     - Root directory of METIS installation (defaults to PARMETIS_ROOT)
#   OMEGA_GKLIB_ROOT     - Root directory of GKlib installation (defaults to PARMETIS_ROOT)

#------------------------------------------------------------------------------
# Find ParMETIS
#------------------------------------------------------------------------------
find_path(Parmetis_INCLUDE_DIR
    NAMES parmetis.h
    PATHS ${OMEGA_PARMETIS_ROOT}
    PATH_SUFFIXES include
    NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH
)

find_library(Parmetis_LIBRARY
    NAMES libparmetis.a parmetis
    HINTS ${Parmetis_INCLUDE_DIR}/../lib
    NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH
)

#------------------------------------------------------------------------------
# Find METIS
#------------------------------------------------------------------------------
# If not defined, assume the METIS path is the same as ParMETIS
if(NOT DEFINED OMEGA_METIS_ROOT)
    set(OMEGA_METIS_ROOT ${OMEGA_PARMETIS_ROOT})
endif()

find_path(Metis_INCLUDE_DIR
    NAMES metis.h
    PATHS ${OMEGA_METIS_ROOT}
    PATH_SUFFIXES include
    NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH
)

find_library(Metis_LIBRARY
    NAMES libmetis.a metis
    HINTS ${Metis_INCLUDE_DIR}/../lib
    NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH
)

#------------------------------------------------------------------------------
# Find GKlib (optional)
#------------------------------------------------------------------------------
# Assume the GKlib path is the same as METIS if it is not defined.
if(NOT DEFINED OMEGA_GKLIB_ROOT)
    set(OMEGA_GKLIB_ROOT ${OMEGA_PARMETIS_ROOT})
endif()

find_path(GKlib_INCLUDE_DIR
    NAMES GKlib.h
    PATHS ${OMEGA_GKLIB_ROOT}
    PATH_SUFFIXES include
    NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH
)

if(GKlib_INCLUDE_DIR AND NOT GKlib_INCLUDE_DIR STREQUAL "")
    find_library(GKlib_LIBRARY
        NAMES libGKlib.a GKlib
        HINTS ${GKlib_INCLUDE_DIR}/../lib
        NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH
    )
endif()

#------------------------------------------------------------------------------
# Set result variables
#------------------------------------------------------------------------------
set(Parmetis_INCLUDE_DIRS)

# ParMETIS
if(Parmetis_INCLUDE_DIR AND Parmetis_LIBRARY)
    set(Parmetis_FOUND TRUE)
    list(APPEND Parmetis_INCLUDE_DIRS ${Parmetis_INCLUDE_DIR})
    message(STATUS "Found ParMETIS: ${Parmetis_LIBRARY}")
    message(STATUS "  Include: ${Parmetis_INCLUDE_DIR}")
else()
    set(Parmetis_FOUND FALSE)
endif()

# METIS
if(Metis_INCLUDE_DIR AND Metis_LIBRARY)
    set(Metis_FOUND TRUE)
    list(APPEND Parmetis_INCLUDE_DIRS ${Metis_INCLUDE_DIR})
    message(STATUS "Found METIS: ${Metis_LIBRARY}")
    message(STATUS "  Include: ${Metis_INCLUDE_DIR}")
else()
    set(Metis_FOUND FALSE)
endif()

# GKlib
if(GKlib_INCLUDE_DIR AND GKlib_LIBRARY)
    set(GKlib_FOUND TRUE)
    list(APPEND Parmetis_INCLUDE_DIRS ${GKlib_INCLUDE_DIR})
    message(STATUS "Found GKlib: ${GKlib_LIBRARY}")
    message(STATUS "  Include: ${GKlib_INCLUDE_DIR}")
else()
    set(GKlib_FOUND FALSE)
endif()

#------------------------------------------------------------------------------
# Create imported targets (modern CMake approach)
#------------------------------------------------------------------------------

# METIS imported target
if(Metis_FOUND AND NOT TARGET Metis::metis)
    add_library(Metis::metis STATIC IMPORTED)
    set_target_properties(Metis::metis PROPERTIES
        IMPORTED_LOCATION "${Metis_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Metis_INCLUDE_DIR}"
    )
endif()

# GKlib imported target
if(GKlib_FOUND AND NOT TARGET GKlib::gklib)
    add_library(GKlib::gklib STATIC IMPORTED)
    set_target_properties(GKlib::gklib PROPERTIES
        IMPORTED_LOCATION "${GKlib_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GKlib_INCLUDE_DIR}"
    )
endif()

# ParMETIS imported target (depends on METIS and optionally GKlib)
if(Parmetis_FOUND AND NOT TARGET Parmetis::parmetis)
    add_library(Parmetis::parmetis STATIC IMPORTED)
    set_target_properties(Parmetis::parmetis PROPERTIES
        IMPORTED_LOCATION "${Parmetis_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Parmetis_INCLUDE_DIR}"
    )

    # Link dependencies
    if(Metis_FOUND)
        set_property(TARGET Parmetis::parmetis APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES Metis::metis
        )
    endif()
    if(GKlib_FOUND)
        set_property(TARGET Parmetis::parmetis APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES GKlib::gklib
        )
    endif()
endif()

#------------------------------------------------------------------------------
# Create backward-compatible targets (non-namespaced)
#------------------------------------------------------------------------------

# These maintain compatibility with existing code that uses 'parmetis', 'metis', 'gklib'
if(Parmetis_FOUND AND NOT TARGET parmetis)
    add_library(parmetis STATIC IMPORTED)
    set_target_properties(parmetis PROPERTIES
        IMPORTED_LOCATION "${Parmetis_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Parmetis_INCLUDE_DIR}"
    )
endif()

if(Metis_FOUND AND NOT TARGET metis)
    add_library(metis STATIC IMPORTED)
    set_target_properties(metis PROPERTIES
        IMPORTED_LOCATION "${Metis_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Metis_INCLUDE_DIR}"
    )
endif()

if(GKlib_FOUND AND NOT TARGET gklib)
    add_library(gklib STATIC IMPORTED)
    set_target_properties(gklib PROPERTIES
        IMPORTED_LOCATION "${GKlib_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GKlib_INCLUDE_DIR}"
    )
endif()

#------------------------------------------------------------------------------
# Handle required components
#------------------------------------------------------------------------------
if(NOT Parmetis_FOUND OR NOT Metis_FOUND)
    message(STATUS "")
    message(STATUS "ParMETIS or METIS not found.")
    message(STATUS "Please set location with -DOMEGA_PARMETIS_ROOT=<path>")
    message(STATUS "")
endif()

# Use CMake's standard handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Parmetis
    REQUIRED_VARS Parmetis_LIBRARY Parmetis_INCLUDE_DIR Metis_LIBRARY Metis_INCLUDE_DIR
    FAIL_MESSAGE "ParMETIS not found. Set OMEGA_PARMETIS_ROOT to the installation directory."
)

# Mark advanced variables
mark_as_advanced(
    Parmetis_INCLUDE_DIR
    Parmetis_LIBRARY
    Metis_INCLUDE_DIR
    Metis_LIBRARY
    GKlib_INCLUDE_DIR
    GKlib_LIBRARY
)
