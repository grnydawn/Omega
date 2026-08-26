# Omega Top-level CMakeLists.txt
#
# - supports two build modes:
#    1) standalone build
#    2) e3sm build
#
# The build runs as six ordered phases: every macro called below declares the
# phase it belongs to, so a call from the wrong place is a configure-time error
# rather than a build that quietly comes out wrong.


# Included scripts do automatic cmake_policy() PUSH and POP
cmake_policy(SET CMP0011 NEW)

# list command no longer ignores empty elements
cmake_policy(SET CMP0007 NEW)

# Only interpret if() arguments as variables or keywords when unquoted
cmake_policy(SET CMP0054 NEW)

#find_package() uses <PackageName>_ROOT variables.
cmake_policy(SET CMP0074 NEW)

# Do not configure Omega at all when it is not the ocean component.
#
# E3SM calls build_omega() unconditionally (components/CMakeLists.txt), unlike
# build_mpas_models(), which is gated on the case's Buildconf. Without this
# guard a CMPASO, DOCN or SOCN case configures the whole of Omega - spdlog,
# yaml-cpp, Kokkos, scorpio - and then fails outright, because
# src/CMakeLists.txt creates an 'ocn' target that collides with the one the
# real ocean component
# creates (mpas-framework/src/build_core.cmake for MPAS-O,
# components/cmake/build_model.cmake for a data or stub ocean):
#
#   add_library cannot create target "ocn" because another target with the same
#   name already exists.
#
# Returning here is the fix that lives inside Omega. The tidier fix is upstream:
# give build_omega() the same participation guard build_mpas_models() has, so
# E3SM does not descend into Omega for cases that do not use it. That change
# belongs in components/cmake/build_omega.cmake and is deliberately not made
# here; see doc/devGuide/CMakeBuild.md.
#
# COMP_OCN is passed by CIME (-DCOMP_OCN=...). The test is written so that Omega
# still configures whenever COMP_OCN is absent, which is every standalone build.
if(DEFINED COMP_OCN AND NOT "${COMP_OCN}" STREQUAL "omega")
  message(STATUS
    "Omega: COMP_OCN='${COMP_OCN}' is not omega; skipping the Omega build.")
  return()
endif()

# define variables and macros for Omega build
include(${CMAKE_CURRENT_SOURCE_DIR}/OmegaBuild.cmake)

###########################################################
# PHASE 1: Setup                                          #
#                                                         #
# Assign values to the Omega build-controlling variables: #
# declare the options, discover the machine and compiler  #
# settings, and detect the target architecture.           #
###########################################################

omega_begin_phase(1 SETUP)

# common pre-processing for standalone and e3sm mode
omega_common()

# handle build modes
if (NOT DEFINED PROJECT_NAME)
  # enter standalone build

  cmake_minimum_required(VERSION 3.21) # the minimum version for HIP support

  # Establish the build mode and build type first, so that the CIME discovery
  # below can pass the right context to the machine macros.
  omega_setup_standalone_build()

  # Collect machine and compiler info from CIME
  omega_init_standalone_build()

else()
  # enter embedded build

  if(DEFINED E3SM_SOURCE_DIR) # E3SM_SOURCE_DIR = E3SM_ROOT/components

    # update variables for e3sm build
    omega_setup_e3sm_build()

  else()
    message(FATAL_ERROR "UNKNOWN OMEGA BUILD MODE")

  endif()

endif()

omega_phase_produces(SETUP OMEGA_BUILD_MODE OMEGA_BUILD_TYPE)

###########################################################
# PHASE 2: Toolchain                                      #
#                                                         #
# Commit the compilers and flags, then call project().    #
# This phase exists because CMake requires the toolchain   #
# to be final before project(); it is a no-op in E3SM     #
# mode, where the parent already did both.                #
###########################################################

omega_begin_phase(2 TOOLCHAIN)

if ("${OMEGA_BUILD_MODE}" STREQUAL "STANDALONE")

  omega_commit_toolchain()

  set(CMAKE_CXX_STANDARD 17) # used in E3SM
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  project(${OMEGA_PROJECT_NAME}
          LANGUAGES C CXX
         )

  set(LINKER_LANGUAGE CXX)   # needed to support Cray C compiler wrapper

endif()

###########################################################
# PHASE 3: Update                                         #
#                                                         #
# update cmake & kokkos variables, and adjust             #
# build-controlling variables                             #
###########################################################

omega_begin_phase(3 UPDATE)

# update CMake and Kokkos variables from the build-controlling variables
omega_update_variables()

omega_phase_produces(UPDATE OMEGA_ARCH CMAKE_BUILD_TYPE)

###########################################################
# PHASE 4: Validate                                       #
#                                                         #
# check that everything phases 1-3 produced is            #
# self-consistent, before anything is built               #
###########################################################

omega_begin_phase(4 VALIDATE)

omega_check_setup()

###########################################################
# PHASE 5: Build                                          #
#                                                         #
# resolve external dependencies, verify they produced the #
# targets Omega links, then build Omega and its tests     #
###########################################################

omega_begin_phase(5 BUILD)

add_subdirectory(external)

# The externals exist now, so this is the first point at which the targets
# src/CMakeLists.txt links can be checked.
omega_check_dependencies()

add_subdirectory(src)

if(OMEGA_BUILD_TEST)

  enable_testing()

  if (OMEGA_TEST_CDASH)
    include(CTest)
  endif()

  add_subdirectory(test)

endif()

###########################################################
# PHASE 6: Output                                         #
#                                                         #
# prepare outputs according to the output-controlling     #
# variables                                               #
###########################################################

omega_begin_phase(6 OUTPUT)

if ("${OMEGA_BUILD_MODE}" STREQUAL "STANDALONE")
  omega_write_dev_scripts()
endif()

omega_wrap_outputs()
