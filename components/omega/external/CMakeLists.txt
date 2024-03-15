# Add external packages

include(ExternalProject)

# Add the spdlog library
add_subdirectory(
  ${E3SM_EXTERNALS_ROOT}/ekat/extern/spdlog
  ${CMAKE_CURRENT_BINARY_DIR}/ekat/extern/spdlog
)

# Add the yaml-cpp library
add_subdirectory(
  ${E3SM_EXTERNALS_ROOT}/ekat/extern/yaml-cpp
  ${CMAKE_CURRENT_BINARY_DIR}/ekat/extern/yaml-cpp
)

if (KOKKOS_OPTIONS)
  unset(KOKKOS_OPTIONS)
endif()

# Add the Kokkos library
if(OMEGA_ARCH STREQUAL "CUDA")
  add_subdirectory(
    ${E3SM_EXTERNALS_ROOT}/ekat/extern/kokkos
    ${CMAKE_CURRENT_BINARY_DIR}/ekat/extern/kokkos
  )

  set(_KOKKOS_LIB Kokkos::kokkos PARENT_SCOPE)

elseif(OMEGA_ARCH STREQUAL "HIP")
  ExternalProject_Add(
    "hip_build"
    SOURCE_DIR "${E3SM_EXTERNALS_ROOT}/ekat/extern/kokkos"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/ekat/extern/kokkos"
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    CMAKE_ARGS
      "-DKokkos_ENABLE_HIP=ON"
      "-DCMAKE_CXX_COMPILER=${OMEGA_HIP_COMPILER}"
      "-DCMAKE_CXX_FLAGS=${OMEGA_HIP_FLAGS}"
      "-DCMAKE_BUILD_TYPE=${OMEGA_BUILD_TYPE}"
      "-DCMAKE_INSTALL_PREFIX=${OMEGA_BUILD_DIR}/kokkos"
      "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
      "-DCRAYPE_LINK_TYPE=dynamic"
    BUILD_COMMAND make -j4 install
  )

  set(_KOKKOS_LIB
    ${OMEGA_BUILD_DIR}/kokkos/lib/libkokkoscontainers.a
    ${OMEGA_BUILD_DIR}/kokkos/lib/libkokkoscore.a
    PARENT_SCOPE
  )

else()
  message(FATAL_ERROR "${OMEGA_ARCH} is not supported yet.")
endif()

if(OMEGA_ARCH STREQUAL "HIP")

else()
endif()

# forward env. variables to Scorpio build
if(DEFINED ENV{NETCDF_PATH})
  set(NetCDF_PATH $ENV{NETCDF_PATH}
		CACHE STRING "Path to NETCDF library" FORCE)
endif()
if(DEFINED ENV{NETCDF_C_PATH})
  set(NetCDF_C_PATH $ENV{NETCDF_C_PATH}
		CACHE STRING "Path to NETCDF-C library" FORCE)
endif()
if(DEFINED ENV{NETCDF_FORTRAN_PATH})
  set(NetCDF_Fortran_PATH $ENV{NETCDF_FORTRAN_PATH}
		CACHE STRING "Path to NETCDF-Fortran library" FORCE)
endif()
if(DEFINED ENV{PNETCDF_PATH})
  set(PnetCDF_PATH $ENV{PNETCDF_PATH}
		CACHE STRING "Path to PNETCDF library" FORCE)
endif()

option(PIO_ENABLE_TOOLS "" OFF)

# Add the Scorpio library
add_subdirectory(
  ${E3SM_EXTERNALS_ROOT}/scorpio
  ${CMAKE_CURRENT_BINARY_DIR}/scorpio
)

# Add the parmetis and related libraries

if(Parmetis_FOUND)
  add_library(parmetis STATIC IMPORTED GLOBAL)
  set_target_properties(parmetis PROPERTIES
     IMPORTED_LOCATION ${Parmetis_LIBRARY})
endif()

if(Metis_FOUND)
  add_library(metis STATIC IMPORTED GLOBAL)
  set_target_properties(metis PROPERTIES
     IMPORTED_LOCATION ${Metis_LIBRARY})
endif()

if(GKlib_FOUND)
  add_library(gklib STATIC IMPORTED GLOBAL)
  set_target_properties(gklib PROPERTIES
     IMPORTED_LOCATION ${GKlib_LIBRARY})
endif()
