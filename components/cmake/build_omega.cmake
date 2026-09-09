function(build_omega)

  # Read the machine's settings here rather than inside Omega.
  #
  # components/CMakeLists.txt loads Macros.cmake inside function(set_compilers_e3sm)
  # deliberately -- "we do not want to pollute the environment with other vars coming
  # from Macros.cmake" -- and exports only the three compilers, E3SM_DEFAULT_BUILD_TYPE,
  # USE_CUDA and USE_HIP. Not USE_SYCL, and no flags. Omega used to recover the rest by
  # including Macros.cmake a second time inside a function of its own and hand-carrying
  # each variable back out with PARENT_SCOPE, which meant a list of names that had to be
  # kept right and was not: it forwarded SYCL_FLAGS alone at first, so the coupled build
  # silently lacked the machine's SYCL settings.
  #
  # add_subdirectory() called inside a function inherits that function's scope, so this
  # include puts every variable Macros.cmake sets in front of components/omega and its
  # children, and in front of no other component -- the same encapsulation set_compilers_e3sm
  # wanted, one level lower. build_eamxx() and build_model() include this same file for the
  # same reason.
  #
  # Note that only the general C++ flags may travel this way. The machine's architecture
  # flags -- -fsycl and friends -- are put on the OmegaLibFlags target inside Omega,
  # because components/omega vendors third-party projects that cannot compile with them.
  include(${CMAKE_SOURCE_DIR}/cmake/common_setup.cmake)

  # Put back the NDEBUG that the include costs us.
  #
  # CMake's own cache default is CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG", and that is what
  # components/omega used before, since nothing shadowed it here. universal.cmake:20-27
  # blanks all six CMAKE_<LANG>_FLAGS_{DEBUG,RELEASE}, and the compiler macro re-adds only
  # an optimization level -- " -O" for gnu, " -O2" for intel, " -fp-model precise -O2
  # --offload-compress" for oneapi-ifxgpu. None of them re-adds -DNDEBUG.
  #
  # CIME supplies it in CPPDEFS instead (common_setup.cmake:58-59), which the other
  # components apply to their targets: build_model.cmake:368 and build_eamxx.cmake:27.
  # Omega applies no CPPDEFS, so without these two lines the whole subtree -- Omega's own
  # sources and the third-party projects under external/ -- would compile a RELEASE build
  # with assert() live.
  #
  # Only these two, rather than add_compile_definitions("${CPPDEFS}") as build_eamxx does:
  # that would inject FORTRANUNDERSCORE, CPRINTEL, HAVE_MPI and the rest into the vendored
  # subtree by directory scope, which is what OmegaBuild.cmake's flag handling exists to
  # avoid, and would double up on share/timing, which applies CPPDEFS to gptl itself.
  # Appending is unconditional because the _RELEASE variables are consulted only when
  # CMAKE_BUILD_TYPE is RELEASE, which is exactly the non-DEBUG case.
  string(APPEND CMAKE_C_FLAGS_RELEASE   " -DNDEBUG")
  string(APPEND CMAKE_CXX_FLAGS_RELEASE " -DNDEBUG")

  # Set CIME source path relative to components
  set(CIMESRC_PATH "../cime/src")

  add_subdirectory("omega")

endfunction(build_omega)
