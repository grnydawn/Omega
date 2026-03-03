# Omega Unit Tests
# Data-driven test registration

#####################
# Omega Test Function
#####################

function(add_omega_test test_name exe_name source_files mpi_args)
  # Copy extra arguments to a local variable so they can be treated
  # as a list
  set(extra_args ${ARGN})

  # Create the executable
  add_executable(${exe_name} ${source_files})

  # Link the library
  target_link_libraries(
    ${exe_name}
    PRIVATE
    OmegaLibFlags
  )

  list(FIND extra_args single_precision use_sp)

  # Link the single precision version if requested and not already configured to
  # use single precision
  if (NOT use_sp EQUAL -1)
    if (OMEGA_SINGLE_PRECISION)
      target_link_libraries(
        ${exe_name}
        PRIVATE
        ${OMEGA_LIB_NAME}
      )
    else()
      target_link_libraries(
        ${exe_name}
        PRIVATE
        ${OMEGA_LIB_NAME}_SP
      )
    endif()
  else()
    target_link_libraries(
      ${exe_name}
      PRIVATE
      ${OMEGA_LIB_NAME}
    )
  endif()

  set_target_properties(${exe_name} PROPERTIES LINKER_LANGUAGE CXX)

  # Add the test command
  if (mpi_args)

    if("${OMEGA_ARCH}" STREQUAL "SYCL")
      add_test(
        NAME ${test_name}
        COMMAND ${OMEGA_MPI_EXEC} ${mpi_args} ${OMEGA_MPI_ARGS} ./${exe_name}
      )
    else()
      add_test(
        NAME ${test_name}
        COMMAND ${OMEGA_MPI_EXEC} ${OMEGA_MPI_ARGS} ${mpi_args} -- ./${exe_name}
      )
    endif()

  else()
    add_test(
      NAME ${test_name}
      COMMAND ../omega_env.sh ./${exe_name}
    )
  endif()

  set_tests_properties(${test_name} PROPERTIES LABELS "${OMEGA_ARCH};Omega-0")

endfunction()

#####################
# Test Definitions  #
#####################

# Standard tests: TEST_NAME;EXE_NAME;SOURCE;MPI_ARGS
# Format: "NAME|EXE|SOURCE|MPI_ARGS"
set(OMEGA_STANDARD_TESTS
    # Base tests
    "DATA_TYPES_TEST|testDataTypes.exe|base/DataTypesTest.cpp|-n;1"
    "MACHINE_ENV_TEST|testMachEnv.exe|base/MachEnvTest.cpp|-n;8"
    "BROADCAST_TEST|testBroadcast.exe|base/BroadcastTest.cpp|-n;8"
    "DECOMP_NTASK1_TEST|testDecompNTask1.exe|base/DecompTest.cpp|-n;1"
    "DECOMP_NTASK8_TEST|testDecompNTask8.exe|base/DecompTest.cpp|-n;8"
    "HALO_TEST|testHalo.exe|base/HaloTest.cpp|-n;8"
    "IO_TEST|testIO.exe|base/IOTest.cpp|-n;8"
    "REDUCTIONS_TEST|testReductions.exe|base/ReductionsTest.cpp|-n;8"
    "TRIDIAGSOLVERS_TEST|testTriDiagSolvers.exe|base/TriDiagSolversTest.cpp|-n;1"

    # Infra tests
    "LOGGING_TEST|testLogging.exe|infra/LoggingTest.cpp|-n;8"
    "ERROR_TEST|testError.exe|infra/ErrorTest.cpp|-n;8"
    "DIMENSION_TEST|testDimension.exe|infra/DimensionTest.cpp|-n;8"
    "CONFIG_TEST|testConfig.exe|infra/ConfigTest.cpp|-n;8"
    "FIELD_TEST|testField.exe|infra/FieldTest.cpp|-n;8"
    "IOSTREAM_TEST|testIOStream.exe|infra/IOStreamTest.cpp|-n;8"
    "TIMEMGR_TEST|testTimeMgr.exe|infra/TimeMgrTest.cpp|-n;1"
    "TIMEINTERVAL_PARSE_TEST|testTimeIntervalParse.exe|infra/TimeIntervalParseTest.cpp|-n;1"
    "TIMEINTERVAL_PARSE_EXTENDED_FORMATS_TEST|testTimeIntervalParseExtendedFormats.exe|infra/TimeIntervalParseExtendedFormatsTest.cpp|-n;1"
    "KOKKOS_FLATPAR_TEST|testKokkosFlatPar.exe|infra/OmegaKokkosFlatParTest.cpp|-n;1"
    "KOKKOS_HIPAR_TEST|testKokkosHiPar.exe|infra/OmegaKokkosHiParTest.cpp|-n;1"

    # Ocn tests
    "HORZMESH_TEST|testHorzMesh.exe|ocn/HorzMeshTest.cpp|-n;8"
    "AUXSTATE_TEST|testAuxiliaryState.exe|ocn/AuxiliaryStateTest.cpp|-n;8"
    "TENDENCIES_TEST|testTendencies.exe|ocn/TendenciesTest.cpp|-n;8"
    "STATE_TEST|testState.exe|ocn/StateTest.cpp|-n;8"
    "TRACERS_TEST|testTracers.exe|ocn/TracersTest.cpp|-n;8"
    "EOS_TEST|testEos.exe|ocn/EosTest.cpp|-n;2"
    "VERTCOORD_TEST|testVertCoord.exe|ocn/VertCoordTest.cpp|-n;8"
    "VERTMIX_TEST|testVertMix.exe|ocn/VertMixTest.cpp|-n;4"

    # Time stepping tests
    "TIMESTEPPER_TEST|testTimeStepper.exe|timeStepping/TimeStepperTest.cpp|-n;2"

    # Driver tests
    "DRIVER_TEST|testDriver.exe|drivers/StandaloneDriverTest.cpp|-n;8"
)

# Tests with compile definitions: TEST_NAME|EXE_NAME|SOURCE|MPI_ARGS|DEFINITION
set(OMEGA_TESTS_WITH_DEFINITIONS
    "HORZOPERATORS_PLANE_TEST|testHorzOperatorsPlane.exe|ocn/HorzOperatorsTest.cpp|-n;8|HORZOPERATORS_TEST_PLANE"
    "HORZOPERATORS_SPHERE_TEST|testHorzOperatorsSphere.exe|ocn/HorzOperatorsTest.cpp|-n;8|HORZOPERATORS_TEST_SPHERE_1"
    "AUXVARS_PLANE_TEST|testAuxiliaryVarsPlane.exe|ocn/AuxiliaryVarsTest.cpp|-n;8|AUXVARS_TEST_PLANE"
    "AUXVARS_SPHERE_TEST|testAuxiliaryVarsSphere.exe|ocn/AuxiliaryVarsTest.cpp|-n;8|AUXVARS_TEST_SPHERE"
    "TEND_PLANE_TEST|testTendencyTermsPlane.exe|ocn/TendencyTermsTest.cpp|-n;8|TENDENCYTERMS_TEST_PLANE"
    "TEND_SPHERE_TEST|testTendencyTermsSphere.exe|ocn/TendencyTermsTest.cpp|-n;8|TENDENCYTERMS_TEST_SPHERE"
)

# Tests with single precision: TEST_NAME|EXE_NAME|SOURCE|MPI_ARGS|DEFINITION
set(OMEGA_SINGLE_PRECISION_TESTS
    "TEND_PLANE_SINGLE_PRECISION_TEST|testTendencyTermsPlaneSinglePrecision.exe|ocn/TendencyTermsTest.cpp|-n;8|TENDENCYTERMS_TEST_PLANE"
)

# Tests that should fail
set(OMEGA_WILL_FAIL_TESTS
    "ERROR_TEST"
)

#####################
# Register Tests    #
#####################

# Register standard tests
foreach(test_spec ${OMEGA_STANDARD_TESTS})
    string(REPLACE "|" ";" test_parts "${test_spec}")
    list(GET test_parts 0 test_name)
    list(GET test_parts 1 exe_name)
    list(GET test_parts 2 source_file)
    list(GET test_parts 3 mpi_args)

    add_omega_test(${test_name} ${exe_name} ${source_file} "${mpi_args}")
endforeach()

# Register tests with compile definitions
foreach(test_spec ${OMEGA_TESTS_WITH_DEFINITIONS})
    string(REPLACE "|" ";" test_parts "${test_spec}")
    list(GET test_parts 0 test_name)
    list(GET test_parts 1 exe_name)
    list(GET test_parts 2 source_file)
    list(GET test_parts 3 mpi_args)
    list(GET test_parts 4 definition)

    add_omega_test(${test_name} ${exe_name} ${source_file} "${mpi_args}")
    target_compile_definitions(${exe_name} PRIVATE ${definition})
endforeach()

# Register single precision tests
foreach(test_spec ${OMEGA_SINGLE_PRECISION_TESTS})
    string(REPLACE "|" ";" test_parts "${test_spec}")
    list(GET test_parts 0 test_name)
    list(GET test_parts 1 exe_name)
    list(GET test_parts 2 source_file)
    list(GET test_parts 3 mpi_args)
    list(GET test_parts 4 definition)

    add_omega_test(${test_name} ${exe_name} ${source_file} "${mpi_args}" single_precision)
    target_compile_definitions(${exe_name} PRIVATE ${definition})
endforeach()

# Set WILL_FAIL property for expected-to-fail tests
foreach(test_name ${OMEGA_WILL_FAIL_TESTS})
    set_tests_properties(${test_name} PROPERTIES WILL_FAIL true)
endforeach()

##################
# External Tests
##################

# GSW-C test (uses direct command, not add_omega_test)
add_test(NAME GSWC_CALL_TEST
    COMMAND ${OMEGA_BUILD_DIR}/external/GSW-C/gsw_check
)
