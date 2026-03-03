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
# Helper macro for registering tests
#####################

# Register a standard test
macro(omega_register_test test_name exe_name source_file mpi_args)
  add_omega_test(${test_name} ${exe_name} ${source_file} "${mpi_args}")
endmacro()

# Register a test with compile definition
macro(omega_register_test_with_def test_name exe_name source_file mpi_args definition)
  add_omega_test(${test_name} ${exe_name} ${source_file} "${mpi_args}")
  target_compile_definitions(${exe_name} PRIVATE ${definition})
endmacro()

# Register a single precision test with compile definition
macro(omega_register_sp_test test_name exe_name source_file mpi_args definition)
  add_omega_test(${test_name} ${exe_name} ${source_file} "${mpi_args}" single_precision)
  target_compile_definitions(${exe_name} PRIVATE ${definition})
endmacro()

#####################
# Base Tests
#####################

omega_register_test(DATA_TYPES_TEST       testDataTypes.exe       base/DataTypesTest.cpp       "-n;1")
omega_register_test(MACHINE_ENV_TEST      testMachEnv.exe         base/MachEnvTest.cpp         "-n;8")
omega_register_test(BROADCAST_TEST        testBroadcast.exe       base/BroadcastTest.cpp       "-n;8")
omega_register_test(DECOMP_NTASK1_TEST    testDecompNTask1.exe    base/DecompTest.cpp          "-n;1")
omega_register_test(DECOMP_NTASK8_TEST    testDecompNTask8.exe    base/DecompTest.cpp          "-n;8")
omega_register_test(HALO_TEST             testHalo.exe            base/HaloTest.cpp            "-n;8")
omega_register_test(IO_TEST               testIO.exe              base/IOTest.cpp              "-n;8")
omega_register_test(REDUCTIONS_TEST       testReductions.exe      base/ReductionsTest.cpp      "-n;8")
omega_register_test(TRIDIAGSOLVERS_TEST   testTriDiagSolvers.exe  base/TriDiagSolversTest.cpp  "-n;1")

#####################
# Infra Tests
#####################

omega_register_test(LOGGING_TEST          testLogging.exe         infra/LoggingTest.cpp        "-n;8")
omega_register_test(ERROR_TEST            testError.exe           infra/ErrorTest.cpp          "-n;8")
omega_register_test(DIMENSION_TEST        testDimension.exe       infra/DimensionTest.cpp      "-n;8")
omega_register_test(CONFIG_TEST           testConfig.exe          infra/ConfigTest.cpp         "-n;8")
omega_register_test(FIELD_TEST            testField.exe           infra/FieldTest.cpp          "-n;8")
omega_register_test(IOSTREAM_TEST         testIOStream.exe        infra/IOStreamTest.cpp       "-n;8")
omega_register_test(TIMEMGR_TEST          testTimeMgr.exe         infra/TimeMgrTest.cpp        "-n;1")
omega_register_test(TIMEINTERVAL_PARSE_TEST
                    testTimeIntervalParse.exe
                    infra/TimeIntervalParseTest.cpp
                    "-n;1")
omega_register_test(TIMEINTERVAL_PARSE_EXTENDED_FORMATS_TEST
                    testTimeIntervalParseExtendedFormats.exe
                    infra/TimeIntervalParseExtendedFormatsTest.cpp
                    "-n;1")
omega_register_test(KOKKOS_FLATPAR_TEST   testKokkosFlatPar.exe   infra/OmegaKokkosFlatParTest.cpp "-n;1")
omega_register_test(KOKKOS_HIPAR_TEST     testKokkosHiPar.exe     infra/OmegaKokkosHiParTest.cpp   "-n;1")

# ERROR_TEST should fail
set_tests_properties(ERROR_TEST PROPERTIES WILL_FAIL true)

#####################
# Ocean Tests
#####################

omega_register_test(HORZMESH_TEST         testHorzMesh.exe        ocn/HorzMeshTest.cpp         "-n;8")
omega_register_test(AUXSTATE_TEST         testAuxiliaryState.exe  ocn/AuxiliaryStateTest.cpp   "-n;8")
omega_register_test(TENDENCIES_TEST       testTendencies.exe      ocn/TendenciesTest.cpp       "-n;8")
omega_register_test(STATE_TEST            testState.exe           ocn/StateTest.cpp            "-n;8")
omega_register_test(TRACERS_TEST          testTracers.exe         ocn/TracersTest.cpp          "-n;8")
omega_register_test(EOS_TEST              testEos.exe             ocn/EosTest.cpp              "-n;2")
omega_register_test(VERTCOORD_TEST        testVertCoord.exe       ocn/VertCoordTest.cpp        "-n;8")
omega_register_test(VERTMIX_TEST          testVertMix.exe         ocn/VertMixTest.cpp          "-n;4")

# Tests with compile definitions
omega_register_test_with_def(HORZOPERATORS_PLANE_TEST
                             testHorzOperatorsPlane.exe
                             ocn/HorzOperatorsTest.cpp
                             "-n;8"
                             HORZOPERATORS_TEST_PLANE)

omega_register_test_with_def(HORZOPERATORS_SPHERE_TEST
                             testHorzOperatorsSphere.exe
                             ocn/HorzOperatorsTest.cpp
                             "-n;8"
                             HORZOPERATORS_TEST_SPHERE_1)

omega_register_test_with_def(AUXVARS_PLANE_TEST
                             testAuxiliaryVarsPlane.exe
                             ocn/AuxiliaryVarsTest.cpp
                             "-n;8"
                             AUXVARS_TEST_PLANE)

omega_register_test_with_def(AUXVARS_SPHERE_TEST
                             testAuxiliaryVarsSphere.exe
                             ocn/AuxiliaryVarsTest.cpp
                             "-n;8"
                             AUXVARS_TEST_SPHERE)

omega_register_test_with_def(TEND_PLANE_TEST
                             testTendencyTermsPlane.exe
                             ocn/TendencyTermsTest.cpp
                             "-n;8"
                             TENDENCYTERMS_TEST_PLANE)

omega_register_test_with_def(TEND_SPHERE_TEST
                             testTendencyTermsSphere.exe
                             ocn/TendencyTermsTest.cpp
                             "-n;8"
                             TENDENCYTERMS_TEST_SPHERE)

# Single precision test with compile definition
omega_register_sp_test(TEND_PLANE_SINGLE_PRECISION_TEST
                       testTendencyTermsPlaneSinglePrecision.exe
                       ocn/TendencyTermsTest.cpp
                       "-n;8"
                       TENDENCYTERMS_TEST_PLANE)

#####################
# Time Stepping Tests
#####################

omega_register_test(TIMESTEPPER_TEST      testTimeStepper.exe     timeStepping/TimeStepperTest.cpp "-n;2")

#####################
# Driver Tests
#####################

omega_register_test(DRIVER_TEST           testDriver.exe          drivers/StandaloneDriverTest.cpp "-n;8")

#####################
# External Tests
#####################

# GSW-C test (uses direct command, not add_omega_test)
add_test(NAME GSWC_CALL_TEST
    COMMAND ${OMEGA_BUILD_DIR}/external/GSW-C/gsw_check
)
