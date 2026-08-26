# Omega Unit Tests

# Directory holding one log file per test, created here so that it exists
# before any test runs
set(OMEGA_TEST_LOG_DIR logs)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${OMEGA_TEST_LOG_DIR})

#####################
# Omega Test Function
#####################

function(omega_add_test test_name exe_name source_files mpi_args)
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

  # Give each test its own log file (relative to the test working directory,
  # which is this binary directory) so that the log of a test can be read on
  # its own rather than mixed in with every other test's messages. The name
  # comes from the test rather than the source file because several tests
  # share a source file.
  set_tests_properties(
    ${test_name}
    PROPERTIES
    LABELS "${OMEGA_ARCH};Omega-0"
    ENVIRONMENT "OMEGA_LOG_FILE=${OMEGA_TEST_LOG_DIR}/${test_name}.log"
  )

endfunction()

##################
# Data type test
##################

omega_add_test(
    DATA_TYPES_TEST
    testDataTypes.exe
    base/DataTypesTest.cpp
    "-n;1"
)

##################
# Machine env test
##################

omega_add_test(
    MACHINE_ENV_TEST
    testMachEnv.exe
    base/MachEnvTest.cpp
    "-n;8"
)

##################
# Broadcast test
##################

omega_add_test(
    BROADCAST_TEST
    testBroadcast.exe
    base/BroadcastTest.cpp
    "-n;8"
)

##################
# Logging test
##################

omega_add_test(
    LOGGING_TEST
    testLogging.exe
    infra/LoggingTest.cpp
    "-n;8"
)

##################
# Error test
##################

omega_add_test(
    ERROR_TEST
    testError.exe
    infra/ErrorTest.cpp
    "-n;8"
)
set_tests_properties(ERROR_TEST PROPERTIES WILL_FAIL true)

##########################
# Decomp test using 1 task
##########################

omega_add_test(
    DECOMP_NTASK1_TEST
    testDecompNTask1.exe
    base/DecompTest.cpp
    "-n;1"
)

###########################
# Decomp test using 8 tasks
###########################

omega_add_test(
    DECOMP_NTASK8_TEST
    testDecompNTask8.exe
    base/DecompTest.cpp
    "-n;8"
)

##################
# Halo test
##################

omega_add_test(
    HALO_TEST
    testHalo.exe
    base/HaloTest.cpp
    "-n;8"
)

##################
# Dimension test
##################

omega_add_test(
    DIMENSION_TEST
    testDimension.exe
    infra/DimensionTest.cpp
    "-n;8"
)

################
# HorzMesh test
################

omega_add_test(
    HORZMESH_TEST
    testHorzMesh.exe
    ocn/HorzMeshTest.cpp
    "-n;8"
)

#######################
# HorzOperators tests
#######################

omega_add_test(
    HORZOPERATORS_PLANE_TEST
    testHorzOperatorsPlane.exe
    ocn/HorzOperatorsTest.cpp
    "-n;8"
)
target_compile_definitions(
  testHorzOperatorsPlane.exe
  PRIVATE
  HORZOPERATORS_TEST_PLANE
)

omega_add_test(
  HORZOPERATORS_SPHERE_TEST
    testHorzOperatorsSphere.exe
    ocn/HorzOperatorsTest.cpp
    "-n;8"
)
target_compile_definitions(
  testHorzOperatorsSphere.exe
  PRIVATE
  HORZOPERATORS_TEST_SPHERE_1
)
################
# AuxVars test
################

omega_add_test(
  AUXVARS_PLANE_TEST
    testAuxiliaryVarsPlane.exe
    ocn/AuxiliaryVarsTest.cpp
    "-n;8"
)
target_compile_definitions(
  testAuxiliaryVarsPlane.exe
  PRIVATE
  AUXVARS_TEST_PLANE
)

omega_add_test(
  AUXVARS_SPHERE_TEST
  testAuxiliaryVarsSphere.exe
    ocn/AuxiliaryVarsTest.cpp
    "-n;8"
)
target_compile_definitions(
  testAuxiliaryVarsSphere.exe
  PRIVATE
  AUXVARS_TEST_SPHERE
)

################
# AuxState test
################

omega_add_test(
  AUXSTATE_TEST
    testAuxiliaryState.exe
    ocn/AuxiliaryStateTest.cpp
    "-n;8"
)

#############
# IO test
#############

omega_add_test(
    IO_TEST
    testIO.exe
    base/IOTest.cpp
    "-n;8"
)

##################
# Config test
##################

omega_add_test(
    CONFIG_TEST
    testConfig.exe
    infra/ConfigTest.cpp
    "-n;8"
)

##################
# Field test
##################

omega_add_test(
    FIELD_TEST
    testField.exe
    infra/FieldTest.cpp
    "-n;8"
)

##################
# IOStream test
##################

omega_add_test(
    IOSTREAM_TEST
    testIOStream.exe
    infra/IOStreamTest.cpp
    "-n;8"
)

#####################
# TendencyTerms test
#####################

omega_add_test(
    TEND_PLANE_TEST
    testTendencyTermsPlane.exe
    ocn/TendencyTermsTest.cpp
    "-n;8"
)
target_compile_definitions(
  testTendencyTermsPlane.exe
  PRIVATE
  TENDENCYTERMS_TEST_PLANE
)

omega_add_test(
    TEND_PLANE_SINGLE_PRECISION_TEST
    testTendencyTermsPlaneSinglePrecision.exe
    ocn/TendencyTermsTest.cpp
    "-n;8"
    single_precision
)
target_compile_definitions(
  testTendencyTermsPlaneSinglePrecision.exe
  PRIVATE
  TENDENCYTERMS_TEST_PLANE
)

omega_add_test(
  TEND_SPHERE_TEST
    testTendencyTermsSphere.exe
    ocn/TendencyTermsTest.cpp
    "-n;8"
)
target_compile_definitions(
  testTendencyTermsSphere.exe
  PRIVATE
  TENDENCYTERMS_TEST_SPHERE
)

################
# Forcing test
################

omega_add_test(
  FORCING_PLANE_TEST
  testForcingPlane.exe
  ocn/ForcingTest.cpp
  "-n;1"
)
target_compile_definitions(
  testForcingPlane.exe
  PRIVATE
  FORCING_TEST_PLANE
)

omega_add_test(
  FORCING_SPHERE_TEST
  testForcingSphere.exe
  ocn/ForcingTest.cpp
  "-n;1"
)
target_compile_definitions(
  testForcingSphere.exe
  PRIVATE
  FORCING_TEST_SPHERE
)

#################
# Tendencies test
#################

omega_add_test(
    TENDENCIES_TEST
    testTendencies.exe
    ocn/TendenciesTest.cpp
    "-n;8"
)

##################
# Pressure gradient test
##################

omega_add_test(
  PGRAD_TEST
  testPGrad.exe
  ocn/PGradTest.cpp
  "-n;1"
)

##################
# State test
##################

omega_add_test(
    STATE_TEST
    testState.exe
    ocn/StateTest.cpp
    "-n;8"
)

##################
# Time Manager test
##################

omega_add_test(
    TIMEMGR_TEST
    testTimeMgr.exe
    infra/TimeMgrTest.cpp
    "-n;1"
)

############################
# TimeInterval parsing tests
############################

omega_add_test(
  TIMEINTERVAL_PARSE_TEST
  testTimeIntervalParse.exe
  infra/TimeIntervalParseTest.cpp
  "-n;1"
)

omega_add_test(
  TIMEINTERVAL_PARSE_EXTENDED_FORMATS_TEST
  testTimeIntervalParseExtendedFormats.exe
  infra/TimeIntervalParseExtendedFormatsTest.cpp
  "-n;1"
)

##################
# Reductions test
##################

omega_add_test(
    REDUCTIONS_TEST
    testReductions.exe
    base/ReductionsTest.cpp
    "-n;8;"
)

###################
# Time Stepper test
###################

omega_add_test(
  TIMESTEPPER_TEST
  testTimeStepper.exe
  timeStepping/TimeStepperTest.cpp
    "-n;2"
)

#######################
# Kokkos wrappers tests
#######################

omega_add_test(
  KOKKOS_FLATPAR_TEST
    testKokkosFlatPar.exe
    infra/OmegaKokkosFlatParTest.cpp
    "-n;1"
)

omega_add_test(
  KOKKOS_HIPAR_TEST
    testKokkosHiPar.exe
    infra/OmegaKokkosHiParTest.cpp
    "-n;1;"
)

##########################
# Build configuration test
##########################

omega_add_test(
    BUILD_CONFIG_TEST
    testBuildConfig.exe
    infra/BuildConfigTest.cpp
    "-n;1"
)

##################
# Driver test
##################

omega_add_test(
    DRIVER_TEST
    testDriver.exe
    drivers/StandaloneDriverTest.cpp
    "-n;8"
)

##################
# Tracers test
##################

omega_add_test(
    TRACERS_TEST
    testTracers.exe
    ocn/TracersTest.cpp
    "-n;8"
)

##########################
# Tridiagonal solvers test
##########################

omega_add_test(
    TRIDIAGSOLVERS_TEST
    testTriDiagSolvers.exe
    base/TriDiagSolversTest.cpp
    "-n;1"
)

##################
# EOS tests
##################
add_test(NAME GSWC_CALL_TEST
    COMMAND ${OMEGA_BUILD_DIR}/external/GSW-C/gsw_check
)

omega_add_test(
    EOS_TEST
    testEos.exe
    ocn/EosTest.cpp
    "-n;2"
)

################
# VertCoord test
################

omega_add_test(
    VERTCOORD_TEST
    testVertCoord.exe
    ocn/VertCoordTest.cpp
    "-n;8"
)

omega_add_test(
    VERTCOORD_SINGLE_PRECISION_TEST
    testVertCoordSinglePrecision.exe
    ocn/VertCoordTest.cpp
    "-n;8"
    single_precision
)

##################
# Fill Value test
##################

omega_add_test(
    FILL_VALUE_TEST
    testFillValue.exe
    ocn/FillValueTest.cpp
    "-n;8"
)

##########################
# Vert Mix test
##########################

omega_add_test(
    VERTMIX_TEST
    testVertMix.exe
    ocn/VertMixTest.cpp
    "-n;4"
)

##################
# VAdv test
##################

omega_add_test(
    VERTADV_TEST
    testVertAdv.exe
    ocn/VertAdvTest.cpp
    "-n;1"
)

##########################
# State Validation test
##########################

omega_add_test(
    STATE_VALIDATION_TEST
    testStateValidation.exe
    ocn/StateValidationTest.cpp
    "-n;8"
)

#####################
# SurfaceCouling test
#####################

omega_add_test(
    SFCCOUPLING_TEST
    testSfcCoupling.exe
    ocn/SfcCouplingTest.cpp
    "-n;8"
)
##################
# Analysis tests
##################

omega_add_test(
    ANALYSIS_OP_TEST
    testAnalysisOps.exe
    analysis/AnalysisOperatorTest.cpp
    "-n;8"
)

omega_add_test(
    ANALYSIS_SYS_TEST
    testAnalysisSys.exe
    analysis/AnalysisSystemTest.cpp
    "-n;8"
)
