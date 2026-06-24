# CTest / CDash configuration for the Omega standalone build.
#
# Read automatically by include(CTest) (invoked from CMakeLists.txt under
# OMEGA_TEST_CDASH). Mirrors the eamxx precedent (components/eamxx/CTestConfig.cmake)
# and wires the gcov coverage collector so ctest_coverage() (called in
# CTestScript.cmake) can report coverage to CDash via the existing ctest_submit
# plumbing (issue #1, decision 4).

set(CTEST_PROJECT_NAME "E3SM")
string(TIMESTAMP CURRTIME "%H:%M:%S" UTC)
set(CTEST_NIGHTLY_START_TIME "${CURRTIME} UTC")

set(CTEST_DROP_METHOD "https")
set(CTEST_CURL_OPTIONS
    "CURLOPT_SSL_VERIFYPEER_OFF"
    "CURLOPT_SSL_VERIFYHOST_OFF"
)
set(CTEST_DROP_SITE "my.cdash.org")
set(CTEST_DROP_LOCATION "/submit.php?project=E3SM")
set(CTEST_DROP_SITE_CDASH TRUE)

# Coverage collector for ctest_coverage() (gcov path). The aggregate
# COVERAGE_REPORT ctest test is the gating signal; CDash coverage is the
# dashboard view of the same data.
set(CTEST_COVERAGE_COMMAND "gcov")
