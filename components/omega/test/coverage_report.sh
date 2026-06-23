#!/usr/bin/env bash
#
# coverage_report.sh - aggregate Omega code coverage and gate it.
#
# Invoked as the FINAL ctest test (COVERAGE_REPORT) registered in
# test/CMakeLists.txt via a CTest FIXTURES_CLEANUP, so it runs after every
# instrumented unit test has produced its coverage data.
#
# It performs a SINGLE aggregate sweep over all test executables' coverage data
# (not a per-test report), excludes vendored + test code (decision 6), prints
# per-toolchain line coverage, and exits non-zero when host/CPU line coverage is
# below the threshold (decision 3). Device-backend numbers are reported without a
# hard gate for v1.
#
# Toolchain is selected by OMEGA_COVERAGE_TOOLCHAIN (set by OmegaBuild.cmake):
#   gcov  -> gcovr sweep of *.gcda  (gcc/CPU baseline; lcov-format output)
#   llvm  -> llvm-profdata merge of *.profraw + llvm-cov (clang/icpx/SYCL)
#
# Environment (exported by the generated omega_coverage.sh, or passed by ctest):
#   OMEGA_COVERAGE_TOOLCHAIN   gcov | llvm
#   OMEGA_COVERAGE_THRESHOLD   host/CPU line-coverage gate, percent (e.g. 90)
#   OMEGA_SOURCE_DIR           components/omega source root
#   OMEGA_BUILD_DIR            the build directory (cwd of the ctest run)
#
# Usage: coverage_report.sh [build_dir]

set -u

BUILD_DIR="${1:-${OMEGA_BUILD_DIR:-$(pwd)}}"
# The ctest working dir for this test is build/test; the coverage data (.gcda /
# .profraw) lives under the whole build tree, so sweep from the build root.
if [ -d "${BUILD_DIR}/test" ] && [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
  # We were given build/test; step up to the build root.
  BUILD_DIR="$(cd "${BUILD_DIR}/.." && pwd)"
fi

TOOLCHAIN="${OMEGA_COVERAGE_TOOLCHAIN:-gcov}"
THRESHOLD="${OMEGA_COVERAGE_THRESHOLD:-90}"
SRC_ROOT="${OMEGA_SOURCE_DIR:-${BUILD_DIR}/../../components/omega}"

# Exclusion patterns (decision 6 / mam4xx pattern): vendored third-party code and
# the test sources themselves are never counted toward Omega's own src/ total.
EXCLUDE_RE='(.*/external/.*|.*/test/.*)'

echo "=============================================================="
echo " Omega COVERAGE_REPORT"
echo "   toolchain : ${TOOLCHAIN}"
echo "   threshold : ${THRESHOLD}% (host/CPU line coverage, hard gate)"
echo "   build dir : ${BUILD_DIR}"
echo "   src root  : ${SRC_ROOT}"
echo "   exclude   : */external/* , */test/*"
echo "=============================================================="

cd "${BUILD_DIR}" || { echo "ERROR: cannot cd to build dir ${BUILD_DIR}"; exit 2; }

# extract_pct <number>: echo an integer-truncated percentage for comparison.
gate_against_threshold() {
  local pct="$1"
  # strip a trailing % and any decimals for an integer compare
  local int_pct
  int_pct="$(printf '%.0f' "${pct%\%}" 2>/dev/null || echo 0)"
  echo ""
  echo "--------------------------------------------------------------"
  printf ' HOST/CPU line coverage: %s%% (gate: %s%%)\n' "${pct%\%}" "${THRESHOLD}"
  if [ "${int_pct}" -lt "${THRESHOLD}" ]; then
    echo " RESULT: FAIL - host coverage below threshold."
    echo "--------------------------------------------------------------"
    return 1
  fi
  echo " RESULT: PASS"
  echo "--------------------------------------------------------------"
  return 0
}

run_gcov_path() {
  # Single aggregate sweep over all .gcda across all test executables.
  local n_gcda
  n_gcda="$(find . -name '*.gcda' 2>/dev/null | wc -l | tr -d ' ')"
  echo "Found ${n_gcda} .gcda coverage data file(s)."
  if [ "${n_gcda}" -eq 0 ]; then
    echo "ERROR: no .gcda files - the suite did not run instrumented, or"
    echo "       coverage data was cleaned. Did you build with -DOMEGA_COVERAGE=ON"
    echo "       and -DOMEGA_BUILD_TYPE=Debug, then run the tests?"
    return 1
  fi

  if ! command -v gcovr >/dev/null 2>&1; then
    echo "ERROR: gcovr not found on PATH (gcc/gcov coverage path)."
    echo "       Install gcovr (see components/omega/dev-conda.txt) or use lcov."
    return 1
  fi

  # gcovr produces lcov-format (coverage.info) AND a Cobertura/JSON summary.
  # --root scopes to Omega; -e excludes vendored + test code.
  echo "Running aggregate gcovr sweep..."
  gcovr \
    --root "${SRC_ROOT}" \
    --exclude '.*/external/.*' \
    --exclude '.*/test/.*' \
    --print-summary \
    --lcov coverage.info \
    --json-summary-pretty -o coverage.json \
    "${BUILD_DIR}" 2>&1 | tee coverage_summary.txt

  # Pull the line-coverage percentage out of the json summary if present,
  # else fall back to parsing the --print-summary "lines:" line.
  local pct=""
  if command -v python3 >/dev/null 2>&1 && [ -f coverage.json ]; then
    pct="$(python3 - <<'PY' 2>/dev/null || true
import json,sys
try:
    d=json.load(open("coverage.json"))
    print(d.get("line_percent", d.get("line_covered_percent","")))
except Exception:
    pass
PY
)"
  fi
  if [ -z "${pct}" ]; then
    pct="$(grep -iE '^lines:' coverage_summary.txt | head -1 | grep -oE '[0-9]+\.?[0-9]*%' | head -1)"
  fi
  if [ -z "${pct}" ]; then
    echo "ERROR: could not parse line coverage from gcovr output."
    return 1
  fi
  gate_against_threshold "${pct}"
}

run_llvm_path() {
  # Clang/icpx/SYCL path: merge all per-test .profraw, then llvm-cov report.
  # On SYCL, host + device coverage merge into one .profdata (decision 2 / the
  # SYCL native device-coverage path shares the llvm-cov toolchain).
  local profdata="coverage.profdata"

  local profraws
  profraws="$(find . -name '*.profraw' 2>/dev/null)"
  local n_profraw
  n_profraw="$(printf '%s\n' "${profraws}" | grep -c . || true)"
  echo "Found ${n_profraw} .profraw coverage data file(s)."
  if [ "${n_profraw}" -eq 0 ]; then
    echo "ERROR: no .profraw files. For the LLVM path each test must run with"
    echo "       LLVM_PROFILE_FILE set so it emits a .profraw (see Testing.md)."
    return 1
  fi

  local PROFDATA_BIN COV_BIN
  PROFDATA_BIN="$(command -v llvm-profdata || true)"
  COV_BIN="$(command -v llvm-cov || true)"
  if [ -z "${PROFDATA_BIN}" ] || [ -z "${COV_BIN}" ]; then
    echo "ERROR: llvm-profdata / llvm-cov not found on PATH (LLVM coverage path)."
    echo "       On Aurora they live under the oneAPI compiler bin dir."
    return 1
  fi

  echo "Merging .profraw -> ${profdata} ..."
  # shellcheck disable=SC2086
  "${PROFDATA_BIN}" merge -sparse ${profraws} -o "${profdata}" || {
    echo "ERROR: llvm-profdata merge failed."; return 1; }

  # Collect every instrumented test executable as an -object for llvm-cov.
  local objs=()
  while IFS= read -r exe; do
    objs+=("-object" "${exe}")
  done < <(find . -path '*/test/*' -type f -name '*.exe' 2>/dev/null)
  if [ "${#objs[@]}" -eq 0 ]; then
    # fall back: any executable test binary
    while IFS= read -r exe; do
      objs+=("-object" "${exe}")
    done < <(find . -type f -name 'test*.exe' 2>/dev/null)
  fi

  echo "Running aggregate llvm-cov report (excluding external/ + test/)..."
  "${COV_BIN}" report "${objs[@]}" \
    -instr-profile="${profdata}" \
    -ignore-filename-regex="${EXCLUDE_RE}" 2>&1 | tee coverage_summary.txt

  # llvm-cov export -> lcov for Codecov/CDash parity.
  "${COV_BIN}" export "${objs[@]}" \
    -instr-profile="${profdata}" \
    -ignore-filename-regex="${EXCLUDE_RE}" \
    -format=lcov > coverage.info 2>/dev/null || \
    echo "WARN: llvm-cov lcov export failed (report still produced)."

  # The TOTAL line in `llvm-cov report` ends with the line-coverage percentage.
  local pct
  pct="$(grep -iE '^TOTAL' coverage_summary.txt | grep -oE '[0-9]+\.?[0-9]*%' | tail -1)"
  if [ -z "${pct}" ]; then
    echo "ERROR: could not parse TOTAL line coverage from llvm-cov output."
    return 1
  fi
  gate_against_threshold "${pct}"
}

# ---- SYCL device-coverage note (decision 1 / decision 3) --------------------
# On SYCL the same .profraw carries host + device line/region coverage when the
# icpx in use contains intel/llvm PR #20710 (built with -fprofile-instr-generate
# -fcoverage-mapping -fno-sycl-use-footer). The llvm path above already merges
# it. If the toolchain lacks #20710, the native_cpu fallback build provides a
# portable reachability cross-check (see doc/devGuide/Testing.md). Device numbers
# are reported best-effort with NO hard gate for v1 - only host/CPU is gated.
# -----------------------------------------------------------------------------

case "${TOOLCHAIN}" in
  gcov) run_gcov_path ;;
  llvm) run_llvm_path ;;
  *)
    echo "ERROR: unknown OMEGA_COVERAGE_TOOLCHAIN='${TOOLCHAIN}' (expected gcov|llvm)."
    exit 2
    ;;
esac
RC=$?

echo ""
if [ "${RC}" -eq 0 ]; then
  echo "COVERAGE_REPORT: PASS  (lcov: ${BUILD_DIR}/coverage.info)"
else
  echo "COVERAGE_REPORT: FAIL  (host coverage below ${THRESHOLD}% or data missing)"
fi
exit "${RC}"
