#!/usr/bin/env bash
#
# verify_omega_508_arch.sh -- verification for Omega issue #508
#   "Set OMEGA_ARCH when Omega is built as part of E3SM"
#
# Creates a real coupled Omega CIME case, builds it, and asserts that the
# E3SM-mode CMake configure:
#   1. printed a NON-EMPTY "OMEGA_ARCH = <arch>" and it matches what the
#      machine/compiler pair implies (CUDA / HIP / SYCL / OPENMP / SERIAL);
#   2. picked up the per-machine Kokkos architecture from KOKKOS_OPTIONS;
#   3. did not trip the "GPU build but no Kokkos_ARCH_*" fatal guard.
#
# This is a discriminating test: before the #508 fix, OMEGA_ARCH was set to ""
# in setup_e3sm_build, so check 1 fails on every machine.
#
# Usage:
#   ./components/omega/test/verification/verify_omega_508_arch.sh --machine frontier --compiler craycray-mphipcc
#   ./components/omega/test/verification/verify_omega_508_arch.sh --machine pm-gpu   --compiler gnugpu --project mXXXX
#
# CIME needs python >= 3.8. It is loaded with a module command, not a hard-coded
# path. Change it for your system by editing PYTHON_MODULE_DEFAULT below, or:
#   --python-module "module load cray-python"
#   OMEGA_VERIFY_PYTHON_MODULE="module load python/3.11" verify_omega_508_arch.sh ...
#   --python-module ""    (skip modules; use the python3 already on PATH)
#
set -euo pipefail

MACH=""; COMPILER=""; PROJECT=""
RES="TL319_EC30to60E2r2"
COMPSET=""                       # auto-detected from config_compsets.xml
DRIVER="mct"
PECOUNT="S"
CASEROOT_BASE="${SCRATCH:-$HOME}/omega-verify"
EXPECT_ARCH=""                   # override the auto-derived expectation
PYTHON_MODULE=""                 # set below from PYTHON_MODULE_DEFAULT/env/--python-module
KEEP=0

usage() { sed -n "2,31p" "$0"; exit 0; }

while [ $# -gt 0 ]; do
  case "$1" in
    --machine)   MACH="$2"; shift 2 ;;
    --compiler)  COMPILER="$2"; shift 2 ;;
    --project)   PROJECT="$2"; shift 2 ;;
    --res)       RES="$2"; shift 2 ;;
    --compset)   COMPSET="$2"; shift 2 ;;
    --driver)    DRIVER="$2"; shift 2 ;;
    --pecount)   PECOUNT="$2"; shift 2 ;;
    --caseroot)  CASEROOT_BASE="$2"; shift 2 ;;
    --expect)    EXPECT_ARCH="$2"; shift 2 ;;
    --python-module) PYTHON_MODULE="$2"; PYTHON_MODULE_SET=1; shift 2 ;;
    --keep)      KEEP=1; shift ;;
    -h|--help)   usage ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ -n "$MACH" ]     || { echo "ERROR: --machine is required" >&2; exit 2; }
[ -n "$COMPILER" ] || { echo "ERROR: --compiler is required" >&2; exit 2; }

# repo root: this script lives at components/omega/test/verification/
SRCROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$SRCROOT"

say()  { printf '\n== %s\n' "$*"; }
pass() { printf '  PASS  %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; FAILED=1; }
FAILED=0

# ------------------------------------------------------- python for CIME ----
# CIME needs python >= 3.8 and many systems default to something older, so the
# right python is loaded with a module command rather than a hard-coded path
# (paths differ per system; module names do too).
#
# EDIT THIS for your system, or override it without editing:
#   --python-module "module load cray-python"
#   OMEGA_VERIFY_PYTHON_MODULE="module load python/3.11" ./this-script ...
# Set it to the empty string to skip module handling entirely and use whatever
# python3 is already on PATH.
PYTHON_MODULE_DEFAULT="module load python"

# precedence: --python-module  >  $OMEGA_VERIFY_PYTHON_MODULE  >  the default
# above. Note ${VAR-...} not ${VAR:-...}, so an explicitly EMPTY env var means
# "skip module handling", rather than falling back to the default.
if [ "${PYTHON_MODULE_SET:-0}" != "1" ]; then
  PYTHON_MODULE="${OMEGA_VERIFY_PYTHON_MODULE-$PYTHON_MODULE_DEFAULT}"
fi

python_ok() {
  command -v python3 >/dev/null 2>&1 &&
    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null
}

# `module` is usually only a shell function in login shells; make it available
# here too before trying to use it.
init_modules() {
  if type module >/dev/null 2>&1; then return 0; fi
  for _init in /usr/share/lmod/lmod/init/bash \
               /opt/cray/pe/lmod/lmod/init/bash \
               "${LMOD_PKG:-/nonexistent}/init/bash" \
               /etc/profile.d/lmod.sh \
               /etc/profile.d/modules.sh; do
    if [ -r "$_init" ]; then
      # shellcheck disable=SC1090
      . "$_init"
      break
    fi
  done
  type module >/dev/null 2>&1
}

setup_python() {
  if python_ok; then return 0; fi
  if [ -z "$PYTHON_MODULE" ]; then return 1; fi
  if ! init_modules; then
    echo "ERROR: no 'module' command available to run: $PYTHON_MODULE" >&2
    return 1
  fi
  say "loading python: $PYTHON_MODULE"
  eval "$PYTHON_MODULE" || true
  # bash caches resolved command paths; without this the old python3 sticks
  # even though the module already prepended the new one to PATH.
  hash -r
  python_ok
}

if ! setup_python; then
  cat >&2 <<ERR
ERROR: need python >= 3.8 for CIME, and could not get one.
       Tried: ${PYTHON_MODULE:-<module handling disabled>}
       Fix by editing PYTHON_MODULE_DEFAULT at the top of this script, or:
         --python-module "module load <your python module>"
         OMEGA_VERIFY_PYTHON_MODULE="module load <...>" $0 ...
ERR
  exit 2
fi
say "python: $(python3 --version 2>&1) ($(command -v python3))"

# The mct driver needs its submodule populated.
if [ "$DRIVER" = "mct" ] && [ ! -f externals/mct/mct/Makefile ]; then
  say "externals/mct submodule is empty -- initializing"
  git submodule update --init --depth=1 externals/mct
fi

# Guard: refuse to "verify" a tree that does not contain the fix.
if ! grep -q 'OMEGA_ARCH = ' components/omega/OmegaBuild.cmake; then
  echo "ERROR: components/omega/OmegaBuild.cmake has no 'OMEGA_ARCH = ' status message." >&2
  echo "       Check out the branch carrying the #508 fix first." >&2
  exit 2
fi

if [ -z "$COMPSET" ]; then
  COMPSET=$(grep -o '<alias>COMEGA[^<]*</alias>' components/omega/cime_config/config_compsets.xml \
            | head -1 | sed 's/<[^>]*>//g')
  [ -n "$COMPSET" ] || { echo "ERROR: no COMEGA* compset found" >&2; exit 2; }
fi
say "compset=$COMPSET  res=$RES  mach=$MACH  compiler=$COMPILER  driver=$DRIVER"

# ------------------------------------------------- expected OMEGA_ARCH ------
MACROS="cime_config/machines/cmake_macros"
MACRO_FILES=""
for f in "$MACROS/universal.cmake" "$MACROS/${COMPILER}.cmake" \
         "$MACROS/${MACH}.cmake" "$MACROS/${MACH}_${COMPILER}.cmake"; do
  [ -f "$f" ] && MACRO_FILES="$MACRO_FILES $f"
done
say "machine macro files:$MACRO_FILES"

if [ -z "$EXPECT_ARCH" ]; then
  if   grep -qE 'set *\( *USE_CUDA +"?TRUE' $MACRO_FILES 2>/dev/null; then EXPECT_ARCH=CUDA
  elif grep -qE 'set *\( *USE_HIP +"?TRUE'  $MACRO_FILES 2>/dev/null; then EXPECT_ARCH=HIP
  elif grep -qE 'set *\( *USE_SYCL +"?TRUE' $MACRO_FILES 2>/dev/null; then EXPECT_ARCH=SYCL
  else EXPECT_ARCH=SERIAL_OR_OPENMP
  fi
fi
EXPECT_KARCH=$(grep -hoE 'Kokkos_ARCH_[A-Za-z0-9_]+' $MACRO_FILES 2>/dev/null | sort -u | tr '\n' ' ')
say "expected OMEGA_ARCH: $EXPECT_ARCH"
if [ "$EXPECT_ARCH" = "SERIAL_OR_OPENMP" ]; then
  cat <<'WARN'

  WARNING: this machine/compiler pair is not a GPU build, so this run is only a
  "does not regress the host path" smoke test. It does NOT discriminate: before
  the #508 fix, OMEGA_ARCH also ended up SERIAL on a host machine (the backend
  chain in update_variables falls through to SERIAL), and the compile defines
  are byte-identical. Run this on a GPU machine/compiler -- pm-gpu/gnugpu,
  frontier/craycray-mphipcc, frontier/gnugpu -- for a real test of #508.

WARN
fi
say "expected Kokkos arch from KOKKOS_OPTIONS: ${EXPECT_KARCH:-<none>}"

# ------------------------------------------------------------- the case -----
CASE="$CASEROOT_BASE/c508.${MACH}.${COMPILER}"
mkdir -p "$CASEROOT_BASE"
rm -rf "$CASE"
say "creating case: $CASE"
PROJ_ARG=""; [ -n "$PROJECT" ] && PROJ_ARG="--project $PROJECT"
./cime/scripts/create_newcase --case "$CASE" --res "$RES" --compset "$COMPSET" \
    --driver "$DRIVER" --machine "$MACH" --compiler "$COMPILER" \
    --pecount "$PECOUNT" --handle-preexisting-dirs r $PROJ_ARG \
    > "$CASEROOT_BASE/create.508.log" 2>&1 \
  || { echo "ERROR: create_newcase failed; see $CASEROOT_BASE/create.508.log" >&2; exit 1; }

cd "$CASE"
say "case.setup"
./case.setup > "$CASEROOT_BASE/setup.508.log" 2>&1 \
  || { echo "ERROR: case.setup failed; see $CASEROOT_BASE/setup.508.log" >&2; exit 1; }

EXEROOT=$(./xmlquery -N --value EXEROOT)
if [ -d "$EXEROOT" ] && [ -n "$(ls -A "$EXEROOT" 2>/dev/null)" ]; then
  say "EXEROOT is already populated -- cleaning first (avoids 'env_build HAS CHANGED')"
  ./case.build --clean-all > "$CASEROOT_BASE/clean.508.log" 2>&1 || true
fi
say "case.build  (EXEROOT=$EXEROOT)"
set +e
./case.build > "$CASEROOT_BASE/build.508.log" 2>&1
BUILD_RC=$?
set -e
say "case.build exit code: $BUILD_RC"

# --------------------------------------------------------------- asserts ----
# The Omega CMake STATUS messages land in the whole-model configure log.
BLDLOG=$(ls -t "$EXEROOT"/e3sm.bldlog.* 2>/dev/null | head -1 || true)
[ -n "$BLDLOG" ] || { echo "ERROR: no e3sm.bldlog.* under $EXEROOT" >&2; exit 1; }
say "inspecting $BLDLOG"
CAT=cat; case "$BLDLOG" in *.gz) CAT=zcat ;; esac

ARCH_LINE=$($CAT "$BLDLOG" | grep -a -m1 -- '-- OMEGA_ARCH = ' || true)
KOPT_LINE=$($CAT "$BLDLOG" | grep -a -m1 -- '-- OMEGA_KOKKOS_OPTIONS = ' || true)
echo "  ${ARCH_LINE:-<no OMEGA_ARCH line>}"
echo "  ${KOPT_LINE:-<no OMEGA_KOKKOS_OPTIONS line>}"

say "assertions"
ARCH_VAL=$(printf '%s' "$ARCH_LINE" | sed 's/.*OMEGA_ARCH = //')
if [ -z "$ARCH_VAL" ]; then
  fail "OMEGA_ARCH is empty or absent  <-- this is the #508 bug"
else
  pass "OMEGA_ARCH is non-empty: '$ARCH_VAL'"
  if [ "$EXPECT_ARCH" = "SERIAL_OR_OPENMP" ]; then
    case "$ARCH_VAL" in
      SERIAL|OPENMP) pass "OMEGA_ARCH '$ARCH_VAL' matches a non-GPU machine" ;;
      *) fail "expected SERIAL or OPENMP on this machine, got '$ARCH_VAL'" ;;
    esac
  elif [ "$ARCH_VAL" = "$EXPECT_ARCH" ]; then
    pass "OMEGA_ARCH matches the machine macros ($EXPECT_ARCH)"
  else
    fail "expected OMEGA_ARCH=$EXPECT_ARCH, got '$ARCH_VAL'"
  fi
fi

if [ -n "$EXPECT_KARCH" ]; then
  MISSING=""
  for k in $EXPECT_KARCH; do
    printf '%s' "$KOPT_LINE" | grep -q "$k" || MISSING="$MISSING $k"
  done
  if [ -z "$MISSING" ]; then
    pass "Kokkos arch reached Omega:$EXPECT_KARCH"
  else
    fail "Kokkos arch missing from Omega's KOKKOS_OPTIONS:$MISSING"
  fi
fi

if $CAT "$BLDLOG" | grep -a -q 'requests a GPU build but no Kokkos_ARCH'; then
  fail "the no-Kokkos_ARCH fatal guard fired"
else
  pass "the no-Kokkos_ARCH fatal guard did not fire"
fi

if [ "$BUILD_RC" -eq 0 ]; then
  pass "case.build completed successfully"
else
  fail "case.build exited $BUILD_RC (see $CASEROOT_BASE/build.508.log)"
fi

[ "$KEEP" -eq 1 ] || say "case kept at $CASE (pass --keep to silence this note)"
say "RESULT: $([ $FAILED -eq 0 ] && echo PASS || echo FAIL)"
exit $FAILED
