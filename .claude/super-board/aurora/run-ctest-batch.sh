#!/usr/bin/env bash
#
# run-ctest-batch.sh — Aurora login-node test dispatch for super-board lane workers.
#
# WHY: super-board lane agents run on an Aurora *login* node (aurora-uan-*), which
# has internet (for gh / Claude API / cmake FetchContent) but NO GPU and must not
# run the MPI/GPU ctest suite directly. This helper submits the Omega build+ctest
# as a PBS batch job to a *compute* node (Intel PVC, SYCL), waits for it, and
# returns the ctest exit code — so a lane worker can "run the tests" with one call.
#
# CONTRACT:
#   - The build dir must ALREADY be configured by the caller ON THE LOGIN NODE
#     (cmake configure triggers FetchContent → needs internet the compute node
#     may lack). This helper only runs `make` + `ctest` inside the batch job.
#   - Exit code == ctest exit code (0 = all tests passed). Non-zero on build
#     failure, ctest failure, submit failure, or timeout.
#
# USAGE:
#   bash .claude/super-board/aurora/run-ctest-batch.sh <build_dir>
#
# OVERRIDES (env vars):
#   OMEGA_PBS_ACCOUNT   (default: E3SM_Dec)
#   OMEGA_PBS_QUEUE     (default: debug)            # debug = 1-2 nodes, <=1h
#   OMEGA_PBS_WALLTIME  (default: 01:00:00)
#   OMEGA_PBS_SELECT    (default: 1)                # nodes
#   OMEGA_MESH_DIR      (default: /lus/flare/projects/E3SM_Dec/youngsun/data/omega/mesh)
#   OMEGA_BATCH_TIMEOUT (default: 5400)             # seconds to wait (queue+run)
#
set -uo pipefail

BUILD_DIR="${1:?usage: run-ctest-batch.sh <build_dir>}"
BUILD_DIR="$(cd "$BUILD_DIR" 2>/dev/null && pwd)" || { echo "ERR: build dir not found: $1" >&2; exit 2; }

ACCOUNT="${OMEGA_PBS_ACCOUNT:-E3SM_Dec}"
QUEUE="${OMEGA_PBS_QUEUE:-debug}"
WALLTIME="${OMEGA_PBS_WALLTIME:-01:00:00}"
SELECT="${OMEGA_PBS_SELECT:-1}"
MESH_DIR="${OMEGA_MESH_DIR:-/lus/flare/projects/E3SM_Dec/youngsun/data/omega/mesh}"
TIMEOUT="${OMEGA_BATCH_TIMEOUT:-5400}"

# --- preconditions: build dir must be configured on the login node first --------
for f in omega_env.sh omega_build.sh omega_ctest.sh; do
  if [ ! -f "$BUILD_DIR/$f" ]; then
    echo "ERR: $BUILD_DIR/$f missing — run 'cmake ... -B $BUILD_DIR' on the LOGIN node first" >&2
    echo "     (configure needs internet for FetchContent; only make+ctest are batched)" >&2
    exit 2
  fi
done
command -v qsub  >/dev/null || { echo "ERR: qsub not on PATH (not an Aurora login node?)" >&2; exit 2; }
command -v qstat >/dev/null || { echo "ERR: qstat not on PATH" >&2; exit 2; }

JOB="$BUILD_DIR/omega_batch_ctest.pbs"
RESULT="$BUILD_DIR/omega_batch_result.txt"
JOBLOG="$BUILD_DIR/omega_batch_ctest.log"
rm -f "$RESULT" "$JOBLOG"

# --- job script: build + link meshes + ctest, on a compute node -----------------
cat > "$JOB" <<PBS
#!/usr/bin/env bash
#PBS -N omega_ctest
#PBS -A ${ACCOUNT}
#PBS -q ${QUEUE}
#PBS -l select=${SELECT}
#PBS -l walltime=${WALLTIME}
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o ${JOBLOG}
set -uo pipefail
# ALCF proxy so any incidental network in make resolves (compute nodes egress via proxy)
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export ftp_proxy=http://proxy.alcf.anl.gov:3128
cd "${BUILD_DIR}"
source ./omega_env.sh
echo "== build (make) on \$(hostname) =="
./omega_build.sh; BUILD_RC=\$?
echo "== link meshes into build test dir =="
ln -sf ${MESH_DIR}/OmegaMesh.nc       test/OmegaMesh.nc
ln -sf ${MESH_DIR}/OmegaSphereMesh.nc test/OmegaSphereMesh.nc
ln -sf ${MESH_DIR}/OmegaPlanarMesh.nc test/OmegaPlanarMesh.nc
CTEST_RC=127
if [ "\$BUILD_RC" -eq 0 ]; then
  echo "== ctest on \$(hostname) =="
  ./omega_ctest.sh; CTEST_RC=\$?
else
  echo "== build failed (rc=\$BUILD_RC) — skipping ctest =="
fi
{ echo "BUILD_RC=\$BUILD_RC"; echo "CTEST_RC=\$CTEST_RC"; echo "HOST=\$(hostname)"; echo "DONE"; } > "${RESULT}"
PBS

# --- submit ---------------------------------------------------------------------
JOBID="$(qsub "$JOB" 2>&1)" || { echo "ERR: qsub failed: $JOBID" >&2; exit 3; }
JOBID="$(echo "$JOBID" | tr -d '[:space:]')"
echo "submitted PBS job: $JOBID  (queue=$QUEUE account=$ACCOUNT select=$SELECT walltime=$WALLTIME)"
echo "job log: $JOBLOG   result: $RESULT"

# --- poll: result-file DONE marker is primary; qstat 'F' state is the fallback --
start=$SECONDS
while true; do
  if [ -f "$RESULT" ] && grep -q '^DONE' "$RESULT"; then
    break
  fi
  state="$(qstat -x -f "$JOBID" 2>/dev/null | awk -F'= ' '/job_state/{print $2}' | tr -d '[:space:]')"
  if [ "$state" = "F" ]; then
    sleep 8   # let the filesystem flush the result file
    if [ -f "$RESULT" ] && grep -q '^DONE' "$RESULT"; then break; fi
    echo "ERR: job $JOBID finished (state=F) without a DONE marker — see $JOBLOG" >&2
    [ -f "$JOBLOG" ] && tail -40 "$JOBLOG" >&2
    exit 4
  fi
  if [ $((SECONDS - start)) -ge "$TIMEOUT" ]; then
    echo "ERR: timed out after ${TIMEOUT}s waiting for $JOBID (state=$state). Still queued/running?" >&2
    echo "     check: qstat -x -f $JOBID ; cancel: qdel $JOBID" >&2
    exit 5
  fi
  sleep 20
done

# --- report + propagate ctest exit code -----------------------------------------
echo "==== batch result ($RESULT) ===="
cat "$RESULT"
BUILD_RC="$(awk -F= '/^BUILD_RC/{print $2}' "$RESULT")"
CTEST_RC="$(awk -F= '/^CTEST_RC/{print $2}' "$RESULT")"
if [ "${BUILD_RC:-1}" -ne 0 ]; then
  echo "==== build failed — tail of job log ===="
  [ -f "$JOBLOG" ] && tail -60 "$JOBLOG"
  exit "${BUILD_RC:-1}"
fi
if [ "${CTEST_RC:-1}" -ne 0 ]; then
  echo "==== ctest failures — tail of LastTest.log ===="
  [ -f "$BUILD_DIR/Testing/Temporary/LastTest.log" ] && tail -80 "$BUILD_DIR/Testing/Temporary/LastTest.log"
fi
exit "${CTEST_RC:-1}"
