# super-board pre-flight — Omega build pipeline

> Re-probed on **ALCF Aurora** (`aurora-uan-0011`, Linux x86_64), 2026-06-17; gates
> re-checked from an interactive **compute node** (`x4309c7s0b0n0`) on **2026-06-23**. Each
> `[ ]` is a **halt gate** for `super-board run` — the loop refuses to start (or a worker
> bounces the card to Blocked) until the items its ticket needs are `[✓]`. Scanned issues:
> **#1 (host + SYCL coverage)**, **#2 (CUDA + HIP device coverage — Backlog, needs
> NVIDIA/AMD machine, not this host)**.

## ⚠️ Top-line verdict

**GREEN — issue #1 can run now.** Aurora is a CIME-supported machine with `icpx`,
`llvm-cov`/`llvm-profdata`, ParMETIS, and meshes in place. The two formerly-open
operational gates are now both satisfied (**2026-06-23**):

1. **Compute node + internet** — running from an interactive PBS compute node that has been
   configured for outbound internet (`gh repo view grnydawn/Omega` succeeds), so the
   autonomous loop (gh / Claude API) can run on the **same** node that builds and runs the
   GPU/MPI ctest suite. (Previously the loop had to halt because compute nodes were
   air-gapped; that is resolved on this node.)
2. **Baseline ctest is fully green** — `100% tests passed, 0 failed out of 40` after
   rebasing `ykim/omega/coverage` onto `origin/develop` (which carries PR #437,
   `omega/fix-debug-tracer-tests`, fixing the 4 `TEND_*` / `TRACERS` failures) and linking
   the meshes into the build `test/` dir. Coverage work starts from a clean baseline.

Issue #2 (CUDA + HIP device coverage) **cannot** run here (no NVIDIA/AMD hardware) and stays
in Backlog for a Perlmutter/MI-series machine.

## 🔑 Credentials the loop will need

- [ ] `CODECOV_TOKEN` — Codecov upload (issue #1, decision 4). Unset. Only needed for the
  **live** Codecov upload; the worker can still land `codecov.yml` + the helper script.
- [~] CDash project + drop — `ctest_submit` already wired in `CTestScript.cmake`; reuse the
  existing Omega CDash project. Confirm at submission time.

## 🛠 Tools the loop will need

Host build toolchain:
- [✓] `gh` CLI authenticated as **`grnydawn`** (scopes: `project, repo, read:org`)
- [✓] `git`, `cmake` 3.31, `gcc`/`g++` 13.4, `mpicc`/`mpicxx` (mpich), `make`, `gcov`
- [✓] `python3` 3.10 (status script / tooling; 3.6 system default too old — symlinked)
- [ ] `ninja` (optional generator) — not checked; Make generator is fine

Host coverage tools (decision 2 — both toolchains):
- [~] `lcov` + `genhtml` — **not installed**; `gcovr 8.6` (installed) produces lcov-format
  output for the gcc/gcov path. Worker: use `gcovr` (or install `lcov` via spack) for the
  `lcov --capture` sweep, or the `llvm-cov` path.
- [✓] `gcovr 8.6` (lcov alternative) — installed under `~/.local`
- [✓] `llvm-cov` + `llvm-profdata` — oneAPI 2025.3 (`/opt/aurora/26.26.0/oneapi/2025.3/bin/compiler/`)
- [✓] `pre-commit 4.6.0` — installed under `~/.local` (format gate; provides its own
  clang-format hook). Standalone `clang-format` binary not on PATH; `conda omega_dev` env
  not built, but `pre-commit run` covers the lint gate.

GPU compilers + device-coverage tools (decision 1 — SYCL leg only on this host):
- [✓] SYCL: `icpx` (oneAPI 2025.3) — present
- [ ] SYCL device coverage **PR #20710** present in this `icpx`? — **UNVERIFIED**
  (build-time discovery). If absent, SYCL device coverage degrades to the `native_cpu`
  fallback + host cross-check (AC notes this fallback).
- [n/a] CUDA `nvcc`+`ncu`, HIP `hipcc`+`rocprof` — not on this host → **issue #2**, Backlog.

## 🌐 Environment

- [✓] **CIME-supported machine** — `aurora` in `config_machines.xml` (`NODENAME_REGEX
  aurora-uan-.*`). Compiler `oneapi-ifx`.
- [✓] **ParMETIS** — `/lus/flare/projects/E3SM_Dec/soft/polaris/aurora/spack/dev_polaris_1.0.0/var/spack/environments/spack_env_oneapi-ifx_mpich/.spack-env/view`
  (has `lib/libparmetis.a`). Worker exports `OMEGA_PARMETIS_ROOT` to it.
- [✓] **GPU hardware (SYCL)** — Aurora compute nodes have 6× Intel PVC; satisfies SYCL
  device coverage **on a compute node**.
- [✓] **Compute-node allocation + internet** — running from an interactive PBS compute node
  (`x4309c7s0b0n0`) that has outbound internet configured, so `super-board` can build, run
  the MPI/GPU ctest suite, **and** drive gh / the Claude API from the same allocation. Grab
  one with `qsub -I` if the session ends.
- [✓] **Test meshes** — canonical copies live at
  `/lus/flare/projects/E3SM_Dec/youngsun/data/omega/mesh/` (`OmegaMesh.nc`,
  `OmegaSphereMesh.nc`, `OmegaPlanarMesh.nc`). ctest runs from the **build** `test/` dir, so
  the three files must be present there — symlink them in after configure:
  `ln -sf /lus/flare/projects/E3SM_Dec/youngsun/data/omega/mesh/Omega*.nc $BUILD/test/`.
  (A copy is also in the source `components/omega/test/`.)
- [~] Required submodules — verify on the target machine with
  `git submodule update --init --recursive externals/ekat externals/scorpio externals/YAKL components/omega/external cime`.

## Recommendation

Issue #1 is correctly scoped (host + SYCL), the baseline ctest is green (40/40), and every
halt gate is now `[✓]` on this Aurora compute node — including outbound internet, so the
loop no longer has to halt for being air-gapped. **Ready to run.** Issue #1 is already in
the **Ready** column; run `super-board run` from this compute-node allocation so the wave
workers build **and** run the ctest suite (including the SYCL device-coverage run)
end-to-end. Issue #2 (CUDA + HIP) stays in Backlog until the pipeline is moved to an
NVIDIA/AMD machine.

The only remaining `[ ]` items are non-blocking for landing the coverage *infrastructure*:
`CODECOV_TOKEN` (needed only for the live Codecov upload, not to land `codecov.yml` + the
helper script) and the build-time check of whether this `icpx` carries SYCL device-coverage
PR #20710 (degrades gracefully to the `native_cpu` fallback if absent).
