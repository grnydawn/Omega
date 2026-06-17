# super-board — HPC handoff / catch-up

> Read this first if you are a fresh Claude Code session picking up the
> super-board pipeline on an HPC machine. It captures everything decided so
> far and the exact steps to start `super-board run`. Authored from a macOS
> session that did onboard + lint but **cannot build Omega locally**; the HPC
> session does the actual building.

## TL;DR

- **Goal:** drain the Omega GitHub Project board autonomously (Build → QA → Review → Done).
- **Right now there is exactly one card:** [grnydawn/Omega#1 "Add coverage test in ctest"](https://github.com/grnydawn/Omega/issues/1), sitting in **Backlog**, fully lint-clarified.
- **Your job on HPC:** verify the pre-flight tools are present, promote #1 to **Ready**, then run `super-board run`.

## What's already set up (committed to branch `ykim/omega/coverage`)

| Artifact | Path | Notes |
|---|---|---|
| super-board skill suite | `.claude/skills/super-board/` | the `/super-board` verbs + lane references |
| wave planner + workflow | `.claude/bin/super-board-wave-plan.sh`, `.claude/workflows/super-board-wave.js` | used by the `run` workflow backend |
| gh rate-limit guard | `.claude/bin/super-board-gh-guard.sh` | |
| active config | `.claude/super-board/configs/omega.json` | committed; see below |
| project context | `docs/super-board/PROJECT.md` | Omega build/test/conventions for lane workers |
| readiness gates | `docs/super-board/pre-flight.md` | **the halt-gate checklist — make these green on HPC** |
| this handoff | `docs/super-board/HANDOFF.md` | |

**NOT committed (per design, recreate locally):** `.claude/super-board/active` (the per-machine
pointer to the active config) is gitignored. You must recreate it — see Step 3.

## The board

- Project **#7 "Omega"** under `grnydawn`: https://github.com/users/grnydawn/projects/7
- Columns (full variant + Backlog intake): `Backlog · Ready · Building · QA · Review · Done · Blocked · Skipped`
- `Backlog` = raw intake (linted in place, never auto-built). `run` drains **Review → QA → Ready** only.

## The config (`.claude/super-board/configs/omega.json`)

- `variant: full`, `worker_backend: workflow` (in-session wave loop — needs **dynamic workflows enabled** in `/config`).
- `base_branch: ykim/omega/coverage` — feature branches are cut from and squash-merged back into this branch.
- `human_approves_merge: true` — Reviewer marks PRs ready; **you click merge** (no auto-merge).
- `bot_identity: grnydawn` — the claim mutex (your GitHub login). Don't run two orchestrators against the board at once.
- `repo.path: "."` (portable), `repo.remote: grnydawn/Omega`.
- `notifications: telegram, chat_id 1559951332` — only fires if a Telegram **bot token** is in the HPC env; otherwise it silently no-ops (harmless).

## Issue #1 — settled scope decisions (don't re-litigate)

The body has full detail; the locked decisions are:

1. **Device-side GPU coverage is in scope for v1** — concrete best-available method per backend: CUDA → Nsight Compute `inst_executed` proxy (`-lineinfo`); HIP → `rocprof`/`rocprof-compute` PC-sampling; SYCL → intel/llvm native device coverage (Level Zero, PR #20710). Difficulty is **high** on all three.
2. **Both toolchains by arch** — `gcc`/`gcov`+`lcov` for the GCC/CPU baseline; LLVM `llvm-cov` for Clang-derived targets (HIP `amdclang++`, SYCL `icpx`, clang CPU, CUDA host-via-clang).
3. **Per-backend thresholds** — host/CPU gated at **90%** (tunable); each device backend reports its own best-effort number, **no hard gate for v1**, no single blended number.
4. **Report to both CDash + Codecov, manually** — reuse `ctest_submit`, add `codecov.yml` + a generated coverage helper script. **No GitHub Actions CI job.**
5. **Production/experimental tooling only for v1** — NVBit (CUDA) and the HIP device-PGO RFC are out of v1 (follow-ups).
6. **Exclude `*/external/*` and `*/test/*`** from the total (mam4xx pattern); measure `src/` only.

Three open questions remain (build-time discovery): exact per-version tool flags on the target GPUs, device-coverage runtime overhead, and Debug-vs-Release/embedded-build behavior. These get resolved during implementation on the target machine.

## Steps to start on HPC

1. **Get the branch:**
   ```sh
   cd <your Omega checkout>
   git fetch origin && git checkout ykim/omega/coverage && git pull
   git submodule update --init --recursive externals/ekat externals/scorpio externals/YAKL components/omega/external cime
   ```
2. **Auth:** `gh auth status` — must be `grnydawn` with `project,repo` scopes (`gh auth refresh -s project,read:project,repo` if needed).
3. **Recreate the active pointer (gitignored, per-machine):**
   ```sh
   printf 'omega\n' > .claude/super-board/active
   ```
4. **Green the pre-flight gates:** open `docs/super-board/pre-flight.md` and confirm/install — a CIME-supported machine + ParMETIS, `lcov`/`llvm-cov`/`llvm-profdata`, the GPU compilers/profilers for the backend(s) you're covering (`nvcc`+`ncu` / `hipcc`+`rocprof` / `icpx` w/ PR #20710), test mesh files, and (if uploading) `CODECOV_TOKEN`. Tick the boxes as they pass; each unticked box its ticket needs is a halt gate.
5. **(Unattended runs only) permissions:** add the allowlist from `.claude/skills/super-board/references/run-workflow.md` ("Mid-run permission prompts") to `.claude/settings.json` so waves don't stall. `gh pr merge` is intentionally excluded (human merge).
6. **(Optional) Telegram:** export the bot token so notifications reach chat `1559951332`.
7. **Promote the card:** on project #7, drag **#1 from Backlog → Ready** (only when its pre-flight gates are green).
8. **Run it:**
   ```
   /super-board run            # default model tier (sonnet/opus/session)
   /super-board run --high     # opus floor — strongest models
   ```
   The orchestrator launches wave(s), reconciles, and reports per wave. Stop anytime via `/workflows` (or `super-board stop`); resume by running again (board state is the only state).

## Caveats

- This is **attended-only** by default: with `human_approves_merge: true`, the Reviewer marks PRs ready and you click merge.
- If `/super-board run` reports the workflow backend is unavailable, enable **dynamic workflows** in `/config` (or set `worker_backend: claude-p` in the config to use the legacy headless dispatcher — see `references/run.md`).
- If the HPC Claude doesn't auto-discover `/super-board`, confirm `.claude/skills/super-board/SKILL.md` is present in the checkout (it is, on this branch).
