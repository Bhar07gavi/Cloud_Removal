# AGENTS.md — Instructions for AI Coding Agents (Antigravity / Claude Code / Cursor / etc.)

This file is persistent context for any coding agent working in this repo.
Read `docs/PRD.md`, `docs/TECH_STACK.md`, and `docs/ARCHITECTURE.md` before
making non-trivial changes. This file tells you *how* to behave, those files
tell you *what* to build.

## Golden rules

1. **Adapt, don't author from scratch.** The generator/discriminator come
   from an existing pix2pix reference implementation
   (`junyanz/pytorch-CycleGAN-and-pix2pix`). Fork/adapt it. Do not write a
   GAN training loop or U-Net architecture from a blank file — this wastes
   the timebox re-deriving well-known code.
2. **Stay inside scope.** If a task references SEN12MS-CR-TS, distributed
   training, Kubernetes, cloud registries, monitoring dashboards, or auth —
   stop and flag it. These are explicit Non-Goals in `docs/PRD.md`. Do not
   quietly implement them because they seem "more correct."
3. **Respect the frozen interfaces** in `docs/ARCHITECTURE.md` §4
   (checkpoint format, image normalization contract, registry model name).
   Changing these breaks the other engineer's parallel work.
4. **Every training run must log to MLflow.** No training code should exist
   that doesn't wrap in `mlflow.start_run()` and log params/metrics/artifacts.
   No exceptions, even for "quick test" runs — use nested runs or a `debug=True`
   tag instead of skipping logging.
5. **The API must always load from the registry, never a hardcoded path.**
   `inference.py` pulls `models:/cloud-removal-gan/Production`. If no
   Production model exists yet, fail loudly with a clear error, don't
   silently fall back to a random path.
6. **Small, runnable increments.** Prefer a script that runs end-to-end on 5
   images in 30 seconds over a "complete" architecture that hasn't been run
   once. Get the full pipeline (data → train → log → register → serve →
   demo) working at toy scale FIRST, then scale up data/epochs.
7. **Timebox discipline.** If a task is taking more than ~2x its estimated
   time in `docs/PROJECT_PLAN.md`, stop, report back with what's blocking,
   and suggest a cut-scope alternative rather than continuing to push through.
8. **Every cut corner gets documented, not hidden.** If you skip something
   (e.g. skip HF `DE` dataset eval, skip full DVC, skip monitoring), add a
   line to the README's "Not attempted due to timebox" section. Do not leave
   it undocumented.

## Code style / conventions
- Python 3.10+, type hints on public functions.
- One shared `transforms.py` for image preprocessing — imported by both
  training and serving code. Never duplicate normalization logic.
- Config via a single `config.yaml` or `.env`, not scattered magic numbers.
- Keep `Dockerfile` single-image, entrypoint decides train vs. serve mode
  (e.g. `docker run image train` vs `docker run image serve`).
- Tests go in `tests/`, named `test_*.py`, runnable via plain `pytest`.

## When starting a new session/task
1. Re-read `docs/PRD.md` §8 (Functional Requirements) and check off what's
   already done in `docs/PROJECT_PLAN.md`.
2. Confirm which FR your task maps to. If it doesn't map to any FR, ask
   before building it.
3. Check `docs/ARCHITECTURE.md` §4 for any interface you're touching.
4. After finishing, update `docs/PROJECT_PLAN.md`'s checklist and note any
   scope cuts in the README.

## When something is ambiguous
Default to the smallest thing that satisfies the PRD's Definition of Done
(§9), not the most complete/general solution. This is a 1.5-day sprint, not
a production system — optimize for "demoable and honest about limitations"
over "theoretically robust."

## Division of labor (do not cross without a note in PR description)
- **Person A** owns: `data/`, `model/` (training + eval).
- **Person B** owns: `mlflow/`, `serving/`, `.github/`, `Dockerfile`, deployment.
- Shared: `transforms.py`-equivalent contracts, `docs/`, `README.md`.
