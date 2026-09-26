# Project Plan — Task Breakdown (1.5 Days / 2 People)

Reference: see `PRD.md` for requirements (FR#), `ARCHITECTURE.md` for structure,
`AGENTS.md` for how the coding agent should operate.

Estimated total: ~32 working hours. Checkboxes are meant to be updated as you go
— treat this as the source of truth for "what % is actually done."

## Block 0 — Kickoff (Hour 0–1) [Both]
- [ ] Freeze scope: confirm RICE as primary dataset, pix2pix fork as model base
- [ ] Create GitHub repo, add `docs/` (this doc set), set up project board with FR IDs as columns
- [ ] Agree on the frozen interfaces in `ARCHITECTURE.md` §4 (image contract, registry model name)

## Block 1 — Data & Env (Hour 1–5)

**Person A — Data (FR1)**
- [ ] Download RICE dataset, inspect structure
- [ ] Write `data/dataset.py`: `RiceCloudDataset` returning `(cloudy, clean)` pairs
- [ ] Implement train/val/test split (80/10/10, fixed seed)
- [ ] Write `data/MANIFEST.json` (source URL, download date, file hashes)
- [ ] Sanity check: visualize 5 pairs, confirm alignment

**Person B — Infra**
- [ ] `Dockerfile` (base image, deps installed)
- [ ] `requirements.txt`
- [ ] `mlflow/docker-compose.yml`: MLflow server w/ SQLite backend + local artifact store
- [ ] Confirm both people can reach the shared MLflow tracking URI

## Block 2 — Model Bring-up (Hour 5–10)

**Person A — Model (FR2 setup)**
- [ ] Fork/vendor pix2pix reference repo into `model/networks.py`
- [ ] Wire `RiceCloudDataset` into the training script
- [ ] Get **1 epoch running end-to-end** on a tiny subset (smoke test, not real training)
- [ ] Confirm output image shape/range matches the image contract in `ARCHITECTURE.md` §4

**Person B — Infra**
- [ ] Stub `serving/inference.py` against a randomly-initialized model (contract-first, doesn't block on real weights)
- [ ] Start `serving/api.py` skeleton: `/health`, `/predict` (accepts image, returns image)
- [ ] `transforms.py` shared module — both training and serving import this

## Block 3 — Real Training (Hour 10–18)

**Person A (FR2, FR3)**
- [ ] Launch real training run on GPU (Colab/Kaggle) — small resolution (256×256), modest epoch count
- [ ] Confirm MLflow logging works: params, per-epoch G/D loss, sample image grid every N epochs
- [ ] Launch a second run with a tweaked hyperparam (proves ">1 run" tracking story)
- [ ] Write `model/eval.py`: PSNR/SSIM on test split, log as MLflow metrics

**Person B**
- [ ] Finish `inference.py`: `load_production_model()` pulling from MLflow registry
- [ ] Write `tests/test_dataset.py`, `tests/test_model.py` (shape/forward-pass tests)
- [ ] Write `.github/workflows/ci.yml`: lint (ruff) + smoke test job

## Block 4 — Registry, Serving, Demo (Hour 18–27)

**Person A**
- [ ] Pick best run by PSNR/SSIM, register model as `cloud-removal-gan` in MLflow
- [ ] Promote best version: `None → Staging → Production`
- [ ] (Stretch, FR10) Run eval on HF `littlebeen/DE` for a generalization check
- [ ] Generate a before/after sample gallery for the README/demo

**Person B (FR4, FR5, FR6, FR7, FR8)**
- [ ] Point `inference.py` at the real Production model, confirm `/predict` returns a correct image
- [ ] Containerize the serving side, confirm it runs via `docker run image serve`
- [ ] Build `serving/demo_app.py` (Gradio): upload cloudy image → before/after display
- [ ] Confirm CI is green on a real push
- [ ] (FR9, stretch) Deploy `demo_app.py` to Hugging Face Spaces

## Block 5 — Wrap-up & Docs (Hour 27–32) [Both]
- [ ] `README.md`: setup, how to reproduce training, how to run serving, link to demo
- [ ] README section: **"Not attempted due to timebox"** — list every dropped stretch item honestly
- [ ] Screenshot/export: MLflow run comparison + registry stage view
- [ ] Short demo GIF or video of the Gradio app in action
- [ ] Final check against `PRD.md` §9 Definition of Done — tick off what's actually true

## Definition-of-done tracker (mirrors PRD §9)
- [ ] Training run produces visibly de-clouded output
- [ ] PSNR/SSIM reported on RICE test split
- [ ] ≥2 MLflow runs with logged metrics + artifacts
- [ ] Model registered + promoted to `Production` in MLflow registry
- [ ] `POST /predict` works end-to-end
- [ ] Public Gradio demo reachable (or documented as not-deployed if cut)
- [ ] CI green on `main`
- [ ] README reproducible by a stranger in <30 min

## Explicit stretch/cut list (fill in as decisions are made)
| Item | Status | Reason |
|---|---|---|
| SEN12MS-CR-TS integration | Cut (planned) | Too heavy for timebox — see PRD Non-Goals |
| Full DVC pipeline | Cut (planned) | Manifest file substitutes |
| HF Spaces deployment | Stretch | Only if Block 4 finishes early |
| HF `DE` dataset eval | Stretch | Only if Block 4 finishes early |
| Monitoring/drift detection | Cut (planned) | Out of scope per PRD |
