# PRD — Cloud Removal from Satellite Imagery (MLOps Sprint)

## 1. Summary
Build an end-to-end MLOps pipeline that trains a GAN-based model (U-Net generator +
PatchGAN discriminator) to remove clouds from optical satellite images, tracks
experiments, registers the best model, and serves it via an API + demo UI.
Timebox: **~1.5 days, 2 engineers, vibecoded with an AI coding agent.**

## 2. Problem Statement
Optical satellite imagery is frequently obstructed by clouds, degrading downstream
tasks (land-cover classification, change detection, disaster response). Cloud
removal reconstructs a cloud-free version of the scene from a cloudy input image.

## 3. Goals (this sprint)
- G1: Train a working image-to-image cloud-removal model on a *paired* dataset.
- G2: Track every training run (params, losses, sample outputs) reproducibly.
- G3: Register the best checkpoint in a model registry with a promotion workflow
  (None → Staging → Production).
- G4: Serve the production model behind an API.
- G5: Provide a public, clickable demo.
- G6: Document architecture + reproduction steps well enough that a third
  person could re-run this in under 30 minutes.

## 4. Non-Goals (explicitly out of scope this sprint)
- SAR+optical fusion / multi-temporal modeling (SEN12MS-CR-TS is noted as future work only).
- Hyperparameter sweeps / NAS.
- Model monitoring, drift detection, A/B testing, canary rollout.
- Multi-GPU / distributed training.
- Full DVC pipeline (a lightweight dataset manifest/hash substitutes).
- Auth, rate limiting, or production-grade scaling of the API.

## 5. Users & Use Case
Primary "user" for this sprint is a reviewer/evaluator assessing MLOps maturity —
so the pipeline must be **legible and demoable**, not just functionally correct.
Secondary real use case: a researcher uploads a cloudy Sentinel-2/RICE-style
image and gets back a cloud-free reconstruction.

## 6. Datasets
| Dataset | Role | Notes |
|---|---|---|
| RICE (RICE1/RICE2) | **Primary** — train/val/test | Small, already paired (cloudy/clean optical), fastest to load |
| HF `littlebeen/DE` | Secondary — extra eval / generalization check | Only if time remains after core loop works |
| SEN12MS-CR-TS | **Not used this sprint** | Documented as future work: multi-temporal SAR+optical, too heavy for the timebox |

## 7. Model
- Generator: U-Net (encoder-decoder with skip connections).
- Discriminator: PatchGAN (classifies overlapping patches as real/fake).
- Approach: fork a working pix2pix implementation and adapt the dataloader —
  do **not** author the GAN training loop from scratch.
- Losses: adversarial loss + L1 reconstruction loss (standard pix2pix recipe).
- Resolution: train small (256×256) to fit the time budget.

## 8. Functional Requirements
| ID | Requirement | Priority |
|---|---|---|
| FR1 | Dataset loader for RICE with train/val/test split | Must |
| FR2 | Training script logging to MLflow (params, per-epoch losses, sample image grids) | Must |
| FR3 | Evaluation script computing PSNR/SSIM on held-out set | Must |
| FR4 | Model registered in MLflow Model Registry with stage transitions | Must |
| FR5 | FastAPI `/predict` endpoint that loads the `Production`-staged model | Must |
| FR6 | Gradio demo (upload cloudy image → see before/after) | Must |
| FR7 | Dockerized inference service | Must |
| FR8 | CI workflow: lint + smoke test (1 train step + 1 inference call) on push | Must |
| FR9 | Deployed public demo (Hugging Face Spaces) | Should |
| FR10 | Second eval on HF `DE` dataset for generalization | Could |
| FR11 | Basic `/health` endpoint + structured logging | Should |

## 9. Success Metrics (Definition of Done)
- [ ] Training run completes and produces visibly de-clouded outputs vs. raw cloudy input.
- [ ] PSNR/SSIM reported on RICE test split.
- [ ] MLflow shows ≥2 runs with logged metrics + artifacts.
- [ ] A model version exists in the registry with a `Production` alias/stage.
- [ ] `POST /predict` returns a cloud-removed image given a cloudy image.
- [ ] Gradio demo is reachable via a public URL.
- [ ] CI pipeline is green on `main`.
- [ ] README allows a stranger to reproduce training + serve locally.

**"60%+ done" bar:** FR1–FR8 complete = core deliverable met, even if FR9–FR11
(stretch) are dropped. Cut items must be listed explicitly in README under
"Not attempted due to timebox," not silently omitted.

## 10. Team & Ownership
- **Person A (Modeling):** FR1, FR2 (training side), FR3, FR10.
- **Person B (MLOps/Infra):** MLflow server, FR4, FR5, FR6, FR7, FR8, FR9, FR11.
- Both: integration pass + docs at the end.

## 11. Risks
| Risk | Mitigation |
|---|---|
| GPU availability/time runs out mid-training | Train small resolution, checkpoint every N epochs, keep epoch count low and configurable |
| Dataset download/licensing friction | RICE chosen specifically because it's small and pre-paired; have a fallback of a tiny synthetic cloud-mask overlay if download fails |
| Agent-generated code drifts from repo conventions | AGENTS.md + ARCHITECTURE.md given to the coding agent as persistent context every session |
| Scope creep into SEN12MS-CR-TS | Explicitly out of scope in this PRD — reference this doc if tempted |
