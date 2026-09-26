# Tech Stack — Cloud Removal MLOps Project

## Principle
Every tool choice here optimizes for **time-to-working-demo** over
"ideal production architecture." Anything heavier (Kubernetes, Postgres,
cloud-managed registries, distributed training) is explicitly rejected for
this timebox — see PRD Non-Goals.

## Languages & Core Libraries
| Layer | Choice | Version pin (target) | Why |
|---|---|---|---|
| Language | Python | 3.10+ | Ecosystem fit for ML + FastAPI |
| DL framework | PyTorch | 2.x | pix2pix reference repos are PyTorch-native |
| Base model code | Forked from `junyanz/pytorch-CycleGAN-and-pix2pix` | pinned commit | U-Net generator + PatchGAN discriminator already implemented and battle-tested; adapt, don't rewrite |
| Image ops | `torchvision`, `Pillow`, `numpy` | latest stable | standard |
| Metrics | `scikit-image` (`peak_signal_noise_ratio`, `structural_similarity`) | latest | PSNR/SSIM out of the box |
| Dataset access | `datasets` (HuggingFace) for `littlebeen/DE`; manual download for RICE | latest | RICE has no official HF loader; keep a small custom `Dataset` class |

## MLOps Layer
| Purpose | Tool | Why this over alternatives |
|---|---|---|
| Experiment tracking | **MLflow Tracking** | Bundled with registry, no external account, runs local/self-hosted |
| Model registry | **MLflow Model Registry** | Native stage transitions (None→Staging→Production), first-class "MLOps" artifact |
| Tracking backend store | SQLite (`mlflow.db`) | Zero setup, fine for 1.5-day single-shared-instance use; Postgres would be overkill |
| Artifact store | Local filesystem (`./mlruns`) mounted into container | No cloud bucket setup needed within timebox |
| Dataset versioning | Lightweight manifest (`data/MANIFEST.json` with file hashes + source URL) | Full DVC pipeline is a stretch goal per PRD; manifest gives traceability without the setup cost |

## Serving & Interface
| Purpose | Tool | Why |
|---|---|---|
| Inference API | **FastAPI** | Fast to stand up, async-friendly, auto-generated OpenAPI docs for free |
| Demo UI | **Gradio** | Fastest path to a drag-and-drop image demo; native image component |
| Model loading in API | `mlflow.pytorch.load_model("models:/cloud-removal-gan/Production")` | API always serves whatever is currently promoted, not a hardcoded path |

## Packaging & Environments
| Purpose | Tool | Why |
|---|---|---|
| Containerization | Docker (single image reused for train + serve, different entrypoints) | Reproducibility, one artifact for both jobs |
| Dependency pinning | `requirements.txt` (pip-compiled) | Simplicity over Poetry/conda for a 1.5-day sprint |
| Env config | `.env` + `pydantic-settings` (or plain `os.environ`) | Avoid hardcoded paths/hosts |

## CI/CD
| Purpose | Tool | Why |
|---|---|---|
| CI | GitHub Actions | Native to GitHub, zero setup cost |
| Checks on push | `ruff` (lint) + one smoke test (1 training step + 1 inference call) | Proves pipeline doesn't silently break without needing a full training run in CI |
| Deployment target | Hugging Face Spaces (Gradio SDK) | Free hosting, zero DevOps, directly deploys a Gradio app |

## Testing
| Purpose | Tool | Why |
|---|---|---|
| Unit tests | `pytest` | Standard, agent-friendly |
| Coverage (minimal) | Dataset shape test, model forward-pass test, API response schema test | Enough to prove core paths work, not full coverage |

## Compute
| Purpose | Tool | Why |
|---|---|---|
| Training compute | Google Colab (Pro if available) or Kaggle Notebooks GPU | No provisioning time; free/cheap GPU access within the timebox |
| Local dev | CPU is fine for wiring/testing, GPU only needed for real training runs | Keeps Person B's infra work decoupled from GPU availability |

## Explicitly Rejected (for this sprint only)
- Kubernetes / Kubeflow — far too much setup overhead for 1.5 days.
- Cloud model registries (SageMaker/Vertex Model Registry) — account/IAM setup cost not worth it.
- Prometheus/Grafana monitoring — no traffic volume to justify it yet; noted as future work.
- DVC (full) — manifest file substitutes; revisit if project continues past this sprint.
- Distributed/multi-GPU training — dataset and timebox don't need it.
