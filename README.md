# ☁️ Satellite Cloud Removal GAN

This repository contains an end-to-end Machine Learning pipeline for removing clouds from optical remote sensing imagery. It implements a **pix2pix Generative Adversarial Network (GAN)** (U-Net Generator + PatchGAN Discriminator) trained on the **RICE dataset**.

This project was built during a 1.5-day timeboxed sprint, focusing on a complete MLOps lifecycle including training, MLflow tracking (via DAGsHub), and local Gradio serving.

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Bhar07gavi/Cloud_Removal.git
cd Cloud_Removal
pip install -r requirements.txt
pip install gradio  # For the web UI
```

### 2. Dataset Setup
We use the RICE dataset. You can download it quickly using the Kaggle API:
```bash
kaggle datasets download -d shubhank/rice-remote-sensing-images-for-cloud-removal
unzip -q rice-remote-sensing-images-for-cloud-removal.zip -d dataset/RICE
```
Ensure the folder structure looks like `dataset/RICE/RICE1` and `dataset/RICE/RICE2`.
Run the preprocessing script to slice the images into 256x256 patches:
```bash
python -m src.data.preprocess --config configs/dataset_rice.yaml
```

## 🧠 Training

Training is fully tracked using **MLflow** hosted on DAGsHub. The training script logs hyperparameters, per-epoch generator/discriminator losses, L1 pixel loss, and generated sample image grids.

To train the model from scratch (best run on a GPU like a Colab Tesla T4):
```bash
python -m src.training.train --config configs/train_rice.yaml
```
The best model weights are automatically saved to `models/baseline/checkpoints/best.pt`.

## 🧪 Testing
The repository uses pytest for unit testing core functionalities such as dataset loading and tensor value transformations. You can run the test suite with:
```bash
pytest -q
```

## 📊 Evaluation
You can evaluate the trained model on the test split using `eval.py`. It computes the Mean PSNR and Mean SSIM for both the Model's output and a "Baseline (cloudy input)" (which simulates doing nothing and outputting the cloudy image directly), and presents a summary table with the Delta:
```bash
python -m src.models.eval --config configs/train_rice.yaml
```

## 🌐 Running the Web Demo (Serving)

We provide a lightweight web interface built with **Gradio** to interactively test the trained model.

1. Ensure your trained model weights are saved at `serving/best.pt`.
2. Run the application:
```bash
python serving/demo_app.py
```
3. Open your browser to the local URL (usually `http://127.0.0.1:7860`).
4. Upload a cloudy satellite image to see the GAN reconstruct the landscape underneath!

## 📦 What We Built
* **Data Layer:** PyTorch datasets with automated 256x256 tiling and normalization `[-1, 1]`.
* **DVC Data Versioning:** Remote dataset synchronization and version tracking via DagsHub DVC remote.
* **Model Layer:** pix2pix fork adapted for 3-band RGB imagery.
* **Tracking Layer:** Remote MLflow integration via DAGsHub logging metrics, traceability parameters, and artifacts.
* **Optuna Tuning (`src/training/tune.py`):** Automated hyperparameter optimization maximizing validation SSIM across trials.
* **PSNR/SSIM Evaluation with Baseline:** Benchmark evaluation on test split comparing model predictions against cloudy input baseline.
* **Pytest Suite:** Test suite covering data transformations, dataset loading, device resolution, checkpoint selection, and MLflow artifact logging.
* **Serving Layer:** Standalone inference pipeline wrapping the generator into an interactive Gradio UI.

## ✂️ Not Attempted Due to Timebox

Given the strict constraint, we intentionally cut the following scope to guarantee we shipped a working end-to-end pipeline:

### Scope / Feature Cuts
| Item | Reason |
|---|---|
| **SEN12MS-CR-TS integration** | Multispectral/SAR data was too heavy and complex for the timebox. Sticking to RGB (RICE) ensured rapid iteration. |
| **HF Spaces deployment** | Left as a stretch goal; prioritized getting the local Gradio demo running reliably instead. |
| **HF `DE` dataset eval** | Cross-dataset generalization check was dropped to focus on optimizing the primary RICE validation metrics. |
| **Monitoring/drift detection** | Out of scope per PRD. Focus was purely on the initial training & deployment phases. |

### Tool / Infrastructure Cuts
* **Full DVC pipeline (dvc.yaml):** Versioned data with DVC, but did not construct a multi-stage `dvc.yaml` workflow pipeline.
* **FastAPI:** Didn't build a standalone API. Wired our Gradio UI directly to the PyTorch model for simplicity.
* **Docker:** Ran the code directly in Colab and local environments instead of building a container.
* **GitHub Actions:** Did not set up automated CI pipelines to run on code push.
* **ruff:** Didn't run an automated code linter.
* **MLflow Model Registry:** Tracked the experiment in MLflow, but skipped pulling the "Production" model via the registry API (`load_production_model()`). Instead, downloaded `best.pt` directly.
