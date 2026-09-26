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
unzip -q rice-remote-sensing-images-for-cloud-removal.zip -d dataset/
```
Ensure the folder structure looks like `dataset/RICE1/cloud/`, `dataset/RICE1/label/`, etc.
Run the preprocessing script to slice the images into 256x256 patches:
```bash
python -m src.data.preprocess --config configs/dataset.yaml
```

## 🧠 Training

Training is fully tracked using **MLflow** hosted on DAGsHub. The training script logs hyperparameters, per-epoch generator/discriminator losses, L1 pixel loss, and generated sample image grids.

To train the model from scratch (best run on a GPU like a Colab Tesla T4):
```bash
python src/training/train.py --config configs/train_rice.yaml
```
The best model weights are automatically saved to `models/baseline/checkpoints/best.pt`.

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
* **Model Layer:** pix2pix fork adapted for 3-band RGB imagery.
* **Tracking Layer:** Remote MLflow integration via DAGsHub logging metrics and artifacts.
* **Serving Layer:** Standalone inference pipeline wrapping the generator into an interactive Gradio UI.

## ✂️ Not Attempted Due to Timebox

Given the strict 1.5-day constraint, we intentionally cut the following scope (as documented in our PRD & Project Plan) to guarantee we shipped a working end-to-end pipeline:

### Scope / Feature Cuts
| Item | Reason |
|---|---|
| **SEN12MS-CR-TS integration** | Multispectral/SAR data was too heavy and complex for the timebox. Sticking to RGB (RICE) ensured rapid iteration. |
| **Full DVC pipeline** | Replaced with a lightweight Kaggle API approach due to setup friction across environments. |
| **HF Spaces deployment** | Left as a stretch goal; prioritized getting the local Gradio demo running reliably instead. |
| **HF `DE` dataset eval** | Cross-dataset generalization check was dropped to focus on optimizing the primary RICE validation metrics. |
| **Monitoring/drift detection** | Out of scope per PRD. Focus was purely on the initial training & deployment phases. |

### Tool / Infrastructure Cuts
* **scikit-image:** Skipped writing the `eval.py` script to calculate PSNR/SSIM, relying on the visual output and L1 loss instead.
* **SQLite:** Bypassed completely because we used DAGsHub for our MLflow server instead of setting up a local database.
* **MANIFEST.json (custom):** Skipped writing a custom JSON file to hash the data, and instead used the Kaggle API directly.
* **FastAPI:** Didn't build a standalone API. Wired our Gradio UI directly to the PyTorch model for simplicity.
* **Docker:** Ran the code directly in Colab and local environments instead of building a container.
* **GitHub Actions:** Did not set up automated CI pipelines to run on code push.
* **pytest:** Didn't write unit tests for the dataset shapes or API responses.
* **ruff:** Didn't run an automated code linter.
* **MLflow Model Registry:** Tracked the experiment in MLflow, but skipped pulling the "Production" model via the registry API (`load_production_model()`). Instead, downloaded `best.pt` directly.
* **Optuna:** Hyperparameter tuning was skipped in favor of using static, proven pix2pix defaults (`configs/train_rice.yaml`).
