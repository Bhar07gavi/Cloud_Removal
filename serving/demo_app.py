"""Gradio Web Demo for Satellite Cloud Removal using the shared inference pipeline."""

import os
import sys
from pathlib import Path
from PIL import Image
import gradio as gr

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from serving.inference import CloudRemovalModel
from serving.settings import settings

print("Initializing Cloud Removal App...")
cloud_remover = None
init_error = None

try:
    # Attempt to load model using local checkpoint if present, or registry
    local_pt = Path(__file__).parent / "best.pt"
    baseline_pt = REPO_ROOT / "models" / "baseline" / "checkpoints" / "best.pt"

    if local_pt.exists():
        cloud_remover = CloudRemovalModel(generator_or_path=str(local_pt))
    elif baseline_pt.exists():
        cloud_remover = CloudRemovalModel(generator_or_path=str(baseline_pt))
    else:
        # Load from registry via settings
        cloud_remover = CloudRemovalModel()
    print(f"Loaded model successfully (version: {cloud_remover.version})")
except Exception as e:
    init_error = str(e)
    print(f"Notice: Model could not be loaded at startup: {e}")
    print("App will launch in status mode. Provide a checkpoint or register a model to enable inference.")


def process_image(input_img: Image.Image, overlap: int = 32):
    if input_img is None:
        return None, "Please upload an image."
    if cloud_remover is None:
        return None, f"Model not loaded: {init_error}"

    try:
        input_img = input_img.convert("RGB")
        # Run tiled inference through the shared pipeline
        clean_img = cloud_remover.predict(input_img, overlap=overlap)
        w, h = input_img.size
        status_msg = f"Successfully processed {w}x{h} image using model version '{cloud_remover.version}' (overlap={overlap}px)."
        return clean_img, status_msg
    except Exception as e:
        return None, f"Error during inference: {e}"


# Build Gradio UI
with gr.Blocks(title="Satellite Cloud Removal", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ☁️ Satellite Cloud Removal GAN")
    gr.Markdown(
        "Upload a cloudy optical satellite image. The Pix2Pix GAN removes cloud cover "
        "using tiled sliding-window inference with edge blending."
    )

    if cloud_remover is None:
        gr.Warning(f"Model not loaded: {init_error}")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input: Cloudy Satellite Image")
            overlap_slider = gr.Slider(
                minimum=0,
                maximum=64,
                value=32,
                step=8,
                label="Tile Overlap (pixels)",
                info="Higher values produce smoother blends across tile seams",
            )
            process_btn = gr.Button("Remove Clouds", variant="primary")
        with gr.Column():
            output_image = gr.Image(type="pil", label="Output: Cloud-Free Image")
            status_box = gr.Textbox(label="Status / Metadata", interactive=False)

    process_btn.click(
        fn=process_image,
        inputs=[input_image, overlap_slider],
        outputs=[output_image, status_box],
    )

if __name__ == "__main__":
    print("Launching Gradio App...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
