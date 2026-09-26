import gradio as gr
from PIL import Image
import os
import sys

# Ensure 'src' can be imported if running directly from the 'serving' folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.inference import CloudRemovalModel

print("Initializing Cloud Removal App...")
try:
    # Initialize the model using the local best.pt file
    model_path = os.path.join(os.path.dirname(__file__), "best.pt")
    cloud_remover = CloudRemovalModel(model_path=model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure 'best.pt' is placed inside the 'serving/' folder.")
    sys.exit(1)

def process_image(input_img: Image.Image):
    if input_img is None:
        return None
    # Ensure RGB
    input_img = input_img.convert("RGB")
    # Run inference
    clean_img = cloud_remover.predict(input_img)
    return clean_img

# Create the Gradio interface
with gr.Blocks(title="Satellite Cloud Removal", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ☁️ Satellite Cloud Removal GAN")
    gr.Markdown("Upload a cloudy satellite image (RICE dataset format), and the pix2pix GAN will remove the clouds!")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input: Cloudy Image")
            process_btn = gr.Button("Remove Clouds", variant="primary")
        with gr.Column():
            output_image = gr.Image(type="pil", label="Output: Cloud-Free Image")
            
    process_btn.click(fn=process_image, inputs=input_image, outputs=output_image)

if __name__ == "__main__":
    print("Launching Gradio App...")
    # Launch the demo locally
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
