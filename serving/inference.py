import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Adjust the import based on your project structure
# If running from root, 'src.models.generator' should work
try:
    from src.models.generator import UNetGenerator
except ImportError:
    # Fallback if someone tries to run it directly from inside 'serving/'
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.models.generator import UNetGenerator

class CloudRemovalModel:
    def __init__(self, model_path: str = "serving/best.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")
        
        self.generator = UNetGenerator(in_channels=3, out_channels=3)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator"])
        
        self.generator.to(self.device)
        self.generator.eval()
        
        # Data contract transforms (256x256, [-1, 1])
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    @torch.no_grad()
    def predict(self, img: Image.Image) -> Image.Image:
        # Preprocess
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Forward pass
        output_tensor = self.generator(input_tensor).squeeze(0).cpu()
        
        # Denormalize [-1, 1] -> [0, 1]
        output_tensor = (output_tensor + 1.0) / 2.0
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
        
        # Convert back to PIL Image
        output_image = T.ToPILImage()(output_tensor)
        
        return output_image
