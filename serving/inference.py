import torch
from src.utils.device import get_device
from src.data.transforms import to_unit_range, unit_to_model_range
import torchvision.transforms as T
from PIL import Image

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
    def __init__(self, model_path: str = "serving/best.pt", device: str = "auto"):
        self.device = get_device(device)
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
            T.Lambda(lambda x: unit_to_model_range(x))
        ])

    @torch.no_grad()
    def predict(self, img: Image.Image) -> Image.Image:
        # Preprocess
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Forward pass
        output_tensor = self.generator(input_tensor).squeeze(0).cpu()
        
        # Denormalize [-1, 1] -> [0, 1]
        output_tensor = to_unit_range(output_tensor)
        
        # Convert back to PIL Image
        output_image = T.ToPILImage()(output_tensor)
        
        return output_image
