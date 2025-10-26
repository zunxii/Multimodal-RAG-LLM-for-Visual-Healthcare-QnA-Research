from transformers import Blip2Processor, Blip2Model
from PIL import Image
import torch
import numpy as np

class BLIPImageEncoder:
    """
    Extracts image embeddings using BLIP-2 model.
    """

    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2Model.from_pretrained(model_name).to(self.device)

    def encode(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype(np.float32)
