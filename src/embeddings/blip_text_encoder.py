from transformers import Blip2Processor, Blip2Model
import torch
import numpy as np

class BLIPTextEncoder:
    """
    Extracts text embeddings using BLIP-2 model.
    """

    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2Model.from_pretrained(model_name).to(self.device)

    def encode(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype(np.float32)
