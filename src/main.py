import json
import numpy as np
from utils.ensure_numpy_2d import ensure_numpy_2d

q_user = 'User query with image details in normal language'
input_image_path = 'data/images/cyanosis_Image_1.jpg'

# Step 1 - Generate dynamic caption cloud
from captioning.dynamic_caption_cloud import DynamicCaptionCloud
caption_cloud_builder = DynamicCaptionCloud()

# path to the generated caption cloud JSON
caption_cloud_path = caption_cloud_builder.build_cloud(input_image_path, n_prompts=3, n_seeds=2)

with open(caption_cloud_path, 'r') as f:
    caption_cloud = json.load(f)
print(f"Generated {len(caption_cloud)} captions in the cloud.")

# Step 2 - make Image Embeddings and text embeddings using encoders (BLIP/CLIP)

from embeddings.blip_image_encoder import BLIPImageEncoder
from embeddings.blip_text_encoder import BLIPTextEncoder
import torch
from embeddings.clip_image_encoder import CLIPImageEncoder
from embeddings.clip_text_encoder import CLIPTextEncoder
from fusion.fusion_mlp import FusionMLP

# choosing Clip for now 

clip_img = CLIPImageEncoder()
clip_txt = CLIPTextEncoder()
fusion = FusionMLP(dv=512, dt=512, d_out=512)

image_vector = clip_img.encode(input_image_path)  # shape (1, 512)
print(f"Image embedding shape: {image_vector.shape}")   


texts = [c["text"] for c in caption_cloud]
text_vector = clip_txt.encode(texts)                    # numpy (N, dt)
text_vector = ensure_numpy_2d(text_vector)

print(f"Text embeddings shape: {text_vector.shape}")

# Convert to torch tensors
image_tensor = torch.from_numpy(image_vector).float()
text_tensor = torch.from_numpy(text_vector).float()

# Step 3 - Fuse image and text embeddings
phi = fusion(image_tensor, text_tensor)
print(f"Fused embeddings shape: {phi.shape}")