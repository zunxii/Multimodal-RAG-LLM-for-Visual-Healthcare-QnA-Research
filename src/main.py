import json
import numpy as np
from retrieval.build_db import build_temp_db_from_caption_cloud
from retrieval.search_db import query_temp_db
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

clip_img = CLIPImageEncoder() # choosing Clip for now 
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

 # Step 4 - Temporary DB creatiion of fused embeddings

index, phi_matrix, metadata = build_temp_db_from_caption_cloud(
    image_vector=image_vector,
    caption_cloud=caption_cloud,
    text_encoder=clip_txt,
    fusion=fusion,
    device=None
)


 # Step 5 - Query Clinicalization, Image Embedding and fusion
from clinicalization.clinicalize_query import clinicalize_query_llm

q_clinical = clinicalize_query_llm(q_user)

neighbors = query_temp_db(
    index=index,
    phi_matrix=phi_matrix,
    metadata=metadata,
    image_vector=image_vector,
    q_clinical=q_clinical,
    text_encoder=clip_txt,
    fusion=fusion,
    top_k=5
)

print("Top neighbors:", neighbors)

