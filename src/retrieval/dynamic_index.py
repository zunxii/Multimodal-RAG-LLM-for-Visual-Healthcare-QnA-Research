# its a temporary test file will be changed later

from src.embeddings.clip_text_encoder import CLIPTextEncoder
from src.embeddings.clip_image_encoder import CLIPImageEncoder
from src.fusion.fusion_mlp import FusionMLP
import torch 


clip_img = CLIPImageEncoder()
clip_txt = CLIPTextEncoder()
fusion = FusionMLP(dv=512, dt=512, d_out=512)

image_vector = clip_img.encode("data/images/cyanosis_Image_1.jpg")  
text_vector = clip_txt.encode("The image shows fingers with a bluish discoloration, which suggests abnormal findings. This appearance can be associated with reduced blood flow or oxygenation issues, potentially indicating a condition like Raynaud's phenomenon or acrocyanosis. It's advisable to consult a medical professional for a proper diagnosis.")

image_tensor = torch.from_numpy(image_vector).float()
text_tensor = torch.from_numpy(text_vector).float()

print(image_vector.shape)  
print(text_vector.shape)   
phi = fusion(image_tensor, text_tensor)
print(phi) 