"""Embedding encoders for images and text"""

from .clip_image_encoder import CLIPImageEncoder
from .clip_text_encoder import CLIPTextEncoder

__all__ = ['CLIPImageEncoder', 'CLIPTextEncoder']