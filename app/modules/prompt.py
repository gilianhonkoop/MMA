from transformers import pipeline, AutoProcessor
from huggingface_hub import login
import os
import torch
import dotenv
import json
import re
from PIL import Image
from typing import List
import uuid

from .image_transformer import ImageTransformer
from .prompt_image import PromptImage
import random

class Prompt():
    def __init__(self, prompt: str, depth: int, input_image: PromptImage = None, suggestion_used: str = None, modified_suggestion: bool = False):
        # maybe uuid is better, otherwise db fetches are needed to get the latest id value that can be used
        self.id = str(uuid.uuid4())
        self.prompt : str = prompt
        self.depth : int = depth
        self.used_suggestion : bool = suggestion_used != None
        self.modified_suggestion : bool = modified_suggestion
        self.suggestion_used : str = suggestion_used
        self.is_enhanced : bool = False
        self.enhanced_prompt : str = None
        self.input_image : PromptImage = input_image
        self.images_out : List[PromptImage] = []    

    def set_image(self, image: PromptImage):
        """Set the input image for this prompt."""
        self.input_image = image

    def get_image(self):
        """Get the input image associated with this prompt."""
        return self.input_image.image
    
    def get_final_prompt(self):
        """Get the final prompt text, enhanced if applicable."""
        if self.is_enhanced:
            return self.enhanced_prompt
        return self.prompt

    def enhance_prompt(self, vlm):        
        new_prompt = vlm.enhance_prompt(self)

        self.is_enhanced = True
        self.enhanced_prompt = new_prompt

    def __get_random_values(self, n, min_val, max_val):
        if n > 1:
            spacing = (max_val - min_val) / (n - 1)
            base_values = [min_val + i * spacing for i in range(n)]
            # Add small random variation to make them roughly evenly spaced
            values = [max(min_val, min(max_val, val + random.uniform(-spacing/4, spacing/4))) for val in base_values]
        else:
            values = [random.uniform(min_val, max_val)]

        random.shuffle(values)
        return values

    def get_new_images(self, image_transformer: ImageTransformer, n=3, save=True):
        if self.is_enhanced:
            prompt = self.enhanced_prompt
        else:
            prompt = self.prompt

        output_images : List[Image.Image] = []

        guidance_scales = self.__get_random_values(n, 1, 15)
        image_guidance_scales = self.__get_random_values(n, 1, 5)

        for (pg, ig) in zip(guidance_scales, image_guidance_scales):
            image = image_transformer.transform(self.get_image(), prompt, guidance_scale=pg, image_guidance_scale=ig, save_path = None)
            pimage = PromptImage(image, pg, ig, input_prompt=None, output_prompt=self.id, save=save)
            output_images.append(pimage)

        self.images_out = output_images

        return output_images