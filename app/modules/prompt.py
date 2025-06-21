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
from concurrent.futures import ThreadPoolExecutor, as_completed

from .image_transformer import ImageTransformer
from .prompt_image import PromptImage
from .model_instances import get_all_image_transformer_instances
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

    def get_new_images(self, image_transformer: ImageTransformer, n=3, save=True, prompt_guidance=None, image_guidance=None):
        if self.is_enhanced:
            prompt = self.enhanced_prompt
        else:
            prompt = self.prompt

        output_images : List[Image.Image] = []

        # guidance_scales = self.__get_random_values(n, 1, 15)
        # image_guidance_scales = self.__get_random_values(n, 1, 5)

        if prompt_guidance is not None:
            if prompt_guidance == 0:  # Low
                guidance_scales = self.__get_random_values(n, 1, 3)
            elif prompt_guidance == 1:  # Medium
                guidance_scales = self.__get_random_values(n, 3, 7)
            elif prompt_guidance == 2:  # High
                guidance_scales = self.__get_random_values(n, 7, 15)
        elif prompt_guidance is None:
            guidance_scales = self.__get_random_values(n, 1, 15)


        if image_guidance is not None:
            if image_guidance == 0:
                image_guidance_scales = self.__get_random_values(n, 1, 1.5)
            elif image_guidance == 1:
                image_guidance_scales = self.__get_random_values(n, 1.5, 2.5)
            elif image_guidance == 2:
                image_guidance_scales = self.__get_random_values(n, 2.5, 5)
        elif image_guidance is None:
            image_guidance_scales = self.__get_random_values(n, 1, 5)

        transformers = get_all_image_transformer_instances()
        
        def generate_single_image(transformer, input_image_copy, prompt_text, pg, ig):
            image = transformer.transform(input_image_copy, prompt_text, guidance_scale=pg, image_guidance_scale=ig)
            return PromptImage(image, pg, ig, input_prompt=None, output_prompt=self.id, save=save)

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i, (pg, ig) in enumerate(zip(guidance_scales, image_guidance_scales)):
                transformer = transformers[i % 3]  # Cycle through the 3 transformers
                input_image_copy = self.get_image().copy()
                futures.append(executor.submit(generate_single_image, transformer, input_image_copy, prompt, pg, ig))
            
            for future in as_completed(futures):
                pimage = future.result()
                output_images.append(pimage)

        self.images_out = output_images

        return output_images