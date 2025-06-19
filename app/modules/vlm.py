import json
import re
from transformers import pipeline, AutoProcessor
from huggingface_hub import login
import os
import sys
import torch
import dotenv

from .prompt import Prompt

class VLM():
    def __init__(self, model="google/gemma-3-4b-it", login_required=True):
        if login_required:
            dotenv.load_dotenv()
            login(os.environ.get("HF_TOKEN"))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model, use_fast=True)

        self.pipe = pipeline(
            "image-text-to-text",
            model=model,
            processor=processor,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            variant="fp16",
            safety_checker=None,
            device=device
        )

    def make_suggestions(self, prompt : Prompt, n_suggestions : int = 3):
        format_example = str([f"suggestion {i+1}" for i in range(n_suggestions)])
        suggestion_prompt = (
            f'Based on the attached image and inspired by the concept: "{prompt.get_final_prompt()}", generate {n_suggestions} completely new and different prompt variations. '
            'Each suggestion should be a fresh, creative prompt that relates to the image content but does not repeat or directly reference the original prompt text. '
            'Focus on different aspects, styles, or interpretations that the image could inspire. '
            f'Return a JSON array of {n_suggestions} strings, like: {format_example}. '
            'Do not include any explanation or other text.'
        )
        # format_example = str([f"suggestion {i+1}" for i in range(n_suggestions)])
        # suggestion_prompt = (
        #     f'Given the prompt: "{prompt.get_final_prompt()}", generate {n_suggestions} variations, each adding an extra dimension to the original prompt. '
        #     'Base your suggestions on the attached image, such that they would make sense in the context of the image. '
        #     'Keep the core instruction of the original prompt intact. '
        #     f'Return a JSON array of {n_suggestions} strings, like: {format_example}. '
        #     'Do not include any explanation or other text.'
        # )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": f"You are a helpful assistant that generates image-to-image generation prompts based on user input."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": prompt.input_image.image},
                    {"type": "text", "text": suggestion_prompt}
                ]
            }
        ]

        response = self.pipe(messages, max_new_tokens=512)[0]['generated_text'][2]['content']

        json_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON array found in the response.")
        json_str = json_match.group(0)
        try:
            suggestions = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")

        return suggestions
    
    def enhance_prompt(self, prompt: Prompt):
        """
        Enhance the prompt by generating a more detailed version based on the input image.
        """
        enhancement_prompt = (
            f'Enhance the following prompt: "{prompt.prompt}". '
            'Extract the main nouns, verbs, and adjectives from the prompt. Write them as a comma-separated list.'
            'For example, the prompt: "Turn this into a green rubber duck, pixar style", would yield: "green, rubber duck, pixar style".'
            'Do not include any explanation or other text.'
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": f"You are a helpful assistant that enhances image-to-image generation prompts based on user input."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": enhancement_prompt}
                ]
            }
        ]

        response = self.pipe(messages, max_new_tokens=512)[0]['generated_text'][2]['content']
        
        return response