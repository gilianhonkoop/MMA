from transformers import pipeline, AutoProcessor
from huggingface_hub import login
import os
import torch
import dotenv
import json
import re
from PIL import Image

class Prompt():
    def __init__(self, initial_prompt, itial_img, model="google/gemma-3-27b-it"):
        self.current_prompt = initial_prompt
        self.current_prompt_img = itial_img
        self.suggestions = []
        self.updates = [initial_prompt] # the updates made to the original prompt through either a suggestion or manual input

        torch.set_float32_matmul_precision('high')
        
        processor = AutoProcessor.from_pretrained(model, use_fast=True)

        self.pipe = pipeline(
            "image-text-to-text",
            model=model,
            device="cuda",
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            processor=processor,
        )
        print(f"{model} loaded successfully.")

    def make_suggestions(self, n_suggestions):
        format_example = str([f"suggestion {i+1}" for i in range(n_suggestions)])
        suggestion_prompt = (
            f'Given the prompt: "{self.current_prompt}", generate {n_suggestions} variations, each adding an extra dimension to the original prompt. '
            'Base your suggestions on the attached image, such that they would make sense in the context of the image. '
            'Keep the core instruction of the original prompt intact. '
            f'Return a JSON array of {n_suggestions} strings, like: {format_example}. '
            'Do not include any explanation or other text.'
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": f"You are a helpful assistant that generates image-to-image generation prompts based on user input."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.current_prompt_img},
                    {"type": "text", "text": suggestion_prompt}
                ]
            }
        ]

        response = self.pipe(messages, max_new_tokens=512)[0]['generated_text'][2]['content']

        # Extract the JSON array from the response
        json_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON array found in the response.")
        json_str = json_match.group(0)
        try:
            suggestions = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}")

        self.suggestions = suggestions
        return suggestions


    def use_suggestion(self, suggestion_index):
        # Set the current prompt, add it to the history, and regenerate suggestions.
        self.current_prompt = self.suggestions[suggestion_index]
        self.updates.append(self.current_prompt)
        self.make_suggestions(len(self.suggestions))
        


dotenv.load_dotenv()
login(os.environ.get("HF_TOKEN"))    

image = Image.open("duck.jpg").convert("RGB")
prompt = Prompt("Turn this into a rubber duck", image, model="google/gemma-3-12b-it")
suggestions = prompt.make_suggestions(3)
print(suggestions)