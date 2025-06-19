from typing import List
from .prompt import Prompt
from PIL import Image

class Chat():
    def __init__(self, user_id: int, title: str):
        self.user_id : int = user_id
        self.title : str = title
        self.prompts : List[Prompt] = []
    
    def add_prompt(self, prompt: Prompt):
        if not isinstance(prompt, Prompt):
            raise TypeError("Expected prompt to be an instance of Prompt class.")

        self.prompts.append(prompt)

    def get_prompts(self) -> List[Prompt]:
        """
        Returns a list of prompts in the chat.
        """
        return self.prompts
    
    def get_images(self) -> List[Image.Image]:
        """
        Returns a list of images associated with the prompts in the chat.
        """
        images = []

        for prompt in self.prompts:
            # every output image is also an input image, except the first user 
            # provided image which we did not generate
            for img in prompt.images_out:
                images.append(img.image)

        return images