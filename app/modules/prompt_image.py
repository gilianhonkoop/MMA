from PIL import Image
import uuid
import os

def get_absolute_path(relative_path):
    """
    Converts a relative path wrt to the file to an absolute path.
    """
    dirname = os.path.dirname(__file__)
    absolute_path = os.path.join(dirname, relative_path)

    return absolute_path

class PromptImage():
    def __init__(self, image: Image.Image, prompt_guidance: float, image_guidance: float, input_prompt = None, output_prompt = None, save: bool = True):
        # maybe uuid is better, otherwise db fetches are needed to get the latest id value that can be used
        self.id = str(uuid.uuid4())
        self.image : Image.Image = image
        self.prompt_guidance : float = prompt_guidance
        self.image_guidance : float = image_guidance
        self.path : str = get_absolute_path("../../images/" + self.id + ".png")
        self.input_prompt = input_prompt
        self.output_prompt = output_prompt

        if save:
            if not os.path.exists("images"):
                os.makedirs("images")
            self.image.save(self.path, format="PNG")

    def set_input_prompt(self, prompt):
        """Set the input prompt for this image."""
        self.input_prompt = prompt
    
    def set_output_prompt(self, prompt):
        """Set the output prompt for this image."""
        self.output_prompt = prompt

    def get_path(self):
        """Get the file path of the image."""
        return self.path