from modules import Prompt, PromptImage, Chat, VLM
from PIL import Image
import os

def get_absolute_path(relative_path):
    """
    Converts a relative path wrt to the file to an absolute path.
    """
    dirname = os.path.dirname(__file__)
    absolute_path = os.path.join(dirname, relative_path)

    return absolute_path

# get this from the login
user_id = 1

def example():
    """
    This function is for demonstration purposes
    """
    vlm = VLM()
    chat = Chat(user_id, "Example Chat")

    image_path = get_absolute_path('../images/duck.jpg')

    init_image = Image.open(image_path).convert("RGB")
    prompt_text = "Turn this into a rubber duck, using a pixar style. It should also be green"

    # Initial user provided prompt + image
    image = PromptImage(init_image, prompt_guidance = None, image_guidance = None, save=False)
    first_prompt = Prompt(prompt_text, depth = 0, input_image = image)
    image.set_input_prompt(first_prompt)

    chat.add_prompt(first_prompt)

    curr_prompt = first_prompt.get_final_prompt()
    print(curr_prompt)

    # first_prompt.enhance_prompt(vlm)

    # curr_prompt = first_prompt.get_final_prompt()
    # print(curr_prompt)

    # return sample_prompt

    suggestions = vlm.make_suggestions(first_prompt, n_suggestions=3)
    
    print(suggestions)

example()