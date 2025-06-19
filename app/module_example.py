from modules import Prompt, PromptImage, Chat, VLM, ImageTransformer
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
    im2im = ImageTransformer()
    chat = Chat(user_id, "Example Chat")

    image_path = get_absolute_path('../images/duck.jpg')

    # Initial user provided prompt + image
    init_image : Image.Image = Image.open(image_path).convert("RGB")
    prompt_text = "Turn this into a rubber duck"
    # prompt_text = "Turn this into a rubber duck, using a pixar style. It should also be green"

    image = PromptImage(init_image, prompt_guidance = None, image_guidance = None, save=False)
    first_prompt  = Prompt(prompt_text, depth = 0, input_image = image)
    image.set_input_prompt(first_prompt)

    chat.add_prompt(first_prompt)

    # Get modified image
    output_image : PromptImage = first_prompt.get_new_images(im2im, n=1, save=False)[0]

    # As for this part I am not sure what the desired implementation is
    # We can either
    # 1) have the second_prompt_text be the final prompt text of the first_prompt. Afterwards we can transform this with a suggestion
    # 2) have the second_prompt_text be a user provided prompt / prompt after a suggestion has been made
    #TODO
    suggestions = vlm.make_suggestions(first_prompt, n_suggestions=3)
    second_prompt_text = ""

    # Second prompt
    second_prompt = Prompt(second_prompt_text, depth = 1, input_image = output_image)

    second_prompt.enhance_prompt(vlm)
    last_prompt = second_prompt.get_final_prompt()
    print(last_prompt)
    
    chat.add_prompt(second_prompt)

example()