from .vlm import VLM
from .image_transformer import ImageTransformer

vlm_instance = None
image_transformer_instance = None

def init_models():
    global vlm_instance, image_transformer_instance
    
    print("Initializing VLM model...")
    vlm_instance = VLM()
    print("VLM model initialized successfully.")
    
    print("Initializing Image Transformer model...")
    image_transformer_instance = ImageTransformer()
    print("Image Transformer model initialized successfully.")
    
    return vlm_instance, image_transformer_instance

def get_vlm_instance():
    global vlm_instance
    if vlm_instance is None:
        print("VLM model not initialized, initializing now...")
        vlm_instance = VLM()
    return vlm_instance

def get_image_transformer_instance():
    global image_transformer_instance
    if image_transformer_instance is None:
        print("Image Transformer model not initialized, initializing now...")
        image_transformer_instance = ImageTransformer()
    return image_transformer_instance