from .vlm import VLM
from .image_transformer import ImageTransformer

vlm_instance = None
image_transformer_instances = [None, None, None]

def init_models():
    global vlm_instance, image_transformer_instances
    
    print("Initializing VLM model on cuda:0...")
    vlm_instance = VLM(device="cuda:0")
    print("VLM model initialized successfully on cuda:0.")
    
    print("Initializing Image Transformer models...")
    for i in range(3):
        device = f"cuda:{i+1}"
        print(f"Initializing Image Transformer model on {device}...")
        image_transformer_instances[i] = ImageTransformer(device=device)
        print(f"Image Transformer model initialized successfully on {device}.")
    
    return vlm_instance, image_transformer_instances

def get_vlm_instance():
    global vlm_instance
    if vlm_instance is None:
        print("VLM model not initialized, initializing now...")
        vlm_instance = VLM(device="cuda:0")
    return vlm_instance

def get_image_transformer_instance(gpu_id=0):
    global image_transformer_instances
    if gpu_id < 0 or gpu_id > 2:
        raise ValueError("gpu_id must be between 0 and 2")
    
    if image_transformer_instances[gpu_id] is None:
        device = f"cuda:{gpu_id+1}"
        print(f"Image Transformer model not initialized on {device}, initializing now...")
        image_transformer_instances[gpu_id] = ImageTransformer(device=device)
    return image_transformer_instances[gpu_id]

def get_all_image_transformer_instances():
    global image_transformer_instances
    for i in range(3):
        if image_transformer_instances[i] is None:
            device = f"cuda:{i+1}"
            print(f"Image Transformer model not initialized on {device}, initializing now...")
            image_transformer_instances[i] = ImageTransformer(device=device)
    return image_transformer_instances