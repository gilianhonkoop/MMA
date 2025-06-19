import os
import torch
import clip
import lpips
import numpy as np
import logging
from PIL import Image
from brisque import BRISQUE
import torchvision.transforms as transforms

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup device and models
device = "cuda" if torch.cuda.is_available() else "cpu"
lpips_model = lpips.LPIPS(net='alex').to(device)
clip_model, preprocess = clip.load("ViT-B/32", device=device)
brisque_model = BRISQUE(url=False)

# Caches
clip_cache = {}
lpips_cache = {}
brisque_cache = {}

def calculate_clip_score(image_path, prompt):
    try:
        key = (image_path, prompt)
        if key in clip_cache:
            return clip_cache[key]

        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        text = clip.tokenize([prompt]).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            text_features = clip_model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()

        clip_cache[key] = similarity
        return similarity
    except Exception as e:
        logger.warning(f"CLIPScore error ({image_path}, '{prompt}'): {e}")
        return None

def calculate_lpips(img1_path, img2_path):
    try:
        key = (img1_path, img2_path)
        if key in lpips_cache:
            return lpips_cache[key]

        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            return None

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0).to(device)
        img2 = transform(Image.open(img2_path).convert('RGB')).unsqueeze(0).to(device)
        dist = lpips_model(img1, img2)

        lpips_cache[key] = dist.item()
        return dist.item()
    except Exception as e:
        logger.warning(f"LPIPS error ({img1_path}, {img2_path}): {e}")
        return None

def calculate_brisque_score(image_path):
    try:
        if image_path in brisque_cache:
            return brisque_cache[image_path]

        if not os.path.exists(image_path):
            return None

        img = Image.open(image_path).convert('RGB')
        img_np = np.asarray(img)
        score = brisque_model.score(img_np)
        brisque_cache[image_path] = score
        return score
    except Exception as e:
        logger.warning(f"BRISQUE error ({image_path}): {e}")
        return None
