import os
import torch
import lpips
import numpy as np
import pandas as pd
import logging
from PIL import Image
import torchvision.transforms as transforms
from bert_score import score as bertscore
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def extract_relevant_words(prompt_text):
    words = re.findall(r'\b\w+\b', prompt_text.lower())
    filtered = [w for w in words if w not in stop_words]
    return filtered, len(words)

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
lpips_model = lpips.LPIPS(net='alex').to(device)
lpips_cache = {}

def calculate_lpips(img1_path, img2_path):
    try:
        # print(f"Calculating LPIPS for: {img1_path} vs {img2_path}")
        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            print(f"Missing image file: {img1_path} or {img2_path}")
            return None

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
        # print(f"LPIPS result: {dist.item()}")
        return dist.item()
    except Exception as e:
        logger.warning(f"LPIPS error ({img1_path}, {img2_path}): {e}")
        return None

def get_or_compute_lpips(image_id, img1_path, img2_path):
    from db.database import Database
    with Database() as db:
        query = "SELECT lpips FROM lpips_metrics WHERE image_id = ?"
        result = db.fetch_dataframe(query, (image_id,))
        if not result.empty and pd.notna(result["lpips"].iloc[0]):
            return result["lpips"].iloc[0]
        score = calculate_lpips(img1_path, img2_path)
        return score

def get_or_compute_bertscore(prompt_id, current_prompt, previous_prompt):
    from db.database import Database
    with Database() as db:
        query = "SELECT bert_novelty FROM bertscore_metrics WHERE prompt_id = ?"
        result = db.fetch_dataframe(query, (prompt_id,))
        if not result.empty and pd.notna(result["bert_novelty"].iloc[0]):
            return result["bert_novelty"].iloc[0]
        try:
            _, _, F1 = bertscore([current_prompt], [previous_prompt], lang='en', verbose=False)
            novelty = 1 - F1.item()
            return novelty
        except Exception as e:
            logger.warning(f"BERTScore error for prompt {prompt_id}: {e}")
            return None
