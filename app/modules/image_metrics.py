# modules/admin_metrics.py

import numpy as np
import plotly.graph_objects as go
from db.database import Database
from modules.image_metrics import calculate_lpips, calculate_clip_score, calculate_brisque_score
from bert_score import score as bertscore

import os

DB_PATH = os.path.abspath(os.path.join("..", "app", "db", "mma.db"))


def get_dataframe(mode='overview'):
    with Database(DB_PATH) as db:
        df_images = db.fetch_all_images()
        df_prompts = db.fetch_all_prompts()
        df = df_images.merge(df_prompts, left_on="prompt_id", right_on="id", suffixes=("_image", "_prompt"))

    def get_mode(row):
        if row.get('enhanced_prompt') and row.get('used_suggestion'):
            return 'Both'
        elif row.get('enhanced_prompt'):
            return 'Enhancement Only'
        elif row.get('used_suggestion'):
            return 'Suggestion Only'
        return 'Baseline'

    df['mode'] = df.apply(get_mode, axis=1)

    if mode.lower() == "overview":
        return df
    elif mode.lower() == "enhancement":
        return df[df["mode"] == "Enhancement Only"]
    elif mode.lower() == "suggestions":
        return df[df["mode"] == "Suggestion Only"]
    elif mode.lower() == "both":
        return df[df["mode"] == "Both"]
    elif mode.lower() == "noai":
        return df[df["mode"] == "Baseline"]
    return df


def prompt_novelty_chart(df):
    scores = []
    for _, row in df.iterrows():
        if row['depth'] <= 1:
            continue
        prev = df[(df['user_id'] == row['user_id']) & (df['depth'] == row['depth'] - 1)]
        if prev.empty:
            continue
        try:
            _, _, F1 = bertscore([row['prompt']], [prev.iloc[0]['prompt']], lang='en', verbose=False)
            scores.append(1 - F1.item())
        except Exception:
            continue

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=scores, mode='lines+markers'))
    fig.update_layout(title="Prompt Novelty", yaxis_title="1 - BERTScore", xaxis_title="Prompt #")
    return fig


def image_novelty_chart(df):
    scores = []
    for _, row in df.iterrows():
        if row['prompt_number'] <= 1:
            continue
        prev = df[(df['user_id'] == row['user_id']) & (df['prompt_number'] == row['prompt_number'] - 1)]
        if prev.empty:
            continue
        score = calculate_lpips(row['image_path'], prev.iloc[0]['image_path'])
        if score is not None:
            scores.append(score)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=scores, mode='lines+markers'))
    fig.update_layout(title="Image Novelty", yaxis_title="LPIPS", xaxis_title="Prompt #")
    return fig


def clip_score_chart(df):
    scores = []
    for _, row in df.iterrows():
        score = calculate_clip_score(row['image_path'], row['prompt'])
        if score is not None:
            scores.append(score)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=scores, mode='lines+markers'))
    fig.update_layout(title="Image–Prompt Fidelity", yaxis_title="CLIP Score", xaxis_title="Prompt #")
    return fig


def brisque_chart(df):
    scores = []
    for _, row in df.iterrows():
        score = calculate_brisque_score(row['image_path'])
        if score is not None:
            scores.append(score)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=scores, mode='lines+markers'))
    fig.update_layout(title="Image Quality (BRISQUE)", yaxis_title="BRISQUE", xaxis_title="Prompt #")
    return fig


def pie_chart(df):
    mode_counts = df['mode'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=mode_counts.index, values=mode_counts.values)])
    fig.update_layout(title="User Interaction Mode Usage")
    return fig



# import os
# import torch
# import clip
# import lpips
# import numpy as np
# import logging
# from PIL import Image
# from brisque import BRISQUE
# import torchvision.transforms as transforms

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Setup device and models
# device = "cuda" if torch.cuda.is_available() else "cpu"
# lpips_model = lpips.LPIPS(net='alex').to(device)
# clip_model, preprocess = clip.load("ViT-B/32", device=device)
# brisque_model = BRISQUE(url=False)

# # Caches
# clip_cache = {}
# lpips_cache = {}
# brisque_cache = {}

# def calculate_clip_score(image_path, prompt):
#     try:
#         key = (image_path, prompt)
#         if key in clip_cache:
#             return clip_cache[key]

#         image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
#         text = clip.tokenize([prompt]).to(device)

#         with torch.no_grad():
#             image_features = clip_model.encode_image(image)
#             text_features = clip_model.encode_text(text)
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             text_features /= text_features.norm(dim=-1, keepdim=True)
#             similarity = (image_features @ text_features.T).item()

#         clip_cache[key] = similarity
#         return similarity
#     except Exception as e:
#         logger.warning(f"CLIPScore error ({image_path}, '{prompt}'): {e}")
#         return None

# def calculate_lpips(img1_path, img2_path):
#     try:
#         key = (img1_path, img2_path)
#         if key in lpips_cache:
#             return lpips_cache[key]

#         if not os.path.exists(img1_path) or not os.path.exists(img2_path):
#             return None

#         transform = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5])
#         ])
#         img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0).to(device)
#         img2 = transform(Image.open(img2_path).convert('RGB')).unsqueeze(0).to(device)
#         dist = lpips_model(img1, img2)

#         lpips_cache[key] = dist.item()
#         return dist.item()
#     except Exception as e:
#         logger.warning(f"LPIPS error ({img1_path}, {img2_path}): {e}")
#         return None

# def calculate_brisque_score(image_path):
#     try:
#         if image_path in brisque_cache:
#             return brisque_cache[image_path]

#         if not os.path.exists(image_path):
#             return None

#         img = Image.open(image_path).convert('RGB')
#         img_np = np.asarray(img)
#         score = brisque_model.score(img_np)
#         brisque_cache[image_path] = score
#         return score
#     except Exception as e:
#         logger.warning(f"BRISQUE error ({image_path}): {e}")
#         return None
