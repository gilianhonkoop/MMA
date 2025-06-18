from dash import dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

import sys
import os
import pandas as pd
import torch
import lpips
import clip
import torchvision.transforms as transforms
from PIL import Image
from bert_score import score as bertscore
import numpy as np
from brisque import BRISQUE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../db')))
from database import Database

# Logging setup
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup (LPIPS and CLIP) models
device = "cuda" if torch.cuda.is_available() else "cpu"
lpips_model = lpips.LPIPS(net='alex').to(device)
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Trying to perform Image Quality Assessment on local images
brisque_model = BRISQUE(url=False)

# Cache dictionaries
clip_cache = {}
lpips_cache = {}
brisque_cache = {}

# --- Utilities ---

# LPIPS distance
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

# CLIPScore
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
        return similarity
    except Exception as e:
        logger.warning(f"CLIPScore error ({image_path}, '{prompt}'): {e}")
        return None
    
# Brisque
def calculate_brisque_score(image_path):
    try:
        if image_path in brisque_cache:
            return brisque_cache[image_path]
        if not os.path.exists(image_path):
            return None
        img = Image.open(image_path).convert('RGB')
        img_np = np.asarray(img)
        score = brisque_model.score(img_np)
        return score
    except Exception as e:
        logger.warning(f"BRISQUE error ({image_path}): {e}")
        return None

# def filter_by_mode(df, mode):
#     if mode == 'overview' or 'interaction_mode' not in df.columns:
#         return df
#     return df[df['interaction_mode'] == mode]


def get_mode(row):
    if row['enhanced_prompt'] and row['used_suggestion']:
        return 'Both'
    elif row['enhanced_prompt']:
        return 'Enhancement Only'
    elif row['used_suggestion']:
        return 'Suggestion Only'
    return 'Baseline'

def filter_by_mode(df, mode):
    if mode == 'overview':
        return df

    df['interaction_mode'] = df.apply(get_mode, axis=1)
    return df[df['interaction_mode'] == mode]


# --- Layout ---

def create_admin_layout():
    return dbc.Container([
        dbc.Button("Refresh", id="refresh-button", color="primary", className="mb-3"),
        dcc.Tabs(
            id='interaction-mode-tabs', 
            value='overview', 
            children=[
                dcc.Tab(label='Overview', value='overview'),
                dcc.Tab(label='Baseline', value='baseline'),
                dcc.Tab(label='Enhancement Only', value='enhancement'),
                dcc.Tab(label='Suggestion Only', value='suggestion'),
                dcc.Tab(label='Both', value='both'),
                ]),
        # dcc.Graph(id='usage-mode-pie'),
        html.Div(dcc.Graph(id='usage-mode-pie'), id='usage-mode-pie-container'),
        dcc.Graph(id={'type': 'graph', 'index': 'image-prompt-fidelity'}),
        dcc.Graph(id={'type': 'graph', 'index': 'prompt-novelty'}),
        dcc.Graph(id={'type': 'graph', 'index': 'image-novelty'}),
        dcc.Graph(id={'type': 'graph', 'index': 'image-quality'}),
        dcc.Store(id='dashboard-data'),
    ])

# --- Callbacks ---

# Refresh
@callback(Output('dashboard-data', 'data'), Input('refresh-button', 'n_clicks'))
def update_dashboard_data(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return {"data_loaded": True}


# Ensure pie chart only shows in overall
@callback(
    Output('usage-mode-pie-container', 'style'),
    Input('interaction-mode-tabs', 'value')
)
def toggle_pie_chart_visibility(tab):
    if tab == 'overview':
        return {'display': 'block'}
    return {'display': 'none'}


# Usage Mode Pie-Chart
@callback(Output('usage-mode-pie', 'figure'), Input('dashboard-data', 'data'))
def update_usage_pie_chart(dashboard_data):
    if not dashboard_data or not dashboard_data.get("data_loaded"):
        raise PreventUpdate
    with Database() as db:
        df = db.fetch_all_images()

    df['mode'] = df.apply(get_mode, axis=1)
    counts = df['mode'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values)])
    fig.update_layout(title="User Interaction Mode Usage")
    return fig

# Image-Prompt Fidelity (CLIPScore)
@callback(Output({'type': 'graph', 'index': 'image-prompt-fidelity'}, 'figure'),
          Input('dashboard-data', 'data'),
          Input('interaction-mode-tabs', 'value'))
def update_image_prompt_fidelity(dashboard_data, mode):
    if dashboard_data is None or not dashboard_data.get("data_loaded"):
        raise PreventUpdate
    with Database() as db:
        df = db.fetch_all_images()
    df = filter_by_mode(df, mode)

    scores = []
    for _, row in df.iterrows():
        score = calculate_clip_score(row['image_path'], row['prompt'])
        if score is not None:
            scores.append(score)

    if not scores:
        raise PreventUpdate

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=scores, mode='lines+markers'))
    fig.update_layout(title="Image-Prompt Fidelity via CLIPScore", yaxis_title="CLIP Similarity", xaxis_title="Prompt Step")
    return fig

# Prompt Novelty (BERTScore)
@callback(Output({'type': 'graph', 'index': 'prompt-novelty'}, 'figure'),
          Input('dashboard-data', 'data'),
          Input('interaction-mode-tabs', 'value'))
def update_prompt_novelty(dashboard_data, mode):
    if dashboard_data is None or not dashboard_data.get("data_loaded"):
        raise PreventUpdate
    
    with Database() as db:
        df = db.fetch_all_images()
    df = filter_by_mode(df, mode)

    if df.empty or 'prompt' not in df.columns or 'prompt_number' not in df.columns:
        raise PreventUpdate

    scores = []
    for _, row in df.iterrows():
        if row['prompt_number'] <= 1:
            continue # Skip the first prompt — no previous to compare
        prev = df[(df['user_id'] == row['user_id']) & (df['prompt_number'] == row['prompt_number'] - 1)]
        if prev.empty:
            continue
        try:
            _, _, F1 = bertscore([row['prompt']], [prev.iloc[0]['prompt']], lang='en', verbose=False)
            scores.append(1 - F1.item())
        except Exception as e:
            print(f"BERTScore error: {e}")
            continue

    if not scores:
        raise PreventUpdate

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=scores, 
                             mode='lines+markers'))
    fig.update_layout(title="Prompt Novelty Over Generations",
                      yaxis_title="Novelty (1-BERTScore)", 
                      xaxis_title="Prompt Step")
    return fig


# Image Novelty (LPIPS)
@callback(Output({'type': 'graph', 'index': 'image-novelty'}, 'figure'),
          Input('dashboard-data', 'data'),
          Input('interaction-mode-tabs', 'value'))
def update_image_novelty(dashboard_data, mode):
    if dashboard_data is None or not dashboard_data.get("data_loaded"):
        raise PreventUpdate
    with Database() as db:
        df = db.fetch_all_images()
    df = filter_by_mode(df, mode)

    # ASSUMPTION: images with prompt_number > 1 have a previous version
    lpips_scores = []
    for _, row in df.iterrows():
        if row['prompt_number'] <= 1:
            continue
        # Try to find matching previous image
        prev = df[(df['user_id'] == row['user_id']) & (df['prompt_number'] == row['prompt_number'] - 1)]
        if prev.empty:
            continue
        score = calculate_lpips(row['image_path'], prev.iloc[0]['image_path'])
        if score is not None:
            lpips_scores.append(score)

    if not lpips_scores:
        raise PreventUpdate

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=lpips_scores,
                              mode='lines+markers'))
    fig.update_layout(title="Image Novelty via LPIPS", 
                      yaxis_title="LPIPS Distance", 
                      xaxis_title="Prompt Step")
    return fig

# Image Quality (BRISQUE image quality)
# @callback(Output({'type': 'graph', 'index': 'image-quality'}, 'figure'),
#           Input('dashboard-data', 'data'),
#           Input('interaction-mode-tabs', 'value'))
# def update_brisque_quality_plot(dashboard_data, mode):
#     if not dashboard_data.get("data_loaded"):
#         raise PreventUpdate
#     # Placeholder BRISQUE plot
#     brisque_score = brisque.score(io.imread(row['image_path']))
#     fig.add_trace(go.Scatter(y=brisque_score,
#                               mode='lines+markers'))
#     fig = go.Figure()
#     fig.update_layout(title="Image Quality (BRISQUE - Placeholder)", yaxis_title="Quality Score", xaxis_title="Prompt Step")
#     return fig

@callback(
    Output({'type': 'graph', 'index': 'image-quality'}, 'figure'),
    Input('dashboard-data', 'data'),
    Input('interaction-mode-tabs', 'value')
)
def update_image_quality(dashboard_data, mode):
    if dashboard_data is None or not dashboard_data.get("data_loaded"):
        raise PreventUpdate

    with Database() as db:
        df = db.fetch_all_images()
    df = filter_by_mode(df, mode)

    scores = []
    for _, row in df.iterrows():
        score = calculate_brisque_score(row['image_path'])
        if score is not None:
            scores.append(score)

    if not scores:
        raise PreventUpdate

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=scores, mode='lines+markers', name='BRISQUE'))
    fig.update_layout(title="Image Quality (BRISQUE)", yaxis_title="BRISQUE Score", xaxis_title="Prompt Step")
    return fig

@callback(
    Output({'type': 'graph', 'index': 'model-parameters'}, 'figure'),
    Input('dashboard-data', 'data')
)
def update_model_parameters(dashboard_data):
    if dashboard_data is None or not dashboard_data.get("data_loaded"):
        raise PreventUpdate
    
    fig = go.Figure()
    fig.update_layout(title="Strength and guidance score plot")
    
    return fig
