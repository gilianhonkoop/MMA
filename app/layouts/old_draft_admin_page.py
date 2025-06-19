from dash import dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../db')))
from database import Database

import lpips
import torchvision.transforms as transforms
from PIL import Image
import torch
import clip
from bert_score import score as bertscore
import pandas as pd

# Load LPIPS and CLIP models
device = "cuda" if torch.cuda.is_available() else "cpu"
lpips_model = lpips.LPIPS(net='alex').to(device)
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Utility: LPIPS distance
def calculate_lpips(img1_path, img2_path):
    try:
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
        return dist.item()
    except:
        return None

# Utility: CLIPScore
def calculate_clip_score(image_path, prompt):
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        text = clip.tokenize([prompt]).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            text_features = clip_model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()
        return similarity
    except:
        return None

# Layout
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
            ]
        ),
        dcc.Graph(id='usage-mode-pie'),
        dcc.Graph(id={'type': 'graph', 'index': 'enhancement-amplitude'}),
        dcc.Graph(id={'type': 'graph', 'index': 'enhancement-quality'}),
        dcc.Graph(id={'type': 'graph', 'index': 'suggestion-amplitude'}),
        dcc.Graph(id={'type': 'graph', 'index': 'suggestion-quality'}),
        dcc.Graph(id={'type': 'graph', 'index': 'model-parameters'}),
        dcc.Graph(id={'type': 'graph', 'index': 'prompt-novelty'}),
        dcc.Graph(id={'type': 'graph', 'index': 'image-prompt-fidelity'}),
        dcc.Store(id='dashboard-data'),
    ])

# Callback: Refresh
@callback(Output('dashboard-data', 'data'), Input('refresh-button', 'n_clicks'))
def update_dashboard_data(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return {"data_loaded": True}

# Callback: Usage Mode Pie Chart
@callback(Output('usage-mode-pie', 'figure'), Input('dashboard-data', 'data'))
def update_usage_pie_chart(dashboard_data):
    if dashboard_data is None or not dashboard_data.get("data_loaded"):
        raise PreventUpdate

    with Database() as db:
        df = db.fetch_all_images()

    if 'enhanced_prompt' not in df.columns or 'used_suggestion' not in df.columns:
        raise PreventUpdate

    def get_mode(row):
        if row['enhanced_prompt'] and row['used_suggestion']:
            return 'Both'
        elif row['enhanced_prompt']:
            return 'Enhancement Only'
        elif row['used_suggestion']:
            return 'Suggestion Only'
        else:
            return 'Baseline'

    df['mode'] = df.apply(get_mode, axis=1)
    mode_counts = df['mode'].value_counts()

    fig = go.Figure(data=[go.Pie(labels=mode_counts.index, values=mode_counts.values)])
    fig.update_layout(title="User Interaction Mode Usage")
    return fig

# Common utility to filter by interaction mode
def filter_by_mode(df, mode):
    if mode != 'all' and 'interaction_mode' in df.columns:
        return df[df['interaction_mode'] == mode]
    return df

# Callback: Image-Prompt Fidelity (CLIPScore)
@callback(
    Output({'type': 'graph', 'index': 'image-prompt-fidelity'}, 'figure'),
    Input('dashboard-data', 'data'),
    Input('interaction-mode-tabs', 'value')
)
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
    fig.add_trace(go.Scatter(y=scores, mode='lines+markers', name='CLIPScore'))
    fig.update_layout(title="Image-Prompt Fidelity via CLIPScore", yaxis_title="CLIP Similarity", xaxis_title="Prompt Step")
    return fig


# Callback: Prompt Novelty (BERTScore)
@callback(
    Output({'type': 'graph', 'index': 'prompt-novelty'}, 'figure'),
    Input('dashboard-data', 'data'),
    Input('interaction-mode-tabs', 'value')
)
def update_prompt_novelty(dashboard_data, mode):
    if dashboard_data is None or not dashboard_data.get("data_loaded"):
        raise PreventUpdate

    with Database() as db:
        df = db.fetch_all_images()
    df = filter_by_mode(df, mode)

    if df.empty or 'prompt' not in df.columns or 'prompt_number' not in df.columns:
        raise PreventUpdate

    novelty_scores = []

    for _, row in df.iterrows():
        if row['prompt_number'] <= 1:
            continue  # Skip the first prompt — no previous to compare

        user_id = row['user_id']
        prev_prompt_num = row['prompt_number'] - 1

        prev_row = df[(df['user_id'] == user_id) & (df['prompt_number'] == prev_prompt_num)]

        if prev_row.empty:
            continue

        prev_prompt = prev_row.iloc[0]['prompt']
        current_prompt = row['prompt']

        try:
            _, _, F1 = bertscore([current_prompt], [prev_prompt], lang='en', verbose=False)
            novelty = 1 - F1.item()
            novelty_scores.append(novelty)
        except Exception as e:
            print(f"BERTScore error: {e}")
            continue

    if not novelty_scores:
        raise PreventUpdate

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=novelty_scores,
        mode='lines+markers',
        name='Prompt Novelty (1 - BERTScore)'
    ))
    fig.update_layout(
        title="Prompt Novelty Over Generations",
        yaxis_title="Novelty (1 - BERTScore)",
        xaxis_title="Prompt Step"
    )

    return fig


# Callback: Image Quality (BRISQUE image quality)
@callback(
    Output({'type': 'graph', 'index': 'enhancement-quality'}, 'figure'),
    Input('dashboard-data', 'data')
)
def update_enhancement_quality(dashboard_data):
    if dashboard_data is None or not dashboard_data.get("data_loaded"):
        raise PreventUpdate
    
    fig = go.Figure()
    fig.update_layout(title="Image quality")
    
    return fig

# Callback: Image Novelty (LPIPS)
@callback(
    Output({'type': 'graph', 'index': 'amplitude'}, 'figure'),
    Input('dashboard-data', 'data')
)
def update_amplitude(dashboard_data):
    if dashboard_data is None or not dashboard_data.get("data_loaded"):
        raise PreventUpdate
    
    with Database() as db:
        df = db.fetch_all_images()

    if df.empty or 'image_path' not in df.columns or 'prompt_number' not in df.columns:
        raise PreventUpdate
    
    # ASSUMPTION: images with prompt_number > 1 have a previous version
    image_changes = []
    for _, row in df.iterrows():
        if row['prompt_number'] <= 1:
            continue
        user_id = row['user_id']
        prev_prompt_num = row['prompt_number'] - 1
        current_img = row['image_path']

        # Try to find matching previous image
        prev_row = df[(df['user_id'] == user_id) & (df['prompt_number'] == prev_prompt_num)]
        if prev_row.empty:
            continue

        prev_img = prev_row.iloc[0]['image_path']
        dist = calculate_lpips(current_img, prev_img)
        if dist is not None:
            image_changes.append(dist)

    if not image_changes:
        raise PreventUpdate
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=image_changes, mode='lines+markers', name='LPIPS'))
    fig.update_layout(title="Suggestion Amplitude (Image Change via LPIPS)",
        yaxis_title="LPIPS Distance",
        xaxis_title="Prompt Step")
    
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
