from dash import dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

# Database from Gillian work
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../db')))
from database import Database

# For lpips
import lpips
import torchvision.transforms as transforms
from PIL import Image
import torch

# CLIP
import clip

# Load LPIPS once globally (uses AlexNet by default)
lpips_model = lpips.LPIPS(net='alex').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='alex')

# Define the function to calculate lpips
def calculate_lpips(img1_path, img2_path):
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        return None

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Optional, normalize sizes
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    try:
        img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0)
        img2 = transform(Image.open(img2_path).convert('RGB')).unsqueeze(0)

        if torch.cuda.is_available():
            img1 = img1.cuda()
            img2 = img2.cuda()

        dist = lpips_model(img1, img2)
        return dist.item()
    except Exception as e:
        print(f"Error comparing images {img1_path} and {img2_path}: {e}")
        return None

# Define clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def create_admin_layout():
    """
    Adds a Refresh button to trigger data loading. Renders 7 graphs:
    3 for enhancement metrics
    3 for suggestion metrics
    1 for model parameters (e.g., strength/guidance)
    Uses dcc.Store to cache the data once it is loaded
    """
    return dbc.Container([
        dbc.Button("Refresh", id="refresh-button", color="primary", className="mb-3"), #refresh button
        dcc.Graph(id={'type': 'graph', 'index': 'enhancement-usage-rate'}), # enhancement
        dcc.Graph(id={'type': 'graph', 'index': 'enhancement-amplitude'}),
        dcc.Graph(id={'type': 'graph', 'index': 'enhancement-quality'}),
        dcc.Graph(id={'type': 'graph', 'index': 'suggestion-usage-rate'}), # suggestion
        dcc.Graph(id={'type': 'graph', 'index': 'suggestion-amplitude'}),
        dcc.Graph(id={'type': 'graph', 'index': 'suggestion-quality'}),
        dcc.Graph(id={'type': 'graph', 'index': 'model-parameters'}), # model parameters
        dcc.Store(id='dashboard-data'),
    ])

# Callback 1: Load Dashboard Data
# What it does: When the user clicks "Refresh", this callback simulates loading 
# data. Currently, it just returns {"data_loaded": True} — a placeholder.


@callback(
    Output('dashboard-data', 'data'),
    Input('refresh-button', 'n_clicks')
)
def update_dashboard_data(n_clicks):
    """When the user clicks 'Refresh', this callback simulates loading data.
    Currently, it just returns {"data_loaded": True} — a placeholder.
    """
    if n_clicks is None:
        raise PreventUpdate
    
    return {
        "data_loaded": True,
    }

# Callbacks 2–8: Update Each Graph
# Each of these callbacks: Waits for 'dashboard-data' to be loaded. Creates an 
# empty go.Figure() with a placeholder title.

@callback(
    Output({'type': 'graph', 'index': 'enhancement-usage-rate'}, 'figure'),
    Input('dashboard-data', 'data')
)
def update_enhancement_usage_rate(dashboard_data):
    """Waits for 'dashboard-data' to be loaded. Creates an empty go.Figure() with a placeholder title."""
    if dashboard_data is None or not dashboard_data.get("data_loaded"):
        raise PreventUpdate
    
    with Database() as db:
        df = db.fetch_all_images()

    if df.empty:
        raise PreventUpdate
    
    usage_rate = df['enhanced_prompt'].mean() * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=usage_rate,
        title={"text": "Enhancement Usage Rate"},
        gauge={'axis': {'range': [0, 100]}},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    
    # fig = go.Figure()
    # fig.update_layout(title="Rate of use")
    
    return fig

@callback(
    Output({'type': 'graph', 'index': 'enhancement-amplitude'}, 'figure'),
    Input('dashboard-data', 'data')
)
def update_enhancement_amplitude(dashboard_data):
    if dashboard_data is None or not dashboard_data.get("data_loaded"):
        raise PreventUpdate
    
    fig = go.Figure()
    fig.update_layout(title="Amplitude of prompt-image change")
    
    return fig

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

@callback(
    Output({'type': 'graph', 'index': 'suggestion-usage-rate'}, 'figure'),
    Input('dashboard-data', 'data')
)
def update_suggestion_usage_rate(dashboard_data):
    if dashboard_data is None or not dashboard_data.get("data_loaded"):
        raise PreventUpdate
    
    fig = go.Figure()
    fig.update_layout(title="Rate of use")
    
    return fig

@callback(
    Output({'type': 'graph', 'index': 'suggestion-amplitude'}, 'figure'),
    Input('dashboard-data', 'data')
)
def update_suggestion_amplitude(dashboard_data):
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
    Output({'type': 'graph', 'index': 'suggestion-quality'}, 'figure'),
    Input('dashboard-data', 'data')
)
def update_suggestion_quality(dashboard_data):
    if dashboard_data is None or not dashboard_data.get("data_loaded"):
        raise PreventUpdate
    
    fig = go.Figure()
    fig.update_layout(title="Image quality")
    
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