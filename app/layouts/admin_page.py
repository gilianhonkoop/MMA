import dash
from dash import html, dcc, Input, Output, callback, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import base64
import io
import os
from db.database import Database

app = dash.Dash(__name__)
server = app.server

# === Constants for styling ===
BG = "#FFEED6"
GREEN = "#38432E"
G_HEIGHT = 260

# === Utility to generate word cloud image from list of words ===
def generate_wordcloud(words):
    if not words:
        return None
    text = " ".join(words)
    wc = WordCloud(width=400, height=200, background_color=BG, colormap="Dark2").generate(text)
    buffer = io.BytesIO()
    wc.to_image().save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# === Layout ===
app.layout = html.Div([
    html.Div(style={"backgroundColor": GREEN, "color": "white", "padding": "16px 20px"}, children=[
        html.Span("AI-D", style={"fontWeight": "bold", "fontSize": "26px", "marginRight": "10px"}),
        html.Span("|", style={"margin": "0 10px"}),
        html.Span("User Dashboard", style={"fontStyle": "italic", "fontSize": "22px"})
    ]),
    html.Div(style={"padding": "12px 20px"}, children=[
        dcc.Input(id="user-id", type="number", placeholder="Enter User ID", style={"marginRight": "10px"}),
        html.Button("Load Dashboard", id="load-button"),
    ]),
    dcc.Store(id="user-data"),
    html.Div(id="dashboard-content")
])

# === Callbacks ===
@callback(
    Output("user-data", "data"),
    Input("load-button", "n_clicks"),
    State("user-id", "value")
)
def fetch_user_data(n_clicks, user_id):
    if not n_clicks or not user_id:
        return dash.no_update

    with Database() as db:
        df = db.fetch_user_dashboard_data(user_id)
    return df.to_dict("records")

@callback(
    Output("dashboard-content", "children"),
    Input("user-data", "data")
)
def render_dashboard(data):
    if not data:
        return dash.no_update

    df = pd.DataFrame(data)

    # === Plot 1: Guidance vs Depth ===
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df['depth'], y=df['prompt_guidance'], mode='lines+markers', name='Prompt Guidance'))
    fig1.add_trace(go.Scatter(x=df['depth'], y=df['image_guidance'], mode='lines+markers', name='Image Guidance'))
    fig1.update_layout(
        title="Guidance vs Prompt Depth",
        height=G_HEIGHT,
        font=dict(family="sans-serif", size=13),
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(t=50, b=30, l=40, r=40),
        xaxis_title="Depth", yaxis_title="Guidance"
    )

    # === Plot 2: Bertscore & LPIPS vs Depth ===
    fig2 = go.Figure()
    if 'bert_novelty' in df.columns:
        fig2.add_trace(go.Scatter(x=df['depth'], y=df['bert_novelty'], mode='lines+markers', name='1 - BERTScore'))
    if 'lpips' in df.columns:
        fig2.add_trace(go.Scatter(x=df['depth'], y=df['lpips'], mode='lines+markers', name='LPIPS'))
    fig2.update_layout(
        title="Prompt and Image Novelty vs Depth",
        height=G_HEIGHT,
        font=dict(family="sans-serif", size=13),
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(t=50, b=30, l=40, r=40),
        xaxis_title="Depth", yaxis_title="Score"
    )

    # === Plot 3: Pie chart of Functionality ===
    func_counts = {
        "Suggestions": df['used_suggestion'].sum(),
        "Enhancement": df['is_enhanced'].sum(),
        "Suggestions & Enhancement": ((df['used_suggestion']) & (df['is_enhanced'])).sum(),
        "No AI": ((~df['used_suggestion']) & (~df['is_enhanced'])).sum()
    }
    fig3 = go.Figure(go.Pie(labels=list(func_counts.keys()), values=list(func_counts.values()),
                            marker_colors=["#7B002C", "#00008B", "#8B4513", "#006400"], textinfo="label+percent"))
    fig3.update_layout(
        title="Functionality Usage",
        height=G_HEIGHT,
        font=dict(family="sans-serif", size=13),
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(t=50, b=30, l=40, r=40),
        showlegend=False
    )

    # === Plot 4: Word Cloud from Relevant Words ===
    all_words = []
    if 'relevant_words' in df.columns:
        for words in df['relevant_words'].dropna():
            all_words.extend(words.split(","))

    wordcloud_img = generate_wordcloud(all_words)
    wordcloud_fig = html.Img(src=wordcloud_img, style={"width": "100%"}) if wordcloud_img else html.Div("No keywords to display")

    return html.Div([
        html.Div(style={"display": "flex"}, children=[
            html.Div(dcc.Graph(figure=fig1), style={"width": "50%"}),
            html.Div(dcc.Graph(figure=fig2), style={"width": "50%"})
        ]),
        html.Div(style={"display": "flex"}, children=[
            html.Div(dcc.Graph(figure=fig3), style={"width": "50%"}),
            html.Div(wordcloud_fig, style={"width": "50%", "padding": "10px"})
        ])
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
