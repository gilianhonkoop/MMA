import dash
from dash import html, dcc, callback, Input, Output
from modules.image_metrics import get_dataframe, prompt_novelty_chart, image_novelty_chart, brisque_chart, clip_score_chart, pie_chart

BG = "#FFEED6"
GREEN = "#38432E"

def get_real_figures(mode):
    df = get_dataframe(mode)
    return [
        clip_score_chart(df),
        brisque_chart(df),
        image_novelty_chart(df),
        prompt_novelty_chart(df),
        pie_chart(df)
    ]

def create_admin_layout():
    return html.Div([
        dcc.Store(id="admin-tab-store", data="overall"),
        html.Div(style={"backgroundColor": GREEN, "color": "white", "padding": "16px 20px"}, children=[
            html.Span("AI-D", style={"fontWeight": "bold", "fontSize": "26px", "marginRight": "10px"}),
            html.Span("|", style={"margin": "0 10px"}),
            html.Span("Admin Dashboard", style={"fontStyle": "italic", "fontSize": "22px"})
        ]),
        html.Div(style={"display": "flex", "padding": "16px 20px 0 20px"}, children=[
            html.Div("Prompt–Image Fidelity", style={"width": "33%", "fontSize": "20px", "fontWeight": "600", "color": GREEN}),
            html.Div("Prompt & Image Novelty", style={"width": "33%", "textAlign": "center", "fontSize": "20px", "fontWeight": "600", "color": GREEN}),
            html.Div("Utility & Quality", style={"width": "33%", "textAlign": "right", "fontSize": "20px", "fontWeight": "600", "color": GREEN})
        ]),
        html.Div(style={"width": "60%", "margin": "0 auto", "marginBottom": "10px"}, children=[
            dcc.Tabs(id="admin-tab", value="overall", children=[
                dcc.Tab(label="Overall", value="overview"),
                dcc.Tab(label="Suggestions", value="suggestions"),
                dcc.Tab(label="Enhancement", value="enhancement"),
                dcc.Tab(label="Suggestions + Enhancement", value="both"),
                dcc.Tab(label="No AI", value="noai")
            ])
        ]),
        html.Div(id="admin-grid-content")
    ])

@callback(
    Output("admin-grid-content", "children"),
    Input("admin-tab-store", "data")
)
def render_admin_figures(tab):
    figs = get_real_figures(tab)
    return html.Div([
        html.Div(style={"display": "flex"}, children=[
            html.Div(dcc.Graph(figure=figs[0]), style={"width": "33%"}),
            html.Div(dcc.Graph(figure=figs[2]), style={"width": "33%"}),
            html.Div(dcc.Graph(figure=figs[1]), style={"width": "33%"})
        ]),
        html.Div(style={"display": "flex"}, children=[
            html.Div(dcc.Graph(figure=figs[3]), style={"width": "50%"}),
            html.Div(dcc.Graph(figure=figs[4]), style={"width": "50%"})
        ])
    ])

@callback(
    Output("admin-tab-store", "data"),
    Input("admin-tab", "value")
)
def update_tab(tab):
    return tab



# from dash import dcc, html, callback, Input, Output
# import plotly.graph_objects as go
# from db.database import Database
# import numpy as np
# from modules.image_metrics import calculate_clip_score, calculate_lpips, calculate_brisque_score
# from bert_score import score as bertscore
# import pandas as pd

# # Styling constants
# BG = "#FFEED6"
# GREEN = "#38432E"
# G_HEIGHT = 260


# def get_mode(row):
#     if row['enhanced_prompt'] and row['used_suggestion']:
#         return 'both'
#     elif row['enhanced_prompt']:
#         return 'enhancement'
#     elif row['used_suggestion']:
#         return 'suggestions'
#     return 'noai'

# def filter_by_tab(df, tab):
#     if tab == 'overall':
#         return df
#     df['mode'] = df.apply(get_mode, axis=1)
#     return df[df['mode'] == tab]

# # Real data-backed chart functions

# def clip_line_chart(df):
#     y = []
#     for _, row in df.iterrows():
#         score = calculate_clip_score(row['path'], row['prompt'])
#         if score is not None:
#             y.append(score)
#     if not y:
#         return go.Figure()
#     fig = go.Figure([go.Scatter(y=y, mode='lines+markers', line=dict(color="#7B002C"))])
#     fig.update_layout(title="Image–Prompt Fidelity (CLIPScore)", height=G_HEIGHT,
#                       paper_bgcolor=BG, plot_bgcolor=BG, margin=dict(t=50, b=30, l=40, r=40))
#     return fig

# def lpips_line_chart(df):
#     scores = []
#     for _, row in df.iterrows():
#         if row['depth'] <= 1:
#             continue
#         prev = df[(df['user_id'] == row['user_id']) & (df['depth'] == row['depth'] - 1)]
#         if prev.empty:
#             continue
#         score = calculate_lpips(row['path'], prev.iloc[0]['path'])
#         if score is not None:
#             scores.append(score)
#     fig = go.Figure([go.Scatter(y=scores, mode='lines+markers', line=dict(color="#00008B"))])
#     fig.update_layout(title="Image Novelty (LPIPS)", height=G_HEIGHT,
#                       paper_bgcolor=BG, plot_bgcolor=BG, margin=dict(t=50, b=30, l=40, r=40))
#     return fig

# def brisque_chart(df):
#     scores = [calculate_brisque_score(row['path']) for _, row in df.iterrows() if calculate_brisque_score(row['path']) is not None]
#     fig = go.Figure([go.Scatter(y=scores, mode='lines+markers', line=dict(color="#8B4513"))])
#     fig.update_layout(title="Image Quality (BRISQUE)", height=G_HEIGHT,
#                       paper_bgcolor=BG, plot_bgcolor=BG, margin=dict(t=50, b=30, l=40, r=40))
#     return fig

# def bertscore_line_chart(df):
#     scores = []
#     for _, row in df.iterrows():
#         if row['depth'] <= 1:
#             continue
#         prev = df[(df['user_id'] == row['user_id']) & (df['depth'] == row['depth'] - 1)]
#         if prev.empty:
#             continue
#         try:
#             _, _, F1 = bertscore([row['prompt']], [prev.iloc[0]['prompt']], lang='en', verbose=False)
#             scores.append(1 - F1.item())
#         except:
#             continue
#     fig = go.Figure([go.Scatter(y=scores, mode='lines+markers')])
#     fig.update_layout(title="Prompt Novelty (BERTScore)", height=G_HEIGHT,
#                       paper_bgcolor=BG, plot_bgcolor=BG, margin=dict(t=50, b=30, l=40, r=40))
#     return fig

# def pie_chart(df):
#     df['mode'] = df.apply(get_mode, axis=1)
#     counts = df['mode'].value_counts()
#     fig = go.Figure([go.Pie(labels=counts.index, values=counts.values)])
#     fig.update_layout(title="Interaction Mode Distribution", height=G_HEIGHT,
#                       paper_bgcolor=BG, plot_bgcolor=BG, margin=dict(t=50, b=30, l=40, r=40))
#     return fig

# def create_admin_layout():
#     return html.Div([
#         dcc.Store(id="insight-tab-store", data="overall"),

#         html.Div(style={"backgroundColor": GREEN, "color": "white", "padding": "16px 20px"}, children=[
#             html.Span("AI-D", style={"fontWeight": "bold", "fontSize": "26px", "marginRight": "10px"}),
#             html.Span("|", style={"margin": "0 10px"}),
#             html.Span("Admin Dashboard", style={"fontStyle": "italic", "fontSize": "22px"})
#         ]),

#         html.Div(style={"display": "flex", "padding": "16px 20px 0 20px"}, children=[
#             html.Div("Parameter Insight", style={"width": "25%", "fontSize": "20px", "fontWeight": "600", "color": GREEN}),
#             html.Div("Prompt-Image Insight", style={"width": "50%", "textAlign": "center", "fontSize": "20px", "fontWeight": "600", "color": GREEN}),
#             html.Div("Dialogue history", style={"width": "25%", "textAlign": "right", "fontSize": "20px", "fontWeight": "600", "color": GREEN})
#         ]),

#         html.Div(style={"width": "50%", "marginLeft": "25%", "marginBottom": "10px"}, children=[
#             dcc.Tabs(id="insight-tab", value="overall", children=[
#                 dcc.Tab(label="Overall", value="overall"),
#                 dcc.Tab(label="Suggestions", value="suggestions"),
#                 dcc.Tab(label="Enhancement", value="enhancement"),
#                 dcc.Tab(label="Suggestions & Enhancement", value="both"),
#                 dcc.Tab(label="No AI", value="noai")
#             ])
#         ]),

#         html.Div(id="grid-content")
#     ])

# @callback(
#     Output("insight-tab-store", "data"),
#     Input("insight-tab", "value")
# )
# def update_tab(tab):
#     return tab

# @callback(
#     Output("grid-content", "children"),
#     Input("insight-tab-store", "data")
# )
# def render_figures(tab):
#     with Database() as db:
#         df_images = db.fetch_all_images()
#         df_prompts = db.fetch_all_prompts()
#     df = df_images.merge(df_prompts, left_on="output_prompt_id", right_on="id", suffixes=("_img", ""))
#     df = filter_by_tab(df, tab)

#     return html.Div([
#         html.Div(style={"display": "flex"}, children=[
#             html.Div(dcc.Graph(figure=clip_line_chart(df)), style={"width": "33%"}),
#             html.Div(dcc.Graph(figure=lpips_line_chart(df)), style={"width": "33%"}),
#             html.Div(dcc.Graph(figure=brisque_chart(df)), style={"width": "33%"})
#         ]),
#         html.Div(style={"display": "flex"}, children=[
#             html.Div(dcc.Graph(figure=bertscore_line_chart(df)), style={"width": "33%"}),
#             html.Div(dcc.Graph(figure=pie_chart(df)), style={"width": "33%"}),
#             html.Div("", style={"width": "33%"})
#         ])
#     ])
