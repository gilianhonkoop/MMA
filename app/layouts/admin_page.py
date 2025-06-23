import dash
from dash import html, dcc, Input, Output, callback, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import base64
import io
import os
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from matplotlib import cm
from sklearn.feature_extraction.text import TfidfVectorizer
from db.database import Database

def create_admin_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Button("Refresh", id="refresh-button", color="primary", className="mb-3"),
                dcc.Dropdown(id="chat-selector", placeholder="Select a Chat", className="mb-3"),
                dcc.RadioItems(
                    id="wordcloud-mode",
                    options=[
                        {"label": "Raw Frequency", "value": "frequency"},
                        {"label": "Color by Depth", "value": "depth"},
                        {"label": "TF-IDF", "value": "tfidf"}
                    ],
                    value="frequency",
                    inline=True,
                    labelStyle={"marginRight": "15px"},
                    className="mb-3"
                ),
            ], width=4),

            dbc.Col([
                dcc.Graph(id={'type': 'graph', 'index': 'model-parameters'}, config={'displayModeBar': False}, style={"height": "280px"}),
                dcc.Graph(id={'type': 'graph', 'index': 'bert-lpips-amplitude'}, config={'displayModeBar': False}, style={"height": "280px"})
            ], width=4),

            dbc.Col([
                dcc.Graph(id={'type': 'graph', 'index': 'functionality-usage-rate'}, config={'displayModeBar': False}, style={"height": "280px"}),
                html.Img(id={'type': 'graph', 'index': 'keywords-evolution'}, style={'width': '100%', 'height': 'auto'}),
                dcc.Graph(id="depth-legend", style={"display": "none", "height": "60px"})
            ], width=4)
        ], className="gy-3"),

        dcc.Store(id='dashboard-data')
    ], fluid=True)

@callback(
    [Output('dashboard-data', 'data'),
     Output('chat-selector', 'options')],
    Input('refresh-button', 'n_clicks'),
    State('app-user-info', 'data'),
    prevent_initial_call=True
)
def update_dashboard_data(n_clicks, user_info):
    if n_clicks is None or not user_info:
        raise PreventUpdate

    user_id = user_info.get("user_id")
    if user_id is None:
        raise PreventUpdate

    with Database() as db:
        chats = db.fetch_chats_by_user(user_id)

    chat_options = [
        {"label": f"{row['title']} (ID {row['id']})", "value": row["id"]}
        for _, row in chats.iterrows()
    ]

    return {"data_loaded": True}, chat_options

@callback(
    Output({'type': 'graph', 'index': 'model-parameters'}, 'figure'),
    Input('chat-selector', 'value'),
    prevent_initial_call=True
)
def update_guidance_plot(chat_id):
    if chat_id is None:
        raise PreventUpdate

    with Database() as db:
        df = db.fetch_all_guidance_metrics()

    df = df[df['chat_id'] == chat_id].sort_values(by='depth')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['depth'], y=df['prompt_guidance'], mode='lines+markers', name='Prompt Guidance'))
    fig.add_trace(go.Scatter(x=df['depth'], y=df['image_guidance'], mode='lines+markers', name='Image Guidance'))
    fig.update_layout(title="Prompt and Image Guidance over Generations", xaxis_title="Generation", yaxis_title="Guidance Value")

    return fig

@callback(
    Output({'type': 'graph', 'index': 'bert-lpips-amplitude'}, 'figure'),
    Input('chat-selector', 'value'),
    prevent_initial_call=True
)
def update_novelty_amplitude(chat_id):
    if chat_id is None:
        raise PreventUpdate

    with Database() as db:
        bert_df = db.fetch_all_bertscore_metrics()
        lpips_df = db.fetch_all_lpips_metrics()

    bert_df = bert_df[bert_df['chat_id'] == chat_id].sort_values(by='depth')
    lpips_df = lpips_df[lpips_df['chat_id'] == chat_id].sort_values(by='depth')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bert_df['depth'], y=bert_df['bert_novelty'], mode='lines+markers', name='BERT Novelty'))
    fig.add_trace(go.Scatter(x=lpips_df['depth'], y=lpips_df['lpips'], mode='lines+markers', name='LPIPS'))
    fig.update_layout(title="Prompt Novelty and Image Change Amplitude", xaxis_title="Generation", yaxis_title="Value")

    return fig

@callback(
    Output({'type': 'graph', 'index': 'functionality-usage-rate'}, 'figure'),
    Input('chat-selector', 'value'),
    prevent_initial_call=True
)
def update_functionality_usage(chat_id):
    if chat_id is None:
        raise PreventUpdate

    with Database() as db:
        df = db.fetch_all_functionality_metrics()

    df = df[df['chat_id'] == chat_id]
    if df.empty:
        raise PreventUpdate

    avg_vals = df[['used_suggestion_pct', 'used_enhancement_pct', 'used_both_pct', 'no_ai_pct']].mean()

    fig = go.Figure(data=[
        go.Pie(
            labels=avg_vals.index,
            values=avg_vals.values,
            hole=0.4,
            textinfo='label+percent'
        )
    ])
    fig.update_layout(title="Functionality Usage Rate (%)")

    return fig

@callback(
    Output({'type': 'graph', 'index': 'keywords-evolution'}, 'src'),
    Input('chat-selector', 'value'),
    Input('wordcloud-mode', 'value'),
    prevent_initial_call=True
)
def update_keywords_wordcloud(chat_id, mode):
    if chat_id is None:
        raise PreventUpdate

    with Database() as db:
        df = db.fetch_all_prompt_word_metrics()
    df = df[df['chat_id'] == chat_id].dropna(subset=['relevant_words', 'depth'])
    # If any row contains an unexpected format (e.g., not a string), this will raise an error. 
    # need to run to check if error is raised
    df['relevant_words'] = df['relevant_words'].apply(lambda s: s.split(','))

    if df.empty:
        raise PreventUpdate

    # RAW FREQUENCY: most common words regardless of time
    if mode == 'frequency':
        all_words = pd.Series([w for words in df['relevant_words'] for w in words])
        word_freq = all_words.value_counts().to_dict()
        wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)

    # COLOR BY DEPTH:  words colored based on when they appeared
    elif mode == 'depth':
        word_depths = {}
        word_counts = {}
        for _, row in df.iterrows():
            for word in row['relevant_words']:
                word_depths[word] = word_depths.get(word, 0) + row['depth']
                word_counts[word] = word_counts.get(word, 0) + 1
        avg_depths = {w: word_depths[w] / word_counts[w] for w in word_depths}
        word_freq = word_counts

        def get_color_func(avg_depths):
            min_d, max_d = min(avg_depths.values()), max(avg_depths.values())
            colormap = cm.get_cmap('coolwarm')
            def color_func(word, **kwargs):
                norm = (avg_depths.get(word, min_d) - min_d) / (max_d - min_d + 1e-5)
                r, g, b, _ = [int(c * 255) for c in colormap(norm)]
                return f"rgb({r},{g},{b})"
            return color_func

        wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
        wc.recolor(color_func=get_color_func(avg_depths))

    # TF-IDF: what makes this chat lexically unique
    elif mode == 'tfidf':
        with Database() as db:
            all_df = db.fetch_all_prompt_word_metrics()
        all_df = all_df.dropna(subset=['relevant_words'])
        all_df['relevant_words'] = all_df['relevant_words'].apply(lambda s: s.split(','))
        chats_grouped = all_df.groupby('chat_id')['relevant_words'].apply(lambda lst: [" ".join(words) for words in lst])
        chat_docs = chats_grouped.apply(lambda lst: " ".join(lst)).tolist()
        chat_ids = chats_grouped.index.tolist()

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(chat_docs)

        if chat_id in chat_ids:
            chat_idx = chat_ids.index(chat_id)
            words = vectorizer.get_feature_names_out()
            scores = tfidf_matrix[chat_idx].toarray().flatten()
            tfidf_scores = {word: scores[i] for i, word in enumerate(words) if scores[i] > 0}
            wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(tfidf_scores)
        else:
            raise PreventUpdate

    else:
        raise PreventUpdate

    img_buffer = io.BytesIO()
    wc.to_image().save(img_buffer, format="PNG")
    encoded_image = base64.b64encode(img_buffer.getvalue()).decode()

    return f"data:image/png;base64,{encoded_image}"

@callback(
    Output("depth-legend", "figure"),
    Output("depth-legend", "style"),
    Input("wordcloud-mode", "value"),
    Input("chat-selector", "value"),
    prevent_initial_call=True
)
def update_legend(mode, chat_id):
    if mode != "depth" or chat_id is None:
        return {}, {"display": "none"}

    with Database() as db:
        df = db.fetch_all_prompt_word_metrics()
    df = df[df['chat_id'] == chat_id]
    if df.empty or 'depth' not in df.columns:
        return {}, {"display": "none"}

    min_d = int(df['depth'].min())
    max_d = int(df['depth'].max())

    fig = go.Figure(go.Heatmap(
        z=[[min_d, max_d]],
        colorscale="coolwarm",
        showscale=True,
        colorbar=dict(
            orientation="h",
            title="Generation Depth",
            titleside="top",
            xanchor="center",
            x=0.5,
            tickmode='array',
            tickvals=[min_d, max_d],
            ticktext=[f"Early ({min_d})", f"Late ({max_d})"]
        )
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=60
    )

    return fig, {"display": "block", "height": "60px"}
