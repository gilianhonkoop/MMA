import dash
from dash import html, dcc, Input, Output, callback, State
from dash.dependencies import Input, Output, ALL, MATCH
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from db.database import Database
from db.recompute_lpips import recompute_lpips_for_chat
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from matplotlib import cm
from sklearn.feature_extraction.text import TfidfVectorizer
import base64
import io
import seaborn as sns


# In order to filter line_chart_change_amplitude and update_keywords_wordcloudby
# by the modes, we implement get_mode and filter_by_mode
def get_mode(row):
    if row['is_enhanced'] and row['used_suggestion']:
        return 'both'
    elif row['is_enhanced'] and not row['used_suggestion']:
        return 'enhancement'
    elif row['used_suggestion'] and not row['is_enhanced']:
        return 'suggestions'
    return 'noai'

def filter_by_mode(df, mode):
    if mode == 'overall':
        return df
    df = df.copy()
    df['used_suggestion'] = df['used_suggestion'].astype(bool)
    df['is_enhanced'] = df['is_enhanced'].astype(bool)
    df['interaction_mode'] = df.apply(get_mode, axis=1)
    return df[df['interaction_mode'] == mode]

# Need to be able to filter by chat or just by user in order to have a "ALL" chat
def filter_by_chat_or_user(df, chat_id, user_id):
    if chat_id == "ALL":
        return df[df["user_id"] == user_id]
    return df[df["chat_id"] == chat_id]

# Recompute LPIPS for all user's chats
def recompute_lpips_for_user(user_id):
    with Database() as db:
        chats = db.fetch_chats_by_user(user_id)
    for chat_id in chats['id'].tolist():
        recompute_lpips_for_chat(chat_id)

# Theme colours 
green_palette = ["#132a13","#31572c","#4f772d","#90a955","#ecf39e"]
pie_chart_palette = ["#adb5bd", "#6c757d","#495057","#f3f3f3"]
GREEN = green_palette[0]
BG = "rgba(177, 183, 143, 0.3)"
Wordcloud_BG = "#e9edd8"
GD = green_palette[0]
GR = "#bc4749"
G_HEIGHT = 350

def create_admin_layout():
    return html.Div([
        dcc.Store(id="insight-tab-store", data="overall"),

        html.Div(style={"display": "flex", "padding": "30px 50px 0 30px"}, children=[
            # COLUMN 1 - Dialogue History
            html.Div([
                html.Div("Dialogue history", style={"fontSize": "30px", "fontWeight": "600", "color": GREEN}),
                dbc.Button("Refresh", id="refresh-button", color="primary", className="mb-3", style={"marginTop": "10px", 'backgroundColor': green_palette[1]}),
                dcc.Dropdown(id="chat-selector", placeholder="Select a Chat", className="mb-3", style={"marginTop": "5px"})
            ], style={"width": "20%"}),

            # COLUMN 2 - Metrics 
            html.Div(id="column-2", style={"width": "37%", "display": "none", "flexDirection": "column", "marginLeft": "3%"}, children=[
                # Header: Dialogue
                html.Div("Dialogue", style={
                    "fontSize": "30px", "fontWeight": "600", "color": GREEN, "marginBottom": "0px", "width":"48%"
                }),
                # ROW 1 - Spheres and pie chart side-by-side
                html.Div(style={"display": "flex", "marginBottom": "10px"}, children=[
                    html.Div(id="dialogue-statistics", style={
                        "width": "48%", "display": "flex", "justifyContent": "center", "alignItems": "center"
                    }),
                    html.Div(dcc.Graph(id="utility-pie", config={"displayModeBar": False}), style={"width": "48%", "marginLeft": "2%"})
                ]),

                # ROW 2 - Bar chart
                html.Div(dcc.Graph(id='functionality-bar'), style={"width": "100%", "marginTop": "20px"}),

                # ROW 3 - Prompt & Image Guidance
                html.Div(dcc.Graph(id={'type': 'graph', 'index': 'model-parameters'}, config={"displayModeBar": False}),
                        style={"width": "100%", "marginTop": "20px"})
            ]),

            # COLUMN 3 - Tabs, Wordcloud, Amplitude chart 
            html.Div(id="column-3", style={"width": "37%", "display": "none", "flexDirection": "column", "marginLeft": "3%"}, children=[
                # Tabs header
                html.Div([
                    html.Div("Dialogue features", style={
                        "fontSize": "30px", "fontWeight": "600", "color": GREEN, "marginBottom": "10px"
                    }),
                    dcc.Tabs(id="insight-tab", value="overall", children=[
                        dcc.Tab(label="Overall", value="overall"),
                        dcc.Tab(label="Suggestions", value="suggestions"),
                        dcc.Tab(label="Enhancement", value="enhancement"),
                        dcc.Tab(label="Suggestions & Enhancement", value="both"),
                        dcc.Tab(label="No AI", value="noai"),
                    ], style={"borderBottom": "2px solid #adb5bd"})
                ]),
                
                # A little line as a separator 
                html.Div(style={
                    "height": "2px",
                    "backgroundColor": "#e9edd8",  
                    "marginTop": "115px", # 48pt for full screen 
                    "marginBottom": "13px",
                    "marginLeft": "0px",
                    "width": "100%" 
                }),

                # Wordcloud mode selector
                html.Div([
                    dcc.RadioItems(
                        id="wordcloud-mode",
                        options=[
                            {"label": "Raw Frequency", "value": "frequency"},
                            {"label": "Color by Depth", "value": "depth"},
                            {"label": "TF-IDF", "value": "tfidf"}
                        ],
                        value="frequency",
                        inline=True,
                        labelStyle={"marginRight": "15px", "fontSize": "18px"},
                        className="mb-3",
                        style={"marginTop": "5px"}
                    )
                ]),

                # Wordcloud image
                html.Div(html.Img(id="wordcloud-img", style={
                    "width": "100%", "marginTop": "5px"
                })),

                # Prompt Novelty and Image Change
                html.Div(dcc.Graph(id={'type': 'graph', 'index': 'bert-lpips-amplitude'},
                                   config={"displayModeBar": False}),
                         style={"width": "100%", "marginTop": "20px"})
            ])
        ])
    ])

@callback(
    [Output("column-2", "style"), Output("column-3", "style")],
    Input("chat-selector", "value"),
    prevent_initial_call=True
)
def toggle_columns_visibility(chat_id):
    if chat_id is None:
        column_2_style = {"width": "37%", "display": "none", "flexDirection": "column", "marginLeft": "3%"}
        column_3_style = {"width": "37%", "display": "none", "flexDirection": "column", "marginLeft": "3%"}
    else:
        column_2_style = {"width": "37%", "display": "flex", "flexDirection": "column", "marginLeft": "3%"}
        column_3_style = {"width": "37%", "display": "flex", "flexDirection": "column", "marginLeft": "3%"}
    
    return column_2_style, column_3_style


@callback(
    Output("insight-tab-store", "data"),
    Input("insight-tab", "value"),
    prevent_initial_call=True
)
def store_selected_tab(tab_value):
    return tab_value

@callback(
    Output("grid-content", "children"),
    Input("insight-tab-store", "data")
)

def render_figures(tab):
    return html.Div([
        html.Div(style={"display": "flex", "padding": "0px 0px 0 0px"}, children=[
            html.Div("", style={
                "width": "20%", "textAlign":"left", "marginLeft": "30px", 
                "fontSize": "30px", "fontWeight": "600", "color": GREEN
            }), 
            
        ]),
        html.Div(style={"display": "flex", "marginTop": "40px"}, children=[
            html.Div([
                
                dcc.RadioItems(
                    id="wordcloud-mode",
                    options=[
                        {"label": "Raw Frequency", "value": "frequency"},
                        {"label": "Color by Depth", "value": "depth"},
                        {"label": "TF-IDF", "value": "tfidf"}
                    ],
                    value="frequency",
                    inline=True,
                    labelStyle={"marginRight": "15px", "fontSize": "18px"},
                    className="mb-3",
                    style={"marginTop": "30px"}
                ),
                html.Div([
                    html.Img(id="wordcloud-img", style={
                        "width": "100%", "display": "block", "margin": "0px auto"
                    })
                ], style={"width": "50%", "marginLeft": "30px", "marginTop": "20px"})
            ], style={"width": "20%", "marginLeft": "3%"}),
        ])
    ])


### STATIC ELEMENTS 
# Prompt Novelty
def calculate_average_prompt_novelty(chat_id, user_id):
    if chat_id is None:
        raise PreventUpdate

    with Database() as db:
        df = db.fetch_all_bertscore_metrics()

    df = filter_by_chat_or_user(df, chat_id, user_id)

    if df.empty:
        return 0.0

    return round(df['bert_novelty'].mean(), 3)

# Image Novelty/Change
def calculate_average_image_change(chat_id, user_id):
    if chat_id is None:
        raise PreventUpdate

    with Database() as db:
        df = db.fetch_all_lpips_metrics()

    df = filter_by_chat_or_user(df, chat_id, user_id)
    if df.empty:
        return 0.0

    return round(df['lpips'].mean(), 3)

# Prompt Length
def calculate_average_prompt_length(chat_id, user_id):
    if chat_id is None:
        raise PreventUpdate

    with Database() as db:
        prompts = db.fetch_prompts_by_user(user_id)

    prompts = filter_by_chat_or_user(prompts, chat_id, user_id)

    if prompts.empty or 'text' not in prompts.columns:
        return 0.0

    prompt_lengths = prompts['text'].dropna().apply(lambda t: len(t.split()))
    return round(prompt_lengths.mean(), 1)


def calculate_summary_statistics(chat_id, user_id):
    return {
        "avg_prompt_novelty": calculate_average_prompt_novelty(chat_id, user_id),
        "avg_image_change": calculate_average_image_change(chat_id, user_id),
        "avg_prompt_length": calculate_average_prompt_length(chat_id, user_id)
    }


### Pie Chart Functionality
# Get the percentages for each of the functionalities
def get_functionality_percentages(chat_id, user_id):
    if chat_id is None:
        raise PreventUpdate

    with Database() as db:
        df = db.fetch_all_functionality_metrics()

    df = filter_by_chat_or_user(df, chat_id, user_id)
    if df.empty:
        raise PreventUpdate

    if chat_id == "ALL":
        with Database() as db:
            prompt_counts = db.fetch_prompts_by_user(user_id)[['chat_id']]
        prompt_counts = prompt_counts.groupby('chat_id').size().rename('num_prompts').reset_index()
        df = df.merge(prompt_counts, on='chat_id', how='left')
        df = df.fillna({'num_prompts': 0})
        total = df['num_prompts'].sum()

        if total == 0:
            return ("0", "0", "0", "0")

        def weighted_avg(col):
            return (df[col] * df['num_prompts']).sum() / total

        return (
            str(round(weighted_avg('used_suggestion_pct'), 1)),
            str(round(weighted_avg('used_enhancement_pct'), 1)),
            str(round(weighted_avg('used_both_pct'), 1)),
            str(round(weighted_avg('no_ai_pct'), 1)),)
    
    elif chat_id != "ALL":
        row = df.iloc[0]
        return (
            str(round(row['used_suggestion_pct'], 1)),
            str(round(row['used_enhancement_pct'], 1)),
            str(round(row['used_both_pct'], 1)),
            str(round(row['no_ai_pct'], 1)))


# View the Percentages of each Functionality as a Pie Chart
def pie_chart_utility(chat_id, user_id):
    values = get_functionality_percentages(chat_id, user_id)
    labels = ["Suggestions", "Enhancement", "S + E", "No AI"]
    colors = pie_chart_palette
    fig = go.Figure(go.Pie(labels=labels, values=values,
                           marker_colors=colors, textinfo="label+percent"))
    fig.update_layout(
        height=165,
        margin=dict(t=10, b=10, l=5, r=5),
        font=dict(family="sans-serif", size=12),
        legend=dict(font=dict(size=12)),
        showlegend=False
    )
    return fig


### The Keyword wordcloud
def update_keywords_wordcloud(chat_id, user_id, mode, tab="overall"):
    if chat_id is None:
        raise PreventUpdate

    with Database() as db:
        df = db.fetch_all_prompt_word_metrics()
        # prompts_df = db.fetch_prompts_by_chat(chat_id)
        if chat_id == "ALL":
            prompts_df = db.fetch_prompts_by_user(user_id)
        else:
            prompts_df = db.fetch_prompts_by_chat(chat_id)
    
    df_merge = df.merge(
    prompts_df[['id', 'chat_id', 'used_suggestion', 'is_enhanced']],
    left_on='prompt_id', right_on='id', how='inner')
    df = df_merge.drop(columns=["chat_id_x"], errors="ignore") \
       .rename(columns={"chat_id_y": "chat_id"})
    df = filter_by_chat_or_user(df, chat_id, user_id)
    df = df.dropna(subset=['relevant_words', 'depth'])
    df['used_suggestion'] = df['used_suggestion'].astype(bool)
    df['is_enhanced'] = df['is_enhanced'].astype(bool)
    df['interaction_mode'] = df.apply(get_mode, axis=1)
    if tab != "overall":
        df = df[df['interaction_mode'] == tab]
    df['relevant_words'] = df['relevant_words'].apply(lambda s: s.split(','))

    if df.empty:
        raise PreventUpdate
    
    # Shared colormap and background
    cubehelix_cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)

    # RAW FREQUENCY: most common words regardless of time
    if mode == 'frequency':
        all_words = pd.Series([w for words in df['relevant_words'] for w in words])
        word_freq = all_words.value_counts().to_dict()
        wc = WordCloud(
            width=800, 
            height=G_HEIGHT, 
            background_color=Wordcloud_BG,
            colormap=cubehelix_cmap
            ).generate_from_frequencies(word_freq)

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
            #colormap = cm.get_cmap('coolwarm')
            def color_func(word, **kwargs):
                norm = (avg_depths.get(word, min_d) - min_d) / (max_d - min_d + 1e-5)
                r, g, b, _ = [int(c * 255) for c in cubehelix_cmap(norm)]
                return f"rgb({r},{g},{b})"
            return color_func

        wc = WordCloud(
                width=800, 
                height=G_HEIGHT, 
                background_color=Wordcloud_BG,
                colormap=cubehelix_cmap
                ).generate_from_frequencies(word_freq)
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
            wc = WordCloud(
                width=800, 
                height=G_HEIGHT, 
                background_color=Wordcloud_BG,
                colormap=cubehelix_cmap
                ).generate_from_frequencies(tfidf_scores)
        else:
            raise PreventUpdate

    else:
        raise PreventUpdate

    img_buffer = io.BytesIO()
    wc.to_image().save(img_buffer, format="PNG")
    encoded_image = base64.b64encode(img_buffer.getvalue()).decode()

    return f"data:image/png;base64,{encoded_image}"


# Creates the Spheres which are needed for visualizing prompt novelty and image change
@callback(
    Output("dialogue-statistics", "children"),
    Input("chat-selector", "value"),
    State('app-user-info', 'data'),
    prevent_initial_call=True
)

def update_statistics_display(chat_id, user_info):
    if chat_id is None:
        raise PreventUpdate
    
    user_id = user_info.get("user_id")
    
    # Recompute LPIPS records before visualizing anything
    if chat_id == "ALL":
        recompute_lpips_for_user(user_id)
    else:
        recompute_lpips_for_chat(chat_id)

    # stats = calculate_summary_statistics(chat_id)
    stats = calculate_summary_statistics(chat_id, user_id)

    def create_sphere(label, value, color, margin_right=False):
        return html.Div([
            html.Div(f"{value}", style={
                "fontSize": "24px",
                "fontWeight": "bold",
                "color": "white" if color != "#FFEED6" else "#000",
                "textAlign": "center",
                "marginTop": "20%"
            }),
            html.Div(label, style={
                "fontSize": "14px",
                "color": "white" if color != "#FFEED6" else "#000",
                "textAlign": "center",
                "marginTop": "5px"
            })
            ], style={
                "width": "100px",
                "height": "100px",
                "borderRadius": "50%",
                "backgroundColor": color,
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "justifyContent": "center",
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.2)",
                "padding": "10px",
                "marginRight": "30px" if margin_right else "0"
            })

    return [
            create_sphere("Avg Prompt Novelty", stats["avg_prompt_novelty"], GD, margin_right=True),
            create_sphere("Avg Image Change", stats["avg_image_change"], GR, margin_right=True)
            #create_sphere("Avg Prompt Length", stats["avg_prompt_length"], "#FFEED6", margin_right=False)
        ]


# This changes the tabs when clicked on 
def update_tab_labels(chat_id, user_id):
    if chat_id is None:
        raise PreventUpdate

    perc_sugg, perc_enh, perc_sugg_enh, perc_no_ai = get_functionality_percentages(chat_id, user_id)

    common_tab_style = {
    "fontSize": "14px",
    "padding": "12px 16px",
    "height": "auto",
    "whiteSpace": "normal",  # allows wrapping
    "overflow": "visible",   # avoids clipping
    "textAlign": "center"}

    return [
    dcc.Tab(label="Overall", value="overall", style=common_tab_style, selected_style=common_tab_style),
    dcc.Tab(label="Suggestions", value="suggestions", style=common_tab_style, selected_style=common_tab_style),
    dcc.Tab(label="Enhancement", value="enhancement", style=common_tab_style, selected_style=common_tab_style),
    dcc.Tab(label="Suggestions & Enhancement", value="both", style=common_tab_style, selected_style=common_tab_style),
    dcc.Tab(label="No AI", value="noai", style=common_tab_style, selected_style=common_tab_style),
    ]

# This returns the Prompt/Image Change Amplitude Chart
def update_functionality_bar(chat_id, user_id):
    if chat_id is None:
        return go.Figure()
    
    # Recompute LPIPS records before visualizing anything
    if chat_id == "ALL":
        recompute_lpips_for_user(user_id)
    else:
        recompute_lpips_for_chat(chat_id)


    with Database() as db:
        prompts = db.fetch_prompts_by_user(user_id)
        bert = db.fetch_all_bertscore_metrics()
        lpips = db.fetch_all_lpips_metrics()

    # Filter each dataframe down to what we need
    prompts = filter_by_chat_or_user(prompts, chat_id, user_id)
    bert = filter_by_chat_or_user(bert, chat_id, user_id)
    lpips = filter_by_chat_or_user(lpips, chat_id, user_id)

    # Preprocessing
    prompts["used_suggestion"] = prompts["used_suggestion"].astype(bool)
    prompts["is_enhanced"] = prompts["is_enhanced"].astype(bool)

    def get_func(row):
        if row["used_suggestion"] and not row["is_enhanced"]:
            return "S"
        elif not row["used_suggestion"] and row["is_enhanced"]:
            return "E"
        elif row["used_suggestion"] and row["is_enhanced"]:
            return "S+E"
        else:
            return "No AI"

    prompts["functionality"] = prompts.apply(get_func, axis=1)

    # === Merge with BERT ===
    merged_bert = prompts.merge(bert, left_on="id", right_on="prompt_id", how="inner")
    bert_group = merged_bert.groupby("functionality")["bert_novelty"].mean()

    # === Merge with LPIPS by user_id, chat_id, and depth ===
    merged_lpips = prompts.merge(lpips, on=["user_id", "chat_id", "depth"], how="inner")
    lpips_group = merged_lpips.groupby("functionality")["lpips"].mean()

    # === Combine & Plot ===
    df = pd.DataFrame({
        "BERTScore": bert_group,
        "LPIPS": lpips_group
    }).fillna(0).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["functionality"], y=df["BERTScore"],
        name="ΔPrompt (BERTScore)", marker_color=GD
    ))
    fig.add_trace(go.Bar(
        x=df["functionality"], y=df["LPIPS"],
        name="ΔPrompt (LPIPS)", marker_color=GR
    ))

    fig.update_layout(
            title={"text": "Mean Prompt Novelty and Image Change<br>per dialogue feature",
                   "x": 0.5,
                   "xanchor": "center",
                   "font": dict(size=16)},
        xaxis_title="Feature",
        yaxis_title="Score",
        barmode="group",
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        margin=dict(t=50, b=30, l=40, r=40),
        font=dict(family="sans-serif", size=14), 
        legend=dict(font=dict(size=12)),
        height=G_HEIGHT
    )

    return fig

# POPULATE WORDCLOUD 
@callback(
    Output("insight-tab", "children"),
    Output("wordcloud-img", "src"),
    Input("chat-selector", "value"),
    Input("wordcloud-mode", "value"),
    Input("insight-tab-store", "data"),
    State("app-user-info", "data"),
    prevent_initial_call=True
)

def render_wordcloud(chat_id, mode, tab, user_info):
    user_id = user_info.get("user_id")
    return update_tab_labels(chat_id, user_id), update_keywords_wordcloud(chat_id, user_id, mode, tab)

# POPULATE FUNCTIONALITY BAR CHART 
@callback(
    Output("functionality-bar", "figure"),
    Input("chat-selector", "value"),
    State("app-user-info", "data"),
    prevent_initial_call=True
)
def render_functionality_bar(chat_id, user_info):
    user_id = user_info.get("user_id")
    return update_functionality_bar(chat_id, user_id)


# POPULATE PIE CHART 
@callback(
    Output("utility-pie", "figure"),
    Input("chat-selector", "value"),
    State('app-user-info', 'data'),
    prevent_initial_call=True
)
def update_pie_chart(chat_id, user_info):
    user_id = user_info.get("user_id")
    return pie_chart_utility(chat_id, user_id)

@callback(
    Output("chat-selector", "options"),
    Input("refresh-button", "n_clicks"),
    State('app-user-info', 'data'),
)
def populate_chat_dropdown(n_clicks, user_info):
    user_id = user_info.get("user_id")
    if user_id is None:
        raise PreventUpdate

    with Database() as db:
        chats = db.fetch_chats_by_user(user_id)
        
    # options = [{"label": f"{chat['id']} — {chat['title']}", "value": chat['id']} for _, chat in chats.iterrows()]
    options = [{"label": "All My Chats", "value": "ALL"}] + [
        {"label": f"{chat['id']} — {chat['title']}", "value": chat['id']} for _, chat in chats.iterrows()]
    return options
######### 

def line_chart_guidances(chat_id, user_id):
    if chat_id is None:
        raise PreventUpdate

    with Database() as db:
        df = db.fetch_all_guidance_metrics()

    df = filter_by_chat_or_user(df, chat_id, user_id).sort_values(by='depth')

    fig = go.Figure()
    
    if chat_id == "ALL":
        unique_chats = sorted(df['chat_id'].unique())
        
        def generate_color_variations(base_color, count):
            variations = [GR, GREEN, "#3357FF", "#FF33F5", "#F5FF33", 
                         "#33FFF5", "#FF8C33", "#8C33FF", "#33FF8C", "#FF3333",
                         "#3333FF", "#FFFF33", "#FF33FF", "#33FFFF", "#8CFF33",
                         "#FF338C", "#338CFF", "#8C8CFF", "#FF8CFF", "#8CFFFF"]
            return variations[:count] if count <= len(variations) else variations * ((count // len(variations)) + 1)
        
        colors = generate_color_variations(None, len(unique_chats))
        
        for i, chat_id_val in enumerate(unique_chats):
            chat_df = df[df['chat_id'] == chat_id_val]
            fig.add_trace(
                go.Scatter(
                    x=chat_df['depth'], 
                    y=chat_df['prompt_guidance'], 
                    mode='lines+markers', 
                    name=f'Prompt Guidance (Chat {chat_id_val})', 
                    line=dict(color=colors[i], width=2, dash='solid'),
                    marker=dict(symbol="circle", size=6),
                    legendgroup="prompt",
                    showlegend=True
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=chat_df['depth'], 
                    y=chat_df['image_guidance'], 
                    mode='lines+markers', 
                    name=f'Image Guidance (Chat {chat_id_val})', 
                    line=dict(color=colors[i], width=2, dash='dash'),
                    marker=dict(symbol="square", size=6),
                    legendgroup="image",
                    showlegend=True
                )
            )
    else:
        fig.add_trace(
                go.Scatter(
                            x=df['depth'], 
                            y=df['prompt_guidance'], 
                            mode='lines+markers', 
                            name='Prompt Guidance', 
                            line=dict(color=GD, width=2),
                            marker=dict(symbol="circle", size=6)
                        )
                    )
        fig.add_trace(
                go.Scatter(
                            x=df['depth'], 
                            y=df['image_guidance'], 
                            mode='lines+markers', 
                            name='Image Guidance', 
                            line=dict(color=GR, width=2),
                            marker=dict(symbol="circle", size=6)
                         )
                    )
    
    fig.update_layout(
                title="Prompt and Image Guidance over Generations", 
                height=G_HEIGHT,
                font=dict(family="sans-serif", size=14), 
                legend=dict(font=dict(size=12)),
                xaxis=dict(
                    title="Generation",
                    tickmode="linear",
                    tick0=1,
                    dtick=1
                ),
                yaxis_title="Guidance Value",  
                paper_bgcolor=BG,
                plot_bgcolor=BG,
                margin=dict(t=50, b=30, l=40, r=40),
                title_font_size=16
                )
    return fig

## Adjust to take tab-dependent values in 
def line_chart_change_amplitude(chat_id, user_id, tab="overall"):
    if chat_id is None:
        raise PreventUpdate

    with Database() as db:
        if chat_id == "ALL":
            prompts = db.fetch_prompts_by_user(user_id)
            bert_df = db.fetch_all_bertscore_metrics()
            bert_df = bert_df[bert_df['user_id'] == user_id]
            lpips_df = db.fetch_all_lpips_metrics()
            lpips_df = lpips_df[lpips_df['user_id'] == user_id]
        else:
            prompts = db.fetch_prompts_by_chat(chat_id)
            bert_df = db.fetch_all_bertscore_metrics()
            lpips_df = db.fetch_all_lpips_metrics()

    # prompts = prompts[prompts['chat_id'] == chat_id]
    prompts = filter_by_chat_or_user(prompts, chat_id, user_id)
    prompts = filter_by_mode(prompts, tab)

    valid_prompt_ids = prompts["id"].tolist()
    valid_depths = prompts[["chat_id", "depth"]].values.tolist() if chat_id == "ALL" else prompts["depth"].tolist()

    if chat_id == "ALL":
        bert_df = bert_df[(bert_df["depth"] > 1)]
        lpips_df = lpips_df[(lpips_df["depth"] > 1)]
        
        valid_chat_depth = set((row[0], row[1]) for row in valid_depths)
        bert_df = bert_df[bert_df.apply(lambda x: (x['chat_id'], x['depth']) in valid_chat_depth, axis=1)]
        lpips_df = lpips_df[lpips_df.apply(lambda x: (x['chat_id'], x['depth']) in valid_chat_depth, axis=1)]
    else:
        bert_df = bert_df[(bert_df["chat_id"] == chat_id) & (bert_df["depth"] > 1)]
        lpips_df = lpips_df[(lpips_df["chat_id"] == chat_id) & (lpips_df["depth"] > 1)]
        bert_df = bert_df[bert_df["prompt_id"].isin(valid_prompt_ids)]
        lpips_df = lpips_df[lpips_df["depth"].isin(valid_depths)]

    # Only keep depth >= 2
    # since depth = 1 corresponds to the initial prompt and image, metrics like
    # bert_novelty and lpips (which are comparative) aren't meaningful at that point.
    bert_df = bert_df[bert_df["depth"] > 1].sort_values(by='depth')
    lpips_df = lpips_df[lpips_df["depth"] > 1].sort_values(by='depth')


    fig = go.Figure()
    
    if chat_id == "ALL":
        unique_chats = sorted(bert_df['chat_id'].unique())
        
        def generate_color_variations(base_color, count):
            variations = [GR, GREEN, "#3357FF", "#FF33F5", "#F5FF33", 
                         "#33FFF5", "#FF8C33", "#8C33FF", "#33FF8C", "#FF3333",
                         "#3333FF", "#FFFF33", "#FF33FF", "#33FFFF", "#8CFF33",
                         "#FF338C", "#338CFF", "#8C8CFF", "#FF8CFF", "#8CFFFF"]
            return variations[:count] if count <= len(variations) else variations * ((count // len(variations)) + 1)
        
        colors = generate_color_variations(None, len(unique_chats))
        
        for i, chat_id_val in enumerate(unique_chats):
            chat_bert_df = bert_df[bert_df['chat_id'] == chat_id_val]
            chat_lpips_df = lpips_df[lpips_df['chat_id'] == chat_id_val]
            
            fig.add_trace(
                go.Scatter(
                    x=chat_bert_df['depth'], 
                    y=chat_bert_df['bert_novelty'], 
                    mode='lines+markers', 
                    name=f'BERT Novelty (Chat {chat_id_val})', 
                    line=dict(color=colors[i], width=2, dash='solid'),
                    marker=dict(symbol="circle", size=6),
                    legendgroup="bert",
                    showlegend=True
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=chat_lpips_df['depth'], 
                    y=chat_lpips_df['lpips'], 
                    mode='lines+markers', 
                    name=f'LPIPS (Chat {chat_id_val})', 
                    line=dict(color=colors[i], width=2, dash='dash'),
                    marker=dict(symbol="square", size=6),
                    legendgroup="lpips",
                    showlegend=True
                )
            )
    else:
        fig.add_trace(
                go.Scatter(
                            x=bert_df['depth'], 
                            y=bert_df['bert_novelty'], 
                            mode='lines+markers', 
                            name='BERT Novelty', 
                            line=dict(color=GD, width=2),
                            marker=dict(symbol="circle", size=6)
                        )
                    )
        fig.add_trace(
                go.Scatter(
                            x=lpips_df['depth'], 
                            y=lpips_df['lpips'], 
                            mode='lines+markers', 
                            name='LPIPS', 
                            line=dict(color=GR, width=2),
                            marker=dict(symbol="circle", size=6)
                         )
                    )
    fig.update_layout(
                # title="Prompt Novelty and Image Change Amplitude", 
                title=f"Prompt Novelty and Image Change Amplitude ({tab.title()})",
                height=G_HEIGHT,
                font=dict(family="sans-serif", size=14), 
                legend=dict(font=dict(size=12)),
                xaxis=dict(
                    title="Generation",
                    tickmode="linear",
                    tick0=2,
                    dtick=1
                ),
                yaxis_title="Value",  
                paper_bgcolor=BG,
                plot_bgcolor=BG,
                margin=dict(t=50, b=30, l=40, r=40),
                title_font_size=16,
                )
    return fig

@callback(
    Output({'type': 'graph', 'index': ALL}, 'figure'),
    Input('chat-selector', 'value'),
    Input('insight-tab-store', 'data'),
    State('app-user-info', 'data'),
    prevent_initial_call=True
)
def update_graphs(chat_id, tab, user_info):
    user_id = user_info.get("user_id")
    return (
        line_chart_guidances(chat_id, user_id),
        line_chart_change_amplitude(chat_id, user_id, tab)
    )
