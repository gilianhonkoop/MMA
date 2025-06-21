from dash import dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from db.database import Database

def create_admin_layout():
    return dbc.Container([
        dbc.Button("Refresh", id="refresh-button", color="primary", className="mb-3"),
        dcc.Dropdown(id="chat-selector", placeholder="Select a Chat", className="mb-3"),
        dcc.Graph(id={'type': 'graph', 'index': 'model-parameters'}),
        dcc.Graph(id={'type': 'graph', 'index': 'bert-lpips-amplitude'}),
        dcc.Graph(id={'type': 'graph', 'index': 'functionality-usage-rate'}),
        dcc.Graph(id={'type': 'graph', 'index': 'keywords-evolution'}),
        dcc.Store(id='dashboard-data'),
    ])

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

    fig = go.Figure(data=[go.Bar(x=avg_vals.index, y=avg_vals.values)])
    fig.update_layout(title="Functionality Usage Rate (%)", xaxis_title="Functionality", yaxis_title="Percentage")

    return fig

@callback(
    Output({'type': 'graph', 'index': 'keywords-evolution'}, 'figure'),
    Input('chat-selector', 'value'),
    prevent_initial_call=True
)
def update_keywords_graph(chat_id):
    if chat_id is None:
        raise PreventUpdate

    with Database() as db:
        df = db.fetch_all_prompt_word_metrics()

    df = df[df['chat_id'] == chat_id].sort_values(by='depth')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['depth'], y=df['word_count'], mode='lines+markers', name='Word Count'))
    fig.update_layout(title="Keyword Count per Prompt over Generations", xaxis_title="Generation", yaxis_title="Word Count")

    return fig