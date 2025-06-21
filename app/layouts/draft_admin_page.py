from dash import dcc, callback, Input, Output, State, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from db.database import Database

def create_admin_layout():
    return dbc.Container([
        dbc.Button("Refresh", id="refresh-button", color="primary", className="mb-3"),
        dcc.Dropdown(id="chat-selector", placeholder="Select a Chat", className="mb-3"),
        
        dcc.Graph(id={'type': 'graph', 'index': 'enhancement-usage-rate'}),
        dcc.Graph(id={'type': 'graph', 'index': 'enhancement-amplitude'}),
        dcc.Graph(id={'type': 'graph', 'index': 'enhancement-quality'}),
        dcc.Graph(id={'type': 'graph', 'index': 'suggestion-usage-rate'}),
        dcc.Graph(id={'type': 'graph', 'index': 'suggestion-amplitude'}),
        dcc.Graph(id={'type': 'graph', 'index': 'suggestion-quality'}),
        dcc.Graph(id={'type': 'graph', 'index': 'model-parameters'}),
        dcc.Store(id='dashboard-data'),
    ])

# 🔧 UPDATED: Include app-user-info for scoping chats to logged-in user
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

    db = Database()
    db.connect()
    chats = db.fetch_chats_by_user(user_id)
    db.close()

    chat_options = [
        {"label": f"{row['title']} (ID {row['id']})", "value": row["id"]}
        for _, row in chats.iterrows()
    ]

    return {"data_loaded": True}, chat_options


# ✅ Each metric callback below should now use the selected chat from the dropdown
def create_placeholder_callback(index, title):
    @callback(
        Output({'type': 'graph', 'index': index}, 'figure'),
        Input('dashboard-data', 'data'),
        State('chat-selector', 'value'),
        prevent_initial_call=True
    )
    def update_graph(dashboard_data, chat_id):
        if not dashboard_data or not dashboard_data.get("data_loaded") or not chat_id:
            raise PreventUpdate

        # Here you would query your db using chat_id to get data
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    return update_graph

# Bind all 7 graphs
update_enhancement_usage_rate = create_placeholder_callback('enhancement-usage-rate', "Enhancement Usage Rate")
update_enhancement_amplitude = create_placeholder_callback('enhancement-amplitude', "Enhancement Amplitude")
update_enhancement_quality = create_placeholder_callback('enhancement-quality', "Enhancement Quality")

update_suggestion_usage_rate = create_placeholder_callback('suggestion-usage-rate', "Suggestion Usage Rate")
update_suggestion_amplitude = create_placeholder_callback('suggestion-amplitude', "Suggestion Amplitude")
update_suggestion_quality = create_placeholder_callback('suggestion-quality', "Suggestion Quality")

update_model_parameters = create_placeholder_callback('model-parameters', "Model Parameters")
