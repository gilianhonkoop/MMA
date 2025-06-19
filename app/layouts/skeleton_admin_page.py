from dash import dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

def create_admin_layout():
    return dbc.Container([
        dbc.Button("Refresh", id="refresh-button", color="primary", className="mb-3"),
        dcc.Graph(id={'type': 'graph', 'index': 'enhancement-usage-rate'}),
        dcc.Graph(id={'type': 'graph', 'index': 'enhancement-amplitude'}),
        dcc.Graph(id={'type': 'graph', 'index': 'enhancement-quality'}),
        dcc.Graph(id={'type': 'graph', 'index': 'suggestion-usage-rate'}),
        dcc.Graph(id={'type': 'graph', 'index': 'suggestion-amplitude'}),
        dcc.Graph(id={'type': 'graph', 'index': 'suggestion-quality'}),
        dcc.Graph(id={'type': 'graph', 'index': 'model-parameters'}),
        dcc.Store(id='dashboard-data'),
    ])

@callback(
    Output('dashboard-data', 'data'),
    Input('refresh-button', 'n_clicks')
)
def update_dashboard_data(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    
    return {
        "data_loaded": True,
    }

@callback(
    Output({'type': 'graph', 'index': 'enhancement-usage-rate'}, 'figure'),
    Input('dashboard-data', 'data')
)
def update_enhancement_usage_rate(dashboard_data):
    if dashboard_data is None or not dashboard_data.get("data_loaded"):
        raise PreventUpdate
    
    fig = go.Figure()
    fig.update_layout(title="Rate of use")
    
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
    
    fig = go.Figure()
    fig.update_layout(title="Amplitude of prompt-image change")
    
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