
from dash import dcc, html, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash.exceptions import PreventUpdate

def create_tree_visualization():
    return cyto.Cytoscape(
        id='tree-graph',
        layout={'name': 'dagre', 'rankDir': 'LR', 'rankSep': 150, 'nodeSep': 100},
        style={'width': '100%', 'height': '600px'},
        elements=[],
        autoungrabify=True,
        autounselectify=True,
        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'label': 'data(label)',
                    'background-image': 'data(image)',
                    'background-fit': 'cover',
                    'shape': 'rectangle',
                    'width': '100px',
                    'height': '100px',
                    'border-width': '1px',
                }
            },
            {
                'selector': ':selected',
                'style': {
                    'border-color': 'red',
                    'border-width': '3px'
                }
            },
            {
                'selector': 'node.disabled',
                'style': {
                    'opacity': '0.4',
                    'events': 'no',
                }
            },
            {
                'selector': 'node.selectable',
                'style': {
                    'border-color': 'blue',
                    'border-width': '2px',
                    'border-style': 'dashed'
                }
            }
        ]
    )

def create_image_display(id_prefix, title):
    return html.Div([
        html.H5(title),
        html.Div(
            id=f'{id_prefix}-image-container',
            children=[html.Img(id=f'{id_prefix}-image', style={'width': '100%', 'display': 'none'})],
            style={'minHeight': '200px', 'border': '1px dashed gray', 'textAlign': 'center'}
        ),
    ])

def create_suggestion_button(idx):
    return dbc.Button(
        id=f'suggestion-button-{idx}',
        children="Suggestion",
        color="secondary",
        className="me-2 mb-2",
        style={'display': 'none'}
    )

def create_user_layout():
    return dbc.Container([
        html.H1("Image Generator Interface", className="my-3"),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div(
                        id="tree-graph-container",
                        children=create_tree_visualization(),
                        style={"display": "none"}
                    ),
                    
                    html.Div([
                        html.H3("Start by uploading an image", className="mb-4 text-center"),
                        dcc.Upload(
                            id='image-upload',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select an Image')
                            ]),
                            style={
                                'width': '100%',
                                'height': '200px',
                                'lineHeight': '200px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '20px 0'
                            },
                            multiple=False
                        ),
                    ], id="upload-container"),
                ], className="mb-4"),
            ], width=12),
        ]),
        
        dbc.Row([
            dbc.Col(
                html.Div([
                    create_suggestion_button(1),
                    create_suggestion_button(2),
                    create_suggestion_button(3),
                ], id='suggestion-container', style={'display': 'none'}),
            width=12, className="text-center mb-3")
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.InputGroup([
                    dbc.Input(id='prompt-input', placeholder="Enter your prompt here...", type="text"),
                    dbc.InputGroupText(
                        dbc.Checkbox(
                            id='ai-enhancement-checkbox',
                            label="Use AI",
                            value=False,
                        )
                    ),
                    dbc.Button("Generate Images", id="submit-button", color="primary", disabled=True),
                ], className="mb-3"),
                
                dbc.Spinner(html.Div(id="loading-output")),
            ], width={"size": 10, "offset": 1}),
        ]),
        
        dcc.Store(id='session-data'),
        dcc.Store(id='tree-data'),
        dcc.Store(id='selected-image-data'),
        dcc.Store(id='original-image-store'),
    ])

@callback(
    Output('submit-button', 'disabled'),
    [Input('image-upload', 'contents'),
     Input('prompt-input', 'value')]
)
def enable_submit_button(image_contents, prompt):
    if image_contents is None or not prompt:
        return True
    return False

@callback(
    [Output('tree-graph-container', 'style'),
     Output('upload-container', 'style'),
     Output('original-image-store', 'data'),
     Output('tree-graph', 'elements', allow_duplicate=True),
     Output('tree-data', 'data', allow_duplicate=True),
     Output('session-data', 'data', allow_duplicate=True)],
    [Input('image-upload', 'contents')],
    prevent_initial_call=True
)
def process_uploaded_image(contents):
    if contents is None:
        raise PreventUpdate
    
    return  None, None, None, None, None, None

@callback(
    [Output('tree-graph', 'elements'),
     Output('session-data', 'data', allow_duplicate=True),
     Output('tree-data', 'data', allow_duplicate=True),
     Output('loading-output', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('original-image-store', 'data'),
     State('prompt-input', 'value'),
     State('ai-enhancement-checkbox', 'value'),
     State('session-data', 'data'),
     State('tree-data', 'data'),
     State('selected-image-data', 'data')],
    prevent_initial_call=True
)
def generate_images(n_clicks, image_src, prompt, use_ai, session_data, tree_data, selected_image_data):
    if n_clicks is None or image_src is None or not prompt:
        raise PreventUpdate
    
    return None, None, None, None

@callback(
    [Output('selected-image-data', 'data'),
     Output('suggestion-container', 'style'),
     Output('suggestion-button-1', 'children'),
     Output('suggestion-button-2', 'children'),
     Output('suggestion-button-3', 'children'),
     Output('suggestion-button-1', 'style'),
     Output('suggestion-button-2', 'style'),
     Output('suggestion-button-3', 'style'),
     Output('ai-enhancement-checkbox', 'disabled'),
     Output('tree-data', 'data', allow_duplicate=True),
     Output('tree-graph', 'elements', allow_duplicate=True)],
    [Input('tree-graph', 'tapNodeData')],
    [State('tree-data', 'data')],
    prevent_initial_call=True
)
def select_tree_node(node_data, tree_data):
    if node_data is None:
        raise PreventUpdate

    return None, None, None, None, None, None, None, None, None, None, None

@callback(
    Output('prompt-input', 'value'),
    [Input('suggestion-button-1', 'n_clicks'),
     Input('suggestion-button-2', 'n_clicks'),
     Input('suggestion-button-3', 'n_clicks')],
    [State('suggestion-button-1', 'children'),
     State('suggestion-button-2', 'children'),
     State('suggestion-button-3', 'children')],
    prevent_initial_call=True
)
def use_suggestion(click1, click2, click3, suggestion1, suggestion2, suggestion3):
    triggered_id = ctx.triggered_id
    
    if triggered_id is None:
        raise PreventUpdate
    
    if triggered_id == 'suggestion-button-1':
        return suggestion1
    elif triggered_id == 'suggestion-button-2':
        return suggestion2
    elif triggered_id == 'suggestion-button-3':
        return suggestion3
    
    raise PreventUpdate