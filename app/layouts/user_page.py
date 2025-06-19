
from dash import dcc, html, callback, Input, Output, State, ctx
# Bootstrap components for styling
import dash_bootstrap_components as dbc
# Cytoscape integration to draw graph/tree visualizations
import dash_cytoscape as cyto
# Stops callbacks from updating outputs under certain conditions.
from dash.exceptions import PreventUpdate


# Tree Visualization
def create_tree_visualization():
    """Creates a graph layout using dagre (a directed acyclic graph layout). 
    It's styled to use images as node backgrounds and includes styles for: 
    Selected nodes, Disabled nodes, Selectable nodes
    """
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

# Image Display Area
def create_image_display(id_prefix, title):
    """ Creates an area to show an image with a title. 
    Initially hidden (display: none), likely toggled via callback.
    """
    return html.Div([
        html.H5(title),
        html.Div(
            id=f'{id_prefix}-image-container',
            children=[html.Img(id=f'{id_prefix}-image', style={'width': '100%', 'display': 'none'})],
            style={'minHeight': '200px', 'border': '1px dashed gray', 'textAlign': 'center'}
        ),
    ])

# Suggestion Button
def create_suggestion_button(idx):
    """ Creates hidden suggestion buttons, which will appear when the user 
    selects a tree node.
    """
    return dbc.Button(
        id=f'suggestion-button-{idx}',
        children="Suggestion",
        color="secondary",
        className="me-2 mb-2",
        style={'display': 'none'}
    )

# Layout Builder
def create_user_layout():
    """Builds the main page layout:
    Header and image uploader.
    Hidden tree visualization container.
    Hidden suggestion button area.
    Input field for prompts.
    Submit button (disabled until conditions are met).
    Several dcc.Store components to hold session state.
    """
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

# Enable Submit Button
def enable_submit_button(image_contents, prompt):
    """ Enables the "Generate Images" button only if: 
    An image is uploaded, A prompt is entered
    """
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

# Image Upload Handler
def process_uploaded_image(contents):
    """Triggered when an image is uploaded. 
    It's currently a placeholder (returns all None) but expected to:
    Process the uploaded image, 
    Show the tree graph, 
    Store image data, 
    Build initial tree nodes and edges
    """
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

# Image Generation Callback
def generate_images(n_clicks, image_src, prompt, use_ai, session_data, tree_data, selected_image_data):
    """
    Triggered on clicking "Generate Images". It uses:
    Image source, Prompt, AI enhancement option, Current session and tree state
    Also currently a placeholder (return Nones). In production, it would:
    Generate images (maybe using AI), Update the tree with new nodes
    """
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

# Tree Node Selection Callback
def select_tree_node(node_data, tree_data):
    """ Runs when a node in the tree is clicked:
    Would update selected node data, Display related suggestions,
    Possibly allow user to enhance/modify based on the selected image
    """
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

# Use Suggestion Button
def use_suggestion(click1, click2, click3, suggestion1, suggestion2, suggestion3):
    """ When one of the suggestion buttons is clicked:
    Detects which was clicked via ctx.triggered_id, 
    Fills the input prompt with that suggestion
    """
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