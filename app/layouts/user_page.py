from dash import dcc, html, callback, Input, Output, State, ctx
# Bootstrap components for styling
import dash_bootstrap_components as dbc
# Cytoscape integration to draw graph/tree visualizations
import dash_cytoscape as cyto
# Stops callbacks from updating outputs under certain conditions.
from dash.exceptions import PreventUpdate
import uuid
import base64
import io
from PIL import Image
import os
import sys
from datetime import datetime
import json
from modules.prompt import Prompt
from modules.prompt_image import PromptImage
from modules.chat import Chat
from db.database import Database
from modules.model_instances import get_vlm_instance, get_image_transformer_instance


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
    [State('app-user-info', 'data')],
    prevent_initial_call=True
)

# Image Upload Handler
def process_uploaded_image(contents, user_info):
    """Triggered when an image is uploaded. 
    Process the uploaded image, 
    Show the tree graph, 
    Store image data, 
    Build initial tree nodes and edges
    """
    if contents is None:
        raise PreventUpdate
    
    # Parse the Base64 string
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    # Convert to PIL Image
    img = Image.open(io.BytesIO(decoded))
    
    # Get user ID from session or default to 1
    user_id = user_info.get('id', 1) if user_info else 1
    
    # Initialize a database connection
    db = Database()
    db.connect()
    
    # Create a unique session ID for the chat
    chat_title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Create a Chat object
    chat = Chat(user_id=user_id, title=chat_title)
    
    # Save the chat to the database
    db_chat_id = db.insert_chat(chat.title, user_id)
    
    # Create initial PromptImage (input)
    prompt_image = PromptImage(img, None, None, input_prompt=None, output_prompt=None, save=True)
    
    # Save image to database
    db.save_image(prompt_image, db_chat_id, user_id)
    
    # Store session data
    session_data = {
        'session_id': db_chat_id,
        'image_count': 1,
        'root_image_id': prompt_image.id,
        'chat_id': db_chat_id
    }
    
    tree_data = {
        'nodes': [
            {'id': 'root', 'label': 'Original Image', 'image': contents, 'level': 0, 'image_id': prompt_image.id}
        ],
        'edges': [],
        'current_node': 'root',
        'max_level': 0,
        'selected_node': 'root'
    }
    
    cy_elements = [{'data': tree_data['nodes'][0], 'selected': True}]

    db.close()
    
    return (
        {'display': 'block'}, 
        {'display': 'none'},
        contents,
        cy_elements,
        tree_data,
        session_data
    )

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
     State('selected-image-data', 'data'),
     State('app-user-info', 'data')],
    prevent_initial_call=True
)

# Image Generation Callback
def generate_images(n_clicks, image_src, prompt_text, use_ai, session_data, tree_data, selected_image_data, user_info):
    """
    Triggered on clicking "Generate Images". It uses:
    Image source, Prompt, AI enhancement option, Current session and tree state
    Generates images using ImageTransformer and adds them to the tree
    """
    if n_clicks is None or image_src is None or not prompt_text:
        raise PreventUpdate
    
    user_id = user_info.get('id', 1) if user_info else 1
    chat_id = session_data.get('chat_id')
    
    db = Database()
    db.connect()
    
    parent_id = tree_data.get('current_node') if tree_data and tree_data.get('current_node') else 'root'
    
    parent_level = 0
    parent_image_id = None
    for node in tree_data['nodes']:
        if node['id'] == parent_id:
            parent_level = node.get('level', 0)
            parent_image_id = node.get('image_id')
            break
    
    parent_image_obj = db.get_image_by_id(parent_image_id)
    
    if parent_image_obj is None:
        if parent_id == 'root':
            content_type, content_string = image_src.split(',')
            decoded = base64.b64decode(content_string)
            parent_image = Image.open(io.BytesIO(decoded))
            
            parent_image_obj = PromptImage(parent_image, None, None)
            
            db.save_image(parent_image_obj, chat_id, user_id)
        else:
            raise PreventUpdate
    
    suggestion_used = None
    modified_suggestion = False
    
    if selected_image_data and 'used_suggestion' in selected_image_data:
        suggestion_used = selected_image_data['used_suggestion']
        modified_suggestion = selected_image_data.get('modified', False)
    
    prompt_obj = Prompt(
        prompt=prompt_text, 
        depth=parent_level + 1, 
        input_image=parent_image_obj,
        suggestion_used=suggestion_used,
        modified_suggestion=modified_suggestion
    )
    
    if use_ai:
        vlm = get_vlm_instance()
        prompt_obj.enhance_prompt(vlm)
    
    db.save_prompt(prompt_obj, chat_id, user_id)
    
    image_transformer = get_image_transformer_instance()
    
    output_images = prompt_obj.get_new_images(image_transformer, n=5, save=True)
    
    for img in output_images:
        img.set_input_prompt(prompt_obj.id)
        db.save_image(img, chat_id, user_id)
    
    new_level = parent_level + 1
    tree_data['max_level'] = max(tree_data.get('max_level', 0), new_level)
    
    if parent_id == 'root' and len(tree_data.get('edges', [])) == 0:
        for i, node in enumerate(tree_data['nodes']):
            if node['id'] == 'root':
                tree_data['nodes'][i]['label'] = prompt_text[:15] + '...' if len(prompt_text) > 15 else prompt_text
                break
    
    for i, img in enumerate(output_images):
        buffered = io.BytesIO()
        img.image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_src = f"data:image/png;base64,{img_str}"
        
        node_id = f"node_{session_data['image_count'] + i}"
        
        tree_data['nodes'].append({
            'id': node_id,
            'label': prompt_text[:15] + '...' if len(prompt_text) > 15 else prompt_text,
            'image': img_src,
            'level': new_level,
            'image_id': img.id
        })
        
        tree_data['edges'].append({
            'id': f"edge_{parent_id}_{node_id}",
            'source': parent_id,
            'target': node_id
        })
    
    session_data['image_count'] += len(output_images)
    
    cy_elements = []
    for node in tree_data['nodes']:
        element = {'data': node.copy()}
        if tree_data.get('selected_node') and node['id'] == tree_data.get('selected_node'):
            element['selected'] = True
        cy_elements.append(element)
    
    cy_elements.extend([{'data': edge} for edge in tree_data['edges']])
    
    db.close()

    return (
        cy_elements,
        session_data, 
        tree_data,
        ""
    )

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
    Updates selected node data, Display related suggestions,
    Allows user to enhance/modify based on the selected image
    """
    if node_data is None:
        raise PreventUpdate
    
    db = Database()
    db.connect()
    
    selected_node_id = node_data['id']
    selected_image_id = node_data.get('image_id')
    
    node_levels = {node['id']: node.get('level', 0) for node in tree_data['nodes']}
    selected_node_level = node_levels.get(selected_node_id, 0)
    
    max_level = tree_data.get('max_level', 0)
    
    if selected_node_level != max_level:
        raise PreventUpdate
    
    tree_data['current_node'] = selected_node_id
    tree_data['selected_node'] = selected_node_id
    
    cy_elements = []
    
    for node in tree_data['nodes']:
        is_selected = (node['id'] == selected_node_id)
        element = {
            'data': node.copy(),
            'classes': '' if is_selected else 'disabled',
            'selected': is_selected
        }
        cy_elements.append(element)
    
    cy_elements.extend([{'data': edge} for edge in tree_data['edges']])

    if selected_image_id:
        db.set_image_selected(selected_image_id, True)
    
    if selected_node_id == 'root':
        db.close()
        return (
            {"selected_node_id": selected_node_id, "image_id": selected_image_id},
            {'display': 'none'},
            "",
            "",
            "",
            {'display': 'none'},
            {'display': 'none'},
            {'display': 'none'},
            False,
            tree_data,
            cy_elements
        )
    else:
        prompt_image = db.get_image_by_id(selected_image_id)
        
        default_suggestions = ["Enhance colors and contrast", "Add artistic effect", "Make more dramatic"]
        suggestions = default_suggestions.copy()
        
        if prompt_image:
            try:
                temp_prompt = Prompt("", 0, input_image=prompt_image)
                
                vlm = get_vlm_instance()
                suggestions = vlm.make_suggestions(temp_prompt, n_suggestions=3)
                
                if len(suggestions) < 3:
                    suggestions = default_suggestions.copy()
            
            except Exception as e:
                print(f"Error generating suggestions: {e}")
                suggestions = default_suggestions.copy()

        db.close()
        
        while len(suggestions) < 3:
            suggestions.append(default_suggestions[len(suggestions) % len(default_suggestions)])
        
        return (
            {"selected_node_id": selected_node_id, "image_id": selected_image_id},
            {'display': 'block'},
            suggestions[0],
            suggestions[1],
            suggestions[2],
            {'display': 'block'},
            {'display': 'block'},
            {'display': 'block'},
            True,
            tree_data,
            cy_elements
        )

@callback(
    [Output('prompt-input', 'value'),
     Output('selected-image-data', 'data', allow_duplicate=True)],
    [Input('suggestion-button-1', 'n_clicks'),
     Input('suggestion-button-2', 'n_clicks'),
     Input('suggestion-button-3', 'n_clicks')],
    [State('suggestion-button-1', 'children'),
     State('suggestion-button-2', 'children'),
     State('suggestion-button-3', 'children'),
     State('selected-image-data', 'data')],
    prevent_initial_call=True
)

# Use Suggestion Button
def use_suggestion(click1, click2, click3, suggestion1, suggestion2, suggestion3, selected_image_data):
    """ When one of the suggestion buttons is clicked:
    Detects which was clicked via ctx.triggered_id, 
    Fills the input prompt with that suggestion,
    Records which suggestion was used in the database
    """
    triggered_id = ctx.triggered_id
    
    if triggered_id is None:
        raise PreventUpdate
    
    if selected_image_data is None:
        selected_image_data = {}
    
    db = Database()
    db.connect()
    
    try:
        if triggered_id == 'suggestion-button-1':
            selected_image_data['used_suggestion'] = suggestion1
            selected_image_data['modified'] = False
            return suggestion1, selected_image_data
        elif triggered_id == 'suggestion-button-2':
            selected_image_data['used_suggestion'] = suggestion2
            selected_image_data['modified'] = False
            return suggestion2, selected_image_data
        elif triggered_id == 'suggestion-button-3':
            selected_image_data['used_suggestion'] = suggestion3
            selected_image_data['modified'] = False
            return suggestion3, selected_image_data
    finally:
        db.close()
    
    raise PreventUpdate

@callback(
    Output('selected-image-data', 'data', allow_duplicate=True),
    [Input('prompt-input', 'value')],
    [State('selected-image-data', 'data')],
    prevent_initial_call=True
)

def track_prompt_modifications(prompt_value, selected_image_data):
    """
    Track when a user modifies a suggestion in the prompt input.
    This allows us to know when to set modified_suggestion=True in the Prompt object.
    """
    if selected_image_data is None or 'used_suggestion' not in selected_image_data:
        raise PreventUpdate
    
    original_suggestion = selected_image_data.get('used_suggestion', '')
    is_modified = prompt_value != original_suggestion
    
    if is_modified != selected_image_data.get('modified', False):
        selected_image_data['modified'] = is_modified
        
        return selected_image_data
    
    raise PreventUpdate