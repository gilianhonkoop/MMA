from dash import dcc, html, callback, Input, Output, State, ctx, ALL
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import base64
import io
from dash.exceptions import PreventUpdate
import uuid
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
        style={'width': '100%', 'height': '100%'},
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
    """ Creates suggestion buttons, which will appear when the user 
    selects a tree node.
    """
    return dbc.Button(
        id=f'suggestion-button-{idx}',
        children="Suggestion",
        color="secondary",
        className="me-2 mb-2",
        style={'display': 'none', 'whiteSpace': 'nowrap'}
    )

# Chat Sidebar
def create_chat_sidebar():
    """Creates a sidebar showing all chats for the logged-in user"""
    return html.Div([
        html.H5("Your Chats", className="mb-3", style={'color': 'white'}),
        dbc.Button(
            "New Chat",
            id="new-chat-button",
            style={'backgroundColor': '#FFEED6', 'borderColor': '#FFEED6', 'color': '#38432E'},
            className="mb-3 w-100",
            size="sm"
        ),
        html.Div(
            id="chat-list-container",
            children=[
                dbc.Spinner(
                    html.Div("Loading chats...", className="text-center", style={'color': 'white'}),
                    size="sm"
                )
            ],
            style={'height': 'calc(100vh - 260px)', 'overflowY': 'auto'}
        )
    ], style={
        'backgroundColor': '#38432E',
        'padding': '15px',
        'height': 'calc(100vh - 60px)',
        'position': 'fixed',
        'left': '0',
        'top': '60px',
        'width': '300px',
        'zIndex': '1000'
    })

# Layout Builder
def create_user_layout():
    """Builds the main page layout with full screen utilization"""
    return html.Div([
        create_chat_sidebar(),
        
        html.Div([
            html.Div([
                html.Div(
                    id="tree-graph-container",
                    children=create_tree_visualization(),
                    style={"display": "none", "height": "100%", "backgroundColor": "#FFEED6", "padding": "10px", "width": "100%"}
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
                ], id="upload-container", style={"backgroundColor": "#FFEED6", "padding": "20px", "height": "100%", "display": "flex", "flexDirection": "column", "justifyContent": "center"}),
            ], style={"flex": "1", "backgroundColor": "#FFEED6", "minHeight": "0", "overflow": "hidden"}),
            
            html.Div([
                html.Div([
                    html.Div([
                        create_suggestion_button(1),
                        create_suggestion_button(2),
                        create_suggestion_button(3),
                    ], style={
                        'display': 'flex',
                        'gap': '10px',
                        'overflowX': 'auto',
                        'whiteSpace': 'nowrap',
                        'padding': '5px 0'
                    })
                ], id='suggestion-container', style={
                    'display': 'none',
                    'backgroundColor': '#F5F5F5',
                    'border': '1px solid #ddd',
                    'borderRadius': '5px',
                    'padding': '8px 12px',
                    'marginBottom': '15px',
                    'maxHeight': '60px',
                    'overflowY': 'hidden'
                }),
                
                # Input and controls
                dbc.InputGroup([
                    dbc.Input(id='prompt-input', placeholder="Enter your prompt here...", type="text"),
                    dbc.InputGroupText(
                        dbc.Checkbox(
                            id='ai-enhancement-checkbox',
                            label="AI-Enhance prompt",
                            value=False,
                        )
                    ),
                    dbc.Button("Generate Images", id="submit-button", disabled=True, 
                              style={'backgroundColor': '#38432E', 'borderColor': '#38432E', 'color': 'white'}),
                ], className="mb-3"),
                
                # Guidance controls
                html.Div([
                    html.Div([
                        dbc.Checkbox(
                            id='guidance-enable-checkbox',
                            label="Enable Custom Guidance Settings",
                            value=False,
                        ),
                        
                        html.Div([
                            html.Label("Prompt Guidance", className="mb-1", style={'fontSize': '12px'}),
                            dcc.Slider(
                                id='prompt-guidance-slider',
                                min=0,
                                max=2,
                                step=1,
                                value=1,
                                marks={
                                    0: 'Low',
                                    1: 'Medium',
                                    2: 'High'
                                },
                                tooltip={"placement": "bottom", "always_visible": False},
                                disabled=True,
                            )
                        ], style={'marginLeft': '20px', 'marginRight': '20px', 'flex': '1', 'maxWidth': '200px'}),
                        
                        html.Div([
                            html.Label("Image Guidance", className="mb-1", style={'fontSize': '12px'}),
                            dcc.Slider(
                                id='image-guidance-slider',
                                min=0,
                                max=2,
                                step=1,
                                value=1,
                                marks={
                                    0: 'Low',
                                    1: 'Medium',
                                    2: 'High'
                                },
                                tooltip={"placement": "bottom", "always_visible": False},
                                disabled=True,
                            )
                        ], style={'marginRight': '20px', 'flex': '1', 'maxWidth': '200px'}),
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '15px'})
                ]),
                
                dbc.Spinner(html.Div(id="loading-output")),
            ], style={
                "backgroundColor": "#FFEED6", 
                "padding": "20px", 
                "minHeight": "180px",
                "borderTop": "1px solid #ccc",
                "flexShrink": "0",
            }),
        ], style={
            "marginLeft": "300px",
            "height": "calc(100vh - 60px)",
            "display": "flex",
            "flexDirection": "column"
        }),
        
        dcc.Store(id='session-data'),
        dcc.Store(id='tree-data'),
        dcc.Store(id='selected-image-data'),
        dcc.Store(id='original-image-store'),
        dcc.Store(id='active-chat-data'),
    ], style={"height": "100vh", "overflow": "hidden"})

@callback(
    Output('submit-button', 'disabled'),
    [Input('image-upload', 'contents'),
     Input('prompt-input', 'value'),
     Input('tree-data', 'data'),
     Input('selected-image-data', 'data')]
)

# Enable Submit Button
def enable_submit_button(image_contents, prompt, tree_data, selected_image_data):
    """ Enables the "Generate Images" button when:
    - New chat: An image is uploaded AND a prompt is entered
    - Existing chat: A node is selected AND a prompt is entered
    """
    if not prompt:
        return True
    
    # For new chats - need uploaded image
    if image_contents is not None:
        return False
    
    # For existing chats - check if we have tree data and either a selected node or root node
    if tree_data and tree_data.get('nodes'):
        # If we have selected image data with a node ID, enable button
        if selected_image_data and selected_image_data.get('selected_node_id'):
            return False
        # If no selected image data but we have tree data (chat loaded), allow generation from current node
        if tree_data.get('current_node'):
            return False
    
    return True

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
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded))
    user_id = user_info.get('user_id', 1) if user_info else 1
    db = Database()
    db.connect()
    
    vlm = get_vlm_instance()
    chat_title = vlm.create_title(img)

    # chat_title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    chat = Chat(user_id=user_id, title=chat_title)
    
    db_chat_id = db.insert_chat(chat.title, user_id)
    prompt_image = PromptImage(img, None, None, input_prompt=None, output_prompt=None, save=True, selected=True)
    
    db.save_image(prompt_image, db_chat_id, user_id)
    
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
        {'display': 'block', 'height': '100%', 'backgroundColor': '#FFEED6', 'padding': '10px', 'width': '100%'}, 
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
     State('guidance-enable-checkbox', 'value'),
     State('prompt-guidance-slider', 'value'),
     State('image-guidance-slider', 'value'),
     State('session-data', 'data'),
     State('tree-data', 'data'),
     State('selected-image-data', 'data'),
     State('app-user-info', 'data')],
    prevent_initial_call=True
)

# Image Generation Callback
def generate_images(n_clicks, image_src, prompt_text, use_ai, guidance_enabled, prompt_guidance, image_guidance, session_data, tree_data, selected_image_data, user_info):
    """
    Triggered on clicking "Generate Images". It uses:
    Image source, Prompt, AI enhancement option, Current session and tree state
    Generates images using ImageTransformer and adds them to the tree
    """
    if n_clicks is None or not prompt_text:
        raise PreventUpdate
    
    user_id = user_info.get('user_id', 1) if user_info else 1
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
    
    # Only pass guidance values if custom guidance is enabled
    guidance_params = {}
    if guidance_enabled:
        guidance_params['prompt_guidance'] = prompt_guidance
        guidance_params['image_guidance'] = image_guidance
    else:
        guidance_params['prompt_guidance'] = None
        guidance_params['image_guidance'] = None
    
    output_images = prompt_obj.get_new_images(image_transformer, n=3, save=True, **guidance_params)
    
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

@callback(
    Output('chat-list-container', 'children'),
    [Input('app-user-info', 'data')],
    prevent_initial_call=False
)
def load_user_chats(user_info):
    """Load and display all chats for the logged-in user"""
    if not user_info:
        return html.Div("Please log in to view your chats.", className="text-center text-muted")
    
    try:
        user_id = user_info.get('user_id', 1)
        
        db = Database()
        db.connect()
        
        chats_df = db.fetch_chats_by_user(user_id, pandas=True)
        db.close()
        
        if chats_df.empty:
            return html.Div("No chats found. Create a new chat to get started!", className="text-center text-muted")
        
        chat_buttons = []
        for _, chat in chats_df.iterrows():
            title = chat['title']
            if len(title) > 30:
                title = title[:27] + "..."
            
            chat_buttons.append(
                dbc.Button(
                    title,
                    id={'type': 'chat-button', 'index': chat['id']},
                    color="outline-secondary",
                    size="sm",
                    className="mb-2 w-100 text-start",
                    style={'textAlign': 'left'}
                )
            )
        
        return chat_buttons
    
    except Exception as e:
        print(f"Error loading chats: {e}")
        return html.Div("Error loading chats.", className="text-center text-danger")

@callback(
    [Output('tree-graph-container', 'style', allow_duplicate=True),
     Output('upload-container', 'style', allow_duplicate=True),
     Output('tree-graph', 'elements', allow_duplicate=True),
     Output('tree-data', 'data', allow_duplicate=True),
     Output('session-data', 'data', allow_duplicate=True),
     Output('active-chat-data', 'data'),
     Output('prompt-input', 'value', allow_duplicate=True),
     Output('suggestion-container', 'style', allow_duplicate=True),
     Output('selected-image-data', 'data', allow_duplicate=True)],
    [Input({'type': 'chat-button', 'index': ALL}, 'n_clicks')],
    [State('app-user-info', 'data')],
    prevent_initial_call=True
)
def load_existing_chat(n_clicks_list, user_info):
    """Load an existing chat when a chat button is clicked"""
    if not any(n_clicks_list) or not user_info:
        raise PreventUpdate
    
    ctx_triggered = ctx.triggered[0] if ctx.triggered else None
    if not ctx_triggered:
        raise PreventUpdate
    
    button_id = json.loads(ctx_triggered['prop_id'].split('.')[0])
    chat_id = button_id['index']
    
    try:
        user_id = user_info.get('user_id', 1)
        
        db = Database()
        db.connect()
        
        chat_df = db.fetch_chat_by_id(chat_id, pandas=True)
        if chat_df.empty:
            db.close()
            raise PreventUpdate
        
        images_df = db.fetch_images_by_chat(chat_id, pandas=True)
        
        prompts_df = db.fetch_prompts_by_chat(chat_id, pandas=True)
        
        db.close()
        
        tree_data = {
            'nodes': [],
            'edges': [],
            'current_node': 'root',
            'max_level': 0,
            'selected_node': 'root'
        }
        
        session_data = {
            'session_id': chat_id,
            'image_count': len(images_df),
            'root_image_id': None,
            'chat_id': chat_id
        }
        
        level_map = {} 
        parent_map = {}
        
        for _, img in images_df.iterrows():
            img_id = img['id']
            input_prompt_id = img['input_prompt_id']
            
            if input_prompt_id is None:
                level_map[img_id] = 0
                parent_map[img_id] = None
                session_data['root_image_id'] = img_id
            else:
                parent_prompt = prompts_df[prompts_df['id'] == input_prompt_id]
                if not parent_prompt.empty:
                    parent_image_id = parent_prompt.iloc[0]['image_in_id']
                    if parent_image_id in level_map:
                        level_map[img_id] = level_map[parent_image_id] + 1
                        parent_map[img_id] = parent_image_id
                    else:
                        level_map[img_id] = 1
                        parent_map[img_id] = session_data['root_image_id']
                else:
                    level_map[img_id] = 1
                    parent_map[img_id] = session_data['root_image_id']
        
        cy_elements = []
        node_counter = 0
        
        for _, img in images_df.iterrows():
            img_id = img['id']
            level = level_map.get(img_id, 0)
            
            try:
                if os.path.exists(img['path']):
                    from PIL import Image
                    pil_img = Image.open(img['path'])
                    buffered = io.BytesIO()
                    pil_img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    img_src = f"data:image/png;base64,{img_str}"
                else:
                    img_src = ""
            except:
                img_src = ""
            
            if level == 0:
                node_id = 'root'
                label = 'Original Image'
            else:
                node_id = f"node_{node_counter}"
                node_counter += 1
                related_prompt = prompts_df[prompts_df['image_in_id'] == parent_map.get(img_id)]
                if not related_prompt.empty:
                    prompt_text = related_prompt.iloc[0]['prompt']
                    label = prompt_text[:15] + '...' if len(prompt_text) > 15 else prompt_text
                else:
                    label = f"Image {node_counter}"
            
            tree_data['nodes'].append({
                'id': node_id,
                'label': label,
                'image': img_src,
                'level': level,
                'image_id': img_id
            })
            
            element = {'data': {
                'id': node_id,
                'label': label,
                'image': img_src,
                'level': level,
                'image_id': img_id
            }}
            
            if level == tree_data['max_level']:
                element['selected'] = (node_id == 'root')
            else:
                element['classes'] = 'disabled'
            
            cy_elements.append(element)
            
            tree_data['max_level'] = max(tree_data['max_level'], level)
        
        for _, img in images_df.iterrows():
            img_id = img['id']
            parent_id = parent_map.get(img_id)
            
            if parent_id is not None:
                source_node = None
                target_node = None
                
                for node in tree_data['nodes']:
                    if node['image_id'] == parent_id:
                        source_node = node['id']
                    if node['image_id'] == img_id:
                        target_node = node['id']
                
                if source_node and target_node:
                    edge = {
                        'id': f"edge_{source_node}_{target_node}",
                        'source': source_node,
                        'target': target_node
                    }
                    tree_data['edges'].append(edge)
                    cy_elements.append({'data': edge})
        
        active_chat_data = {
            'chat_id': chat_id,
            'title': chat_df.iloc[0]['title'],
            'user_id': user_id
        }
        
        root_image_id = session_data.get('root_image_id')
        selected_image_data = {
            'selected_node_id': 'root',
            'image_id': root_image_id
        }
        
        return (
            {'display': 'block', 'height': '100%', 'backgroundColor': '#FFEED6', 'padding': '10px', 'width': '100%'},
            {'display': 'none'}, 
            cy_elements,
            tree_data,
            session_data,
            active_chat_data,
            "", 
            {'display': 'none'},
            selected_image_data
        )
    
    except Exception as e:
        print(f"Error loading chat: {e}")
        raise PreventUpdate

@callback(
    Output('chat-list-container', 'children', allow_duplicate=True),
    [Input('new-chat-button', 'n_clicks')],
    [State('app-user-info', 'data')],
    prevent_initial_call=True
)
def create_new_chat(n_clicks, user_info):
    """Reset the interface for a new chat"""
    if not n_clicks or not user_info:
        raise PreventUpdate
    
    return load_user_chats(user_info)

@callback(
    [Output('tree-graph-container', 'style', allow_duplicate=True),
     Output('upload-container', 'style', allow_duplicate=True),
     Output('session-data', 'data', allow_duplicate=True),
     Output('tree-data', 'data', allow_duplicate=True),
     Output('active-chat-data', 'data', allow_duplicate=True),
     Output('prompt-input', 'value', allow_duplicate=True),
     Output('suggestion-container', 'style', allow_duplicate=True),
     Output('selected-image-data', 'data', allow_duplicate=True)],
    [Input('new-chat-button', 'n_clicks')],
    prevent_initial_call=True
)
def reset_for_new_chat(n_clicks):
    """Reset the interface for a new chat"""
    if not n_clicks:
        raise PreventUpdate
    
    return (
        {'display': 'none'}, 
        {'display': 'block'},
        {},
        {},
        {},
        "",
        {'display': 'none'},
        {}
    )

@callback(
    [Output('prompt-guidance-slider', 'disabled'),
     Output('image-guidance-slider', 'disabled')],
    [Input('guidance-enable-checkbox', 'value')]
)
def toggle_guidance_sliders(enable_guidance):
    """Enable or disable the guidance sliders based on checkbox state"""
    return not enable_guidance, not enable_guidance