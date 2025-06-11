import dash
from dash import dcc, html, Input, Output, State, callback, ctx
from dash.exceptions import PreventUpdate
import dash_cytoscape as cyto
import json
import random

cyto.load_extra_layouts()

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.Div([
        cyto.Cytoscape(
            id='graph',
            layout={'name': 'dagre', 'rankDir': 'LR', 'rankSep': 150, 'nodeSep': 100},
            style={'width': '100%', 'height': '600px'},
            elements=[
                        {'data': {'id': 'node1', 'label': 'Node 1', 'image': 'https://picsum.photos/id/1/200/200'}},
                    ],
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
                        'border-color': 'red'
                    }
                }
            ]
        )
    ], style={'border': '1px solid black'}),
    
    html.Div([
        html.Div([
            html.Button("Add Edge", id="add-edge-button", n_clicks=0),
            html.Div([
                html.P("Selected Node:"),
                html.Div(id="selected-node-info", style={"margin-bottom": "10px"}),
                html.Img(id="selected-node-image", style={"max-width": "100px", "max-height": "100px", "display": "block", "margin-top": "10px"})
            ])
        ]),
        
        html.Div([
            html.H3("Graph Data"),
            html.Pre(id="graph-data-json", style={"border": "1px", "padding": "10px", "max-height": "300px"})
        ])
    ])
])

app.selected_node_id = None

# callback to display selected node information
@callback(
    Output("selected-node-info", "children"),
    Output("selected-node-image", "src"),
    Input("graph", "selectedNodeData")
)
def display_selected_node(node_data):
    if not node_data:
        return "No node selected", ""
    
    app.selected_node_id = node_data[0]['id']
    label = node_data[0].get('label', node_data[0]['id'])
    image = node_data[0].get('image', '')
    
    return (
        f"ID: {app.selected_node_id}, Label: {label}", 
        image, 
    )

# callback for adding edges
@callback(
    Output("graph", "elements"),
    Input("add-edge-button", "n_clicks"),
    State("graph", "elements"),
    State("graph", "mouseoverNodeData"),
)
def update_graph(add_edge_clicks, elements, mouseover_data):
    trigger = ctx.triggered_id
    
    if trigger is None:
        raise PreventUpdate
        
    if trigger == "add-edge-button" and app.selected_node_id:
        source_id = app.selected_node_id
        target_id = f"node{len([e for e in elements if 'source' not in e['data']]) + 1}"
        target_label = f"Node {target_id[4:]}"
        
        new_node = {
            'data': {'id': target_id, 'label': target_label, 'image': f'https://picsum.photos/id/{random.randint(1, 100)}/200/200'},
        }
        
        new_edge = {
            'data': {
                'id': f"edge{len([e for e in elements if 'source' in e['data']]) + 1}",
                'source': source_id,
                'target': target_id
            }
        }
        
        elements.extend([new_node, new_edge])
    
    return elements

# callback to display graph data
@callback(
    Output("graph-data-json", "children"),
    Input("graph", "elements")
)
def display_graph_json(elements):
    return json.dumps(elements, indent=2)

if __name__ == '__main__':
    app.run_server(debug=True)