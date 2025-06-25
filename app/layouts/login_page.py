from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from db import Database

GREEN = "#132a13"
BG = "rgba(177, 183, 143, 0.3)"


def create_login_layout():
    return html.Div([
        html.Div([
            html.H1("Image Tree", className="text-center mb-4", style={'color': GREEN, 'fontWeight': 'bold'}),
            dbc.Card([
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab([
                            html.Div([
                                dbc.Input(
                                    id="login-username", 
                                    placeholder="Username", 
                                    type="text", 
                                    className="mb-3",
                                    style={'borderColor': GREEN}
                                ),
                                dbc.Input(
                                    id="login-password", 
                                    placeholder="Password", 
                                    type="password", 
                                    className="mb-3",
                                    style={'borderColor': GREEN}
                                ),
                                dbc.Button(
                                    "Login", 
                                    id="login-button", 
                                    className="w-100",
                                    style={'backgroundColor': GREEN, 'borderColor': GREEN, 'color': 'white'}
                                ),
                                html.Div(id="login-error", className="text-danger mt-3")
                            ], className="p-3"),
                        ], label="Login", tab_style={'color': GREEN}, active_tab_style={'color': GREEN, 'fontWeight': 'bold'}),
                        dbc.Tab([
                            html.Div([
                                dbc.Input(
                                    id="register-username", 
                                    placeholder="Username", 
                                    type="text", 
                                    className="mb-3",
                                    style={'borderColor': GREEN}
                                ),
                                dbc.Input(
                                    id="register-password", 
                                    placeholder="Password", 
                                    type="password", 
                                    className="mb-3",
                                    style={'borderColor': GREEN}
                                ),
                                dbc.Input(
                                    id="register-confirm-password", 
                                    placeholder="Confirm Password", 
                                    type="password", 
                                    className="mb-3",
                                    style={'borderColor': GREEN}
                                ),
                                dbc.Button(
                                    "Register", 
                                    id="register-button", 
                                    className="w-100",
                                    style={'backgroundColor': GREEN, 'borderColor': GREEN, 'color': 'white'}
                                ),
                                html.Div(id="register-error", className="text-danger mt-3"),
                                html.Div(id="register-success", className="text-success mt-3")
                            ], className="p-3"),
                        ], label="Register", tab_style={'color': GREEN}, active_tab_style={'color': GREEN, 'fontWeight': 'bold'}),
                    ], style={'borderColor': GREEN}),
                ]),
            ], className="shadow", style={'border': '2px solid #38432E', 'maxWidth': '450px', 'width': '100%'}),
        ], style={
            'position': 'absolute',
            'top': '50%',
            'left': '50%',
            'transform': 'translate(-50%, -50%)',
            'width': '100%',
            'maxWidth': '450px',
            'padding': '20px'
        }),
        
        dcc.Store(id="user-info"),
    ], style={'minHeight': '100vh', 'height': '100vh', 'position': 'relative'})

@callback(
    [Output("user-info", "data"),
     Output("login-error", "children")],
    [Input("login-button", "n_clicks")],
    [State("login-username", "value"),
     State("login-password", "value")],
    prevent_initial_call=True
)
def login_user(n_clicks, username, password):
    if n_clicks is None or not username or not password:
        raise PreventUpdate
    
    with Database() as db:
        user_data = db.fetch_user_by_username(username)
    
    if user_data.empty:
        return None, "User not found."
    
    stored_password = user_data.iloc[0]["password"]
    if password != stored_password: 
        return None, "Incorrect password."
    
    user_id = user_data.iloc[0]["id"]
    return {"username": username, "user_id": int(user_id)}, ""

@callback(
    [Output("register-error", "children"),
     Output("register-success", "children"),
     Output("register-username", "value"),
     Output("register-password", "value"),
     Output("register-confirm-password", "value")],
    [Input("register-button", "n_clicks")],
    [State("register-username", "value"),
     State("register-password", "value"),
     State("register-confirm-password", "value")],
    prevent_initial_call=True
)
def register_user(n_clicks, username, password, confirm_password):
    if n_clicks is None:
        raise PreventUpdate
    
    if not username:
        return "Username cannot be empty.", "", None, None, None
    
    if not password:
        return "Password cannot be empty.", "", username, None, None
    
    if password != confirm_password:
        return "Passwords do not match.", "", username, None, None
    
    with Database() as db:
        user_data = db.fetch_user_by_username(username)
        
        if not user_data.empty:
            return f"Username '{username}' already exists.", "", username, None, None
        
        db.insert_user(username, password)
    
    return "", f"User '{username}' registered successfully! You can now login.", "", "", ""

@callback(
    Output("url", "pathname"),
    [Input("user-info", "data")],
    prevent_initial_call=True
)
def redirect_after_login(user_data):
    if user_data:
        return "/"
    raise PreventUpdate