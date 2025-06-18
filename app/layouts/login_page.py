from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from db import Database

def create_login_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Login", className="text-center my-4"),
                dbc.Card([
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab([
                                html.Div([
                                    dbc.Input(id="login-username", placeholder="Username", type="text", className="mb-3"),
                                    dbc.Input(id="login-password", placeholder="Password", type="password", className="mb-3"),
                                    dbc.Button("Login", id="login-button", color="primary", className="w-100"),
                                    html.Div(id="login-error", className="text-danger mt-3")
                                ], className="p-3"),
                            ], label="Login"),
                            dbc.Tab([
                                html.Div([
                                    dbc.Input(id="register-username", placeholder="Username", type="text", className="mb-3"),
                                    dbc.Input(id="register-password", placeholder="Password", type="password", className="mb-3"),
                                    dbc.Input(id="register-confirm-password", placeholder="Confirm Password", type="password", className="mb-3"),
                                    dbc.Button("Register", id="register-button", color="success", className="w-100"),
                                    html.Div(id="register-error", className="text-danger mt-3"),
                                    html.Div(id="register-success", className="text-success mt-3")
                                ], className="p-3"),
                            ], label="Register"),
                        ]),
                    ]),
                ], className="shadow"),
            ], width={"size": 6, "offset": 3}),
        ], className="vh-100 d-flex align-items-center justify-content-center"),
        
        dcc.Store(id="user-info"),
    ])

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