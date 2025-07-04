import dash
from dash import dcc, html, callback, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from layouts.user_page import create_user_layout
from layouts.admin_side import create_admin_layout
from layouts.login_page import create_login_layout

from modules.model_instances import init_models

print("Initializing AI models...")
init_models()
print("AI models initialized successfully!")

GREEN = "#132a13"
BG = "rgba(177, 183, 143, 0.3)"

app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id="app-user-info", storage_type="session", data=None),
    html.Div(id='navbar-container', style={'position': 'fixed', 'top': '0', 'left': '0', 'right': '0', 'zIndex': '999', 'height': '60px', 'backgroundColor': GREEN}),
    html.Div(id='page-content', style={'paddingTop': '60px'})
], style={'height': '100vh'})

@callback(
    Output('navbar-container', 'children'),
    [Input('app-user-info', 'data')]
)
def update_navbar(user_info):
    if user_info:
        return dbc.Navbar([
            dbc.NavbarBrand("Image Tree", className="me-auto"),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Home", href="/")),
                dbc.NavItem(dbc.NavLink("Statistics", href="/statistics")),
                dbc.NavItem(dbc.NavLink(f"Logged in as {user_info.get('username')}", href="#")),
                dbc.NavItem(dbc.NavLink("Logout", href="/logout", id="logout-link")),
            ], className="ms-auto", navbar=True)
        ], color=GREEN, dark=True, style={'backgroundColor': GREEN, 'borderColor': GREEN, 'height': '100%', 'margin': '0', 'borderRadius': '0', 'paddingLeft': '20px', 'paddingRight': '20px'})
    return html.Div()

@callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('app-user-info', 'data')],
    prevent_initial_call=False
)
def display_page(pathname, user_info):
    if pathname == '/logout':
        return create_login_layout()
    
    if not user_info:
        return create_login_layout()
    
    if pathname == '/login' or pathname == '/register' or pathname == '/':
        return create_user_layout()
    
    if pathname == '/statistics':
        return create_admin_layout()
    
    return create_user_layout()

@callback(
    Output('app-user-info', 'data'),
    [Input('user-info', 'data')],
    [State('app-user-info', 'data')],
    prevent_initial_call=True
)
def update_app_user_info(login_user_data, current_app_user_data):
    if login_user_data is None:
        raise PreventUpdate
    
    print(f"Updating app user info with: {login_user_data}")
    return login_user_data

@callback(
    Output('app-user-info', 'data', allow_duplicate=True),
    [Input('url', 'pathname')],
    [State('app-user-info', 'data')],
    prevent_initial_call=True
)
def handle_logout(pathname, current_user_data):
    if pathname == '/logout' and current_user_data is not None:
        return None
    raise PreventUpdate

if __name__ == '__main__':
    cyto.load_extra_layouts()
    app.run_server(debug=True)