import dash
from dash import dcc, html, callback, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from layouts.user_page import create_user_layout
from layouts.admin_page import create_admin_layout
from layouts.login_page import create_login_layout

app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id="app-user-info", storage_type="session", data=None),
    html.Div(id='navbar-container'),
    html.Div(id='page-content')
])

@callback(
    Output('navbar-container', 'children'),
    [Input('app-user-info', 'data')]
)
def update_navbar(user_info):
    if user_info:
        return html.Div([
            dbc.NavbarSimple(
                children=[
                    dbc.NavItem(dbc.NavLink("User Interface", href="/")),
                    dbc.NavItem(dbc.NavLink("Admin Dashboard", href="/admin")),
                    dbc.NavItem(dbc.NavLink(f"Logged in as {user_info.get('username')}", href="#")),
                    dbc.NavItem(dbc.NavLink("Logout", href="/logout", id="logout-link")),
                ],
                brand="Image Generator",
                color="primary",
                dark=True,
            ),
        ])
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
    
    if pathname == '/admin':
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