import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

from layouts.admin_side import create_admin_layout

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)

# App layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id="app-user-info", storage_type="session", data={"username": "admin", "user_id": 1}),
    html.Div(id='navbar-container', style={'position': 'fixed', 'top': '0', 'left': '0', 'right': '0', 'zIndex': '999', 'height': '60px', 'backgroundColor': '#38432E'}),
    html.Div(id='page-content', style={'paddingTop': '60px'})
], style={'height': '100vh'})
# style={'height': '100vh', 'overflow': 'hidden'})

# Dummy navbar (auto-logged in as admin for local testing)
@callback(
    Output('navbar-container', 'children'),
    [Input('app-user-info', 'data')]
)
def update_navbar(user_info):
    return dbc.Navbar([
        dbc.NavbarBrand("Statistics Dashboard", className="me-auto"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Statistics", href="/statistics")),
        ], className="ms-auto", navbar=True)
    ], color='#38432E', dark=True)

# Render only the admin layout
@callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')],
    prevent_initial_call=False
)
def display_page(pathname):
    return create_admin_layout()

if __name__ == '__main__':
    cyto.load_extra_layouts()
    app.run_server(debug=True)
    #app.run(debug=True)
