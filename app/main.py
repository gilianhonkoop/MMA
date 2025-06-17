import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from layouts.user_page import create_user_layout
from layouts.admin_page import create_admin_layout

app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("User Interface", href="/")),
                dbc.NavItem(dbc.NavLink("Admin Dashboard", href="/admin")),
            ],
        ),
    ]),
    html.Div(id='page-content')
])

@callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/admin':
        return create_admin_layout()
    else:
        return create_user_layout()

if __name__ == '__main__':
    cyto.load_extra_layouts()
    app.run_server(debug=True)