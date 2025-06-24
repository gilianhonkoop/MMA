from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from user_page import create_user_layout
from admin_side_for_sync import create_admin_layout  # updated import

app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="app-user-info", data={"user_id": 1}),
    dbc.NavbarSimple(
        brand="AI-D Platform",
        color="primary",
        dark=True,
        children=[
            dbc.NavItem(dbc.NavLink("User Page", href="/user")),
            dbc.NavItem(dbc.NavLink("Admin Page", href="/admin")),
        ]
    ),
    html.Div(id="page-content")
])

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    print(f"🔁 Routing to: {pathname}")
    if pathname == "/" or pathname == "/user":
        return create_user_layout()
    elif pathname == "/admin":
        return create_admin_layout()
    else:
        return html.Div("404 Not Found")

if __name__ == "__main__":
    app.run(debug=True)
