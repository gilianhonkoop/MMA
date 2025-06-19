
import dash
from dash import html, dcc, Input, Output
import plotly.graph_objs as go
import numpy as np

app = dash.Dash(__name__)
server = app.server

BG = "#FFEED6"
GREEN = "#38432E"
G_HEIGHT = 260  # Reduced for better vertical alignment

def line_chart(seed, title, subtitle=None):
    np.random.seed(seed)
    x = np.arange(1, 7)
    y1 = np.random.uniform(0.2, 0.6, 6)
    y2 = np.random.uniform(0.2, 0.6, 6)
    fig = go.Figure([
        go.Scatter(x=x, y=y1, mode="lines+markers",
                   line=dict(color="#7B002C", width=2),
                   marker=dict(symbol="circle", size=6)),
        go.Scatter(x=x, y=y2, mode="lines+markers",
                   line=dict(color="#00008B", width=2),
                   marker=dict(symbol="circle", size=6))
    ])
    fig.update_layout(
        title=f"{title}<br><sup>{subtitle or ''}</sup>",
        height=G_HEIGHT,
        font=dict(family="sans-serif", size=13),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(t=50, b=30, l=40, r=40),
        title_font_size=14
    )
    fig.update_xaxes(title="Generation", showgrid=False)
    fig.update_yaxes(title="Score", showgrid=False, range=[0, 1])
    return fig

def heatmap_chart():
    np.random.seed(42)
    x = np.linspace(0, 1, 10)
    y = np.arange(1, 11)
    y_norm = (y - 1) / 9
    X, Y = np.meshgrid(x, y_norm)
    z = (1 - np.hypot(X - 0.5, Y - 0.5) / 0.7071) * np.random.rand(*X.shape)
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y,
                               colorscale=[[0, "#D2B48C"], [1, "#7B002C"]],
                               showscale=False))
    fig.update_layout(
        title="Strength–Guidance Balance",
        height=G_HEIGHT,
        font=dict(family="sans-serif", size=13),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(t=50, b=30, l=40, r=40)
    )
    fig.update_xaxes(title="Strength")
    fig.update_yaxes(title="Guidance")
    return fig

def bar_chart(seed, title, subtitle=None):
    np.random.seed(seed)
    data = np.random.randint(0, 101, (4, 2))
    labels = ["S", "E", "S+E", "No AI"]
    fig = go.Figure([
        go.Bar(x=labels, y=data[:, 0], marker_color="#7B002C"),
        go.Bar(x=labels, y=data[:, 1], marker_color="#00008B")
    ])
    fig.update_layout(
        title=f"{title}<br><sup>{subtitle or ''}</sup>",
        barmode="group",
        height=G_HEIGHT,
        font=dict(family="sans-serif", size=13),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(t=50, b=30, l=40, r=40),
        showlegend=False
    )
    fig.update_yaxes(range=[0, 100], title="Score")
    return fig

def pie_chart():
    np.random.seed(42)
    values = np.random.randint(1, 100, 4)
    labels = ["Suggestions", "Enhancement", "S + E", "No AI"]
    colors = ["#7B002C", "#00008B", "#8B4513", "#006400"]
    fig = go.Figure(go.Pie(labels=labels, values=values,
                           marker_colors=colors, textinfo="label+percent"))
    fig.update_layout(
        title="Utility (rates of use)",
        height=G_HEIGHT,
        font=dict(family="sans-serif", size=13),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        margin=dict(t=50, b=30, l=40, r=40),
        showlegend=False
    )
    return fig

def get_figures(tab):
    seeds = {
        "overall": [13, 26, 5],
        "suggestions": [11, 21, 8],
        "enhancement": [22, 33, 44],
        "both": [55, 66, 77],
        "noai": [99, 101, 88]
    }[tab]

    return [
        line_chart(seeds[0], "Strength (normalised)", "Over generations"),
        line_chart(seeds[2], "Aesthetic (BRISQUE)", "Image–Prompt Fidelity"),
        heatmap_chart(),
        line_chart(seeds[1], "Prompt change (Bertscore)", "Image change (LPIPS)"),
        bar_chart(13, "Prompt change (Bertscore), mean", "Image change (LPIPS), mean"),
        bar_chart(77, "Aesthetic (BRISQUE), mean", "Image–Prompt Fidelity, mean"),
        pie_chart()
    ]

app.layout = html.Div([
    dcc.Store(id="insight-tab-store", data="overall"),
    html.Div(style={"backgroundColor": GREEN, "color": "white", "padding": "16px 20px"}, children=[
        html.Span("AI-D", style={"fontWeight": "bold", "fontSize": "26px", "marginRight": "10px"}),
        html.Span("|", style={"margin": "0 10px"}),
        html.Span("Developer", style={"fontStyle": "italic", "fontSize": "22px"})
    ]),
    html.Div(style={"display": "flex", "padding": "16px 20px 0 20px"}, children=[
        html.Div("Parameter Insight", style={"width": "25%", "fontSize": "20px", "fontWeight": "600", "color": GREEN}),
        html.Div("Prompt-Image Insight", style={"width": "50%", "textAlign": "center", "fontSize": "20px", "fontWeight": "600", "color": GREEN}),
        html.Div("Dialogue history", style={"width": "25%", "textAlign": "right", "fontSize": "20px", "fontWeight": "600", "color": GREEN})
    ]),
    html.Div(style={"width": "50%", "marginLeft": "25%", "marginBottom": "10px"}, children=[
        dcc.Tabs(id="insight-tab", value="overall", children=[
            dcc.Tab(label="Overall", value="overall"),
            dcc.Tab(label="Suggestions", value="suggestions"),
            dcc.Tab(label="Enhancement", value="enhancement"),
            dcc.Tab(label="Suggestions & Enhancement", value="both"),
            dcc.Tab(label="No AI", value="noai"),
        ])
    ]),
    html.Div(id="grid-content")
])

@app.callback(
    Output("grid-content", "children"),
    Input("insight-tab-store", "data")
)
def render_figures(tab):
    figs = get_figures(tab)
    return html.Div([
        html.Div(style={"display": "flex"}, children=[
            html.Div(dcc.Graph(figure=figs[0]), style={"width": "25%"}),
            html.Div(dcc.Graph(figure=figs[3]), style={"width": "25%"}),
            html.Div(dcc.Graph(figure=figs[1]), style={"width": "25%"}),
            html.Div(style={"width": "25%", "padding": "10px"}, children=[
                html.Ul([
                    html.Li("Fragrance Bottle Stone"),
                    html.Li("Water Bottle Sticker"),
                    html.Li("Water Bottle Sticker Funky Urban"),
                    html.Li("Necklace Presentation Soft Cushion")
                ], style={"listStyle": "none", "padding": 0, "lineHeight": "1.6em"})
            ])
        ]),
        html.Div(style={"display": "flex"}, children=[
            html.Div(dcc.Graph(figure=figs[2]), style={"width": "25%"}),
            html.Div(dcc.Graph(figure=figs[4]), style={"width": "25%"}),
            html.Div(dcc.Graph(figure=figs[5]), style={"width": "25%"}),
            html.Div(dcc.Graph(figure=figs[6]), style={"width": "25%"})
        ])
    ])

@app.callback(
    Output("insight-tab-store", "data"),
    Input("insight-tab", "value")
)
def update_tab(tab): return tab

if __name__ == "__main__":
    app.run_server(debug=True)
