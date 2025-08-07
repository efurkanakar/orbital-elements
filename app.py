import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

def kepler_3d(a, e, inc, w, Om, f):
    r = a*(1 - e**2)/(1 + e*np.cos(f))
    x, y = r*np.cos(f), r*np.sin(f)
    z    = np.zeros_like(f)
    Rz = lambda t: np.array([[np.cos(t), -np.sin(t), 0],
                             [np.sin(t),  np.cos(t), 0],
                             [0,           0,        1]])
    Rx = lambda t: np.array([[1, 0,           0],
                             [0, np.cos(t), -np.sin(t)],
                             [0, np.sin(t),  np.cos(t)]])
    return (Rz(Om) @ Rx(inc) @ Rz(w)) @ np.vstack((x, y, z))

# sabitler
sma, ecc = 1.0, 0.5
f_line   = np.linspace(0, 2*np.pi, 400)
initial_inc, initial_w, initial_Om = map(np.deg2rad, (45, 90, 90))

# Dash app
app    = Dash(__name__)
server = app.server

# yardımcı: bir figür oluştur
def make_figures(i_deg, w_deg, Om_deg):
    inc = np.deg2rad(i_deg)
    w   = np.deg2rad(w_deg)
    Om  = np.deg2rad(Om_deg)

    # yörünge hattı
    Xo, Yo, Zo = kepler_3d(sma, ecc, inc, w, Om, f_line)

    # 3D plane
    U = np.linspace(-1.2, 1.2, 30)
    UU, VV = np.meshgrid(U, U)
    plane = go.Surface(x=UU, y=VV, z=np.zeros_like(UU),
                       showscale=False, opacity=0.15,
                       colorscale=[[0,'cornflowerblue'],
                                   [1,'cornflowerblue']],
                       name="Plane")

    # 3D orbit trace
    trace3d = go.Scatter3d(x=Xo, y=Yo, z=Zo,
                           mode='lines',
                           line=dict(color='black', width=2),
                           name="Orbit")

    # 3D figür
    fig3d = go.Figure([plane, trace3d])
    fig3d.update_layout(
        scene=dict(
            xaxis=dict(range=[-1.5,1.5], title="North"),
            yaxis=dict(range=[-1.5,1.5], title="East"),
            zaxis=dict(range=[-1.5,1.5], title="LoS"),
            aspectmode='cube'
        ),
        margin=dict(l=0,r=0,b=0,t=30),
        title="3D Orbit"
    )

    # 2D projeksiyon (ΔRA = -Y, ΔDec = X)
    fig2d = go.Figure([
        go.Scatter(x=-Yo, y=Xo, mode='lines',
                   line=dict(color='black'), name="Orbit")
    ])
    fig2d.update_layout(
        xaxis=dict(range=[-1.5,1.5], autorange=False, title="ΔRA"),
        yaxis=dict(range=[-1.5,1.5], autorange=False, title="ΔDec"),
        margin=dict(l=40,r=10,b=40,t=30),
        title="Sky-Plane Projection",
        width=450, height=450
    )

    return fig3d, fig2d

# layout; ilk figürleri hemen yazıyoruz
init3d, init2d = make_figures(45, 90, 90)
app.layout = html.Div([
    html.Div([
        html.Label("Inclination i (°)"),
        dcc.Slider(0,180,step=1,value=45,id="slider-i",
                   marks={i:str(i) for i in range(0,181,30)},
                   updatemode='mouseup'),
        html.Br(),
        html.Label("Argument ω (°)"),
        dcc.Slider(0,360,step=1,value=90,id="slider-w",
                   marks={i:str(i) for i in range(0,361,60)},
                   updatemode='mouseup'),
        html.Br(),
        html.Label("Node Ω (°)"),
        dcc.Slider(0,360,step=1,value=90,id="slider-Om",
                   marks={i:str(i) for i in range(0,361,60)},
                   updatemode='mouseup'),
    ], style={"width":"20%","display":"inline-block","padding":"20px","verticalAlign":"top"}),

    html.Div([
        dcc.Graph(id="orbit-3d", figure=init3d,
                  style={"display":"inline-block","width":"49%"}),
        dcc.Graph(id="orbit-2d", figure=init2d,
                  style={"display":"inline-block","width":"49%"})
    ], style={"width":"75%","display":"inline-block","verticalAlign":"top"})
])

# callback sadece slider’lara bağlı
@app.callback(
    Output("orbit-3d","figure"),
    Output("orbit-2d","figure"),
    Input("slider-i","value"),
    Input("slider-w","value"),
    Input("slider-Om","value"),
)
def update(i_deg, w_deg, Om_deg):
    return make_figures(i_deg, w_deg, Om_deg)

if __name__=="__main__":
    app.run_server(debug=True)
