import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# ——— YÖRÜNGE HESAPLAMA ———
def kepler_3d(a, e, inc, w, Om, f):
    r = a * (1 - e**2) / (1 + e * np.cos(f))
    x, y = r * np.cos(f), r * np.sin(f)
    z    = np.zeros_like(f)
    Rz = lambda t: np.array([
        [ np.cos(t), -np.sin(t), 0],
        [ np.sin(t),  np.cos(t), 0],
        [ 0,           0,        1]
    ])
    Rx = lambda t: np.array([
        [1,           0,        0],
        [0, np.cos(t), -np.sin(t)],
        [0, np.sin(t),  np.cos(t)]
    ])
    return (Rz(Om) @ Rx(inc) @ Rz(w)) @ np.vstack((x, y, z))

# ——— SABİTLER ———
sma, ecc = 1.0, 0.5
# Daha hızlı güncelleme için nokta sayısını yarıya indirdik
f_line = np.linspace(0, 2*np.pi, 200)

# Orbital düzlemi mesh’i (statik)
U = np.linspace(-1.2, 1.2, 10)
UU, VV = np.meshgrid(U, U)

# ——— DASH UYGULAMASI ———
app    = Dash(__name__)
server = app.server

def make_base_figures(i_deg, w_deg, Om_deg):
    # Slider değerlerini radyana çevir
    inc = np.deg2rad(i_deg)
    w   = np.deg2rad(w_deg)
    Om  = np.deg2rad(Om_deg)

    # 3D orbit hattı
    Xo, Yo, Zo = kepler_3d(sma, ecc, inc, w, Om, f_line)

    # Statik orbital düzlemi
    plane = go.Surface(
        x=UU, y=VV, z=np.zeros_like(UU),
        showscale=False, opacity=0.15,
        colorscale=[[0,'cornflowerblue'],[1,'cornflowerblue']],
        name="Orbital plane"
    )

    # Dinamik orbit trace
    orbit3d = go.Scatter3d(
        x=Xo, y=Yo, z=Zo,
        mode='lines', line=dict(color='black', width=2),
        name="Orbit"
    )

    fig3d = go.Figure([plane, orbit3d])
    fig3d.update_layout(
        scene=dict(
            xaxis=dict(range=[-1.5,1.5], title="North"),
            yaxis=dict(range=[-1.5,1.5], title="East"),
            zaxis=dict(range=[-1.5,1.5], title="LoS"),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="3D Orbit"
    )

    # 2D projeksiyon (ΔRA = -Y, ΔDec = X)
    orbit2d = go.Scatter(
        x=-Yo, y=Xo,
        mode='lines', line=dict(color='black'),
        name="Orbit"
    )
    fig2d = go.Figure([orbit2d])
    fig2d.update_layout(
        xaxis=dict(range=[-1.5,1.5], autorange=False, title="ΔRA"),
        yaxis=dict(range=[-1.5,1.5], autorange=False, title="ΔDec"),
        margin=dict(l=40, r=10, b=40, t=30),
        title="Sky-Plane Projection"
    )

    return fig3d, fig2d

# Başlangıç figürlerini oluştur
base3d, base2d = make_base_figures(45, 90, 90)

# Layout
app.layout = html.Div([
    html.Div([
        html.Label("Inclination i (°)"),
        dcc.Slider(0, 180, step=1, value=45, id="slider-i",
                   marks={i: str(i) for i in range(0,181,30)},
                   updatemode='mouseup'),
        html.Br(),
        html.Label("Argument ω (°)"),
        dcc.Slider(0, 360, step=1, value=90, id="slider-w",
                   marks={i: str(i) for i in range(0,361,60)},
                   updatemode='mouseup'),
        html.Br(),
        html.Label("Node Ω (°)"),
        dcc.Slider(0, 360, step=1, value=90, id="slider-Om",
                   marks={i: str(i) for i in range(0,361,60)},
                   updatemode='mouseup'),
    ], style={"width":"20%","display":"inline-block",
              "verticalAlign":"top","padding":"20px"}),

    html.Div([
        dcc.Graph(id="orbit-3d", figure=base3d,
                  style={"display":"inline-block","width":"49%"}),
        dcc.Graph(id="orbit-2d", figure=base2d,
                  style={"display":"inline-block","width":"49%"})
    ], style={"width":"75%","display":"inline-block",
              "verticalAlign":"top"})
])

# Callback: sadece orbit trace’lerini güncelle
@app.callback(
    Output("orbit-3d","figure"),
    Output("orbit-2d","figure"),
    Input("slider-i","value"),
    Input("slider-w","value"),
    Input("slider-Om","value"),
)
def update(i_deg, w_deg, Om_deg):
    # Yeni orbit hattını hesapla
    inc = np.deg2rad(i_deg)
    w   = np.deg2rad(w_deg)
    Om  = np.deg2rad(Om_deg)
    Xo, Yo, Zo = kepler_3d(sma, ecc, inc, w, Om, f_line)

    # Sadece 3D figürdeki orbit trace (index 1) güncellensin
    fig3d = base3d
    fig3d.data[1].x = Xo
    fig3d.data[1].y = Yo
    fig3d.data[1].z = Zo

    # Sadece 2D figürdeki orbit trace (index 0) güncellensin
    fig2d = base2d
    fig2d.data[0].x = -Yo
    fig2d.data[0].y =  Xo

    return fig3d, fig2d

if __name__ == "__main__":
    app.run_server(debug=True)
