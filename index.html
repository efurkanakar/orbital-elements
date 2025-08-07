import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State

# ——— YÖRÜNGE HESAPLAMA FONKSİYONLARI ———
def solve_kepler(M, e, tol=1e-10, n_iter=32):
    E = M.copy()
    for _ in range(n_iter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= dE
        if np.all(np.abs(dE) < tol):
            break
    return E

def kepler_3d(a, e, inc, w, Om, f):
    r = a * (1 - e**2) / (1 + e * np.cos(f))
    x, y = r * np.cos(f), r * np.sin(f)
    z = np.zeros_like(f)
    Rz = lambda t: np.array([[ np.cos(t), -np.sin(t), 0],
                             [ np.sin(t),  np.cos(t), 0],
                             [ 0,           0,        1]])
    Rx = lambda t: np.array([[1, 0,          0      ],
                             [0, np.cos(t), -np.sin(t)],
                             [0, np.sin(t),  np.cos(t)]])
    return (Rz(Om) @ Rx(inc) @ Rz(w)) @ np.vstack((x, y, z))

# ——— GLOBAL SABİTLER ———
sma, ecc = 1.0, 0.5
N_fr      = 200
M_seq     = np.linspace(0, 2*np.pi, N_fr)
E_seq     = solve_kepler(M_seq, ecc)
f_true    = (2 * np.arctan2(np.sqrt(1+ecc) * np.sin(E_seq/2),
                            np.sqrt(1-ecc) * np.cos(E_seq/2))) % (2*np.pi)
f_line    = np.linspace(0, 2*np.pi, 400)   # orbit çizgisi

# ——— DASH UYGULAMASI ———
app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Label("İnclination i (°)"),
            dcc.Slider(0, 180, step=1, value=45, id="slider-i")
        ], style={"padding":"0 20px"}),
        html.Div([
            html.Label("Argument ω (°)"),
            dcc.Slider(0, 360, step=1, value=90, id="slider-w")
        ], style={"padding":"0 20px"}),
        html.Div([
            html.Label("Node Ω (°)"),
            dcc.Slider(0, 360, step=1, value=90, id="slider-Om")
        ], style={"padding":"0 20px"}),
        html.Button("▶ Play / || Pause", id="btn-play", n_clicks=0),
        dcc.Interval(id="interval", interval=100, disabled=True)
    ], style={"width":"20%", "display":"inline-block", "verticalAlign":"top"}),

    html.Div([
        dcc.Graph(id="orbit-3d", style={"height":"45vh"}),
        dcc.Graph(id="orbit-2d", style={"height":"45vh"})
    ], style={"width":"75%", "display":"inline-block"})
])

# ——— CALLBACK: Play / Pause butonuna basınca interval’ı aç/kapa ———
@app.callback(
    Output("interval", "disabled"),
    Input("btn-play", "n_clicks"),
    State("interval", "disabled"),
)
def toggle_interval(n, currently_disabled):
    # her tıklamada tam tersine çevir
    return not currently_disabled

# ——— CALLBACK: Slider veya interval tetikleyince grafikleri güncelle ———
@app.callback(
    Output("orbit-3d", "figure"),
    Output("orbit-2d", "figure"),
    Input("slider-i", "value"),
    Input("slider-w", "value"),
    Input("slider-Om", "value"),
    Input("interval", "n_intervals"),
)
def update_graph(i_deg, w_deg, Om_deg, frame_idx):
    inc = np.deg2rad(i_deg)
    w   = np.deg2rad(w_deg)
    Om  = np.deg2rad(Om_deg)
    k   = frame_idx % N_fr     # sürekli dönen index
    ν   = f_true[k]

    # 1) Tam orbit çizgisi
    Xo, Yo, Zo = kepler_3d(sma, ecc, inc, w, Om, f_line)

    # 2) Güneş (origin) ve yörünge noktası
    Xb, Yb, Zb = kepler_3d(sma, ecc, inc, w, Om, np.array([ν]))

    # 3D Figure
    fig3d = go.Figure([
        go.Scatter3d(x=Xo, y=Yo, z=Zo, mode='lines', line=dict(color='black')),
        go.Scatter3d(x=[0],[0],[0], mode='markers', marker=dict(size=5,color='purple')),
        go.Scatter3d(x=[Xb],[Yb],[Zb], mode='markers', marker=dict(size=6,color='orange'))
    ])
    fig3d.update_layout(
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title=f"3D Orbit (frame {k+1}/{N_fr})"
    )

    # 2D projeksiyon: ΔRA vs ΔDec (basitçe X–Y’yi al diyelim)
    fig2d = go.Figure([
        go.Scatter(x=Xo, y=Yo, mode='lines', line=dict(color='black')),
        go.Scatter(x=[Xb], y=[Yb], mode='markers', marker=dict(size=8,color='orange'))
    ])
    fig2d.update_layout(
        xaxis_title="ΔRA", yaxis_title="ΔDec",
        margin=dict(l=40, r=10, b=40, t=30),
        title="Sky-Plane Projection"
    )

    return fig3d, fig2d

if __name__ == "__main__":
    app.run_server(debug=True)
