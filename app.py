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

def rodrigues(v, k, th):
    v, k = np.asarray(v), np.asarray(k)
    return (v * np.cos(th) +
            np.cross(k, v) * np.sin(th) +
            k * np.dot(k, v) * (1 - np.cos(th)))

# ——— GLOBAL SABİTLER ———
sma, ecc = 1.0, 0.5
N_fr      = 200
M_seq     = np.linspace(0, 2*np.pi, N_fr)
E_seq     = solve_kepler(M_seq, ecc)
f_true    = (2 * np.arctan2(np.sqrt(1+ecc) * np.sin(E_seq/2),
                            np.sqrt(1-ecc) * np.cos(E_seq/2))) % (2*np.pi)
f_line    = np.linspace(0, 2*np.pi, 400)

# ——— DASH UYGULAMASI ———
app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.Div([
        html.Label("İnclination i (°)"),
        dcc.Slider(
            0, 180, step=1, value=45, id="slider-i",
            marks={i: str(i) for i in range(0, 181, 30)},
            updatemode='mouseup'
        ),
        html.Br(),
        html.Label("Argument ω (°)"),
        dcc.Slider(
            0, 360, step=1, value=90, id="slider-w",
            marks={i: str(i) for i in range(0, 361, 60)},
            updatemode='mouseup'
        ),
        html.Br(),
        html.Label("Node Ω (°)"),
        dcc.Slider(
            0, 360, step=1, value=90, id="slider-Om",
            marks={i: str(i) for i in range(0, 361, 60)},
            updatemode='mouseup'
        ),
        html.Br(),
        html.Button("▶ Play / || Pause", id="btn-play", n_clicks=0),
        dcc.Interval(id="interval", interval=50, disabled=True)
    ], style={"width":"20%", "display":"inline-block", "verticalAlign":"top", "padding":"20px"}),

    html.Div([
        dcc.Graph(id="orbit-3d", style={"height":"50vh"}),
        dcc.Graph(id="orbit-2d", style={"height":"45vh"})
    ], style={"width":"75%", "display":"inline-block", "padding":"20px"})
])

@app.callback(
    Output("interval", "disabled"),
    Input("btn-play", "n_clicks"),
    State("interval", "disabled"),
)
def toggle_interval(n, disabled):
    return not disabled

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
    k   = frame_idx % N_fr
    ν   = f_true[k]

    # Orbit hattı
    Xo, Yo, Zo = kepler_3d(sma, ecc, inc, w, Om, f_line)
    # Body pozisyonu
    Xb, Yb, Zb = kepler_3d(sma, ecc, inc, w, Om, np.array([ν]))

    # Plane yüzeyi
    U = np.linspace(-1.2, 1.2, 30)
    UU, VV = np.meshgrid(U, U)
    plane = go.Surface(
        x=UU, y=VV, z=np.zeros_like(UU),
        showscale=False, opacity=0.15,
        colorscale=[[0,'cornflowerblue'],[1,'cornflowerblue']],
        name="Orbital plane"
    )

    # Eksen okları
    axes = []
    for vec, lbl in [([1.2,0,0],"North"),([0,1.2,0],"East"),([0,0,1.2],"LoS")]:
        axes.append(go.Scatter3d(
            x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
            mode='lines+text',
            line=dict(color='gray', width=2),
            text=[None, lbl], textposition="top",
            showlegend=False
        ))

    # Node hattı
    t_nodes = np.array([-1.2, 1.2])
    x_nd = t_nodes * np.cos(Om)
    y_nd = t_nodes * np.sin(Om)
    nodes_line = go.Scatter3d(
        x=x_nd, y=y_nd, z=[0,0],
        mode='lines', line=dict(color='gray', width=2, dash='dash'),
        name="Line of nodes"
    )

    # Asc/Des node
    f_asc = (2*np.pi - w) % (2*np.pi)
    Xa, Ya, Za = kepler_3d(sma, ecc, inc, w, Om, np.array([f_asc]))
    Xd, Yd, Zd = kepler_3d(sma, ecc, inc, w, Om, np.array([f_asc+np.pi]))
    asc_node = go.Scatter3d(x=[Xa[0]], y=[Ya[0]], z=[Za[0]],
        mode='markers', marker=dict(symbol='triangle-up', size=6, color='dodgerblue'),
        name="Ascending node")
    des_node = go.Scatter3d(x=[Xd[0]], y=[Yd[0]], z=[Zd[0]],
        mode='markers', marker=dict(symbol='triangle-down', size=6, color='firebrick'),
        name="Descending node")

    # Periastron
    peri = go.Scatter3d(x=[Xo[0]], y=[Yo[0]], z=[Zo[0]],
        mode='markers', marker=dict(symbol='diamond', size=6, color='gold'),
        name="Periastron")

    # Ω yay
    th_Om = np.linspace(0, Om, 60)
    Om_arc = go.Scatter3d(
        x=1.05*np.cos(th_Om), y=1.05*np.sin(th_Om), z=[0]*60,
        mode='lines', line=dict(color='seagreen', width=3),
        name="Ω arc"
    )

    # ω yay
    sign = +1 if inc < np.pi/2 else -1
    th_w = np.linspace(0, w, 60)
    f_arc = (f_asc + sign*th_w) % (2*np.pi)
    W_X, W_Y, W_Z = kepler_3d(sma, ecc, inc, w, Om, f_arc)
    w_arc = go.Scatter3d(
        x=W_X, y=W_Y, z=W_Z,
        mode='lines', line=dict(color='darkorange', width=3),
        name="ω arc"
    )

    # Inclination wedge için
    axis = np.cross([0,0,1], [np.sin(inc)*np.sin(Om),
                              -np.sin(inc)*np.cos(Om),
                              np.cos(inc)])
    sin_i = np.linalg.norm(axis)
    if sin_i > 1e-6:
        l_hat = axis/sin_i
        e1 = np.cross(l_hat, [0,0,1]); e1 /= np.linalg.norm(e1)
        pts = np.vstack([rodrigues(e1*1.05, l_hat, th)
                         for th in np.linspace(0, inc, 40)])
        wedge = go.Mesh3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            opacity=0.35, color='cyan',
            alphahull=0, name="Inclination wedge"
        )
    else:
        wedge = None

    # 3D figür toplama
    traces3d = [
        go.Scatter3d(x=Xo, y=Yo, z=Zo,
                     mode='lines', line=dict(color='black', width=2),
                     name="Orbit"),
        plane,
        go.Scatter3d(x=[0], y=[0], z=[0],
                     mode='markers', marker=dict(size=6, color='black'),
                     name="Focus"),
        go.Scatter3d(x=[Xb], y=[Yb], z=[Zb],
                     mode='markers', marker=dict(size=8, color='purple'),
                     name="Body"),
        nodes_line, asc_node, des_node, peri, Om_arc, w_arc,
        *axes
    ]
    if wedge is not None:
        traces3d.append(wedge)

    fig3d = go.Figure(traces3d)
    fig3d.update_layout(
        scene=dict(xaxis_title="North", yaxis_title="East", zaxis_title="LoS",
                   aspectmode='cube'),
        legend=dict(x=0.75, y=0.9),
        margin=dict(l=0, r=0, b=0, t=30),
        title=f"3D Orbit Geometry (frame {k+1}/{N_fr})"
    )

    # 2D Projeksyon
    fig2d = go.Figure([
        go.Scatter(x=-Yo, y=Xo, mode='lines',
                   line=dict(color='black'), name="Orbit"),
        go.Scatter(x=[-Yb], y=[Xb], mode='markers',
                   marker=dict(size=8, color='purple'), name="Body"),
        go.Scatter(x=-y_nd, y=x_nd, mode='lines',
                   line=dict(color='gray', width=2, dash='dash'),
                   name="Line of nodes"),
        go.Scatter(x=[-Ya[0]], y=[Xa[0]], mode='markers',
                   marker=dict(symbol='triangle-up', size=6, color='dodgerblue'),
                   name="Asc node"),
        go.Scatter(x=[-Yd[0]], y=[Xd[0]], mode='markers',
                   marker=dict(symbol='triangle-down', size=6, color='firebrick'),
                   name="Des node"),
        go.Scatter(x=[-Yo[0]], y=[Xo[0]], mode='markers',
                   marker=dict(symbol='diamond', size=6, color='gold'),
                   name="Periastron"),
        go.Scatter(x=1.05*np.cos(th_Om), y=1.05*np.sin(th_Om),
                   mode='lines', line=dict(color='seagreen', width=3),
                   name="Ω arc"),
        go.Scatter(x=-W_Y, y=W_X, mode='lines',
                   line=dict(color='darkorange', width=3), name="ω arc"),
        go.Scatter(x=np.hstack(([0], -pts[:,1], [0])),
                   y=np.hstack(([0], pts[:,0], [0])),
                   fill='toself', fillcolor='cyan',
                   opacity=0.3, line=dict(width=0),
                   name="Inclination wedge")
    ])
    fig2d.update_layout(
        xaxis=dict(title="ΔRA", autorange='reversed', scaleanchor="y", scaleratio=1),
        yaxis=dict(title="ΔDec"),
        margin=dict(l=40, r=10, b=40, t=30),
        title="Sky-Plane Projection"
    )

    return fig3d, fig2d

if __name__ == "__main__":
    app.run_server(debug=True)
