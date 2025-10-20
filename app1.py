# -*- coding: utf-8 -*-
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="3D Staging Demo", layout="wide")

# ---- Axes ranges ----
X_MIN, X_MAX = 0, 3   # ypN (x)
Y_MIN, Y_MAX = 0, 3   # TRG (y)
Z_MIN, Z_MAX = 0, 4   # ypT (z)

# ---- Staging config (4 stages => 3 thresholds) ----
RADII_THRESHOLDS  = [1.3, 3.8, 5.0]   # draw 3 boundary surfaces
STAGE_LABELS = ['Stage I', 'Stage II', 'Stage IIIA', 'Stage IIIB']  # 4 stages
STAGE_COLORS = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']         # 4 colors

assert len(STAGE_LABELS) == len(STAGE_COLORS) == len(RADII_THRESHOLDS) + 1, \
    "For N stages, provide N labels/colors and N-1 (or N?) thresholds. Here we use N-1=3 thresholds for 4 stages."

def stage_of(x, y, z):
    """Return (stage_idx, stage_label, stage_color, r) based on Euclidean distance."""
    r = float(np.hypot(np.hypot(x, y), z))  # sqrt(x^2 + y^2 + z^2)
    # right-closed bins: <= t1 -> stage 0; <= t2 -> stage 1; <= t3 -> stage 2; else stage 3
    if r <= RADII_THRESHOLDS[0]:
        idx = 0
    elif r <= RADII_THRESHOLDS[1]:
        idx = 1
    elif r <= RADII_THRESHOLDS[2]:
        idx = 2
    else:
        idx = 3
    return idx, STAGE_LABELS[idx], STAGE_COLORS[idx], r

@st.cache_data
def quarter_surfaces(n=220):
    """Generate 3 quarter-sphere boundary surfaces at the given thresholds."""
    traces = []
    theta = np.linspace(0, np.pi/2, n)   # polar
    phi   = np.linspace(0, np.pi/2, n)   # azimuth
    T, P  = np.meshgrid(theta, phi)

    for r, color, name in zip(RADII_THRESHOLDS, STAGE_COLORS[:-1], STAGE_LABELS[:-1]):
        X = r*np.sin(T)*np.cos(P)
        Y = r*np.sin(T)*np.sin(P)
        Z = r*np.cos(T)
        m = (X>=X_MIN)&(X<=X_MAX)&(Y>=Y_MIN)&(Y<=Y_MAX)&(Z>=Z_MIN)&(Z<=Z_MAX)
        X = np.where(m, X, np.nan); Y = np.where(m, Y, np.nan); Z = np.where(m, Z, np.nan)
        traces.append(go.Surface(
            x=X, y=Y, z=Z, name=f"Boundary {name}",
            showscale=False, opacity=0.25,
            colorscale=[[0, color],[1, color]],
            hoverinfo="skip"
        ))
    return traces

def cube_edges():
    V = np.array([
        [X_MIN,Y_MIN,Z_MIN],[X_MAX,Y_MIN,Z_MIN],[X_MAX,Y_MAX,Z_MIN],[X_MIN,Y_MAX,Z_MIN],
        [X_MIN,Y_MIN,Z_MAX],[X_MAX,Y_MIN,Z_MAX],[X_MAX,Y_MAX,Z_MAX],[X_MIN,Y_MAX,Z_MAX]
    ], float)
    E = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    traces = []
    for i,j in E:
        traces.append(go.Scatter3d(
            x=[V[i,0],V[j,0]], y=[V[i,1],V[j,1]], z=[V[i,2],V[j,2]],
            mode="lines", line=dict(color="black", width=4),
            hoverinfo="skip", showlegend=False
        ))
    return traces

@st.cache_data
def lattice():
    """All integer grid points in the cuboid, colored by stage."""
    xs, ys, zs, cols = [], [], [], []
    for xi in range(X_MIN, X_MAX+1):
        for yi in range(Y_MIN, Y_MAX+1):
            for zi in range(Z_MIN, Z_MAX+1):
                idx,_,c,_ = stage_of(xi, yi, zi)
                xs.append(xi); ys.append(yi); zs.append(zi); cols.append(c)
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="markers",
        marker=dict(size=3.5, color=cols, line=dict(width=0.5, color="black")),
        name="Integer grid points", showlegend=False,
        hovertemplate="ypN=%{x}<br>TRG=%{y}<br>ypT=%{z}<extra></extra>"
    )

# ---- Sidebar inputs ----
with st.sidebar:
    st.header("Input Parameters")
    trg = st.number_input("TRG", 0, 3, 1, step=1)
    ypn = st.number_input("ypN", 0, 3, 1, step=1)
    ypt = st.number_input("ypT", 0, 4, 1, step=1)
    show = st.checkbox("Show all integer lattice points", True)

st.title("Gastric Cancer TRG–ypT–ypN 3D Staging")

fig = go.Figure()
for tr in cube_edges(): fig.add_trace(tr)
for tr in quarter_surfaces(): fig.add_trace(tr)
if show: fig.add_trace(lattice())

idx, label, color, r = stage_of(ypn, trg, ypt)
fig.add_trace(go.Scatter3d(
    x=[ypn], y=[trg], z=[ypt], mode="markers",
    marker=dict(size=8, color=color, line=dict(width=1, color="black")),
    name="Input point", showlegend=False,
    hovertemplate="<b>Input</b><br>ypN=%{x}<br>TRG=%{y}<br>ypT=%{z}<extra></extra>"
))

fig.update_scenes(
    xaxis=dict(title="ypN", range=[X_MIN, X_MAX], dtick=1, zeroline=False),
    yaxis=dict(title="TRG", range=[Y_MIN, Y_MAX], dtick=1, zeroline=False),
    zaxis=dict(title="ypT", range=[Z_MIN, Z_MAX], dtick=1, zeroline=False),
    camera=dict(eye=dict(x=1.6, y=1.4, z=1.1))
)
fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=720)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Staging Result")
st.markdown(
    f"- **Stage**: <span style='color:{color};font-weight:700'>{label}</span><br>"
    f"- **r** (distance to origin): **{r:.3f}**",
    unsafe_allow_html=True
)
