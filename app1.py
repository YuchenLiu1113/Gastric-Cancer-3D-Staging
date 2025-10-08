# -*- coding: utf-8 -*-
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="3D Staging Demo", layout="wide")

X_MIN, X_MAX = 0, 3   # ypN
Y_MIN, Y_MAX = 0, 3   # TRG
Z_MIN, Z_MAX = 0, 4   # ypT

RADII  = [1.001, 3.001, 4.20, 5.001,7]
COLORS = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#4C1F3A']
LABELS = ['Stage I', 'Stage II', 'Stage IIIA', 'Stage IIIB', 'Stage IIIC']

def stage_of(x, y, z):
    r = (x*x + y*y + z*z) ** 0.5
    idx = next((i for i, R in enumerate(RADII) if r <= R), len(RADII)-1)
    return idx, LABELS[idx], COLORS[idx], r

@st.cache_data
def quarter_surfaces(n=220):
    traces = []
    theta = np.linspace(0, np.pi/2, n)
    phi   = np.linspace(0, np.pi/2, n)
    T, P  = np.meshgrid(theta, phi)
    for r, color, name in zip(RADII, COLORS, LABELS):
        X = r*np.sin(T)*np.cos(P)
        Y = r*np.sin(T)*np.sin(P)
        Z = r*np.cos(T)
        m = (X>=X_MIN)&(X<=X_MAX)&(Y>=Y_MIN)&(Y<=Y_MAX)&(Z>=Z_MIN)&(Z<=Z_MAX)
        X = np.where(m, X, np.nan); Y = np.where(m, Y, np.nan); Z = np.where(m, Z, np.nan)
        traces.append(go.Surface(x=X, y=Y, z=Z, name=name, showscale=False,
                                 opacity=0.25, colorscale=[[0,color],[1,color]],
                                 hoverinfo="skip"))
    return traces

def cube_edges():
    V = np.array([[X_MIN,Y_MIN,Z_MIN],[X_MAX,Y_MIN,Z_MIN],[X_MAX,Y_MAX,Z_MIN],[X_MIN,Y_MAX,Z_MIN],
                  [X_MIN,Y_MIN,Z_MAX],[X_MAX,Y_MIN,Z_MAX],[X_MAX,Y_MAX,Z_MAX],[X_MIN,Y_MAX,Z_MAX]], float)
    E = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    traces = []
    for i,j in E:
        traces.append(go.Scatter3d(x=[V[i,0],V[j,0]], y=[V[i,1],V[j,1]], z=[V[i,2],V[j,2]],
                                   mode="lines", line=dict(color="black", width=4),
                                   hoverinfo="skip", showlegend=False))
    return traces

@st.cache_data
def lattice():
    xs, ys, zs, cols = [], [], [], []
    for xi in range(X_MIN, X_MAX+1):
        for yi in range(Y_MIN, Y_MAX+1):
            for zi in range(Z_MIN, Z_MAX+1):
                idx,_,c,_ = stage_of(xi, yi, zi)
                xs.append(xi); ys.append(yi); zs.append(zi); cols.append(c)
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="markers",
                        marker=dict(size=3.5, color=cols, line=dict(width=0.5, color="black")),
                        name="整数点", showlegend=False,
                        hovertemplate="ypN=%{x}<br>TRG=%{y}<br>ypT=%{z}<extra></extra>")

with st.sidebar:
    st.header("Input")
    trg = st.number_input("TRG", 0, 3, 1, step=1)
    ypn = st.number_input("ypN", 0, 3, 1, step=1)
    ypt = st.number_input("ypT", 0, 4, 1, step=1)
    show = st.checkbox("Show All Integer Grid Points", True)

st.title("Gastric Cancer TRG-ypT-ypN 3D Staging")

fig = go.Figure()
for tr in cube_edges(): fig.add_trace(tr)
for tr in quarter_surfaces(): fig.add_trace(tr)
if show: fig.add_trace(lattice())

idx, label, color, r = stage_of(ypn, trg, ypt)
fig.add_trace(go.Scatter3d(x=[ypn], y=[trg], z=[ypt], mode="markers",
                           marker=dict(size=8, color=color, line=dict(width=1, color="black")),
                           name="Input", showlegend=False,
                           hovertemplate="<b>Input</b><br>ypN=%{x}<br>TRG=%{y}<br>ypT=%{z}<extra></extra>"))
fig.update_scenes(xaxis=dict(title="ypN", range=[X_MIN,X_MAX], dtick=1),
                  yaxis=dict(title="TRG", range=[Y_MIN,Y_MAX], dtick=1),
                  zaxis=dict(title="ypT", range=[Z_MIN,Z_MAX], dtick=1),
                  camera=dict(eye=dict(x=1.6,y=1.4,z=1.1)))
fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=720)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Staging Result")
st.markdown(f"- **Stage**：<span style='color:{color};font-weight:700'>{label}</span><br>- **r**：**{r:.3f}**", unsafe_allow_html=True)
