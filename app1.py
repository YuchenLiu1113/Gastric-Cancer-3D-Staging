# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Gastric Cancer 3D Staging", layout="wide")
APP_DIR = Path(__file__).resolve().parent

# ---- Axes ranges ----
X_MIN, X_MAX = 0, 3   # ypN (x)
Y_MIN, Y_MAX = 0, 3   # TRG (y)
Z_MIN, Z_MAX = 0, 4   # ypT (z)

# ---- 3D visual boundary config ----
RADII_THRESHOLDS = [1.3, 3.8, 5.05]

STAGE_LABELS = ["Stage I", "Stage II", "Stage IIIA", "Stage IIIB"]
STAGE_SHORT = ["I", "II", "IIIA", "IIIB"]
STAGE_COLORS = ["#2f68c5", "#2c9f59", "#e57f2e", "#c83d3d"]
STAGE_BACKGROUNDS = ["#b9ccef", "#bdddc8", "#f2c4a5", "#d77d76"]
UNCLASSIFIED_COLOR = "#9aa4b2"

# Staging matrix reconstructed from the supplied TRG/ypT/ypN table.
# Values are zero-based indexes into STAGE_LABELS.
TRG_STAGE_GRID = {
    1: [
        [1, 1, 1, 2],
        [1, 1, 1, 2],
        [1, 1, 1, 2],
        [1, 2, 2, 3],
    ],
    2: [
        [1, 1, 1, 2],
        [1, 1, 1, 2],
        [1, 1, 2, 2],
        [1, 2, 2, 3],
    ],
    3: [
        [1, 1, 2, 2],
        [1, 1, 2, 3],
        [1, 2, 2, 3],
        [2, 2, 3, 3],
    ],
}

STAGE_SURVIVAL_DATA = {
    0: {
        "stage": "Stage I",
        "image": "assets/survival/km_stage_i_curve.png",
        "1-year OS": "98.6%",
        "1-year 95% CI": "96.7%-100.0%",
        "3-year OS": "91.9%",
        "3-year 95% CI": "86.0%-98.3%",
    },
    1: {
        "stage": "Stage II",
        "image": "assets/survival/km_stage_ii_curve.png",
        "1-year OS": "92.4%",
        "1-year 95% CI": "90.2%-94.6%",
        "3-year OS": "78.1%",
        "3-year 95% CI": "74.1%-82.2%",
    },
    2: {
        "stage": "Stage IIIA",
        "image": "assets/survival/km_stage_iiia_curve.png",
        "1-year OS": "87.5%",
        "1-year 95% CI": "83.9%-91.2%",
        "3-year OS": "54.9%",
        "3-year 95% CI": "48.8%-61.7%",
    },
    3: {
        "stage": "Stage IIIB",
        "image": "assets/survival/km_stage_iiib_curve.png",
        "1-year OS": "79.8%",
        "1-year 95% CI": "74.8%-85.1%",
        "3-year OS": "45.8%",
        "3-year 95% CI": "39.2%-53.5%",
    },
}


def build_staging_map():
    mapping = {(0, 0, 0): 0}
    for trg_value, rows in TRG_STAGE_GRID.items():
        for ypn_value, row in enumerate(rows):
            for ypt_value, stage_idx in enumerate(row, start=1):
                mapping[(trg_value, ypn_value, ypt_value)] = stage_idx
    return mapping


STAGING_MAP = build_staging_map()


def stage_of(ypn, trg, ypt):
    """Return stage details for a selected ypN/TRG/ypT coordinate."""
    r = float(np.hypot(np.hypot(ypn, trg), ypt))
    stage_idx = STAGING_MAP.get((trg, ypn, ypt))
    if stage_idx is None:
        return None, "Outside supplied matrix", UNCLASSIFIED_COLOR, r
    return stage_idx, STAGE_LABELS[stage_idx], STAGE_COLORS[stage_idx], r


def stage_short(stage_idx):
    if stage_idx is None:
        return "-"
    return STAGE_SHORT[stage_idx]


@st.cache_data
def quarter_surfaces(n=180):
    """Generate 3 quarter-sphere reference boundary surfaces."""
    traces = []
    theta = np.linspace(0, np.pi / 2, n)
    phi = np.linspace(0, np.pi / 2, n)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    for radius, color, name in zip(RADII_THRESHOLDS, STAGE_COLORS[:-1], STAGE_LABELS[:-1]):
        x = radius * np.sin(theta_grid) * np.cos(phi_grid)
        y = radius * np.sin(theta_grid) * np.sin(phi_grid)
        z = radius * np.cos(theta_grid)
        mask = (
            (x >= X_MIN) & (x <= X_MAX) &
            (y >= Y_MIN) & (y <= Y_MAX) &
            (z >= Z_MIN) & (z <= Z_MAX)
        )
        x = np.where(mask, x, np.nan)
        y = np.where(mask, y, np.nan)
        z = np.where(mask, z, np.nan)
        traces.append(go.Surface(
            x=x,
            y=y,
            z=z,
            name=f"Reference boundary: {name}",
            showscale=False,
            opacity=0.22,
            colorscale=[[0, color], [1, color]],
            hoverinfo="skip",
        ))
    return traces


def cube_edges():
    vertices = np.array([
        [X_MIN, Y_MIN, Z_MIN], [X_MAX, Y_MIN, Z_MIN],
        [X_MAX, Y_MAX, Z_MIN], [X_MIN, Y_MAX, Z_MIN],
        [X_MIN, Y_MIN, Z_MAX], [X_MAX, Y_MIN, Z_MAX],
        [X_MAX, Y_MAX, Z_MAX], [X_MIN, Y_MAX, Z_MAX],
    ], float)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    traces = []
    for i, j in edges:
        traces.append(go.Scatter3d(
            x=[vertices[i, 0], vertices[j, 0]],
            y=[vertices[i, 1], vertices[j, 1]],
            z=[vertices[i, 2], vertices[j, 2]],
            mode="lines",
            line=dict(color="black", width=4),
            hoverinfo="skip",
            showlegend=False,
        ))
    return traces


@st.cache_data
def lattice_points():
    points = []
    for ypn_value in range(X_MIN, X_MAX + 1):
        for trg_value in range(Y_MIN, Y_MAX + 1):
            for ypt_value in range(Z_MIN, Z_MAX + 1):
                stage_idx, label, color, _ = stage_of(ypn_value, trg_value, ypt_value)
                points.append({
                    "ypn": ypn_value,
                    "trg": trg_value,
                    "ypt": ypt_value,
                    "stage_idx": stage_idx,
                    "stage_label": label,
                    "color": color,
                })
    return points


def add_lattice_traces(fig, selected_stage_idx, highlight_same_stage):
    points = lattice_points()
    fig.add_trace(go.Scatter3d(
        x=[point["ypn"] for point in points],
        y=[point["trg"] for point in points],
        z=[point["ypt"] for point in points],
        mode="markers",
        marker=dict(
            size=3.2,
            color=[point["color"] for point in points],
            opacity=0.55,
            line=dict(width=0.5, color="black"),
        ),
        name="Integer grid points",
        showlegend=False,
        customdata=[
            [point["stage_label"], point["ypn"], point["trg"], point["ypt"]]
            for point in points
        ],
        hovertemplate=(
            "ypN=%{customdata[1]}<br>TRG=%{customdata[2]}"
            "<br>ypT=%{customdata[3]}<br>%{customdata[0]}<extra></extra>"
        ),
    ))

    if not highlight_same_stage or selected_stage_idx is None:
        return

    selected_points = [point for point in points if point["stage_idx"] == selected_stage_idx]
    fig.add_trace(go.Scatter3d(
        x=[point["ypn"] for point in selected_points],
        y=[point["trg"] for point in selected_points],
        z=[point["ypt"] for point in selected_points],
        mode="markers",
        marker=dict(
            size=5.5,
            color=STAGE_COLORS[selected_stage_idx],
            opacity=0.92,
            line=dict(width=1.0, color="white"),
        ),
        name=f"All {STAGE_LABELS[selected_stage_idx]} points",
        showlegend=False,
        hovertemplate=(
            "ypN=%{x}<br>TRG=%{y}<br>ypT=%{z}"
            f"<br>{STAGE_LABELS[selected_stage_idx]}<extra></extra>"
        ),
    ))


def selected_axis_guides(ypn, trg, ypt, color):
    guide_style = dict(color=color, width=5, dash="dash")
    return [
        go.Scatter3d(
            x=[ypn, ypn], y=[trg, trg], z=[Z_MIN, ypt],
            mode="lines", line=guide_style, hoverinfo="skip", showlegend=False,
        ),
        go.Scatter3d(
            x=[X_MIN, ypn], y=[trg, trg], z=[ypt, ypt],
            mode="lines", line=guide_style, hoverinfo="skip", showlegend=False,
        ),
        go.Scatter3d(
            x=[ypn, ypn], y=[Y_MIN, trg], z=[ypt, ypt],
            mode="lines", line=guide_style, hoverinfo="skip", showlegend=False,
        ),
    ]


def style_block():
    return """
    <style>
        .result-card {
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-radius: 8px;
            padding: 12px 14px;
            background: #ffffff;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        }
        .result-label {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }
        .result-value {
            color: #111827;
            font-size: 1.3rem;
            font-weight: 800;
            line-height: 1.15;
        }
        .stage-table {
            border-collapse: collapse;
            width: 100%;
            table-layout: fixed;
            margin: 8px 0 18px;
            font-size: 0.75rem;
            color: #111827;
        }
        .stage-table caption {
            caption-side: top;
            text-align: left;
            color: #0f172a;
            font-size: 0.86rem;
            font-weight: 800;
            margin-bottom: 6px;
        }
        .stage-table th,
        .stage-table td {
            border: 1px solid rgba(15, 23, 42, 0.18);
            padding: 6px 4px;
            text-align: center;
            height: 30px;
        }
        .stage-table th {
            background: #eef2f7;
            font-weight: 800;
        }
        .stage-table .active-head {
            background: #111827;
            color: #ffffff;
        }
        .stage-cell {
            font-weight: 800;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stage-cell.selected {
            outline: 3px solid #111827;
            outline-offset: -3px;
            box-shadow: inset 0 0 0 2px #ffffff, 0 0 0 3px rgba(17, 24, 39, 0.22);
            transform: scale(1.02);
        }
        .mini-note {
            color: #64748b;
            font-size: 0.76rem;
            line-height: 1.35;
            margin: -4px 0 10px;
        }
        .risk-table {
            border-collapse: collapse;
            width: 100%;
            table-layout: fixed;
            color: #111827;
            font-size: 0.86rem;
            margin-top: -4px;
        }
        .risk-table th,
        .risk-table td {
            border: 1px solid rgba(15, 23, 42, 0.14);
            padding: 7px 4px;
            text-align: center;
        }
        .risk-table th {
            background: #f1f5f9;
            font-weight: 800;
        }
    </style>
    """


def stage_cell_html(stage_idx, selected=False):
    classes = "stage-cell selected" if selected else "stage-cell"
    return (
        f"<td class='{classes}' style='background:{STAGE_BACKGROUNDS[stage_idx]};'>"
        f"{STAGE_SHORT[stage_idx]}</td>"
    )


def staging_table_html(selected_trg, selected_ypn, selected_ypt):
    special_selected = selected_trg == 0 and selected_ypn == 0 and selected_ypt == 0
    html = [style_block()]
    html.append("<table class='stage-table'>")
    html.append("<caption>TRG 0</caption>")
    html.append("<tr><th></th><th>T0</th></tr>")
    active_n = " class='active-head'" if special_selected else ""
    html.append(f"<tr><th{active_n}>N0</th>{stage_cell_html(0, special_selected)}</tr>")
    html.append("</table>")

    for trg_value, rows in TRG_STAGE_GRID.items():
        active_trg = selected_trg == trg_value
        html.append("<table class='stage-table'>")
        html.append(f"<caption>TRG {trg_value}</caption>")
        header_class = " class='active-head'" if active_trg else ""
        html.append(f"<tr><th{header_class}>ypN / ypT</th>")
        for ypt_value in range(1, 5):
            active_t = active_trg and selected_ypt == ypt_value
            th_class = " class='active-head'" if active_t else ""
            html.append(f"<th{th_class}>T{ypt_value}</th>")
        html.append("</tr>")

        for ypn_value, row in enumerate(rows):
            active_n = active_trg and selected_ypn == ypn_value
            th_class = " class='active-head'" if active_n else ""
            html.append(f"<tr><th{th_class}>N{ypn_value}</th>")
            for ypt_value, stage_idx in enumerate(row, start=1):
                selected = active_trg and selected_ypn == ypn_value and selected_ypt == ypt_value
                html.append(stage_cell_html(stage_idx, selected))
            html.append("</tr>")
        html.append("</table>")

    return "".join(html)


def survival_summary_html(survival_data):
    return (
        style_block()
        + "<div class='result-card'>"
        + "<div class='result-label'>Selected stage</div>"
        + f"<div class='result-value'>{survival_data['stage']}</div>"
        + "</div>"
        + "<div style='height:10px;'></div>"
        + "<div style='display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:0 0 10px;'>"
        + "<div class='result-card'>"
        + "<div class='result-label'>1-year OS</div>"
        + f"<div class='result-value'>{survival_data['1-year OS']}</div>"
        + "</div>"
        + "<div class='result-card'>"
        + "<div class='result-label'>3-year OS</div>"
        + f"<div class='result-value'>{survival_data['3-year OS']}</div>"
        + "</div>"
        + "</div>"
        + "<div class='result-card'>"
        + "<div class='result-label'>1-year 95% CI</div>"
        + f"<div class='result-value'>{survival_data['1-year 95% CI']}</div>"
        + "</div>"
        + "<div style='height:10px;'></div>"
        + "<div class='result-card'>"
        + "<div class='result-label'>3-year 95% CI</div>"
        + f"<div class='result-value'>{survival_data['3-year 95% CI']}</div>"
        + "</div>"
    )


@st.cache_data
def load_survival_image(relative_path):
    image_path = APP_DIR / relative_path
    if not image_path.exists():
        return None, str(image_path)
    return image_path.read_bytes(), str(image_path)


# ---- Sidebar inputs ----
with st.sidebar:
    st.header("Input Parameters")
    trg = st.number_input("TRG", 0, 3, 1, step=1)
    ypn = st.number_input("ypN", 0, 3, 1, step=1)
    ypt = st.number_input("ypT", 0, 4, 1, step=1)
    show_boundaries = st.checkbox("Show reference boundary shells", True)
    show_lattice = st.checkbox("Show all integer lattice points", True)
    highlight_stage = st.checkbox("Brighten matching stage points", True)

    selected_idx, selected_label, selected_color, selected_radius = stage_of(ypn, trg, ypt)

    st.markdown("### Staging Matrix")
    st.markdown(staging_table_html(trg, ypn, ypt), unsafe_allow_html=True)
    if selected_idx is None:
        st.warning("This coordinate is not defined in the supplied matrix.")
    else:
        st.caption(f"Selected cell: TRG {trg}, N{ypn}, T{ypt} -> {selected_label}")


st.title("Gastric Cancer TRG-ypT-ypN 3D Staging")

top_left, top_right = st.columns([2.1, 1], gap="large")

with top_left:
    fig = go.Figure()
    for trace in cube_edges():
        fig.add_trace(trace)
    if show_boundaries:
        for trace in quarter_surfaces():
            fig.add_trace(trace)
    if show_lattice:
        add_lattice_traces(fig, selected_idx, highlight_stage)

    for trace in selected_axis_guides(ypn, trg, ypt, selected_color):
        fig.add_trace(trace)

    fig.add_trace(go.Scatter3d(
        x=[ypn],
        y=[trg],
        z=[ypt],
        mode="markers",
        marker=dict(
            size=18,
            color=selected_color,
            opacity=0.18,
            line=dict(width=0),
        ),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter3d(
        x=[ypn],
        y=[trg],
        z=[ypt],
        mode="markers",
        marker=dict(size=10, color=selected_color, line=dict(width=1.2, color="black")),
        name="Selected coordinate",
        showlegend=False,
        hovertemplate=(
            "<b>Selected coordinate</b><br>ypN=%{x}<br>TRG=%{y}<br>ypT=%{z}"
            f"<br>{selected_label}<extra></extra>"
        ),
    ))

    fig.update_scenes(
        xaxis=dict(title="ypN", range=[X_MIN, X_MAX], dtick=1, zeroline=False),
        yaxis=dict(title="TRG", range=[Y_MIN, Y_MAX], dtick=1, zeroline=False),
        zaxis=dict(title="ypT", range=[Z_MIN, Z_MAX], dtick=1, zeroline=False),
        camera=dict(eye=dict(x=1.6, y=1.4, z=1.1)),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1.1),
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=680)
    st.plotly_chart(fig, use_container_width=True)

with top_right:
    st.subheader("Staging Result")
    stage_text = selected_label if selected_idx is not None else "Not classified"
    stage_short_text = stage_short(selected_idx)
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Stage</div>
            <div class="result-value" style="color:{selected_color};">{stage_text}</div>
        </div>
        <div style="height:10px;"></div>
        <div class="result-card">
            <div class="result-label">Selected coordinate</div>
            <div class="result-value">TRG {trg} / N{ypn} / T{ypt}</div>
        </div>
        <div style="height:10px;"></div>
        <div class="result-card">
            <div class="result-label">Matrix value</div>
            <div class="result-value">{stage_short_text}</div>
        </div>
        <div style="height:10px;"></div>
        <div class="result-card">
            <div class="result-label">3D radius</div>
            <div class="result-value">{selected_radius:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Stage is read from the TRG/ypN/ypT matrix; radius is retained for 3D orientation.")

st.divider()
survival_left, survival_right = st.columns([1.7, 1], gap="large")
selected_survival = STAGE_SURVIVAL_DATA.get(selected_idx)

with survival_left:
    st.subheader("Stage-Specific Overall Survival Curve")
    if selected_survival is None:
        st.info("Select a matrix-defined stage to show the corresponding survival curve.")
    else:
        image_bytes, image_path = load_survival_image(selected_survival["image"])
        if image_bytes is None:
            st.warning(f"Survival curve image was not found: {image_path}")
        else:
            st.image(image_bytes, use_column_width=True)
        st.caption("KM curve rendered from the supplied stage-specific PDF; number-at-risk panel omitted.")

with survival_right:
    st.subheader("Survival Summary")
    if selected_survival is None:
        st.info("No survival summary for this undefined coordinate.")
    else:
        st.markdown(survival_summary_html(selected_survival), unsafe_allow_html=True)
