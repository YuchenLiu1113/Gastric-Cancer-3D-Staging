# -*- coding: utf-8 -*-
import csv
import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Gastric Cancer 3D Staging", layout="wide")

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

SURVIVAL_POINTS = [
    {"month": 0, "os": 1.000, "lower": 1.000, "upper": 1.000},
    {"month": 3, "os": 0.985, "lower": 0.973, "upper": 0.994},
    {"month": 6, "os": 0.960, "lower": 0.944, "upper": 0.974},
    {"month": 9, "os": 0.925, "lower": 0.907, "upper": 0.943},
    {"month": 12, "os": 0.894, "lower": 0.877, "upper": 0.912},
    {"month": 15, "os": 0.865, "lower": 0.842, "upper": 0.887},
    {"month": 18, "os": 0.840, "lower": 0.815, "upper": 0.864},
    {"month": 21, "os": 0.805, "lower": 0.778, "upper": 0.831},
    {"month": 24, "os": 0.766, "lower": 0.741, "upper": 0.792},
    {"month": 27, "os": 0.735, "lower": 0.707, "upper": 0.763},
    {"month": 30, "os": 0.705, "lower": 0.675, "upper": 0.735},
    {"month": 33, "os": 0.685, "lower": 0.654, "upper": 0.716},
    {"month": 36, "os": 0.667, "lower": 0.636, "upper": 0.698},
]

SURVIVAL_SUMMARY = {
    "1-year OS": "89.4%",
    "1-year 95% CI": "87.7%-91.2%",
    "3-year OS": "66.7%",
    "3-year 95% CI": "63.6%-69.8%",
}

NUMBER_AT_RISK = [
    {"month": 0, "at_risk": 1270},
    {"month": 6, "at_risk": 1222},
    {"month": 12, "at_risk": 1063},
    {"month": 18, "at_risk": 871},
    {"month": 24, "at_risk": 679},
    {"month": 30, "at_risk": 506},
    {"month": 36, "at_risk": 410},
]


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


def survival_figure(stage_idx):
    months = [point["month"] for point in SURVIVAL_POINTS]
    survival = [point["os"] * 100 for point in SURVIVAL_POINTS]
    lower = [point["lower"] * 100 for point in SURVIVAL_POINTS]
    upper = [point["upper"] * 100 for point in SURVIVAL_POINTS]
    curve_color = STAGE_COLORS[stage_idx] if stage_idx is not None else "#334155"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months,
        y=upper,
        mode="lines",
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=months,
        y=lower,
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(51, 65, 85, 0.14)",
        line=dict(width=0),
        hoverinfo="skip",
        name="95% CI",
    ))
    fig.add_trace(go.Scatter(
        x=months,
        y=survival,
        mode="lines",
        line=dict(color=curve_color, width=3, shape="hv"),
        name="Overall survival",
        hovertemplate="Month %{x}<br>OS %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[12, 36],
        y=[89.4, 66.7],
        mode="markers+text",
        marker=dict(size=9, color=curve_color, line=dict(color="white", width=1.5)),
        text=["1-year", "3-year"],
        textposition="top center",
        hovertemplate="%{text}<br>OS %{y:.1f}%<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title="Time (months)", range=[0, 36], dtick=6, gridcolor="#e2e8f0"),
        yaxis=dict(title="Overall survival (%)", range=[0, 102], dtick=20, gridcolor="#e2e8f0"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def risk_table_html():
    html = [style_block(), "<table class='risk-table'>"]
    html.append("<tr><th>Months</th>")
    for row in NUMBER_AT_RISK:
        html.append(f"<th>{row['month']}</th>")
    html.append("</tr><tr><th>Number at risk</th>")
    for row in NUMBER_AT_RISK:
        html.append(f"<td>{row['at_risk']}</td>")
    html.append("</tr></table>")
    return "".join(html)


def survival_summary_html():
    return (
        style_block()
        + "<div style='display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:0 0 14px;'>"
        + "<div class='result-card'>"
        + "<div class='result-label'>1-year OS</div>"
        + f"<div class='result-value'>{SURVIVAL_SUMMARY['1-year OS']}</div>"
        + f"<div class='mini-note'>95% CI {SURVIVAL_SUMMARY['1-year 95% CI']}</div>"
        + "</div>"
        + "<div class='result-card'>"
        + "<div class='result-label'>3-year OS</div>"
        + f"<div class='result-value'>{SURVIVAL_SUMMARY['3-year OS']}</div>"
        + f"<div class='mini-note'>95% CI {SURVIVAL_SUMMARY['3-year 95% CI']}</div>"
        + "</div>"
        + "</div>"
    )


def survival_csv():
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["month", "os", "lower", "upper"])
    writer.writeheader()
    writer.writerows(SURVIVAL_POINTS)
    return output.getvalue()


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

with survival_left:
    st.subheader("Overall Survival Curve")
    st.plotly_chart(survival_figure(selected_idx), use_container_width=True)
    st.markdown(survival_summary_html(), unsafe_allow_html=True)
    st.markdown(risk_table_html(), unsafe_allow_html=True)
    st.caption("Curve redrawn from supplied aggregate OS values; confirm final citation against the source dataset or manuscript.")

with survival_right:
    st.subheader("Survival Summary")
    metric_1, metric_3 = st.columns(2)
    metric_1.metric("1-year OS", SURVIVAL_SUMMARY["1-year OS"])
    metric_3.metric("3-year OS", SURVIVAL_SUMMARY["3-year OS"])
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">1-year 95% CI</div>
            <div class="result-value">{SURVIVAL_SUMMARY["1-year 95% CI"]}</div>
        </div>
        <div style="height:10px;"></div>
        <div class="result-card">
            <div class="result-label">3-year 95% CI</div>
            <div class="result-value">{SURVIVAL_SUMMARY["3-year 95% CI"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download survival curve data",
        data=survival_csv(),
        file_name="overall_survival_curve.csv",
        mime="text/csv",
        use_container_width=True,
    )
