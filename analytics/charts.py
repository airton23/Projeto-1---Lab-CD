"""Figuras que reutilizam as mesmas medidas exibidas nas tabelas."""
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from analytics.descriptive import valid_values
from analytics.schema import SCHEMA


def style_figure(fig, title, x_title, y_title, height=410):
    fig.update_layout(
        title={"text": title, "font": {"size": 17}, "x": 0.02},
        template="plotly_white", height=height, autosize=True,
        font={"family": "Arial, sans-serif", "color": "#253849", "size": 12},
        paper_bgcolor="white", plot_bgcolor="white",
        margin={"l": 65, "r": 28, "t": 65, "b": 100},
        legend={"orientation": "h", "yanchor": "top", "y": -0.22, "x": 0},
        separators=",.", hoverlabel={"bgcolor": "white"},
        xaxis_title=x_title, yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#edf0ef", zerolinecolor="#d9e1dd")
    return fig


def build_boxplot(df, variable, summaries, teams):
    fig = go.Figure()
    meta = SCHEMA[variable]
    extreme_legends = set()
    for position, (team_id, team) in enumerate(teams.items()):
        summary = summaries[(team_id, variable)]
        if not summary["n"]:
            continue
        fig.add_trace(go.Box(
            x=[position], q1=[summary["q1"]], median=[summary["median"]],
            q3=[summary["q3"]], lowerfence=[summary["whisker_low"]],
            upperfence=[summary["whisker_high"]], width=0.34,
            name=team["name"], marker_color=team["color"], fillcolor=team["color"],
            opacity=0.6, boxpoints=False,
        ))
        for name, values, symbol, color in [
            ("Média", [summary["mean"]], "diamond", "#192d3d"),
            ("Moda(s)", summary["modes"], "x", "#a87420"),
        ]:
            fig.add_trace(go.Scatter(x=[position] * len(values), y=values, mode="markers",
                                     name=name, marker={"symbol": symbol, "size": 10, "color": color},
                                     showlegend=position == 0,
                                     hovertemplate=name + ": %{y:.2f}<extra></extra>"))
        for i, (label, limit) in enumerate([("Limite inferior", summary["lower"]), ("Limite superior", summary["upper"]) ]):
            fig.add_trace(go.Scatter(x=[position - 0.24, position + 0.24], y=[limit, limit],
                                     mode="lines", line={"color": "#7a8a83", "dash": "dot", "width": 1.5},
                                     name="Limites teóricos (1,5 × IQR)", showlegend=position == 0 and i == 0,
                                     hovertemplate=label + ": %{y:.2f}<extra></extra>"))
        observations = df.loc[df["time_id"] == team_id]
        values = valid_values(observations[variable])
        for label, mask, symbol in [
            ("Outliers inferiores", values < summary["lower"], "triangle-down"),
            ("Outliers superiores", values > summary["upper"], "triangle-up"),
        ]:
            extreme = values.loc[mask]
            custom = [[str(observations.at[index, "partida_id"]),
                       observations.at[index, "data_partida"].strftime("%d/%m/%Y")
                       if str(observations.at[index, "data_partida"]) != "NaT" else "Data indisponível",
                       str(observations.at[index, "confronto"])] for index in extreme.index]
            fig.add_trace(go.Scatter(x=[position] * len(extreme), y=extreme.to_list(),
                                     name=label, mode="markers", customdata=custom,
                                     marker={"color": team["color"], "size": 10, "symbol": symbol},
                                     showlegend=bool(len(extreme)) and label not in extreme_legends,
                                     hovertemplate="%{y:.0f}<br>%{customdata[1]} · %{customdata[2]}<br>ID %{customdata[0]}<extra>" + label + "</extra>"))
            if len(extreme):
                extreme_legends.add(label)
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(teams))),
                     ticktext=[t["name"] for t in teams.values()])
    return style_figure(fig, f"Distribuição de {meta['label'].lower()} · boxplot", "Equipe", meta["unit"], 450)


def build_histogram(series, variable, team):
    values = valid_values(series)
    meta = SCHEMA[variable]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, histnorm="probability density", name="Histograma (densidade)",
                               marker_color=team["color"], opacity=0.6,
                               xbins={"start": float(values.min() - 0.5) if len(values) else 0,
                                      "end": float(values.max() + 0.5) if len(values) else 1,
                                      "size": max(1, float(np.ceil((values.max() - values.min()) / 15))) if len(values) else 1}))
    if len(values) >= 3 and values.nunique() > 1:
        grid = np.linspace(max(0, values.min() - 2 * values.std(ddof=1)),
                           values.max() + 2 * values.std(ddof=1), 250)
        density = gaussian_kde(values, bw_method="scott")
        fig.add_trace(go.Scatter(x=grid, y=density(grid), mode="lines", name="KDE · Scott",
                                 line={"color": "#203649", "width": 2.5}))
    fig.update_xaxes(rangemode="nonnegative")
    return style_figure(fig, f"{meta['label']} · {team['name']}", meta["unit"],
                        f"Densidade [1/({meta['unit']})]")


def build_bars(values, variable, teams, measure_label, unit):
    fig = go.Figure()
    for team_id, team in teams.items():
        value = values.get(team_id, np.nan)
        fig.add_trace(go.Bar(x=[team["short"]], y=[value], name=team["name"],
                            marker_color=team["color"], width=0.42,
                            text=[f"{value:.2f}".replace(".", ",") if np.isfinite(value) else "Indisponível"],
                            textposition="outside", cliponaxis=False,
                            hovertemplate=f"{team['name']}<br>{measure_label}: %{{y:.2f}} {unit}<extra></extra>"))
    return style_figure(fig, f"{SCHEMA[variable]['label']} · {measure_label.lower()}", "Equipe", unit)


def build_timeline(df, teams):
    fig = go.Figure()
    for team_id, team in teams.items():
        values = df.loc[df["time_id"] == team_id].sort_values("data_partida")
        fig.add_trace(go.Scatter(x=values["data_partida"], y=values["pontos"].astype(float),
                                 name=team["name"], mode="lines+markers",
                                 line={"color": team["color"], "width": 1.5},
                                 marker={"size": 4},
                                 hovertemplate="%{x|%d/%m/%Y}<br>%{y:.0f} pontos<extra>%{fullData.name}</extra>"))
    fig.update_xaxes(tickformat="%d/%m/%Y", nticks=6)
    return style_figure(fig, "Pontuação ao longo dos jogos", "Data da partida", "pontos/jogo")


def build_detection_heatmap(counts, variables, method_names):
    fig = go.Figure(go.Heatmap(
        z=counts, x=list(method_names), y=[SCHEMA[v]["label"] for v in variables],
        text=counts, texttemplate="%{text}",
        colorscale=[[0, "#f2f6f3"], [0.4, "#9ac8ad"], [1, "#1c705c"]],
        colorbar={"title": {"text": "Jogos<br>sinalizados"}},
        hovertemplate="%{y}<br>%{x}: %{z} sinalizações<extra></extra>",
    ))
    return style_figure(fig, "Sinalizações por variável e método", "Método", "Variável", 460)

