"""Integração dos cálculos com o relatório; sem fórmulas nos templates."""
import numpy as np
import pandas as pd

from analytics.charts import (build_bars, build_boxplot, build_detection_heatmap,
                              build_histogram, build_timeline)
from analytics.data import load_dataset, prepare_dataset, source_metadata
from analytics.descriptive import describe_by_team, describe_series, standardized_team_means
from analytics.outliers import METHODS, detect_outliers
from analytics.quality import audit_quality
from analytics.schema import MEASUREMENTS, SCHEMA, dictionary_rows
from app.formatting import chart_html, fmt, table_html
from config import IQR_FACTOR, TEAMS, VARIABLES

MEASURES = {"mean": "Média", "median": "Mediana", "std": "Desvio padrão",
            "variance": "Variância", "cv": "CV", "iqr": "IQR", "zscore": "Média em z-score"}


def load_project(path):
    original = load_dataset(path)
    raw, prepared, transformations = prepare_dataset(original, TEAMS)
    quality = audit_quality(prepared, raw, TEAMS, original)
    cleaned = quality["cleaned"]
    summaries = describe_by_team(cleaned, VARIABLES, TEAMS, IQR_FACTOR)
    detectors = {}
    for team_id in TEAMS:
        group = cleaned.loc[cleaned["time_id"] == team_id]
        for variable in VARIABLES:
            for method in METHODS:
                detectors[(team_id, variable, method)] = detect_outliers(group[variable], method)
    return {"original": original, "raw": raw, "prepared": prepared, "df": cleaned,
            "transformations": transformations, "quality": quality, "summaries": summaries,
            "detectors": detectors, "source": source_metadata(path, original, prepared)}


def base_context(project, team_id, variable, method, measure):
    teams = TEAMS if team_id == "ambos" else {team_id: TEAMS[team_id]}
    return {"teams": TEAMS, "visible_teams": teams, "variables": VARIABLES,
            "schema": SCHEMA, "methods": METHODS, "measures": MEASURES,
            "selected_team": team_id, "variable": variable, "method": method, "measure": measure,
            "source": project["source"], "scope_label": "Chicago Bulls e Minnesota Timberwolves" if team_id == "ambos" else TEAMS[team_id]["name"],
            "active_rows": int(project["df"]["time_id"].isin(teams).sum())}


def overview_context(project, context):
    teams = context["visible_teams"]
    df = project["df"]
    cards = []
    for team_id, team in teams.items():
        group = df.loc[df["time_id"] == team_id]
        summary = project["summaries"][(team_id, "pontos")]
        cards.append({**team, "id": team_id, "games": len(group),
                      "wins": int(group["venceu"].sum()), "summary": summary})
    return {"cards": cards, "timeline": chart_html(build_timeline(df, teams)),
            "overview_box": chart_html(build_boxplot(df, context["variable"], project["summaries"], teams))}


def preparation_context(project):
    return {"transformations_table": table_html(project["transformations"])}


def descriptive_context(project, context):
    variable = context["variable"]
    df = project["df"]
    teams = context["visible_teams"]
    rows, distribution_cards, extreme_rows, sensitivity = [], [], [], []
    meta = SCHEMA[variable]
    for team_id, team in teams.items():
        summary = project["summaries"][(team_id, variable)]
        series = df.loc[df["time_id"] == team_id, variable]
        for label, key in [("Valores válidos (jogos)", "n"), ("Ausentes/inválidos (jogos)", "missing"),
                           ("Média", "mean"), ("Mediana", "median"), ("Moda(s)", "modes"),
                           ("Mínimo", "min"), ("Máximo", "max"), ("Amplitude", "range"),
                           ("Desvio padrão amostral", "std"), ("Variância amostral", "variance"),
                           ("Q1", "q1"), ("Q3", "q3"), ("IQR", "iqr"), ("CV", "cv"),
                           ("Assimetria ajustada G1", "skew"), ("IQR / mediana", "robust_ratio"),
                           ("Limite inferior teórico", "lower"), ("Limite superior teórico", "upper"),
                           ("Bigode inferior observado", "whisker_low"), ("Bigode superior observado", "whisker_high"),
                           ("Outliers IQR (jogos)", "outlier_count")]:
            unit = ("jogos" if key in {"n", "missing", "outlier_count"} else "%" if key == "cv"
                    else "adimensional" if key in {"skew", "robust_ratio"}
                    else f"({meta['unit']})²" if key == "variance" else meta["unit"])
            existing = next((r for r in rows if r["Medida"] == label), None)
            if existing is None:
                existing = {"Medida": label, "Unidade": unit}
                rows.append(existing)
            existing[team["name"]] = summary[key]
        distribution_cards.append({"team": team, "summary": summary,
                                   "histogram": chart_html(build_histogram(series, variable, team))})
        for row_id, value in summary["outliers"].items():
            observation = df.loc[row_id]
            extreme_rows.append({"Equipe": team["name"], "Data": observation["data_partida"],
                                 "Confronto": observation["confronto"], "Partida": observation["partida_id"],
                                 "Linha CSV": int(row_id), f"Valor ({meta['unit']})": value,
                                 "Extremo": "Inferior" if value < summary["lower"] else "Superior"})
        retained = series.drop(index=summary["outliers"].index)
        without = describe_series(retained)
        sensitivity.append({"Equipe": team["name"], "Cenário": "Todos os valores válidos", "n": summary["n"],
                            f"Desvio ({meta['unit']})": summary["std"], f"IQR ({meta['unit']})": summary["iqr"],
                            f"Média ({meta['unit']})": summary["mean"]})
        sensitivity.append({"Equipe": team["name"], "Cenário": "Sem sinalizados IQR (sensibilidade)", "n": without["n"],
                            f"Desvio ({meta['unit']})": without["std"], f"IQR ({meta['unit']})": without["iqr"],
                            f"Média ({meta['unit']})": without["mean"]})
    return {"statistics_table": table_html(pd.DataFrame(rows)), "distributions": distribution_cards,
            "boxplot": chart_html(build_boxplot(df, variable, project["summaries"], teams)),
            "extremes_table": table_html(pd.DataFrame(extreme_rows)),
            "sensitivity_table": table_html(pd.DataFrame(sensitivity))}


def matrix_table(project, key, teams):
    rows = []
    for variable in VARIABLES:
        row = {"Variável": SCHEMA[variable]["label"],
               "Unidade": "%" if key == "cv" else "adimensional" if key == "skew" else SCHEMA[variable]["unit"]}
        row.update({team["name"]: project["summaries"][(team_id, variable)][key] for team_id, team in teams.items()})
        rows.append(row)
    return table_html(pd.DataFrame(rows))


def comparison_context(project, context):
    variable, measure, teams = context["variable"], context["measure"], context["visible_teams"]
    reference = None
    if measure == "zscore":
        values, reference = standardized_team_means(project["df"], variable, TEAMS)
        unit = "desvios padrão (adimensional)"
    else:
        values = {team_id: project["summaries"][(team_id, variable)][measure] for team_id in teams}
        unit = "%" if measure == "cv" else f"({SCHEMA[variable]['unit']})²" if measure == "variance" else SCHEMA[variable]["unit"]
    details, comparisons = [], []
    for v in VARIABLES:
        for team_id, team in teams.items():
            s = project["summaries"][(team_id, v)]
            details.append({"Variável": SCHEMA[v]["label"], "Equipe": team["short"], "Unidade": SCHEMA[v]["unit"],
                            "n válido": s["n"], "Média": s["mean"], "Desvio": s["std"], "Amplitude": s["range"],
                            "Q1": s["q1"], "Q3": s["q3"], "IQR": s["iqr"],
                            "CV (%)": s["cv"], "G1 (adimensional)": s["skew"]})
        valid_cv = [(tid, project["summaries"][(tid, v)]["cv"]) for tid in teams
                    if np.isfinite(project["summaries"][(tid, v)]["cv"])]
        if len(valid_cv) == 2:
            low, high = sorted(valid_cv, key=lambda pair: pair[1])
            iqrs = sorted(((tid, project["summaries"][(tid, v)]["iqr"]) for tid in teams), key=lambda pair: pair[1])
            comparisons.append({"variable": v, "label": SCHEMA[v]["label"],
                                "low": teams[low[0]]["name"], "low_cv": low[1],
                                "high": teams[high[0]]["name"], "high_cv": high[1],
                                "tie": bool(np.isclose(low[1], high[1])),
                                "iqr_low_team": teams[iqrs[0][0]]["name"], "iqr_low": iqrs[0][1],
                                "iqr_high_team": teams[iqrs[1][0]]["name"], "iqr_high": iqrs[1][1],
                                "iqr_tie": bool(np.isclose(iqrs[0][1], iqrs[1][1]))})
    return {"bar_chart": chart_html(build_bars(values, variable, teams, MEASURES[measure], unit)),
            "reference": reference, "cv_table": matrix_table(project, "cv", teams),
            "iqr_table": matrix_table(project, "iqr", teams),
            "skew_table": matrix_table(project, "skew", teams),
            "comparison_details": table_html(pd.DataFrame(details)), "comparisons": comparisons}


def quality_context(project, context):
    teams, variable, method = context["visible_teams"], context["variable"], context["method"]
    if context["selected_team"] == "ambos":
        audit = project["quality"]
    else:
        mask = project["prepared"]["time_id"].isin(teams)
        audit = audit_quality(project["prepared"].loc[mask], project["raw"].loc[mask], TEAMS, project["original"])
    details, occurrence_rows, detector_summary, counts = [], [], [], []
    for v in VARIABLES:
        heat_row = []
        for m in METHODS:
            total = 0
            for team_id, team in teams.items():
                result = project["detectors"][(team_id, v, m)]
                detector_summary.append({"Equipe": team["short"], "Variável": SCHEMA[v]["label"],
                                         "Método": METHODS[m], "n válido": result["n"], "Sinalizados": result["count"],
                                         "Estado": result["status"]})
                total += result["count"] or 0
            all_na = all(project["detectors"][(tid, v, m)]["count"] is None for tid in teams)
            heat_row.append(None if all_na else total)
        counts.append(heat_row)
    for team_id, team in teams.items():
        result = project["detectors"][(team_id, variable, method)]
        details.append({"team": team, "result": result})
        for row_id, observation in result["rows"].loc[result["rows"]["sinalizado"]].iterrows():
            source = project["df"].loc[row_id]
            occurrence_rows.append({"Equipe": team["name"], "Data": source["data_partida"],
                                    "Confronto": source["confronto"], "Partida": source["partida_id"],
                                    "Linha CSV": int(row_id), f"Valor ({SCHEMA[variable]['unit']})": observation["valor"],
                                    "Escore": observation["escore"], "Método": result["name"],
                                    "Limiar": result["threshold"]})
    summary = audit["summary"]
    return {"audit_table": table_html(summary), "issues_table": table_html(audit["issues"]),
            "missing_count": int(summary["Ausentes na leitura"].sum()),
            "invalid_count": int(audit["issues"]["Linha CSV"].nunique()) if len(audit["issues"]) else 0,
            "duplicate_count": audit["exact_duplicates"], "duplicate_key_rows": audit["duplicate_key_rows"],
            "detector_details": details, "outlier_occurrences": table_html(pd.DataFrame(occurrence_rows)),
            "detectors_table": table_html(pd.DataFrame(detector_summary)),
            "detection_heatmap": chart_html(build_detection_heatmap(counts, VARIABLES, METHODS.values()))}


def synthesis_context(project, context):
    teams = context["visible_teams"]
    team_insights, sensitivity_rows, influence = [], [], []
    for team_id, team in teams.items():
        values = [(v, project["summaries"][(team_id, v)]) for v in VARIABLES]
        valid_skew = [(v, s) for v, s in values if np.isfinite(s["skew"])]
        valid_cv = [(v, s) for v, s in values if np.isfinite(s["cv"])]
        team_insights.append({"team": team, "skewed": max(valid_skew, key=lambda p: abs(p[1]["skew"])) if valid_skew else None,
                              "stable": min(valid_cv, key=lambda p: p[1]["cv"]) if valid_cv else None,
                              "variable": max(valid_cv, key=lambda p: p[1]["cv"]) if valid_cv else None,
                              "summaries": values})
        for variable, summary in values:
            series = project["df"].loc[project["df"]["time_id"] == team_id, variable]
            without = describe_series(series.drop(index=summary["outliers"].index))
            std_drop = 100 * (summary["std"] - without["std"]) / summary["std"] if summary["std"] > 0 else np.nan
            iqr_change = 100 * abs(summary["iqr"] - without["iqr"]) / summary["iqr"] if summary["iqr"] > 0 else np.nan
            sensitivity_rows.append({"Equipe": team["short"], "Variável": SCHEMA[variable]["label"],
                                     "Unidade": SCHEMA[variable]["unit"], "Sinalizados IQR": summary["outlier_count"],
                                     "Desvio completo": summary["std"], "Desvio sem sinalizados": without["std"],
                                     "IQR completo": summary["iqr"], "IQR sem sinalizados": without["iqr"],
                                     "Queda do desvio (%)": std_drop, "Mudança absoluta do IQR (%)": iqr_change})
            if summary["outlier_count"] and np.isfinite(std_drop) and np.isfinite(iqr_change) and std_drop > iqr_change:
                influence.append({"team": team["name"], "variable": SCHEMA[variable]["label"],
                                  "std_drop": std_drop, "iqr_change": iqr_change})
    comparisons = comparison_context(project, context)["comparisons"]
    skew_patterns = [insight["skewed"][0] for insight in team_insights if insight["skewed"]]
    return {"team_insights": team_insights, "comparisons": comparisons,
            "cv_table": matrix_table(project, "cv", teams), "iqr_table": matrix_table(project, "iqr", teams),
            "repeated_skew": len(skew_patterns) == 2 and len(set(skew_patterns)) == 1,
            "influence": influence, "sensitivity_all": table_html(pd.DataFrame(sensitivity_rows))}


def dictionary_context(project):
    return {"dictionary_table": table_html(pd.DataFrame(dictionary_rows(project["prepared"])))}


def full_report_context(project, context):
    """Relatório completo do recorte, incluindo as sete variáveis na mesma página."""
    report = {
        **dictionary_context(project),
        **preparation_context(project),
        **quality_context(project, context),
        **comparison_context(project, context),
        **synthesis_context(project, context),
    }
    chapters = []
    occurrences = []
    for variable in VARIABLES:
        chapter_context = {**context, "variable": variable}
        means = {team_id: project["summaries"][(team_id, variable)]["mean"]
                 for team_id in context["visible_teams"]}
        chapters.append({
            "variable": variable,
            **descriptive_context(project, chapter_context),
            "bar_chart": chart_html(build_bars(means, variable, context["visible_teams"],
                                               "Média", SCHEMA[variable]["unit"])),
        })
        for team_id, team in context["visible_teams"].items():
            for method, name in METHODS.items():
                result = project["detectors"][(team_id, variable, method)]
                for row_id, observation in result["rows"].loc[result["rows"]["sinalizado"]].iterrows():
                    source = project["df"].loc[row_id]
                    occurrences.append({
                        "Equipe": team["name"], "Data": source["data_partida"],
                        "Partida": source["partida_id"], "Confronto": source["confronto"],
                        "Variável": SCHEMA[variable]["label"], "Unidade": SCHEMA[variable]["unit"],
                        "Método": name, "Valor": observation["valor"],
                        "Escore": observation["escore"], "Limiar": result["threshold"],
                    })
    report["variable_chapters"] = chapters
    report["all_outlier_occurrences"] = table_html(pd.DataFrame(occurrences))
    return report

