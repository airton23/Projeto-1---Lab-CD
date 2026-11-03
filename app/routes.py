"""Rotas do relatório e validação da seleção enviada pelo navegador."""
from pathlib import Path

import plotly
from flask import Blueprint, abort, current_app, redirect, render_template, request, send_file, url_for

from app.context import (MEASURES, base_context, comparison_context, descriptive_context,
                         dictionary_context, overview_context, preparation_context,
                         quality_context, full_report_context)
from analytics.outliers import METHODS
from config import TEAMS, VARIABLES

report = Blueprint("report", __name__)
PAGES = {
    "overview": ("/", "Visão geral", overview_context),
    "dictionary": ("/dicionario", "Dicionário de dados", lambda p, c: dictionary_context(p)),
    "preparation": ("/preparacao", "Preparação dos dados", lambda p, c: preparation_context(p)),
    "quality": ("/qualidade", "Qualidade dos dados", quality_context),
    "descriptive": ("/descritiva", "Análise univariada", descriptive_context),
    "comparison": ("/comparacao", "Comparação dos times", comparison_context),
    "full_report": ("/relatorio", "Relatório completo", full_report_context),
}


def selection():
    values = {"team_id": request.args.get("time", "ambos"),
              "variable": request.args.get("variavel", "pontos"),
              "method": request.args.get("metodo", "iqr"),
              "measure": request.args.get("medida", "mean")}
    allowed = {"team_id": {"ambos", *TEAMS}, "variable": set(VARIABLES),
               "method": set(METHODS), "measure": set(MEASURES)}
    for key, value in values.items():
        if value not in allowed[key]:
            abort(400, description="Seleção inválida. Escolha uma das duas equipes, variáveis e opções disponíveis.")
    return values


def render_report(page):
    project = current_app.extensions["project"]
    context = base_context(project, **selection())
    context.update(PAGES[page][2](project, context))
    context.update(page=page, page_title=PAGES[page][1], pages=PAGES)
    return render_template(f"report/{page}.html", **context)


for page_name, (path, _, _) in PAGES.items():
    report.add_url_rule(path, endpoint=page_name, view_func=lambda page=page_name: render_report(page))


def redirect_to_report(anchor):
    filters = {key: request.args[key] for key in ("time", "variavel", "metodo", "medida")
               if key in request.args}
    return redirect(url_for("report.full_report", _anchor=anchor, **filters))


@report.get("/metodologia")
def methodology():
    return redirect_to_report("metodologia")


@report.get("/sintese")
def synthesis():
    return redirect_to_report("sintese")


@report.get("/assets/plotly.min.js")
def plotly_asset():
    return send_file(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", max_age=86400)


@report.app_errorhandler(400)
@report.app_errorhandler(404)
def error_page(error):
    return render_template("error.html", error=error), error.code

