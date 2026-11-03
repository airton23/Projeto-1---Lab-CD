"""Apresentação uniforme, sem arredondar os valores usados nos cálculos."""
from datetime import date, datetime
from numbers import Integral, Real

import numpy as np
import pandas as pd
from markupsafe import Markup


def fmt(value, decimals=2):
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return "—"
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.strftime("%d/%m/%Y")
    if isinstance(value, (bool, np.bool_)):
        return "Sim" if value else "Não"
    if isinstance(value, Integral):
        return f"{value:,}".replace(",", ".")
    if isinstance(value, Real):
        if not np.isfinite(value):
            return "Indefinido"
        return f"{value:,.{decimals}f}".translate(str.maketrans({",": ".", ".": ","}))
    if isinstance(value, (list, tuple)):
        return "; ".join(fmt(v) for v in value) if value else "Sem moda informativa"
    return str(value)


def table_html(frame, caption=None):
    if frame.empty:
        return Markup('<p class="empty-state">Nenhuma ocorrência neste recorte.</p>')
    display = frame.copy().astype(object)
    for column in display.columns:
        display[column] = display[column].map(fmt)
    spaces = {column: 110 for column in display.columns if column in {"Data", "Partida", "Linha CSV"}}
    html = display.to_html(index=False, border=0, classes="data-table", escape=True, col_space=spaces)
    if caption:
        from markupsafe import escape
        html = html.replace('<thead>', f'<caption>{escape(caption)}</caption><thead>', 1)
    return Markup(html)


def chart_html(figure):
    html = figure.to_html(full_html=False, include_plotlyjs=False,
                         config={"responsive": True, "displaylogo": False,
                                 "toImageButtonOptions": {"format": "svg"}})
    return Markup(html)

