"""Leitura e preparação reproduzível, sem dependência do Flask."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from analytics.schema import NUMERIC, RENAME, SCHEMA


def load_dataset(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Base não encontrada: {path}. Consulte o README.")
    df = pd.read_csv(path, dtype="string", encoding="utf-8-sig",
                     keep_default_na=False)
    missing = set(RENAME) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {', '.join(sorted(missing))}")
    # O índice identifica a linha física no CSV, incluindo o cabeçalho.
    df.index = pd.Index(np.arange(2, len(df) + 2), name="linha_csv")
    return df


def prepare_dataset(original, teams):
    selected = original.loc[original["Team_ID"].str.strip().isin(teams)].copy()
    if selected.empty:
        raise ValueError("A base não contém registros das equipes configuradas.")
    raw = selected.rename(columns=RENAME).loc[:, list(SCHEMA)].copy()
    raw = raw.apply(lambda column: column.str.strip())
    raw = raw.replace(["", "NA", "N/A", "NaN", "nan", "null", "NULL", "None"], pd.NA)
    prepared = raw.copy()
    changes = []
    for name, meta in SCHEMA.items():
        if name in NUMERIC:
            parsed = pd.to_numeric(raw[name], errors="coerce").astype("Float64")
            if meta["kind"] in {"contagem", "contagem acumulada", "binária"}:
                finite = parsed.dropna()
                if np.isfinite(finite.to_numpy(dtype=float)).all() and finite.mod(1).eq(0).all():
                    parsed = parsed.astype("Int64")
            prepared[name] = parsed
        elif meta["kind"] == "data":
            prepared[name] = pd.to_datetime(raw[name], format="%Y-%m-%d", errors="coerce")
        elif meta["kind"] == "categórica":
            prepared[name] = raw[name].astype("category")
        failures = int((raw[name].notna() & prepared[name].isna()).sum())
        changes.append({"Original": meta["original"], "Nome interno": name,
                        "Tipo inicial": "texto (leitura preservada)", "Tipo final": str(prepared[name].dtype),
                        "Valores presentes": int(raw[name].notna().sum()), "Falhas de conversão": failures})
    return raw, prepared, pd.DataFrame(changes)


def source_metadata(path, original, prepared):
    path = Path(path)
    dates = prepared["data_partida"].dropna()
    return {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size, "original_rows": len(original),
            "original_columns": len(original.columns), "original_teams": original["Team_ID"].nunique(),
            "rows": len(prepared), "columns": len(prepared.columns),
            "games": prepared["partida_id"].nunique(), "teams": prepared["time_id"].nunique(),
            "date_min": dates.min() if not dates.empty else None,
            "date_max": dates.max() if not dates.empty else None,
            "memory_bytes": int(prepared.memory_usage(deep=True).sum())}

