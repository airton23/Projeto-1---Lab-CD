"""Auditoria de células e relações, preservando ocorrências rastreáveis."""
import numpy as np
import pandas as pd

from analytics.descriptive import iqr_details
from analytics.schema import MEASUREMENTS, NUMERIC, SCHEMA


def audit_quality(df, raw, teams, original=None):
    issues = []
    invalid = pd.DataFrame(False, index=df.index, columns=df.columns)

    def record(mask, columns, rule, exclude=True):
        mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
        for row_id in df.index[mask]:
            for column in columns:
                issues.append({"Linha CSV": int(row_id), "Time": teams.get(str(df.at[row_id, "time_id"]), {}).get("name", "Desconhecido"),
                               "Partida": df.at[row_id, "partida_id"], "Coluna": column,
                               "Valor original": raw.at[row_id, column], "Regra": rule})
                if exclude:
                    invalid.at[row_id, column] = True

    for column, meta in SCHEMA.items():
        values = df[column]
        record(raw[column].notna() & values.isna(), [column], "Falha de conversão")
        if column in NUMERIC:
            numeric = pd.to_numeric(values, errors="coerce").astype(float)
            record(values.notna() & ~np.isfinite(numeric), [column], "Valor não finito")
            record(numeric < 0, [column], "Valor negativo fora do domínio")
            if "contagem" in meta["kind"]:
                record(numeric.notna() & np.isfinite(numeric) & (numeric % 1 != 0), [column], "Contagem fracionária")
            if "proporção" in meta["kind"]:
                record((numeric < 0) | (numeric > 1), [column], "Proporção fora de [0, 1]")
            if meta["kind"] == "binária":
                record(values.notna() & ~values.isin([0, 1]), [column], "Resultado fora de {0, 1}")
        if column in {"time_id", "partida_id", "time", "data_partida"}:
            record(raw[column].isna(), [column], "Campo obrigatório ausente")

    record(df["minutos_equipe"].le(0), ["minutos_equipe"], "Minutos devem ser positivos")
    for team_id, meta in teams.items():
        record(df["time_id"].eq(team_id) & df["time"].astype("string").ne(meta["name"]),
               ["time"], "Nome incompatível com ID da equipe", exclude=False)

    relationships = [
        (["rebotes", "rebotes_ofensivos", "rebotes_defensivos"],
         df["rebotes"] != df["rebotes_ofensivos"] + df["rebotes_defensivos"], "REB ≠ OREB + DREB"),
        (["pontos", "arremessos_quadra_convertidos", "cestas_tres_convertidas", "lances_livres_convertidos"],
         df["pontos"] != 2 * df["arremessos_quadra_convertidos"] + df["cestas_tres_convertidas"] + df["lances_livres_convertidos"],
         "PTS ≠ 2 × FGM + FG3M + FTM"),
    ]
    for made, attempted in [("arremessos_quadra_convertidos", "arremessos_quadra_tentados"),
                             ("cestas_tres_convertidas", "cestas_tres_tentadas"),
                             ("lances_livres_convertidos", "lances_livres_tentados"),
                             ("cestas_tres_convertidas", "arremessos_quadra_convertidos"),
                             ("cestas_tres_tentadas", "arremessos_quadra_tentados")]:
        relationships.append(([made, attempted], df[made] > df[attempted], "Convertidos/tentados incompatíveis"))
    for columns, mask, rule in relationships:
        present = df[columns].notna().all(axis=1)
        record(present & mask, columns, rule)

    ratios = [("arremessos_quadra_convertidos", "arremessos_quadra_tentados", "aproveitamento_quadra"),
              ("cestas_tres_convertidas", "cestas_tres_tentadas", "aproveitamento_tres"),
              ("lances_livres_convertidos", "lances_livres_tentados", "aproveitamento_lance_livre")]
    for made, attempted, ratio in ratios:
        expected = df[made] / df[attempted].where(df[attempted] > 0)
        record(df[ratio].notna() & expected.notna() & (df[ratio] - expected).abs().gt(0.00050001),
               [ratio], "Aproveitamento incompatível com razão (tolerância 0,0005)")
    total = df["vitorias_acumuladas"] + df["derrotas_acumuladas"]
    expected = df["vitorias_acumuladas"] / total.where(total > 0)
    record(expected.notna() & df["proporcao_vitorias"].notna() & (df["proporcao_vitorias"] - expected).abs().gt(0.00050001),
           ["proporcao_vitorias"], "Aproveitamento acumulado incompatível")

    duplicate_keys = df.duplicated(["time_id", "partida_id"], keep=False)
    record(duplicate_keys, ["partida_id"], "Chave time + partida duplicada", exclude=False)
    exact_duplicates = int(raw.duplicated().sum())
    for team_id in teams:
        ordered = df.loc[df["time_id"].eq(team_id)].sort_values("data_partida")
        # Diferenças só são verificadas entre jogos consecutivos do acumulado:
        # um salto pode significar apenas que a coleta não contém todos os jogos.
        games_played = ordered["vitorias_acumuladas"] + ordered["derrotas_acumuladas"]
        consecutive = games_played.diff().eq(1)
        bad = consecutive & ((ordered["vitorias_acumuladas"].diff() != ordered["venceu"]) |
                             (ordered["derrotas_acumuladas"].diff() != 1 - ordered["venceu"]))
        record(bad.reindex(df.index, fill_value=False), ["vitorias_acumuladas", "derrotas_acumuladas"],
               "Evolução acumulada incompatível com resultado", exclude=False)

    if original is not None:
        original_groups = {str(key): group for key, group in original.groupby("Game_ID")}
        for row_id, row in df.iterrows():
            pair = original_groups.get(str(row["partida_id"]))
            if pair is None or len(pair) != 2 or pair["Team_ID"].nunique() != 2:
                record(df.index == row_id, ["partida_id"], "Estrutura da partida na origem incompleta", exclude=False)
                continue
            opponent = pair.loc[pair["Team_ID"] != row["time_id"]].iloc[0]
            opponent_points = pd.to_numeric(opponent["PTS"], errors="coerce")
            if pd.notna(row["pontos"]) and pd.notna(row["venceu"]) and pd.notna(opponent_points):
                if row["pontos"] == opponent_points or int(row["venceu"]) != int(row["pontos"] > opponent_points):
                    record(df.index == row_id, ["venceu", "pontos"], "Resultado incompatível com placar do adversário")
            for original_col, internal_col in [("GAME_DATE", "data_partida"), ("MIN", "minutos_equipe")]:
                if pair[original_col].nunique() != 1:
                    record(df.index == row_id, [internal_col], "Divergência entre as duas equipes na origem", exclude=False)

    cleaned = df.copy()
    for column in NUMERIC:
        cleaned.loc[invalid[column], column] = pd.NA
    cleaned.loc[invalid["data_partida"], "data_partida"] = pd.NaT
    issue_frame = pd.DataFrame(issues, columns=["Linha CSV", "Time", "Partida", "Coluna", "Valor original", "Regra"])
    summaries = []
    for column, meta in SCHEMA.items():
        affected = (issue_frame.loc[issue_frame["Coluna"] == column, "Linha CSV"].nunique()
                    if len(issue_frame) else 0)
        extremes = 0
        if column in MEASUREMENTS:
            for _, group in cleaned.groupby("time_id", observed=True):
                extremes += int(iqr_details(group[column])["outlier_mask"].sum())
        summaries.append({"Coluna": meta["original"], "Nome": column,
                          "Ausentes na leitura": int(raw[column].isna().sum()),
                          "Ausentes após conversão": int(df[column].isna().sum()),
                          "Ausentes (%)": 100 * float(df[column].isna().sum()) / len(df) if len(df) else np.nan,
                          "Zeros": int(df[column].eq(0).sum()) if column in NUMERIC else "Não aplicável",
                          "Inconsistentes": int(affected),
                          "Outliers IQR por equipe": extremes if column in MEASUREMENTS else "Não aplicável"})
    return {"summary": pd.DataFrame(summaries), "issues": issue_frame, "cleaned": cleaned,
            "invalid": invalid, "exact_duplicates": exact_duplicates,
            "duplicate_key_rows": int(duplicate_keys.sum())}

