"""Convenções compartilhadas: amostra, quartis lineares e regra IQR."""
import numpy as np
import pandas as pd
from scipy import stats
from config import IQR_FACTOR


def valid_values(series):
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values.loc[np.isfinite(values)]


def iqr_details(series, factor=IQR_FACTOR):
    values = valid_values(series)
    q1, median, q3 = (values.quantile([0.25, 0.5, 0.75], interpolation="linear").tolist()
                      if len(values) else (np.nan, np.nan, np.nan))
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    mask = (values < lower) | (values > upper)
    central = values.loc[~mask]
    return {"q1": q1, "median": median, "q3": q3, "iqr": iqr,
            "lower": lower, "upper": upper,
            "whisker_low": central.min(), "whisker_high": central.max(),
            "outlier_mask": mask, "outliers": values.loc[mask]}


def describe_series(series, factor=IQR_FACTOR):
    values = valid_values(series)
    n = len(values)
    details = iqr_details(values, factor)
    mean = values.mean()
    std = values.std(ddof=1)
    modes = values.mode().tolist()
    if len(modes) == n:
        modes = []
    skew = float(stats.skew(values, bias=False)) if n >= 3 and values.nunique() > 1 else np.nan
    direction = ("indefinida" if not np.isfinite(skew) else "positiva (direita)" if skew > 0
                 else "negativa (esquerda)" if skew < 0 else "simétrica")
    intensity = ("indefinida" if not np.isfinite(skew) else "aproximadamente simétrica" if abs(skew) < 0.5
                 else "moderada" if abs(skew) < 1 else "forte")
    return {**details, "n": n, "missing": len(series) - n, "mean": mean,
            "modes": modes, "min": values.min(), "max": values.max(),
            "range": values.max() - values.min(), "std": std, "variance": values.var(ddof=1),
            "cv": 100 * std / mean if n > 1 and mean > 0 else np.nan,
            "robust_ratio": details["iqr"] / details["median"] if details["median"] > 0 else np.nan,
            "skew": skew, "direction": direction, "intensity": intensity,
            "outlier_count": int(details["outlier_mask"].sum())}


def describe_by_team(df, variables, teams, factor=IQR_FACTOR):
    results = {}
    for team_id in teams:
        team_df = df.loc[df["time_id"] == team_id]
        for variable in variables:
            results[(team_id, variable)] = describe_series(team_df[variable], factor)
    return results


def standardized_team_means(df, variable, team_ids):
    """Referência sempre conjunta, independente do time atualmente visível."""
    reference = valid_values(df[variable])
    mean, std = reference.mean(), reference.std(ddof=1)
    means = {}
    for team_id in team_ids:
        values = valid_values(df.loc[df["time_id"] == team_id, variable])
        means[team_id] = float(((values - mean) / std).mean()) if std > 0 else np.nan
    return means, {"mean": mean, "std": std, "n": len(reference)}

