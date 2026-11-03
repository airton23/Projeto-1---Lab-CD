"""Detectores univariados com retorno e identificação de observações comuns."""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from pyod.models.ecod import ECOD

from analytics.descriptive import iqr_details, valid_values
from config import (CONTAMINATION, DBSCAN_EPS, DBSCAN_MIN_SAMPLES, GRUBBS_ALPHA,
                    IQR_FACTOR, KNN_NEIGHBORS, LOF_NEIGHBORS, ZSCORE_LIMIT)

METHODS = {"iqr": "IQR / Boxplot", "zscore": "Z-score", "grubbs": "Grubbs",
           "ecod": "ECOD", "knn": "KNN", "dbscan": "DBSCAN", "lof": "LOF"}


def detect_outliers(series, method, parameters=None):
    if method not in METHODS:
        raise ValueError(f"Método desconhecido: {method}")
    parameters = parameters or {}
    values = valid_values(series)
    n = len(values)
    scores = pd.Series(np.nan, index=values.index, dtype=float)
    flags = pd.Series(False, index=values.index)
    result = {"method": method, "name": METHODS[method], "n": n, "status": "calculado",
              "notes": [], "parameters": {}, "threshold": None}
    minimum = 3 if method == "grubbs" else 2
    if n < minimum or values.nunique() < 2:
        result["status"] = "não aplicável"
        result["notes"].append(f"Exige pelo menos {minimum} valores válidos e variabilidade.")
    else:
        x = values.to_numpy(dtype=float).reshape(-1, 1)
        scaled = StandardScaler().fit_transform(x)
        if method == "iqr":
            factor = parameters.get("factor", IQR_FACTOR)
            details = iqr_details(values, factor)
            flags = details["outlier_mask"]
            scores = (values - values.median()).abs()
            result.update(parameters={"k": factor, "quartis": "interpolação linear"},
                          threshold=(details["lower"], details["upper"]))
            if details["iqr"] == 0:
                result["notes"].append("IQR zero: qualquer valor fora do intervalo central degenerado é sinalizado pela regra.")
        elif method == "zscore":
            limit = parameters.get("limit", ZSCORE_LIMIT)
            scores = ((values - values.mean()) / values.std(ddof=1)).abs()
            flags = scores > limit
            result.update(parameters={"ddof": 1, "limiar |z|": limit}, threshold=limit)
        elif method == "grubbs":
            alpha = parameters.get("alpha", GRUBBS_ALPHA)
            scores = ((values - values.mean()) / values.std(ddof=1)).abs()
            t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
            critical = (n - 1) / np.sqrt(n) * np.sqrt(t * t / (n - 2 + t * t))
            winner = scores.idxmax()
            if scores.loc[winner] > critical:
                flags.loc[winner] = True
            normality_p = float(stats.shapiro(values).pvalue) if n <= 5000 else None
            result.update(parameters={"alpha": alpha, "bilateral": True, "candidatos": 1,
                                      "p Shapiro–Wilk": normality_p}, threshold=float(critical))
            result["notes"].append("Teste de um único extremo, sem remoção iterativa. Pressupõe normalidade aproximada e independência; testes entre séries são exploratórios.")
            if normality_p is not None and normality_p < 0.05:
                result["status"] = "exploratório"
                result["notes"].append("Shapiro–Wilk rejeita normalidade a 5%; interpretar Grubbs com ressalva, especialmente para contagens discretas.")
        elif method == "ecod":
            contamination = parameters.get("contamination", CONTAMINATION)
            model = ECOD(contamination=contamination, n_jobs=1).fit(x)
            scores = pd.Series(model.decision_scores_, index=values.index)
            flags = scores > model.threshold_
            result.update(parameters={"contamination": contamination}, threshold=float(model.threshold_))
            result["notes"].append("A fração de contaminação define o limiar exploratório; não estima a proporção verdadeira de erros. Empates podem alterar a quantidade sinalizada.")
        elif method == "knn":
            k = min(parameters.get("neighbors", KNN_NEIGHBORS), n - 1)
            distances, neighbors = NearestNeighbors(n_neighbors=k + 1).fit(scaled).kneighbors(scaled)
            # Excluir pelo índice: com empates, a própria linha nem sempre é a primeira.
            kth = [dist[indices != i][:k][-1] for i, (dist, indices) in enumerate(zip(distances, neighbors))]
            scores = pd.Series(kth, index=values.index)
            threshold = float(scores.quantile(0.95, interpolation="linear"))
            flags = scores > threshold
            result.update(parameters={"k": k, "limiar": "percentil 95 linear", "escala": "StandardScaler, ddof=0"}, threshold=threshold)
            result["notes"].append("Distância ao k-ésimo outro vizinho; sinalização estrita acima do percentil 95, sem forçar quantidade de extremos.")
        elif method == "dbscan":
            eps = parameters.get("eps", DBSCAN_EPS)
            min_samples = parameters.get("min_samples", DBSCAN_MIN_SAMPLES)
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
            flags = pd.Series(labels == -1, index=values.index)
            result.update(parameters={"eps": eps, "min_samples": min_samples,
                                      "escala": "StandardScaler, ddof=0"}, threshold="rótulo −1 (ruído)")
            result["notes"].append("Não há escore contínuo nativo nesta saída; ruído depende da escolha de eps e min_samples.")
        elif method == "lof":
            k = min(parameters.get("neighbors", LOF_NEIGHBORS), n - 1)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", UserWarning)
                model = LocalOutlierFactor(n_neighbors=k, contamination="auto")
                labels = model.fit_predict(scaled)
            scores = pd.Series(-model.negative_outlier_factor_, index=values.index)
            flags = pd.Series(labels == -1, index=values.index)
            result.update(parameters={"n_neighbors": k, "contamination": "auto",
                                      "escala": "StandardScaler, ddof=0"}, threshold=float(-model.offset_))
            result["notes"].extend(str(w.message) for w in caught)
        if values.nunique() < min(n, 10) and method in {"ecod", "knn", "lof", "dbscan"}:
            result["notes"].append("Poucos valores distintos: empates e discretização podem tornar a separação por densidade pouco informativa.")
    result["rows"] = pd.DataFrame({"valor": values, "escore": scores, "sinalizado": flags})
    result["count"] = int(flags.sum()) if result["status"] != "não aplicável" else None
    return result

