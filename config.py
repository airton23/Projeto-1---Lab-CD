"""Configuração do recorte e das convenções do projeto."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "stats_times_regular_final.csv"
TEAMS = {
    "1610612741": {"name": "Chicago Bulls", "short": "CHI", "color": "#bc3545"},
    "1610612750": {"name": "Minnesota Timberwolves", "short": "MIN", "color": "#137b69"},
}
VARIABLES = (
    "pontos", "rebotes", "assistencias", "roubos_bola", "tocos",
    "faltas_pessoais", "cestas_tres_convertidas",
)
IQR_FACTOR = 1.5
ZSCORE_LIMIT = 3.0
GRUBBS_ALPHA = 0.05
CONTAMINATION = 0.05
KNN_NEIGHBORS = 5
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5
LOF_NEIGHBORS = 20

