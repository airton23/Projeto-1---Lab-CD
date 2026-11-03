import unittest

import numpy as np

from app import create_app
from analytics.charts import build_boxplot
from app.formatting import chart_html, fmt
from config import TEAMS


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app({"TESTING": True})
        cls.client = cls.app.test_client()
        cls.project = cls.app.extensions["project"]

    def test_all_pages_and_local_plotly_asset(self):
        for path in ["/", "/dicionario", "/preparacao", "/qualidade", "/descritiva", "/comparacao", "/relatorio"]:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn('lang="pt-BR"', response.get_data(as_text=True))
        with self.client.get("/assets/plotly.min.js") as response:
            self.assertEqual(response.status_code, 200)

    def test_filters_update_results(self):
        for team_id, team in TEAMS.items():
            html = self.client.get(f"/descritiva?time={team_id}&variavel=tocos").get_data(as_text=True)
            summary = self.project["summaries"][(team_id, "tocos")]
            self.assertIn(team["name"], html)
            self.assertIn(fmt(summary["mean"]), html)
            self.assertIn("74 registros time-partida", html)
            self.assertNotIn("149 registros", html)

    def test_explanations_are_consolidated_in_complete_report(self):
        for path in ["/", "/dicionario", "/preparacao", "/qualidade", "/descritiva", "/comparacao"]:
            with self.subTest(path=path):
                html = self.client.get(path).get_data(as_text=True)
                self.assertNotIn('class="eyebrow"', html)
                self.assertNotIn('class="reading-label"', html)
                self.assertNotIn('class="interpretation"', html)
                self.assertNotIn("Um formato explícito", html)
                self.assertNotIn("Uma visão própria para cada finalidade", html)
                self.assertNotIn("Listing–Schlittgen", html)
        html = self.client.get("/relatorio").get_data(as_text=True)
        self.assertEqual(html.count('data-report-variable="'), 7)
        self.assertEqual(html.count('class="interpretation"'), 14)
        self.assertEqual(html.count("<h1>"), 1)
        for topic in ["%Y-%m-%d", "Resultado binário", "MCAR", "MAR", "MNAR",
                      "Listing–Schlittgen", "Fisher–Pearson", "Grubbs", "Limitações", "Grupo 5"]:
            self.assertIn(topic, html)
        self.assertNotIn("BASQUETE, CONTEXTO E ESTATÍSTICA", html)

    def test_legacy_report_routes_redirect_to_complete_report(self):
        for path, anchor in [("/metodologia", "metodologia"), ("/sintese", "sintese")]:
            with self.subTest(path=path):
                response = self.client.get(f"{path}?time=1610612741")
                self.assertEqual(response.status_code, 302)
                self.assertEqual(response.headers["Location"], f"/relatorio?time=1610612741#{anchor}")

    def test_all_variables_and_detectors_can_be_selected(self):
        for variable in self.project["df"][["pontos", "rebotes", "assistencias", "roubos_bola", "tocos", "faltas_pessoais", "cestas_tres_convertidas"]]:
            with self.subTest(variable=variable):
                self.assertEqual(self.client.get(f"/descritiva?variavel={variable}").status_code, 200)
        for method in ["iqr", "zscore", "grubbs", "ecod", "knn", "dbscan", "lof"]:
            with self.subTest(method=method):
                self.assertEqual(self.client.get(f"/qualidade?metodo={method}&time=1610612741").status_code, 200)

    def test_foreign_teams_and_bad_parameters_are_rejected(self):
        for query in ["time=1610612737", "time=<script>", "variavel=venceu", "metodo=unknown", "medida=total"]:
            self.assertEqual(self.client.get(f"/?{query}").status_code, 400)
        self.assertEqual(self.client.get("/nao-existe").status_code, 404)

    def test_plot_uses_shared_quartiles_and_outliers(self):
        figure = build_boxplot(self.project["df"], "pontos", self.project["summaries"], TEAMS)
        for team_id, team in TEAMS.items():
            summary = self.project["summaries"][(team_id, "pontos")]
            box = next(trace for trace in figure.data if trace.type == "box" and trace.name == team["name"])
            self.assertEqual(box.q1[0], summary["q1"])
            self.assertEqual(box.upperfence[0], summary["whisker_high"])
        plotted = sum(len(trace.y) for trace in figure.data if trace.name in {"Outliers inferiores", "Outliers superiores"})
        expected = sum(self.project["summaries"][(tid, "pontos")]["outlier_count"] for tid in TEAMS)
        self.assertEqual(plotted, expected)
        html = str(chart_html(figure))
        self.assertIn('Plotly.newPlot', html)
        self.assertNotIn("cdn.plot.ly", html)

    def test_comparison_zscore_and_formatting(self):
        html = self.client.get("/comparacao?medida=zscore&time=1610612741").get_data(as_text=True)
        self.assertIn("148 valores válidos", html)
        self.assertIn("Referência conjunta das duas equipes", html)
        self.assertEqual(fmt(1234.5), "1.234,50")
        self.assertEqual(fmt(np.nan), "—")


if __name__ == "__main__":
    unittest.main()

