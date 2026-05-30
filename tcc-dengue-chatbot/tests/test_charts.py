"""Testes de gráficos comparativos (sem PostgreSQL)."""

from __future__ import annotations

import unittest

import pandas as pd

from app import charts, nlu


class TestCharts(unittest.TestCase):
    def test_wants_comparative_chart(self) -> None:
        self.assertTrue(
            charts.wants_comparative_chart(
                "Gere um gráfico comparativo entre Palmas TO e o Paraná em 2025"
            )
        )
        self.assertFalse(charts.wants_comparative_chart("Quantos casos em Palmas?"))

    def test_nlu_grafico_comparativo(self) -> None:
        result = nlu.interpret_question(
            "Gere um gráfico comparativo entre Palmas TO e o Paraná em 2025"
        )
        self.assertEqual(result["intencao"], "comparacao")
        self.assertTrue(result.get("gerar_grafico"))
        entidades = result.get("entidades") or []
        municipios = result.get("municipios") or []
        self.assertTrue(len(entidades) >= 2 or len(municipios) >= 2)

    def test_prepare_municipios_labels(self) -> None:
        df = pd.DataFrame(
            [
                {"nome_municipio": "Palmas", "uf_sigla": "TO", "total_casos": 100, "incidencia_100k": 50.0},
                {"nome_municipio": "Araguaína", "uf_sigla": "TO", "total_casos": 80, "incidencia_100k": 40.0},
            ]
        )
        prepared = charts.prepare_comparacao_df(df)
        self.assertIn("entidade_label", prepared.columns)
        self.assertEqual(len(prepared), 2)
        self.assertTrue(charts.comparacao_chartable(prepared))

    def test_build_figure_requires_two_rows(self) -> None:
        df = pd.DataFrame([{"entidade_label": "A", "total_casos": 1, "incidencia_100k": 1.0}])
        self.assertFalse(charts.comparacao_chartable(df))
        self.assertIsNone(charts.build_comparacao_figure(df))

    def test_build_figure_with_plotly(self) -> None:
        try:
            import plotly  # noqa: F401
        except ImportError:
            self.skipTest("plotly não instalado")
        df = pd.DataFrame(
            [
                {"entidade_label": "Palmas/TO", "total_casos": 100, "incidencia_100k": 50.0},
                {"entidade_label": "Parana (UF)", "total_casos": 5000, "incidencia_100k": 120.0},
            ]
        )
        self.assertIsNotNone(charts.build_comparacao_figure(df, metrica="casos"))


if __name__ == "__main__":
    unittest.main()
