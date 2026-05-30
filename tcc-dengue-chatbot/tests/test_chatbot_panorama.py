"""Testes do branch panorama (sem PostgreSQL)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from app import chatbot


class TestChatbotPanorama(unittest.TestCase):
    def test_panorama_has_data_with_indicador(self) -> None:
        dados = {
            "indicador": pd.DataFrame([{"total_casos": 10}]),
            "tendencia": {"serie": pd.DataFrame(), "resumo": {}},
        }
        self.assertTrue(chatbot._panorama_has_data(dados))

    def test_panorama_has_data_with_tendencia_resumo(self) -> None:
        dados = {
            "tendencia": {"serie": pd.DataFrame(), "resumo": {"casos_ultimas_n": 5}},
        }
        self.assertTrue(chatbot._panorama_has_data(dados))

    def test_panorama_empty(self) -> None:
        dados = {
            "indicador": pd.DataFrame(),
            "tendencia": {"serie": pd.DataFrame(), "resumo": {}},
        }
        self.assertFalse(chatbot._panorama_has_data(dados))

    @patch("app.chatbot.queries.get_indicador_municipio")
    @patch("app.chatbot.queries.get_tendencia_municipio")
    @patch("app.chatbot.queries.comparar_municipio_com_estado")
    @patch("app.chatbot.retrieve_context")
    @patch("app.chatbot.llm.generate_answer")
    @patch("app.chatbot.nlu.interpret_question")
    @patch("app.chatbot._log_consulta")
    def test_answer_panorama_municipio(
        self,
        _log: object,
        mock_nlu: object,
        mock_llm: object,
        mock_rag: object,
        mock_comp: object,
        mock_trend: object,
        mock_ind: object,
    ) -> None:
        mock_nlu.return_value = {
            "intencao": "panorama",
            "municipios": ["Serra Talhada"],
            "ufs": ["PE"],
            "ano": 2025,
            "metrica": "casos",
            "entidades": [],
            "observacoes": [],
        }
        mock_ind.return_value = pd.DataFrame(
            [{"nome_municipio": "Serra Talhada", "uf_sigla": "PE", "total_casos": 42, "incidencia_100k": 100.0}]
        )
        mock_trend.return_value = {
            "serie": pd.DataFrame([{"semana_epidemiologica": 1, "casos": 5}]),
            "resumo": {"classificacao": "aumento", "variacao_percentual": 10.0},
        }
        mock_comp.return_value = pd.DataFrame([{"acima_media_estadual": True}])
        mock_rag.return_value = []
        mock_llm.return_value = "Panorama gerado."

        out = chatbot.answer_question("Quero saber sobre Serra Talhada")
        self.assertEqual(out["tipo_consulta"], "panorama")
        self.assertEqual(out["resposta"], "Panorama gerado.")
        mock_llm.assert_called_once()
        payload = mock_llm.call_args.kwargs["dados_calculados"]
        self.assertEqual(payload.get("tipo_consulta"), "panorama")
        self.assertIn("indicador", payload)


if __name__ == "__main__":
    unittest.main()
