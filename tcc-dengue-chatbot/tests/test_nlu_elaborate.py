"""Perguntas elaboradas (estilo banca) — NLU deve extrair localidade corretamente."""

from __future__ import annotations

import unittest

from app import nlu


ELABORATE_CASES: list[tuple[str, dict]] = [
    (
        "qual foi a incidencia da dengue em palmas durante o ano de 2025?",
        {"intencao": "incidencia", "ano": 2025, "municipio_contains": "palmas"},
    ),
    (
        "Poderia informar quantos casos de dengue foram notificados em Araguaína-TO no ano de 2024?",
        {"intencao": "total", "ano": 2024, "municipio_contains": "aragua"},
    ),
    (
        "Gostaria de saber a taxa de incidência por 100 mil habitantes na cidade de Palmas Tocantins em 2025",
        {"intencao": "incidencia", "ano": 2025, "municipio_contains": "palmas", "uf": "TO"},
    ),
    (
        "Como está a situação epidemiológica da dengue no município de Serra Talhada Pernambuco?",
        {"intencao": "panorama", "municipio_contains": "serra talhada"},
    ),
    (
        "Me diga o total de casos confirmados em Palmas/TO ao longo de 2025",
        {"intencao": "total", "ano": 2025, "municipio_contains": "palmas", "uf": "TO"},
    ),
    (
        "Quais foram os 5 municípios com maior incidência de dengue no Tocantins em 2025?",
        {"intencao": "ranking", "ano": 2025, "uf": "TO"},
    ),
    (
        "Em qual semana epidemiológica ocorreu o pico de casos de dengue em Palmas em 2025?",
        {"intencao": "pico", "ano": 2025, "municipio_contains": "palmas"},
    ),
    (
        "Palmas está acima da média estadual do Tocantins em 2025?",
        {"intencao": "media_estado", "ano": 2025, "municipio_contains": "palmas", "uf": "TO"},
    ),
]


class TestNLUElaborate(unittest.TestCase):
    def test_elaborate_questions(self) -> None:
        for pergunta, expected in ELABORATE_CASES:
            result = nlu.interpret_question(pergunta)
            self.assertEqual(
                result.get("intencao"),
                expected["intencao"],
                msg=f"intent: {pergunta!r} -> {result}",
            )
            if expected.get("ano") is not None:
                self.assertEqual(result.get("ano"), expected["ano"], msg=pergunta)
            if expected.get("municipio_contains"):
                municipios = [str(m).lower() for m in result.get("municipios") or []]
                self.assertTrue(
                    municipios,
                    msg=f"sem município: {pergunta!r} -> {result}",
                )
                self.assertTrue(
                    any(expected["municipio_contains"] in m for m in municipios),
                    msg=f"município: {pergunta!r} -> {municipios}",
                )
                for m in municipios:
                    self.assertNotIn("durante", m, msg=pergunta)
                    self.assertNotIn("dengue", m, msg=pergunta)
                    self.assertFalse(
                        nlu.is_suspect_municipio_name(m),
                        msg=f"suspeito: {pergunta!r} -> {m}",
                    )
            if expected.get("uf"):
                self.assertIn(expected["uf"], result.get("ufs") or [], msg=pergunta)


if __name__ == "__main__":
    unittest.main()
