"""Testes unitários do NLU (sem dependência de PostgreSQL)."""

from __future__ import annotations

import unittest

from app import nlu


class TestNLU(unittest.TestCase):
    def test_compare_media_parana_vs_palmas(self) -> None:
        result = nlu.interpret_question(
            "compare a média do parana com a média de palmas to"
        )
        self.assertEqual(result["intencao"], "comparacao")
        self.assertEqual(result["metrica"], "incidencia")
        self.assertGreaterEqual(len(result["entidades"]), 2)
        tipos = {entity["tipo"] for entity in result["entidades"]}
        self.assertIn("uf", tipos)
        self.assertIn("municipio", tipos)
        ufs = {entity.get("uf") for entity in result["entidades"] if entity["tipo"] == "uf"}
        self.assertIn("PR", ufs)

    def test_compare_ufs(self) -> None:
        result = nlu.interpret_question("compare sao paulo e rio de janeiro em 2025")
        self.assertEqual(result["intencao"], "comparacao")
        self.assertEqual(result["ano"], 2025)
        self.assertEqual(len(result["entidades"]), 2)
        self.assertTrue(all(entity["tipo"] == "uf" for entity in result["entidades"]))

    def test_vs_pattern(self) -> None:
        result = nlu.interpret_question("parana vs palmas to")
        self.assertEqual(result["intencao"], "comparacao")
        self.assertGreaterEqual(len(result["entidades"]), 2)

    def test_incidencia_araguaina(self) -> None:
        result = nlu.interpret_question("qual a incidencia em araguaina")
        self.assertEqual(result["intencao"], "incidencia")
        self.assertTrue(result["municipios"])

    def test_media_estado_palmas_tocantins(self) -> None:
        result = nlu.interpret_question("Palmas está acima da média do Tocantins em 2025")
        self.assertEqual(result["intencao"], "media_estado")
        self.assertIn("Palmas", result["municipios"])
        self.assertIn("TO", result["ufs"])
        self.assertEqual(result["ano"], 2025)

    def test_resolve_uf_typos(self) -> None:
        self.assertEqual(nlu.resolve_uf("tocatins"), "TO")
        self.assertEqual(nlu.resolve_uf("parana"), "PR")
        self.assertEqual(nlu.resolve_uf("PR"), "PR")

    def test_count_valid_entities(self) -> None:
        parsed = {
            "intencao": "comparacao",
            "entidades": [{"tipo": "uf", "uf": "PR"}, {"tipo": "municipio", "nome": "Palmas"}],
            "municipios": [],
            "ufs": [],
        }
        self.assertEqual(nlu.count_valid_entities(parsed), 2)


if __name__ == "__main__":
    unittest.main()
