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

    def test_compare_palmas_estado_parana(self) -> None:
        for pergunta in (
            "Compare palmas com o estado do paraná",
            "Compare palmas com o estado do parana em 2025",
        ):
            result = nlu.interpret_question(pergunta)
            self.assertEqual(result["intencao"], "comparacao", msg=pergunta)
            self.assertGreaterEqual(len(result["entidades"]), 2, msg=pergunta)
            tipos = {entity["tipo"] for entity in result["entidades"]}
            self.assertIn("uf", tipos, msg=pergunta)
            self.assertIn("municipio", tipos, msg=pergunta)
            ufs = {
                entity.get("uf")
                for entity in result["entidades"]
                if entity.get("tipo") == "uf"
            }
            self.assertIn("PR", ufs, msg=pergunta)

    def test_compare_palmas_serra_talhada(self) -> None:
        result = nlu.interpret_question("Compare Palmas e Serra Talhada em 2025")
        self.assertEqual(result["intencao"], "comparacao")
        self.assertEqual(result["ano"], 2025)
        self.assertGreaterEqual(len(result["entidades"]), 2)
        self.assertTrue(
            all(entity.get("tipo") == "municipio" for entity in result["entidades"])
        )

    def test_incidencia_palmas_durante_ano(self) -> None:
        result = nlu.interpret_question(
            "qual foi a incidencia da dengue em palmas durante o ano de 2025?"
        )
        self.assertEqual(result["intencao"], "incidencia")
        self.assertEqual(result["ano"], 2025)
        self.assertTrue(result["municipios"])
        self.assertIn("palmas", result["municipios"][0].lower())
        self.assertNotIn("2025", result["municipios"][0])
        self.assertNotIn("durante", result["municipios"][0].lower())

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

    def test_serra_talhada_vague(self) -> None:
        result = nlu.interpret_question("Quero saber sobre Serra Talhada")
        self.assertEqual(result["intencao"], "panorama")
        self.assertTrue(result["municipios"])
        self.assertIn("Serra Talhada", result["municipios"][0])

    def test_palmas_tocantins_vague(self) -> None:
        result = nlu.interpret_question("palmas tocantins")
        self.assertEqual(result["intencao"], "panorama")
        self.assertTrue(result["municipios"])
        self.assertTrue(any("palmas" in m.lower() for m in result["municipios"]))
        self.assertIn("TO", result["ufs"])

    def test_tocantins_uf_only_panorama(self) -> None:
        result = nlu.interpret_question("tocantins")
        self.assertEqual(result["intencao"], "panorama")
        self.assertIn("TO", result["ufs"])

    def test_infer_intent_when_vague(self) -> None:
        parsed = {
            "intencao": "desconhecida",
            "municipios": ["Araguaína"],
            "ufs": ["TO"],
        }
        out = nlu.infer_intent_when_vague(parsed)
        self.assertEqual(out["intencao"], "panorama")

    def test_explicit_intent_not_overridden(self) -> None:
        parsed = {
            "intencao": "incidencia",
            "municipios": ["Palmas"],
            "ufs": ["TO"],
        }
        out = nlu.infer_intent_when_vague(parsed)
        self.assertEqual(out["intencao"], "incidencia")


if __name__ == "__main__":
    unittest.main()
