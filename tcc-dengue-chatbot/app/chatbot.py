"""Orquestração do assistente analítico epidemiológico conversacional."""

from __future__ import annotations

import json
import logging
import unicodedata
from typing import Any

import pandas as pd
from sqlalchemy import text

from app import llm, nlu, queries
from app.db import get_engine
from app.rag import retrieve_context

logger = logging.getLogger(__name__)

MAX_REGISTROS_LLM = 20
MAX_RANKING_LLM = 10
MAX_RAG_CARTAS_LLM = 5


def _strip_accents(value: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFD", value)
        if unicodedata.category(c) != "Mn"
    )


def _norm(value: str) -> str:
    return _strip_accents(str(value or "").strip().lower())


def _dedupe_strs(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        s = str(raw or "").strip()
        if not s:
            continue
        key = _norm(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _normalize_uf(uf: Any) -> str | None:
    if uf is None:
        return None
    s = str(uf).strip().upper()[:2]
    if len(s) != 2 or s.startswith("("):
        return None
    return s


def _log_consulta(pergunta: str, resposta: str, filtros: dict[str, Any]) -> None:
    try:
        engine = get_engine()
        with engine.begin() as conn:
            filtros_json = json.dumps(filtros, ensure_ascii=False, default=str)
            conn.execute(
                text(
                    """
                    INSERT INTO log_consulta (pergunta, resposta, filtros)
                    VALUES (:pergunta, :resposta, :filtros::jsonb)
                    """
                ),
                {"pergunta": pergunta, "resposta": resposta, "filtros": filtros_json},
            )
    except Exception:
        return


def _merge_ui_filters(
    interpretacao: dict[str, Any], filtros_ui: dict[str, Any] | None
) -> dict[str, Any]:
    merged = dict(interpretacao or {})
    ui = dict(filtros_ui or {})

    ui_municipio = ui.get("municipio")
    if ui_municipio and str(ui_municipio).strip() not in {"(qualquer)", "(sem dados)"}:
        merged["municipios"] = [str(ui_municipio).strip()]

    ui_uf = _normalize_uf(ui.get("uf") or ui.get("uf_sigla"))
    if ui_uf:
        merged["ufs"] = [ui_uf]

    if ui.get("ano") is not None:
        try:
            merged["ano"] = int(ui["ano"])
        except Exception:
            pass

    if ui.get("semana_epidemiologica") is not None:
        try:
            merged["semana"] = int(ui["semana_epidemiologica"])
        except Exception:
            pass

    merged["municipios"] = _dedupe_strs(list(merged.get("municipios") or []))
    merged["ufs"] = _dedupe_strs([str(x).upper()[:2] for x in list(merged.get("ufs") or [])])
    return merged


def _ambiguity_message(term: str, df: pd.DataFrame) -> str:
    linhas = [f"Encontrei mais de um município relacionado a '{term}'. Você quis dizer:"]
    for i, row in enumerate(df.itertuples(index=False), start=1):
        linhas.append(f"{i}. {row.nome_municipio}/{row.uf_sigla}")
    linhas.append("Responda com o município e a UF para eu calcular corretamente.")
    return "\n".join(linhas)


def _resolve_single_municipio(
    termo: str,
    uf_hint: str | None = None,
) -> dict[str, Any]:
    nome = str(termo or "").strip()
    if not nome:
        return {"nome": None, "uf": uf_hint, "nota": None, "resposta": None}

    df = queries.buscar_municipios(nome, uf=uf_hint)
    if df.empty and uf_hint:
        df = queries.buscar_municipios(nome, uf=None)
    if df.empty:
        if nome_norm := _norm(nome):
            if nome_norm == "palmas" and not uf_hint:
                return {
                    "nome": "Palmas",
                    "uf": "TO",
                    "nota": "Assumi Palmas/TO. Se você quis dizer Palmas/PR ou Palmas de Monte Alto/BA, informe a UF.",
                    "resposta": None,
                }
        return {"nome": nome, "uf": uf_hint, "nota": None, "resposta": None}

    nome_norm = _norm(nome)
    exact = df[df["nome_municipio"].astype(str).map(_norm) == nome_norm].copy()
    pool = exact if not exact.empty else df.copy()

    if nome_norm == "palmas":
        palmas_to = pool[
            (pool["nome_municipio"].astype(str).map(_norm) == "palmas")
            & (pool["uf_sigla"].astype(str).str.upper() == "TO")
        ]
        if not palmas_to.empty:
            chosen = palmas_to.iloc[0]
            other_rows = df[df["id_municipio_ibge"] != chosen["id_municipio_ibge"]].head(10)
            nota = "Assumi Palmas/TO."
            if not other_rows.empty:
                options = [f"{r.nome_municipio}/{r.uf_sigla}" for r in other_rows.itertuples(index=False)]
                nota += " Se você quis dizer " + " ou ".join(options) + ", informe a UF."
            return {
                "nome": str(chosen["nome_municipio"]),
                "uf": str(chosen["uf_sigla"]).upper(),
                "nota": nota,
                "resposta": None,
            }

    if len(pool) == 1:
        chosen = pool.iloc[0]
        return {
            "nome": str(chosen["nome_municipio"]),
            "uf": str(chosen["uf_sigla"]).upper(),
            "nota": None,
            "resposta": None,
        }

    if len(exact) == 1:
        chosen = exact.iloc[0]
        return {
            "nome": str(chosen["nome_municipio"]),
            "uf": str(chosen["uf_sigla"]).upper(),
            "nota": None,
            "resposta": None,
        }

    return {"nome": None, "uf": uf_hint, "nota": None, "resposta": _ambiguity_message(nome, df.head(10))}


def _resolve_municipios(
    interpretacao: dict[str, Any],
) -> tuple[list[str], list[str], list[str], str | None]:
    requested = _dedupe_strs(list(interpretacao.get("municipios") or []))
    uf_hint = _normalize_uf((interpretacao.get("ufs") or [None])[0])

    if not requested:
        return [], _dedupe_strs(list(interpretacao.get("ufs") or [])), [], None

    resolved_names: list[str] = []
    resolved_ufs: list[str] = []
    notes: list[str] = []

    for termo in requested:
        result = _resolve_single_municipio(termo, uf_hint=uf_hint)
        if result["resposta"]:
            return [], [], notes, str(result["resposta"])
        if result["nome"]:
            resolved_names.append(str(result["nome"]))
        if result["uf"]:
            resolved_ufs.append(str(result["uf"]))
        if result["nota"]:
            notes.append(str(result["nota"]))

    return _dedupe_strs(resolved_names), _dedupe_strs(resolved_ufs), _dedupe_strs(notes), None


def _need_more_filters_message(interp: dict[str, Any]) -> str | None:
    intent = str(interp.get("intencao") or "desconhecida")
    municipios = list(interp.get("municipios") or [])
    ufs = list(interp.get("ufs") or [])
    ano = interp.get("ano")

    if intent == "incidencia" and not (municipios or ufs):
        return "Informe o município ou UF para calcular a incidência."
    if intent == "ranking" and not (ufs or ano):
        return "Informe pelo menos a UF ou o ano para montar o ranking."
    if intent == "comparacao" and len(municipios) < 2:
        return "Informe pelo menos dois municípios para comparar."
    if intent == "tendencia" and not municipios:
        return "Informe o município para analisar a tendência nas últimas semanas."
    if intent == "media_estado" and not municipios:
        return "Informe o município para comparar com a média estadual."
    if intent == "pico" and not municipios:
        return "Informe o município para localizar o pico epidemiológico."
    if intent == "total" and not (municipios or ufs or ano):
        return "Informe pelo menos município, UF ou ano para calcular o total."
    return None


def _build_rag_filters(interp: dict[str, Any]) -> dict[str, Any]:
    municipios = list(interp.get("municipios") or [])
    ufs = list(interp.get("ufs") or [])
    return {
        "municipio": municipios[0] if municipios else None,
        "uf": ufs[0] if ufs else None,
        "ano": interp.get("ano"),
        "semana_epidemiologica": interp.get("semana"),
    }


def _compact_dados_para_llm(
    dados_calculados: dict[str, Any],
    tipo_consulta: str,
    df_contexto: pd.DataFrame | None,
) -> dict[str, Any]:
    out = {k: v for k, v in dados_calculados.items() if k != "resultado"}
    lim = MAX_RANKING_LLM if tipo_consulta == "ranking" else MAX_REGISTROS_LLM

    if isinstance(df_contexto, pd.DataFrame):
        total = len(df_contexto)
        head = df_contexto.head(lim)
        out["resultado"] = head.to_dict(orient="records")
        out["_limite_llm"] = {
            "linhas_sql": total,
            "linhas_enviadas_ao_llm": len(head),
            "maximo": lim,
        }
        if total > lim:
            out["_aviso"] = (
                f"Resultado truncado para o LLM: {len(head)} de {total} linhas."
            )
        return out

    result = dados_calculados.get("resultado")
    if isinstance(result, list):
        out["resultado"] = result[:lim]
        out["_limite_llm"] = {
            "linhas_sql": len(result),
            "linhas_enviadas_ao_llm": min(len(result), lim),
            "maximo": lim,
        }
        return out

    out["resultado"] = result
    return out


def _empty_result(
    resposta: str,
    *,
    interpretacao: dict[str, Any],
    filtros_ui: dict[str, Any],
) -> dict[str, Any]:
    filtros = _build_rag_filters(interpretacao)
    _log_consulta(
        pergunta=interpretacao.get("_pergunta_original", ""),
        resposta=resposta,
        filtros={"tipo_consulta": interpretacao.get("intencao"), **filtros, "filtros_ui": filtros_ui},
    )
    return {
        "resposta": resposta,
        "dados": pd.DataFrame(),
        "contexto": [],
        "filtros": filtros,
        "tipo_consulta": interpretacao.get("intencao", "desconhecida"),
        "interpretacao": {k: v for k, v in interpretacao.items() if not str(k).startswith("_")},
    }


def answer_question(pergunta: str, filtros_ui: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Fluxo:
    1. NLU interpreta a pergunta.
    2. Resolve municípios / ambiguidades.
    3. Executa SQL/Python.
    4. Recupera contexto RAG.
    5. Gera resposta via OpenAI sem deixar o LLM calcular números.
    """
    filtros_ui = dict(filtros_ui or {})
    logger.info("[CHATBOT] antes NLU")
    interpretacao = nlu.interpret_question(pergunta)
    interpretacao = _merge_ui_filters(interpretacao, filtros_ui)
    interpretacao["_pergunta_original"] = pergunta
    logger.info("[CHATBOT] depois NLU interpretacao=%s", interpretacao)

    resolved_names, resolved_ufs, notes, ambiguity = _resolve_municipios(interpretacao)
    if ambiguity:
        logger.info("[CHATBOT] ambiguidade de município: %s", ambiguity)
        interpretacao["municipios"] = list(interpretacao.get("municipios") or [])
        interpretacao["observacoes"] = notes
        return _empty_result(ambiguity, interpretacao=interpretacao, filtros_ui=filtros_ui)

    if resolved_names:
        interpretacao["municipios"] = resolved_names
    if resolved_ufs:
        interpretacao["ufs"] = _dedupe_strs(resolved_ufs + list(interpretacao.get("ufs") or []))
    if notes:
        interpretacao["observacoes"] = notes

    filtro_msg = _need_more_filters_message(interpretacao)
    if filtro_msg:
        logger.info("[CHATBOT] consulta ampla/insuficiente: %s", filtro_msg)
        return _empty_result(filtro_msg, interpretacao=interpretacao, filtros_ui=filtros_ui)

    intent = str(interpretacao.get("intencao") or "desconhecida")
    municipios = list(interpretacao.get("municipios") or [])
    ufs = list(interpretacao.get("ufs") or [])
    municipio = municipios[0] if municipios else None
    uf = _normalize_uf(ufs[0]) if ufs else None
    ano = interpretacao.get("ano")
    semana = interpretacao.get("semana")
    metrica = str(interpretacao.get("metrica") or "casos")
    limite = int(interpretacao.get("limite") or 10)

    if intent == "desconhecida":
        return _empty_result(
            "Não consegui identificar o tipo de análise desejada. Tente perguntar sobre incidência, total de casos, ranking, comparação, tendência, média estadual ou pico epidemiológico.",
            interpretacao=interpretacao,
            filtros_ui=filtros_ui,
        )

    logger.info(
        "[CHATBOT] antes SQL intent=%s municipio=%s uf=%s ano=%s semana=%s",
        intent,
        municipio,
        uf,
        ano,
        semana,
    )

    dados_ui: Any = pd.DataFrame()
    dados_calculados: dict[str, Any] = {
        "tipo_consulta": intent,
        "filtros": {
            "municipios": municipios,
            "ufs": ufs,
            "ano": ano,
            "semana": semana,
            "metrica": metrica,
        },
    }

    try:
        if intent in {"incidencia", "total"}:
            df = queries.get_indicador_municipio(municipio=municipio, uf=uf, ano=ano)
            dados_ui = df
        elif intent == "ranking":
            df = queries.get_ranking(uf=uf, ano=ano, metrica=metrica, limit=limite)
            dados_ui = df
        elif intent == "comparacao":
            df = queries.comparar_municipios(municipios=municipios, uf=uf, ano=ano, metrica=metrica)
            dados_ui = df
        elif intent == "tendencia":
            trend = queries.get_tendencia_municipio(municipio=municipio, uf=uf, ano=ano, ultimas_n=6)
            dados_ui = trend.get("serie", pd.DataFrame())
            dados_calculados["resumo"] = trend.get("resumo", {})
        elif intent == "media_estado":
            df = queries.comparar_municipio_com_estado(municipio=municipio, uf=uf, ano=ano)
            dados_ui = df
        elif intent == "pico":
            df = queries.get_pico_epidemiologico(municipio=municipio, uf=uf, ano=ano)
            dados_ui = df
        else:
            dados_ui = pd.DataFrame()
    except Exception as exc:
        dados_ui = pd.DataFrame()
        dados_calculados["erro_query"] = str(exc)

    logger.info(
        "[CHATBOT] depois SQL linhas=%s erro=%s",
        len(dados_ui) if isinstance(dados_ui, pd.DataFrame) else 0,
        dados_calculados.get("erro_query"),
    )

    if isinstance(dados_ui, pd.DataFrame) and dados_ui.empty and not dados_calculados.get("erro_query"):
        resposta = "Não encontrei dados para essa combinação de filtros."
        if notes:
            resposta = "\n\n".join(notes + [resposta])
        _log_consulta(
            pergunta=pergunta,
            resposta=resposta,
            filtros={"tipo_consulta": intent, **_build_rag_filters(interpretacao), "filtros_ui": filtros_ui},
        )
        return {
            "resposta": resposta,
            "dados": dados_ui,
            "contexto": [],
            "filtros": _build_rag_filters(interpretacao),
            "tipo_consulta": intent,
            "interpretacao": {k: v for k, v in interpretacao.items() if not str(k).startswith("_")},
        }

    logger.info("[CHATBOT] antes RAG")
    filtros_rag = _build_rag_filters(interpretacao)
    contexto = retrieve_context(pergunta, filtros=filtros_rag, top_k=5)
    contexto_llm = list(contexto or [])[:MAX_RAG_CARTAS_LLM]
    logger.info("[CHATBOT] depois RAG cartas=%s", len(contexto_llm))

    dados_para_llm = _compact_dados_para_llm(dados_calculados, intent, dados_ui if isinstance(dados_ui, pd.DataFrame) else None)
    logger.info(
        "[CHATBOT] payload LLM linhas=%s rag=%s",
        dados_para_llm.get("_limite_llm", {}).get("linhas_enviadas_ao_llm", 0),
        len(contexto_llm),
    )

    logger.info("[CHATBOT] antes LLM")
    resposta = llm.generate_answer(
        pergunta=pergunta,
        dados_calculados=dados_para_llm,
        contexto_rag=contexto_llm,
    )
    logger.info("[CHATBOT] depois LLM chars=%s", len(resposta or ""))

    if notes:
        resposta = "\n\n".join(notes + [resposta])

    _log_consulta(
        pergunta=pergunta,
        resposta=resposta,
        filtros={"tipo_consulta": intent, **filtros_rag, "filtros_ui": filtros_ui},
    )

    return {
        "resposta": resposta,
        "dados": dados_ui,
        "contexto": contexto,
        "filtros": filtros_rag,
        "tipo_consulta": intent,
        "interpretacao": {k: v for k, v in interpretacao.items() if not str(k).startswith("_")},
    }


def reply(user_message: str) -> str:
    out = answer_question(user_message, filtros_ui=None)
    return str(out.get("resposta", ""))
