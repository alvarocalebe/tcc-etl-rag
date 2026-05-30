"""Orquestração do assistente analítico epidemiológico conversacional."""

from __future__ import annotations

import json
import logging
import unicodedata
from typing import Any

import pandas as pd
from sqlalchemy import text

from app import charts, llm, nlu, queries
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
    elif ui.get("uf") or ui.get("uf_sigla"):
        resolved = nlu.resolve_uf(str(ui.get("uf") or ui.get("uf_sigla")))
        if resolved:
            merged["ufs"] = [resolved]

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
    linhas.append(
        "Responda com o município e a UF (ex.: Serra Talhada/PE) para eu calcular corretamente."
    )
    return "\n".join(linhas)


def _resolve_single_municipio(
    termo: str,
    uf_hint: str | None = None,
) -> dict[str, Any]:
    nome = nlu._clean_location_phrase(nlu._strip_temporal_suffix(str(termo or "").strip()))
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
    fuzzy_note: str | None = None
    if exact.empty and not pool.empty:
        chosen_name = str(pool.iloc[0]["nome_municipio"])
        if _norm(chosen_name) != nome_norm:
            fuzzy_note = f"Interpretei '{nome}' como '{chosen_name}'."

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
            "nota": fuzzy_note,
            "resposta": None,
        }

    if len(exact) == 1:
        chosen = exact.iloc[0]
        return {
            "nome": str(chosen["nome_municipio"]),
            "uf": str(chosen["uf_sigla"]).upper(),
            "nota": fuzzy_note,
            "resposta": None,
        }

    return {"nome": None, "uf": uf_hint, "nota": None, "resposta": _ambiguity_message(nome, df.head(10))}


def _uf_hint_for_index(interpretacao: dict[str, Any], index: int) -> str | None:
    ufs = [_normalize_uf(u) for u in list(interpretacao.get("ufs") or []) if _normalize_uf(u)]
    if not ufs:
        return None
    if index < len(ufs):
        return ufs[index]
    if len(ufs) == 1:
        return ufs[0]
    return None


def _fetch_indicador_resilient(
    *,
    municipio: str | None,
    uf: str | None,
    ano: int | None,
) -> tuple[pd.DataFrame, str | None]:
    """Busca indicador com fallbacks (UF, resolução de nome, ano mais recente)."""
    if not municipio and not uf:
        return pd.DataFrame(), None

    extra_note: str | None = None
    df = queries.get_indicador_municipio(municipio=municipio, uf=uf, ano=ano)
    if not df.empty:
        return df, None

    if municipio and uf:
        df = queries.get_indicador_municipio(municipio=municipio, uf=None, ano=ano)
        if not df.empty and "uf_sigla" in df.columns:
            scoped = df[df["uf_sigla"].astype(str).str.upper() == str(uf).upper()]
            if not scoped.empty:
                return scoped, None

    if municipio:
        resolved = _resolve_single_municipio(str(municipio), uf_hint=uf)
        if resolved.get("nome"):
            res_name = str(resolved["nome"])
            res_uf = _normalize_uf(resolved.get("uf")) or uf
            if res_name.lower() != str(municipio).lower() or res_uf != uf:
                df = queries.get_indicador_municipio(
                    municipio=res_name, uf=res_uf, ano=ano
                )
                if not df.empty:
                    return df, resolved.get("nota")

    if municipio and ano is not None:
        df_years = queries.get_indicador_municipio(municipio=municipio, uf=uf, ano=None)
        if not df_years.empty and "ano" in df_years.columns:
            anos = pd.to_numeric(df_years["ano"], errors="coerce").dropna()
            if not anos.empty:
                alt_ano = int(anos.max())
                subset = df_years[df_years["ano"] == alt_ano]
                if not subset.empty:
                    extra_note = (
                        f"Não há dados para {ano} na base; usei o ano mais recente disponível ({alt_ano})."
                    )
                    return subset, extra_note

    return pd.DataFrame(), extra_note


def _resolve_municipios(
    interpretacao: dict[str, Any],
) -> tuple[list[str], list[str], list[str], str | None]:
    requested = _dedupe_strs(list(interpretacao.get("municipios") or []))

    if not requested:
        return [], _dedupe_strs(list(interpretacao.get("ufs") or [])), [], None

    resolved_names: list[str] = []
    resolved_ufs: list[str] = []
    notes: list[str] = []

    for index, termo in enumerate(requested):
        uf_hint = _uf_hint_for_index(interpretacao, index)
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


def _resolve_entidades(
    interpretacao: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], str | None]:
    entidades = list(interpretacao.get("entidades") or [])
    notes: list[str] = []
    resolved: list[dict[str, Any]] = []

    for entidade in entidades:
        tipo = str(entidade.get("tipo") or "").strip().lower()
        if tipo == "uf":
            uf = _normalize_uf(entidade.get("uf"))
            if not uf:
                continue
            resolved.append(
                {
                    "tipo": "uf",
                    "uf": uf,
                    "nome": None,
                    "label": str(entidade.get("label") or f"{uf} (UF)"),
                }
            )
            continue

        if tipo != "municipio":
            continue

        nome = str(entidade.get("nome") or "").strip()
        uf_hint = _normalize_uf(entidade.get("uf"))
        if not nome:
            continue

        if nlu.is_likely_state_name(nome):
            uf_state = _normalize_uf(nlu.resolve_uf(nome))
            if uf_state:
                resolved.append(
                    {
                        "tipo": "uf",
                        "uf": uf_state,
                        "nome": None,
                        "label": str(entidade.get("label") or f"{uf_state} (UF)"),
                    }
                )
                continue

        result = _resolve_single_municipio(nome, uf_hint=uf_hint)
        if result["resposta"]:
            return [], notes, str(result["resposta"])
        if not result["nome"]:
            continue

        resolved_name = str(result["nome"])
        resolved_uf = _normalize_uf(result["uf"]) or uf_hint
        label = f"{resolved_name}/{resolved_uf}" if resolved_uf else resolved_name
        resolved.append(
            {
                "tipo": "municipio",
                "nome": resolved_name,
                "uf": resolved_uf,
                "label": str(entidade.get("label") or label),
            }
        )
        if result["nota"]:
            notes.append(str(result["nota"]))

    return resolved, _dedupe_strs(notes), None


def _apply_default_year(interpretacao: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
    data = dict(interpretacao or {})
    if data.get("ano") is not None:
        return data, None
    ano = queries.get_ano_mais_recente()
    if ano is None:
        return data, None
    data["ano"] = int(ano)
    return data, int(ano)


def _comparison_scope_uf(interpretacao: dict[str, Any]) -> str | None:
    """
    UF global só quando há uma única UF na comparação.
    Comparações entre estados ou municípios de UFs diferentes não devem filtrar por uma UF só.
    """
    intent = str(interpretacao.get("intencao") or "")
    if intent != "comparacao":
        ufs = list(interpretacao.get("ufs") or [])
        return _normalize_uf(ufs[0]) if ufs else None

    entidades = list(interpretacao.get("entidades") or [])
    if entidades:
        scoped: list[str] = []
        for item in entidades:
            tipo = str(item.get("tipo") or "").strip().lower()
            if tipo == "uf":
                uf = _normalize_uf(item.get("uf"))
                if uf:
                    scoped.append(uf)
            elif tipo == "municipio":
                uf = _normalize_uf(item.get("uf"))
                if uf:
                    scoped.append(uf)
        unique = _dedupe_strs(scoped)
        if len(unique) == 1:
            return unique[0]
        return None

    municipios = list(interpretacao.get("municipios") or [])
    ufs = [_normalize_uf(u) for u in list(interpretacao.get("ufs") or []) if _normalize_uf(u)]
    unique = _dedupe_strs(ufs)
    if len(municipios) >= 2:
        return unique[0] if len(unique) == 1 else None
    return unique[0] if unique else None


def _municipios_uf_hints(entidades: list[dict[str, Any]]) -> list[str | None] | None:
    if not entidades or not all(str(item.get("tipo") or "") == "municipio" for item in entidades):
        return None
    hints = [_normalize_uf(item.get("uf")) for item in entidades]
    if not any(hints):
        return None
    return hints


def _need_more_filters_message(interp: dict[str, Any]) -> str | None:
    intent = str(interp.get("intencao") or "desconhecida")
    municipios = list(interp.get("municipios") or [])
    ufs = list(interp.get("ufs") or [])
    entidades = list(interp.get("entidades") or [])
    ano = interp.get("ano")

    if intent == "incidencia" and not (municipios or ufs):
        return "Informe o município ou UF para calcular a incidência."
    if intent == "ranking" and not (ufs or ano):
        return "Informe pelo menos a UF ou o ano para montar o ranking."
    if intent == "comparacao" and len(entidades) < 2 and len(municipios) < 2:
        return "Informe pelo menos duas localidades para comparar (municípios, UFs ou uma combinação)."
    if intent == "tendencia" and not municipios:
        return "Informe o município para analisar a tendência nas últimas semanas."
    if intent == "media_estado" and not municipios:
        return "Informe o município para comparar com a média estadual."
    if intent == "pico" and not municipios:
        return "Informe o município para localizar o pico epidemiológico."
    if intent == "total" and not (municipios or ufs or ano):
        return "Informe pelo menos município, UF ou ano para calcular o total."
    if intent == "panorama" and not (municipios or ufs):
        return "Informe o município ou o estado para montar o panorama epidemiológico."
    return None


def _unknown_intent_message(interpretacao: dict[str, Any]) -> str:
    municipios = list(interpretacao.get("municipios") or [])
    ufs = list(interpretacao.get("ufs") or [])
    if municipios or ufs:
        local = municipios[0] if municipios else ufs[0]
        uf_hint = f"/{ufs[0]}" if municipios and ufs else ""
        return (
            f"Identifiquei a localidade ({local}{uf_hint}), mas não entendi o tipo de análise. "
            "Tente, por exemplo:\n"
            f"- Quantos casos de dengue houve em {local}{uf_hint}?\n"
            f"- Qual a incidência em {local}{uf_hint}?\n"
            f"- Como está a tendência em {local}{uf_hint} nas últimas semanas?"
        )
    return (
        "Não identifiquei localidade nem tipo de análise. Exemplos:\n"
        "- Quero saber sobre Serra Talhada\n"
        "- Qual a incidência de dengue em Palmas TO?\n"
        "- Quais municípios do TO tiveram mais casos em 2025?"
    )


def _panorama_has_data(dados_calculados: dict[str, Any]) -> bool:
    for key in ("indicador", "comparacao_estadual", "indicador_uf", "ranking_uf"):
        block = dados_calculados.get(key)
        if isinstance(block, pd.DataFrame) and not block.empty:
            return True
        if isinstance(block, list) and block:
            return True
    tendencia = dados_calculados.get("tendencia") or {}
    if isinstance(tendencia, dict):
        serie = tendencia.get("serie")
        if isinstance(serie, pd.DataFrame) and not serie.empty:
            return True
        resumo = tendencia.get("resumo") or {}
        if resumo.get("casos_ultimas_n") is not None:
            return True
    return False


def _execute_panorama(
    *,
    municipio: str | None,
    uf: str | None,
    ano: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Monta payload de panorama municipal ou estadual."""
    dados_calculados: dict[str, Any] = {
        "tipo_consulta": "panorama",
        "local_resolvida": {"municipio": municipio, "uf": uf, "ano": ano},
    }
    dados_ui = pd.DataFrame()

    if municipio:
        indicador = queries.get_indicador_municipio(municipio=municipio, uf=uf, ano=ano)
        trend = queries.get_tendencia_municipio(municipio=municipio, uf=uf, ano=ano, ultimas_n=6)
        comparacao = queries.comparar_municipio_com_estado(municipio=municipio, uf=uf, ano=ano)
        dados_calculados["indicador"] = indicador
        dados_calculados["tendencia"] = {
            "serie": trend.get("serie", pd.DataFrame()),
            "resumo": trend.get("resumo", {}),
        }
        dados_calculados["comparacao_estadual"] = comparacao
        serie = trend.get("serie", pd.DataFrame())
        dados_ui = serie if isinstance(serie, pd.DataFrame) and not serie.empty else indicador
    elif uf:
        indicador_uf = queries.get_indicador_municipio(uf=uf, ano=ano)
        ranking = queries.get_ranking(uf=uf, ano=ano, metrica="casos", limit=5)
        dados_calculados["indicador_uf"] = indicador_uf
        dados_calculados["ranking_uf"] = ranking
        dados_ui = ranking if isinstance(ranking, pd.DataFrame) and not ranking.empty else indicador_uf
    else:
        dados_calculados["erro"] = "Localidade não informada para panorama."

    return dados_ui, dados_calculados


def _empty_data_message(
    interpretacao: dict[str, Any],
    *,
    municipio: str | None,
    uf: str | None,
    ano: int | None,
    notes: list[str],
) -> str:
    parts = []
    if municipio:
        parts.append(f"município {municipio}" + (f"/{uf}" if uf else ""))
    elif uf:
        parts.append(f"UF {uf}")
    local = ", ".join(parts) if parts else "local informado"
    ano_txt = f" no ano {ano}" if ano else ""
    msg = f"Não encontrei registros de dengue para {local}{ano_txt} na base carregada."
    if notes:
        return "\n\n".join(notes + [msg])
    return msg


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


def _df_to_records(df: Any) -> list[dict[str, Any]]:
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df.head(MAX_REGISTROS_LLM).to_dict(orient="records")
    return []


def _compact_panorama_para_llm(dados_calculados: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "tipo_consulta": "panorama",
        "local_resolvida": dados_calculados.get("local_resolvida"),
    }
    if dados_calculados.get("ano_assumido") is not None:
        out["ano_assumido"] = dados_calculados["ano_assumido"]
    if dados_calculados.get("filtros"):
        out["filtros"] = dados_calculados["filtros"]

    for key in ("indicador", "comparacao_estadual", "indicador_uf", "ranking_uf"):
        block = dados_calculados.get(key)
        if isinstance(block, pd.DataFrame):
            out[key] = _df_to_records(block)

    tendencia = dados_calculados.get("tendencia")
    if isinstance(tendencia, dict):
        serie = tendencia.get("serie")
        out["tendencia"] = {
            "serie": _df_to_records(serie),
            "resumo": tendencia.get("resumo") or {},
        }

    if dados_calculados.get("erro_query"):
        out["erro_query"] = dados_calculados["erro_query"]
    return out


def _should_show_comparacao_chart(
    interpretacao: dict[str, Any],
    pergunta: str,
    dados_ui: pd.DataFrame,
    intent: str,
) -> bool:
    if intent != "comparacao":
        return False
    if not charts.comparacao_chartable(dados_ui):
        return bool(interpretacao.get("gerar_grafico") or charts.wants_comparative_chart(pergunta, interpretacao))
    return True


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

    intent = str(interpretacao.get("intencao") or "desconhecida")
    notes: list[str] = list(interpretacao.get("observacoes") or [])

    if intent == "comparacao" and interpretacao.get("entidades"):
        entidades, ent_notes, ambiguity = _resolve_entidades(interpretacao)
        if ambiguity:
            logger.info("[CHATBOT] ambiguidade de entidade: %s", ambiguity)
            interpretacao["observacoes"] = ent_notes
            return _empty_result(ambiguity, interpretacao=interpretacao, filtros_ui=filtros_ui)
        interpretacao["entidades"] = entidades
        notes.extend(ent_notes)
        interpretacao["municipios"] = [
            str(item["nome"])
            for item in entidades
            if item.get("tipo") == "municipio" and item.get("nome")
        ]
        interpretacao["ufs"] = _dedupe_strs(
            [str(item["uf"]) for item in entidades if item.get("tipo") == "uf" and item.get("uf")]
            + [str(item["uf"]) for item in entidades if item.get("tipo") == "municipio" and item.get("uf")]
        )
    else:
        resolved_names, resolved_ufs, mun_notes, ambiguity = _resolve_municipios(interpretacao)
        if ambiguity:
            logger.info("[CHATBOT] ambiguidade de município: %s", ambiguity)
            interpretacao["municipios"] = list(interpretacao.get("municipios") or [])
            interpretacao["observacoes"] = mun_notes
            return _empty_result(ambiguity, interpretacao=interpretacao, filtros_ui=filtros_ui)

        if resolved_names:
            interpretacao["municipios"] = resolved_names
        if resolved_ufs:
            interpretacao["ufs"] = _dedupe_strs(resolved_ufs + list(interpretacao.get("ufs") or []))
        notes.extend(mun_notes)

    interpretacao, ano_assumido = _apply_default_year(interpretacao)
    interpretacao = nlu.infer_intent_when_vague(interpretacao, pergunta)
    if notes:
        interpretacao["observacoes"] = _dedupe_strs(notes)

    filtro_msg = _need_more_filters_message(interpretacao)
    if filtro_msg:
        logger.info("[CHATBOT] consulta ampla/insuficiente: %s", filtro_msg)
        return _empty_result(filtro_msg, interpretacao=interpretacao, filtros_ui=filtros_ui)

    intent = str(interpretacao.get("intencao") or "desconhecida")
    municipios = list(interpretacao.get("municipios") or [])
    ufs = list(interpretacao.get("ufs") or [])
    entidades = list(interpretacao.get("entidades") or [])
    municipio = municipios[0] if municipios else None
    uf = _comparison_scope_uf(interpretacao)
    ano = interpretacao.get("ano")
    semana = interpretacao.get("semana")
    metrica = str(interpretacao.get("metrica") or "casos")
    limite = int(interpretacao.get("limite") or 10)

    if intent == "desconhecida":
        return _empty_result(
            _unknown_intent_message(interpretacao),
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
            "entidades": entidades,
            "ano": ano,
            "semana": semana,
            "metrica": metrica,
        },
    }
    if ano_assumido is not None:
        dados_calculados["ano_assumido"] = ano_assumido

    try:
        if intent in {"incidencia", "total"}:
            df, fetch_note = _fetch_indicador_resilient(municipio=municipio, uf=uf, ano=ano)
            if fetch_note:
                notes.append(fetch_note)
            dados_ui = df
        elif intent == "ranking":
            df = queries.get_ranking(uf=uf, ano=ano, metrica=metrica, limit=limite)
            dados_ui = df
        elif intent == "comparacao" and len(entidades) >= 2:
            df = queries.comparar_entidades(entidades=entidades, ano=ano, metrica=metrica)
            if isinstance(df, pd.DataFrame) and df.empty:
                mun_hints = _municipios_uf_hints(entidades)
                df = queries.comparar_municipios(
                    municipios=[str(item["nome"]) for item in entidades if item.get("nome")],
                    uf=uf,
                    ano=ano,
                    metrica=metrica,
                    municipios_uf=mun_hints,
                )
            dados_ui = charts.prepare_comparacao_df(df)
            dados_calculados["entidades_comparadas"] = entidades
        elif intent == "comparacao":
            df = queries.comparar_municipios(municipios=municipios, uf=uf, ano=ano, metrica=metrica)
            dados_ui = charts.prepare_comparacao_df(df)
        elif intent == "tendencia":
            trend = queries.get_tendencia_municipio(municipio=municipio, uf=uf, ano=ano, ultimas_n=6)
            serie = trend.get("serie", pd.DataFrame())
            if isinstance(serie, pd.DataFrame) and serie.empty and municipio:
                resolved = _resolve_single_municipio(str(municipio), uf_hint=uf)
                if resolved.get("nome"):
                    trend = queries.get_tendencia_municipio(
                        municipio=resolved["nome"],
                        uf=_normalize_uf(resolved.get("uf")) or uf,
                        ano=ano,
                        ultimas_n=6,
                    )
                    if resolved.get("nota"):
                        notes.append(str(resolved["nota"]))
            dados_ui = trend.get("serie", pd.DataFrame())
            dados_calculados["resumo"] = trend.get("resumo", {})
        elif intent == "media_estado":
            df = queries.comparar_municipio_com_estado(municipio=municipio, uf=uf, ano=ano)
            if isinstance(df, pd.DataFrame) and df.empty and municipio:
                resolved = _resolve_single_municipio(str(municipio), uf_hint=uf)
                res_uf = _normalize_uf(resolved.get("uf")) or uf
                if resolved.get("nome") and res_uf:
                    df = queries.comparar_municipio_com_estado(
                        municipio=resolved["nome"], uf=res_uf, ano=ano
                    )
                    if resolved.get("nota"):
                        notes.append(str(resolved["nota"]))
            dados_ui = df
        elif intent == "pico":
            df = queries.get_pico_epidemiologico(municipio=municipio, uf=uf, ano=ano)
            if isinstance(df, pd.DataFrame) and df.empty and municipio:
                resolved = _resolve_single_municipio(str(municipio), uf_hint=uf)
                if resolved.get("nome"):
                    df = queries.get_pico_epidemiologico(
                        municipio=resolved["nome"],
                        uf=_normalize_uf(resolved.get("uf")) or uf,
                        ano=ano,
                    )
                    if resolved.get("nota"):
                        notes.append(str(resolved["nota"]))
            dados_ui = df
        elif intent == "panorama":
            dados_ui, dados_calculados = _execute_panorama(
                municipio=municipio,
                uf=uf,
                ano=ano,
            )
            dados_calculados["filtros"] = {
                "municipios": municipios,
                "ufs": ufs,
                "ano": ano,
                "metrica": metrica,
            }
            if ano_assumido is not None:
                dados_calculados["ano_assumido"] = ano_assumido
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

    panorama_sem_dados = intent == "panorama" and not _panorama_has_data(dados_calculados)
    if (
        (isinstance(dados_ui, pd.DataFrame) and dados_ui.empty and not dados_calculados.get("erro_query"))
        or panorama_sem_dados
    ):
        resposta = _empty_data_message(
            interpretacao,
            municipio=municipio,
            uf=uf,
            ano=ano,
            notes=notes,
        )
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

    if intent == "panorama":
        dados_para_llm = _compact_panorama_para_llm(dados_calculados)
    else:
        dados_para_llm = _compact_dados_para_llm(
            dados_calculados, intent, dados_ui if isinstance(dados_ui, pd.DataFrame) else None
        )
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

    gerar_grafico = _should_show_comparacao_chart(
        interpretacao, pergunta, dados_ui if isinstance(dados_ui, pd.DataFrame) else pd.DataFrame(), intent
    )

    return {
        "resposta": resposta,
        "dados": dados_ui,
        "contexto": contexto,
        "filtros": filtros_rag,
        "tipo_consulta": intent,
        "gerar_grafico": gerar_grafico,
        "metrica_grafico": metrica,
        "interpretacao": {k: v for k, v in interpretacao.items() if not str(k).startswith("_")},
    }


def reply(user_message: str) -> str:
    out = answer_question(user_message, filtros_ui=None)
    return str(out.get("resposta", ""))
