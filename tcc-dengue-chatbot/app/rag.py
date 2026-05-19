"""Recuperação simples de contexto (cartas_de_fato) no PostgreSQL (MVP)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import text

from app.db import get_engine

# Nunca retornar ou solicitar mais que este número de cartas (evita carga excessiva).
MAX_CARTAS_RAG = 5

# Palavras muito comuns na pergunta que não aparecem nas cartas de fato.
_STOPWORDS = frozenset(
    {
        "a",
        "ao",
        "as",
        "com",
        "da",
        "de",
        "do",
        "dos",
        "das",
        "e",
        "em",
        "foi",
        "foram",
        "na",
        "no",
        "nos",
        "nas",
        "o",
        "os",
        "ou",
        "para",
        "por",
        "qual",
        "quais",
        "que",
        "se",
        "sem",
        "um",
        "uma",
        "como",
        "mostre",
        "mostrar",
        "dengue",
        "incidencia",
        "incidência",
        "casos",
        "total",
        "evolucao",
        "evolução",
    }
)


def _normalize_text(s: Any) -> str:
    return str(s or "").strip()


def _terms_from_question(pergunta: str, *, max_terms: int = 4) -> list[str]:
    """Extrai termos buscáveis da pergunta (não usa a frase inteira)."""
    import re
    import unicodedata

    s = unicodedata.normalize("NFD", pergunta.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    raw = re.findall(r"[a-z0-9]{3,}", s)
    out: list[str] = []
    for tok in raw:
        if tok in _STOPWORDS or tok.isdigit():
            continue
        if tok not in out:
            out.append(tok)
        if len(out) >= max_terms:
            break
    return out


def retrieve_context(
    pergunta: str, filtros: dict[str, Any] | None, top_k: int = 5
) -> list[dict]:
    """
    Recupera cartas_de_fato relevantes no banco, usando:
    - filtros opcionais (municipio, uf, ano, semana_epidemiologica)
    - busca por similaridade textual via ILIKE na coluna `texto`

    Retorna no máximo MAX_CARTAS_RAG registros (nunca carrega a tabela inteira: sempre LIMIT 5 no SQL).
    """
    try:
        _ = top_k  # compatibilidade de chamada; limite fixo no SQL

        pergunta_norm = _normalize_text(pergunta)
        if not pergunta_norm:
            return []

        filtros = filtros or {}
        lim = MAX_CARTAS_RAG

        conditions: list[str] = []
        params: dict[str, Any] = {}

        # Filtros estruturados (NLU / sidebar)
        municipio = filtros.get("municipio")
        if municipio:
            params["municipio"] = f"%{str(municipio).strip()}%"
            conditions.append("cf.municipio ILIKE :municipio")

        uf = filtros.get("uf")
        if uf:
            params["uf_sigla"] = str(uf).strip().upper()[:2]
            conditions.append("cf.uf_sigla = :uf_sigla")

        ano = filtros.get("ano")
        if ano is not None:
            params["ano"] = int(ano)
            conditions.append("cf.ano = :ano")

        semana = filtros.get("semana_epidemiologica")
        if semana is not None:
            params["semana"] = int(semana)
            conditions.append("cf.semana_epidemiologica = :semana")

        has_structural = bool(
            municipio or uf or ano is not None or semana is not None
        )

        # Busca textual: a pergunta inteira quase nunca aparece na carta ("Em Palmas/TO...").
        # Com filtros estruturados, basta município/UF/ano/semana; sem filtros, usa termos-chave.
        if not has_structural:
            terms = _terms_from_question(pergunta_norm)
            if terms:
                term_conds: list[str] = []
                for i, term in enumerate(terms):
                    key = f"term_{i}"
                    params[key] = f"%{term}%"
                    term_conds.append(f"cf.texto ILIKE :{key}")
                conditions.append("(" + " OR ".join(term_conds) + ")")
            else:
                params["q"] = f"%{pergunta_norm}%"
                conditions.append("cf.texto ILIKE :q")

        if not conditions:
            return []

        where = " AND ".join(conditions)
        order_by = "cf.casos DESC NULLS LAST, cf.semana_epidemiologica DESC"

        query = f"""
            SELECT
                cf.texto,
                cf.municipio,
                cf.uf_sigla,
                cf.ano,
                cf.semana_epidemiologica,
                fi.casos,
                fi.populacao,
                fi.incidencia_100k,
                fi.fonte_populacao AS fonte
            FROM carta_de_fato cf
            JOIN fato_indicador fi ON fi.id_indicador = cf.id_indicador
            WHERE {where}
            ORDER BY {order_by}
            LIMIT {lim}
        """

        engine = get_engine()
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)

        if df.empty:
            return []

        # Garantia extra: nunca mais que MAX_CARTAS_RAG linhas
        df = df.head(MAX_CARTAS_RAG)

        # Retorna tipos JSON-friendly
        out: list[dict] = []
        for row in df.to_dict(orient="records"):
            out.append(
                {
                    "texto": row.get("texto"),
                    "municipio": row.get("municipio"),
                    "uf_sigla": row.get("uf_sigla"),
                    "ano": row.get("ano"),
                    "semana_epidemiologica": row.get("semana_epidemiologica"),
                    "casos": row.get("casos"),
                    "populacao": row.get("populacao"),
                    "incidencia_100k": row.get("incidencia_100k"),
                    "fonte": row.get("fonte"),
                }
            )
        return out
    except Exception:
        # MVP: falhas devem retornar [] para não quebrar o chatbot
        return []
