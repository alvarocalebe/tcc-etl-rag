"""Recuperação simples de contexto (cartas_de_fato) no PostgreSQL (MVP)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import text

from app.db import get_engine

# Nunca retornar ou solicitar mais que este número de cartas (evita carga excessiva).
MAX_CARTAS_RAG = 5


def _normalize_text(s: Any) -> str:
    return str(s or "").strip()


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

        # Busca textual simples
        params["q"] = f"%{pergunta_norm}%"
        conditions.append("cf.texto ILIKE :q")

        # Filtros opcionais
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

        where = " AND ".join(conditions)

        # Ordenação simples: prioriza correspondência no texto (mais "curta" fica mais alta)
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
            ORDER BY cf.texto ASC
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
