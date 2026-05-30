"""Consultas analíticas reutilizáveis (SQL parametrizado + pandas)."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import pandas as pd
from sqlalchemy import text

from app.db import get_engine

DEFAULT_LIMIT = 10
MAX_RESULT_LIMIT = 20
MAX_SERIE_LIMIT = 104


def count_municipio_sql():
    return text("SELECT COUNT(*) AS n FROM municipio")


def count_fato_dengue_sql():
    return text("SELECT COUNT(*) AS n FROM fato_dengue")


def count_log_consulta_sql():
    return text("SELECT COUNT(*) AS n FROM log_consulta")


def _read_df(query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Executa query parametrizada e retorna DataFrame (ou DataFrame vazio)."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params or {})
    except Exception:
        return pd.DataFrame()


def _safe_limit(limit: Any, *, default: int = DEFAULT_LIMIT, maximum: int = MAX_RESULT_LIMIT) -> int:
    try:
        value = int(limit)
    except Exception:
        value = default
    return max(1, min(value, maximum))


def _municipio_condition(
    alias: str,
    municipio: Any,
    params: dict[str, Any],
    *,
    param_prefix: str = "municipio",
) -> tuple[str, dict[str, Any]]:
    """
    Retorna (condition_sql, params_update) para filtrar município.

    - Se `municipio` for int (ou string numérica) -> id_municipio_ibge
    - Se `municipio` for texto -> nome_municipio ILIKE %texto%
    """
    if municipio is None:
        return "", params

    if isinstance(municipio, int) and not isinstance(municipio, bool):
        params = {**params, f"{param_prefix}_id": int(municipio)}
        return f"{alias}.id_municipio_ibge = :{param_prefix}_id", params

    if isinstance(municipio, str) and municipio.isdigit():
        params = {**params, f"{param_prefix}_id": int(municipio)}
        return f"{alias}.id_municipio_ibge = :{param_prefix}_id", params

    nome = str(municipio).strip()
    if not nome:
        return "", params
    params = {**params, f"{param_prefix}_nome": f"%{nome}%"}
    return f"{alias}.nome_municipio ILIKE :{param_prefix}_nome", params


def _uf_condition(
    alias: str, uf: Any, params: dict[str, Any], *, param_name: str = "uf_sigla"
) -> tuple[str, dict[str, Any]]:
    if uf is None:
        return "", params
    sigla = str(uf).strip().upper()[:2]
    if len(sigla) != 2 or sigla.startswith("("):
        return "", params
    params = {**params, param_name: sigla}
    return f"{alias}.uf_sigla = :{param_name}", params


def _municipios_exact_condition(
    alias: str,
    municipios: list[str],
    params: dict[str, Any],
    *,
    param_prefix: str = "mun_lista",
    ufs: list[str | None] | None = None,
) -> tuple[str, dict[str, Any]]:
    clean = [str(m).strip() for m in municipios if str(m).strip()]
    if not clean:
        return "", params
    hints = list(ufs or [])
    if hints and len(hints) == len(clean):
        clauses: list[str] = []
        out = dict(params)
        for idx, nome in enumerate(clean):
            name_key = f"{param_prefix}_nome_{idx}"
            out[name_key] = nome
            uf_hint = str(hints[idx] or "").strip().upper()[:2]
            if len(uf_hint) == 2 and uf_hint.isalpha():
                uf_key = f"{param_prefix}_uf_{idx}"
                out[uf_key] = uf_hint
                clauses.append(
                    f"(LOWER(TRIM({alias}.nome_municipio)) = LOWER(TRIM(:{name_key})) "
                    f"AND {alias}.uf_sigla = :{uf_key})"
                )
            else:
                clauses.append(
                    f"LOWER(TRIM({alias}.nome_municipio)) = LOWER(TRIM(:{name_key}))"
                )
        return "(" + " OR ".join(clauses) + ")", out

    clauses: list[str] = []
    out = dict(params)
    for idx, nome in enumerate(clean):
        key = f"{param_prefix}_{idx}"
        out[key] = nome
        clauses.append(f"LOWER(TRIM({alias}.nome_municipio)) = LOWER(TRIM(:{key}))")
    return "(" + " OR ".join(clauses) + ")", out


def get_indicador_municipio(
    municipio: Any = None,
    uf: Any = None,
    ano: int | None = None,
) -> pd.DataFrame:
    """
    Retorna indicador agregado por município, UF ou país (uma linha por escopo/ano).

    Colunas principais:
    - id_municipio_ibge
    - nome_municipio
    - uf_sigla
    - ano
    - total_casos
    - populacao
    - incidencia_100k
    """
    params: dict[str, Any] = {"limit": MAX_RESULT_LIMIT}
    conditions: list[str] = []

    cond_m, params = _municipio_condition("m", municipio, params)
    if cond_m:
        conditions.append(cond_m)

    cond_uf, params = _uf_condition("m", uf, params)
    if cond_uf:
        conditions.append(cond_uf)

    if ano is not None:
        params["ano"] = int(ano)
        conditions.append("t.ano = :ano")

    if not conditions:
        return pd.DataFrame()

    where = "WHERE " + " AND ".join(conditions)

    if municipio is not None:
        query = f"""
            WITH base AS (
                SELECT
                    m.id_municipio_ibge,
                    m.nome_municipio,
                    m.uf_sigla,
                    t.ano,
                    SUM(fi.casos) AS total_casos,
                    MAX(fi.populacao) AS populacao
                FROM fato_indicador fi
                JOIN tempo t ON t.id_tempo = fi.id_tempo
                JOIN municipio m ON m.id_municipio_ibge = fi.id_municipio_ibge
                {where}
                GROUP BY m.id_municipio_ibge, m.nome_municipio, m.uf_sigla, t.ano
            )
            SELECT
                id_municipio_ibge,
                nome_municipio,
                uf_sigla,
                ano,
                total_casos,
                populacao,
                ROUND(
                    total_casos::numeric
                    / NULLIF(populacao, 0)::numeric
                    * 100000,
                    2
                ) AS incidencia_100k
            FROM base
            ORDER BY ano DESC, uf_sigla, nome_municipio
            LIMIT :limit
        """
    elif uf is not None:
        query = f"""
            WITH base AS (
                SELECT
                    m.id_municipio_ibge,
                    m.uf_sigla,
                    t.ano,
                    SUM(fi.casos) AS total_casos,
                    MAX(fi.populacao) AS populacao
                FROM fato_indicador fi
                JOIN tempo t ON t.id_tempo = fi.id_tempo
                JOIN municipio m ON m.id_municipio_ibge = fi.id_municipio_ibge
                {where}
                GROUP BY m.id_municipio_ibge, m.uf_sigla, t.ano
            )
            SELECT
                NULL::BIGINT AS id_municipio_ibge,
                NULL::VARCHAR AS nome_municipio,
                uf_sigla,
                ano,
                SUM(total_casos) AS total_casos,
                SUM(populacao) AS populacao,
                ROUND(
                    SUM(total_casos)::numeric
                    / NULLIF(SUM(populacao), 0)::numeric
                    * 100000,
                    2
                ) AS incidencia_100k
            FROM base
            GROUP BY uf_sigla, ano
            ORDER BY ano DESC, uf_sigla
            LIMIT :limit
        """
    else:
        query = f"""
            WITH base AS (
                SELECT
                    m.id_municipio_ibge,
                    t.ano,
                    SUM(fi.casos) AS total_casos,
                    MAX(fi.populacao) AS populacao
                FROM fato_indicador fi
                JOIN tempo t ON t.id_tempo = fi.id_tempo
                JOIN municipio m ON m.id_municipio_ibge = fi.id_municipio_ibge
                {where}
                GROUP BY m.id_municipio_ibge, t.ano
            )
            SELECT
                NULL::BIGINT AS id_municipio_ibge,
                NULL::VARCHAR AS nome_municipio,
                NULL::VARCHAR AS uf_sigla,
                ano,
                SUM(total_casos) AS total_casos,
                SUM(populacao) AS populacao,
                ROUND(
                    SUM(total_casos)::numeric
                    / NULLIF(SUM(populacao), 0)::numeric
                    * 100000,
                    2
                ) AS incidencia_100k
            FROM base
            GROUP BY ano
            ORDER BY ano DESC
            LIMIT :limit
        """
    return _read_df(query, params)


def get_ranking(
    uf: Any = None,
    ano: int | None = None,
    metrica: str = "incidencia",
    limit: int = 10,
) -> pd.DataFrame:
    """
    Ranking municipal por incidência ou casos.
    """
    if uf is None and ano is None:
        return pd.DataFrame()

    metric = str(metrica or "incidencia").strip().lower()
    if metric not in {"incidencia", "casos"}:
        metric = "incidencia"

    params: dict[str, Any] = {"limit": _safe_limit(limit, default=10, maximum=10)}
    conditions: list[str] = []

    cond_uf, params = _uf_condition("m", uf, params)
    if cond_uf:
        conditions.append(cond_uf)

    if ano is not None:
        params["ano"] = int(ano)
        conditions.append("t.ano = :ano")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    order_metric = "incidencia_100k DESC NULLS LAST, total_casos DESC" if metric == "incidencia" else "total_casos DESC, incidencia_100k DESC NULLS LAST"

    query = f"""
        WITH base AS (
            SELECT
                m.id_municipio_ibge,
                m.nome_municipio,
                m.uf_sigla,
                t.ano,
                SUM(fi.casos) AS total_casos,
                MAX(fi.populacao) AS populacao,
                ROUND(
                    SUM(fi.casos)::numeric
                    / NULLIF(MAX(fi.populacao), 0)::numeric
                    * 100000,
                    2
                ) AS incidencia_100k
            FROM fato_indicador fi
            JOIN tempo t ON t.id_tempo = fi.id_tempo
            JOIN municipio m ON m.id_municipio_ibge = fi.id_municipio_ibge
            {where}
            GROUP BY m.id_municipio_ibge, m.nome_municipio, m.uf_sigla, t.ano
        )
        SELECT
            DENSE_RANK() OVER (ORDER BY {order_metric}) AS rank,
            id_municipio_ibge,
            nome_municipio,
            uf_sigla,
            ano,
            total_casos,
            populacao,
            incidencia_100k
        FROM base
        ORDER BY {order_metric}
        LIMIT :limit
    """
    return _read_df(query, params)


def comparar_municipios(
    municipios: list[str],
    uf: Any = None,
    ano: int | None = None,
    metrica: str = "incidencia",
    *,
    municipios_uf: list[str | None] | None = None,
) -> pd.DataFrame:
    """
    Retorna uma linha por município com casos, população e incidência.

    municipios_uf: UF opcional por município (mesmo tamanho que municipios).
    """
    del metrica  # A saída sempre traz casos e incidência; a interpretação usa ambas.

    params: dict[str, Any] = {"limit": MAX_RESULT_LIMIT}
    conditions: list[str] = []

    hints = list(municipios_uf or [])
    per_uf = hints and len(hints) == len(municipios) and len({h for h in hints if h}) > 1
    cond_list, params = _municipios_exact_condition(
        "m",
        municipios,
        params,
        ufs=hints if hints and len(hints) == len(municipios) else None,
    )
    if cond_list:
        conditions.append(cond_list)

    if not per_uf:
        cond_uf, params = _uf_condition("m", uf, params)
        if cond_uf:
            conditions.append(cond_uf)

    if ano is not None:
        params["ano"] = int(ano)
        conditions.append("t.ano = :ano")

    if not conditions:
        return pd.DataFrame()

    where = "WHERE " + " AND ".join(conditions)
    query = f"""
        SELECT
            m.id_municipio_ibge,
            m.nome_municipio,
            m.uf_sigla,
            t.ano,
            SUM(fi.casos) AS total_casos,
            MAX(fi.populacao) AS populacao,
            ROUND(
                SUM(fi.casos)::numeric
                / NULLIF(MAX(fi.populacao), 0)::numeric
                * 100000,
                2
            ) AS incidencia_100k
        FROM fato_indicador fi
        JOIN tempo t ON t.id_tempo = fi.id_tempo
        JOIN municipio m ON m.id_municipio_ibge = fi.id_municipio_ibge
        {where}
        GROUP BY m.id_municipio_ibge, m.nome_municipio, m.uf_sigla, t.ano
        ORDER BY m.uf_sigla, m.nome_municipio, t.ano
        LIMIT :limit
    """
    df = _read_df(query, params)
    if df.empty:
        return df
    labels = []
    for _, row in df.iterrows():
        nome = row.get("nome_municipio")
        uf_sigla = row.get("uf_sigla")
        if pd.notna(nome) and str(nome).strip():
            label = str(nome).strip()
            if pd.notna(uf_sigla) and str(uf_sigla).strip():
                label = f"{label}/{uf_sigla}"
        else:
            label = str(uf_sigla or "Município")
        labels.append(label)
    df = df.copy()
    df["entidade_label"] = labels
    df["tipo_entidade"] = "municipio"
    return df


def get_serie_semanal(
    municipio: Any = None, uf: Any = None, ano: int | None = None
) -> pd.DataFrame:
    """
    Série semanal (casos e incidência) filtrando por município/UF/ano.
    """
    params: dict[str, Any] = {"limit": MAX_SERIE_LIMIT}
    conditions: list[str] = []

    group_cols = ["t.ano", "t.semana_epidemiologica"]
    incidencia_expr = "ROUND(SUM(fi.casos)::numeric / NULLIF(SUM(fi.populacao), 0)::numeric * 100000, 2)"
    select_cols = [
        "t.ano",
        "t.semana_epidemiologica",
        "SUM(fi.casos) AS casos",
        f"{incidencia_expr} AS incidencia_100k",
    ]

    cond_m, params = _municipio_condition("m", municipio, params)
    if cond_m:
        conditions.append(cond_m)
        incidencia_expr = "ROUND(SUM(fi.casos)::numeric / NULLIF(MAX(fi.populacao), 0)::numeric * 100000, 2)"
        select_cols.insert(0, "m.nome_municipio")
        select_cols.insert(1, "m.uf_sigla")
        group_cols.insert(0, "m.nome_municipio")
        group_cols.insert(1, "m.uf_sigla")
    else:
        cond_uf, params = _uf_condition("m", uf, params)
        if cond_uf:
            conditions.append(cond_uf)
            select_cols.insert(0, "m.uf_sigla")
            group_cols.insert(0, "m.uf_sigla")

    if ano is not None:
        params["ano"] = int(ano)
        conditions.append("t.ano = :ano")

    if not conditions:
        return pd.DataFrame()

    where = "WHERE " + " AND ".join(conditions)
    query = f"""
        SELECT
            {', '.join(select_cols[:-1])},
            {incidencia_expr} AS incidencia_100k
        FROM fato_indicador fi
        JOIN tempo t ON t.id_tempo = fi.id_tempo
        JOIN municipio m ON m.id_municipio_ibge = fi.id_municipio_ibge
        {where}
        GROUP BY {', '.join(group_cols)}
        ORDER BY t.ano, t.semana_epidemiologica
        LIMIT :limit
    """
    return _read_df(query, params)


def get_tendencia_municipio(
    municipio: Any,
    uf: Any = None,
    ano: int | None = None,
    ultimas_n: int = 6,
) -> dict[str, Any]:
    """
    Retorna série semanal e resumo de tendência.
    """
    serie = get_serie_semanal(municipio=municipio, uf=uf, ano=ano)
    resumo: dict[str, Any] = {
        "ultimas_n": max(1, int(ultimas_n)),
        "casos_ultimas_n": None,
        "casos_n_anteriores": None,
        "variacao_percentual": None,
        "classificacao": "sem dados",
    }

    if serie.empty or "casos" not in serie.columns:
        return {"serie": serie, "resumo": resumo}

    n = max(1, int(ultimas_n))
    serie = serie.sort_values(["ano", "semana_epidemiologica"]).reset_index(drop=True)
    ultimas = serie.tail(n)
    anteriores = serie.iloc[max(0, len(serie) - 2 * n) : max(0, len(serie) - n)]

    casos_ultimas = int(pd.to_numeric(ultimas["casos"], errors="coerce").fillna(0).sum())
    resumo["casos_ultimas_n"] = casos_ultimas

    if not anteriores.empty:
        casos_anteriores = int(pd.to_numeric(anteriores["casos"], errors="coerce").fillna(0).sum())
        resumo["casos_n_anteriores"] = casos_anteriores
        if casos_anteriores > 0:
            variacao = round((casos_ultimas - casos_anteriores) / casos_anteriores * 100, 2)
            resumo["variacao_percentual"] = variacao
            if variacao > 10:
                resumo["classificacao"] = "aumento"
            elif variacao < -10:
                resumo["classificacao"] = "queda"
            else:
                resumo["classificacao"] = "estável"
        elif casos_ultimas > 0:
            resumo["classificacao"] = "aumento"
    return {"serie": serie, "resumo": resumo}


def comparar_municipio_com_estado(
    municipio: Any,
    uf: Any,
    ano: int | None,
) -> pd.DataFrame:
    """
    Compara a incidência do município com a média da UF no ano.
    """
    if municipio is None or uf is None or ano is None:
        return pd.DataFrame()

    params: dict[str, Any] = {"ano": int(ano)}
    conditions_m: list[str] = []
    conditions_uf: list[str] = ["t.ano = :ano"]

    cond_m, params = _municipio_condition("m", municipio, params)
    if cond_m:
        conditions_m.append(cond_m)
    cond_uf, params = _uf_condition("m", uf, params, param_name="uf_estado")
    if cond_uf:
        conditions_uf.append(cond_uf.replace(":uf_estado", ":uf_estado"))
        conditions_m.append(cond_uf.replace(":uf_estado", ":uf_estado"))

    if not conditions_m:
        return pd.DataFrame()

    where_m = "WHERE " + " AND ".join(conditions_m)
    where_uf = "WHERE " + " AND ".join(conditions_uf)
    query = f"""
        WITH municipio_base AS (
            SELECT
                m.id_municipio_ibge,
                m.nome_municipio,
                m.uf_sigla,
                t.ano,
                SUM(fi.casos) AS total_casos,
                MAX(fi.populacao) AS populacao,
                ROUND(
                    SUM(fi.casos)::numeric
                    / NULLIF(MAX(fi.populacao), 0)::numeric
                    * 100000,
                    2
                ) AS incidencia_municipio_100k
            FROM fato_indicador fi
            JOIN tempo t ON t.id_tempo = fi.id_tempo
            JOIN municipio m ON m.id_municipio_ibge = fi.id_municipio_ibge
            {where_m}
            GROUP BY m.id_municipio_ibge, m.nome_municipio, m.uf_sigla, t.ano
        ),
        estado_base AS (
            SELECT
                m.uf_sigla,
                t.ano,
                m.id_municipio_ibge,
                ROUND(
                    SUM(fi.casos)::numeric
                    / NULLIF(MAX(fi.populacao), 0)::numeric
                    * 100000,
                    2
                ) AS incidencia_100k
            FROM fato_indicador fi
            JOIN tempo t ON t.id_tempo = fi.id_tempo
            JOIN municipio m ON m.id_municipio_ibge = fi.id_municipio_ibge
            {where_uf}
            GROUP BY m.uf_sigla, t.ano, m.id_municipio_ibge
        )
        SELECT
            mb.id_municipio_ibge,
            mb.nome_municipio,
            mb.uf_sigla,
            mb.ano,
            mb.total_casos,
            mb.populacao,
            mb.incidencia_municipio_100k,
            ROUND(AVG(eb.incidencia_100k), 2) AS incidencia_media_estado_100k,
            ROUND(
                (
                    mb.incidencia_municipio_100k - AVG(eb.incidencia_100k)
                ) / NULLIF(AVG(eb.incidencia_100k), 0) * 100,
                2
            ) AS diferenca_percentual,
            CASE
                WHEN mb.incidencia_municipio_100k > AVG(eb.incidencia_100k) THEN 'acima'
                WHEN mb.incidencia_municipio_100k < AVG(eb.incidencia_100k) THEN 'abaixo'
                ELSE 'igual'
            END AS posicao_relativa
        FROM municipio_base mb
        JOIN estado_base eb
          ON eb.uf_sigla = mb.uf_sigla
         AND eb.ano = mb.ano
        GROUP BY
            mb.id_municipio_ibge,
            mb.nome_municipio,
            mb.uf_sigla,
            mb.ano,
            mb.total_casos,
            mb.populacao,
            mb.incidencia_municipio_100k
        LIMIT 1
    """
    return _read_df(query, params)


def get_pico_epidemiologico(
    municipio: Any,
    uf: Any = None,
    ano: int | None = None,
) -> pd.DataFrame:
    """
    Retorna a semana com maior número de casos para o município/UF/ano.
    """
    params: dict[str, Any] = {}
    conditions: list[str] = []

    cond_m, params = _municipio_condition("m", municipio, params)
    if cond_m:
        conditions.append(cond_m)

    cond_uf, params = _uf_condition("m", uf, params)
    if cond_uf:
        conditions.append(cond_uf)

    if ano is not None:
        params["ano"] = int(ano)
        conditions.append("t.ano = :ano")

    if not conditions:
        return pd.DataFrame()

    where = "WHERE " + " AND ".join(conditions)
    query = f"""
        SELECT
            m.id_municipio_ibge,
            m.nome_municipio,
            m.uf_sigla,
            t.ano,
            t.semana_epidemiologica,
            SUM(fi.casos) AS casos,
            MAX(fi.populacao) AS populacao,
            ROUND(
                SUM(fi.casos)::numeric
                / NULLIF(MAX(fi.populacao), 0)::numeric
                * 100000,
                2
            ) AS incidencia_100k
        FROM fato_indicador fi
        JOIN tempo t ON t.id_tempo = fi.id_tempo
        JOIN municipio m ON m.id_municipio_ibge = fi.id_municipio_ibge
        {where}
        GROUP BY m.id_municipio_ibge, m.nome_municipio, m.uf_sigla, t.ano, t.semana_epidemiologica
        ORDER BY casos DESC, incidencia_100k DESC NULLS LAST, t.semana_epidemiologica
        LIMIT 1
    """
    return _read_df(query, params)


# -----------------------------------------------------------------------------
# Wrappers legados / compatibilidade
# -----------------------------------------------------------------------------
def get_total_casos(municipio: Any = None, uf: Any = None, ano: int | None = None) -> pd.DataFrame:
    df = get_indicador_municipio(municipio=municipio, uf=uf, ano=ano)
    if df.empty or "total_casos" not in df.columns:
        return pd.DataFrame(columns=["total_casos"])
    total = int(pd.to_numeric(df["total_casos"], errors="coerce").fillna(0).sum())
    return pd.DataFrame([{"total_casos": total}])


def get_incidencia_municipio(
    municipio: Any = None,
    uf: Any = None,
    ano: int | None = None,
    semana: int | None = None,
) -> pd.DataFrame:
    if semana is not None:
        serie = get_serie_semanal(municipio=municipio, uf=uf, ano=ano)
        if serie.empty:
            return serie
        serie = serie[serie["semana_epidemiologica"] == int(semana)].copy()
        if serie.empty:
            return serie
        serie = serie.rename(columns={"casos": "total_casos"})
        return serie
    return get_indicador_municipio(municipio=municipio, uf=uf, ano=ano)


def get_ranking_municipios(uf: Any = None, ano: int | None = None, limit: int = 10) -> pd.DataFrame:
    return get_ranking(uf=uf, ano=ano, metrica="casos", limit=limit)


def comparar_anos(
    municipio: Any = None,
    uf: Any = None,
    ano1: int | None = None,
    ano2: int | None = None,
) -> pd.DataFrame:
    anos = [int(a) for a in [ano1, ano2] if a is not None]
    if not anos:
        return pd.DataFrame()

    params: dict[str, Any] = {"anos": anos, "limit": MAX_RESULT_LIMIT}
    conditions: list[str] = ["t.ano = ANY(:anos)"]

    cond_m, params = _municipio_condition("m", municipio, params)
    if cond_m:
        conditions.append(cond_m)
    elif uf is not None:
        cond_uf, params = _uf_condition("m", uf, params)
        if cond_uf:
            conditions.append(cond_uf)

    where = "WHERE " + " AND ".join(conditions)
    query = f"""
        SELECT
            t.ano,
            SUM(fi.casos) AS casos_total,
            MAX(fi.populacao) AS populacao_total,
            ROUND(
                SUM(fi.casos)::numeric
                / NULLIF(MAX(fi.populacao), 0)::numeric
                * 100000,
                2
            ) AS incidencia_100k
        FROM fato_indicador fi
        JOIN tempo t ON t.id_tempo = fi.id_tempo
        JOIN municipio m ON m.id_municipio_ibge = fi.id_municipio_ibge
        {where}
        GROUP BY t.ano
        ORDER BY t.ano
        LIMIT :limit
    """
    return _read_df(query, params)


def buscar_municipios(nome: str, uf: Any = None) -> pd.DataFrame:
    """
    Busca municípios por nome: prioriza nome exato (case-insensitive), depois ILIKE.
    Opcionalmente restringe por UF. No máximo 10 linhas.
    Retorna: id_municipio_ibge, nome_municipio, uf_sigla.
    """
    nome_base = str(nome or "").strip()
    if not nome_base:
        return pd.DataFrame()

    params: dict[str, Any] = {
        "nome_like": f"%{nome_base}%",
        "nome_exato": nome_base,
        "prioriza_palmas_to": 1 if nome_base.lower() == "palmas" else 0,
    }
    uf_clause = ""
    if uf is not None:
        u = str(uf).strip().upper()[:2]
        if len(u) == 2 and not u.startswith("("):
            params["uf_filtro"] = u
            uf_clause = " AND m.uf_sigla = :uf_filtro"

    query = f"""
        SELECT
            m.id_municipio_ibge,
            m.nome_municipio,
            m.uf_sigla
        FROM municipio m
        WHERE m.nome_municipio ILIKE :nome_like
        {uf_clause}
        ORDER BY
            (LOWER(TRIM(m.nome_municipio)) = LOWER(TRIM(:nome_exato))) DESC,
            CASE
                WHEN :prioriza_palmas_to = 1
                    AND m.uf_sigla = 'TO'
                    AND LOWER(TRIM(m.nome_municipio)) = 'palmas'
                THEN 0
                ELSE 1
            END,
            LENGTH(TRIM(m.nome_municipio)),
            m.uf_sigla,
            m.nome_municipio
        LIMIT 10
    """
    df = _read_df(query, params)
    if not df.empty:
        return df

    return _buscar_municipios_fuzzy(nome_base, uf=uf)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _buscar_municipios_fuzzy(nome: str, uf: Any = None) -> pd.DataFrame:
    nome_base = str(nome or "").strip()
    if len(nome_base) < 3:
        return pd.DataFrame()

    prefix = nome_base[: max(3, min(len(nome_base), 5))]
    params: dict[str, Any] = {"nome_prefix": f"{prefix}%"}
    uf_clause = ""
    if uf is not None:
        u = str(uf).strip().upper()[:2]
        if len(u) == 2 and not u.startswith("("):
            params["uf_filtro"] = u
            uf_clause = " AND m.uf_sigla = :uf_filtro"

    query = f"""
        SELECT
            m.id_municipio_ibge,
            m.nome_municipio,
            m.uf_sigla
        FROM municipio m
        WHERE m.nome_municipio ILIKE :nome_prefix
        {uf_clause}
        ORDER BY m.nome_municipio
        LIMIT 200
    """
    candidates = _read_df(query, params)
    if candidates.empty:
        return pd.DataFrame()

    target = nome_base.lower()
    scored = candidates.copy()
    scored["_score"] = scored["nome_municipio"].astype(str).map(
        lambda value: _similarity(target, value.lower())
    )
    scored = scored[scored["_score"] >= 0.82].sort_values("_score", ascending=False)
    if scored.empty:
        return pd.DataFrame()
    return scored.drop(columns=["_score"]).head(10)


def get_ano_mais_recente() -> int | None:
    df = listar_anos()
    if df.empty or "ano" not in df.columns:
        return None
    try:
        return int(df["ano"].iloc[0])
    except Exception:
        return None


def comparar_entidades(
    entidades: list[dict[str, Any]],
    ano: int | None = None,
    metrica: str = "incidencia",
) -> pd.DataFrame:
    """
    Compara indicadores de UFs e/ou municípios em um DataFrame unificado.
    """
    del metrica  # Saída sempre inclui casos e incidência.

    rows: list[dict[str, Any]] = []
    for entidade in entidades or []:
        tipo = str(entidade.get("tipo") or "").strip().lower()
        label = str(entidade.get("label") or "").strip()
        if tipo == "uf":
            uf = str(entidade.get("uf") or "").strip().upper()[:2]
            if len(uf) != 2:
                continue
            df = get_indicador_municipio(uf=uf, ano=ano)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            row["entidade_label"] = label or uf
            row["tipo_entidade"] = "uf"
            rows.append(row)
        elif tipo == "municipio":
            nome = str(entidade.get("nome") or "").strip()
            if not nome:
                continue
            uf_hint = entidade.get("uf")
            df = get_indicador_municipio(municipio=nome, uf=uf_hint, ano=ano)
            if df.empty and uf_hint:
                df = get_indicador_municipio(municipio=nome, uf=None, ano=ano)
            if df.empty:
                continue
            if uf_hint and "uf_sigla" in df.columns:
                uf_norm = str(uf_hint).strip().upper()[:2]
                scoped = df[df["uf_sigla"].astype(str).str.upper() == uf_norm]
                if not scoped.empty:
                    df = scoped
            row = df.iloc[0].to_dict()
            municipio_nome = row.get("nome_municipio") or nome
            uf_sigla = row.get("uf_sigla") or uf_hint
            row["entidade_label"] = label or (
                f"{municipio_nome}/{uf_sigla}" if uf_sigla else str(municipio_nome)
            )
            row["tipo_entidade"] = "municipio"
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    if len(result) == 2 and "incidencia_100k" in result.columns:
        first = pd.to_numeric(result.iloc[0]["incidencia_100k"], errors="coerce")
        second = pd.to_numeric(result.iloc[1]["incidencia_100k"], errors="coerce")
        if pd.notna(first) and pd.notna(second) and float(second) != 0.0:
            diff = round((float(first) - float(second)) / float(second) * 100, 2)
            result["diferenca_percentual_vs_segunda"] = [diff, -diff]

    return result


def listar_ufs() -> pd.DataFrame:
    query = """
        SELECT DISTINCT uf_sigla
        FROM municipio
        WHERE uf_sigla IS NOT NULL AND TRIM(uf_sigla) <> ''
        ORDER BY uf_sigla
        LIMIT 100
    """
    return _read_df(query, {})


def listar_anos() -> pd.DataFrame:
    query = """
        SELECT DISTINCT ano
        FROM tempo
        WHERE ano IS NOT NULL
        ORDER BY ano DESC
        LIMIT 100
    """
    return _read_df(query, {})


def listar_municipios_por_uf(uf: Any) -> pd.DataFrame:
    params = {"uf_sigla": str(uf).strip().upper()[:2]}
    query = """
        SELECT id_municipio_ibge, nome_municipio, uf_sigla
        FROM municipio
        WHERE uf_sigla = :uf_sigla
        ORDER BY nome_municipio
        LIMIT 1000
    """
    return _read_df(query, params)
