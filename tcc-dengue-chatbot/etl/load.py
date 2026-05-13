"""
Carga das tabelas analíticas no PostgreSQL (upsert + cartas recriadas).
"""

from __future__ import annotations

import decimal
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.db import get_engine

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def _clean_value(v: Any) -> Any:
    """Limpa um valor escalar para insert SQL seguro."""
    if v is None:
        return None

    # pd.NaT / pd.NA / np.nan / float('nan')
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, (float, np.floating)):
        if math.isnan(float(v)):
            return None
        return float(v)
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, pd.Timestamp):
        # Se horário for 00:00:00, converte para date; caso contrário, mantém datetime.
        if v.hour == 0 and v.minute == 0 and v.second == 0 and v.microsecond == 0:
            return v.date()
        return v.to_pydatetime()
    return v


def clean_record(record: dict) -> dict:
    """
    Limpa um registro antes do insert:
    - pd.NaT / pd.NA / np.nan / float('nan') -> None
    - pd.Timestamp -> date/datetime nativo
    - mantém int/str/float normais
    """
    return {k: _clean_value(v) for k, v in record.items()}


def _df_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    records = []
    for row in df.to_dict("records"):
        records.append(clean_record(row))
    return records


def _statements_from_sql(content: str) -> list[str]:
    """Separa comandos SQL simples (arquivos DDL sem strings com ';')."""
    lines: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        lines.append(line)
    buf = "\n".join(lines)
    out: list[str] = []
    for part in buf.split(";"):
        s = part.strip()
        if s:
            out.append(s)
    return out


def apply_schema(engine=None) -> None:
    """Executa sql/01_schema.sql e sql/02_indexes.sql."""
    eng = engine or get_engine()
    files = [SQL_DIR / "01_schema.sql", SQL_DIR / "02_indexes.sql"]
    for fp in files:
        if not fp.is_file():
            raise FileNotFoundError(f"Arquivo SQL não encontrado: {fp}")
        raw = fp.read_text(encoding="utf-8")
        stmts = _statements_from_sql(raw)
        with eng.begin() as conn:
            for stmt in stmts:
                conn.execute(text(stmt))


def _fetch_tempo_map(conn) -> dict[tuple[int, int], int]:
    result = conn.execute(
        text("SELECT id_tempo, ano, semana_epidemiologica FROM tempo")
    ).mappings()
    return {
        (int(m["ano"]), int(m["semana_epidemiologica"])): int(m["id_tempo"])
        for m in result
    }


def load_all(tables: dict[str, pd.DataFrame], engine=None) -> None:
    """
    Insere dados na ordem: municipio → tempo → fato_dengue → populacao_municipio
    → fato_indicador; apaga e recria carta_de_fato.

    Usa ON CONFLICT onde há restrição única. Resolve id_tempo pelo par (ano, semana_epidemiológica).
    """
    eng = engine or get_engine()
    apply_schema(eng)

    mun = tables.get("municipio", pd.DataFrame())
    tempo_df = tables.get("tempo", pd.DataFrame())
    fd = tables.get("fato_dengue", pd.DataFrame())
    pm = tables.get("populacao_municipio", pd.DataFrame())
    fi = tables.get("fato_indicador", pd.DataFrame())
    carta = tables.get("carta_de_fato", pd.DataFrame())

    ins_mun = text(
        """
        INSERT INTO municipio (id_municipio_ibge, nome_municipio, uf_sigla, uf_codigo)
        VALUES (:id_municipio_ibge, :nome_municipio, :uf_sigla, :uf_codigo)
        ON CONFLICT (id_municipio_ibge) DO UPDATE SET
            nome_municipio = EXCLUDED.nome_municipio,
            uf_sigla = EXCLUDED.uf_sigla,
            uf_codigo = EXCLUDED.uf_codigo
        """
    )

    ins_tempo = text(
        """
        INSERT INTO tempo (ano, semana_epidemiologica, mes, dt_inicio, dt_fim)
        VALUES (:ano, :semana_epidemiologica, :mes, :dt_inicio, :dt_fim)
        ON CONFLICT (ano, semana_epidemiologica) DO UPDATE SET
            mes = EXCLUDED.mes,
            dt_inicio = EXCLUDED.dt_inicio,
            dt_fim = EXCLUDED.dt_fim
        """
    )

    ins_fd = text(
        """
        INSERT INTO fato_dengue (id_municipio_ibge, id_tempo, casos, fonte, data_extracao)
        VALUES (:id_municipio_ibge, :id_tempo, :casos, :fonte, :data_extracao)
        ON CONFLICT (id_municipio_ibge, id_tempo) DO UPDATE SET
            casos = EXCLUDED.casos,
            fonte = EXCLUDED.fonte,
            data_extracao = EXCLUDED.data_extracao
        """
    )

    ins_pm = text(
        """
        INSERT INTO populacao_municipio (id_municipio_ibge, ano, populacao, fonte, data_extracao)
        VALUES (:id_municipio_ibge, :ano, :populacao, :fonte, :data_extracao)
        ON CONFLICT (id_municipio_ibge, ano) DO UPDATE SET
            populacao = EXCLUDED.populacao,
            fonte = EXCLUDED.fonte,
            data_extracao = EXCLUDED.data_extracao
        """
    )

    ins_fi = text(
        """
        INSERT INTO fato_indicador (
            id_municipio_ibge, id_tempo, casos, populacao, incidencia_100k,
            fonte_casos, fonte_populacao, data_calculo
        )
        VALUES (
            :id_municipio_ibge, :id_tempo, :casos, :populacao, :incidencia_100k,
            :fonte_casos, :fonte_populacao, :data_calculo
        )
        ON CONFLICT (id_municipio_ibge, id_tempo) DO UPDATE SET
            casos = EXCLUDED.casos,
            populacao = EXCLUDED.populacao,
            incidencia_100k = EXCLUDED.incidencia_100k,
            fonte_casos = EXCLUDED.fonte_casos,
            fonte_populacao = EXCLUDED.fonte_populacao,
            data_calculo = EXCLUDED.data_calculo
        """
    )

    ins_carta = text(
        """
        INSERT INTO carta_de_fato (
            id_indicador, texto, municipio, uf_sigla, ano, semana_epidemiologica,
            incidencia_100k, casos, populacao, fonte, data_extracao
        )
        VALUES (
            :id_indicador, :texto, :municipio, :uf_sigla, :ano, :semana_epidemiologica,
            :incidencia_100k, :casos, :populacao, :fonte, :data_extracao
        )
        """
    )

    with eng.begin() as conn:
        for r in _df_records(mun):
            conn.execute(ins_mun, r)

        if not tempo_df.empty:
            tempo_rows = _df_records(
                tempo_df[
                    ["ano", "semana_epidemiologica", "mes", "dt_inicio", "dt_fim"]
                ]
            )
        else:
            tempo_rows = []
        for r in tempo_rows:
            conn.execute(ins_tempo, r)

        tempo_map = _fetch_tempo_map(conn)

        if not fd.empty and not tempo_df.empty:
            tref = tempo_df[["id_tempo", "ano", "semana_epidemiologica"]].copy()
            fd_loc = fd.drop(columns=["id_fato"], errors="ignore").merge(
                tref, on="id_tempo", how="left"
            )
            fd_loc["id_tempo"] = [
                tempo_map[(int(a), int(s))]
                for a, s in zip(fd_loc["ano"], fd_loc["semana_epidemiologica"])
            ]
            fd_ins = fd_loc[
                [
                    "id_municipio_ibge",
                    "id_tempo",
                    "casos",
                    "fonte",
                    "data_extracao",
                ]
            ]
        elif not fd.empty:
            fd_ins = fd.drop(columns=["id_fato"], errors="ignore")[
                ["id_municipio_ibge", "id_tempo", "casos", "fonte", "data_extracao"]
            ]
        else:
            fd_ins = fd

        for r in _df_records(fd_ins):
            conn.execute(ins_fd, r)

        pm_ins = pm.drop(columns=["id_pop"], errors="ignore") if not pm.empty else pm
        for r in _df_records(pm_ins):
            conn.execute(ins_pm, r)

        if not fi.empty and not tempo_df.empty:
            tref = tempo_df[["id_tempo", "ano", "semana_epidemiologica"]].copy()
            fi_loc = fi.drop(columns=["id_indicador"], errors="ignore").merge(
                tref, on="id_tempo", how="left"
            )
            fi_loc["id_tempo"] = [
                tempo_map[(int(a), int(s))]
                for a, s in zip(fi_loc["ano"], fi_loc["semana_epidemiologica"])
            ]
            fi_loc = fi_loc.drop(columns=["ano", "semana_epidemiologica"], errors="ignore")
            fi_ins = fi_loc[
                [
                    "id_municipio_ibge",
                    "id_tempo",
                    "casos",
                    "populacao",
                    "incidencia_100k",
                    "fonte_casos",
                    "fonte_populacao",
                    "data_calculo",
                ]
            ]
        elif not fi.empty:
            fi_ins = fi.drop(columns=["id_indicador"], errors="ignore")[
                [
                    "id_municipio_ibge",
                    "id_tempo",
                    "casos",
                    "populacao",
                    "incidencia_100k",
                    "fonte_casos",
                    "fonte_populacao",
                    "data_calculo",
                ]
            ]
        else:
            fi_ins = fi

        for r in _df_records(fi_ins):
            conn.execute(ins_fi, r)

        conn.execute(text("DELETE FROM carta_de_fato"))

        fi_keys = pd.read_sql(
            text(
                """
                SELECT
                    fi.id_indicador,
                    fi.id_municipio_ibge,
                    t.ano,
                    t.semana_epidemiologica
                FROM fato_indicador fi
                JOIN tempo t ON t.id_tempo = fi.id_tempo
                """
            ),
            conn,
        )
        # Só colunas de junção + id vindo do banco (sem colidir com df_cartas)
        fi_keys = fi_keys[
            ["id_municipio_ibge", "ano", "semana_epidemiologica", "id_indicador"]
        ]
        print(f"[LOAD] indicadores encontrados no banco: {len(fi_keys)}")

        print(f"[LOAD] cartas recebidas do transform: {len(carta)}")
        if not carta.empty:
            df_cartas = carta.copy()

            # Garante id_municipio_ibge (cartas do transform não trazem essa coluna).
            # Só id_municipio_ibge no lado direito: evita ano_x/ano_y e semanas nulas no merge com fi_keys.
            if "id_municipio_ibge" not in df_cartas.columns:
                if (
                    "id_indicador" in df_cartas.columns
                    and not fi.empty
                    and "id_municipio_ibge" in fi.columns
                ):
                    df_cartas = df_cartas.merge(
                        fi[["id_indicador", "id_municipio_ibge"]].drop_duplicates(
                            subset=["id_indicador"], keep="first"
                        ),
                        on="id_indicador",
                        how="left",
                    )
                else:
                    df_cartas["id_municipio_ibge"] = np.nan

            # Obrigatório antes do merge com fi_keys: remove ids do transform (só id_indicador do banco após o join)
            df_cartas = df_cartas.drop(columns=["id_indicador", "id_carta"], errors="ignore")

            needed_cols = [
                "id_municipio_ibge",
                "texto",
                "municipio",
                "uf_sigla",
                "ano",
                "semana_epidemiologica",
                "incidencia_100k",
                "casos",
                "populacao",
                "fonte",
                "data_extracao",
            ]
            for c in needed_cols:
                if c not in df_cartas.columns:
                    df_cartas[c] = np.nan
            df_cartas = df_cartas[needed_cols].copy()

            merge_keys = ["id_municipio_ibge", "ano", "semana_epidemiologica"]

            def _normalize_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
                d = df.copy()
                for k in merge_keys:
                    d[k] = pd.to_numeric(d[k], errors="coerce")
                d = d.dropna(subset=merge_keys, how="any")
                for k in merge_keys:
                    d[k] = d[k].round(0).astype(np.int64)
                return d

            df_cartas = _normalize_merge_keys(df_cartas)
            fi_keys_m = _normalize_merge_keys(fi_keys)
            fi_keys_m = fi_keys_m.drop_duplicates(subset=merge_keys, keep="first")

            cartas_linkadas = df_cartas.merge(
                fi_keys_m,
                on=merge_keys,
                how="left",
            )
            print(f"[LOAD] cartas após merge: {len(cartas_linkadas)}")
            print(cartas_linkadas.columns.tolist())
            print(cartas_linkadas[merge_keys + ["id_indicador"]].head())

            cartas_linkadas = cartas_linkadas[cartas_linkadas["id_indicador"].notna()].copy()
            ncom = len(cartas_linkadas)
            print(f"[LOAD] cartas com id_indicador: {ncom}")

            if not cartas_linkadas.empty:
                cartas_linkadas["id_indicador"] = cartas_linkadas["id_indicador"].astype(int)
                carta_ins = cartas_linkadas[
                    [
                        "id_indicador",
                        "texto",
                        "municipio",
                        "uf_sigla",
                        "ano",
                        "semana_epidemiologica",
                        "incidencia_100k",
                        "casos",
                        "populacao",
                        "fonte",
                        "data_extracao",
                    ]
                ]
                inserted = 0
                for r in _df_records(carta_ins):
                    conn.execute(ins_carta, r)
                    inserted += 1
                print(f"[LOAD] cartas inseridas: {inserted}")
            else:
                print("[LOAD] cartas inseridas: 0")


def run(*args, **kwargs) -> None:
    """Compatível com chamadas antigas; delega para load_all se receber tabelas."""
    if args and isinstance(args[0], dict):
        load_all(args[0], **kwargs)

