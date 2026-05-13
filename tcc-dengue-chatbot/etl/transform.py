"""
Transforma DataFrames de dengue (Sinan) e IBGE em dimensões e fatos para carga no DW.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def _first_non_null(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return None
    return s.iloc[0]


def _norm_name(v: object) -> str | None:
    if v is None or pd.isna(v):
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    s = (
        s.replace("á", "a")
        .replace("à", "a")
        .replace("ã", "a")
        .replace("â", "a")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ô", "o")
        .replace("õ", "o")
        .replace("ú", "u")
        .replace("ç", "c")
    )
    return " ".join(s.split())


def transform_data(
    df_dengue: pd.DataFrame,
    df_ibge: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Agrupa dengue por município/ano/semana, integra IBGE, calcula incidência e cartas.

    Retorna dicionário com DataFrames: municipio, tempo, fato_dengue,
    populacao_municipio, fato_indicador, carta_de_fato.
    """
    hoje = date.today()
    cols = {
        "municipio": [
            "id_municipio_ibge",
            "nome_municipio",
            "uf_sigla",
            "uf_codigo",
        ],
        "tempo": [
            "id_tempo",
            "ano",
            "semana_epidemiologica",
            "mes",
            "dt_inicio",
            "dt_fim",
        ],
        "fato_dengue": [
            "id_fato",
            "id_municipio_ibge",
            "id_tempo",
            "casos",
            "fonte",
            "data_extracao",
        ],
        "populacao_municipio": [
            "id_pop",
            "id_municipio_ibge",
            "ano",
            "populacao",
            "fonte",
            "data_extracao",
        ],
        "fato_indicador": [
            "id_indicador",
            "id_municipio_ibge",
            "id_tempo",
            "casos",
            "populacao",
            "incidencia_100k",
            "fonte_casos",
            "fonte_populacao",
            "data_calculo",
        ],
        "carta_de_fato": [
            "id_carta",
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
        ],
    }

    if df_dengue.empty:
        return {k: pd.DataFrame(columns=v) for k, v in cols.items()}

    if df_ibge is None:
        df_i = pd.DataFrame()
    else:
        df_i = df_ibge.copy()
        if not df_i.empty and "id_municipio_ibge" in df_i.columns:
            df_i["id_municipio_ibge"] = pd.to_numeric(
                df_i["id_municipio_ibge"], errors="coerce"
            )
            df_i = df_i.dropna(subset=["id_municipio_ibge"]).copy()
            df_i["id_municipio_ibge"] = (
                df_i["id_municipio_ibge"].round(0).astype(np.int64)
            )
            # Código oficial IBGE do município: 7 dígitos (UF 2 + município 5)
            _id_len = df_i["id_municipio_ibge"].astype(str).str.len()
            df_i = df_i[_id_len == 7].copy()

    df_d = df_dengue.copy()
    if "uf_sigla" not in df_d.columns:
        df_d["uf_sigla"] = np.nan

    # ------------------------------------------------------------------
    # Padronização de id_municipio_ibge (dengue -> código oficial IBGE 7 dígitos)
    # ------------------------------------------------------------------
    if "id_municipio_ibge" not in df_d.columns:
        raise ValueError("df_dengue deve conter a coluna 'id_municipio_ibge'.")

    # Nome do município (se existir) para matching por nome+UF
    dengue_name_col = None
    for cand in ["nome_municipio", "municipio", "municipio_residencia"]:
        if cand in df_d.columns:
            dengue_name_col = cand
            break
    if dengue_name_col is None:
        df_d["__mun_name_norm"] = None
    else:
        df_d["__mun_name_norm"] = df_d[dengue_name_col].map(_norm_name)

    df_d["id_municipio_ibge"] = pd.to_numeric(df_d["id_municipio_ibge"], errors="coerce")
    df_d["id_municipio_ibge"] = df_d["id_municipio_ibge"].astype("Int64")
    df_d["__id_len"] = df_d["id_municipio_ibge"].astype(str).str.replace("<NA>", "", regex=False).str.len()
    df_d["__id6"] = np.where(
        df_d["__id_len"] == 6,
        pd.to_numeric(df_d["id_municipio_ibge"], errors="coerce"),
        np.nan,
    )

    matched_rows = 0
    unmatched_rows = 0
    corrected_examples: list[tuple[int, int]] = []

    if not df_i.empty and "id_municipio_ibge" in df_i.columns:
        ibge_lookup = df_i.copy()
        ibge_lookup["id_municipio_ibge"] = pd.to_numeric(
            ibge_lookup["id_municipio_ibge"], errors="coerce"
        )
        ibge_lookup = ibge_lookup.dropna(subset=["id_municipio_ibge"]).copy()
        ibge_lookup["id_municipio_ibge"] = ibge_lookup["id_municipio_ibge"].astype(np.int64)
        ibge_lookup["__id6"] = (ibge_lookup["id_municipio_ibge"] // 10).astype(np.int64)
        if "uf_sigla" not in ibge_lookup.columns:
            ibge_lookup["uf_sigla"] = np.nan
        if "nome_municipio" not in ibge_lookup.columns:
            ibge_lookup["nome_municipio"] = np.nan
        ibge_lookup["uf_sigla"] = (
            ibge_lookup["uf_sigla"].astype(str).str.strip().str.upper().str.slice(0, 2)
        )
        ibge_lookup["__mun_name_norm"] = ibge_lookup["nome_municipio"].map(_norm_name)

        # 1) match por nome+UF (prioritário) para linhas de código 6 dígitos
        six_mask = df_d["__id_len"] == 6
        if six_mask.any():
            left = df_d.loc[six_mask, ["id_municipio_ibge", "uf_sigla", "__mun_name_norm", "__id6"]].copy()
            left["uf_sigla"] = left["uf_sigla"].astype(str).str.strip().str.upper().str.slice(0, 2)
            name_uf_match = left.merge(
                ibge_lookup[["id_municipio_ibge", "uf_sigla", "__mun_name_norm"]].dropna(
                    subset=["uf_sigla", "__mun_name_norm"]
                ),
                on=["uf_sigla", "__mun_name_norm"],
                how="left",
                suffixes=("_dengue", "_ibge"),
            )
            # índice alinhado com df_d[six_mask]
            idx = df_d.index[six_mask]
            resolved_name = pd.to_numeric(
                name_uf_match["id_municipio_ibge_ibge"], errors="coerce"
            )
            name_ok = resolved_name.notna()
            if name_ok.any():
                df_d.loc[idx[name_ok], "id_municipio_ibge"] = resolved_name[name_ok].astype("Int64").values

            # 2) fallback por __id6 -> id7 quando houver mapeamento único
            still_six_mask = (df_d["__id_len"] == 6) & df_d["id_municipio_ibge"].notna()
            still_idx = df_d.index[still_six_mask]
            if len(still_idx) > 0:
                unique_map = (
                    ibge_lookup.groupby("__id6")["id_municipio_ibge"]
                    .nunique()
                    .reset_index(name="n")
                )
                unique_map = unique_map[unique_map["n"] == 1][["__id6"]]
                unique_map = unique_map.merge(
                    ibge_lookup[["__id6", "id_municipio_ibge"]].drop_duplicates("__id6"),
                    on="__id6",
                    how="left",
                )
                map_dict = dict(zip(unique_map["__id6"], unique_map["id_municipio_ibge"]))
                old_vals = pd.to_numeric(df_d.loc[still_idx, "id_municipio_ibge"], errors="coerce")
                keys = pd.to_numeric(df_d.loc[still_idx, "__id6"], errors="coerce")
                new_vals = keys.map(map_dict)
                ok = new_vals.notna()
                if ok.any():
                    df_d.loc[still_idx[ok], "id_municipio_ibge"] = new_vals[ok].astype("Int64").values
                    # exemplos de correção
                    for old, new in zip(old_vals[ok].astype(int).tolist(), new_vals[ok].astype(int).tolist()):
                        if old != new and len(corrected_examples) < 10:
                            corrected_examples.append((old, new))

        # métricas de match
        final_numeric = pd.to_numeric(df_d["id_municipio_ibge"], errors="coerce")
        final_ids_set = set(ibge_lookup["id_municipio_ibge"].astype(np.int64).tolist())
        matched_mask = final_numeric.notna() & final_numeric.astype(np.int64).isin(final_ids_set)
        matched_rows = int(matched_mask.sum())
        unmatched_rows = int((final_numeric.notna() & ~matched_mask).sum())
    else:
        final_numeric = pd.to_numeric(df_d["id_municipio_ibge"], errors="coerce")
        matched_rows = int(final_numeric.notna().sum())
        unmatched_rows = 0

    # limpeza final do dataframe dengue
    df_d["id_municipio_ibge"] = pd.to_numeric(df_d["id_municipio_ibge"], errors="coerce").astype("Int64")
    df_d = df_d.dropna(subset=["id_municipio_ibge"]).copy()
    df_d["id_municipio_ibge"] = df_d["id_municipio_ibge"].astype(np.int64)
    print(f"[TRANSFORM] municípios com match correto IBGE: {matched_rows}")
    print(f"[TRANSFORM] municípios sem match IBGE: {unmatched_rows}")
    if corrected_examples:
        print(f"[TRANSFORM] exemplos de códigos corrigidos (6->7 dígitos): {corrected_examples[:5]}")

    for c in ["id_municipio_ibge", "ano", "semana_epidemiologica"]:
        if c not in df_d.columns:
            raise ValueError(f"df_dengue deve conter a coluna '{c}'.")

    agrupado = (
        df_d.groupby(["id_municipio_ibge", "ano", "semana_epidemiologica"], as_index=False)
        .agg(
            casos=("id_municipio_ibge", "count"),
            uf_sigla=("uf_sigla", _first_non_null),
        )
    )
    agrupado["casos"] = agrupado["casos"].astype(int)

    tempo_unique = (
        agrupado[["ano", "semana_epidemiologica"]]
        .drop_duplicates()
        .sort_values(["ano", "semana_epidemiologica"])
        .reset_index(drop=True)
    )
    tempo_unique["id_tempo"] = np.arange(1, len(tempo_unique) + 1, dtype=np.int64)
    tempo = tempo_unique.assign(mes=np.nan, dt_inicio=None, dt_fim=None)[
        ["id_tempo", "ano", "semana_epidemiologica", "mes", "dt_inicio", "dt_fim"]
    ]

    fato_base = agrupado.merge(
        tempo_unique[["id_tempo", "ano", "semana_epidemiologica"]],
        on=["ano", "semana_epidemiologica"],
        how="left",
    )
    fato_base["id_fato"] = np.arange(1, len(fato_base) + 1, dtype=np.int64)
    fato_dengue = fato_base[
        ["id_fato", "id_municipio_ibge", "id_tempo", "casos"]
    ].assign(
        fonte="Sinan/Dengue",
        data_extracao=hoje,
    )

    # ------------------------------------------------------------------
    # Dimensão municipio: união de municípios da dengue + municípios do IBGE
    # ------------------------------------------------------------------
    ids_dengue = pd.Series(agrupado["id_municipio_ibge"].unique(), name="id_municipio_ibge")
    ids_ibge = (
        pd.Series(df_i["id_municipio_ibge"].dropna().unique(), name="id_municipio_ibge")
        if ("id_municipio_ibge" in df_i.columns and not df_i.empty)
        else pd.Series([], dtype="int64", name="id_municipio_ibge")
    )
    all_ids = pd.concat([ids_dengue, ids_ibge], ignore_index=True).dropna().drop_duplicates()
    mun_rows = pd.DataFrame({"id_municipio_ibge": all_ids.astype(np.int64)})

    ibge_dim = pd.DataFrame(columns=["id_municipio_ibge", "nome_municipio", "uf_ibge"])
    if not df_i.empty and all(
        c in df_i.columns for c in ["id_municipio_ibge", "nome_municipio", "uf_sigla"]
    ):
        ibge_dim = (
            df_i.sort_values("ano", ascending=False)
            .drop_duplicates(subset=["id_municipio_ibge"], keep="first")[
                ["id_municipio_ibge", "nome_municipio", "uf_sigla"]
            ]
            .rename(columns={"uf_sigla": "uf_ibge"})
        )
    mun_rows = mun_rows.merge(ibge_dim, on="id_municipio_ibge", how="left")

    uf_por_mun = (
        agrupado.groupby("id_municipio_ibge")["uf_sigla"]
        .agg(_first_non_null)
        .rename("uf_dengue")
    )
    mun_rows = mun_rows.merge(
        uf_por_mun, left_on="id_municipio_ibge", right_index=True, how="left"
    )

    def _norm_uf_u2(v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
            return "NA"
        s = str(v).strip().upper()
        if s in ("NAN", "NONE", "", "<NA>"):
            return "NA"
        return s[:2]

    # Prioridade para dados do IBGE
    uf_joined = mun_rows["uf_ibge"].combine_first(mun_rows["uf_dengue"])
    mun_rows["uf_sigla"] = uf_joined.map(_norm_uf_u2)

    # Fallback para municípios só da dengue sem correspondência no IBGE
    mun_rows["nome_municipio"] = mun_rows["nome_municipio"].where(
        mun_rows["nome_municipio"].notna(),
        mun_rows["id_municipio_ibge"].map(lambda x: f"Município {int(x)}"),
    )

    # uf_codigo: primeiros 2 dígitos do código IBGE
    mun_rows["uf_codigo"] = (
        mun_rows["id_municipio_ibge"]
        .astype(str)
        .str.zfill(7)
        .str[:2]
        .astype("Int64")
    )

    municipio = mun_rows[
        ["id_municipio_ibge", "nome_municipio", "uf_sigla", "uf_codigo"]
    ].drop_duplicates(subset=["id_municipio_ibge"], keep="first")

    print(f"[TRANSFORM] municípios da dengue: {len(ids_dengue)}")
    print(f"[TRANSFORM] municípios do IBGE: {len(ids_ibge)}")
    print(f"[TRANSFORM] municípios finais na dimensão municipio: {len(municipio)}")

    if not df_i.empty and all(
        c in df_i.columns
        for c in ["id_municipio_ibge", "ano", "populacao"]
    ):
        # Merge de população sempre por id_municipio_ibge de 7 dígitos (já filtrado em df_i)
        pop_src = df_i[["id_municipio_ibge", "ano", "populacao"]].drop_duplicates(
            subset=["id_municipio_ibge", "ano"], keep="first"
        )
        pop_src["populacao"] = pd.to_numeric(pop_src["populacao"], errors="coerce")
        pop_src = pop_src.dropna(subset=["populacao"])
        pop_src["populacao"] = pop_src["populacao"].round().astype(np.int64)
        # Garante FK: toda população precisa existir na dimensão municipio
        pop_src = pop_src[
            pop_src["id_municipio_ibge"].isin(municipio["id_municipio_ibge"])
        ].copy()
        pop_src["id_pop"] = np.arange(1, len(pop_src) + 1, dtype=np.int64)
        populacao_municipio = pop_src.assign(
            fonte="IBGE", data_extracao=hoje
        )[["id_pop", "id_municipio_ibge", "ano", "populacao", "fonte", "data_extracao"]]
        print(f"[TRANSFORM] populações carregadas: {len(populacao_municipio)}")
    else:
        populacao_municipio = pd.DataFrame(
            columns=[
                "id_pop",
                "id_municipio_ibge",
                "ano",
                "populacao",
                "fonte",
                "data_extracao",
            ]
        )
        print("[TRANSFORM] populações carregadas: 0")

    fd_keys = fato_dengue.merge(
        tempo[["id_tempo", "ano", "semana_epidemiologica"]], on="id_tempo", how="left"
    )
    if not populacao_municipio.empty:
        pm_merge = populacao_municipio[["id_municipio_ibge", "ano", "populacao"]]
        fi_merge = fd_keys.merge(
            pm_merge,
            on=["id_municipio_ibge", "ano"],
            how="left",
        )
    else:
        fi_merge = fd_keys.assign(populacao=pd.NA)

    fi_merge["populacao"] = pd.to_numeric(fi_merge["populacao"], errors="coerce")
    tem_pop = fi_merge["populacao"].notna() & (fi_merge["populacao"] > 0)
    fi_merge["incidencia_100k"] = np.where(
        tem_pop,
        np.round(
            fi_merge["casos"].astype(float)
            / fi_merge["populacao"].astype(float)
            * 100_000,
            2,
        ),
        np.nan,
    )

    fi_merge = fi_merge.reset_index(drop=True)

    fato_indicador = pd.DataFrame(
        {
            "id_indicador": np.arange(1, len(fi_merge) + 1, dtype=np.int64),
            "id_municipio_ibge": fi_merge["id_municipio_ibge"].values,
            "id_tempo": fi_merge["id_tempo"].values,
            "casos": fi_merge["casos"].values,
            "populacao": fi_merge["populacao"],
            "incidencia_100k": fi_merge["incidencia_100k"],
            "fonte_casos": "Sinan/Dengue",
            "fonte_populacao": np.where(tem_pop, "IBGE", None),
            "data_calculo": hoje,
        }
    )
    fato_indicador["populacao"] = fato_indicador["populacao"].astype("Int64")

    nomes = municipio.set_index("id_municipio_ibge")["nome_municipio"].to_dict()
    ufs = municipio.set_index("id_municipio_ibge")["uf_sigla"].to_dict()

    cartas: list[dict] = []
    for i in range(len(fato_indicador)):
        row = fato_indicador.iloc[i]
        ctx = fi_merge.iloc[i]
        mid = int(row["id_municipio_ibge"])
        mun_nome = nomes.get(mid, "Nome não disponível")
        uf = ufs.get(mid, "NA")
        ano = int(ctx["ano"])
        sem = int(ctx["semana_epidemiologica"])
        casos = int(row["casos"])
        pop = row["populacao"]
        inc = row["incidencia_100k"]

        com_incidencia = (
            pd.notna(pop)
            and int(pop) > 0
            and pd.notna(inc)
        )

        if com_incidencia:
            texto = (
                f"Em {mun_nome}/{uf}, na semana epidemiológica {sem} de {ano}, "
                f"foram registrados {casos} casos de dengue na base Sinan/Dengue. "
                f"A população estimada usada foi {int(pop)} habitantes, resultando em "
                f"incidência de {float(inc):.2f} casos por 100 mil habitantes. "
                f"Fontes: Sinan/Dengue e IBGE."
            )
            fonte_txt = "Sinan/Dengue e IBGE"
        else:
            texto = (
                f"Em {mun_nome}/{uf}, na semana epidemiológica {sem} de {ano}, "
                f"foram registrados {casos} casos de dengue na base Sinan/Dengue. "
                "Não foi encontrada população correspondente do IBGE para calcular incidência. "
                "Fonte: Sinan/Dengue."
            )
            fonte_txt = "Sinan/Dengue"

        cartas.append(
            {
                "id_carta": i + 1,
                "id_indicador": int(row["id_indicador"]),
                "texto": texto,
                "municipio": mun_nome,
                "uf_sigla": uf,
                "ano": ano,
                "semana_epidemiologica": sem,
                "incidencia_100k": float(inc) if pd.notna(inc) else np.nan,
                "casos": casos,
                "populacao": int(pop)
                if pd.notna(pop) and int(pop) > 0
                else np.nan,
                "fonte": fonte_txt,
                "data_extracao": hoje,
            }
        )

    carta_de_fato = pd.DataFrame(cartas)

    return {
        "municipio": municipio,
        "tempo": tempo,
        "fato_dengue": fato_dengue,
        "populacao_municipio": populacao_municipio,
        "fato_indicador": fato_indicador,
        "carta_de_fato": carta_de_fato,
    }
