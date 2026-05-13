"""
Extração de arquivo de população municipal (IBGE) em data/raw/ibge.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

COL_CODIGO_IBGE = [
    "id_municipio_ibge",
    "cod_ibge",
    "geocodigo",
    "codigo",
]
COL_COD_UF = ["uf", "cod_uf", "codigo_uf"]
COL_COD_MUNICIPIO = ["cod_municipio", "codigo_municipio", "cod_mun", "codigo"]
COL_NOME = ["nome_municipio", "municipio", "nome"]
COL_UF = ["uf_sigla", "uf", "sigla_uf"]
COL_ANO = ["ano"]
COL_POP = [
    "populacao",
    "pop",
    "habitantes",
    "populacao_estimada",
    "populacao_residente",
]

MSG_SEM_ARQUIVO = "Coloque o arquivo de população do IBGE em data/raw/ibge."
HEADERS_TRY = list(range(10))

UF_CODE_TO_SIGLA = {
    "11": "RO",
    "12": "AC",
    "13": "AM",
    "14": "RR",
    "15": "PA",
    "16": "AP",
    "17": "TO",
    "21": "MA",
    "22": "PI",
    "23": "CE",
    "24": "RN",
    "25": "PB",
    "26": "PE",
    "27": "AL",
    "28": "SE",
    "29": "BA",
    "31": "MG",
    "32": "ES",
    "33": "RJ",
    "35": "SP",
    "41": "PR",
    "42": "SC",
    "43": "RS",
    "50": "MS",
    "51": "MT",
    "52": "GO",
    "53": "DF",
}


def _normalize_col(name: str) -> str:
    s = str(name).strip().lower()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )
    s = s.replace(".", " ")
    s = s.replace("-", " ")
    s = s.replace("/", " ")
    s = re.sub(r"\s+", "_", s)
    return s


def _dedupe_columns(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def _read_ibge_file(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, low_memory=False, encoding_errors="replace")
    if ext in (".xlsx", ".xlsm"):
        return pd.read_excel(path, engine="openpyxl")
    if ext == ".xls":
        # .xls requer xlrd instalado
        return pd.read_excel(path)
    raise ValueError(f"Formato não aceito (use .csv, .xlsx ou .xls): {path.suffix}")


def _is_pop2025_official(path: Path) -> bool:
    return "pop2025" in path.name.lower()


def _has_municipios_sheet(path: Path) -> bool:
    try:
        xls = pd.ExcelFile(path)
        if "Municípios" in xls.sheet_names:
            return True
        return _resolve_sheet_municipios_name(xls) is not None
    except Exception:
        return False


def _resolve_sheet_municipios_name(xls: pd.ExcelFile) -> str | None:
    """
    Retorna o nome exato da aba cujo título normalizado é só "municipios"
    (ex.: "Municípios", "MUNICIPIOS"). Não usa outras abas.
    """
    best: str | None = None
    for raw in xls.sheet_names:
        s = str(raw).strip()
        norm = _normalize_col(s)
        if norm == "municipios":
            if s == "Municípios":
                return s
            if best is None:
                best = s
    return best


def _read_official_pop2025(path: Path) -> pd.DataFrame:
    """
    Leitura do arquivo oficial IBGE POP2025 (aba Municípios).

    Leitura obrigatória: header=1, todas as colunas como str, colunas fixas do IBGE.
    """
    PALMAS_ID_ESPERADO = 1_721_000
    POP_UF_TOCANTINS_ERR = 3_284_990

    print(f"[IBGE] arquivo lido: {path.name}")

    try:
        xls = pd.ExcelFile(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Não foi possível abrir o Excel: {path.name}: {exc}") from exc
    if "Municípios" not in xls.sheet_names:
        raise ValueError(
            "Aba 'Municípios' não encontrada no arquivo. "
            f"Abas disponíveis: {xls.sheet_names!r}"
        )

    df = pd.read_excel(path, sheet_name="Municípios", header=1, dtype=str)
    df = df.dropna(how="all").copy()

    print("[IBGE DEBUG] primeiras 10 linhas brutas:")
    print(df.head(10).to_string())

    required = ["UF", "COD. UF", "COD. MUNIC", "NOME DO MUNICÍPIO", "POPULAÇÃO ESTIMADA"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no POP2025: {missing}")

    dbg_u = df["UF"].fillna("").astype(str).str.strip().str.upper()
    dbg_n = df["NOME DO MUNICÍPIO"].fillna("").astype(str).str.strip()
    palmas_bruto = df.loc[(dbg_u == "TO") & (dbg_n == "Palmas")].copy()
    print("[IBGE DEBUG] linha(s) brutas UF=TO, NOME DO MUNICÍPIO=Palmas:")
    if palmas_bruto.empty:
        print("  (nenhuma)")
    else:
        print(palmas_bruto.to_string())

    # Somente colunas oficiais; população sempre da coluna POPULAÇÃO ESTIMADA da mesma linha
    df = df[required].copy()
    for col in required:
        s = df[col].fillna("").astype(str).str.strip()
        s = s.mask(s.str.upper().isin(["NAN", "NONE"]), "")
        df[col] = s

    cod_mun_raw = df["COD. MUNIC"].str.replace(r"\D", "", regex=True)
    cod_uf_raw = df["COD. UF"].str.replace(r"\D", "", regex=True)

    pop_digits = (
        df["POPULAÇÃO ESTIMADA"]
        .str.replace(".", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(r"\D", "", regex=True)
    )
    pop_num = pd.to_numeric(pop_digits, errors="coerce")

    cod_mun_nulo = cod_mun_raw.isna() | (cod_mun_raw.str.len() == 0)
    cod_mun_total_uf = cod_mun_raw == "00000"
    nome_nulo = df["NOME DO MUNICÍPIO"].str.len() == 0
    pop_nula = pop_num.isna()

    mask_ok = ~(cod_mun_nulo | cod_mun_total_uf | nome_nulo | pop_nula)
    df = df.loc[mask_ok].copy()
    cod_mun_raw = df["COD. MUNIC"].str.replace(r"\D", "", regex=True)
    cod_uf_raw = df["COD. UF"].str.replace(r"\D", "", regex=True)
    pop_digits = (
        df["POPULAÇÃO ESTIMADA"]
        .str.replace(".", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(r"\D", "", regex=True)
    )
    pop_num = pd.to_numeric(pop_digits, errors="coerce")

    cod_uf = cod_uf_raw.str.zfill(2).str[-2:]
    cod_mun = cod_mun_raw.str.zfill(5).str[-5:]
    id_ibge_str = cod_uf + cod_mun
    id_ibge = pd.to_numeric(id_ibge_str, errors="coerce")

    out = pd.DataFrame(
        {
            "id_municipio_ibge": id_ibge,
            "nome_municipio": df["NOME DO MUNICÍPIO"].str.strip(),
            "uf_sigla": df["UF"].str.strip().str.upper().str[:2],
            "ano": 2025,
            "populacao": pop_num,
        }
    )

    out = out.dropna(subset=["id_municipio_ibge", "nome_municipio", "uf_sigla", "populacao"], how="any")
    out = out[out["uf_sigla"].astype(str).str.len() == 2]

    out["id_municipio_ibge"] = out["id_municipio_ibge"].astype(np.int64)
    out["ano"] = out["ano"].astype(np.int64)
    out["populacao"] = out["populacao"].astype(np.int64)
    out = out.drop_duplicates(subset=["id_municipio_ibge", "ano"], keep="first")
    out = out.reset_index(drop=True)

    print(f"[IBGE] colunas usadas: {required}")
    print(f"[IBGE] total de municípios carregados: {len(out)}")

    palmas_to = out[
        (out["nome_municipio"].str.strip().str.upper() == "PALMAS")
        & (out["uf_sigla"].astype(str).str.upper().str[:2] == "TO")
    ]
    if palmas_to.empty:
        print("[IBGE] AVISO: Palmas/TO não encontrada após filtros (COD. MUNIC, nome, pop).")
    else:
        if (palmas_to["id_municipio_ibge"] == PALMAS_ID_ESPERADO).any():
            row = palmas_to.loc[palmas_to["id_municipio_ibge"] == PALMAS_ID_ESPERADO].iloc[0]
        else:
            row = palmas_to.iloc[0]
        pid = int(row["id_municipio_ibge"])
        ppop = int(row["populacao"])
        pnome = str(row["nome_municipio"])
        puf = str(row["uf_sigla"])
        print(
            f"[IBGE DEBUG] Palmas/TO encontrada: id={pid}, nome={pnome!r}, uf={puf!r}, populacao={ppop}"
        )
        if ppop == POP_UF_TOCANTINS_ERR:
            raise ValueError(
                "Leitura IBGE incorreta: a população associada a Palmas/TO foi 3.284.990, "
                "valor que corresponde à população estimada do estado do Tocantins (UF), "
                "não à do município de Palmas. O arquivo foi interpretado de forma incorreta. "
                "Confira: (1) aba exatamente 'Municípios' com header=1; (2) coluna "
                "'POPULAÇÃO ESTIMADA' alinhada à linha do município (COD. MUNIC ≠ '00000'); "
                "(3) ausência de deslocamento de colunas ou linha de total de UF misturada ao bloco municipal."
            )
        if pid != PALMAS_ID_ESPERADO:
            raise ValueError(
                f"Palmas/TO: id_municipio_ibge esperado {PALMAS_ID_ESPERADO} (COD. UF 17 + COD. MUNIC 21000), "
                f"obtido {pid}. Verifique COD. UF e COD. MUNIC na linha do município no arquivo."
            )

    return out


def _list_candidate_files(raw_path: Path) -> list[Path]:
    if not raw_path.is_dir():
        raise ValueError(MSG_SEM_ARQUIVO)
    exts = {".csv", ".xlsx", ".xlsm", ".xls"}
    files = sorted(p for p in raw_path.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if not files:
        raise ValueError(MSG_SEM_ARQUIVO)
    return files


def _first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _try_map_columns(df: pd.DataFrame) -> dict[str, str | None] | None:
    """
    Retorna mapa canônico -> coluna interna; None se faltar população/nome e
    forma de identificar município.
    """
    col_cod_ibge = _first_column(df, COL_CODIGO_IBGE)
    col_cod_uf = _first_column(df, COL_COD_UF)
    col_cod_mun = _first_column(df, COL_COD_MUNICIPIO)
    col_nome = _first_column(df, COL_NOME)
    col_uf = _first_column(df, COL_UF)
    col_ano = _first_column(df, COL_ANO)
    col_pop = _first_column(df, COL_POP)

    has_municipio_id = col_cod_ibge is not None or (
        col_cod_uf is not None and col_cod_mun is not None
    )
    if not has_municipio_id:
        return None
    if col_nome is None or col_pop is None:
        return None

    return {
        "id_municipio_ibge": col_cod_ibge,
        "cod_uf": col_cod_uf,
        "cod_municipio": col_cod_mun,
        "nome_municipio": col_nome,
        "uf_sigla": col_uf,
        "ano": col_ano,
        "populacao": col_pop,
    }


def _normalize_code(v: object) -> str | None:
    if v is None or pd.isna(v):
        return None
    s = str(v).strip()
    # remove parte decimal comum de excel (ex.: 17.0)
    s = re.sub(r"\.0+$", "", s)
    s = re.sub(r"\D", "", s)
    return s or None


def _clean_non_data_rows(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    out = df.copy()
    nome_col = mapping["nome_municipio"]
    pop_col = mapping["populacao"]

    if nome_col and nome_col in out.columns:
        nome = out[nome_col].astype(str).str.strip()
        # remove totais e cabeçalhos repetidos
        mask_total = nome.str.contains(r"^total\b", case=False, na=False)
        mask_header = nome.str.contains(r"municipio|nome_do_municipio|nome_municipio", case=False, na=False)
        out = out.loc[~(mask_total | mask_header)].copy()

    if pop_col and pop_col in out.columns:
        out[pop_col] = pd.to_numeric(out[pop_col], errors="coerce")
        out = out[out[pop_col].notna()].copy()

    return out


def _build_output(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    df = _clean_non_data_rows(df, mapping)

    nm = df[mapping["nome_municipio"]].astype(str).str.strip()
    nm = nm.replace({"nan": np.nan, "None": np.nan, "<na>": np.nan})

    # ID município: preferir coluna pronta; se não houver, compor cod_uf + cod_municipio
    if mapping["id_municipio_ibge"] is not None:
        cod_ibge = (
            df[mapping["id_municipio_ibge"]]
            .map(_normalize_code)
            .astype("string")
        )
    else:
        cod_uf = df[mapping["cod_uf"]].map(_normalize_code).astype("string")
        cod_mun = df[mapping["cod_municipio"]].map(_normalize_code).astype("string")
        # municipal geralmente 5 dígitos no IBGE ao concatenar com UF
        cod_mun = cod_mun.str.zfill(5)
        cod_ibge = (cod_uf.fillna("") + cod_mun.fillna("")).replace("", pd.NA)

    # UF sigla: usa coluna textual se existir; caso contrário, deriva de código UF
    if mapping["uf_sigla"] is not None:
        uf_sigla = (
            df[mapping["uf_sigla"]]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.slice(0, 2)
            .replace({"": np.nan, "NA": np.nan, "NAN": np.nan})
        )
    elif mapping["cod_uf"] is not None:
        cod_uf = df[mapping["cod_uf"]].map(_normalize_code)
        uf_sigla = cod_uf.map(UF_CODE_TO_SIGLA).astype("string")
    else:
        uf_sigla = pd.Series([pd.NA] * len(df), index=df.index, dtype="string")

    # Ano default 2025 quando ausente
    if mapping["ano"] is not None:
        ano_series = pd.to_numeric(df[mapping["ano"]], errors="coerce").fillna(2025)
    else:
        ano_series = pd.Series([2025] * len(df), index=df.index)

    out = pd.DataFrame(
        {
            "id_municipio_ibge": pd.to_numeric(cod_ibge, errors="coerce"),
            "nome_municipio": nm,
            "uf_sigla": uf_sigla,
            "ano": pd.to_numeric(ano_series, errors="coerce"),
            "populacao": pd.to_numeric(df[mapping["populacao"]], errors="coerce"),
        }
    )
    out["populacao"] = out["populacao"].round().astype("Int64")
    out["id_municipio_ibge"] = out["id_municipio_ibge"].astype("Int64")
    out["ano"] = out["ano"].astype("Int64")
    out = out.dropna(
        subset=["id_municipio_ibge", "nome_municipio", "uf_sigla", "ano", "populacao"],
        how="any",
    )
    out["id_municipio_ibge"] = out["id_municipio_ibge"].astype(np.int64)
    out["ano"] = out["ano"].astype(np.int64)
    out["populacao"] = out["populacao"].astype(np.int64)
    out = out.drop_duplicates(subset=["id_municipio_ibge", "ano"], keep="first")
    return out.reset_index(drop=True)


def _try_excel_with_headers(path: Path) -> tuple[pd.DataFrame, dict[str, str | None], int]:
    """
    Para Excel oficial do IBGE, tenta header=0..9 até encontrar mapeamento válido.
    """
    ext = path.suffix.lower()
    if ext not in (".xls", ".xlsx", ".xlsm"):
        raise ValueError("Função de header múltiplo só vale para Excel.")

    last_error: Exception | None = None
    for h in HEADERS_TRY:
        try:
            if ext in (".xlsx", ".xlsm"):
                df = pd.read_excel(path, engine="openpyxl", header=h)
            else:
                df = pd.read_excel(path, header=h)
            df.columns = _dedupe_columns([_normalize_col(c) for c in df.columns])
            mapping = _try_map_columns(df)
            if mapping is not None and not df.empty:
                print(f"[IBGE] header usado: {h}")
                print(f"[IBGE] colunas detectadas: {list(df.columns)}")
                return df, mapping, h
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    if last_error is not None:
        raise ValueError(f"Erro ao tentar headers 0..9: {last_error}")
    raise ValueError("Não foi possível detectar colunas compatíveis nos headers 0..9.")


def extract_ibge_population(raw_path: str | Path = "data/raw/ibge") -> pd.DataFrame:
    """
    Lê o primeiro arquivo .csv, .xlsx ou .xls utilizável na pasta e devolve colunas padronizadas.

    Retorna: id_municipio_ibge, nome_municipio, uf_sigla, ano, populacao.
    """
    raw = Path(raw_path)
    files = _list_candidate_files(raw)

    last_error: Exception | None = None
    for path in files:
        try:
            # Prioridade: layout oficial POP2025 / arquivos com aba Municípios
            if path.suffix.lower() in (".xls", ".xlsx", ".xlsm") and (
                _is_pop2025_official(path) or _has_municipios_sheet(path)
            ):
                return _read_official_pop2025(path)

            ext = path.suffix.lower()
            if ext in (".xls", ".xlsx", ".xlsm"):
                chunk, mapping, _ = _try_excel_with_headers(path)
            else:
                chunk = _read_ibge_file(path)
                chunk.columns = _dedupe_columns([_normalize_col(c) for c in chunk.columns])
                mapping = _try_map_columns(chunk)
                if mapping is None or chunk.empty:
                    continue
                print("[IBGE] header usado: 0 (csv)")
                print(f"[IBGE] colunas detectadas: {list(chunk.columns)}")

            out = _build_output(chunk, mapping)
            print(f"[IBGE] total de municípios lidos: {len(out)}")
            return out
        except Exception as exc:  # noqa: BLE001 — tentar próximo arquivo
            last_error = ValueError(f"Erro ao ler '{path.name}': {exc}")
            continue

    detail = f" Último erro: {last_error}" if last_error else ""
    raise ValueError(
        "Nenhum arquivo .csv/.xlsx/.xls em data/raw/ibge pôde ser lido com as colunas "
        "esperadas (código IBGE, nome, UF, ano, população)."
        + detail
    )

