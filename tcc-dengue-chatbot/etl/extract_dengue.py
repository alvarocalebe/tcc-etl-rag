
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

COL_MUNICIPIO = [
    "id_mn_resi",
    "id_municip",
    "cod_municipio",
    "municipio_residencia",
    "id_municipio_ibge",
]
COL_UF = ["sg_uf_not", "uf", "uf_sigla"]
COL_DATA = ["dt_notific", "data_notificacao"]
COL_ANO = ["nu_ano", "ano"]
COL_SEMANA = ["sem_not", "semana_epidemiologica", "semana"]


def _normalize_col(name: str) -> str:
    s = str(name).strip().lower()
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


def _read_one_file(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, low_memory=False, encoding_errors="replace")
    if ext == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Leitura de .parquet requer o pacote 'pyarrow' (ou 'fastparquet'). "
                "Instale com: pip install pyarrow"
            ) from exc
    if ext in (".xlsx", ".xlsm"):
        return pd.read_excel(path, engine="openpyxl")
    if ext == ".xls":
        return pd.read_excel(path)
    raise ValueError(f"Formato não suportado: {path.suffix}")


def _list_raw_files(raw_path: Path) -> list[Path]:
    exts = {".csv", ".parquet", ".xlsx", ".xlsm", ".xls"}
    if not raw_path.is_dir():
        raise FileNotFoundError(
            f"Pasta de dados brutos não encontrada: {raw_path}. "
            "Crie a pasta e adicione arquivos .csv, .parquet ou .xlsx."
        )
    files = sorted(
        p for p in raw_path.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    if not files:
        raise ValueError(
            f"Nenhum arquivo .csv, .parquet ou .xlsx em {raw_path}. "
            "Adicione ao menos um arquivo nesses formatos."
        )
    return files


def _first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Retorna o nome interno da coluna (já normalizado) se existir."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _build_standard_frame(
    df: pd.DataFrame,
    col_mun: str,
    col_uf: str | None,
    col_data: str | None,
    col_ano: str | None,
    col_sem: str | None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["id_municipio_ibge"] = pd.to_numeric(df[col_mun], errors="coerce")

    if col_uf:
        u = df[col_uf].astype(str).str.strip().str.upper()
        out["uf_sigla"] = u.replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
        out["uf_sigla"] = out["uf_sigla"].str.slice(0, 2)
    else:
        out["uf_sigla"] = np.nan

    if col_data:
        out["data_notificacao"] = pd.to_datetime(
            df[col_data],
            errors="coerce",
            dayfirst=True,
            format="mixed",
        )
    else:
        out["data_notificacao"] = pd.NaT

    ano_series = (
        pd.to_numeric(df[col_ano], errors="coerce")
        if col_ano
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    sem_series = (
        pd.to_numeric(df[col_sem], errors="coerce")
        if col_sem
        else pd.Series(np.nan, index=df.index, dtype=float)
    )

    dt = out["data_notificacao"]
    ano_series = ano_series.where(ano_series.notna(), dt.dt.year)

    # Semana epidemiológica ausente: ISO 8601 a partir da data (compatível com várias bases Sinan)
    sem_iso = pd.to_numeric(dt.dt.isocalendar().week, errors="coerce")
    sem_series = sem_series.where(sem_series.notna(), sem_iso)

    out["ano"] = ano_series
    out["semana_epidemiologica"] = sem_series

    mask_ano = out["ano"].isna() & dt.notna()
    out.loc[mask_ano, "ano"] = out.loc[mask_ano, "data_notificacao"].dt.year.astype(float)

    out["ano"] = pd.to_numeric(out["ano"], errors="coerce")
    out["semana_epidemiologica"] = pd.to_numeric(out["semana_epidemiologica"], errors="coerce")

    return out[
        [
            "id_municipio_ibge",
            "uf_sigla",
            "data_notificacao",
            "ano",
            "semana_epidemiologica",
        ]
    ]


def extract_dengue_data(raw_path: str | Path = "data/raw/dengue") -> pd.DataFrame:
    """
    Lê todos os .csv / .parquet / .xlsx da pasta, unifica e devolve colunas padronizadas.

    Retorna colunas: id_municipio_ibge, uf_sigla, data_notificacao, ano,
    semana_epidemiologica.
    """
    raw = Path(raw_path)
    files = _list_raw_files(raw)

    frames: list[pd.DataFrame] = []
    for fp in files:
        chunk = _read_one_file(fp)
        chunk.columns = _dedupe_columns([_normalize_col(c) for c in chunk.columns])
        frames.append(chunk)

    merged = pd.concat(frames, ignore_index=True, sort=False)

    if merged.empty:
        raise ValueError(
            f"Nenhuma linha foi lida dos arquivos em {raw.resolve()}. "
            "Verifique se os arquivos contêm dados."
        )

    col_mun = _first_column(merged, COL_MUNICIPIO)
    if col_mun is None:
        raise ValueError(
            "Nenhuma coluna de município encontrada após normalizar nomes. "
            f"Esperado uma das colunas: {', '.join(COL_MUNICIPIO)}. "
            f"Colunas presentes: {list(merged.columns)}"
        )

    col_uf = _first_column(merged, COL_UF)
    col_data = _first_column(merged, COL_DATA)
    col_ano = _first_column(merged, COL_ANO)
    col_sem = _first_column(merged, COL_SEMANA)

    # Precisa haver data OU par ano+semana OU data para derivar ambos
    if col_data is None and col_ano is None:
        raise ValueError(
            "Não foi possível localizar colunas de data (dt_notific, data_notificacao) "
            "nem de ano (nu_ano, ano). Pelo menos uma dessas é necessária para definir ano "
            "e semana."
        )
    if col_data is None and col_sem is None:
        raise ValueError(
            "Sem coluna de data e sem coluna de semana epidemiológica: "
            "inclua dt_notific/data_notificacao ou sem_not/semana_epidemiologica/semana."
        )

    result = _build_standard_frame(
        merged,
        col_mun=col_mun,
        col_uf=col_uf,
        col_data=col_data,
        col_ano=col_ano,
        col_sem=col_sem,
    )

    before = len(result)
    result = result.dropna(subset=["id_municipio_ibge", "ano", "semana_epidemiologica"])
    result["id_municipio_ibge"] = result["id_municipio_ibge"].astype(np.int64)
    result["ano"] = result["ano"].astype(int)
    result["semana_epidemiologica"] = result["semana_epidemiologica"].astype(int)

    if result.empty and before > 0:
        raise ValueError(
            "Todos os registros foram descartados: verifique códigos IBGE, ano e semana."
        )

    return result.reset_index(drop=True)
