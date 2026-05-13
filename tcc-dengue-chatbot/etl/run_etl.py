"""Orquestra extração, transformação e carga (ETL). Execute: python -m etl.run_etl"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from etl.extract_dengue import extract_dengue_data
from etl.extract_ibge import MSG_SEM_ARQUIVO, extract_ibge_population
from etl.load import load_all
from etl.transform import transform_data


def main() -> None:
    df_dengue = extract_dengue_data()
    print(f"Linhas lidas (dengue): {len(df_dengue)}")

    try:
        df_ibge = extract_ibge_population()
    except ValueError as exc:
        if str(exc) == MSG_SEM_ARQUIVO or str(exc).startswith("Coloque o arquivo"):
            print(f"Aviso IBGE: {exc}")
            df_ibge = pd.DataFrame()
        else:
            raise
    print(f"Linhas lidas (IBGE): {len(df_ibge)}")

    tables = transform_data(df_dengue, df_ibge)
    print(f"Municípios: {len(tables['municipio'])}")
    print(f"Fatos dengue: {len(tables['fato_dengue'])}")
    print(f"Indicadores: {len(tables['fato_indicador'])}")
    print(f"Cartas: {len(tables['carta_de_fato'])}")

    load_all(tables)
    print("Sucesso: ETL finalizado e dados carregados no PostgreSQL.")


if __name__ == "__main__":
    main()
