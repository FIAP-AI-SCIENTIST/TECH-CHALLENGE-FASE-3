"""Leitura do snapshot da Gold gravado em `data/`, sem credencial GCP."""

from pathlib import Path

import pandas as pd

from src.config import DIRETORIO_DADOS, TABELAS_SNAPSHOT


def carregar_tabela(nome: str, diretorio: Path = DIRETORIO_DADOS) -> pd.DataFrame:
    """Lê `<diretorio>/<nome>.parquet` como DataFrame.

    Falha com `ValueError` se o nome não for uma tabela do snapshot e com
    `FileNotFoundError` se o arquivo não existir (nesse caso, rode
    `python -m src.preprocessing.export_gold`).
    """
    if nome not in TABELAS_SNAPSHOT:
        raise ValueError(f"Tabela desconhecida: {nome}. Opções: {', '.join(TABELAS_SNAPSHOT)}.")
    caminho = diretorio / f"{nome}.parquet"
    if not caminho.exists():
        raise FileNotFoundError(
            f"Snapshot não encontrado em {caminho}. "
            "Gere-o com `python -m src.preprocessing.export_gold`."
        )
    return pd.read_parquet(caminho)
