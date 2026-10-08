"""Leitura do snapshot da Gold gravado em `data/`, sem credencial GCP.

NOTA DE DESIGN: este projeto lê arquivos Parquet locais, e não o BigQuery, de
propósito. A Gold da Fase 2 é efêmera (a infraestrutura é destruída depois do uso
para não gerar custo) e o snapshot permite que qualquer pessoa rode o projeto sem
projeto GCP, faturamento nem credencial, com os mesmos dados em toda execução.

Isso poderia ser trocado para ler direto do GCP sem mexer no resto do código:
`carregar_tabela` é o único ponto de leitura dos dados, e basta recriar a Gold com
o pipeline do trabalho da Fase 2 (`TECH-CHALLENGE-FASE-2`: `make bronze silver gold`,
dataset `alfabetizacao_analytics`) e fazer esta função devolver o DataFrame de uma
consulta ao dataset em vez de ler o arquivo. `export_gold.py` já mostra as
consultas e as colunas de cada tabela. Os tipos dos DataFrames podem diferir um
pouco entre as duas fontes e precisariam ser conferidos nos testes.
"""

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
