"""Exporta a camada Gold da Fase 2 (BigQuery) para Parquet em `data/`.

Roda uma vez por quem tem acesso ao projeto GCP; o restante do projeto lê o
snapshot gerado, sem credencial. A autenticação é por Application Default
Credentials (somente leitura): `gcloud auth application-default login`.

Este script é a ponte com o GCP: o projeto lê o snapshot por escolha, mas poderia
ler direto do BigQuery (ver a nota de design em `loader.py`).

Uso:
    GCP_PROJECT_ID=<id-do-projeto> python -m src.preprocessing.export_gold
"""

import os
from pathlib import Path

import pyarrow.parquet as pq
from google.cloud import bigquery

from src.config import (
    COLUNAS_FORA_DO_SNAPSHOT,
    DATASET_GOLD,
    DIRETORIO_DADOS,
    PREFIXO_CHAVE_SUBSTITUTA,
    TABELAS_SNAPSHOT,
    VARIAVEL_PROJETO_GCP,
)


def _colunas_do_snapshot(cliente: bigquery.Client, projeto: str, tabela: str) -> list[str]:
    """Colunas da tabela sem as chaves substitutas e sem as de aluno fora do snapshot."""
    esquema = cliente.get_table(f"{projeto}.{DATASET_GOLD}.{tabela}").schema
    removidas = COLUNAS_FORA_DO_SNAPSHOT.get(tabela, ())
    return [
        campo.name
        for campo in esquema
        if not campo.name.startswith(PREFIXO_CHAVE_SUBSTITUTA) and campo.name not in removidas
    ]


def exportar_gold(projeto: str | None = None, destino: Path = DIRETORIO_DADOS) -> dict[str, int]:
    """Grava cada tabela de `TABELAS_SNAPSHOT` como `<destino>/<tabela>.parquet`.

    Retorna o número de linhas gravadas por tabela. Falha cedo com `ValueError`
    se o projeto não for informado nem estiver na variável de ambiente.
    """
    projeto = projeto or os.environ.get(VARIAVEL_PROJETO_GCP)
    if not projeto:
        raise ValueError(
            f"Informe o projeto GCP no argumento ou na variável de ambiente {VARIAVEL_PROJETO_GCP}."
        )

    cliente = bigquery.Client(project=projeto)
    destino.mkdir(parents=True, exist_ok=True)

    linhas_por_tabela: dict[str, int] = {}
    for tabela in TABELAS_SNAPSHOT:
        colunas = ", ".join(f"`{coluna}`" for coluna in _colunas_do_snapshot(cliente, projeto, tabela))
        consulta = f"SELECT {colunas} FROM `{projeto}.{DATASET_GOLD}.{tabela}`"
        arrow = cliente.query(consulta).to_arrow()
        caminho = destino / f"{tabela}.parquet"
        pq.write_table(arrow, caminho, compression="zstd")
        linhas_por_tabela[tabela] = arrow.num_rows
        print(f"{tabela}: {arrow.num_rows} linhas, {caminho.stat().st_size / 1_048_576:.2f} MiB")
    return linhas_por_tabela


if __name__ == "__main__":
    exportar_gold()
