"""Gráficos reutilizáveis do projeto.

Cada função recebe os dados por argumento, grava a figura em `images/` e devolve
o caminho do arquivo. Nenhuma função lê dados nem treina modelos, e este módulo
não importa de `preprocessing`, `modeling` nem `evaluation`.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src.config import DIRETORIO_IMAGENS

RESOLUCAO_DPI: int = 120


def salvar_figura(figura: Figure, nome_arquivo: str, diretorio: Path = DIRETORIO_IMAGENS) -> Path:
    """Grava a figura como PNG em `diretorio` e a fecha para liberar memória."""
    diretorio.mkdir(parents=True, exist_ok=True)
    caminho = diretorio / nome_arquivo
    figura.tight_layout()
    figura.savefig(caminho, dpi=RESOLUCAO_DPI)
    plt.close(figura)
    return caminho


def histograma(
    dados: pd.DataFrame,
    coluna: str,
    titulo: str,
    nome_arquivo: str,
    diretorio: Path = DIRETORIO_IMAGENS,
    bins: int = 40,
) -> Path:
    """Histograma de uma coluna numérica, ignorando os valores nulos."""
    figura, eixo = plt.subplots(figsize=(7, 4))
    sns.histplot(dados[coluna].dropna(), bins=bins, ax=eixo)
    eixo.set_title(titulo)
    eixo.set_xlabel(coluna)
    eixo.set_ylabel("Frequência")
    return salvar_figura(figura, nome_arquivo, diretorio)


def barras_proporcao_alvo(
    dados: pd.DataFrame,
    categoria: str,
    alvo: str,
    titulo: str,
    nome_arquivo: str,
    diretorio: Path = DIRETORIO_IMAGENS,
) -> Path:
    """Proporção média do alvo (0 ou 1) em cada categoria, com a média geral como linha."""
    proporcoes = dados.groupby(categoria, observed=True)[alvo].mean().sort_values()
    figura, eixo = plt.subplots(figsize=(7, 4))
    sns.barplot(x=proporcoes.values, y=proporcoes.index.astype(str), ax=eixo, orient="h")
    eixo.axvline(dados[alvo].mean(), color="black", linestyle="--", label="média geral")
    eixo.set_title(titulo)
    eixo.set_xlabel(f"Proporção de {alvo}")
    eixo.set_ylabel(categoria)
    eixo.legend()
    return salvar_figura(figura, nome_arquivo, diretorio)


def mapa_correlacao(
    dados: pd.DataFrame,
    colunas: Sequence[str],
    metodo: str,
    titulo: str,
    nome_arquivo: str,
    diretorio: Path = DIRETORIO_IMAGENS,
) -> Path:
    """Mapa de calor da correlação entre colunas numéricas (`pearson` ou `spearman`)."""
    if metodo not in ("pearson", "spearman"):
        raise ValueError(f"Método de correlação desconhecido: {metodo}. Use 'pearson' ou 'spearman'.")
    correlacao = dados[list(colunas)].corr(method=metodo)
    figura, eixo = plt.subplots(figsize=(7, 6))
    sns.heatmap(correlacao, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=eixo)
    eixo.set_title(titulo)
    return salvar_figura(figura, nome_arquivo, diretorio)
