"""Ranking de risco dos municípios, com probabilidades fora da dobra.

O risco de cada município vem de um modelo que não o viu: a tabela é dividida em
dobras por município (`GroupKFold`), e cada dobra é prevista por uma cópia do modelo
ajustada nas demais. O modelo final do treino viu 80% dos municípios, e usá-lo no
ranking daria a eles um risco otimista, decorado. A configuração do modelo (o
algoritmo e os hiperparâmetros) é a escolhida pelo treino; só o ajuste é refeito.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.config import SEED
from src.modeling.treino import amostrar
from src.modeling.validacao_cruzada import N_DOBRAS_PADRAO, gerar_dobras
from src.preprocessing.tabelas import TabelaModelagem

N_QUINTIS: int = 5


def probabilidades_fora_da_dobra(
    modelo: Pipeline,
    tabela: TabelaModelagem,
    n_dobras: int = N_DOBRAS_PADRAO,
    max_linhas_treino: int | None = None,
    semente: int = SEED,
) -> pd.Series:
    """Probabilidade da classe positiva de cada linha, prevista por um modelo ajustado sem o seu município.

    `max_linhas_treino` sorteia uma amostra do treino de cada dobra, para a trilha do
    aluno; a previsão é sempre feita em todas as linhas da dobra.
    """
    probabilidades = np.empty(len(tabela.alvo))
    for indices_treino, indices_validacao in gerar_dobras(tabela.grupos, n_dobras):
        treino = TabelaModelagem(
            atributos=tabela.atributos.iloc[indices_treino].reset_index(drop=True),
            alvo=tabela.alvo.iloc[indices_treino].reset_index(drop=True),
            grupos=tabela.grupos.iloc[indices_treino].reset_index(drop=True),
        )
        if max_linhas_treino is not None:
            treino = amostrar(treino, max_linhas_treino, semente)
        ajustado = clone(modelo).fit(treino.atributos, treino.alvo)
        probabilidades[indices_validacao] = ajustado.predict_proba(tabela.atributos.iloc[indices_validacao])[:, 1]
    return pd.Series(probabilidades, index=tabela.alvo.index, name="probabilidade")


def taxa_por_municipio(valores: pd.Series, grupos: pd.Series) -> pd.Series:
    """Média dos `valores` (alvo ou probabilidade) em cada município."""
    return valores.groupby(grupos.to_numpy()).mean()


def montar_ranking(risco: pd.Series, municipios: pd.DataFrame, nome_risco: str) -> pd.DataFrame:
    """Uma linha por município, do maior para o menor risco, com a posição geral, na UF e o quintil.

    `risco` é indexado por `id_municipio`; `municipios` traz `id_municipio`, `nome` e
    `sigla_uf`. O quintil 5 é o de maior risco.
    """
    sem_cadastro = sorted(set(risco.index) - set(municipios["id_municipio"]))
    if sem_cadastro:
        raise ValueError(f"Municípios sem cadastro: {', '.join(map(str, sem_cadastro))}.")
    ranking = (
        risco.rename(nome_risco)
        .rename_axis("id_municipio")
        .reset_index()
        .merge(municipios, on="id_municipio", how="left", validate="one_to_one")
        .sort_values([nome_risco, "id_municipio"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranking.insert(0, "posicao", np.arange(1, len(ranking) + 1))
    ranking["posicao_na_uf"] = ranking.groupby("sigla_uf").cumcount() + 1
    ordem_crescente = ranking[nome_risco].rank(method="first")
    ranking["quintil_risco"] = pd.qcut(ordem_crescente, N_QUINTIS, labels=range(1, N_QUINTIS + 1)).astype(int)
    return ranking
