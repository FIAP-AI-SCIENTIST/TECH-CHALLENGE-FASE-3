"""Tabelas de modelagem das trilhas supervisionadas, a partir do snapshot da Gold.

Cada construtor devolve os atributos, o alvo (classe positiva = o risco) e o
grupo da separação (o município). Nada aqui ajusta estatísticas: imputação,
escalonamento e codificação são do pré-processador, ajustado só com o treino.
"""

from dataclasses import dataclass

import pandas as pd

from src.config import ALVO_ALUNO, COLUNAS_T1
from src.preprocessing.atributos import validar_atributos

ALVO_NAO_ALFABETIZADO: str = "nao_alfabetizado"


@dataclass(frozen=True)
class TabelaModelagem:
    """Atributos, alvo e grupo (município) alinhados pelo mesmo índice."""

    atributos: pd.DataFrame
    alvo: pd.Series
    grupos: pd.Series


def construir_tabela_aluno(alunos: pd.DataFrame, municipios: pd.DataFrame) -> TabelaModelagem:
    """Tabela da trilha T1: só quem fez a prova, alvo `nao_alfabetizado` (1 = não alfabetizado).

    Quem deixou o caderno em branco tem `alfabetizado` falso por ausência, não por
    leitura, e fica fora. Os atributos municipais vêm de `dim_municipio` pelo
    `id_municipio`, que é devolvido como grupo e nunca como atributo.
    """
    validar_atributos(COLUNAS_T1.atributos, COLUNAS_T1)
    com_prova = alunos[alunos["preenchimento_caderno"]]
    colunas_municipio = ["id_municipio", *(c for c in COLUNAS_T1.atributos if c in municipios.columns)]
    tabela = com_prova.merge(municipios[colunas_municipio], on="id_municipio", how="left", validate="many_to_one")
    return TabelaModelagem(
        atributos=tabela[list(COLUNAS_T1.atributos)],
        alvo=(~tabela[ALVO_ALUNO]).astype(int).rename(ALVO_NAO_ALFABETIZADO),
        grupos=tabela["id_municipio"],
    )
