"""Separação treino, validação e teste por município.

O município é a unidade sorteada: todas as linhas de um município vão para o mesmo
conjunto, porque os atributos municipais o identificam e um município em dois
conjuntos vazaria informação do treino para a avaliação. O sorteio usa
`train_test_split` estratificado pela região do município, que é o que as aulas
ensinam, e leva as linhas junto.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SEED
from src.preprocessing.tabelas import TabelaModelagem

FRACOES_PADRAO: tuple[float, float, float] = (0.6, 0.2, 0.2)
ATRIBUTO_ESTRATO: str = "nome_regiao"


def _subconjunto(tabela: TabelaModelagem, municipios: set[str]) -> TabelaModelagem:
    mascara = tabela.grupos.isin(municipios).to_numpy()
    return TabelaModelagem(
        atributos=tabela.atributos[mascara].reset_index(drop=True),
        alvo=tabela.alvo[mascara].reset_index(drop=True),
        grupos=tabela.grupos[mascara].reset_index(drop=True),
    )


def separar_por_municipio(
    tabela: TabelaModelagem,
    fracoes: tuple[float, float, float] = FRACOES_PADRAO,
    semente: int = SEED,
    atributo_estrato: str = ATRIBUTO_ESTRATO,
) -> tuple[TabelaModelagem, TabelaModelagem, TabelaModelagem]:
    """Divide a tabela em treino, validação e teste sem repetir município entre eles.

    Os municípios são sorteados com estratificação pela coluna `atributo_estrato` dos
    atributos, e as proporções valem para municípios; a proporção de linhas fica
    próxima, mas depende do tamanho de cada município.
    """
    if abs(sum(fracoes) - 1.0) > 1e-9:
        raise ValueError(f"As frações devem somar 1, mas somam {sum(fracoes)}.")
    fracao_treino, fracao_validacao, fracao_teste = fracoes

    municipios = (
        pd.DataFrame({"municipio": tabela.grupos, "estrato": tabela.atributos[atributo_estrato].fillna("desconhecido")})
        .drop_duplicates("municipio")
        .reset_index(drop=True)
    )
    treino, resto = train_test_split(
        municipios, train_size=fracao_treino, stratify=municipios["estrato"], random_state=semente
    )
    validacao, teste = train_test_split(
        resto,
        train_size=fracao_validacao / (fracao_validacao + fracao_teste),
        stratify=resto["estrato"],
        random_state=semente,
    )
    return (
        _subconjunto(tabela, set(treino["municipio"])),
        _subconjunto(tabela, set(validacao["municipio"])),
        _subconjunto(tabela, set(teste["municipio"])),
    )
