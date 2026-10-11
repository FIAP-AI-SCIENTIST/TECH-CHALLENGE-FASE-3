"""Validação cruzada agrupada por município.

Cada dobra de validação reúne municípios inteiros que não aparecem no treino da
mesma dobra. `GroupKFold` é exigido pelo enunciado para separar por grupo e não é
coberto nas aulas; ele não estratifica o alvo, o que aqui pesa pouco porque as
classes são equilibradas.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupKFold, cross_validate

from src.preprocessing.tabelas import TabelaModelagem

N_DOBRAS_PADRAO: int = 5


def gerar_dobras(grupos: pd.Series, n_dobras: int = N_DOBRAS_PADRAO) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Pares (índices de treino, índices de validação), sem município repetido na mesma dobra."""
    if grupos.nunique() < n_dobras:
        raise ValueError(f"São necessários pelo menos {n_dobras} municípios para {n_dobras} dobras; há {grupos.nunique()}.")
    posicoes = np.arange(len(grupos))
    yield from GroupKFold(n_splits=n_dobras).split(posicoes, groups=grupos)


def validar_com_cv(
    modelo: BaseEstimator, tabela: TabelaModelagem, n_dobras: int = N_DOBRAS_PADRAO
) -> dict[str, object]:
    """AUC-ROC por dobra e sua média e desvio, mais a média da AUC-ROC de treino.

    O `modelo` é clonado a cada dobra, então o pré-processamento é reajustado só com
    o treino da dobra e o objeto recebido não é alterado.
    """
    dobras = list(gerar_dobras(tabela.grupos, n_dobras))
    resultado = cross_validate(
        modelo,
        tabela.atributos,
        tabela.alvo,
        cv=dobras,
        scoring="roc_auc",
        return_train_score=True,
    )
    por_dobra = [float(valor) for valor in resultado["test_score"]]
    return {
        "auc_roc_por_dobra": por_dobra,
        "auc_roc_media": float(np.mean(por_dobra)),
        "auc_roc_desvio": float(np.std(por_dobra)),
        "auc_roc_treino_media": float(np.mean(resultado["train_score"])),
    }
