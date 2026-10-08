"""Métricas de classificação do projeto.

A classe positiva é o risco (não alfabetizado, não atingiu a meta). A AUC-ROC e a
AUC-PR usam as probabilidades; precisão, recall, F1 e a matriz de confusão usam um
limiar (0,5 por padrão), porque as aulas não ensinam como escolher o limiar.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

LIMIAR_PADRAO: float = 0.5


def calcular_metricas(
    y_verdadeiro: Sequence[int], probabilidades: Sequence[float], limiar: float = LIMIAR_PADRAO
) -> dict[str, object]:
    """AUC-ROC, AUC-PR (precisão média), precisão, recall, F1 e matriz de confusão.

    Falha com `ValueError` se os tamanhos diferem ou se o alvo tem uma só classe.
    """
    y = np.asarray(y_verdadeiro)
    probabilidade = np.asarray(probabilidades, dtype=float)
    if len(y) != len(probabilidade):
        raise ValueError("O alvo e as probabilidades devem ter o mesmo tamanho.")
    if len(np.unique(y)) < 2:
        raise ValueError("O alvo precisa das duas classes para calcular a AUC-ROC.")

    previsto = (probabilidade >= limiar).astype(int)
    verdadeiro_negativo, falso_positivo, falso_negativo, verdadeiro_positivo = confusion_matrix(
        y, previsto, labels=[0, 1]
    ).ravel()
    return {
        "auc_roc": float(roc_auc_score(y, probabilidade)),
        "auc_pr": float(average_precision_score(y, probabilidade)),
        "precisao": float(precision_score(y, previsto, zero_division=0)),
        "recall": float(recall_score(y, previsto, zero_division=0)),
        "f1": float(f1_score(y, previsto, zero_division=0)),
        "matriz_confusao": {
            "verdadeiro_negativo": int(verdadeiro_negativo),
            "falso_positivo": int(falso_positivo),
            "falso_negativo": int(falso_negativo),
            "verdadeiro_positivo": int(verdadeiro_positivo),
        },
    }
