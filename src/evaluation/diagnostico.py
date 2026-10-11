"""Diagnóstico de overfitting e comparação de modelos.

A regra vem da Aula de Otimização 5: se o erro de treino é pelo menos 2 vezes menor
que o de outro conjunto, há overfitting. Aqui o erro é `1 - AUC-ROC` e o outro
conjunto é a validação, nunca o teste, que fica fechado até a avaliação final.
"""

import math

import pandas as pd

LIMITE_RAZAO_PADRAO: float = 2.0


def diagnosticar_overfitting(
    auc_treino: float, auc_validacao: float, limite: float = LIMITE_RAZAO_PADRAO
) -> dict[str, object]:
    """Erros, razão entre eles, lacuna de AUC e a marca `overfitting` (razão >= `limite`)."""
    erro_treino = 1.0 - auc_treino
    erro_validacao = 1.0 - auc_validacao
    if erro_treino > 0:
        razao = erro_validacao / erro_treino
    else:
        razao = math.inf if erro_validacao > 0 else 1.0
    return {
        "erro_treino": erro_treino,
        "erro_validacao": erro_validacao,
        "razao_erro": razao,
        "lacuna_auc": auc_treino - auc_validacao,
        "overfitting": bool(razao >= limite),
    }


def comparar_modelos(resultados: dict[str, dict[str, object]], limite: float = LIMITE_RAZAO_PADRAO) -> pd.DataFrame:
    """Tabela dos modelos ordenada pela AUC-ROC média de validação, com o diagnóstico de cada um.

    `resultados` mapeia o nome do modelo ao dicionário de `validar_com_cv`.
    """
    linhas = []
    for modelo, resultado in resultados.items():
        diagnostico = diagnosticar_overfitting(
            float(resultado["auc_roc_treino_media"]), float(resultado["auc_roc_media"]), limite
        )
        linhas.append(
            {
                "modelo": modelo,
                "auc_roc_media": float(resultado["auc_roc_media"]),
                "auc_roc_desvio": float(resultado["auc_roc_desvio"]),
                "auc_roc_treino_media": float(resultado["auc_roc_treino_media"]),
                "lacuna_auc": diagnostico["lacuna_auc"],
                "razao_erro": diagnostico["razao_erro"],
                "overfitting": diagnostico["overfitting"],
            }
        )
    return pd.DataFrame(linhas).sort_values("auc_roc_media", ascending=False).reset_index(drop=True)


def melhor_modelo(comparacao: pd.DataFrame) -> str:
    """Nome do modelo de maior AUC-ROC média; a marca de overfitting é informada, não elimina."""
    return str(comparacao.sort_values("auc_roc_media", ascending=False)["modelo"].iloc[0])
