"""Interpretação dos modelos finais: importância nativa, por permutação e SHAP.

As três respondem à mesma pergunta (quais atributos pesam na previsão) por caminhos
diferentes, e a interpretação lê as três juntas:

- **Importância nativa:** os coeficientes da Regressão Logística (com sinal, na escala
  padronizada) ou a `feature_importances_` das árvores. Fica no espaço transformado
  (uma linha por coluna do One-Hot).
- **Permutação** (`permutation_importance`): a queda da AUC-ROC quando uma coluna
  original é embaralhada. Mede o atributo inteiro, inclusive todas as suas colunas
  One-Hot, e não depende do tipo de modelo.
- **SHAP:** a contribuição de cada atributo para cada previsão, em log-odds. Recomendado
  pelo enunciado e não coberto nas aulas.

Tudo descreve associação nos dados, não o efeito de uma política.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import SEED

PREFIXOS_TRANSFORMADOS: tuple[str, ...] = ("numericos__", "categoricos__")
# Linhas de fundo do `LinearExplainer` (a média e a covariância de referência).
MAXIMO_FUNDO_SHAP: int = 1_000


def atributo_original(nome_transformado: str, originais: Sequence[str]) -> str:
    """Coluna original de um nome do pré-processador (`categoricos__sigla_uf_SP` -> `sigla_uf`).

    Usa o nome original mais longo que casa, para `idhm` não capturar `idhm_renda`.
    """
    nome = nome_transformado
    for prefixo in PREFIXOS_TRANSFORMADOS:
        if nome.startswith(prefixo):
            nome = nome[len(prefixo) :]
            break
    candidatos = [original for original in originais if nome == original or nome.startswith(f"{original}_")]
    if not candidatos:
        raise ValueError(f"Atributo transformado desconhecido: {nome_transformado}.")
    return max(candidatos, key=len)


def importancia_nativa(modelo: Pipeline) -> pd.DataFrame:
    """Coeficientes (Regressão Logística) ou `feature_importances_` (árvores), do maior para o menor em módulo."""
    estimador = modelo.named_steps["estimador"]
    nomes = modelo.named_steps["pre_processamento"].get_feature_names_out()
    if isinstance(estimador, LogisticRegression):
        valores = estimador.coef_[0]
    else:
        valores = estimador.feature_importances_
    tabela = pd.DataFrame({"atributo_transformado": nomes, "importancia": valores})
    ordem = tabela["importancia"].abs().sort_values(ascending=False).index
    return tabela.loc[ordem].reset_index(drop=True)


def importancia_por_permutacao(
    modelo: Pipeline, atributos: pd.DataFrame, alvo: pd.Series, n_repeticoes: int = 10, semente: int = SEED
) -> pd.DataFrame:
    """Queda média (e desvio) da AUC-ROC ao embaralhar cada coluna original, da maior para a menor."""
    resultado = permutation_importance(
        modelo, atributos, alvo, scoring="roc_auc", n_repeats=n_repeticoes, random_state=semente
    )
    tabela = pd.DataFrame(
        {
            "atributo": atributos.columns,
            "queda_auc_media": resultado.importances_mean,
            "queda_auc_desvio": resultado.importances_std,
        }
    )
    return tabela.sort_values("queda_auc_media", ascending=False).reset_index(drop=True)


def valores_shap(modelo: Pipeline, atributos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valores SHAP (log-odds) de cada linha e os atributos transformados que os geraram.

    A Regressão Logística usa o `LinearExplainer`, com até 1.000 linhas dos próprios dados,
    sorteadas com a semente do projeto, como fundo; as
    árvores usam o `TreeExplainer`. Os dois explicam o estimador sobre a saída do
    pré-processador, então as colunas são as do espaço transformado.
    """
    transformados = modelo.named_steps["pre_processamento"].transform(atributos)
    estimador = modelo.named_steps["estimador"]
    if isinstance(estimador, LogisticRegression):
        fundo = transformados.sample(n=min(len(transformados), MAXIMO_FUNDO_SHAP), random_state=SEED)
        explicador = shap.LinearExplainer(estimador, shap.maskers.Independent(fundo, max_samples=len(fundo)))
    else:
        explicador = shap.TreeExplainer(estimador)
    valores = np.asarray(explicador.shap_values(transformados))
    if valores.ndim == 3:  # alguns explicadores devolvem uma matriz por classe
        valores = valores[..., 1]
    return pd.DataFrame(valores, columns=transformados.columns, index=transformados.index), transformados


def agregar_por_atributo(shap_valores: pd.DataFrame, originais: Sequence[str]) -> pd.DataFrame:
    """Média do |SHAP| por atributo original, somando antes as colunas One-Hot de cada linha.

    Somar por linha antes do módulo respeita a aditividade do SHAP: a contribuição de
    `sigla_uf` numa previsão é a soma das contribuições das suas colunas One-Hot.
    """
    mapa = {coluna: atributo_original(coluna, originais) for coluna in shap_valores.columns}
    por_atributo = shap_valores.T.groupby(mapa).sum().T
    tabela = por_atributo.abs().mean().rename("shap_medio_absoluto").rename_axis("atributo").reset_index()
    return tabela.sort_values("shap_medio_absoluto", ascending=False).reset_index(drop=True)
