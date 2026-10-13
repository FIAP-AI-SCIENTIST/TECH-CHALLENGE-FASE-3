"""Montagem do `Pipeline` de modelo: pré-processador e estimador num único objeto."""

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.config import ColunasTrilha
from src.preprocessing.transformadores import construir_preprocessador


def montar_pipeline(trilha: ColunasTrilha, estimador: BaseEstimator, escalonar: bool) -> Pipeline:
    """`Pipeline` com os passos `pre_processamento` e `estimador`.

    O pré-processador (não ajustado, de `preprocessing`) e o estimador são ajustados
    juntos por `fit`, então a imputação, o escalonamento e as categorias vêm só do
    treino que o `fit` recebe. Isso vale também dentro de cada dobra da validação
    cruzada, que clona o `Pipeline` inteiro.
    """
    return Pipeline(
        [
            ("pre_processamento", construir_preprocessador(trilha, escalonar)),
            ("estimador", estimador),
        ]
    )
