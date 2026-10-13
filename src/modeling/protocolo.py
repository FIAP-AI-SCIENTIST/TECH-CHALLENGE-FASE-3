"""Protocolo genérico de modelagem, parametrizado pelo alvo e pela trilha.

Serve às trilhas T1 (aluno) e T2 (município): cada uma passa a própria tabela e as
mesmas funções fazem os baselines e a busca de hiperparâmetros. A busca é feita só
com a tabela de treino recebida e com as dobras agrupadas por município; o teste
nunca passa por aqui. `DummyClassifier` é exigido pelo enunciado e não é coberto
nas aulas; a busca em grade e a aleatória são as ensinadas (Aula: Otimização 2).
"""

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.config import SEED, ColunasTrilha
from src.modeling.pipeline_modelo import montar_pipeline
from src.modeling.validacao_cruzada import N_DOBRAS_PADRAO, gerar_dobras, validar_com_cv
from src.preprocessing.tabelas import TabelaModelagem

ITERACOES_PADRAO: int = 10


def executar_baselines(
    trilha: ColunasTrilha, tabela: TabelaModelagem, n_dobras: int = N_DOBRAS_PADRAO
) -> dict[str, dict[str, object]]:
    """Validação cruzada agrupada do `DummyClassifier` e da Regressão Logística com hiperparâmetros padrão."""
    dummy = montar_pipeline(trilha, DummyClassifier(strategy="prior"), escalonar=False)
    regressao = montar_pipeline(trilha, LogisticRegression(max_iter=200, random_state=SEED), escalonar=True)
    return {
        "dummy": validar_com_cv(dummy, tabela, n_dobras),
        "regressao_logistica": validar_com_cv(regressao, tabela, n_dobras),
    }


def buscar_hiperparametros(
    modelo: BaseEstimator,
    grade: dict[str, list[Any]],
    tabela: TabelaModelagem,
    busca: str = "grade",
    n_iteracoes: int = ITERACOES_PADRAO,
    n_dobras: int = N_DOBRAS_PADRAO,
    semente: int = SEED,
) -> dict[str, Any]:
    """Busca em grade (`grade`) ou aleatória (`aleatoria`) pela AUC-ROC de validação cruzada agrupada.

    Devolve o modelo reajustado com os melhores parâmetros (sobre toda a `tabela`),
    esses parâmetros e a AUC-ROC média da validação cruzada. O `modelo` recebido não
    é alterado.
    """
    if busca not in ("grade", "aleatoria"):
        raise ValueError(f"Tipo de busca desconhecido: {busca}. Use 'grade' ou 'aleatoria'.")
    dobras = list(gerar_dobras(tabela.grupos, n_dobras))
    if busca == "grade":
        procura = GridSearchCV(modelo, grade, scoring="roc_auc", cv=dobras, refit=True)
    else:
        procura = RandomizedSearchCV(
            modelo, grade, n_iter=n_iteracoes, scoring="roc_auc", cv=dobras, refit=True, random_state=semente
        )
    procura.fit(tabela.atributos, tabela.alvo)
    return {
        "modelo": procura.best_estimator_,
        "melhores_parametros": procura.best_params_,
        "auc_roc_cv": float(procura.best_score_),
    }
