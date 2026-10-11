import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import ColunasTrilha
from src.modeling.pipeline_modelo import montar_pipeline
from src.modeling.protocolo import buscar_hiperparametros, executar_baselines
from src.modeling.validacao_cruzada import validar_com_cv
from src.preprocessing.tabelas import TabelaModelagem

TRILHA = ColunasTrilha(categoricas=("cor",), numericas=("x",), proibidas=())
GRADE = {"estimador__C": [0.001, 1.0]}


def _tabela(n_municipios: int = 60, alunos_por_municipio: int = 20, semente: int = 3) -> TabelaModelagem:
    gerador = np.random.default_rng(semente)
    grupos = np.repeat(np.arange(n_municipios).astype(str), alunos_por_municipio)
    n = len(grupos)
    x = gerador.normal(size=n)
    cor = gerador.choice(["azul", "verde"], n)
    alvo = (x + (cor == "azul") + gerador.normal(scale=0.7, size=n) > 0.5).astype(int)
    return TabelaModelagem(pd.DataFrame({"x": x, "cor": cor}), pd.Series(alvo, name="alvo"), pd.Series(grupos, name="id_municipio"))


def _modelo():
    return montar_pipeline(TRILHA, LogisticRegression(max_iter=200), escalonar=True)


def test_baselines_trazem_dummy_em_meio_e_regressao_logistica_acima() -> None:
    resultados = executar_baselines(TRILHA, _tabela())
    assert set(resultados) == {"dummy", "regressao_logistica"}
    assert resultados["dummy"]["auc_roc_media"] == pytest.approx(0.5)
    assert resultados["regressao_logistica"]["auc_roc_media"] > 0.7


def test_busca_em_grade_devolve_o_melhor_conjunto_dentro_da_grade_e_um_modelo_ajustado() -> None:
    resultado = buscar_hiperparametros(_modelo(), GRADE, _tabela())
    assert resultado["melhores_parametros"]["estimador__C"] in GRADE["estimador__C"]
    assert hasattr(resultado["modelo"].named_steps["estimador"], "coef_")


def test_busca_escolhe_o_valor_de_c_que_a_validacao_cruzada_prefere() -> None:
    tabela = _tabela()
    resultado = buscar_hiperparametros(_modelo(), GRADE, tabela)
    pontuacoes = {}
    for c in GRADE["estimador__C"]:
        modelo = _modelo().set_params(estimador__C=c)
        pontuacoes[c] = validar_com_cv(modelo, tabela)["auc_roc_media"]
    assert resultado["melhores_parametros"]["estimador__C"] == max(pontuacoes, key=pontuacoes.get)
    assert resultado["auc_roc_cv"] == pytest.approx(max(pontuacoes.values()))


def test_busca_aleatoria_e_reprodutivel_pela_semente() -> None:
    grade = {"estimador__C": [0.001, 0.01, 0.1, 1.0, 10.0]}
    primeira = buscar_hiperparametros(_modelo(), grade, _tabela(), busca="aleatoria", n_iteracoes=3)
    segunda = buscar_hiperparametros(_modelo(), grade, _tabela(), busca="aleatoria", n_iteracoes=3)
    assert primeira["melhores_parametros"] == segunda["melhores_parametros"]


def test_tipo_de_busca_desconhecido_e_recusado() -> None:
    with pytest.raises(ValueError, match="grade.*aleatoria"):
        buscar_hiperparametros(_modelo(), GRADE, _tabela(), busca="bayesiana")


def test_o_modelo_original_nao_e_ajustado_pela_busca() -> None:
    modelo = _modelo()
    buscar_hiperparametros(modelo, GRADE, _tabela())
    assert not hasattr(modelo.named_steps["estimador"], "coef_")
