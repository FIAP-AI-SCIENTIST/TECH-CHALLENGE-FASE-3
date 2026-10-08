import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.config import ColunasTrilha
from src.modeling.pipeline_modelo import montar_pipeline
from src.modeling.validacao_cruzada import gerar_dobras, validar_com_cv
from src.preprocessing.tabelas import TabelaModelagem

TRILHA = ColunasTrilha(categoricas=("cor",), numericas=("x",), proibidas=())


def _tabela(n_municipios: int = 60, alunos_por_municipio: int = 20, semente: int = 3) -> TabelaModelagem:
    gerador = np.random.default_rng(semente)
    grupos = np.repeat(np.arange(n_municipios).astype(str), alunos_por_municipio)
    n = len(grupos)
    x = gerador.normal(size=n)
    cor = gerador.choice(["azul", "verde"], n)
    alvo = (x + (cor == "azul") + gerador.normal(scale=0.7, size=n) > 0.5).astype(int)
    return TabelaModelagem(
        atributos=pd.DataFrame({"x": x, "cor": cor}),
        alvo=pd.Series(alvo, name="alvo"),
        grupos=pd.Series(grupos, name="id_municipio"),
    )


def test_nenhum_municipio_cai_no_treino_e_na_validacao_da_mesma_dobra() -> None:
    tabela = _tabela()
    for treino, validacao in gerar_dobras(tabela.grupos, n_dobras=5):
        assert set(tabela.grupos.iloc[treino]).isdisjoint(set(tabela.grupos.iloc[validacao]))


def test_cada_linha_e_validada_exatamente_uma_vez() -> None:
    tabela = _tabela()
    validadas = np.concatenate([validacao for _, validacao in gerar_dobras(tabela.grupos, n_dobras=5)])
    assert sorted(validadas.tolist()) == list(range(len(tabela.grupos)))


def test_gera_o_numero_pedido_de_dobras() -> None:
    assert len(list(gerar_dobras(_tabela().grupos, n_dobras=4))) == 4


def test_resultado_traz_media_desvio_e_uma_auc_por_dobra() -> None:
    resultado = validar_com_cv(montar_pipeline(TRILHA, LogisticRegression(max_iter=200), escalonar=True), _tabela())
    assert len(resultado["auc_roc_por_dobra"]) == 5
    assert resultado["auc_roc_media"] == pytest.approx(np.mean(resultado["auc_roc_por_dobra"]))
    assert resultado["auc_roc_desvio"] == pytest.approx(np.std(resultado["auc_roc_por_dobra"]))


def test_modelo_com_sinal_passa_de_meio_e_o_dummy_fica_em_meio() -> None:
    tabela = _tabela()
    com_sinal = validar_com_cv(montar_pipeline(TRILHA, LogisticRegression(max_iter=200), escalonar=True), tabela)
    dummy = validar_com_cv(montar_pipeline(TRILHA, DummyClassifier(strategy="prior"), escalonar=False), tabela)
    assert com_sinal["auc_roc_media"] > 0.7
    assert dummy["auc_roc_media"] == pytest.approx(0.5)


def test_traz_a_auc_de_treino_para_o_diagnostico_de_overfitting() -> None:
    resultado = validar_com_cv(montar_pipeline(TRILHA, LogisticRegression(max_iter=200), escalonar=True), _tabela())
    assert 0.5 < resultado["auc_roc_treino_media"] <= 1.0


def test_o_modelo_recebido_nao_e_ajustado_pela_validacao() -> None:
    modelo = montar_pipeline(TRILHA, LogisticRegression(max_iter=200), escalonar=True)
    validar_com_cv(modelo, _tabela())
    assert not hasattr(modelo.named_steps["estimador"], "coef_")


def test_resultado_e_reprodutivel() -> None:
    tabela = _tabela()
    modelo = montar_pipeline(TRILHA, LogisticRegression(max_iter=200), escalonar=True)
    assert validar_com_cv(modelo, tabela) == validar_com_cv(modelo, tabela)


def test_menos_municipios_que_dobras_e_recusado_com_mensagem_clara() -> None:
    with pytest.raises(ValueError, match="municípios"):
        validar_com_cv(montar_pipeline(TRILHA, LogisticRegression(), escalonar=False), _tabela(n_municipios=3), n_dobras=5)
