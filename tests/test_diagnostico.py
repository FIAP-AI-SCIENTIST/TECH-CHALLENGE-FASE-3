import math

import pandas as pd
import pytest

from src.evaluation.diagnostico import comparar_modelos, diagnosticar_overfitting, melhor_modelo


def test_erro_e_um_menos_a_auc_roc_e_a_razao_compara_validacao_com_treino() -> None:
    diagnostico = diagnosticar_overfitting(auc_treino=0.95, auc_validacao=0.70)
    assert diagnostico["erro_treino"] == pytest.approx(0.05)
    assert diagnostico["erro_validacao"] == pytest.approx(0.30)
    assert diagnostico["razao_erro"] == pytest.approx(6.0)
    assert diagnostico["lacuna_auc"] == pytest.approx(0.25)


def test_razao_de_dois_ou_mais_marca_overfitting() -> None:
    assert diagnosticar_overfitting(0.95, 0.70)["overfitting"] is True
    assert diagnosticar_overfitting(0.75, 0.50)["overfitting"] is True  # razão exatamente 2


def test_razao_menor_que_dois_nao_marca_overfitting() -> None:
    assert diagnosticar_overfitting(0.72, 0.70)["overfitting"] is False


def test_treino_perfeito_tem_razao_infinita_e_marca_overfitting_se_a_validacao_for_pior() -> None:
    diagnostico = diagnosticar_overfitting(1.0, 0.8)
    assert math.isinf(diagnostico["razao_erro"]) and diagnostico["overfitting"] is True


def test_validacao_igual_ou_melhor_que_o_treino_nao_e_overfitting() -> None:
    assert diagnosticar_overfitting(0.6, 0.62)["overfitting"] is False
    assert diagnosticar_overfitting(1.0, 1.0)["overfitting"] is False


def test_limite_da_razao_pode_ser_mudado() -> None:
    assert diagnosticar_overfitting(0.72, 0.70, limite=1.0)["overfitting"] is True


RESULTADOS = {
    "dummy": {"auc_roc_media": 0.5, "auc_roc_desvio": 0.0, "auc_roc_treino_media": 0.5},
    "regressao_logistica": {"auc_roc_media": 0.63, "auc_roc_desvio": 0.01, "auc_roc_treino_media": 0.64},
    "floresta": {"auc_roc_media": 0.66, "auc_roc_desvio": 0.02, "auc_roc_treino_media": 0.99},
}


def test_comparacao_ordena_pela_auc_roc_media_e_traz_o_diagnostico() -> None:
    tabela = comparar_modelos(RESULTADOS)
    assert isinstance(tabela, pd.DataFrame)
    assert tabela["modelo"].tolist() == ["floresta", "regressao_logistica", "dummy"]
    assert tabela.set_index("modelo").loc["floresta", "overfitting"] == True  # noqa: E712
    assert tabela.set_index("modelo").loc["regressao_logistica", "overfitting"] == False  # noqa: E712


def test_melhor_modelo_e_o_de_maior_auc_roc_media_mesmo_com_overfitting_sinalizado() -> None:
    assert melhor_modelo(comparar_modelos(RESULTADOS)) == "floresta"
