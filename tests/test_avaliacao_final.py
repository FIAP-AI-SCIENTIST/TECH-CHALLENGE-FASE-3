import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import ColunasTrilha
from src.evaluation.avaliacao_final import conferir_com_relatorio, desenhar_avaliacao
from src.evaluation.metricas import calcular_metricas
from src.modeling.pipeline_modelo import montar_pipeline

ROTULOS = ("Alfabetizado", "Não alfabetizado")


def _previsoes(n: int = 300, semente: int = 2) -> tuple[pd.Series, np.ndarray]:
    gerador = np.random.default_rng(semente)
    x = gerador.normal(size=n)
    y = (x + gerador.normal(scale=1.0, size=n) > 0).astype(int)
    trilha = ColunasTrilha(categoricas=(), numericas=("x",), proibidas=())
    atributos = pd.DataFrame({"x": x})
    modelo = montar_pipeline(trilha, LogisticRegression(), escalonar=True).fit(atributos, y)
    return pd.Series(y), modelo.predict_proba(atributos)[:, 1]


def test_desenha_os_tres_graficos_com_o_prefixo_da_trilha(tmp_path) -> None:
    y, probabilidades = _previsoes()
    caminhos = desenhar_avaliacao("t9", y, probabilidades, ROTULOS, "Modelo", tmp_path)
    assert {caminho.name for caminho in caminhos} == {
        "avaliacao_matriz_confusao_t9.png",
        "avaliacao_curva_roc_t9.png",
        "avaliacao_probabilidades_t9.png",
    }
    assert all(caminho.stat().st_size > 0 for caminho in caminhos)


def test_conferencia_aceita_metricas_iguais_as_do_relatorio() -> None:
    y, probabilidades = _previsoes()
    relatorio = {"teste": calcular_metricas(y, probabilidades)}
    conferir_com_relatorio(calcular_metricas(y, probabilidades), relatorio)


def test_conferencia_recusa_modelo_desatualizado() -> None:
    y, probabilidades = _previsoes()
    relatorio = {"teste": {**calcular_metricas(y, probabilidades), "auc_roc": 0.99}}
    with pytest.raises(RuntimeError, match="treino"):
        conferir_com_relatorio(calcular_metricas(y, probabilidades), relatorio)
