import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import ColunasTrilha
from src.evaluation.avaliacao_final import conferir_com_relatorio, desenhar_avaliacao
from src.evaluation.metricas import calcular_metricas
from src.modeling.pipeline_modelo import montar_pipeline
from src.evaluation import avaliacao_final

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


def test_conferencia_recusa_auc_pr_diferente() -> None:
    y, probabilidades = _previsoes()
    metricas = calcular_metricas(y, probabilidades)

    relatorio = {
        "teste": {
            **metricas,
            "auc_pr": 0.0,
        }
    }

    with pytest.raises(RuntimeError, match="treino"):
        conferir_com_relatorio(metricas, relatorio)


def test_conferencia_recusa_matriz_de_confusao_diferente() -> None:
    y, probabilidades = _previsoes()
    metricas = calcular_metricas(y, probabilidades)

    matriz_alterada = metricas["matriz_confusao"].copy()
    matriz_alterada["falso_positivo"] += 1

    relatorio = {
        "teste": {
            **metricas,
            "matriz_confusao": matriz_alterada,
        }
    }

    with pytest.raises(RuntimeError, match="treino"):
        conferir_com_relatorio(metricas, relatorio)


def test_desenho_usa_o_mesmo_limiar_na_matriz_e_no_histograma(monkeypatch):
    capturado = {}

    def capturar_matriz(matriz, rotulos, titulo, *args):
        capturado["matriz"] = matriz
        capturado["titulo"] = titulo

    def capturar_histograma(*args, **kwargs):
        capturado["limiar_histograma"] = kwargs["limiar"]

    monkeypatch.setattr(avaliacao_final, "LIMIAR_PADRAO", 0.7)
    monkeypatch.setattr(
        avaliacao_final, "matriz_confusao", capturar_matriz
    )
    monkeypatch.setattr(
        avaliacao_final, "curva_roc", lambda *args: None
    )
    monkeypatch.setattr(
        avaliacao_final,
        "distribuicao_probabilidades",
        capturar_histograma,
    )

    avaliacao_final.desenhar_avaliacao(
        "teste",
        [0, 0, 1, 1],
        [0.1, 0.6, 0.65, 0.9],
        ROTULOS,
        "Modelo",
    )

    assert capturado["matriz"] == {
        "verdadeiro_negativo": 2,
        "falso_positivo": 0,
        "falso_negativo": 1,
        "verdadeiro_positivo": 1,
    }
    assert capturado["limiar_histograma"] == 0.7
    assert "limiar 0,7" in capturado["titulo"]