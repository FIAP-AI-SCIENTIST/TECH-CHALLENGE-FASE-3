import numpy as np
import pytest

from src.visualization.graficos_avaliacao import curva_roc, distribuicao_probabilidades, matriz_confusao

MATRIZ = {"verdadeiro_negativo": 50, "falso_positivo": 10, "falso_negativo": 15, "verdadeiro_positivo": 25}
ROTULOS = ("Atingiu a meta", "Não atingiu a meta")


def _previsoes(n: int = 200, semente: int = 1) -> tuple[np.ndarray, np.ndarray]:
    gerador = np.random.default_rng(semente)
    y = gerador.integers(0, 2, n)
    probabilidades = np.clip(0.3 * y + gerador.uniform(0, 0.7, n), 0, 1)
    return y, probabilidades


def test_matriz_de_confusao_grava_png(tmp_path) -> None:
    caminho = matriz_confusao(MATRIZ, ROTULOS, "Matriz", "matriz.png", tmp_path)
    assert caminho == tmp_path / "matriz.png"
    assert caminho.stat().st_size > 0


def test_matriz_de_confusao_recusa_chave_faltando(tmp_path) -> None:
    with pytest.raises(ValueError, match="verdadeiro_positivo"):
        matriz_confusao({"verdadeiro_negativo": 1, "falso_positivo": 1, "falso_negativo": 1}, ROTULOS, "M", "m.png", tmp_path)


def test_curva_roc_grava_png(tmp_path) -> None:
    y, probabilidades = _previsoes()
    caminho = curva_roc(y, probabilidades, "Modelo", "ROC", "roc.png", tmp_path)
    assert caminho.exists() and caminho.stat().st_size > 0


def test_distribuicao_das_probabilidades_grava_png(tmp_path) -> None:
    y, probabilidades = _previsoes()
    caminho = distribuicao_probabilidades(y, probabilidades, ROTULOS, "Distribuição", "dist.png", tmp_path)
    assert caminho.exists() and caminho.stat().st_size > 0


def test_graficos_recusam_tamanhos_diferentes(tmp_path) -> None:
    y, probabilidades = _previsoes()
    with pytest.raises(ValueError, match="mesmo tamanho"):
        curva_roc(y, probabilidades[:-1], "Modelo", "ROC", "roc.png", tmp_path)
    with pytest.raises(ValueError, match="mesmo tamanho"):
        distribuicao_probabilidades(y[:-1], probabilidades, ROTULOS, "D", "d.png", tmp_path)
