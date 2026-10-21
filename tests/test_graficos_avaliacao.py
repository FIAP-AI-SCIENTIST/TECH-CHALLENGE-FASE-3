import numpy as np
import pytest
import matplotlib.pyplot as plt

from src.visualization import graficos_avaliacao
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


@pytest.mark.parametrize(
    "matriz",
    [
        {
            "verdadeiro_negativo": 0,
            "falso_positivo": 0,
            "falso_negativo": 2,
            "verdadeiro_positivo": 3,
        },
        {
            "verdadeiro_negativo": 3,
            "falso_positivo": 2,
            "falso_negativo": 0,
            "verdadeiro_positivo": 0,
        },
    ],
)
def test_matriz_recusa_classe_real_ausente(matriz, tmp_path) -> None:
    with pytest.raises(ValueError, match="classes"):
        matriz_confusao(matriz, ROTULOS, "Matriz", "matriz.png", tmp_path)


@pytest.mark.parametrize("classe", [0, 1])
def test_roc_recusa_classe_real_ausente(classe, tmp_path) -> None:
    y = [classe, classe, classe]
    probabilidades = [0.1, 0.5, 0.9]

    with pytest.raises(ValueError, match="classes"):
        curva_roc(y, probabilidades, "Modelo", "ROC", "roc.png", tmp_path)


def test_matriz_desenha_valores_percentuais_e_classes_corretos(monkeypatch):
    figuras = []

    def capturar_figura(figura, nome_arquivo, diretorio):
        figuras.append(figura)
        return diretorio / nome_arquivo

    monkeypatch.setattr(
        graficos_avaliacao, "salvar_figura", capturar_figura
    )

    try:
        matriz_confusao(MATRIZ, ROTULOS, "Matriz", "matriz.png")

        eixo = figuras[0].axes[0]

        np.testing.assert_allclose(
            eixo.images[0].get_array(),
            [[50 / 60, 10 / 60], [15 / 40, 25 / 40]],
        )

        textos = {
            texto.get_position(): texto.get_text()
            for texto in eixo.texts
        }
        assert textos == {
            (0, 0): "50\n(83,3%)",
            (1, 0): "10\n(16,7%)",
            (0, 1): "15\n(37,5%)",
            (1, 1): "25\n(62,5%)",
        }

        assert [item.get_text() for item in eixo.get_xticklabels()] == [
            f"Previsto:\n{ROTULOS[0]}",
            f"Previsto:\n{ROTULOS[1]}",
        ]
        assert [item.get_text() for item in eixo.get_yticklabels()] == [
            f"Real:\n{ROTULOS[0]}",
            f"Real:\n{ROTULOS[1]}",
        ]
    finally:
        for figura in figuras:
            plt.close(figura)


def test_roc_desenha_curva_e_referencia_corretas(monkeypatch):
    figuras = []

    def capturar_figura(figura, nome_arquivo, diretorio):
        figuras.append(figura)
        return diretorio / nome_arquivo

    monkeypatch.setattr(
        graficos_avaliacao, "salvar_figura", capturar_figura
    )

    try:
        curva_roc(
            [0, 0, 1, 1],
            [0.1, 0.7, 0.4, 0.9],
            "Modelo",
            "ROC",
            "roc.png",
        )

        eixo = figuras[0].axes[0]
        curva, referencia = eixo.lines

        np.testing.assert_allclose(
            curva.get_xdata(), [0, 0, 0.5, 0.5, 1]
        )
        np.testing.assert_allclose(
            curva.get_ydata(), [0, 0.5, 0.5, 1, 1]
        )
        assert curva.get_label() == "Modelo (AUC-ROC 0,750)"

        np.testing.assert_allclose(referencia.get_xdata(), [0, 1])
        np.testing.assert_allclose(referencia.get_ydata(), [0, 1])
        assert referencia.get_linestyle() == "--"

        assert eixo.get_xlabel() == "Taxa de falsos positivos"
        assert eixo.get_ylabel() == "Taxa de verdadeiros positivos (recall)"
    finally:
        for figura in figuras:
            plt.close(figura)


def test_histograma_normaliza_classes_e_marca_limiar(monkeypatch):
    figuras = []

    def capturar_figura(figura, nome_arquivo, diretorio):
        figuras.append(figura)
        return diretorio / nome_arquivo

    monkeypatch.setattr(
        graficos_avaliacao, "salvar_figura", capturar_figura
    )

    try:
        distribuicao_probabilidades(
            [0, 0, 0, 1],
            [0.1, 0.2, 0.8, 0.9],
            ROTULOS,
            "Distribuição",
            "distribuicao.png",
            limiar=0.6,
            bins=2,
        )

        eixo = figuras[0].axes[0]
        alturas = [barra.get_height() for barra in eixo.patches]

        np.testing.assert_allclose(
            alturas,
            [2 / 3, 1 / 3, 0, 1],
        )

        assert len(eixo.lines) == 1
        linha_limiar = eixo.lines[0]
        np.testing.assert_allclose(
            linha_limiar.get_xdata(), [0.6, 0.6]
        )
        assert linha_limiar.get_linestyle() == "--"
        assert linha_limiar.get_label() == "Limiar 0,6"

        assert eixo.get_ylabel() == "Proporção dentro da classe"
        assert eixo.get_xlim() == (0, 1)
    finally:
        for figura in figuras:
            plt.close(figura)


@pytest.mark.parametrize("grafico", ["matriz", "roc", "histograma"])
def test_escalas_usam_formatacao_coerente(monkeypatch, grafico):
    figuras = []

    def capturar_figura(figura, nome_arquivo, diretorio):
        figuras.append(figura)
        return diretorio / nome_arquivo

    monkeypatch.setattr(
        graficos_avaliacao, "salvar_figura", capturar_figura
    )

    try:
        if grafico == "matriz":
            matriz_confusao(MATRIZ, ROTULOS, "Matriz", "matriz.png")
            barra = figuras[0].axes[1]
            formatador = barra.yaxis.get_major_formatter()
            assert formatador(0.2, 0) == "20,0%"
        else:
            y = [0, 0, 1, 1]
            probabilidades = [0.1, 0.7, 0.4, 0.9]

            if grafico == "roc":
                curva_roc(y, probabilidades, "Modelo", "ROC", "roc.png")
            else:
                distribuicao_probabilidades(
                    y, probabilidades, ROTULOS, "Distribuição", "dist.png"
                )

            eixo = figuras[0].axes[0]
            assert eixo.xaxis.get_major_formatter()(0.2, 0) == "0,2"

            esperado_y = "0,20" if grafico == "histograma" else "0,2"
            assert eixo.yaxis.get_major_formatter()(0.2, 0) == esperado_y

            if grafico == "histograma":
                assert eixo.yaxis.get_major_formatter()(0.02, 0) == "0,02"
                assert eixo.yaxis.get_major_formatter()(0.04, 0) == "0,04"
    finally:
        for figura in figuras:
            plt.close(figura)
