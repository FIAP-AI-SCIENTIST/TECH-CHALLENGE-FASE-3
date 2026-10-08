from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.visualization.graficos import barras_proporcao_alvo, histograma, mapa_correlacao


@pytest.fixture
def dados() -> pd.DataFrame:
    gerador = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "x": gerador.normal(size=300),
            "y": gerador.normal(size=300),
            "categoria": gerador.choice(["a", "b", "c"], 300),
            "alvo": gerador.integers(0, 2, 300),
        }
    )


def test_histograma_grava_a_figura_e_devolve_o_caminho(dados: pd.DataFrame, tmp_path: Path) -> None:
    caminho = histograma(dados, "x", "Título", "h.png", tmp_path)
    assert caminho == tmp_path / "h.png" and caminho.stat().st_size > 0


def test_histograma_ignora_valores_nulos(dados: pd.DataFrame, tmp_path: Path) -> None:
    dados.loc[:10, "x"] = np.nan
    assert histograma(dados, "x", "Título", "h.png", tmp_path).exists()


def test_barras_do_alvo_gravam_a_figura(dados: pd.DataFrame, tmp_path: Path) -> None:
    assert barras_proporcao_alvo(dados, "categoria", "alvo", "Título", "b.png", tmp_path).stat().st_size > 0


@pytest.mark.parametrize("metodo", ["pearson", "spearman"])
def test_mapa_de_correlacao_aceita_pearson_e_spearman(dados: pd.DataFrame, tmp_path: Path, metodo: str) -> None:
    assert mapa_correlacao(dados, ["x", "y", "alvo"], metodo, "Título", f"{metodo}.png", tmp_path).exists()


def test_mapa_de_correlacao_rejeita_metodo_desconhecido(dados: pd.DataFrame, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="kendall"):
        mapa_correlacao(dados, ["x", "y"], "kendall", "Título", "c.png", tmp_path)


def test_cria_o_diretorio_de_destino_se_nao_existir(dados: pd.DataFrame, tmp_path: Path) -> None:
    destino = tmp_path / "novo" / "imagens"
    assert histograma(dados, "x", "Título", "h.png", destino).parent == destino
