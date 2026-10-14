from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest

from src.visualization.graficos_agrupamento import dispersao_pca, escolha_de_k

ASSINATURA_PNG = b"\x89PNG\r\n\x1a\n"


def _varredura() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "k": [2, 3, 4, 5],
            "inercia": [900.0, 500.0, 380.0, 330.0],
            "silhouette": [0.41, 0.55, 0.47, 0.40],
            "davies_bouldin": [1.0, 0.7, 0.9, 1.1],
        }
    )


def _projecao(n: int = 60) -> np.ndarray:
    return np.random.default_rng(1).normal(size=(n, 2))


def test_dispersao_grava_um_png_com_conteudo(tmp_path: Path) -> None:
    caminho = dispersao_pca(_projecao(), np.repeat([0, 1, 2], 20), "Clusters no PCA", "pca.png", tmp_path)
    assert caminho == tmp_path / "pca.png"
    assert caminho.read_bytes().startswith(ASSINATURA_PNG)
    altura, largura = mpimg.imread(caminho).shape[:2]
    assert altura > 100 and largura > 100


def test_dispersao_cria_o_diretorio_de_destino(tmp_path: Path) -> None:
    destino = tmp_path / "nao" / "existe"
    assert dispersao_pca(_projecao(), np.zeros(60, dtype=int), "t", "a.png", destino).exists()


def test_dispersao_aceita_ruido_e_um_cluster_so(tmp_path: Path) -> None:
    com_ruido = np.repeat([0, 1, -1], 20)
    assert dispersao_pca(_projecao(), com_ruido, "t", "ruido.png", tmp_path).exists()
    assert dispersao_pca(_projecao(), np.zeros(60, dtype=int), "t", "um.png", tmp_path).exists()


def test_dispersao_recusa_projecao_sem_duas_colunas_ou_rotulos_de_outro_tamanho(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duas colunas"):
        dispersao_pca(np.zeros((10, 3)), np.zeros(10, dtype=int), "t", "x.png", tmp_path)
    with pytest.raises(ValueError, match="mesmo tamanho"):
        dispersao_pca(_projecao(60), np.zeros(59, dtype=int), "t", "x.png", tmp_path)


def test_escolha_de_k_grava_um_png_com_conteudo(tmp_path: Path) -> None:
    caminho = escolha_de_k(_varredura(), "Escolha de K", "k.png", tmp_path)
    assert caminho == tmp_path / "k.png"
    assert caminho.read_bytes().startswith(ASSINATURA_PNG)


def test_escolha_de_k_recusa_tabela_sem_as_colunas(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="silhouette"):
        escolha_de_k(pd.DataFrame({"k": [2, 3], "inercia": [1.0, 0.5]}), "t", "k.png", tmp_path)


def test_escolha_de_k_com_um_unico_k_nao_quebra(tmp_path: Path) -> None:
    assert escolha_de_k(_varredura().head(1), "t", "um.png", tmp_path).exists()
