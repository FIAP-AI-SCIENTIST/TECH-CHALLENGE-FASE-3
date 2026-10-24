"""Gráficos do agrupamento de municípios.

Mesma convenção de `graficos.py`: cada função recebe os dados por argumento, grava a
figura em `images/` e devolve o caminho; nada aqui lê dados nem agrupa. Ficam num
módulo à parte para não disputar o fim de `graficos.py` com os gráficos das outras
unidades.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import DIRETORIO_IMAGENS
from src.visualization.graficos import salvar_figura

RUIDO: int = -1


def dispersao_pca(
    projecao_2d: np.ndarray,
    rotulos: np.ndarray,
    titulo: str,
    nome_arquivo: str,
    diretorio: Path = DIRETORIO_IMAGENS,
) -> Path:
    """Os dois primeiros componentes do PCA, coloridos pelo cluster; o ruído (−1) aparece em cinza.

    Falha com `ValueError` se `projecao_2d` não tiver duas colunas ou se `rotulos` tiver
    outro tamanho.
    """
    projecao_2d = np.asarray(projecao_2d)
    rotulos = np.asarray(rotulos)
    if projecao_2d.ndim != 2 or projecao_2d.shape[1] != 2:
        raise ValueError("A projeção precisa de exatamente duas colunas.")
    if len(rotulos) != len(projecao_2d):
        raise ValueError("A projeção e os rótulos precisam ter o mesmo tamanho.")

    figura, eixo = plt.subplots(figsize=(7, 5))
    for cluster in sorted(set(rotulos.tolist()) - {RUIDO}):
        marcados = rotulos == cluster
        eixo.scatter(projecao_2d[marcados, 0], projecao_2d[marcados, 1], s=8, alpha=0.6, label=f"cluster {cluster}")
    ruido = rotulos == RUIDO
    if ruido.any():
        eixo.scatter(projecao_2d[ruido, 0], projecao_2d[ruido, 1], s=14, color="gray", marker="x", label="ruído")
    eixo.set_title(titulo)
    eixo.set_xlabel("Componente principal 1")
    eixo.set_ylabel("Componente principal 2")
    eixo.legend(markerscale=2)
    return salvar_figura(figura, nome_arquivo, diretorio)


def escolha_de_k(
    varredura: pd.DataFrame, titulo: str, nome_arquivo: str, diretorio: Path = DIRETORIO_IMAGENS
) -> Path:
    """Cotovelo (inércia por K) e silhouette médio por K lado a lado, com o K de maior silhouette marcado.

    `varredura` precisa das colunas `k`, `inercia` e `silhouette`. Falha com `ValueError` se faltar alguma.
    """
    faltando = [coluna for coluna in ("k", "inercia", "silhouette") if coluna not in varredura.columns]
    if faltando:
        raise ValueError(f"A varredura precisa das colunas {', '.join(faltando)}.")

    figura, (eixo_cotovelo, eixo_silhouette) = plt.subplots(1, 2, figsize=(10, 4))
    eixo_cotovelo.plot(varredura["k"], varredura["inercia"], marker="o")
    eixo_cotovelo.set_title("Cotovelo")
    eixo_cotovelo.set_xlabel("K")
    eixo_cotovelo.set_ylabel("Inércia (WCSS)")

    eixo_silhouette.plot(varredura["k"], varredura["silhouette"], marker="o")
    melhor = varredura.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    eixo_silhouette.axvline(melhor["k"], color="gray", linestyle="--", label=f"maior silhouette: K={int(melhor['k'])}")
    eixo_silhouette.set_title("Silhouette médio")
    eixo_silhouette.set_xlabel("K")
    eixo_silhouette.set_ylabel("Silhouette")
    eixo_silhouette.legend()
    figura.suptitle(titulo)
    return salvar_figura(figura, nome_arquivo, diretorio)
