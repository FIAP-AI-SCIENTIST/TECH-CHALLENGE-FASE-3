"""Trilha T3: agrupamento de municípios por perfil socioeducacional.

Lê o snapshot, monta o perfil por município (U3), padroniza, reduz com PCA e agrupa
com K-Means (a atribuição final), Ward (comparação) e DBSCAN (outliers), seguindo as
regras de `agrupamento.py`. Grava em `reports/` e `images/`:

- `agrupamento_municipios.json`: PCA, varredura de K e de ε, escolhas e métricas;
- `clusters_municipios.csv`: o cluster de cada município (tabela de FR9.3);
- `perfil_clusters.csv` e `clusters_por_regiao.csv`: o que cada cluster é;
- `agrupamento_pca_2d.png` e `agrupamento_escolha_k.png`.

Os clusters são numerados do menor para o maior IDHM médio.

Uso:
    python -m src.modeling.agrupamento_municipios
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DIRETORIO_DADOS, DIRETORIO_IMAGENS, DIRETORIO_RELATORIOS
from src.modeling.agrupamento import (
    ATRIBUTOS_AGRUPAMENTO,
    RUIDO,
    VARIANCIA_PCA,
    Reducao,
    agrupar_dbscan,
    agrupar_kmeans,
    agrupar_ward,
    avaliar,
    clusters_por_regiao,
    detalhar_silhouette,
    escolher_eps,
    escolher_k,
    grade_eps,
    min_pts,
    padronizar_e_reduzir,
    perfil_dos_clusters,
    selecionar_completos,
    varrer_eps,
    varrer_k,
)
from src.preprocessing.loader import carregar_tabela
from src.preprocessing.tabelas import construir_atributos_municipio
from src.visualization.graficos_agrupamento import dispersao_pca, escolha_de_k

logger = logging.getLogger(__name__)

K_COMPLEMENTAR: int = 3


@dataclass(frozen=True)
class ResultadoAgrupamento:
    """O que o agrupamento produz, para relatório e para a resposta de negócio."""

    completos: pd.DataFrame
    excluidos: list[str]
    reducao: Reducao
    varredura_k: pd.DataFrame
    k: int
    rotulos_kmeans: np.ndarray
    rotulos_ward: np.ndarray
    varredura_eps: pd.DataFrame
    eps: float | None
    rotulos_dbscan: np.ndarray | None
    avaliacao: dict[str, dict[str, float] | None]
    rotulos_kmeans_k3: np.ndarray
    rotulos_ward_k3: np.ndarray


def _numerar_por_idhm(completos: pd.DataFrame, rotulos: np.ndarray) -> np.ndarray:
    """Renumera os clusters do menor para o maior IDHM médio; o ruído continua −1."""
    medias = completos.assign(cluster=rotulos).loc[rotulos != RUIDO].groupby("cluster")["idhm"].mean()
    nova_numeracao = {antigo: novo for novo, antigo in enumerate(medias.sort_values().index)}
    return np.array([nova_numeracao.get(r, RUIDO) for r in rotulos])


def _para_json(valor: Any) -> Any:
    """Converte tipos do numpy e do pandas em tipos que o JSON aceita; arredonda os decimais; NaN e infinito viram nulo."""
    if isinstance(valor, dict):
        return {str(chave): _para_json(item) for chave, item in valor.items()}
    if isinstance(valor, (list, tuple, np.ndarray)):
        return [_para_json(item) for item in valor]
    if isinstance(valor, pd.DataFrame):
        return _para_json(valor.to_dict(orient="records"))
    if isinstance(valor, np.generic):
        valor = valor.item()
    if isinstance(valor, float):
        # 6 casas: a ordem das somas em paralelo muda os últimos dígitos (a inércia do K-Means,
        # por exemplo) entre execuções, e o arquivo não deve variar por causa disso
        return round(valor, 6) if math.isfinite(valor) else None
    return valor


def executar(
    diretorio_dados: Path = DIRETORIO_DADOS,
    diretorio_relatorios: Path = DIRETORIO_RELATORIOS,
    diretorio_imagens: Path = DIRETORIO_IMAGENS,
) -> ResultadoAgrupamento:
    """Agrupa os municípios do snapshot em `diretorio_dados` e grava o resultado e as figuras."""
    perfil = construir_atributos_municipio(
        carregar_tabela("dim_municipio", diretorio_dados), carregar_tabela("fact_meta_resultado_municipio", diretorio_dados)
    )
    completos, excluidos = selecionar_completos(perfil)
    logger.info(
        "Perfil: %d municípios, %d com os %d atributos completos e %d excluídos por ausência",
        len(perfil), len(completos), len(ATRIBUTOS_AGRUPAMENTO), len(excluidos),
    )

    reducao = padronizar_e_reduzir(completos[list(ATRIBUTOS_AGRUPAMENTO)])
    espaco = reducao.espaco
    logger.info(
        "PCA: %d componentes explicam %.1f%% da variância (alvo %.0f%%)",
        reducao.n_componentes, 100 * reducao.variancia_acumulada[reducao.n_componentes - 1], 100 * VARIANCIA_PCA,
    )

    varredura_k = varrer_k(espaco)
    k = escolher_k(varredura_k)
    rotulos_kmeans = _numerar_por_idhm(completos, agrupar_kmeans(espaco, k))
    avaliacao_kmeans = {**avaliar(espaco, rotulos_kmeans), **detalhar_silhouette(espaco, rotulos_kmeans)}
    logger.info(
        "K-Means: K=%d (maior silhouette da varredura de %d a %d); silhouette %.4f, Davies-Bouldin %.4f",
        k, varredura_k["k"].min(), varredura_k["k"].max(), avaliacao_kmeans["silhouette"], avaliacao_kmeans["davies_bouldin"],
    )

    rotulos_ward = _numerar_por_idhm(completos, agrupar_ward(espaco, k))
    avaliacao_ward = avaliar(espaco, rotulos_ward)
    logger.info("Ward: K=%d; silhouette %.4f, Davies-Bouldin %.4f", k, avaliacao_ward["silhouette"], avaliacao_ward["davies_bouldin"])

    # Leitura complementar, exploratória: o K=3 não vem da regra do K, e sim de olhar os resultados.
    rotulos_kmeans_k3 = _numerar_por_idhm(completos, agrupar_kmeans(espaco, K_COMPLEMENTAR))
    rotulos_ward_k3 = _numerar_por_idhm(completos, agrupar_ward(espaco, K_COMPLEMENTAR))
    avaliacao_k3 = {**avaliar(espaco, rotulos_kmeans_k3), **detalhar_silhouette(espaco, rotulos_kmeans_k3)}
    concordancia_k3 = float((rotulos_kmeans_k3 == rotulos_ward_k3).mean())
    logger.info(
        "K-Means K=%d (complementar, exploratório): silhouette %.4f, Davies-Bouldin %.4f; concordância com o Ward %.1f%%",
        K_COMPLEMENTAR, avaliacao_k3["silhouette"], avaliacao_k3["davies_bouldin"], 100 * concordancia_k3,
    )

    pts = min_pts(reducao.n_componentes)
    varredura_eps = varrer_eps(espaco, pts, grade_eps(espaco, pts))
    eps = escolher_eps(varredura_eps)
    rotulos_dbscan = None
    avaliacao_dbscan = None
    if eps is None:
        logger.warning("DBSCAN: nenhum ε dá de 2 a 10 clusters com até 10%% de ruído (minPts=%d); sem estrutura útil", pts)
    else:
        rotulos_dbscan = agrupar_dbscan(espaco, eps, pts)
        avaliacao_dbscan = avaliar(espaco, rotulos_dbscan)
        logger.info(
            "DBSCAN: ε=%.4f, minPts=%d, %d clusters e %d outliers (%.1f%%); silhouette %.4f, Davies-Bouldin %.4f",
            eps, pts, avaliacao_dbscan["n_clusters"], int((rotulos_dbscan == RUIDO).sum()), 100 * avaliacao_dbscan["pct_ruido"],
            avaliacao_dbscan["silhouette"], avaliacao_dbscan["davies_bouldin"],
        )

    diretorio_relatorios.mkdir(parents=True, exist_ok=True)
    clusters = completos[["id_municipio", "sigla_uf", "nome_regiao"]].assign(
        cluster=rotulos_kmeans,
        cluster_k3=rotulos_kmeans_k3,
        cluster_ward=rotulos_ward,
        outlier_dbscan=pd.NA if rotulos_dbscan is None else (rotulos_dbscan == RUIDO).astype(int),
    )
    clusters.to_csv(diretorio_relatorios / "clusters_municipios.csv", index=False)
    perfil_dos_clusters(completos, rotulos_kmeans).to_csv(diretorio_relatorios / "perfil_clusters.csv")
    clusters_por_regiao(completos, rotulos_kmeans).to_csv(diretorio_relatorios / "clusters_por_regiao.csv")
    perfil_dos_clusters(completos, rotulos_kmeans_k3).to_csv(diretorio_relatorios / "perfil_clusters_k3.csv")
    clusters_por_regiao(completos, rotulos_kmeans_k3).to_csv(diretorio_relatorios / "clusters_por_regiao_k3.csv")
    dispersao_pca(
        reducao.projecao_2d, rotulos_kmeans, f"Municípios no PCA 2D, clusters do K-Means (K={k})", "agrupamento_pca_2d.png", diretorio_imagens
    )
    escolha_de_k(varredura_k, "Escolha de K no K-Means", "agrupamento_escolha_k.png", diretorio_imagens)
    dispersao_pca(
        reducao.projecao_2d,
        rotulos_kmeans_k3,
        f"Municípios no PCA 2D, clusters do K-Means (K={K_COMPLEMENTAR}, complementar)",
        "agrupamento_pca_2d_k3.png",
        diretorio_imagens,
    )

    avaliacao = {"kmeans": avaliacao_kmeans, "ward": avaliacao_ward, "dbscan": avaliacao_dbscan}
    conteudo = {
        "atributos": list(ATRIBUTOS_AGRUPAMENTO),
        "n_municipios_perfil": len(perfil),
        "n_municipios_agrupados": len(completos),
        "excluidos": excluidos,
        "pca": {
            "variancia_alvo": VARIANCIA_PCA,
            "n_componentes": reducao.n_componentes,
            "variancia_acumulada": reducao.variancia_acumulada,
        },
        "kmeans": {
            "regra": "K de maior silhouette médio",
            "escolhido_pela_regra": True,
            "k_escolhido": k,
            "varredura": varredura_k,
            "tamanhos": pd.Series(rotulos_kmeans).value_counts().sort_index().to_dict(),
        },
        "dbscan": {
            "min_pts": pts,
            "eps_escolhido": eps,
            "varredura": varredura_eps,
            "n_outliers": None if rotulos_dbscan is None else int((rotulos_dbscan == RUIDO).sum()),
        },
        "avaliacao": avaliacao,
        "complementar": {
            "descricao": "leitura exploratória com K=3, escolhida depois de ver os resultados e não pela regra do K",
            "k": K_COMPLEMENTAR,
            "escolhido_pela_regra": False,
            "avaliacao": avaliacao_k3,
            "concordancia_com_ward": concordancia_k3,
            "tamanhos": pd.Series(rotulos_kmeans_k3).value_counts().sort_index().to_dict(),
        },
    }
    (diretorio_relatorios / "agrupamento_municipios.json").write_text(
        json.dumps(_para_json(conteudo), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Resultados gravados em %s e %s", diretorio_relatorios, diretorio_imagens)
    return ResultadoAgrupamento(
        completos, excluidos, reducao, varredura_k, k, rotulos_kmeans, rotulos_ward, varredura_eps, eps, rotulos_dbscan, avaliacao,
        rotulos_kmeans_k3, rotulos_ward_k3,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    executar()
