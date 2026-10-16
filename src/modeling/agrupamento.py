"""Agrupamento de municípios por perfil socioeducacional (trilha T3).

Aprendizado não supervisionado: não há alvo, então não há separação treino/teste e o
escalonador e o PCA ajustam em todos os municípios completos. O pipeline é o das
Aulas de Não supervisionados 1 e 6: padronizar (`StandardScaler`), reduzir com PCA até
95% da variância acumulada e agrupar. Cada regra de decisão (K, `minPts`, ε) foi
fixada no plano antes de qualquer resultado e cita a aula:

- K-Means (Aula 3): K com o maior silhouette médio; o cotovelo (WCSS) só é registrado.
- Ward (Aula 5): comparação com o mesmo K.
- DBSCAN (Aula 4): `minPts = max(4, 2×d)`, ε varrido nos quantis da distância ao
  `(minPts−1)`-ésimo vizinho (o gráfico k-distância) e escolhido pelo maior silhouette
  calculado só sem o ruído; o ruído (rótulo −1) são os outliers.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.config import SEED

ATRIBUTOS_AGRUPAMENTO: tuple[str, ...] = (
    "idhm",
    "idhm_educacao",
    "idhm_renda",
    "idhm_longevidade",
    "taxa_alfabetizacao_2023",
    "taxa_alfabetizacao_2024",
)
VARIANCIA_PCA: float = 0.95
INTERVALO_K: range = range(2, 11)
QUANTIS_EPS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
MAX_CLUSTERS_DBSCAN: int = 10
LIMITE_RUIDO: float = 0.10
LIMITE_FRONTEIRA: float = 0.10
N_INICIALIZACOES: int = 10
RUIDO: int = -1


@dataclass(frozen=True)
class Reducao:
    """O espaço do PCA que os algoritmos usam e a projeção em 2D para o gráfico."""

    espaco: np.ndarray
    projecao_2d: np.ndarray
    n_componentes: int
    variancia_acumulada: np.ndarray


def selecionar_completos(perfil: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Municípios com os 6 atributos completos (índice reiniciado) e os ids dos excluídos.

    Ausente em outra coluna, como `capital_uf` ou `sigla_uf`, não exclui o município,
    porque essas colunas só descrevem os clusters.
    """
    completo = perfil[list(ATRIBUTOS_AGRUPAMENTO)].notna().all(axis=1)
    return perfil.loc[completo].reset_index(drop=True), perfil.loc[~completo, "id_municipio"].tolist()


def padronizar_e_reduzir(valores: pd.DataFrame, variancia: float = VARIANCIA_PCA) -> Reducao:
    """Padroniza e projeta no menor número de componentes que chega a `variancia` acumulada."""
    if valores.shape[1] < 2:
        raise ValueError("O agrupamento precisa de pelo menos dois atributos.")
    padronizados = StandardScaler().fit_transform(valores)
    pca = PCA(n_components=min(padronizados.shape), svd_solver="full").fit(padronizados)
    acumulada = np.cumsum(pca.explained_variance_ratio_)
    n_componentes = int(np.argmax(acumulada >= variancia - 1e-12)) + 1
    projecao = pca.transform(padronizados)
    return Reducao(
        espaco=projecao[:, :n_componentes],
        projecao_2d=projecao[:, :2],
        n_componentes=n_componentes,
        variancia_acumulada=acumulada,
    )


def agrupar_kmeans(espaco: np.ndarray, k: int, semente: int = SEED) -> np.ndarray:
    """K-Means com inicialização K-Means++ (padrão do scikit-learn)."""
    return KMeans(n_clusters=k, n_init=N_INICIALIZACOES, random_state=semente).fit_predict(espaco)


def agrupar_ward(espaco: np.ndarray, k: int) -> np.ndarray:
    """Aglomerativo com ligação de Ward, o critério que as aulas recomendam para explorar."""
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(espaco)


def avaliar(espaco: np.ndarray, rotulos: np.ndarray) -> dict[str, float]:
    """Clusters, percentual de ruído, silhouette e Davies-Bouldin, calculados só sem o ruído.

    Com menos de dois clusters as duas métricas não existem e saem `nan`, sem erro.
    """
    rotulos = np.asarray(rotulos)
    ruido = rotulos == RUIDO
    validos = ~ruido
    n_clusters = int(len(np.unique(rotulos[validos])))
    silhouette = davies_bouldin = float("nan")
    if n_clusters >= 2 and int(validos.sum()) > n_clusters:
        silhouette = float(silhouette_score(espaco[validos], rotulos[validos]))
        davies_bouldin = float(davies_bouldin_score(espaco[validos], rotulos[validos]))
    return {
        "n_clusters": n_clusters,
        "pct_ruido": float(ruido.mean()),
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
    }


def detalhar_silhouette(
    espaco: np.ndarray, rotulos: np.ndarray, limite_fronteira: float = LIMITE_FRONTEIRA
) -> dict[str, object]:
    """Silhouette médio de cada cluster e o percentual de municípios na fronteira, sem o ruído.

    Um cluster de silhouette médio baixo é difuso, e um município com silhouette abaixo de
    `limite_fronteira` está perto demais do vizinho para ser atribuído com firmeza. Com menos
    de dois clusters o detalhe não existe: o dicionário sai vazio e a fronteira, `nan`.
    """
    rotulos = np.asarray(rotulos)
    validos = rotulos != RUIDO
    clusters = np.unique(rotulos[validos])
    if len(clusters) < 2 or int(validos.sum()) <= len(clusters):
        return {"silhouette_por_cluster": {}, "pct_fronteira": float("nan")}
    amostras = silhouette_samples(espaco[validos], rotulos[validos])
    rotulos_validos = rotulos[validos]
    return {
        "silhouette_por_cluster": {int(c): float(amostras[rotulos_validos == c].mean()) for c in clusters},
        "pct_fronteira": float((amostras < limite_fronteira).mean()),
    }


def varrer_k(espaco: np.ndarray, ks: range = INTERVALO_K, semente: int = SEED) -> pd.DataFrame:
    """Inércia (WCSS, para o cotovelo), silhouette e Davies-Bouldin do K-Means para cada K."""
    linhas = []
    for k in ks:
        modelo = KMeans(n_clusters=k, n_init=N_INICIALIZACOES, random_state=semente).fit(espaco)
        metricas = avaliar(espaco, modelo.labels_)
        linhas.append(
            {"k": k, "inercia": float(modelo.inertia_), "silhouette": metricas["silhouette"], "davies_bouldin": metricas["davies_bouldin"]}
        )
    return pd.DataFrame(linhas)


def escolher_k(varredura: pd.DataFrame) -> int:
    """K de maior silhouette médio (Aula 3); em empate, o menor K."""
    melhor = varredura.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    return int(melhor["k"])


def min_pts(n_componentes: int) -> int:
    """`minPts` do DBSCAN: o maior entre 4 e 2×d (Aula 4: "≥ d+1, entre 4 e 10, ou 2×d")."""
    return max(4, 2 * n_componentes)


def grade_eps(espaco: np.ndarray, min_pts: int, quantis: tuple[float, ...] = QUANTIS_EPS) -> np.ndarray:
    """Candidatos a ε: quantis da distância de cada ponto ao `(min_pts−1)`-ésimo vizinho (gráfico k-distância)."""
    distancias, _ = NearestNeighbors(n_neighbors=min_pts).fit(espaco).kneighbors(espaco)
    candidatos = np.unique(np.quantile(distancias[:, -1], quantis))  # a coluna 0 é o próprio ponto
    return candidatos[candidatos > 0]


def agrupar_dbscan(espaco: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """DBSCAN; o rótulo −1 marca o ruído (os outliers)."""
    return DBSCAN(eps=eps, min_samples=min_pts).fit_predict(espaco)


def varrer_eps(espaco: np.ndarray, min_pts: int, eps_valores: np.ndarray) -> pd.DataFrame:
    """Clusters, ruído e métricas sem ruído para cada ε, com `minPts` fixo (Aula 4)."""
    linhas = []
    for eps in eps_valores:
        metricas = avaliar(espaco, agrupar_dbscan(espaco, float(eps), min_pts))
        linhas.append({"eps": float(eps), **metricas})
    return pd.DataFrame(linhas)[["eps", "n_clusters", "pct_ruido", "silhouette", "davies_bouldin"]]


def escolher_eps(
    varredura: pd.DataFrame, max_clusters: int = MAX_CLUSTERS_DBSCAN, limite_ruido: float = LIMITE_RUIDO
) -> float | None:
    """ε de maior silhouette entre os que dão de 2 a `max_clusters` clusters e no máximo `limite_ruido` de ruído.

    Devolve `None` se nenhum ε qualifica: o DBSCAN não achou estrutura útil.
    """
    validos = varredura[
        varredura["n_clusters"].between(2, max_clusters)
        & (varredura["pct_ruido"] <= limite_ruido)
        & varredura["silhouette"].notna()
    ]
    if validos.empty:
        return None
    return float(validos.sort_values(["silhouette", "eps"], ascending=[False, True]).iloc[0]["eps"])


def _sem_ruido(completos: pd.DataFrame, rotulos: np.ndarray) -> pd.DataFrame:
    dados = completos.assign(cluster=np.asarray(rotulos))
    return dados[dados["cluster"] != RUIDO]


def perfil_dos_clusters(completos: pd.DataFrame, rotulos: np.ndarray) -> pd.DataFrame:
    """Tamanho, média dos 6 atributos na escala original e percentual de capitais de cada cluster (sem o ruído)."""
    grupos = _sem_ruido(completos, rotulos).groupby("cluster")
    perfil = grupos[list(ATRIBUTOS_AGRUPAMENTO)].mean()
    perfil.insert(0, "n", grupos.size())
    perfil["pct_capital"] = grupos["capital_uf"].mean()
    return perfil


def clusters_por_regiao(completos: pd.DataFrame, rotulos: np.ndarray) -> pd.DataFrame:
    """Quantos municípios de cada região caem em cada cluster (sem o ruído)."""
    dados = _sem_ruido(completos, rotulos)
    return pd.crosstab(dados["cluster"], dados["nome_regiao"])
