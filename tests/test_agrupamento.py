import numpy as np
import pandas as pd
import pytest

from src.modeling.agrupamento import (
    ATRIBUTOS_AGRUPAMENTO,
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

CENTROS = {
    0: {"idhm": 0.50, "taxa_2023": 35.0, "taxa_2024": 38.0, "regiao": "Norte"},
    1: {"idhm": 0.65, "taxa_2023": 55.0, "taxa_2024": 58.0, "regiao": "Nordeste"},
    2: {"idhm": 0.80, "taxa_2023": 75.0, "taxa_2024": 80.0, "regiao": "Sul"},
}
OUTLIERS = [(0.15, 99.0), (0.95, 2.0), (0.10, 5.0), (0.99, 100.0)]  # (idhm, taxa) extremos


def _perfil(por_grupo: int = 100, semente: int = 3, com_outliers: bool = False) -> pd.DataFrame:
    """3 grupos bem separados nos 6 atributos; `grupo_real` guarda a resposta certa (-1 = outlier plantado)."""
    gerador = np.random.default_rng(semente)
    linhas = []
    for grupo, centro in CENTROS.items():
        for i in range(por_grupo):
            base = centro["idhm"] + gerador.normal(0, 0.01)
            linhas.append(
                {
                    "id_municipio": f"{grupo}{i:03d}",
                    "sigla_uf": "XX",
                    "nome_regiao": centro["regiao"],
                    "capital_uf": 0.0,
                    "idhm": base,
                    "idhm_educacao": base - 0.08 + gerador.normal(0, 0.01),
                    "idhm_renda": base + gerador.normal(0, 0.01),
                    "idhm_longevidade": base + 0.10 + gerador.normal(0, 0.01),
                    "taxa_alfabetizacao_2023": centro["taxa_2023"] + gerador.normal(0, 2),
                    "taxa_alfabetizacao_2024": centro["taxa_2024"] + gerador.normal(0, 2),
                    "grupo_real": grupo,
                }
            )
    if com_outliers:
        for j, (idhm, taxa) in enumerate(OUTLIERS):
            linhas.append(
                {
                    "id_municipio": f"o{j}",
                    "sigla_uf": "XX",
                    "nome_regiao": "Norte",
                    "capital_uf": 0.0,
                    "idhm": idhm,
                    "idhm_educacao": idhm,
                    "idhm_renda": idhm,
                    "idhm_longevidade": idhm,
                    "taxa_alfabetizacao_2023": taxa,
                    "taxa_alfabetizacao_2024": taxa,
                    "grupo_real": -1,
                }
            )
    return pd.DataFrame(linhas)


def _espaco(perfil: pd.DataFrame):
    return padronizar_e_reduzir(perfil[list(ATRIBUTOS_AGRUPAMENTO)])


def _um_grupo_real_por_cluster(rotulos: np.ndarray, grupos_reais: pd.Series) -> bool:
    tabela = pd.crosstab(grupos_reais.to_numpy(), rotulos)
    return bool((tabela.gt(0).sum(axis=1) == 1).all() and (tabela.gt(0).sum(axis=0) == 1).all())


# ---------- seleção dos atributos ----------


def test_atributos_do_agrupamento_sao_os_seis_numericos_sem_capital() -> None:
    assert ATRIBUTOS_AGRUPAMENTO == (
        "idhm", "idhm_educacao", "idhm_renda", "idhm_longevidade", "taxa_alfabetizacao_2023", "taxa_alfabetizacao_2024",
    )


def test_so_municipios_com_os_seis_atributos_completos_ficam() -> None:
    perfil = _perfil(por_grupo=3)
    perfil.loc[0, "idhm"] = np.nan
    perfil.loc[4, "taxa_alfabetizacao_2023"] = np.nan
    perfil.loc[7, "taxa_alfabetizacao_2024"] = np.nan
    completos, excluidos = selecionar_completos(perfil)
    assert excluidos == [perfil.loc[i, "id_municipio"] for i in (0, 4, 7)]
    assert len(completos) == len(perfil) - 3
    assert completos[list(ATRIBUTOS_AGRUPAMENTO)].notna().all().all()
    assert completos.index.tolist() == list(range(len(completos)))


def test_ausente_em_coluna_que_nao_e_atributo_nao_exclui_o_municipio() -> None:
    perfil = _perfil(por_grupo=3)
    perfil.loc[2, "capital_uf"] = np.nan
    perfil.loc[3, "sigla_uf"] = None
    completos, excluidos = selecionar_completos(perfil)
    assert excluidos == [] and len(completos) == len(perfil)


# ---------- padronização e PCA ----------


def test_pca_mantem_o_menor_numero_de_componentes_que_chega_a_95_por_cento() -> None:
    reducao = _espaco(_perfil())
    acumulada = reducao.variancia_acumulada
    n = reducao.n_componentes
    assert acumulada[n - 1] >= 0.95
    assert n == 1 or acumulada[n - 2] < 0.95
    assert reducao.espaco.shape == (300, n)


def test_a_variancia_acumulada_cresce_ate_um() -> None:
    acumulada = _espaco(_perfil()).variancia_acumulada
    assert np.all(np.diff(acumulada) >= -1e-12)
    assert acumulada[-1] == pytest.approx(1.0)


def test_padronizar_torna_o_resultado_independente_da_escala_de_cada_atributo() -> None:
    perfil = _perfil()
    ampliado = perfil.copy()
    ampliado["taxa_alfabetizacao_2023"] *= 1000.0
    a, b = _espaco(perfil), _espaco(ampliado)
    assert a.n_componentes == b.n_componentes
    assert np.allclose(np.abs(a.espaco), np.abs(b.espaco), atol=1e-6)  # o sinal de um componente é arbitrário


def test_projecao_2d_tem_duas_colunas_mesmo_se_um_componente_basta() -> None:
    perfil = _perfil()
    for coluna in ATRIBUTOS_AGRUPAMENTO[1:]:
        perfil[coluna] = perfil["idhm"] * 2.0 + 1.0  # todos colineares: 1 componente explica tudo
    reducao = _espaco(perfil)
    assert reducao.n_componentes == 1
    assert reducao.projecao_2d.shape == (300, 2)


# ---------- K-Means: varredura e escolha de K ----------


def test_varredura_de_k_traz_inercia_silhouette_e_davies_bouldin_por_k() -> None:
    varredura = varrer_k(_espaco(_perfil()).espaco, ks=range(2, 7))
    assert list(varredura.columns) == ["k", "inercia", "silhouette", "davies_bouldin"]
    assert varredura["k"].tolist() == [2, 3, 4, 5, 6]


def test_a_inercia_cai_a_cada_k_e_o_silhouette_e_maximo_no_k_real() -> None:
    varredura = varrer_k(_espaco(_perfil()).espaco, ks=range(2, 8))
    assert varredura["inercia"].is_monotonic_decreasing
    assert escolher_k(varredura) == 3


def test_escolher_k_pega_o_maior_silhouette_e_desempata_pelo_menor_k() -> None:
    varredura = pd.DataFrame(
        {"k": [2, 3, 4, 5], "inercia": [9.0, 5.0, 3.0, 2.0], "silhouette": [0.40, 0.55, 0.55, 0.30], "davies_bouldin": [1.0, 0.7, 0.8, 1.2]}
    )
    assert escolher_k(varredura) == 3


def test_varredura_de_k_e_reprodutivel() -> None:
    espaco = _espaco(_perfil()).espaco
    pd.testing.assert_frame_equal(varrer_k(espaco, ks=range(2, 6)), varrer_k(espaco, ks=range(2, 6)))


# ---------- K-Means e Ward recuperam os grupos ----------


@pytest.mark.parametrize("algoritmo", [agrupar_kmeans, agrupar_ward])
def test_kmeans_e_ward_recuperam_os_tres_grupos_reais(algoritmo) -> None:
    perfil = _perfil()
    rotulos = algoritmo(_espaco(perfil).espaco, 3)
    assert sorted(set(rotulos)) == [0, 1, 2]
    assert _um_grupo_real_por_cluster(rotulos, perfil["grupo_real"])


def test_kmeans_e_reprodutivel_e_ward_e_deterministico() -> None:
    espaco = _espaco(_perfil()).espaco
    assert np.array_equal(agrupar_kmeans(espaco, 3), agrupar_kmeans(espaco, 3))
    assert np.array_equal(agrupar_ward(espaco, 3), agrupar_ward(espaco, 3))


# ---------- avaliação: silhouette e Davies-Bouldin sem o ruído ----------


def test_avaliacao_ignora_o_ruido_e_informa_clusters_e_percentual_de_ruido() -> None:
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    perfil = _perfil(com_outliers=True)
    espaco = _espaco(perfil).espaco
    rotulos = agrupar_kmeans(espaco, 3).copy()
    ruido = perfil["grupo_real"].to_numpy() == -1
    rotulos[ruido] = -1  # os 4 outliers viram ruído
    resultado = avaliar(espaco, rotulos)
    sem_ruido = ~ruido
    assert resultado["silhouette"] == pytest.approx(silhouette_score(espaco[sem_ruido], rotulos[sem_ruido]))
    assert resultado["davies_bouldin"] == pytest.approx(davies_bouldin_score(espaco[sem_ruido], rotulos[sem_ruido]))
    assert resultado["silhouette"] != pytest.approx(silhouette_score(espaco, rotulos))  # contar o ruído muda o valor
    assert resultado["n_clusters"] == 3
    assert resultado["pct_ruido"] == pytest.approx(4 / len(perfil))


def test_avaliacao_com_menos_de_dois_clusters_devolve_nan_em_vez_de_quebrar() -> None:
    espaco = _espaco(_perfil()).espaco
    um_so = avaliar(espaco, np.zeros(len(espaco), dtype=int))
    tudo_ruido = avaliar(espaco, np.full(len(espaco), -1))
    for resultado in (um_so, tudo_ruido):
        assert np.isnan(resultado["silhouette"]) and np.isnan(resultado["davies_bouldin"])
    assert um_so["n_clusters"] == 1 and tudo_ruido["n_clusters"] == 0
    assert tudo_ruido["pct_ruido"] == 1.0


# ---------- DBSCAN: minPts, grade de ε, varredura e escolha ----------


@pytest.mark.parametrize("n_componentes, esperado", [(1, 4), (2, 4), (3, 6), (4, 8), (6, 12)])
def test_min_pts_e_o_maior_entre_quatro_e_duas_vezes_as_dimensoes(n_componentes: int, esperado: int) -> None:
    assert min_pts(n_componentes) == esperado


def test_grade_de_eps_vem_dos_quantis_da_distancia_ao_vizinho_k() -> None:
    # posições 0, 1, 3, 7, 15 (espaçamento dobrando). Sem contar o próprio ponto:
    #   1º vizinho a 1, 1, 2, 4 e 8; 2º vizinho a 3, 2, 3, 6 e 12 (0→1,3; 1→0,3; 3→1,0; 7→3,1; 15→7,3)
    pontos = np.array([0.0, 1.0, 3.0, 7.0, 15.0]).reshape(-1, 1)
    assert grade_eps(pontos, min_pts=2, quantis=(0.0, 0.5, 1.0)).tolist() == [1.0, 2.0, 8.0]
    assert grade_eps(pontos, min_pts=3, quantis=(0.0, 0.5, 1.0)).tolist() == [2.0, 3.0, 12.0]


def test_quantis_que_dao_a_mesma_distancia_viram_um_candidato_so() -> None:
    pontos = np.array([[0.0], [1.0]])  # os dois pontos têm o vizinho a 1
    assert grade_eps(pontos, min_pts=2, quantis=(0.1, 0.5, 0.9)).tolist() == [1.0]


def test_grade_de_eps_e_crescente_e_sem_repeticoes() -> None:
    espaco = _espaco(_perfil(com_outliers=True)).espaco
    grade = grade_eps(espaco, min_pts=8)
    assert len(grade) > 1 and (np.diff(grade) > 0).all() and (grade > 0).all()


def test_varredura_de_eps_traz_clusters_ruido_e_metricas_sem_ruido() -> None:
    espaco = _espaco(_perfil(com_outliers=True)).espaco
    varredura = varrer_eps(espaco, min_pts=8, eps_valores=grade_eps(espaco, 8))
    assert list(varredura.columns) == ["eps", "n_clusters", "pct_ruido", "silhouette", "davies_bouldin"]
    assert varredura["pct_ruido"].between(0, 1).all()


def test_escolher_eps_segue_a_regra_maior_silhouette_dentro_dos_limites() -> None:
    varredura = pd.DataFrame(
        {
            "eps": [0.5, 1.0, 1.5, 2.0, 2.5],
            "n_clusters": [3, 3, 4, 1, 12],
            "pct_ruido": [0.30, 0.05, 0.08, 0.0, 0.02],
            "silhouette": [0.90, 0.60, 0.70, np.nan, 0.95],
            "davies_bouldin": [0.2, 0.5, 0.4, np.nan, 0.1],
        }
    )
    # 0,5 tem ruído demais; 2,0 tem um cluster só; 2,5 tem clusters demais; sobra 1,5 com o maior silhouette
    assert escolher_eps(varredura) == 1.5


def test_escolher_eps_devolve_none_se_nenhum_valor_qualifica() -> None:
    varredura = pd.DataFrame(
        {"eps": [0.5, 1.0], "n_clusters": [1, 15], "pct_ruido": [0.0, 0.5], "silhouette": [np.nan, 0.8], "davies_bouldin": [np.nan, 0.3]}
    )
    assert escolher_eps(varredura) is None


def test_dbscan_marca_como_ruido_os_outliers_plantados_e_acha_os_tres_grupos() -> None:
    perfil = _perfil(com_outliers=True)
    reducao = _espaco(perfil)
    varredura = varrer_eps(reducao.espaco, min_pts(reducao.n_componentes), grade_eps(reducao.espaco, min_pts(reducao.n_componentes)))
    eps = escolher_eps(varredura)
    assert eps is not None
    rotulos = agrupar_dbscan(reducao.espaco, eps, min_pts(reducao.n_componentes))
    ruido = rotulos == -1
    assert ruido[perfil["grupo_real"].to_numpy() == -1].all()  # todos os outliers plantados
    assert sorted(set(rotulos[~ruido])) == [0, 1, 2]
    assert ruido.mean() <= 0.10


# ---------- perfil dos clusters ----------


def _pequeno() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_municipio": list("abcdef"),
            "nome_regiao": ["Norte", "Norte", "Sul", "Sul", "Sul", "Nordeste"],
            "capital_uf": [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "idhm": [0.4, 0.6, 0.8, 0.8, 1.0, 0.5],
            "idhm_educacao": [0.3, 0.5, 0.7, 0.7, 0.9, 0.4],
            "idhm_renda": [0.4, 0.6, 0.8, 0.8, 1.0, 0.5],
            "idhm_longevidade": [0.5, 0.7, 0.9, 0.9, 1.1, 0.6],
            "taxa_alfabetizacao_2023": [30.0, 50.0, 70.0, 70.0, 90.0, 40.0],
            "taxa_alfabetizacao_2024": [32.0, 52.0, 72.0, 72.0, 92.0, 42.0],
        }
    )


def test_perfil_traz_tamanho_medias_na_escala_original_e_percentual_de_capitais() -> None:
    perfil = perfil_dos_clusters(_pequeno(), np.array([0, 0, 1, 1, 1, 0]))
    assert perfil.index.tolist() == [0, 1]
    assert perfil["n"].tolist() == [3, 3]
    assert perfil.loc[0, "idhm"] == pytest.approx(0.5)  # (0,4 + 0,6 + 0,5) / 3, nada padronizado
    assert perfil.loc[1, "taxa_alfabetizacao_2023"] == pytest.approx(230 / 3)
    assert perfil.loc[0, "pct_capital"] == pytest.approx(2 / 3)
    assert perfil.loc[1, "pct_capital"] == 0.0


def test_perfil_deixa_o_ruido_de_fora() -> None:
    perfil = perfil_dos_clusters(_pequeno(), np.array([0, 0, 1, 1, -1, -1]))
    assert perfil.index.tolist() == [0, 1] and perfil["n"].sum() == 4


def test_clusters_por_regiao_conta_os_municipios_de_cada_regiao_em_cada_cluster() -> None:
    tabela = clusters_por_regiao(_pequeno(), np.array([0, 0, 1, 1, 1, 0]))
    assert tabela.loc[0, "Norte"] == 2 and tabela.loc[0, "Nordeste"] == 1 and tabela.loc[0, "Sul"] == 0
    assert tabela.loc[1, "Sul"] == 3 and tabela.loc[1, "Norte"] == 0
    assert tabela.to_numpy().sum() == 6


# ---------- detalhe do silhouette: por cluster e na fronteira ----------

DOIS_E_DOIS = np.array([[0.0], [1.0], [10.0], [11.0]])  # dois pares bem separados


def test_silhouette_por_cluster_confere_com_a_conta_a_mao() -> None:
    # ponto 0: a=1, b=(10+11)/2=10,5 -> 9,5/10,5; ponto 1: a=1, b=(9+10)/2=9,5 -> 8,5/9,5; o cluster 1 é simétrico
    esperado = (9.5 / 10.5 + 8.5 / 9.5) / 2
    detalhe = detalhar_silhouette(DOIS_E_DOIS, np.array([0, 0, 1, 1]))
    assert detalhe["silhouette_por_cluster"] == pytest.approx({0: esperado, 1: esperado})
    assert detalhe["pct_fronteira"] == 0.0


def test_percentual_de_fronteira_conta_os_pontos_abaixo_do_limite() -> None:
    # os silhouettes são 0,9048, 0,8947, 0,8947 e 0,9048: com limite 0,9 ficam abaixo os dois do meio
    assert detalhar_silhouette(DOIS_E_DOIS, np.array([0, 0, 1, 1]), limite_fronteira=0.9)["pct_fronteira"] == 0.5
    assert detalhar_silhouette(DOIS_E_DOIS, np.array([0, 0, 1, 1]), limite_fronteira=0.95)["pct_fronteira"] == 1.0


def test_o_ruido_fica_fora_do_detalhe_e_do_denominador() -> None:
    com_ruido = np.vstack([DOIS_E_DOIS, [[500.0]]])
    sem = detalhar_silhouette(DOIS_E_DOIS, np.array([0, 0, 1, 1]), limite_fronteira=0.9)
    com = detalhar_silhouette(com_ruido, np.array([0, 0, 1, 1, -1]), limite_fronteira=0.9)
    assert com["silhouette_por_cluster"] == pytest.approx(sem["silhouette_por_cluster"])
    assert com["pct_fronteira"] == sem["pct_fronteira"]  # o ponto de ruído não entra na conta


def test_com_menos_de_dois_clusters_o_detalhe_fica_vazio_sem_quebrar() -> None:
    um_so = detalhar_silhouette(DOIS_E_DOIS, np.zeros(4, dtype=int))
    assert um_so["silhouette_por_cluster"] == {} and np.isnan(um_so["pct_fronteira"])
