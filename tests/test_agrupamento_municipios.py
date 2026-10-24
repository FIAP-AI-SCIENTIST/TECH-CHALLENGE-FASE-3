import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling import agrupamento_municipios
from src.modeling.agrupamento_municipios import executar

CENTROS = {0: (0.50, 35.0, 38.0, "Norte"), 1: (0.65, 55.0, 58.0, "Nordeste"), 2: (0.80, 75.0, 80.0, "Sul")}
POR_GRUPO = 100
SEM_IDHM, SEM_TAXA_2023, SEM_TAXA_2024 = "0001", "1001", "2001"
OUTLIERS = [("o0", 0.15, 99.0), ("o1", 0.95, 2.0), ("o2", 0.10, 5.0), ("o3", 0.99, 100.0)]


def _gravar_snapshot(diretorio: Path, centros: dict = CENTROS, com_outliers: bool = True) -> None:
    gerador = np.random.default_rng(3)
    municipios, metas = [], []

    def adiciona(id_municipio, idhm, taxa_2023, taxa_2024, regiao, capital=0.0):
        municipios.append(
            {
                "id_municipio": id_municipio,
                "sigla_uf": "XX",
                "nome_regiao": regiao,
                "capital_uf": capital,
                "idhm": idhm,
                "idhm_educacao": idhm - 0.08,
                "idhm_renda": idhm,
                "idhm_longevidade": idhm + 0.10,
            }
        )
        metas.append((id_municipio, "Municipal", 2023, taxa_2023))
        metas.append((id_municipio, "Municipal", 2024, taxa_2024))

    for grupo, (idhm, taxa_2023, taxa_2024, regiao) in centros.items():
        for i in range(POR_GRUPO):
            adiciona(
                f"{grupo}{i:03d}",
                idhm + gerador.normal(0, 0.01),
                taxa_2023 + gerador.normal(0, 2),
                taxa_2024 + gerador.normal(0, 2),
                regiao,
                capital=1.0 if (grupo == 2 and i < 2) else 0.0,
            )
    for id_municipio, idhm, taxa in OUTLIERS if com_outliers else []:
        adiciona(id_municipio, idhm, taxa, taxa, "Norte")

    tabela_municipios = pd.DataFrame(municipios)
    tabela_municipios.loc[tabela_municipios["id_municipio"] == SEM_IDHM, ["idhm", "idhm_educacao", "idhm_renda", "idhm_longevidade"]] = np.nan
    tabela_metas = pd.DataFrame(metas, columns=["id_municipio", "rede", "ano", "taxa_alfabetizacao"])
    tabela_metas = tabela_metas[~((tabela_metas["id_municipio"] == SEM_TAXA_2023) & (tabela_metas["ano"] == 2023))]
    tabela_metas.loc[(tabela_metas["id_municipio"] == SEM_TAXA_2024) & (tabela_metas["ano"] == 2024), "taxa_alfabetizacao"] = np.nan
    tabela_municipios.to_parquet(diretorio / "dim_municipio.parquet")
    tabela_metas.to_parquet(diretorio / "fact_meta_resultado_municipio.parquet")


@pytest.fixture
def execucao(tmp_path: Path):
    dados, relatorios, imagens = tmp_path / "data", tmp_path / "reports", tmp_path / "images"
    dados.mkdir()
    _gravar_snapshot(dados)
    resultado = executar(dados, relatorios, imagens)
    return resultado, relatorios, imagens


def _json(relatorios: Path) -> dict:
    return json.loads((relatorios / "agrupamento_municipios.json").read_text(encoding="utf-8"))


def _clusters(relatorios: Path) -> pd.DataFrame:
    return pd.read_csv(relatorios / "clusters_municipios.csv", dtype={"id_municipio": str})


def test_grava_o_json_os_tres_csvs_e_as_duas_figuras(execucao) -> None:
    _, relatorios, imagens = execucao
    for nome in ("agrupamento_municipios.json", "clusters_municipios.csv", "perfil_clusters.csv", "clusters_por_regiao.csv"):
        assert (relatorios / nome).exists(), nome
    for nome in ("agrupamento_pca_2d.png", "agrupamento_escolha_k.png"):
        assert (imagens / nome).stat().st_size > 0, nome


def test_so_municipios_com_dados_completos_sao_agrupados_e_os_excluidos_ficam_listados(execucao) -> None:
    _, relatorios, _ = execucao
    conteudo, clusters = _json(relatorios), _clusters(relatorios)
    excluidos = {SEM_IDHM, SEM_TAXA_2023, SEM_TAXA_2024}
    assert set(conteudo["excluidos"]) == excluidos
    assert conteudo["n_municipios_perfil"] == 3 * POR_GRUPO + len(OUTLIERS)
    assert conteudo["n_municipios_agrupados"] == conteudo["n_municipios_perfil"] - len(excluidos)
    assert len(clusters) == conteudo["n_municipios_agrupados"] and clusters["id_municipio"].is_unique
    assert excluidos.isdisjoint(set(clusters["id_municipio"]))


def test_o_csv_de_clusters_traz_uf_regiao_e_as_tres_atribuicoes(execucao) -> None:
    _, relatorios, _ = execucao
    assert list(_clusters(relatorios).columns) == [
        "id_municipio", "sigla_uf", "nome_regiao", "cluster", "cluster_k3", "cluster_ward", "outlier_dbscan",
    ]


def test_os_grupos_reais_caem_cada_um_num_cluster_proprio_de_kmeans(execucao) -> None:
    _, relatorios, _ = execucao
    clusters = _clusters(relatorios)
    normais = clusters[~clusters["id_municipio"].str.startswith("o")]
    tabela = pd.crosstab(normais["id_municipio"].str[0], normais["cluster"])
    assert (tabela.gt(0).sum(axis=1) == 1).all()  # cada grupo real num cluster só


def test_k_escolhido_e_o_de_maior_silhouette_da_varredura_registrada(execucao) -> None:
    _, relatorios, _ = execucao
    conteudo = json.loads((relatorios / "agrupamento_municipios.json").read_text(encoding="utf-8"))
    varredura = pd.DataFrame(conteudo["kmeans"]["varredura"])
    assert conteudo["kmeans"]["k_escolhido"] == int(varredura.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    assert _clusters(relatorios)["cluster"].nunique() == conteudo["kmeans"]["k_escolhido"]
    assert varredura["k"].tolist() == list(range(2, 11))


def test_pca_registra_componentes_e_variancia_e_dbscan_usa_a_regra_de_min_pts(execucao) -> None:
    _, relatorios, _ = execucao
    conteudo = _json(relatorios)
    n = conteudo["pca"]["n_componentes"]
    assert conteudo["pca"]["variancia_acumulada"][n - 1] >= 0.95
    assert conteudo["dbscan"]["min_pts"] == max(4, 2 * n)


def test_dbscan_aponta_como_outliers_os_quatro_plantados(execucao) -> None:
    _, relatorios, _ = execucao
    clusters = _clusters(relatorios)
    plantados = clusters["id_municipio"].str.startswith("o")
    assert clusters.loc[plantados, "outlier_dbscan"].eq(1).all()
    assert clusters["outlier_dbscan"].mean() <= 0.10
    conteudo = _json(relatorios)
    assert conteudo["dbscan"]["n_outliers"] == int(clusters["outlier_dbscan"].sum())
    assert conteudo["dbscan"]["eps_escolhido"] in [linha["eps"] for linha in conteudo["dbscan"]["varredura"]]


def test_avaliacao_traz_silhouette_e_davies_bouldin_dos_tres_algoritmos(execucao) -> None:
    _, relatorios, _ = execucao
    avaliacao = _json(relatorios)["avaliacao"]
    assert set(avaliacao) == {"kmeans", "ward", "dbscan"}
    for algoritmo in avaliacao.values():
        assert {"n_clusters", "pct_ruido", "silhouette", "davies_bouldin"} <= set(algoritmo)
    assert avaliacao["kmeans"]["silhouette"] > 0.5  # grupos bem separados


def test_perfil_tem_um_cluster_por_linha_e_os_tamanhos_somam_os_agrupados(execucao) -> None:
    _, relatorios, _ = execucao
    perfil = pd.read_csv(relatorios / "perfil_clusters.csv", index_col="cluster")
    assert perfil["n"].sum() == _json(relatorios)["n_municipios_agrupados"]
    assert {"idhm", "taxa_alfabetizacao_2023", "pct_capital"} <= set(perfil.columns)
    assert len(perfil) == _json(relatorios)["kmeans"]["k_escolhido"]


def test_tabela_por_regiao_soma_os_municipios_agrupados(execucao) -> None:
    _, relatorios, _ = execucao
    por_regiao = pd.read_csv(relatorios / "clusters_por_regiao.csv", index_col="cluster")
    assert por_regiao.to_numpy().sum() == _json(relatorios)["n_municipios_agrupados"]
    assert {"Norte", "Nordeste", "Sul"} <= set(por_regiao.columns)


def test_resultado_e_reprodutivel(tmp_path: Path) -> None:
    saidas = []
    for nome in ("a", "b"):
        dados, relatorios = tmp_path / nome / "data", tmp_path / nome / "reports"
        dados.mkdir(parents=True)
        _gravar_snapshot(dados)
        executar(dados, relatorios, tmp_path / nome / "images")
        saidas.append(((relatorios / "clusters_municipios.csv").read_bytes(), (relatorios / "agrupamento_municipios.json").read_bytes()))
    assert saidas[0] == saidas[1]


def test_sem_epsilon_valido_o_dbscan_fica_vazio_sem_quebrar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agrupamento_municipios, "escolher_eps", lambda varredura: None)
    dados, relatorios = tmp_path / "data", tmp_path / "reports"
    dados.mkdir()
    _gravar_snapshot(dados)
    executar(dados, relatorios, tmp_path / "images")
    conteudo, clusters = _json(relatorios), _clusters(relatorios)
    assert conteudo["dbscan"]["eps_escolhido"] is None and conteudo["dbscan"]["n_outliers"] is None
    assert conteudo["avaliacao"]["dbscan"] is None
    assert clusters["outlier_dbscan"].isna().all()
    assert len(clusters) == conteudo["n_municipios_agrupados"]  # o K-Means e o Ward seguem normais


def test_informa_o_progresso_de_cada_etapa(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    dados = tmp_path / "data"
    dados.mkdir()
    _gravar_snapshot(dados)
    caplog.set_level(logging.INFO, logger="src")
    executar(dados, tmp_path / "r", tmp_path / "i")
    mensagens = [registro.getMessage() for registro in caplog.records]
    for trecho in ("Perfil:", "PCA:", "K-Means:", "Ward:", "DBSCAN:", "Resultados gravados"):
        assert any(trecho in m for m in mensagens), trecho


def test_clusters_sao_numerados_do_menor_para_o_maior_idhm_medio(execucao) -> None:
    _, relatorios, _ = execucao
    perfil = pd.read_csv(relatorios / "perfil_clusters.csv", index_col="cluster")
    assert perfil.index.tolist() == list(range(len(perfil)))
    assert perfil["idhm"].is_monotonic_increasing


def test_avaliacao_do_dbscan_e_coerente_com_os_outliers_apontados(execucao) -> None:
    _, relatorios, _ = execucao
    conteudo = _json(relatorios)
    dbscan = conteudo["avaliacao"]["dbscan"]
    esperado = conteudo["dbscan"]["n_outliers"] / conteudo["n_municipios_agrupados"]
    assert dbscan["pct_ruido"] == pytest.approx(esperado, abs=1e-6)  # o JSON guarda 6 casas decimais
    assert dbscan["n_clusters"] == 3  # os três grupos reais, com os outliers fora
    assert dbscan["silhouette"] > 0.5 and dbscan["davies_bouldin"] < 1.0  # calculados sem o ruído


def _numeros(valor):
    if isinstance(valor, dict):
        for item in valor.values():
            yield from _numeros(item)
    elif isinstance(valor, list):
        for item in valor:
            yield from _numeros(item)
    elif isinstance(valor, float):
        yield valor


def test_json_arredonda_os_decimais_para_nao_variar_com_o_ruido_do_ponto_flutuante(execucao) -> None:
    # a ordem das somas em paralelo muda os últimos dígitos da inércia do K-Means entre execuções
    _, relatorios, _ = execucao
    numeros = list(_numeros(_json(relatorios)))
    assert numeros and all(x == round(x, 6) for x in numeros)


# ---------- leitura complementar com K=3 (exploratória, não escolhida pela regra) ----------

DOIS_GRUPOS = {0: CENTROS[0], 2: CENTROS[2]}  # a regra escolhe K=2, então o K=3 complementar é outra coisa


@pytest.fixture
def execucao_dois_grupos(tmp_path: Path):
    dados, relatorios, imagens = tmp_path / "data", tmp_path / "reports", tmp_path / "images"
    dados.mkdir()
    _gravar_snapshot(dados, DOIS_GRUPOS, com_outliers=False)
    resultado = executar(dados, relatorios, imagens)
    return resultado, relatorios, imagens


def test_o_k_oficial_continua_sendo_o_da_regra_e_o_k3_e_um_corte_a_mais(execucao_dois_grupos) -> None:
    _, relatorios, _ = execucao_dois_grupos
    conteudo, clusters = _json(relatorios), _clusters(relatorios)
    assert conteudo["kmeans"]["k_escolhido"] == 2  # a regra, nos dois grupos reais
    assert clusters["cluster"].nunique() == 2 and clusters["cluster_k3"].nunique() == 3


def test_os_clusters_do_k3_sao_numerados_do_menor_para_o_maior_idhm_medio(execucao_dois_grupos) -> None:
    _, relatorios, _ = execucao_dois_grupos
    perfil = pd.read_csv(relatorios / "perfil_clusters_k3.csv", index_col="cluster")
    assert perfil.index.tolist() == [0, 1, 2]
    assert perfil["idhm"].is_monotonic_increasing
    assert perfil["n"].sum() == _json(relatorios)["n_municipios_agrupados"]


def test_a_composicao_por_regiao_do_k3_soma_os_municipios_agrupados(execucao_dois_grupos) -> None:
    _, relatorios, _ = execucao_dois_grupos
    por_regiao = pd.read_csv(relatorios / "clusters_por_regiao_k3.csv", index_col="cluster")
    assert por_regiao.to_numpy().sum() == _json(relatorios)["n_municipios_agrupados"]
    assert len(por_regiao) == 3


def test_o_json_marca_o_k3_como_complementar_e_o_k_da_regra_como_escolhido_pela_regra(execucao_dois_grupos) -> None:
    _, relatorios, _ = execucao_dois_grupos
    conteudo = _json(relatorios)
    assert conteudo["kmeans"]["escolhido_pela_regra"] is True
    assert conteudo["complementar"]["k"] == 3 and conteudo["complementar"]["escolhido_pela_regra"] is False


def test_o_complemento_traz_avaliacao_concordancia_com_o_ward_e_a_robustez(execucao_dois_grupos) -> None:
    resultado, relatorios, _ = execucao_dois_grupos
    complementar = _json(relatorios)["complementar"]
    assert {"n_clusters", "pct_ruido", "silhouette", "davies_bouldin"} <= set(complementar["avaliacao"])
    assert complementar["avaliacao"]["n_clusters"] == 3
    esperado = float((resultado.rotulos_kmeans_k3 == resultado.rotulos_ward_k3).mean())
    assert complementar["concordancia_com_ward"] == pytest.approx(esperado, abs=1e-6)
    assert len(complementar["avaliacao"]["silhouette_por_cluster"]) == 3
    assert 0.0 <= complementar["avaliacao"]["pct_fronteira"] <= 1.0


def test_a_robustez_tambem_vale_para_o_k_da_regra_para_comparar_os_dois(execucao_dois_grupos) -> None:
    _, relatorios, _ = execucao_dois_grupos
    kmeans = _json(relatorios)["avaliacao"]["kmeans"]
    assert len(kmeans["silhouette_por_cluster"]) == 2 and 0.0 <= kmeans["pct_fronteira"] <= 1.0


def test_grava_a_figura_do_pca_2d_com_os_clusters_do_k3(execucao_dois_grupos) -> None:
    _, _, imagens = execucao_dois_grupos
    assert (imagens / "agrupamento_pca_2d_k3.png").stat().st_size > 0


def test_informa_o_k3_como_complementar_no_log(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    dados = tmp_path / "data"
    dados.mkdir()
    _gravar_snapshot(dados, DOIS_GRUPOS, com_outliers=False)
    caplog.set_level(logging.INFO, logger="src")
    executar(dados, tmp_path / "r", tmp_path / "i")
    assert any("K=3" in m and "complementar" in m for m in (r.getMessage() for r in caplog.records))
