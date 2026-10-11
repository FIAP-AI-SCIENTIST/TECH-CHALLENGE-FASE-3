import numpy as np
import pandas as pd

from src.config import COLUNAS_T2
from src.preprocessing.atributos import colunas_proibidas_presentes
from src.preprocessing.tabelas import construir_atributos_municipio, construir_tabela_municipio

NAN = np.nan


def _municipios() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_municipio": ["1", "2", "3", "4"],
            "sigla_uf": ["SP", "BA", "RS", "AM"],
            "nome_regiao": ["Sudeste", "Nordeste", "Sul", "Norte"],
            "capital_uf": [1.0, 0.0, 0.0, 0.0],
            "idhm": [0.8, 0.6, 0.7, 0.5],
            "idhm_educacao": [0.75, 0.55, 0.65, 0.45],
            "idhm_renda": [0.8, 0.6, 0.7, 0.5],
            "idhm_longevidade": [0.85, 0.7, 0.8, 0.6],
        }
    )


def _metas(taxa_2024_municipio_1: float = 60.0) -> pd.DataFrame:
    linhas = [
        # id, rede, ano, taxa, meta, atingiu
        ("1", "Municipal", 2023, 50.0, NAN, None),
        ("1", "Municipal", 2024, taxa_2024_municipio_1, 55.0, True),
        ("2", "Municipal", 2023, 40.0, NAN, None),
        ("2", "Municipal", 2024, 45.0, 50.0, False),
        ("3", "Municipal", 2023, 70.0, NAN, None),
        ("3", "Municipal", 2024, 80.0, NAN, None),  # sem meta vigente: sem rótulo
        ("4", "Municipal", 2024, 30.0, 35.0, False),  # sem dados de 2023
    ]
    return pd.DataFrame(linhas, columns=["id_municipio", "rede", "ano", "taxa_alfabetizacao", "meta_indicador", "atingiu_meta"])


def _indicadores() -> pd.DataFrame:
    linhas = [
        # id, rede (código), ano, media_portugues
        ("1", "3", 2023, 5.0),
        ("2", "3", 2023, 4.0),
        ("3", "3", 2023, 6.0),
        ("1", "3", 2024, 9.9),  # 2024 não pode entrar
        ("1", "2", 2023, 1.0),  # outra rede não pode entrar
    ]
    return pd.DataFrame(linhas, columns=["id_municipio", "rede", "ano", "media_portugues"])


def _redes() -> pd.DataFrame:
    return pd.DataFrame({"rede": ["3", "2"], "rede_desc": ["Municipal", "Estadual"]})


def _tabela(taxa_2024_municipio_1: float = 60.0):
    return construir_tabela_municipio(_metas(taxa_2024_municipio_1), _indicadores(), _municipios(), _redes())


def test_tabela_municipio_tem_so_municipios_com_rotulo_de_2024() -> None:
    tabela = _tabela()
    assert tabela.grupos.tolist() == ["1", "2", "4"]


def test_alvo_tem_classe_positiva_em_nao_atingiu_a_meta() -> None:
    assert _tabela().alvo.tolist() == [0, 1, 1]


def test_atributos_sao_exatamente_os_da_trilha_e_sem_colunas_proibidas() -> None:
    atributos = _tabela().atributos
    assert list(atributos.columns) == list(COLUNAS_T2.atributos)
    assert colunas_proibidas_presentes(atributos.columns, COLUNAS_T2) == []


def test_atributos_defasados_vem_de_2023_na_rede_municipal() -> None:
    atributos = _tabela().atributos
    assert atributos["taxa_alfabetizacao_2023"].iloc[:2].tolist() == [50.0, 40.0]
    assert atributos["media_portugues_2023"].iloc[:2].tolist() == [5.0, 4.0]


def test_meta_de_2024_e_folga_de_2023_sao_calculadas() -> None:
    atributos = _tabela().atributos
    assert atributos["meta_indicador_2024"].tolist() == [55.0, 50.0, 35.0]
    assert atributos["folga_2023"].iloc[:2].tolist() == [-5.0, -10.0]


def test_municipio_sem_dados_de_2023_continua_com_atributos_nulos() -> None:
    atributos = _tabela().atributos
    assert atributos["taxa_alfabetizacao_2023"].isna().tolist() == [False, False, True]
    assert atributos["folga_2023"].isna().tolist() == [False, False, True]


def test_resultado_de_2024_nao_altera_os_atributos() -> None:
    sem_mudanca = _tabela(taxa_2024_municipio_1=60.0).atributos
    com_mudanca = _tabela(taxa_2024_municipio_1=99.0).atributos
    pd.testing.assert_frame_equal(sem_mudanca, com_mudanca)


def test_atributos_para_agrupamento_tem_uma_linha_por_municipio_da_rede_municipal() -> None:
    tabela = construir_atributos_municipio(_municipios(), _metas())
    assert tabela["id_municipio"].tolist() == ["1", "2", "3", "4"]
    assert tabela["id_municipio"].is_unique


def test_atributos_para_agrupamento_trazem_taxas_dos_dois_anos_e_o_perfil() -> None:
    tabela = construir_atributos_municipio(_municipios(), _metas()).set_index("id_municipio")
    assert tabela.loc["1", "taxa_alfabetizacao_2023"] == 50.0
    assert tabela.loc["1", "taxa_alfabetizacao_2024"] == 60.0
    assert tabela.loc["1", "idhm"] == 0.8
    assert tabela.loc["1", "nome_regiao"] == "Sudeste"
    assert np.isnan(tabela.loc["4", "taxa_alfabetizacao_2023"])
