import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from src.config import COLUNAS_T1, ColunasTrilha
from src.preprocessing.transformadores import construir_preprocessador

TRILHA = ColunasTrilha(categoricas=("cor",), numericas=("x",), proibidas=())


def _treino() -> pd.DataFrame:
    return pd.DataFrame({"x": [0.0, 10.0, np.nan, 20.0], "cor": ["azul", "verde", "azul", "verde"], "id_municipio": ["1", "2", "3", "4"]})


def test_preprocessador_nasce_sem_ajuste() -> None:
    pre = construir_preprocessador(TRILHA, escalonar=False)
    assert isinstance(pre, ColumnTransformer)
    with pytest.raises(NotFittedError):
        check_is_fitted(pre)


def test_imputa_valores_numericos_ausentes_e_a_saida_nao_tem_nulos() -> None:
    pre = construir_preprocessador(TRILHA, escalonar=False).fit(_treino())
    saida = pre.transform(pd.DataFrame({"x": [np.nan, 5.0], "cor": ["azul", "verde"]}))
    assert not saida.isna().any().any()


def test_imputacao_usa_a_mediana_do_treino_e_nao_a_dos_dados_novos() -> None:
    pre = construir_preprocessador(TRILHA, escalonar=False).fit(_treino())  # mediana do treino = 10
    novos = pd.DataFrame({"x": [1000.0, np.nan], "cor": ["azul", "azul"]})  # mediana dos novos seria 1000
    assert pre.transform(novos)["numericos__x"].tolist() == [1000.0, 10.0]


def test_escalonamento_usa_media_e_desvio_do_treino() -> None:
    treino = pd.DataFrame({"x": [0.0, 10.0], "cor": ["azul", "verde"]})  # média 5, desvio 5
    pre = construir_preprocessador(TRILHA, escalonar=True).fit(treino)
    saida = pre.transform(pd.DataFrame({"x": [5.0, 15.0], "cor": ["azul", "azul"]}))
    assert saida["numericos__x"].tolist() == [0.0, 2.0]


def test_sem_escalonar_mantem_a_escala_original() -> None:
    treino = pd.DataFrame({"x": [0.0, 10.0], "cor": ["azul", "verde"]})
    pre = construir_preprocessador(TRILHA, escalonar=False).fit(treino)
    assert pre.transform(pd.DataFrame({"x": [5.0], "cor": ["azul"]}))["numericos__x"].tolist() == [5.0]


def test_one_hot_cria_uma_coluna_por_categoria_com_nome_legivel() -> None:
    pre = construir_preprocessador(TRILHA, escalonar=False).fit(_treino())
    saida = pre.transform(pd.DataFrame({"x": [1.0], "cor": ["verde"]}))
    assert saida["categoricos__cor_azul"].tolist() == [0.0]
    assert saida["categoricos__cor_verde"].tolist() == [1.0]


def test_categoria_nova_nos_dados_novos_nao_quebra_e_zera_o_one_hot() -> None:
    pre = construir_preprocessador(TRILHA, escalonar=False).fit(_treino())
    saida = pre.transform(pd.DataFrame({"x": [1.0], "cor": ["roxo"]}))
    assert saida[["categoricos__cor_azul", "categoricos__cor_verde"]].to_numpy().tolist() == [[0.0, 0.0]]


def test_colunas_fora_da_trilha_nao_chegam_a_saida() -> None:
    pre = construir_preprocessador(TRILHA, escalonar=False).fit(_treino())  # o treino tem id_municipio
    assert not any("id_municipio" in coluna for coluna in pre.get_feature_names_out())


def test_funciona_com_as_colunas_da_trilha_do_aluno() -> None:
    pre = construir_preprocessador(COLUNAS_T1, escalonar=True)
    dados = pd.DataFrame(
        {
            "rede": ["3", "2"],
            "caderno": ["1", "2"],
            "sigla_uf": ["SP", "BA"],
            "nome_regiao": ["Sudeste", "Nordeste"],
            "capital_uf": [1.0, 0.0],
            "idhm": [0.8, np.nan],
            "idhm_educacao": [0.7, 0.5],
            "idhm_renda": [0.8, 0.6],
            "idhm_longevidade": [0.8, 0.7],
        }
    )
    saida = pre.fit_transform(dados)
    assert len(saida) == 2 and not saida.isna().any().any()
