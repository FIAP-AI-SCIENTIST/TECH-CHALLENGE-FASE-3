import pandas as pd

from src.config import COLUNAS_T1
from src.preprocessing.atributos import colunas_proibidas_presentes
from src.preprocessing.tabelas import construir_tabela_aluno


def _municipios() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_municipio": ["1", "2"],
            "sigla_uf": ["SP", "BA"],
            "nome_regiao": ["Sudeste", "Nordeste"],
            "capital_uf": [1.0, 0.0],
            "idhm": [0.8, 0.6],
            "idhm_educacao": [0.75, 0.55],
            "idhm_renda": [0.8, 0.6],
            "idhm_longevidade": [0.85, 0.7],
            "nome": ["A", "B"],
        }
    )


def _alunos() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ano": [2024] * 5,
            "id_municipio": ["1", "1", "2", "2", "2"],
            "id_escola": ["e1", "e1", "e2", "e2", "e3"],
            "caderno": ["1", "1", "2", "2", "2"],
            "rede": ["3", "3", "2", "2", "3"],
            "presenca": [True, True, True, False, True],
            "preenchimento_caderno": [True, True, True, False, True],
            "alfabetizado": [True, False, False, False, True],
            "proficiencia": [760.0, 700.0, 710.0, None, 790.0],
            "peso_aluno": [1.0, 1.0, 1.0, None, 1.0],
        }
    )


def test_tabela_aluno_exclui_quem_nao_preencheu_o_caderno() -> None:
    tabela = construir_tabela_aluno(_alunos(), _municipios())
    assert len(tabela.atributos) == 4
    assert len(tabela.alvo) == len(tabela.grupos) == 4


def test_alvo_tem_classe_positiva_em_nao_alfabetizado() -> None:
    tabela = construir_tabela_aluno(_alunos(), _municipios())
    # alfabetizados dos alunos que fizeram a prova: True, False, False, True
    assert tabela.alvo.tolist() == [0, 1, 1, 0]


def test_atributos_sao_exatamente_os_da_trilha_e_sem_colunas_proibidas() -> None:
    tabela = construir_tabela_aluno(_alunos(), _municipios())
    assert list(tabela.atributos.columns) == list(COLUNAS_T1.atributos)
    assert colunas_proibidas_presentes(tabela.atributos.columns, COLUNAS_T1) == []


def test_grupo_e_o_municipio_e_nao_vira_atributo() -> None:
    tabela = construir_tabela_aluno(_alunos(), _municipios())
    assert tabela.grupos.tolist() == ["1", "1", "2", "2"]
    assert "id_municipio" not in tabela.atributos.columns


def test_atributos_municipais_chegam_pelo_municipio_do_aluno() -> None:
    tabela = construir_tabela_aluno(_alunos(), _municipios())
    assert tabela.atributos["sigla_uf"].tolist() == ["SP", "SP", "BA", "BA"]
    assert tabela.atributos["idhm"].tolist() == [0.8, 0.8, 0.6, 0.6]
