import numpy as np
import pandas as pd

from src.modeling.baseline import ALVO_NAO_ALFABETIZADO, avaliar_baselines, montar_tabela_aluno


def _municipios() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_municipio": ["1", "2"],
            "sigla_uf": ["SP", "BA"],
            "nome_regiao": ["Sudeste", "Nordeste"],
            "capital_uf": [1.0, 0.0],
            "idhm": [0.8, 0.5],
            "idhm_educacao": [0.8, 0.5],
            "idhm_renda": [0.8, 0.5],
            "idhm_longevidade": [0.8, 0.5],
        }
    )


def _alunos(n: int = 400) -> pd.DataFrame:
    gerador = np.random.default_rng(42)
    municipio = gerador.choice(["1", "2"], n)
    # O município 1 (IDHM alto) tem mais alfabetizados: sinal fraco, mas real.
    alfabetizado = gerador.random(n) < np.where(municipio == "1", 0.7, 0.4)
    return pd.DataFrame({"id_municipio": municipio, "caderno": gerador.choice(["1", "2"], n), "rede": "3", "alfabetizado": alfabetizado})


def test_alvo_da_tabela_tem_classe_positiva_em_nao_alfabetizado() -> None:
    alunos = _alunos()
    tabela = montar_tabela_aluno(alunos, _municipios())
    assert tabela[ALVO_NAO_ALFABETIZADO].tolist() == (~alunos["alfabetizado"]).astype(int).tolist()


def test_dummy_tem_auc_roc_de_meio_e_a_regressao_logistica_passa_disso() -> None:
    resultado = avaliar_baselines(montar_tabela_aluno(_alunos(), _municipios()))
    assert resultado["auc_roc_dummy"] == 0.5
    assert resultado["auc_roc_regressao_logistica"] > 0.5


def test_separacao_80_20_e_resultado_reprodutivel() -> None:
    tabela = montar_tabela_aluno(_alunos(), _municipios())
    primeiro, segundo = avaliar_baselines(tabela), avaliar_baselines(tabela)
    assert primeiro["linhas_treino"] == 320 and primeiro["linhas_validacao"] == 80
    assert primeiro == segundo
