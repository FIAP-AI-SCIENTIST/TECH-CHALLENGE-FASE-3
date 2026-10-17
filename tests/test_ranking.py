import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import ColunasTrilha
from src.modeling.pipeline_modelo import montar_pipeline
from src.modeling.ranking import montar_ranking, probabilidades_fora_da_dobra, taxa_por_municipio
from src.preprocessing.tabelas import TabelaModelagem

TRILHA = ColunasTrilha(categoricas=("cor",), numericas=("x",), proibidas=())


def _tabela(n_municipios: int = 40, por_municipio: int = 10, semente: int = 8) -> TabelaModelagem:
    gerador = np.random.default_rng(semente)
    grupos = np.repeat([f"m{i:02d}" for i in range(n_municipios)], por_municipio)
    x = np.repeat(gerador.normal(size=n_municipios), por_municipio) + gerador.normal(scale=0.3, size=len(grupos))
    cor = gerador.choice(["azul", "verde"], len(grupos))
    alvo = (x + gerador.normal(scale=0.5, size=len(grupos)) > 0).astype(int)
    return TabelaModelagem(pd.DataFrame({"x": x, "cor": cor}), pd.Series(alvo), pd.Series(grupos))


def _modelo():
    return montar_pipeline(TRILHA, LogisticRegression(), escalonar=True)


def test_probabilidade_fora_da_dobra_tem_uma_por_linha_entre_zero_e_um() -> None:
    tabela = _tabela()
    probabilidades = probabilidades_fora_da_dobra(_modelo(), tabela)
    assert len(probabilidades) == len(tabela.alvo)
    assert probabilidades.between(0, 1).all()


def test_probabilidade_de_um_municipio_vem_de_modelo_que_nao_o_viu() -> None:
    tabela = _tabela()
    probabilidades = probabilidades_fora_da_dobra(_modelo(), tabela)
    ajustado_em_tudo = _modelo().fit(tabela.atributos, tabela.alvo).predict_proba(tabela.atributos)[:, 1]
    assert not np.allclose(probabilidades.to_numpy(), ajustado_em_tudo)


def test_probabilidade_fora_da_dobra_e_reprodutivel_e_aceita_limite_de_linhas() -> None:
    tabela = _tabela()
    primeira = probabilidades_fora_da_dobra(_modelo(), tabela, max_linhas_treino=150)
    segunda = probabilidades_fora_da_dobra(_modelo(), tabela, max_linhas_treino=150)
    assert primeira.equals(segunda)


def test_taxa_por_municipio_e_a_media_por_grupo() -> None:
    valores = pd.Series([1.0, 0.0, 0.5, 0.5])
    grupos = pd.Series(["a", "a", "b", "b"])
    assert taxa_por_municipio(valores, grupos).to_dict() == {"a": 0.5, "b": 0.5}


def test_ranking_ordena_pelo_risco_e_numera_a_posicao_geral_e_na_uf() -> None:
    risco = pd.Series({"1": 0.9, "2": 0.2, "3": 0.6, "4": 0.7})
    municipios = pd.DataFrame(
        {"id_municipio": ["1", "2", "3", "4"], "nome": ["A", "B", "C", "D"], "sigla_uf": ["SP", "SP", "BA", "BA"]}
    )
    ranking = montar_ranking(risco, municipios, nome_risco="prob_nao_atingir_meta")
    assert list(ranking["id_municipio"]) == ["1", "4", "3", "2"]
    assert list(ranking["posicao"]) == [1, 2, 3, 4]
    assert ranking.set_index("id_municipio")["posicao_na_uf"].to_dict() == {"1": 1, "2": 2, "4": 1, "3": 2}
    assert ranking.set_index("id_municipio")["quintil_risco"].to_dict()["1"] == 5


def test_ranking_recusa_municipio_sem_cadastro() -> None:
    with pytest.raises(ValueError, match="99"):
        montar_ranking(pd.Series({"99": 0.5}), pd.DataFrame({"id_municipio": ["1"], "nome": ["A"], "sigla_uf": ["SP"]}), "risco")
