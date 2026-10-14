import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import ColunasTrilha
from src.evaluation.interpretacao_modelos import gravar_interpretacao, interpretar
from src.modeling.pipeline_modelo import montar_pipeline
from src.preprocessing.tabelas import TabelaModelagem

TRILHA = ColunasTrilha(categoricas=("cor",), numericas=("forte", "ruido"), proibidas=())


def _tabela_e_modelo(n: int = 500, semente: int = 4):
    gerador = np.random.default_rng(semente)
    forte = gerador.normal(size=n)
    cor = gerador.choice(["azul", "verde"], n)
    alvo = pd.Series((forte + gerador.normal(scale=0.5, size=n) > 0).astype(int))
    atributos = pd.DataFrame({"forte": forte, "ruido": gerador.normal(size=n), "cor": cor})
    modelo = montar_pipeline(TRILHA, LogisticRegression(), escalonar=True).fit(atributos, alvo)
    return TabelaModelagem(atributos, alvo, pd.Series(np.arange(n).astype(str))), modelo


def test_interpretacao_traz_as_tres_leituras_com_os_atributos_da_trilha() -> None:
    tabela, modelo = _tabela_e_modelo()
    resultado = interpretar(modelo, TRILHA, tabela, n_permutacao=200, n_shap=100, n_repeticoes=2)
    assert set(resultado.permutacao["atributo"]) == set(TRILHA.atributos)
    assert set(resultado.shap_por_atributo["atributo"]) == set(TRILHA.atributos)
    assert resultado.shap_por_atributo.iloc[0]["atributo"] == "forte"
    assert len(resultado.shap_valores) == len(resultado.transformados) == 100
    assert len(resultado.nativa) == len(modelo.named_steps["pre_processamento"].get_feature_names_out())


def test_interpretacao_e_reprodutivel_pela_semente() -> None:
    tabela, modelo = _tabela_e_modelo()
    primeira = interpretar(modelo, TRILHA, tabela, n_permutacao=200, n_shap=100, n_repeticoes=2)
    segunda = interpretar(modelo, TRILHA, tabela, n_permutacao=200, n_shap=100, n_repeticoes=2)
    assert primeira.permutacao.equals(segunda.permutacao)
    assert primeira.shap_por_atributo.equals(segunda.shap_por_atributo)


def test_gravar_interpretacao_escreve_tabelas_e_graficos(tmp_path) -> None:
    tabela, modelo = _tabela_e_modelo()
    resultado = interpretar(modelo, TRILHA, tabela, n_permutacao=200, n_shap=100, n_repeticoes=2)
    caminhos = gravar_interpretacao(resultado, "t9", "T9 (teste)", tmp_path, tmp_path)
    nomes = {caminho.name for caminho in caminhos}
    assert {
        "interpretacao_nativa_t9.csv",
        "interpretacao_permutacao_t9.csv",
        "interpretacao_shap_t9.csv",
        "interpretacao_nativa_t9.png",
        "interpretacao_permutacao_t9.png",
        "interpretacao_shap_atributos_t9.png",
        "interpretacao_shap_resumo_t9.png",
    } == nomes
    assert all(caminho.stat().st_size > 0 for caminho in caminhos)
