import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import COLUNAS_T1, ColunasTrilha
from src.modeling.pipeline_modelo import montar_pipeline

TRILHA = ColunasTrilha(categoricas=("cor",), numericas=("x",), proibidas=())


def _dados(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    gerador = np.random.default_rng(42)
    x = gerador.normal(size=n)
    cor = gerador.choice(["azul", "verde"], n)
    alvo = pd.Series((x + (cor == "azul") + gerador.normal(scale=0.5, size=n) > 0.5).astype(int))
    atributos = pd.DataFrame({"x": x, "cor": cor, "id_municipio": np.arange(n).astype(str)})
    atributos.loc[::10, "x"] = np.nan
    return atributos, alvo


def test_pipeline_ajusta_e_preve_direto_dos_dados_brutos() -> None:
    atributos, alvo = _dados()
    modelo = montar_pipeline(TRILHA, LogisticRegression(max_iter=200), escalonar=True).fit(atributos, alvo)
    probabilidades = modelo.predict_proba(atributos)
    assert probabilidades.shape == (len(atributos), 2)
    assert np.allclose(probabilidades.sum(axis=1), 1.0)


def test_pipeline_tem_o_preprocessamento_e_o_estimador_como_passos() -> None:
    modelo = montar_pipeline(TRILHA, LogisticRegression(), escalonar=False)
    assert isinstance(modelo, Pipeline)
    assert list(modelo.named_steps) == ["pre_processamento", "estimador"]


def test_preprocessamento_do_pipeline_ajusta_so_com_o_treino() -> None:
    treino = pd.DataFrame({"x": [1.0, 2.0, 3.0, 2.0], "cor": ["azul", "verde", "azul", "verde"]})  # mediana 2
    modelo = montar_pipeline(TRILHA, LogisticRegression(), escalonar=False).fit(treino, [0, 1, 0, 1])
    novos = pd.DataFrame({"x": [np.nan, 1000.0], "cor": ["azul", "azul"]})  # a mediana dos novos seria 1000
    assert modelo.named_steps["pre_processamento"].transform(novos)["numericos__x"].tolist() == [2.0, 1000.0]


def test_estimador_enxerga_as_colunas_com_nomes_legiveis() -> None:
    atributos, alvo = _dados()
    modelo = montar_pipeline(TRILHA, LogisticRegression(max_iter=200), escalonar=False).fit(atributos, alvo)
    nomes = list(modelo[:-1].get_feature_names_out())
    assert nomes == ["numericos__x", "categoricos__cor_azul", "categoricos__cor_verde"]


def test_clone_do_pipeline_nao_carrega_o_ajuste() -> None:
    atributos, alvo = _dados()
    ajustado = montar_pipeline(TRILHA, LogisticRegression(max_iter=200), escalonar=True).fit(atributos, alvo)
    copia = clone(ajustado)
    assert not hasattr(copia.named_steps["estimador"], "coef_")


def test_funciona_com_a_trilha_do_aluno_e_uma_floresta() -> None:
    gerador = np.random.default_rng(42)
    n = 120
    atributos = pd.DataFrame(
        {
            "rede": gerador.choice(["2", "3"], n),
            "caderno": gerador.choice(["1", "2"], n),
            "sigla_uf": gerador.choice(["SP", "BA", "RS"], n),
            "nome_regiao": gerador.choice(["Sudeste", "Nordeste", "Sul"], n),
            "capital_uf": gerador.integers(0, 2, n).astype(float),
            "idhm": gerador.random(n),
            "idhm_educacao": gerador.random(n),
            "idhm_renda": gerador.random(n),
            "idhm_longevidade": gerador.random(n),
        }
    )
    alvo = pd.Series(gerador.integers(0, 2, n))
    modelo = montar_pipeline(COLUNAS_T1, RandomForestClassifier(n_estimators=5, random_state=42), escalonar=False).fit(atributos, alvo)
    assert modelo.predict(atributos).shape == (n,)
