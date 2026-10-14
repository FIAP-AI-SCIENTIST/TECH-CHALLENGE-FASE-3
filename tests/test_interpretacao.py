import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import ColunasTrilha
from src.evaluation.interpretacao import (
    agregar_por_atributo,
    atributo_original,
    importancia_nativa,
    importancia_por_permutacao,
    valores_shap,
)
from src.modeling.pipeline_modelo import montar_pipeline

TRILHA = ColunasTrilha(categoricas=("cor",), numericas=("forte", "ruido"), proibidas=())


def _dados(n: int = 400, semente: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    gerador = np.random.default_rng(semente)
    forte = gerador.normal(size=n)
    cor = gerador.choice(["azul", "verde", "roxo"], n)
    alvo = (2 * forte + 0.5 * (cor == "azul") + gerador.normal(scale=0.5, size=n) > 0).astype(int)
    return pd.DataFrame({"forte": forte, "ruido": gerador.normal(size=n), "cor": cor}), pd.Series(alvo)


def _modelo(estimador, escalonar: bool):
    atributos, alvo = _dados()
    return montar_pipeline(TRILHA, estimador, escalonar).fit(atributos, alvo), atributos, alvo


@pytest.mark.parametrize(
    ("nome", "esperado"),
    [
        ("numericos__idhm", "idhm"),
        ("numericos__taxa_alfabetizacao_2023", "taxa_alfabetizacao_2023"),
        ("categoricos__sigla_uf_SP", "sigla_uf"),
        ("categoricos__nome_regiao_Centro-Oeste", "nome_regiao"),
        ("categoricos__caderno_1A", "caderno"),
    ],
)
def test_nome_transformado_volta_ao_atributo_original(nome: str, esperado: str) -> None:
    originais = ("idhm", "taxa_alfabetizacao_2023", "sigla_uf", "nome_regiao", "caderno")
    assert atributo_original(nome, originais) == esperado


def test_nome_desconhecido_e_recusado() -> None:
    with pytest.raises(ValueError, match="desconhecido"):
        atributo_original("numericos__outra", ("idhm",))


@pytest.mark.parametrize(
    ("estimador", "escalonar"),
    [
        (LogisticRegression(), True),
        (RandomForestClassifier(n_estimators=20, random_state=42), False),
        (GradientBoostingClassifier(n_estimators=20, random_state=42), False),
    ],
)
def test_importancia_nativa_poe_o_atributo_forte_em_primeiro(estimador, escalonar: bool) -> None:
    modelo, _, _ = _modelo(estimador, escalonar)
    importancia = importancia_nativa(modelo)
    assert list(importancia.columns) == ["atributo_transformado", "importancia"]
    assert importancia.iloc[0]["atributo_transformado"] == "numericos__forte"


def test_importancia_nativa_da_regressao_logistica_mantem_o_sinal_do_coeficiente() -> None:
    modelo, _, _ = _modelo(LogisticRegression(), True)
    importancia = importancia_nativa(modelo).set_index("atributo_transformado")["importancia"]
    assert importancia["numericos__forte"] > 0


def test_permutacao_usa_as_colunas_originais_e_poe_o_forte_em_primeiro() -> None:
    modelo, atributos, alvo = _modelo(LogisticRegression(), True)
    importancia = importancia_por_permutacao(modelo, atributos, alvo, n_repeticoes=3)
    assert set(importancia["atributo"]) == {"forte", "ruido", "cor"}
    assert importancia.iloc[0]["atributo"] == "forte"
    assert {"queda_auc_media", "queda_auc_desvio"} <= set(importancia.columns)


@pytest.mark.parametrize(
    ("estimador", "escalonar"),
    [(LogisticRegression(), True), (GradientBoostingClassifier(n_estimators=20, random_state=42), False)],
)
def test_shap_tem_uma_coluna_por_atributo_transformado_e_uma_linha_por_amostra(estimador, escalonar: bool) -> None:
    modelo, atributos, _ = _modelo(estimador, escalonar)
    shap_valores, transformados = valores_shap(modelo, atributos.head(50))
    nomes = list(modelo.named_steps["pre_processamento"].get_feature_names_out())
    assert list(shap_valores.columns) == nomes == list(transformados.columns)
    assert len(shap_valores) == len(transformados) == 50


def test_agregacao_soma_as_colunas_one_hot_e_ordena_pela_media_absoluta() -> None:
    shap_valores = pd.DataFrame(
        {
            "numericos__forte": [1.0, -1.0],
            "categoricos__cor_azul": [0.2, 0.2],
            "categoricos__cor_verde": [-0.1, 0.3],
        }
    )
    agregado = agregar_por_atributo(shap_valores, ("forte", "cor"))
    assert list(agregado["atributo"]) == ["forte", "cor"]
    assert agregado.set_index("atributo")["shap_medio_absoluto"].to_dict() == pytest.approx({"forte": 1.0, "cor": 0.3})
