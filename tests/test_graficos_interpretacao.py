import numpy as np
import pandas as pd
import pytest

from src.visualization.graficos_interpretacao import barras_importancia, nome_legivel, resumo_shap


def test_barras_de_importancia_gravam_png(tmp_path) -> None:
    tabela = pd.DataFrame({"atributo": ["a", "b", "c"], "valor": [0.3, 0.2, 0.05], "desvio": [0.01, 0.02, 0.01]})
    caminho = barras_importancia(tabela, "atributo", "valor", "Título", "Queda da AUC-ROC", "b.png", tmp_path, coluna_erro="desvio")
    assert caminho == tmp_path / "b.png" and caminho.stat().st_size > 0


def test_barras_de_importancia_aceitam_valores_negativos_e_limitam_o_numero(tmp_path) -> None:
    tabela = pd.DataFrame({"atributo": list("abcdef"), "valor": [0.5, -0.4, 0.3, -0.2, 0.1, 0.0]})
    assert barras_importancia(tabela, "atributo", "valor", "T", "Coeficiente", "n.png", tmp_path, maximo=3).exists()


def test_barras_de_importancia_recusam_coluna_ausente(tmp_path) -> None:
    with pytest.raises(ValueError, match="valor"):
        barras_importancia(pd.DataFrame({"atributo": ["a"]}), "atributo", "valor", "T", "X", "x.png", tmp_path)


def test_resumo_shap_grava_png(tmp_path) -> None:
    gerador = np.random.default_rng(0)
    transformados = pd.DataFrame(gerador.normal(size=(60, 3)), columns=["numericos__a", "numericos__b", "categoricos__c_x"])
    shap_valores = transformados * [0.5, 0.1, 0.2]
    caminho = resumo_shap(shap_valores, transformados, "SHAP", "s.png", tmp_path)
    assert caminho.exists() and caminho.stat().st_size > 0


def test_nome_legivel_tira_o_prefixo_do_pre_processador() -> None:
    assert nome_legivel("categoricos__sigla_uf_SP") == "sigla_uf_SP"
    assert nome_legivel("numericos__idhm") == "idhm"
    assert nome_legivel("idhm") == "idhm"
