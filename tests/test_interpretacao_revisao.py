import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from src.config import ColunasTrilha
from src.evaluation.interpretacao import atributo_original, importancia_nativa, importancia_por_permutacao, valores_shap
from src.modeling.pipeline_modelo import montar_pipeline


def dados(n=1300):
    rng = np.random.default_rng(13)
    x = pd.DataFrame(rng.normal(size=(n, 2)), columns=['positivo', 'negativo'])
    y = (2 * x.positivo - 2 * x.negativo + rng.normal(size=n) > 0).astype(int)
    return x, y


def modelo(estimador, n=1300):
    x, y = dados(n)
    trilha = ColunasTrilha(categoricas=(), numericas=tuple(x.columns), proibidas=())
    return montar_pipeline(trilha, estimador, escalonar=True).fit(x, y), x, y


def test_nome_original_escolhe_prefixo_mais_longo():
    assert atributo_original('numericos__idhm_renda', ('idhm', 'idhm_renda')) == 'idhm_renda'
    assert atributo_original('categoricos__idhm_renda_alta', ('idhm', 'idhm_renda')) == 'idhm_renda'


def test_coeficientes_preservam_sinais_opostos():
    m, _, _ = modelo(LogisticRegression())
    tabela = importancia_nativa(m).set_index('atributo_transformado').importancia
    assert tabela['numericos__positivo'] > 0
    assert tabela['numericos__negativo'] < 0


def test_permutacao_confere_auc_roc_e_nao_acuracia():
    m, x, y = modelo(LogisticRegression(), 300)
    esperado = permutation_importance(m, x, y, scoring='roc_auc', n_repeats=3, random_state=17)
    obtido = importancia_por_permutacao(m, x, y, n_repeticoes=3, semente=17).set_index('atributo').loc[x.columns]
    np.testing.assert_allclose(obtido.queda_auc_media, esperado.importances_mean)
    np.testing.assert_allclose(obtido.queda_auc_desvio, esperado.importances_std)
    acuracia = permutation_importance(m, x, y, scoring='accuracy', n_repeats=3, random_state=17)
    assert not np.allclose(esperado.importances_mean, acuracia.importances_mean)


@pytest.mark.parametrize('estimador,unidade', [
    (LogisticRegression(), 'log-odds'),
    (GradientBoostingClassifier(n_estimators=15, random_state=13), 'log-odds'),
    (RandomForestClassifier(n_estimators=15, random_state=13), 'probabilidade'),
])
def test_shap_recompoe_previsao_na_escala_do_modelo(estimador, unidade):
    m, x, _ = modelo(estimador, 300)
    amostra = x.head(35)
    valores, _ = valores_shap(m, amostra)
    esperado = m.predict_proba(amostra)[:, 1] if unidade == 'probabilidade' else m.decision_function(amostra)
    np.testing.assert_allclose(valores.sum(axis=1) + valores.attrs['valor_base'], esperado, atol=1e-7)
    assert valores.attrs['unidade'] == unidade


def test_fundo_shap_acima_do_limite_respeita_semente():
    m, x, _ = modelo(LogisticRegression())
    a, _ = valores_shap(m, x, semente=17)
    b, _ = valores_shap(m, x, semente=17)
    c, _ = valores_shap(m, x, semente=18)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a, c)

def test_resumo_shap_repete_png_com_a_mesma_semente(tmp_path):
    from src.visualization.graficos_interpretacao import resumo_shap
    rng = np.random.default_rng(17)
    valores = pd.DataFrame(rng.normal(size=(80, 3)), columns=['a', 'b', 'c'])
    atributos = pd.DataFrame(rng.normal(size=(80, 3)), columns=valores.columns)
    a = resumo_shap(valores, atributos, 'SHAP', 'a.png', tmp_path, semente=17)
    b = resumo_shap(valores, atributos, 'SHAP', 'b.png', tmp_path, semente=17)
    assert a.read_bytes() == b.read_bytes()
