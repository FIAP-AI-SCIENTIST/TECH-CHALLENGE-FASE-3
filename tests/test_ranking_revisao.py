import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from src.modeling.ranking import probabilidades_fora_da_dobra
from src.modeling.ranking_municipios import ranking_taxa_observada, abaixo_do_perfil, proporcao_alto_risco_por_uf
from src.preprocessing.tabelas import TabelaModelagem


class DetectorDeMunicipio(ClassifierMixin, BaseEstimator):
    tamanhos = []

    def fit(self, x, y):
        self.vistos_ = set(x['municipio'])
        self.classes_ = np.array([0, 1])
        type(self).tamanhos.append(len(x))
        return self

    def predict_proba(self, x):
        p = np.where(x['municipio'].isin(self.vistos_), 1.0, 0.5)
        return np.column_stack([1 - p, p])


class ProbabilidadeConstante(ClassifierMixin, BaseEstimator):
    def fit(self, x, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x):
        return np.tile([0.1, 0.9], (len(x), 1))


def tabela():
    grupos = pd.Series(np.repeat(np.arange(20), 20))
    return TabelaModelagem(pd.DataFrame({'municipio': grupos}), pd.Series(np.tile([0, 1], 200)), grupos)


def test_dobras_nao_misturam_registros_do_mesmo_municipio():
    p = probabilidades_fora_da_dobra(DetectorDeMunicipio(), tabela())
    np.testing.assert_allclose(p, 0.5)


def test_fora_da_dobra_usa_probabilidade_da_classe_positiva():
    p = probabilidades_fora_da_dobra(ProbabilidadeConstante(), tabela())
    np.testing.assert_allclose(p, 0.9)


def test_limite_e_aplicado_ao_fit_de_todas_as_dobras():
    DetectorDeMunicipio.tamanhos = []
    p = probabilidades_fora_da_dobra(DetectorDeMunicipio(), tabela(), max_linhas_treino=150)
    assert DetectorDeMunicipio.tamanhos == [150] * 5
    assert len(p) == 400


def test_taxa_observada_e_excesso_respondem_perguntas_diferentes():
    perfil = pd.DataFrame({'id_municipio': ['a', 'b', 'c'], 'n_alunos': [100, 100, 49],
        'taxa_nao_alfabetizados_real': [0.9, 0.7, 1.0],
        'taxa_nao_alfabetizados_esperada': [0.85, 0.2, 0.1]})
    assert ranking_taxa_observada(perfil).id_municipio.tolist() == ['a', 'b']
    assert abaixo_do_perfil(perfil).id_municipio.tolist() == ['b', 'a']


def test_empate_entre_ufs_tem_ordem_alfabetica_estavel():
    x = pd.DataFrame({'sigla_uf': ['SP', 'BA', 'AC'], 'quintil_risco': [1, 1, 1]})
    assert proporcao_alto_risco_por_uf(x).sigla_uf.tolist() == ['AC', 'BA', 'SP']
    pd.testing.assert_frame_equal(proporcao_alto_risco_por_uf(x), proporcao_alto_risco_por_uf(x.iloc[::-1]))
