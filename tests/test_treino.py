import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import ColunasTrilha
from src.modeling.treino import Candidato, amostrar, gravar_relatorio, treinar_trilha
from src.preprocessing.tabelas import TabelaModelagem

TRILHA = ColunasTrilha(categoricas=("cor", "nome_regiao"), numericas=("x",), proibidas=())
CANDIDATOS = (
    Candidato("regressao_logistica", LogisticRegression(max_iter=200), True, {"estimador__C": [0.01, 1.0]}),
    Candidato(
        "random_forest",
        RandomForestClassifier(n_estimators=20, random_state=42),
        False,
        {"estimador__max_depth": [2, 4, None]},
        busca="aleatoria",
        n_iteracoes=2,
    ),
)


def _tabela(n_municipios: int = 80, alunos_por_municipio: int = 15, semente: int = 5) -> TabelaModelagem:
    gerador = np.random.default_rng(semente)
    grupos = np.repeat(np.arange(n_municipios).astype(str), alunos_por_municipio)
    regiao_do_municipio = gerador.choice(["Norte", "Sul"], n_municipios)
    n = len(grupos)
    x = gerador.normal(size=n)
    cor = gerador.choice(["azul", "verde"], n)
    alvo = (x + (cor == "azul") + gerador.normal(scale=0.7, size=n) > 0.5).astype(int)
    atributos = pd.DataFrame({"x": x, "cor": cor, "nome_regiao": np.repeat(regiao_do_municipio, alunos_por_municipio)})
    return TabelaModelagem(atributos, pd.Series(alvo, name="alvo"), pd.Series(grupos, name="id_municipio"))


def test_amostra_tem_o_tamanho_pedido_e_mantem_atributos_alvo_e_grupo_alinhados() -> None:
    tabela = _tabela()
    amostra = amostrar(tabela, 300)
    assert len(amostra.atributos) == len(amostra.alvo) == len(amostra.grupos) == 300
    original = tabela.atributos.assign(alvo=tabela.alvo, grupo=tabela.grupos)
    sorteada = amostra.atributos.assign(alvo=amostra.alvo, grupo=amostra.grupos)
    assert len(sorteada.merge(original, how="inner")) >= 300


def test_amostra_e_reprodutivel_pela_semente_e_nao_corta_tabela_menor() -> None:
    tabela = _tabela()
    assert amostrar(tabela, 300).atributos.equals(amostrar(tabela, 300).atributos)
    assert len(amostrar(tabela, 10_000).atributos) == len(tabela.atributos)


def test_relatorio_traz_tamanhos_baselines_comparacao_validacao_e_teste() -> None:
    relatorio, _ = treinar_trilha(TRILHA, _tabela(), CANDIDATOS)
    assert set(relatorio) >= {"tamanhos", "baselines", "candidatos", "comparacao", "modelo_escolhido", "validacao", "teste"}
    assert set(relatorio["tamanhos"]) == {"treino", "validacao", "teste"}
    assert set(relatorio["baselines"]) == {"dummy", "regressao_logistica"}
    assert {linha["modelo"] for linha in relatorio["comparacao"]} == {
        "dummy",
        "regressao_logistica_padrao",
        "regressao_logistica",
        "random_forest",
    }
    assert {"auc_roc", "auc_pr", "precisao", "recall", "f1", "matriz_confusao"} <= set(relatorio["teste"])
    assert "overfitting" in relatorio["validacao"]["diagnostico"]


def test_escolhe_o_candidato_de_maior_auc_roc_de_validacao_cruzada() -> None:
    relatorio, _ = treinar_trilha(TRILHA, _tabela(), CANDIDATOS)
    candidatos = {nome: dados["auc_roc_cv"] for nome, dados in relatorio["candidatos"].items()}
    assert relatorio["modelo_escolhido"] == max(candidatos, key=candidatos.get)


def test_nenhum_municipio_aparece_em_dois_conjuntos() -> None:
    relatorio, _ = treinar_trilha(TRILHA, _tabela(), CANDIDATOS)
    municipios = [relatorio["tamanhos"][conjunto]["municipios"] for conjunto in ("treino", "validacao", "teste")]
    assert sum(municipios) == 80


def test_limite_de_linhas_reduz_treino_e_validacao_mas_nao_o_teste() -> None:
    tabela = _tabela()
    completo, _ = treinar_trilha(TRILHA, tabela, CANDIDATOS)
    limitado, _ = treinar_trilha(TRILHA, tabela, CANDIDATOS, max_linhas_treino=200, max_linhas_validacao=100)
    assert limitado["tamanhos"]["treino"]["linhas"] == 200
    assert limitado["tamanhos"]["validacao"]["linhas"] == 100
    assert limitado["tamanhos"]["teste"]["linhas"] == completo["tamanhos"]["teste"]["linhas"]


def test_modelo_final_preve_probabilidades_para_dados_novos() -> None:
    tabela = _tabela()
    _, modelo = treinar_trilha(TRILHA, tabela, CANDIDATOS)
    probabilidades = modelo.predict_proba(tabela.atributos.head(5))[:, 1]
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()


def test_relatorio_vira_json_e_se_repete_entre_execucoes() -> None:
    primeiro, _ = treinar_trilha(TRILHA, _tabela(), CANDIDATOS)
    segundo, _ = treinar_trilha(TRILHA, _tabela(), CANDIDATOS)
    assert json.dumps(primeiro, sort_keys=True) == json.dumps(segundo, sort_keys=True)


def test_nome_de_candidato_repetido_e_recusado() -> None:
    with pytest.raises(ValueError, match="repetido"):
        treinar_trilha(TRILHA, _tabela(), (CANDIDATOS[0], CANDIDATOS[0]))


def test_gravar_relatorio_escreve_o_json_e_a_tabela_de_comparacao(tmp_path) -> None:
    relatorio, _ = treinar_trilha(TRILHA, _tabela(), CANDIDATOS)
    caminho_json, caminho_csv = gravar_relatorio(relatorio, "teste", tmp_path)
    assert json.loads(caminho_json.read_text(encoding="utf-8"))["modelo_escolhido"] == relatorio["modelo_escolhido"]
    comparacao = pd.read_csv(caminho_csv)
    assert list(comparacao["modelo"]) == [linha["modelo"] for linha in relatorio["comparacao"]]
