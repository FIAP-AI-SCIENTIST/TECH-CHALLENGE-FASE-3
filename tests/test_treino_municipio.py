import pandas as pd

from src.modeling import treino_municipio
from src.preprocessing.tabelas import TabelaModelagem
from sklearn.model_selection import ParameterGrid

from src.config import COLUNAS_T2, SEED
from src.modeling.pipeline_modelo import montar_pipeline
from src.modeling.treino_municipio import CANDIDATOS_T2


def test_candidatos_de_t2_tem_nomes_unicos_e_cobrem_os_tres_algoritmos() -> None:
    nomes = [candidato.nome for candidato in CANDIDATOS_T2]
    assert len(nomes) == len(set(nomes))
    assert set(nomes) == {"regressao_logistica", "random_forest", "gradient_boosting"}


def test_toda_chave_da_grade_e_um_parametro_do_pipeline() -> None:
    for candidato in CANDIDATOS_T2:
        parametros = montar_pipeline(COLUNAS_T2, candidato.estimador, candidato.escalonar).get_params()
        assert set(candidato.grade) <= set(parametros), candidato.nome


def test_estimadores_aleatorios_usam_a_semente_do_projeto() -> None:
    for candidato in CANDIDATOS_T2:
        if "random_state" in candidato.estimador.get_params():
            assert candidato.estimador.get_params()["random_state"] == SEED, candidato.nome


def test_analise_uf_usa_os_atributos_e_grupos_corretos(monkeypatch) -> None:
    atributos = pd.DataFrame({
        coluna: [0.6, 0.8, 0.7, 0.9]
        for coluna in COLUNAS_T2.numericas
    })
    atributos["sigla_uf"] = ["AC", "AC", "SP", "SP"]
    atributos["nome_regiao"] = ["Norte", "Norte", "Sudeste", "Sudeste"]

    treino = TabelaModelagem(
        atributos=atributos,
        alvo=pd.Series([0, 1, 0, 1]),
        grupos=pd.Series(["A", "B", "C", "D"]),
    )
    chamadas = []

    def validar_falso(modelo, tabela, n_dobras):
        transformadores = modelo.named_steps["pre_processamento"].transformers
        colunas = {
            nome: list(selecionadas)
            for nome, _, selecionadas in transformadores
        }
        chamadas.append((tabela, colunas, n_dobras))
        return {"auc_roc_media": 0.5}

    monkeypatch.setattr(
        treino_municipio, "validar_com_cv", validar_falso
    )

    resultado = treino_municipio.analisar_sinal_uf(
        treino, n_dobras=2
    )

    assert resultado["escopo"] == "somente treino"
    assert resultado["n_municipios"] == 4
    assert len(chamadas) == 3
    assert all(tabela.atributos is atributos for tabela, _, _ in chamadas)
    assert all(tabela.alvo is treino.alvo for tabela, _, _ in chamadas)
    assert all(dobras == 2 for _, _, dobras in chamadas)

    assert chamadas[0][1]["numericos"] == []
    assert chamadas[0][1]["categoricos"] == ["sigla_uf", "nome_regiao"]
    assert chamadas[1][1]["categoricos"] == []
    assert chamadas[1][1]["numericos"] == list(COLUNAS_T2.numericas)
    assert chamadas[2][1]["categoricos"] == list(COLUNAS_T2.categoricas)
    assert chamadas[2][1]["numericos"] == list(COLUNAS_T2.numericas)

    assert chamadas[0][0].grupos.equals(treino.grupos)
    assert chamadas[1][0].grupos.equals(treino.grupos)
    assert chamadas[2][0].grupos.equals(atributos["sigla_uf"])


def test_executar_analisa_so_o_treino_e_grava_o_resultado(monkeypatch) -> None:
    atributos = pd.DataFrame({
        "sigla_uf": ["AC", "AC", "SP", "SP", "CE", "CE"],
        "nome_regiao": ["Norte"] * 6,
    })
    tabela = TabelaModelagem(
        atributos=atributos,
        alvo=pd.Series([0, 1, 0, 1, 0, 1]),
        grupos=pd.Series(["A", "B", "C", "D", "E", "F"]),
    )

    def parte(inicio, fim):
        return TabelaModelagem(
            atributos=tabela.atributos.iloc[inicio:fim].reset_index(drop=True),
            alvo=tabela.alvo.iloc[inicio:fim].reset_index(drop=True),
            grupos=tabela.grupos.iloc[inicio:fim].reset_index(drop=True),
        )

    treino = parte(0, 2)
    validacao = parte(2, 4)
    teste = parte(4, 6)
    eventos = []
    analise = {"escopo": "somente treino", "n_municipios": 2}

    def separar_falso(recebida, semente):
        assert recebida is tabela
        assert semente == SEED
        return treino, validacao, teste

    def analisar_falso(recebida):
        assert recebida is treino
        eventos.append("analise")
        return analise

    def treinar_falso(trilha, recebida, candidatos, semente):
        assert recebida is tabela
        assert semente == SEED
        eventos.append("treinamento")
        return {
            "modelo_escolhido": "exemplo",
            "teste": {"auc_roc": 0.5},
        }, None

    def gravar_falso(relatorio, prefixo):
        assert relatorio["analise_uf"] == analise
        assert prefixo == "t2"
        eventos.append("gravacao")
        return "relatorio.json", "comparacao.csv"

    monkeypatch.setattr(
        treino_municipio, "carregar_tabela_t2", lambda: tabela
    )
    monkeypatch.setattr(
        treino_municipio, "separar_por_municipio", separar_falso
    )
    monkeypatch.setattr(
        treino_municipio, "analisar_sinal_uf", analisar_falso
    )
    monkeypatch.setattr(
        treino_municipio, "treinar_trilha", treinar_falso
    )
    monkeypatch.setattr(
        treino_municipio, "gravar_relatorio", gravar_falso
    )

    resultado = treino_municipio.executar()

    assert resultado["analise_uf"] == analise
    assert eventos == ["analise", "treinamento", "gravacao"]


def test_random_forest_avalia_a_grade_completa() -> None:
    candidato = next(
        item for item in CANDIDATOS_T2
        if item.nome == "random_forest"
    )

    assert candidato.busca == "grade"

    combinacoes = list(ParameterGrid(candidato.grade))
    assert len(combinacoes) == 24
    assert any(
        parametros["estimador__max_depth"] is None
        and parametros["estimador__min_samples_leaf"] == 20
        for parametros in combinacoes
    )