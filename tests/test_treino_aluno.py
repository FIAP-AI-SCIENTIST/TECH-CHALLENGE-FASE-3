from src.config import COLUNAS_T1
from src.modeling.pipeline_modelo import montar_pipeline
from src.modeling import treino_aluno
from src.modeling.treino_aluno import CANDIDATOS_T1


def test_candidatos_de_t1_tem_nomes_unicos_e_cobrem_os_tres_algoritmos() -> None:
    nomes = [candidato.nome for candidato in CANDIDATOS_T1]
    assert len(nomes) == len(set(nomes))
    assert set(nomes) == {"regressao_logistica", "random_forest", "gradient_boosting"}


def test_toda_chave_da_grade_e_um_parametro_do_pipeline() -> None:
    for candidato in CANDIDATOS_T1:
        parametros = montar_pipeline(COLUNAS_T1, candidato.estimador, candidato.escalonar).get_params()
        assert set(candidato.grade) <= set(parametros), candidato.nome


def test_estimadores_aleatorios_usam_a_semente_do_projeto() -> None:
    for candidato in CANDIDATOS_T1:
        if "random_state" in candidato.estimador.get_params():
            assert candidato.estimador.get_params()["random_state"] == 42, candidato.nome


def test_executar_encaminha_os_limites_de_amostragem(monkeypatch) -> None:
    tabela = object()
    chamadas = []

    def treinar_falso(
        trilha,
        recebida,
        candidatos,
        max_linhas_treino=None,
        max_linhas_validacao=None,
    ):
        assert recebida is tabela
        chamadas.append((max_linhas_treino, max_linhas_validacao))
        return {
            "modelo_escolhido": "exemplo",
            "teste": {"auc_roc": 0.5},
        }, None

    monkeypatch.setattr(
        treino_aluno, "carregar_tabela_t1", lambda: tabela
    )
    monkeypatch.setattr(
        treino_aluno, "MAX_LINHAS_TREINO", 100
    )
    monkeypatch.setattr(
        treino_aluno, "MAX_LINHAS_VALIDACAO", 50
    )
    monkeypatch.setattr(
        treino_aluno, "treinar_trilha", treinar_falso
    )
    monkeypatch.setattr(
        treino_aluno,
        "gravar_relatorio",
        lambda *args: ("relatorio.json", "comparacao.csv"),
    )

    treino_aluno.executar()

    assert chamadas == [(100, 50)]


def test_execucao_respeita_amostras_sem_misturar_municipios(monkeypatch) -> None:
    from src.modeling import treino
    from tests.test_treino import TRILHA, CANDIDATOS, _tabela

    tabela = _tabela()
    amostras = []
    amostrar_original = treino.amostrar

    def amostrar_observando(recebida, n_linhas, semente):
        resultado = amostrar_original(recebida, n_linhas, semente)
        amostras.append(resultado)
        return resultado

    monkeypatch.setattr(treino, "amostrar", amostrar_observando)
    monkeypatch.setattr(
        treino_aluno, "carregar_tabela_t1", lambda: tabela
    )
    monkeypatch.setattr(treino_aluno, "COLUNAS_T1", TRILHA)
    monkeypatch.setattr(treino_aluno, "CANDIDATOS_T1", CANDIDATOS[:1])
    monkeypatch.setattr(treino_aluno, "MAX_LINHAS_TREINO", 200)
    monkeypatch.setattr(treino_aluno, "MAX_LINHAS_VALIDACAO", 100)
    monkeypatch.setattr(
        treino_aluno,
        "gravar_relatorio",
        lambda *args: ("relatorio.json", "comparacao.csv"),
    )

    relatorio = treino_aluno.executar()

    assert len(amostras) == 2
    assert len(amostras[0].alvo) == 200
    assert len(amostras[1].alvo) == 100
    assert set(amostras[0].grupos).isdisjoint(amostras[1].grupos)
    assert relatorio["tamanhos"]["treino"]["linhas"] == 200
    assert relatorio["tamanhos"]["validacao"]["linhas"] == 100
    assert relatorio["tamanhos"]["teste"]["linhas"] == 240