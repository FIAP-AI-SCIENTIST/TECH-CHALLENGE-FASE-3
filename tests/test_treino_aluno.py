from src.config import COLUNAS_T1
from src.modeling.pipeline_modelo import montar_pipeline
from src.modeling.treino_aluno import CANDIDATOS_T1, MAX_LINHAS_TREINO, MAX_LINHAS_VALIDACAO


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


def test_amostra_do_aluno_e_menor_que_a_tabela_e_a_validacao_menor_que_o_treino() -> None:
    assert 0 < MAX_LINHAS_VALIDACAO < MAX_LINHAS_TREINO < 3_354_661
