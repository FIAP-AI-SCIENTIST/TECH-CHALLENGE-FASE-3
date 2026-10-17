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
