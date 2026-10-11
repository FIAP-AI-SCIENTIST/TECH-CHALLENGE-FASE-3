import pytest

from src.config import COLUNAS_T1, COLUNAS_T2
from src.preprocessing.atributos import colunas_proibidas_presentes, validar_atributos


def test_atributos_de_cada_trilha_nao_incluem_colunas_proibidas() -> None:
    validar_atributos(COLUNAS_T1.atributos, COLUNAS_T1)
    validar_atributos(COLUNAS_T2.atributos, COLUNAS_T2)


@pytest.mark.parametrize("coluna", ["proficiencia", "presenca", "preenchimento_caderno", "id_municipio", "taxa_alfabetizacao"])
def test_t1_rejeita_coluna_que_define_ou_vaza_o_rotulo(coluna: str) -> None:
    with pytest.raises(ValueError, match=coluna):
        validar_atributos(["idhm", coluna], COLUNAS_T1)


def test_t1_rejeita_proporcao_por_nivel_do_mesmo_ano_pelo_prefixo() -> None:
    assert colunas_proibidas_presentes(["idhm", "proporcao_aluno_nivel_3"], COLUNAS_T1) == ["proporcao_aluno_nivel_3"]


def test_t2_rejeita_resultado_de_2024_e_aceita_meta_e_folga_defasadas() -> None:
    with pytest.raises(ValueError, match="taxa_alfabetizacao_2024"):
        validar_atributos(["taxa_alfabetizacao_2024"], COLUNAS_T2)
    validar_atributos(["taxa_alfabetizacao_2023", "meta_indicador_2024", "folga_2023"], COLUNAS_T2)


def test_mensagem_lista_todas_as_proibidas_em_ordem() -> None:
    with pytest.raises(ValueError) as erro:
        validar_atributos(["proficiencia", "idhm", "alfabetizado"], COLUNAS_T1)
    assert "alfabetizado, proficiencia" in str(erro.value)
