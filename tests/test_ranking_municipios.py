import pandas as pd
import pytest

from src.modeling.ranking_municipios import MINIMO_ALUNOS, abaixo_do_perfil, proporcao_alto_risco_por_uf


def _perfil() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_municipio": ["1", "2", "3", "4"],
            "nome": ["A", "B", "C", "D"],
            "sigla_uf": ["SP", "SP", "BA", "BA"],
            "n_alunos": [200, 200, MINIMO_ALUNOS - 1, 80],
            "taxa_nao_alfabetizados_real": [0.50, 0.30, 0.90, 0.45],
            "taxa_nao_alfabetizados_esperada": [0.40, 0.35, 0.40, 0.40],
        }
    )


def test_abaixo_do_perfil_ordena_pelo_excesso_e_descarta_municipio_pequeno() -> None:
    resultado = abaixo_do_perfil(_perfil())
    assert list(resultado["id_municipio"]) == ["1", "4", "2"]
    assert resultado["excesso_nao_alfabetizados"].tolist() == pytest.approx([0.10, 0.05, -0.05])
    assert list(resultado["posicao"]) == [1, 2, 3]


def test_proporcao_de_alto_risco_por_uf_conta_o_quintil_5() -> None:
    ranking = pd.DataFrame({"sigla_uf": ["SP", "SP", "BA", "BA"], "quintil_risco": [5, 1, 5, 5]})
    proporcao = proporcao_alto_risco_por_uf(ranking).set_index("sigla_uf")
    assert proporcao.loc["BA", "proporcao_quintil_5"] == pytest.approx(1.0)
    assert proporcao.loc["SP", "proporcao_quintil_5"] == pytest.approx(0.5)
    assert proporcao.loc["SP", "municipios"] == 2
