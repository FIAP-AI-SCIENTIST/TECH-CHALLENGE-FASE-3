import numpy as np
import pandas as pd
import pytest

from src.modeling.separacao import separar_por_municipio
from src.preprocessing.tabelas import TabelaModelagem

REGIOES = ["Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste"]


def _tabela(n_municipios: int = 1000, semente: int = 7) -> TabelaModelagem:
    gerador = np.random.default_rng(semente)
    regiao_do_municipio = gerador.choice(REGIOES, n_municipios)
    alunos_por_municipio = gerador.integers(3, 15, n_municipios)
    grupos = np.repeat(np.arange(n_municipios).astype(str), alunos_por_municipio)
    regiao = np.repeat(regiao_do_municipio, alunos_por_municipio)
    n = len(grupos)
    return TabelaModelagem(
        atributos=pd.DataFrame({"nome_regiao": regiao, "x": gerador.normal(size=n)}),
        alvo=pd.Series(gerador.integers(0, 2, n), name="alvo"),
        grupos=pd.Series(grupos, name="id_municipio"),
    )


def test_nenhum_municipio_aparece_em_mais_de_um_conjunto() -> None:
    treino, validacao, teste = separar_por_municipio(_tabela())
    conjuntos = [set(treino.grupos), set(validacao.grupos), set(teste.grupos)]
    assert conjuntos[0].isdisjoint(conjuntos[1])
    assert conjuntos[0].isdisjoint(conjuntos[2])
    assert conjuntos[1].isdisjoint(conjuntos[2])


def test_todas_as_linhas_vao_para_algum_conjunto() -> None:
    tabela = _tabela()
    partes = separar_por_municipio(tabela)
    assert sum(len(parte.alvo) for parte in partes) == len(tabela.alvo)
    assert set().union(*(set(parte.grupos) for parte in partes)) == set(tabela.grupos)


def test_proporcoes_de_linhas_ficam_a_menos_de_dois_pontos_de_60_20_20() -> None:
    tabela = _tabela()
    treino, validacao, teste = separar_por_municipio(tabela)
    total = len(tabela.alvo)
    for parte, esperado in ((treino, 0.6), (validacao, 0.2), (teste, 0.2)):
        assert abs(len(parte.alvo) / total - esperado) < 0.02


def test_regioes_ficam_balanceadas_nos_tres_conjuntos() -> None:
    tabela = _tabela()
    partes = separar_por_municipio(tabela)
    por_municipio = pd.DataFrame({"g": tabela.grupos, "r": tabela.atributos["nome_regiao"]}).drop_duplicates("g")
    for regiao in REGIOES:
        total_regiao = (por_municipio["r"] == regiao).sum()
        no_treino = len({g for g in partes[0].grupos if g in set(por_municipio.loc[por_municipio["r"] == regiao, "g"])})
        assert abs(no_treino / total_regiao - 0.6) < 0.02


def test_atributos_alvo_e_grupos_ficam_alinhados_em_cada_conjunto() -> None:
    for parte in separar_por_municipio(_tabela()):
        assert len(parte.atributos) == len(parte.alvo) == len(parte.grupos)
        assert parte.atributos.index.equals(parte.alvo.index) and parte.alvo.index.equals(parte.grupos.index)


def test_mesma_semente_repete_a_separacao_e_outra_semente_muda() -> None:
    tabela = _tabela()
    primeira = separar_por_municipio(tabela, semente=1)
    repetida = separar_por_municipio(tabela, semente=1)
    outra = separar_por_municipio(tabela, semente=2)
    assert set(primeira[0].grupos) == set(repetida[0].grupos)
    assert set(primeira[0].grupos) != set(outra[0].grupos)


def test_fracoes_que_nao_somam_um_sao_recusadas() -> None:
    with pytest.raises(ValueError, match="somar 1"):
        separar_por_municipio(_tabela(50), fracoes=(0.6, 0.3, 0.3))


def test_usa_a_semente_do_projeto_por_padrao() -> None:
    tabela = _tabela()
    padrao = separar_por_municipio(tabela)
    explicita = separar_por_municipio(tabela, semente=42)
    assert set(padrao[2].grupos) == set(explicita[2].grupos)
