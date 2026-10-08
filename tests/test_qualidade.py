from pathlib import Path

import pandas as pd

from src.config import TABELAS_SNAPSHOT
from src.preprocessing.qualidade import (
    anos_da_tabela,
    anos_suspeitos,
    colunas_totalmente_ausentes,
    descrever_colunas,
    gerar_dicionario,
    papel_da_coluna,
)


def test_papel_pelo_nome_exato_pelo_prefixo_e_o_padrao() -> None:
    assert papel_da_coluna("alfabetizado") == "alvo do aluno (T1)"
    assert papel_da_coluna("proporcao_aluno_nivel_3").startswith("resultado do mesmo ano")
    assert papel_da_coluna("rede") == "atributo candidato"


def test_descrever_colunas_calcula_ausentes_e_percentual() -> None:
    descricao = descrever_colunas("t", pd.DataFrame({"a": [1.0, None, 3.0, None], "b": [1, 2, 3, 4]}))
    por_coluna = descricao.set_index("coluna")
    assert por_coluna.loc["a", "ausentes"] == 2
    assert por_coluna.loc["a", "pct_ausentes"] == 50.0
    assert por_coluna.loc["b", "pct_ausentes"] == 0.0


def test_anos_da_tabela_conta_linhas_por_ano_e_ignora_tabela_sem_ano() -> None:
    assert anos_da_tabela(pd.DataFrame({"ano": [2024, 2023, 2024]})) == {2023: 1, 2024: 2}
    assert anos_da_tabela(pd.DataFrame({"x": [1]})) == {}


def test_so_fatos_com_ano_fora_dos_dados_reais_sao_suspeitos() -> None:
    suspeitos = anos_suspeitos({"fact_alunos": {2023: 1, 2026: 2}, "dim_tempo": {2030: 1}, "fact_indicador_uf": {2024: 3}})
    assert suspeitos == {"fact_alunos": [2026]}


def test_colunas_totalmente_ausentes_listam_tabela_e_coluna() -> None:
    descricoes = [
        descrever_colunas("t1", pd.DataFrame({"a": [None, None], "b": [1, 2]})),
        descrever_colunas("t2", pd.DataFrame({"c": [1, None]})),
    ]
    assert colunas_totalmente_ausentes(descricoes) == ["t1.a"]


def _gravar_snapshot(diretorio: Path, ano_extra: int | None = None) -> None:
    for tabela in TABELAS_SNAPSHOT:
        dados = pd.DataFrame({"x": [1, 2]})
        if tabela.startswith("fact_") or tabela == "dim_tempo":
            dados["ano"] = [2023, ano_extra or 2024]
        dados.to_parquet(diretorio / f"{tabela}.parquet")


def test_dicionario_tem_as_secoes_e_nao_aponta_ano_suspeito_em_dados_reais(tmp_path: Path) -> None:
    _gravar_snapshot(tmp_path)
    texto = gerar_dicionario(tmp_path)
    for secao in ("## Anos cobertos por tabela", "## Anos fora dos dados reais", "## Colunas totalmente ausentes", "## Colunas"):
        assert secao in texto
    assert "Nenhuma tabela de fato tem linhas fora de 2023 e 2024" in texto
    assert "### `fact_alunos`" in texto


def test_dicionario_aponta_fato_com_ano_fora_dos_dados_reais(tmp_path: Path) -> None:
    _gravar_snapshot(tmp_path, ano_extra=2026)
    texto = gerar_dicionario(tmp_path)
    assert "`fact_alunos`: anos 2026" in texto
