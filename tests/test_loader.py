from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.loader import carregar_tabela


def test_le_o_parquet_da_tabela_pelo_nome(tmp_path: Path) -> None:
    pd.DataFrame({"sigla_uf": ["SP", "BA"]}).to_parquet(tmp_path / "dim_uf.parquet")
    assert carregar_tabela("dim_uf", tmp_path)["sigla_uf"].tolist() == ["SP", "BA"]


def test_nome_fora_do_snapshot_falha_com_as_opcoes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Tabela desconhecida: inexistente.*fact_alunos"):
        carregar_tabela("inexistente", tmp_path)


def test_arquivo_ausente_orienta_a_gerar_o_snapshot(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="export_gold"):
        carregar_tabela("dim_uf", tmp_path)
