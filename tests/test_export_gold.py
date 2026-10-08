from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.config import TABELAS_SNAPSHOT
from src.preprocessing import export_gold


class _Job:
    def __init__(self, tabela: pa.Table) -> None:
        self._tabela = tabela

    def to_arrow(self) -> pa.Table:
        return self._tabela


class _ClienteFalso:
    """Imita só o que `exportar_gold` usa do cliente do BigQuery."""

    def __init__(self) -> None:
        self.consultas: list[str] = []

    def get_table(self, referencia: str) -> SimpleNamespace:
        colunas = ["id_municipio", "sk_municipio", "valor"]
        if referencia.endswith(".fact_alunos"):
            colunas.append("id_aluno")
        return SimpleNamespace(schema=[SimpleNamespace(name=nome) for nome in colunas])

    def query(self, consulta: str) -> _Job:
        self.consultas.append(consulta)
        colunas = [c.strip("` ") for c in consulta.split("FROM")[0].replace("SELECT", "").split(",")]
        return _Job(pa.table({coluna: [1, 2] for coluna in colunas}))


def test_sem_projeto_falha_cedo_com_mensagem_clara(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    with pytest.raises(ValueError, match="GCP_PROJECT_ID"):
        export_gold.exportar_gold(destino=tmp_path)


def test_grava_uma_tabela_por_arquivo_sem_chaves_substitutas_nem_id_aluno(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cliente = _ClienteFalso()
    monkeypatch.setattr(export_gold.bigquery, "Client", lambda project: cliente)

    linhas = export_gold.exportar_gold(projeto="projeto-teste", destino=tmp_path)

    assert set(linhas) == set(TABELAS_SNAPSHOT)
    assert all((tmp_path / f"{tabela}.parquet").exists() for tabela in TABELAS_SNAPSHOT)
    assert pq.read_table(tmp_path / "fact_alunos.parquet").column_names == ["id_municipio", "valor"]
    assert not any("sk_municipio" in consulta or "id_aluno" in consulta for consulta in cliente.consultas)


def test_le_o_projeto_da_variavel_de_ambiente(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GCP_PROJECT_ID", "projeto-do-ambiente")
    projetos: list[str] = []

    def _cliente(project: str) -> _ClienteFalso:
        projetos.append(project)
        return _ClienteFalso()

    monkeypatch.setattr(export_gold.bigquery, "Client", _cliente)
    export_gold.exportar_gold(destino=tmp_path)
    assert projetos == ["projeto-do-ambiente"]
