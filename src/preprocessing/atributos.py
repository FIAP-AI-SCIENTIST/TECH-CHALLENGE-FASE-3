"""Validação dos atributos de cada trilha contra as colunas proibidas."""

from collections.abc import Iterable

from src.config import ColunasTrilha


def colunas_proibidas_presentes(colunas: Iterable[str], trilha: ColunasTrilha) -> list[str]:
    """Colunas da lista que a trilha proíbe, pelo nome exato ou pelo prefixo, em ordem alfabética."""
    return sorted(
        {
            coluna
            for coluna in colunas
            if coluna in trilha.proibidas or coluna.startswith(trilha.prefixos_proibidos)
        }
    )


def validar_atributos(colunas: Iterable[str], trilha: ColunasTrilha) -> None:
    """Falha com `ValueError` se alguma coluna proibida da trilha estiver entre as `colunas`."""
    proibidas = colunas_proibidas_presentes(colunas, trilha)
    if proibidas:
        raise ValueError(f"Colunas proibidas como feature: {', '.join(proibidas)}.")
