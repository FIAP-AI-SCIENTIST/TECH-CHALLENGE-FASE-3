"""Tabelas de modelagem das trilhas supervisionadas, a partir do snapshot da Gold.

Cada construtor devolve os atributos, o alvo (classe positiva = o risco) e o
grupo da separação (o município). Nada aqui ajusta estatísticas: imputação,
escalonamento e codificação são do pré-processador, ajustado só com o treino.
"""

from dataclasses import dataclass

import pandas as pd

from src.config import ALVO_ALUNO, ATRIBUTOS_MUNICIPAIS_CATEGORICOS, ATRIBUTOS_MUNICIPAIS_NUMERICOS, COLUNAS_T1, COLUNAS_T2
from src.preprocessing.atributos import validar_atributos

ALVO_NAO_ALFABETIZADO: str = "nao_alfabetizado"
ALVO_NAO_ATINGIU_META: str = "nao_atingiu_meta"

# O rótulo de T2 só existe para 2024; os atributos defasados são de 2023.
ANO_ROTULO: int = 2024
ANO_DEFASADO: int = 2023
# Descrição da rede municipal em `dim_rede` e em `fact_meta_resultado_municipio`
# (em `fact_alunos` e `fact_indicador_municipio` a rede vem como código).
REDE_MUNICIPAL: str = "Municipal"

# Perfil socioeducacional usado no agrupamento de municípios (T3): o IDHM e seus
# componentes, a capital e as taxas de alfabetização dos dois anos.
ATRIBUTOS_AGRUPAMENTO_NUMERICOS: tuple[str, ...] = (
    "idhm",
    "idhm_educacao",
    "idhm_renda",
    "idhm_longevidade",
    "capital_uf",
    "taxa_alfabetizacao_2023",
    "taxa_alfabetizacao_2024",
)


@dataclass(frozen=True)
class TabelaModelagem:
    """Atributos, alvo e grupo (município) alinhados pelo mesmo índice."""

    atributos: pd.DataFrame
    alvo: pd.Series
    grupos: pd.Series


def construir_tabela_aluno(alunos: pd.DataFrame, municipios: pd.DataFrame) -> TabelaModelagem:
    """Tabela da trilha T1: só quem fez a prova, alvo `nao_alfabetizado` (1 = não alfabetizado).

    Quem deixou o caderno em branco tem `alfabetizado` falso por ausência, não por
    leitura, e fica fora. Os atributos municipais vêm de `dim_municipio` pelo
    `id_municipio`, que é devolvido como grupo e nunca como atributo.
    """
    validar_atributos(COLUNAS_T1.atributos, COLUNAS_T1)
    com_prova = alunos[alunos["preenchimento_caderno"]]
    colunas_municipio = ["id_municipio", *(c for c in COLUNAS_T1.atributos if c in municipios.columns)]
    tabela = com_prova.merge(municipios[colunas_municipio], on="id_municipio", how="left", validate="many_to_one")
    return TabelaModelagem(
        atributos=tabela[list(COLUNAS_T1.atributos)],
        alvo=(~tabela[ALVO_ALUNO]).astype(int).rename(ALVO_NAO_ALFABETIZADO),
        grupos=tabela["id_municipio"],
    )


def construir_tabela_municipio(
    metas: pd.DataFrame, indicadores: pd.DataFrame, municipios: pd.DataFrame, redes: pd.DataFrame
) -> TabelaModelagem:
    """Tabela da trilha T2: município e rede em 2024 com rótulo, alvo `nao_atingiu_meta` (1 = não atingiu).

    Os atributos são de antes do resultado: a taxa e a média de português de 2023, a
    meta de 2024 (conhecida de antemão), a folga (taxa de 2023 menos a meta de 2024) e
    os atributos municipais estáticos. Nada de 2024 além da meta entra, e municípios
    sem dados de 2023 ficam com os atributos defasados nulos para a imputação do treino.
    """
    validar_atributos(COLUNAS_T2.atributos, COLUNAS_T2)
    chave = ["id_municipio", "rede"]

    rotulos = metas[(metas["ano"] == ANO_ROTULO) & metas["atingiu_meta"].notna()][
        [*chave, "meta_indicador", "atingiu_meta"]
    ].rename(columns={"meta_indicador": "meta_indicador_2024"})
    taxas = metas[metas["ano"] == ANO_DEFASADO][[*chave, "taxa_alfabetizacao"]].rename(
        columns={"taxa_alfabetizacao": "taxa_alfabetizacao_2023"}
    )
    descricao_da_rede = redes.set_index("rede")["rede_desc"]
    medias = (
        indicadores[indicadores["ano"] == ANO_DEFASADO]
        .assign(rede=lambda dados: dados["rede"].map(descricao_da_rede))[[*chave, "media_portugues"]]
        .rename(columns={"media_portugues": "media_portugues_2023"})
    )
    estaticos = ["id_municipio", *(c for c in COLUNAS_T2.atributos if c in municipios.columns)]

    tabela = (
        rotulos.merge(taxas, on=chave, how="left", validate="one_to_one")
        .merge(medias, on=chave, how="left", validate="one_to_one")
        .merge(municipios[estaticos], on="id_municipio", how="left", validate="many_to_one")
        .sort_values(chave)
        .reset_index(drop=True)
    )
    tabela["folga_2023"] = tabela["taxa_alfabetizacao_2023"] - tabela["meta_indicador_2024"]
    return TabelaModelagem(
        atributos=tabela[list(COLUNAS_T2.atributos)],
        alvo=(~tabela["atingiu_meta"].astype(bool)).astype(int).rename(ALVO_NAO_ATINGIU_META),
        grupos=tabela["id_municipio"],
    )


def construir_atributos_municipio(municipios: pd.DataFrame, metas: pd.DataFrame) -> pd.DataFrame:
    """Perfil por município da rede municipal para o agrupamento (T3), uma linha por município.

    Reúne o IDHM e seus componentes, a capital e as taxas de alfabetização de 2023 e
    2024. Só entram municípios que têm linha na rede municipal de `metas`.
    """
    taxas = (
        metas[(metas["rede"] == REDE_MUNICIPAL) & metas["ano"].isin([ANO_DEFASADO, ANO_ROTULO])]
        .pivot(index="id_municipio", columns="ano", values="taxa_alfabetizacao")
        .rename(columns=lambda ano: f"taxa_alfabetizacao_{ano}")
        .reset_index()
    )
    descritivos = [*ATRIBUTOS_MUNICIPAIS_CATEGORICOS, *ATRIBUTOS_MUNICIPAIS_NUMERICOS]
    tabela = taxas.merge(municipios[["id_municipio", *descritivos]], on="id_municipio", how="left", validate="one_to_one")
    return tabela[["id_municipio", *ATRIBUTOS_MUNICIPAIS_CATEGORICOS, *ATRIBUTOS_AGRUPAMENTO_NUMERICOS]]
