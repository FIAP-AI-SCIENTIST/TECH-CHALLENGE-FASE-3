"""Dicionário de dados e qualidade do snapshot da Gold.

Para cada tabela do snapshot, descreve as colunas (tipo, percentual de ausentes e
papel provisório) e os anos cobertos, e verifica se há anos fora dos dados reais.
Os papéis aqui são PROVISÓRIOS: a lista definitiva de atributos e de colunas
proibidas, por trilha, é da unidade de pré-processamento.

Uso:
    python -m src.preprocessing.qualidade
"""

from pathlib import Path

import pandas as pd

from src.config import DIRETORIO_DADOS, DIRETORIO_RELATORIOS, TABELAS_SNAPSHOT
from src.preprocessing.loader import carregar_tabela

ARQUIVO_DICIONARIO: str = "dicionario_dados.md"

# A Fase 2 só traz dados reais de 2023 e 2024 e, rodada sem o streaming, não
# tem eventos sintéticos. Qualquer linha de fato fora desses anos é suspeita.
ANOS_REAIS: tuple[int, ...] = (2023, 2024)
TABELAS_DE_FATO: tuple[str, ...] = tuple(t for t in TABELAS_SNAPSHOT if t.startswith("fact_"))

PAPEL_PADRAO: str = "atributo candidato"
PAPEIS: dict[str, str] = {
    "alfabetizado": "alvo do aluno (T1)",
    "atingiu_meta": "alvo do município (T2); nulo quando não há meta vigente",
    "id_municipio": "identificador; nunca feature",
    "id_escola": "identificador; nunca feature",
    "proficiencia": "proibida: define alfabetizado (corte 743)",
    "presenca": "proibida: caderno em branco implica alfabetizado falso",
    "preenchimento_caderno": "proibida: caderno em branco implica alfabetizado falso",
    "peso_aluno": "peso amostral; não é feature",
    "taxa_alfabetizacao": "proibida no mesmo ano do rótulo; só defasada",
    "meta_indicador": "conhecida de antemão; uso a decidir (define atingiu_meta com a taxa)",
    "gap_pontos": "proibida: diferença entre taxa e meta do mesmo ano",
    "media_portugues": "resultado do mesmo ano; só defasada",
    "nivel_alfabetizacao": "resultado do mesmo ano; só defasada",
    "valid_from": "controle SCD2",
    "valid_to": "controle SCD2",
    "is_current": "controle SCD2",
}
PREFIXO_PAPEIS: dict[str, str] = {
    "proporcao_aluno_nivel_": "resultado do mesmo ano (distribuição de proficiência); só defasada",
}


def papel_da_coluna(coluna: str) -> str:
    """Papel provisório de uma coluna, pelo nome exato ou pelo prefixo."""
    if coluna in PAPEIS:
        return PAPEIS[coluna]
    for prefixo, papel in PREFIXO_PAPEIS.items():
        if coluna.startswith(prefixo):
            return papel
    return PAPEL_PADRAO


def descrever_colunas(tabela: str, dados: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por coluna: tipo, número e percentual de ausentes, e papel provisório."""
    ausentes = dados.isna().sum()
    return pd.DataFrame(
        {
            "tabela": tabela,
            "coluna": dados.columns,
            "tipo": [str(tipo) for tipo in dados.dtypes],
            "ausentes": ausentes.values,
            "pct_ausentes": (ausentes.values / max(len(dados), 1)) * 100,
            "papel": [papel_da_coluna(coluna) for coluna in dados.columns],
        }
    )


def anos_da_tabela(dados: pd.DataFrame) -> dict[int, int]:
    """Linhas por ano; vazio se a tabela não tem a coluna `ano`."""
    if "ano" not in dados.columns:
        return {}
    return {int(ano): int(n) for ano, n in dados["ano"].value_counts().sort_index().items()}


def anos_suspeitos(anos_por_tabela: dict[str, dict[int, int]]) -> dict[str, list[int]]:
    """Anos fora de `ANOS_REAIS` encontrados nas tabelas de fato."""
    return {
        tabela: sorted(set(anos) - set(ANOS_REAIS))
        for tabela, anos in anos_por_tabela.items()
        if tabela in TABELAS_DE_FATO and set(anos) - set(ANOS_REAIS)
    }


def colunas_totalmente_ausentes(descricoes: list[pd.DataFrame]) -> list[str]:
    """Colunas (no formato `tabela.coluna`) em que todos os valores são nulos."""
    return [
        f"{linha.tabela}.{linha.coluna}"
        for descricao in descricoes
        for linha in descricao.itertuples()
        if linha.pct_ausentes >= 100
    ]


def _numero(valor: float) -> str:
    return f"{valor:.2f}".replace(".", ",")


def gerar_dicionario(diretorio: Path = DIRETORIO_DADOS) -> str:
    """Monta o dicionário de dados em Markdown a partir do snapshot em `diretorio`."""
    descricoes: list[pd.DataFrame] = []
    anos_por_tabela: dict[str, dict[int, int]] = {}
    linhas_por_tabela: dict[str, int] = {}
    for tabela in TABELAS_SNAPSHOT:
        dados = carregar_tabela(tabela, diretorio)
        descricoes.append(descrever_colunas(tabela, dados))
        anos_por_tabela[tabela] = anos_da_tabela(dados)
        linhas_por_tabela[tabela] = len(dados)

    saida = [
        "# Dicionário de dados do snapshot da Gold",
        "",
        "Gerado por `python -m src.preprocessing.qualidade` a partir de `data/`. Os papéis são "
        "provisórios; a lista definitiva de atributos e de colunas proibidas, por trilha, "
        "é da unidade de pré-processamento.",
        "",
        "## Anos cobertos por tabela",
        "",
        "| Tabela | Linhas | Linhas por ano |",
        "|---|---|---|",
    ]
    for tabela in TABELAS_SNAPSHOT:
        anos = anos_por_tabela[tabela]
        por_ano = ", ".join(f"{ano}: {n}" for ano, n in anos.items()) if anos else "sem coluna `ano`"
        saida.append(f"| `{tabela}` | {linhas_por_tabela[tabela]} | {por_ano} |")

    suspeitos = anos_suspeitos(anos_por_tabela)
    saida += ["", "## Anos fora dos dados reais (eventos sintéticos)", ""]
    if suspeitos:
        for tabela, anos in suspeitos.items():
            saida.append(f"- `{tabela}`: anos {', '.join(map(str, anos))} fora de {ANOS_REAIS}.")
    else:
        saida.append(
            f"Nenhuma tabela de fato tem linhas fora de {ANOS_REAIS[0]} e {ANOS_REAIS[-1]}. "
            "A Gold foi recriada sem o streaming, então não há eventos sintéticos."
        )

    vazias = colunas_totalmente_ausentes(descricoes)
    saida += ["", "## Colunas totalmente ausentes", ""]
    if vazias:
        saida += [f"- `{coluna}`" for coluna in vazias]
    else:
        saida.append("Nenhuma coluna tem todos os valores nulos.")

    saida += ["", "## Colunas"]
    for descricao in descricoes:
        saida += ["", f"### `{descricao['tabela'].iloc[0]}`", "", "| Coluna | Tipo | Ausentes | % ausentes | Papel provisório |", "|---|---|---|---|---|"]
        for linha in descricao.itertuples():
            saida.append(
                f"| `{linha.coluna}` | {linha.tipo} | {linha.ausentes} | {_numero(linha.pct_ausentes)} | {linha.papel} |"
            )
    return "\n".join(saida) + "\n"


def executar() -> Path:
    """Grava o dicionário em `reports/dicionario_dados.md` e devolve o caminho."""
    DIRETORIO_RELATORIOS.mkdir(parents=True, exist_ok=True)
    caminho = DIRETORIO_RELATORIOS / ARQUIVO_DICIONARIO
    caminho.write_text(gerar_dicionario(), encoding="utf-8")
    print(f"Dicionário gravado em {caminho}")
    return caminho


if __name__ == "__main__":
    executar()
