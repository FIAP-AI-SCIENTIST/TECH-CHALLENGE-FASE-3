"""Interpretação dos modelos finais de T1 e T2, sobre o conjunto de teste.

`python -m src.evaluation.interpretacao_modelos` carrega os modelos gravados pelo
treino, reconstrói o teste de cada trilha e grava em `reports/` as tabelas e em
`images/` os gráficos da importância nativa, da permutação e do SHAP.

O teste é usado porque são dados que o modelo não viu: a importância medida no
treino premiaria o que o modelo decorou. Nada aqui reajusta o modelo. O teste de T1
tem mais de 600 mil alunos, então a permutação e o SHAP usam amostras sorteadas.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import COLUNAS_T1, COLUNAS_T2, DIRETORIO_IMAGENS, DIRETORIO_RELATORIOS, SEED, ColunasTrilha
from src.evaluation.interpretacao import (
    agregar_por_atributo,
    importancia_nativa,
    importancia_por_permutacao,
    valores_shap,
)
from src.modeling.treino import amostrar, carregar_modelo, conjunto_de_teste
from src.modeling.treino_aluno import carregar_tabela_t1
from src.modeling.treino_municipio import carregar_tabela_t2
from src.preprocessing.tabelas import TabelaModelagem
from src.visualization.graficos_interpretacao import barras_importancia, resumo_shap

# Tamanho das amostras do teste: o de T2 (cerca de mil municípios) entra inteiro.
AMOSTRAS: dict[str, dict[str, int]] = {
    "t1": {"n_permutacao": 20_000, "n_shap": 2_000},
    "t2": {"n_permutacao": 20_000, "n_shap": 2_000},
}
TITULOS: dict[str, str] = {"t1": "T1 (aluno)", "t2": "T2 (município)"}


@dataclass(frozen=True)
class Interpretacao:
    """As três leituras de um modelo e o que o gráfico de resumo do SHAP precisa."""

    nativa: pd.DataFrame
    permutacao: pd.DataFrame
    shap_por_atributo: pd.DataFrame
    shap_valores: pd.DataFrame
    transformados: pd.DataFrame
    unidade_shap: str = "log-odds"
    semente: int = SEED


def interpretar(
    modelo: Pipeline,
    trilha: ColunasTrilha,
    tabela: TabelaModelagem,
    n_permutacao: int,
    n_shap: int,
    n_repeticoes: int = 10,
    semente: int = SEED,
) -> Interpretacao:
    """Importância nativa, permutação (AUC-ROC) e SHAP do `modelo` sobre amostras da `tabela`."""
    amostra_permutacao = amostrar(tabela, n_permutacao, semente)
    amostra_shap = amostrar(tabela, n_shap, semente)
    atributos_shap = amostra_shap.atributos[list(trilha.atributos)]
    shap_valores, transformados = valores_shap(modelo, atributos_shap, semente=semente)
    return Interpretacao(
        nativa=importancia_nativa(modelo),
        permutacao=importancia_por_permutacao(
            modelo, amostra_permutacao.atributos[list(trilha.atributos)], amostra_permutacao.alvo, n_repeticoes, semente
        ),
        shap_por_atributo=agregar_por_atributo(shap_valores, trilha.atributos),
        shap_valores=shap_valores,
        transformados=transformados,
        unidade_shap=shap_valores.attrs["unidade"],
        semente=semente,
    )


def gravar_interpretacao(
    resultado: Interpretacao,
    prefixo: str,
    titulo_trilha: str,
    diretorio_relatorios: Path = DIRETORIO_RELATORIOS,
    diretorio_imagens: Path = DIRETORIO_IMAGENS,
) -> list[Path]:
    """Grava as três tabelas em CSV e os quatro gráficos, e devolve os caminhos."""
    diretorio_relatorios.mkdir(parents=True, exist_ok=True)
    caminhos = []
    for nome, tabela in (
        ("nativa", resultado.nativa),
        ("permutacao", resultado.permutacao),
        ("shap", resultado.shap_por_atributo),
    ):
        caminho = diretorio_relatorios / f"interpretacao_{nome}_{prefixo}.csv"
        tabela.round(6).to_csv(caminho, index=False)
        caminhos.append(caminho)

    caminhos += [
        barras_importancia(
            resultado.nativa,
            "atributo_transformado",
            "importancia",
            f"Importância nativa do modelo - {titulo_trilha}",
            "Coeficiente (escala padronizada) ou importância das árvores",
            f"interpretacao_nativa_{prefixo}.png",
            diretorio_imagens,
        ),
        barras_importancia(
            resultado.permutacao,
            "atributo",
            "queda_auc_media",
            f"Importância por permutação no teste - {titulo_trilha}",
            "Queda da AUC-ROC ao embaralhar o atributo",
            f"interpretacao_permutacao_{prefixo}.png",
            diretorio_imagens,
            coluna_erro="queda_auc_desvio",
        ),
        barras_importancia(
            resultado.shap_por_atributo,
            "atributo",
            "shap_medio_absoluto",
            f"SHAP por atributo original - {titulo_trilha}",
            f"Média do |SHAP| ({resultado.unidade_shap}), One-Hot somado por atributo",
            f"interpretacao_shap_atributos_{prefixo}.png",
            diretorio_imagens,
        ),
        resumo_shap(
            resultado.shap_valores,
            resultado.transformados,
            f"SHAP no teste - {titulo_trilha}",
            f"interpretacao_shap_resumo_{prefixo}.png",
            diretorio_imagens,
            semente=resultado.semente,
            unidade=resultado.unidade_shap,
        ),
    ]
    return caminhos


def executar(prefixos: tuple[str, ...] = ("t2", "t1")) -> dict[str, Interpretacao]:
    """Interpreta cada trilha a partir do modelo gravado e grava tabelas e gráficos."""
    configuracao = {"t1": (COLUNAS_T1, carregar_tabela_t1), "t2": (COLUNAS_T2, carregar_tabela_t2)}
    resultados: dict[str, Interpretacao] = {}
    for prefixo in prefixos:
        trilha, carregar = configuracao[prefixo]
        teste = conjunto_de_teste(carregar())
        resultados[prefixo] = interpretar(carregar_modelo(prefixo), trilha, teste, **AMOSTRAS[prefixo])
        for caminho in gravar_interpretacao(resultados[prefixo], prefixo, TITULOS[prefixo]):
            print(caminho)
        print(resultados[prefixo].shap_por_atributo.to_string(index=False))
    return resultados


if __name__ == "__main__":
    executar()
