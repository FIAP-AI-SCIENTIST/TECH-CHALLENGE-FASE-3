"""Gráficos da avaliação final de T1 e T2, a partir dos modelos gravados pelo treino.

`python -m src.evaluation.avaliacao_final` carrega `models/modelo_t1.joblib` e
`models/modelo_t2.joblib`, reconstrói o conjunto de teste de cada trilha pela mesma
semente da separação, prevê as probabilidades e grava em `images/` a matriz de
confusão, a curva ROC e a distribuição das probabilidades por classe.

Antes de desenhar, confere que as métricas recalculadas batem com o
`reports/treino_<trilha>.json`: se não baterem, o modelo gravado não é o do relatório.
Nada aqui reajusta modelo nem escolhe limiar; o teste já foi aberto pelo treino.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.config import DIRETORIO_IMAGENS, DIRETORIO_RELATORIOS
from src.evaluation.metricas import calcular_metricas
from src.modeling.treino import carregar_modelo, conjunto_de_teste
from src.modeling.treino_aluno import carregar_tabela_t1
from src.modeling.treino_municipio import carregar_tabela_t2
from src.preprocessing.tabelas import TabelaModelagem
from src.visualization.graficos_avaliacao import curva_roc, distribuicao_probabilidades, matriz_confusao

TOLERANCIA: float = 1e-3
NOMES_DOS_MODELOS: dict[str, str] = {
    "regressao_logistica": "Regressão Logística",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}
TRILHAS: dict[str, dict[str, object]] = {
    "t1": {"titulo": "T1 (aluno)", "rotulos": ("Alfabetizado", "Não alfabetizado")},
    "t2": {"titulo": "T2 (município)", "rotulos": ("Atingiu a meta", "Não atingiu a meta")},
}


def desenhar_avaliacao(
    prefixo: str,
    y_verdadeiro: Sequence[int],
    probabilidades: Sequence[float],
    rotulos: tuple[str, str],
    nome_modelo: str,
    diretorio: Path = DIRETORIO_IMAGENS,
    titulo_trilha: str = "",
) -> list[Path]:
    """Matriz de confusão (limiar 0,5), curva ROC e distribuição das probabilidades da trilha."""
    metricas = calcular_metricas(y_verdadeiro, probabilidades)
    sufixo = f" - {titulo_trilha}" if titulo_trilha else ""
    return [
        matriz_confusao(
            metricas["matriz_confusao"],
            rotulos,
            f"Matriz de confusão no teste{sufixo}",
            f"avaliacao_matriz_confusao_{prefixo}.png",
            diretorio,
        ),
        curva_roc(
            y_verdadeiro, probabilidades, nome_modelo, f"Curva ROC no teste{sufixo}", f"avaliacao_curva_roc_{prefixo}.png", diretorio
        ),
        distribuicao_probabilidades(
            y_verdadeiro,
            probabilidades,
            rotulos,
            f"Probabilidade prevista por classe real{sufixo}",
            f"avaliacao_probabilidades_{prefixo}.png",
            diretorio,
        ),
    ]


def conferir_com_relatorio(metricas: dict[str, object], relatorio: dict[str, object]) -> None:
    """Falha se a AUC-ROC, a AUC-PR ou a matriz de confusão diferem das do relatório do treino."""
    esperado = relatorio["teste"]
    diferentes = [
        chave for chave in ("auc_roc", "auc_pr") if abs(float(metricas[chave]) - float(esperado[chave])) > TOLERANCIA
    ]
    if metricas["matriz_confusao"] != esperado["matriz_confusao"]:
        diferentes.append("matriz_confusao")
    if diferentes:
        raise RuntimeError(
            f"O modelo gravado não reproduz o relatório ({', '.join(diferentes)}). Rode de novo o script de treino da trilha."
        )


def _tabela_da_trilha(prefixo: str) -> TabelaModelagem:
    return carregar_tabela_t1() if prefixo == "t1" else carregar_tabela_t2()


def executar(prefixos: Sequence[str] = ("t2", "t1")) -> dict[str, list[Path]]:
    """Desenha a avaliação de cada trilha a partir do modelo gravado."""
    caminhos: dict[str, list[Path]] = {}
    for prefixo in prefixos:
        relatorio = json.loads((DIRETORIO_RELATORIOS / f"treino_{prefixo}.json").read_text(encoding="utf-8"))
        modelo = carregar_modelo(prefixo)
        teste = conjunto_de_teste(_tabela_da_trilha(prefixo))
        probabilidades = modelo.predict_proba(teste.atributos)[:, 1]
        conferir_com_relatorio(calcular_metricas(teste.alvo, probabilidades), relatorio)
        configuracao = TRILHAS[prefixo]
        caminhos[prefixo] = desenhar_avaliacao(
            prefixo,
            teste.alvo.to_numpy(),
            np.asarray(probabilidades),
            configuracao["rotulos"],
            NOMES_DOS_MODELOS.get(relatorio["modelo_escolhido"], relatorio["modelo_escolhido"]),
            titulo_trilha=configuracao["titulo"],
        )
        for caminho in caminhos[prefixo]:
            print(caminho)
    return caminhos


if __name__ == "__main__":
    executar()
