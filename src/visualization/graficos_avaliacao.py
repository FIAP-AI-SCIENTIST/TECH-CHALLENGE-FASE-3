"""Gráficos da avaliação final dos modelos supervisionados.

Seguem a regra de `graficos.py`: cada função recebe os dados por argumento, grava a
figura em `images/` e devolve o caminho, sem ler dados nem treinar modelos. A curva
ROC calcula os pontos com `roc_curve`, porque eles só servem ao desenho.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import DIRETORIO_IMAGENS
from src.visualization.graficos import salvar_figura

# Duas primeiras cores da paleta categórica (azul e laranja), separáveis por
# daltônicos; a referência do acaso fica em cinza.
COR_SERIE_1: str = "#2a78d6"
COR_SERIE_2: str = "#eb6834"
COR_REFERENCIA: str = "#8a8984"
CHAVES_MATRIZ: tuple[str, ...] = ("verdadeiro_negativo", "falso_positivo", "falso_negativo", "verdadeiro_positivo")


def _decimal(valor: float, casas: int) -> str:
    """Número com vírgula decimal e ponto de milhar, como no resto do projeto."""
    return f"{valor:,.{casas}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _validar_tamanhos(y_verdadeiro: Sequence[int], probabilidades: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_verdadeiro)
    probabilidade = np.asarray(probabilidades, dtype=float)
    if len(y) != len(probabilidade):
        raise ValueError("O alvo e as probabilidades devem ter o mesmo tamanho.")
    return y, probabilidade


def matriz_confusao(
    matriz: dict[str, int],
    rotulos: tuple[str, str],
    titulo: str,
    nome_arquivo: str,
    diretorio: Path = DIRETORIO_IMAGENS,
) -> Path:
    """Matriz de confusão a partir do dicionário de `calcular_metricas`, com contagem e % da linha.

    `rotulos` são os nomes das classes 0 e 1, nessa ordem.
    """
    faltando = [chave for chave in CHAVES_MATRIZ if chave not in matriz]
    if faltando:
        raise ValueError(f"Faltam na matriz: {', '.join(faltando)}.")
    valores = np.array(
        [
            [matriz["verdadeiro_negativo"], matriz["falso_positivo"]],
            [matriz["falso_negativo"], matriz["verdadeiro_positivo"]],
        ]
    )
    totais_por_classe = valores.sum(axis=1, keepdims=True)
    if (totais_por_classe == 0).any():
        raise ValueError("A matriz precisa de registros nas duas classes reais.")

    percentual_da_linha = valores / totais_por_classe

    figura, eixo = plt.subplots(figsize=(5.5, 4.5))
    imagem = eixo.imshow(percentual_da_linha, cmap="Blues", vmin=0, vmax=1)
    for linha in range(2):
        for coluna in range(2):
            escuro = percentual_da_linha[linha, coluna] > 0.55
            eixo.text(
                coluna,
                linha,
                f"{_decimal(valores[linha, coluna], 0)}\n({_decimal(100 * percentual_da_linha[linha, coluna], 1)}%)",
                ha="center",
                va="center",
                color="white" if escuro else "#0b0b0b",
                fontsize=11,
            )
    eixo.set_xticks([0, 1], [f"Previsto:\n{rotulos[0]}", f"Previsto:\n{rotulos[1]}"])
    eixo.set_yticks([0, 1], [f"Real:\n{rotulos[0]}", f"Real:\n{rotulos[1]}"])
    eixo.set_title(titulo)
    figura.colorbar(imagem, ax=eixo, label="% da classe real", fraction=0.046, pad=0.04)
    return salvar_figura(figura, nome_arquivo, diretorio)


def curva_roc(
    y_verdadeiro: Sequence[int],
    probabilidades: Sequence[float],
    nome_modelo: str,
    titulo: str,
    nome_arquivo: str,
    diretorio: Path = DIRETORIO_IMAGENS,
) -> Path:
    """Curva ROC do modelo contra a diagonal do acaso, com a AUC-ROC na legenda."""
    y, probabilidade = _validar_tamanhos(y_verdadeiro, probabilidades)
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("A curva ROC precisa das duas classes reais: 0 e 1.")
    falsos_positivos, verdadeiros_positivos, _ = roc_curve(y, probabilidade)
    auc = roc_auc_score(y, probabilidade)

    figura, eixo = plt.subplots(figsize=(5.5, 5))
    eixo.plot(falsos_positivos, verdadeiros_positivos, color=COR_SERIE_1, linewidth=2, label=f"{nome_modelo} (AUC-ROC {_decimal(auc, 3)})")
    eixo.plot([0, 1], [0, 1], color=COR_REFERENCIA, linewidth=1.5, linestyle="--", label="Acaso (AUC-ROC 0,500)")
    eixo.set_xlim(0, 1)
    eixo.set_ylim(0, 1)
    eixo.set_xlabel("Taxa de falsos positivos")
    eixo.set_ylabel("Taxa de verdadeiros positivos (recall)")
    eixo.set_title(titulo)
    eixo.grid(color="#e4e3df", linewidth=0.8)
    eixo.legend(loc="lower right")
    return salvar_figura(figura, nome_arquivo, diretorio)


def distribuicao_probabilidades(
    y_verdadeiro: Sequence[int],
    probabilidades: Sequence[float],
    rotulos: tuple[str, str],
    titulo: str,
    nome_arquivo: str,
    diretorio: Path = DIRETORIO_IMAGENS,
    limiar: float = 0.5,
    bins: int = 40,
) -> Path:
    """Histogramas da probabilidade prevista para cada classe real, com o limiar marcado.

    As alturas são a proporção dentro de cada classe, para as duas serem comparáveis
    mesmo com tamanhos diferentes.
    """
    y, probabilidade = _validar_tamanhos(y_verdadeiro, probabilidades)
    bordas = np.linspace(0, 1, bins + 1)

    figura, eixo = plt.subplots(figsize=(7, 4))
    for classe, cor in ((0, COR_SERIE_1), (1, COR_SERIE_2)):
        da_classe = probabilidade[y == classe]
        pesos = np.full(len(da_classe), 1 / max(len(da_classe), 1))
        eixo.hist(da_classe, bins=bordas, weights=pesos, color=cor, alpha=0.55, label=f"Real: {rotulos[classe]}")
    eixo.axvline(limiar, color="#0b0b0b", linewidth=1.5, linestyle="--", label=f"Limiar {_decimal(limiar, 1)}")
    eixo.set_xlim(0, 1)
    eixo.set_xlabel(f"Probabilidade prevista de “{rotulos[1]}”")
    eixo.set_ylabel("Proporção dentro da classe")
    eixo.set_title(titulo)
    eixo.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return salvar_figura(figura, nome_arquivo, diretorio)
