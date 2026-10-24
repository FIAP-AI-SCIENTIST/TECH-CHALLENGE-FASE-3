"""Gráficos da interpretação dos modelos (importância e SHAP).

Seguem a regra de `graficos.py`: recebem os dados prontos, gravam em `images/` e
devolvem o caminho, sem calcular importância nem SHAP.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.config import DIRETORIO_IMAGENS, SEED
from src.visualization.graficos import salvar_figura
from src.visualization.graficos_avaliacao import COR_SERIE_1, COR_SERIE_2

MAXIMO_PADRAO: int = 15


def nome_legivel(nome_transformado: str) -> str:
    """Tira o prefixo do pré-processador (`categoricos__sigla_uf_SP` -> `sigla_uf_SP`)."""
    return nome_transformado.split("__", 1)[-1]


def barras_importancia(
    tabela: pd.DataFrame,
    coluna_nome: str,
    coluna_valor: str,
    titulo: str,
    rotulo_eixo: str,
    nome_arquivo: str,
    diretorio: Path = DIRETORIO_IMAGENS,
    coluna_erro: str | None = None,
    maximo: int = MAXIMO_PADRAO,
) -> Path:
    """Barras horizontais dos `maximo` maiores valores em módulo, o maior no topo.

    Valores positivos ficam em azul e negativos em laranja, com a linha do zero: assim
    o gráfico serve tanto à importância (sempre positiva) quanto ao coeficiente (com sinal).
    """
    faltando = [coluna for coluna in (coluna_nome, coluna_valor, coluna_erro) if coluna and coluna not in tabela]
    if faltando:
        raise ValueError(f"Colunas ausentes na tabela: {', '.join(faltando)}.")
    dados = tabela.loc[tabela[coluna_valor].abs().sort_values(ascending=False).index].head(maximo).iloc[::-1]
    cores = [COR_SERIE_1 if valor >= 0 else COR_SERIE_2 for valor in dados[coluna_valor]]

    figura, eixo = plt.subplots(figsize=(7, max(2.5, 0.35 * len(dados) + 1)))
    eixo.barh(
        dados[coluna_nome].astype(str).map(nome_legivel),
        dados[coluna_valor],
        xerr=dados[coluna_erro] if coluna_erro else None,
        color=cores,
        height=0.7,
        error_kw={"ecolor": "#52514e", "elinewidth": 1, "capsize": 2},
    )
    eixo.axvline(0, color="#52514e", linewidth=0.8)
    eixo.set_xlabel(rotulo_eixo)
    eixo.set_title(titulo)
    eixo.grid(axis="x", color="#e4e3df", linewidth=0.8)
    eixo.set_axisbelow(True)
    return salvar_figura(figura, nome_arquivo, diretorio)


def resumo_shap(
    shap_valores: pd.DataFrame,
    transformados: pd.DataFrame,
    titulo: str,
    nome_arquivo: str,
    diretorio: Path = DIRETORIO_IMAGENS,
    maximo: int = MAXIMO_PADRAO,
    semente: int = SEED,
    unidade: str = "log-odds",
) -> Path:
    """Gráfico de resumo do SHAP (beeswarm): um ponto por previsão, cor pelo valor do atributo."""
    shap.summary_plot(
        shap_valores.to_numpy(),
        transformados,
        feature_names=[nome_legivel(coluna) for coluna in transformados.columns],
        max_display=maximo,
        show=False,
        rng=semente,
        plot_size=(8, max(3.0, 0.4 * min(maximo, transformados.shape[1]) + 1.5)),
    )
    figura = plt.gcf()
    barra_de_cores = figura.axes[-1]
    barra_de_cores.set_ylabel("Valor do atributo")
    barra_de_cores.set_yticks(barra_de_cores.get_yticks(), ["Baixo", "Alto"])
    figura.axes[0].set_title(titulo)
    figura.axes[0].set_xlabel(f"Valor SHAP (contribuição em {unidade} para o risco)")
    return salvar_figura(figura, nome_arquivo, diretorio)
