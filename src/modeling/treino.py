"""Treino completo de uma trilha supervisionada, do particionamento ao teste.

Junta as peças de `modeling` e `evaluation` numa ordem fixa, que vale para T1 e T2:

1. separa treino, validação e teste por município (`separar_por_municipio`);
2. roda os baselines com validação cruzada agrupada, só no treino;
3. busca os hiperparâmetros de cada candidato, só no treino, e o compara com os
   baselines pela AUC-ROC de validação cruzada, com o diagnóstico de overfitting;
4. escolhe o candidato de maior AUC-ROC de validação cruzada, nunca pelo teste;
5. confere o escolhido na validação (ajustado no treino) e diagnostica overfitting;
6. reajusta o escolhido em treino + validação e avalia o teste uma única vez.

O teste só é tocado no passo 6. Os limites de linhas servem para a trilha do
aluno, grande demais para a busca: a amostra é sorteada depois da separação, então
o teste é sempre avaliado inteiro.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

from src.config import DIRETORIO_RELATORIOS, SEED, ColunasTrilha
from src.evaluation.diagnostico import comparar_modelos, diagnosticar_overfitting, melhor_modelo
from src.evaluation.metricas import calcular_metricas
from src.modeling.pipeline_modelo import montar_pipeline
from src.modeling.protocolo import ITERACOES_PADRAO, buscar_hiperparametros, executar_baselines
from src.modeling.separacao import separar_por_municipio
from src.modeling.validacao_cruzada import N_DOBRAS_PADRAO, validar_com_cv
from src.preprocessing.tabelas import TabelaModelagem

# Casas decimais do relatório, para o JSON não variar entre execuções.
CASAS_DECIMAIS: int = 4


@dataclass(frozen=True)
class Candidato:
    """Um estimador a comparar: nome, estimador não ajustado, se escalona e a grade da busca.

    As chaves da `grade` usam o prefixo do passo do `Pipeline` (`estimador__C`).
    """

    nome: str
    estimador: BaseEstimator
    escalonar: bool
    grade: dict[str, list[Any]] = field(default_factory=dict)
    busca: str = "grade"
    n_iteracoes: int = ITERACOES_PADRAO


def amostrar(tabela: TabelaModelagem, n_linhas: int, semente: int = SEED) -> TabelaModelagem:
    """Sorteio simples de `n_linhas` linhas, sem reposição; a tabela menor volta inteira."""
    if len(tabela.atributos) <= n_linhas:
        return tabela
    posicoes = tabela.alvo.sample(n=n_linhas, random_state=semente).index
    return TabelaModelagem(
        atributos=tabela.atributos.loc[posicoes].reset_index(drop=True),
        alvo=tabela.alvo.loc[posicoes].reset_index(drop=True),
        grupos=tabela.grupos.loc[posicoes].reset_index(drop=True),
    )


def _juntar(primeira: TabelaModelagem, segunda: TabelaModelagem) -> TabelaModelagem:
    return TabelaModelagem(
        atributos=pd.concat([primeira.atributos, segunda.atributos], ignore_index=True),
        alvo=pd.concat([primeira.alvo, segunda.alvo], ignore_index=True),
        grupos=pd.concat([primeira.grupos, segunda.grupos], ignore_index=True),
    )


def _tamanho(tabela: TabelaModelagem) -> dict[str, float]:
    return {
        "linhas": int(len(tabela.alvo)),
        "municipios": int(tabela.grupos.nunique()),
        "proporcao_positiva": float(tabela.alvo.mean()),
    }


def _arredondar(valor: Any) -> Any:
    if isinstance(valor, dict):
        return {chave: _arredondar(item) for chave, item in valor.items()}
    if isinstance(valor, list):
        return [_arredondar(item) for item in valor]
    if isinstance(valor, bool):
        return valor
    if isinstance(valor, float):
        return round(valor, CASAS_DECIMAIS) if math.isfinite(valor) else None
    if hasattr(valor, "item"):  # escalares do numpy
        return _arredondar(valor.item())
    return valor


def _resumo_cv(resultado: dict[str, object]) -> dict[str, object]:
    return {
        "auc_roc_media": resultado["auc_roc_media"],
        "auc_roc_desvio": resultado["auc_roc_desvio"],
        "auc_roc_treino_media": resultado["auc_roc_treino_media"],
    }


def treinar_trilha(
    trilha: ColunasTrilha,
    tabela: TabelaModelagem,
    candidatos: tuple[Candidato, ...],
    max_linhas_treino: int | None = None,
    max_linhas_validacao: int | None = None,
    n_dobras: int = N_DOBRAS_PADRAO,
    semente: int = SEED,
) -> tuple[dict[str, Any], Pipeline]:
    """Devolve o relatório de avaliação e o modelo final.

    As métricas do bloco `validacao` são calculadas com o modelo ajustado
    somente no treino, antes do reajuste final.

    O modelo retornado é reajustado em treino + validação e usado para
    calcular as métricas do bloco `teste`. Ele não deve ser usado para
    reproduzir as métricas de validação, pois já foi treinado nesses dados.

    Para curvas ou diagnósticos de validação, deve-se usar um modelo
    ajustado exclusivamente no treino.
    """
    nomes = [candidato.nome for candidato in candidatos]
    if len(set(nomes)) != len(nomes):
        raise ValueError(f"Nome de candidato repetido: {nomes}.")

    treino, validacao, teste = separar_por_municipio(tabela, semente=semente)
    if max_linhas_treino is not None:
        treino = amostrar(treino, max_linhas_treino, semente)
    if max_linhas_validacao is not None:
        validacao = amostrar(validacao, max_linhas_validacao, semente)

    baselines = executar_baselines(trilha, treino, n_dobras)

    buscas: dict[str, dict[str, Any]] = {}
    resultados_cv: dict[str, dict[str, object]] = {
        "dummy": baselines["dummy"],
        "regressao_logistica_padrao": baselines["regressao_logistica"],
    }
    for candidato in candidatos:
        pipeline = montar_pipeline(trilha, clone(candidato.estimador), candidato.escalonar)
        buscas[candidato.nome] = buscar_hiperparametros(
            pipeline, candidato.grade, treino, candidato.busca, candidato.n_iteracoes, n_dobras, semente
        )
        resultados_cv[candidato.nome] = validar_com_cv(buscas[candidato.nome]["modelo"], treino, n_dobras)

    comparacao = comparar_modelos(resultados_cv)
    escolhido = melhor_modelo(comparacao[comparacao["modelo"].isin(nomes)])

    ajustado_no_treino = buscas[escolhido]["modelo"]
    auc_treino = calcular_metricas(treino.alvo, ajustado_no_treino.predict_proba(treino.atributos)[:, 1])["auc_roc"]
    metricas_validacao = calcular_metricas(validacao.alvo, ajustado_no_treino.predict_proba(validacao.atributos)[:, 1])

    treino_e_validacao = _juntar(treino, validacao)
    modelo_final = clone(ajustado_no_treino).fit(treino_e_validacao.atributos, treino_e_validacao.alvo)
    metricas_teste = calcular_metricas(teste.alvo, modelo_final.predict_proba(teste.atributos)[:, 1])

    relatorio = {
        "tamanhos": {"treino": _tamanho(treino), "validacao": _tamanho(validacao), "teste": _tamanho(teste)},
        "baselines": {nome: _resumo_cv(resultado) for nome, resultado in baselines.items()},
        "candidatos": {
            nome: {"melhores_parametros": busca["melhores_parametros"], "auc_roc_cv": busca["auc_roc_cv"]}
            for nome, busca in buscas.items()
        },
        "comparacao": comparacao.to_dict(orient="records"),
        "modelo_escolhido": escolhido,
        "validacao": {
            "metricas": metricas_validacao,
            "diagnostico": diagnosticar_overfitting(auc_treino, metricas_validacao["auc_roc"]),
        },
        "teste": metricas_teste,
    }
    return _arredondar(relatorio), modelo_final


def gravar_relatorio(relatorio: dict[str, Any], prefixo: str, diretorio: Path = DIRETORIO_RELATORIOS) -> tuple[Path, Path]:
    """Grava `treino_<prefixo>.json` e `comparacao_modelos_<prefixo>.csv` e devolve os dois caminhos."""
    diretorio.mkdir(parents=True, exist_ok=True)
    caminho_json = diretorio / f"treino_{prefixo}.json"
    caminho_json.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    caminho_csv = diretorio / f"comparacao_modelos_{prefixo}.csv"
    pd.DataFrame(relatorio["comparacao"]).to_csv(caminho_csv, index=False)
    return caminho_json, caminho_csv
