"""Baseline DESCARTÁVEL do aluno, só para provar a fatia ponta a ponta do esqueleto.

Não é resultado do projeto. A separação aqui é um `train_test_split`
estratificado, que NÃO separa por município: o mesmo município aparece no treino
e na validação, e os atributos municipais o identificam, então a AUC-ROC sai
otimista. As unidades de pré-processamento, validação e modelagem substituem
este módulo.

Uso:
    python -m src.modeling.baseline
"""

import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ALVO_ALUNO,
    COLUNAS_PROIBIDAS_PROVISORIAS,
    DIRETORIO_RELATORIOS,
    SEED,
)
from src.preprocessing.loader import carregar_tabela

ALVO_NAO_ALFABETIZADO: str = "nao_alfabetizado"
ATRIBUTOS_CATEGORICOS: tuple[str, ...] = ("rede", "caderno", "sigla_uf", "nome_regiao")
ATRIBUTOS_NUMERICOS: tuple[str, ...] = (
    "capital_uf",
    "idhm",
    "idhm_educacao",
    "idhm_renda",
    "idhm_longevidade",
)
FRACAO_VALIDACAO: float = 0.2
ARQUIVO_RESULTADO: str = "baseline_provisorio.json"


def montar_tabela_aluno(alunos: pd.DataFrame, municipios: pd.DataFrame) -> pd.DataFrame:
    """Une `fact_alunos` aos atributos de `dim_municipio` e cria o alvo.

    O alvo `nao_alfabetizado` tem a classe positiva = não alfabetizado (o risco).
    Falha com `ValueError` se algum atributo estiver na lista de colunas proibidas.
    """
    atributos = [*ATRIBUTOS_CATEGORICOS, *ATRIBUTOS_NUMERICOS]
    proibidos = sorted(set(atributos) & set(COLUNAS_PROIBIDAS_PROVISORIAS))
    if proibidos:
        raise ValueError(f"Atributos proibidos como feature: {', '.join(proibidos)}.")

    colunas_municipio = ["id_municipio", *(c for c in atributos if c in municipios.columns)]
    tabela = alunos.merge(
        municipios[colunas_municipio], on="id_municipio", how="left", validate="many_to_one"
    )
    tabela[ALVO_NAO_ALFABETIZADO] = (~tabela[ALVO_ALUNO]).astype(int)
    return tabela[[*atributos, ALVO_NAO_ALFABETIZADO]]


def _pipeline(estimador: object) -> Pipeline:
    """Pré-processamento e estimador no mesmo objeto, ajustado só com o treino."""
    pre_processador = ColumnTransformer(
        [
            (
                "numericos",
                Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]),
                list(ATRIBUTOS_NUMERICOS),
            ),
            ("categoricos", OneHotEncoder(handle_unknown="ignore"), list(ATRIBUTOS_CATEGORICOS)),
        ]
    )
    return Pipeline([("pre_processamento", pre_processador), ("estimador", estimador)])


def avaliar_baselines(tabela: pd.DataFrame) -> dict[str, float]:
    """Treina `DummyClassifier` e Regressão Logística e devolve a AUC-ROC na validação."""
    atributos = tabela.drop(columns=[ALVO_NAO_ALFABETIZADO])
    alvo = tabela[ALVO_NAO_ALFABETIZADO]
    x_treino, x_validacao, y_treino, y_validacao = train_test_split(
        atributos, alvo, test_size=FRACAO_VALIDACAO, stratify=alvo, random_state=SEED
    )

    resultado: dict[str, float] = {
        "linhas_treino": float(len(x_treino)),
        "linhas_validacao": float(len(x_validacao)),
        "proporcao_positiva": float(alvo.mean()),
    }
    estimadores = {
        "dummy": DummyClassifier(strategy="prior"),
        "regressao_logistica": LogisticRegression(max_iter=200, random_state=SEED),
    }
    for nome, estimador in estimadores.items():
        modelo = _pipeline(estimador).fit(x_treino, y_treino)
        probabilidades = modelo.predict_proba(x_validacao)[:, 1]
        resultado[f"auc_roc_{nome}"] = float(roc_auc_score(y_validacao, probabilidades))
    return resultado


def executar() -> dict[str, float]:
    """Carrega o snapshot, avalia os baselines e grava o resultado em `reports/`."""
    tabela = montar_tabela_aluno(carregar_tabela("fact_alunos"), carregar_tabela("dim_municipio"))
    resultado = avaliar_baselines(tabela)
    DIRETORIO_RELATORIOS.mkdir(parents=True, exist_ok=True)
    caminho = DIRETORIO_RELATORIOS / ARQUIVO_RESULTADO
    caminho.write_text(json.dumps(resultado, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(resultado, indent=2))
    return resultado


if __name__ == "__main__":
    executar()
