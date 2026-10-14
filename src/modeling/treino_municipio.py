"""Treino da trilha T2: o município (rede municipal) vai atingir a meta de 2024?

`python -m src.modeling.treino_municipio` lê o snapshot, monta a tabela de T2, roda
`treinar_trilha` e grava `reports/treino_t2.json`, `reports/comparacao_modelos_t2.csv` e o modelo
final em `models/modelo_t2.joblib`.
A tabela tem cerca de 5 mil municípios, então a busca roda sem amostra.

Os três algoritmos são os ensinados (Regressão Logística, Random Forest e Gradient
Boosting). A Regressão Logística tem poucas combinações e usa a busca em grade; as
árvores têm mais e usam a busca aleatória (Aula: Otimização 2).
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import COLUNAS_T2, SEED
from src.modeling.treino import Candidato, gravar_modelo, gravar_relatorio, treinar_trilha
from src.preprocessing.loader import carregar_tabela
from src.preprocessing.tabelas import TabelaModelagem, construir_tabela_municipio

PREFIXO_RELATORIO: str = "t2"

CANDIDATOS_T2: tuple[Candidato, ...] = (
    Candidato(
        "regressao_logistica",
        LogisticRegression(max_iter=1000, random_state=SEED),
        escalonar=True,
        grade={"estimador__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    Candidato(
        "random_forest",
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        escalonar=False,
        grade={
            "estimador__n_estimators": [100, 300],
            "estimador__max_depth": [3, 5, 10, None],
            "estimador__min_samples_leaf": [1, 5, 20],
        },
        busca="aleatoria",
        n_iteracoes=10,
    ),
    Candidato(
        "gradient_boosting",
        GradientBoostingClassifier(random_state=SEED),
        escalonar=False,
        grade={
            "estimador__n_estimators": [100, 200],
            "estimador__learning_rate": [0.03, 0.1],
            "estimador__max_depth": [2, 3],
        },
        busca="aleatoria",
        n_iteracoes=6,
    ),
)


def carregar_tabela_t2() -> TabelaModelagem:
    """Tabela de T2 a partir do snapshot em `data/`."""
    return construir_tabela_municipio(
        carregar_tabela("fact_meta_resultado_municipio"),
        carregar_tabela("fact_indicador_municipio"),
        carregar_tabela("dim_municipio"),
        carregar_tabela("dim_rede"),
    )


def executar() -> dict:
    """Treina T2, grava o relatório em `reports/` e o modelo em `models/` e imprime o resumo."""
    relatorio, modelo = treinar_trilha(COLUNAS_T2, carregar_tabela_t2(), CANDIDATOS_T2)
    caminho_json, caminho_csv = gravar_relatorio(relatorio, PREFIXO_RELATORIO)
    caminho_modelo = gravar_modelo(modelo, PREFIXO_RELATORIO)
    print(f"Modelo escolhido: {relatorio['modelo_escolhido']}")
    print(f"AUC-ROC no teste: {relatorio['teste']['auc_roc']}")
    print(f"Relatório: {caminho_json} e {caminho_csv}")
    print(f"Modelo: {caminho_modelo}")
    return relatorio


if __name__ == "__main__":
    executar()
