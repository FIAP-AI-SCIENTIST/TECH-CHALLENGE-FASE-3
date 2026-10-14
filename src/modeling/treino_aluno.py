"""Treino da trilha T1: o aluno que fez a prova é alfabetizado?

`python -m src.modeling.treino_aluno` lê o snapshot, monta a tabela de T1 (3,35
milhões de alunos), roda `treinar_trilha` e grava `reports/treino_t1.json` e
`reports/comparacao_modelos_t1.csv` e o modelo final em `models/modelo_t1.joblib`.
Leva cerca de 7 minutos.

A tabela inteira não cabe na busca de hiperparâmetros com validação cruzada, então o
treino e a validação são amostrados depois da separação por município, por sorteio
simples de alunos. Como os atributos de T1 são do município, da rede e do caderno, a
tabela tem só cerca de 120 mil combinações distintas, e uma amostra de 100 mil alunos
cobre quase todos os municípios do treino. O teste é avaliado inteiro.
"""

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import COLUNAS_T1, SEED
from src.modeling.treino import Candidato, gravar_modelo, gravar_relatorio, treinar_trilha
from src.preprocessing.loader import carregar_tabela
from src.preprocessing.tabelas import TabelaModelagem, construir_tabela_aluno

PREFIXO_RELATORIO: str = "t1"
MAX_LINHAS_TREINO: int = 100_000
MAX_LINHAS_VALIDACAO: int = 50_000

# Grades menores que as de T2: um ajuste de Gradient Boosting com 100 mil alunos
# leva cerca de 25 segundos, e a busca repete o ajuste a cada dobra.
CANDIDATOS_T1: tuple[Candidato, ...] = (
    Candidato(
        "regressao_logistica",
        LogisticRegression(max_iter=1000, random_state=SEED),
        escalonar=True,
        grade={"estimador__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    Candidato(
        "random_forest",
        RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        escalonar=False,
        grade={
            "estimador__max_depth": [5, 10, 20],
            "estimador__min_samples_leaf": [20, 100],
        },
        busca="aleatoria",
        n_iteracoes=4,
    ),
    Candidato(
        "gradient_boosting",
        GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        escalonar=False,
        grade={
            "estimador__learning_rate": [0.1, 0.3],
            "estimador__max_depth": [2, 3],
        },
        busca="aleatoria",
        n_iteracoes=3,
    ),
)


def carregar_tabela_t1() -> TabelaModelagem:
    """Tabela de T1 a partir do snapshot em `data/`."""
    return construir_tabela_aluno(carregar_tabela("fact_alunos"), carregar_tabela("dim_municipio"))


def executar() -> dict:
    """Treina T1 com amostra no treino e na validação, grava o relatório em `reports/` e o modelo em `models/`."""
    relatorio, modelo = treinar_trilha(
        COLUNAS_T1,
        carregar_tabela_t1(),
        CANDIDATOS_T1,
        max_linhas_treino=MAX_LINHAS_TREINO,
        max_linhas_validacao=MAX_LINHAS_VALIDACAO,
    )
    caminho_json, caminho_csv = gravar_relatorio(relatorio, PREFIXO_RELATORIO)
    caminho_modelo = gravar_modelo(modelo, PREFIXO_RELATORIO)
    print(f"Modelo escolhido: {relatorio['modelo_escolhido']}")
    print(f"AUC-ROC no teste: {relatorio['teste']['auc_roc']}")
    print(f"Relatório: {caminho_json} e {caminho_csv}")
    print(f"Modelo: {caminho_modelo}")
    return relatorio


if __name__ == "__main__":
    executar()
