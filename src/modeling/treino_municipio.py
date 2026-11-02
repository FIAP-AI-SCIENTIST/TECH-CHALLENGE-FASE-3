"""Treino da trilha T2: o município (rede municipal) vai atingir a meta de 2024?

`python -m src.modeling.treino_municipio` lê o snapshot, monta a tabela de T2, roda
`treinar_trilha` e grava `reports/treino_t2.json`, `reports/comparacao_modelos_t2.csv` e o modelo
final em `models/modelo_t2.joblib`.
A tabela tem cerca de 5 mil municípios, então a busca roda sem amostra.

Os três algoritmos são Regressão Logística, Random Forest e Gradient Boosting.
A Regressão Logística e a Random Forest usam busca em grade; a Random Forest
avalia as 24 combinações configuradas. O Gradient Boosting usa busca aleatória.
"""

from dataclasses import replace

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import COLUNAS_T2, SEED
from src.modeling.pipeline_modelo import montar_pipeline
from src.modeling.separacao import separar_por_municipio
from src.modeling.treino import Candidato, gravar_modelo, gravar_relatorio, treinar_trilha
from src.modeling.validacao_cruzada import N_DOBRAS_PADRAO, validar_com_cv
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
        busca="grade",
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


def analisar_sinal_uf(
    treino: TabelaModelagem,
    n_dobras: int = N_DOBRAS_PADRAO,
) -> dict:
    """Compara o sinal territorial usando somente a tabela de treino recebida.

    A avaliação por UF mede transferência entre estados, não entre anos.
    """
    somente_uf = replace(
        COLUNAS_T2,
        categoricas=("sigla_uf", "nome_regiao"),
        numericas=(),
    )
    sem_uf = replace(
        COLUNAS_T2,
        categoricas=tuple(
            coluna for coluna in COLUNAS_T2.categoricas
            if coluna not in ("sigla_uf", "nome_regiao")
        ),
    )
    por_uf = TabelaModelagem(
        atributos=treino.atributos,
        alvo=treino.alvo,
        grupos=treino.atributos["sigla_uf"],
    )

    resultado = {
        "escopo": "somente treino",
        "n_municipios": int(treino.grupos.nunique()),
    }

    for nome, trilha, tabela in [
        ("somente_uf_regiao", somente_uf, treino),
        ("sem_uf_regiao", sem_uf, treino),
        ("agrupado_por_uf", COLUNAS_T2, por_uf),
    ]:
        modelo = montar_pipeline(
            trilha,
            LogisticRegression(max_iter=200, random_state=SEED),
            escalonar=True,
        )
        resultado[nome] = validar_com_cv(
            modelo, tabela, n_dobras=n_dobras
        )

    return resultado


def executar() -> dict:
    """Treina T2, registra a análise territorial e grava relatório e modelo."""
    tabela = carregar_tabela_t2()

    treino, _, _ = separar_por_municipio(tabela, semente=SEED)
    analise_uf = analisar_sinal_uf(treino)

    relatorio, modelo = treinar_trilha(
        COLUNAS_T2,
        tabela,
        CANDIDATOS_T2,
        semente=SEED,
    )
    relatorio["analise_uf"] = analise_uf

    caminho_json, caminho_csv = gravar_relatorio(
        relatorio, PREFIXO_RELATORIO
    )
    caminho_modelo = gravar_modelo(modelo, PREFIXO_RELATORIO)

    print(f"Modelo escolhido: {relatorio['modelo_escolhido']}")
    print(f"AUC-ROC no teste: {relatorio['teste']['auc_roc']}")
    print(f"Relatório: {caminho_json} e {caminho_csv}")
    print(f"Modelo: {caminho_modelo}")
    return relatorio


if __name__ == "__main__":
    executar()
