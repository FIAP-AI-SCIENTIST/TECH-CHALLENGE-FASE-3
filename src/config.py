"""Constantes compartilhadas do projeto.

Fica fora das camadas de `src/` de propósito: `preprocessing`, `modeling`,
`evaluation` e `visualization` importam daqui, e este módulo não importa de
nenhuma delas.
"""

from pathlib import Path

# Semente única de todos os `random_state` do projeto. O valor 42 é o que o
# código das aulas usa em `train_test_split`, `RandomForestClassifier`, `KMeans`
# e demais exemplos.
SEED: int = 42

# Caminhos relativos à raiz do repositório, nunca absolutos.
RAIZ: Path = Path(__file__).resolve().parent.parent
DIRETORIO_DADOS: Path = RAIZ / "data"
DIRETORIO_RELATORIOS: Path = RAIZ / "reports"
DIRETORIO_IMAGENS: Path = RAIZ / "images"

# Camada Gold da Fase 2, lida do BigQuery por Application Default Credentials
# (somente leitura). O projeto vem da variável de ambiente `GCP_PROJECT_ID`.
VARIAVEL_PROJETO_GCP: str = "GCP_PROJECT_ID"
DATASET_GOLD: str = "alfabetizacao_analytics"

# Tabelas da Gold que entram no snapshot em `data/`.
TABELAS_SNAPSHOT: tuple[str, ...] = (
    "fact_alunos",
    "fact_meta_resultado_municipio",
    "fact_indicador_municipio",
    "fact_alfabetizacao_municipio",
    "dim_municipio",
    "dim_rede",
    "dim_serie",
    "dim_tempo",
    "dim_uf",
)

# Chaves substitutas (`sk_*`) da Gold: servem só para joins no BigQuery. O
# snapshot mantém as chaves naturais (`id_municipio`, `rede`), então as `sk_*`
# são removidas para reduzir o tamanho do arquivo.
PREFIXO_CHAVE_SUBSTITUTA: str = "sk_"

# Colunas que identificam o aluno e não entram no snapshot: não são features
# e não há razão para versionar identificadores individuais.
COLUNAS_FORA_DO_SNAPSHOT: dict[str, tuple[str, ...]] = {
    "fact_alunos": ("id_aluno",),
}

# Alvos das trilhas T1 (aluno) e T2 (município).
ALVO_ALUNO: str = "alfabetizado"
ALVO_MUNICIPIO: str = "atingiu_meta"

# Lista PROVISÓRIA de colunas proibidas como feature, usada só pelo baseline
# descartável do esqueleto. A lista definitiva, por trilha, é da unidade de
# pré-processamento (requisitos FR3.4).
COLUNAS_PROIBIDAS_PROVISORIAS: tuple[str, ...] = (
    "alfabetizado",
    "proficiencia",
    "id_aluno",
    "id_escola",
    "id_municipio",
    "atingiu_meta",
    "gap_pontos",
    "taxa_alfabetizacao",
    "meta_indicador",
    "media_portugues",
    "nivel_alfabetizacao",
)
