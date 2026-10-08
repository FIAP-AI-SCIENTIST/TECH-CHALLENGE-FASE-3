"""Constantes compartilhadas do projeto.

Fica fora das camadas de `src/` de propósito: `preprocessing`, `modeling`,
`evaluation` e `visualization` importam daqui, e este módulo não importa de
nenhuma delas.
"""

from dataclasses import dataclass
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
# pré-processamento (requisitos FR3.4). `presenca` e `preenchimento_caderno`
# entram aqui porque, nos dados, caderno em branco implica `alfabetizado` falso
# (ver `data/README.md`).
COLUNAS_PROIBIDAS_PROVISORIAS: tuple[str, ...] = (
    "alfabetizado",
    "presenca",
    "preenchimento_caderno",
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


@dataclass(frozen=True)
class ColunasTrilha:
    """Atributos e colunas proibidas de uma trilha supervisionada.

    `proibidas` e `prefixos_proibidos` listam o que nunca pode virar feature
    (requisitos FR3.4); `atributos` é o que o pré-processador recebe.
    """

    categoricas: tuple[str, ...]
    numericas: tuple[str, ...]
    proibidas: tuple[str, ...]
    prefixos_proibidos: tuple[str, ...] = ()

    @property
    def atributos(self) -> tuple[str, ...]:
        return (*self.categoricas, *self.numericas)


# Atributos municipais estáticos (Atlas do Desenvolvimento Humano e região),
# usados pelas duas trilhas.
ATRIBUTOS_MUNICIPAIS_CATEGORICOS: tuple[str, ...] = ("sigla_uf", "nome_regiao")
ATRIBUTOS_MUNICIPAIS_NUMERICOS: tuple[str, ...] = (
    "capital_uf",
    "idhm",
    "idhm_educacao",
    "idhm_renda",
    "idhm_longevidade",
)

# T1 (aluno). Só quem fez a prova. `id_municipio` é o grupo da separação e nunca
# é feature. As colunas municipais do mesmo ano (taxa, média de português,
# proporção por nível, nível de alfabetização) são resultados da própria
# avaliação e ficam proibidas.
COLUNAS_T1: ColunasTrilha = ColunasTrilha(
    categoricas=("rede", "caderno", *ATRIBUTOS_MUNICIPAIS_CATEGORICOS),
    numericas=ATRIBUTOS_MUNICIPAIS_NUMERICOS,
    proibidas=(
        "alfabetizado",
        "nao_alfabetizado",
        "proficiencia",
        "presenca",
        "preenchimento_caderno",
        "peso_aluno",
        "id_aluno",
        "id_escola",
        "id_municipio",
        "atingiu_meta",
        "gap_pontos",
        "taxa_alfabetizacao",
        "meta_indicador",
        "media_portugues",
        "nivel_alfabetizacao",
        "percentual_participacao",
    ),
    prefixos_proibidos=("proporcao_aluno_nivel_",),
)

# T2 (município, rede municipal, rótulo de 2024). Atributos de 2023 (defasados),
# a meta de 2024 (conhecida de antemão) e a folga entre a taxa de 2023 e essa meta.
# A taxa, a média de português e a participação de 2024 são o resultado que o
# rótulo mede e ficam proibidas.
COLUNAS_T2: ColunasTrilha = ColunasTrilha(
    categoricas=ATRIBUTOS_MUNICIPAIS_CATEGORICOS,
    numericas=(
        *ATRIBUTOS_MUNICIPAIS_NUMERICOS,
        "taxa_alfabetizacao_2023",
        "media_portugues_2023",
        "meta_indicador_2024",
        "folga_2023",
    ),
    proibidas=(
        "atingiu_meta",
        "nao_atingiu_meta",
        "gap_pontos",
        "id_municipio",
        "rede",
        "taxa_alfabetizacao_2024",
        "media_portugues_2024",
        "percentual_participacao_2024",
        "nivel_alfabetizacao",
        "valid_from",
        "valid_to",
        "is_current",
    ),
    prefixos_proibidos=("proporcao_aluno_nivel_",),
)
