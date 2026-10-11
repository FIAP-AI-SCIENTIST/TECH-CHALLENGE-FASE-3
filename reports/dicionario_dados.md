# Dicionário de dados do snapshot da Gold

Gerado por `python -m src.preprocessing.qualidade` a partir de `data/`. Os papéis são provisórios; a lista definitiva de atributos e de colunas proibidas, por trilha, é da unidade de pré-processamento.

## Anos cobertos por tabela

| Tabela | Linhas | Linhas por ano |
|---|---|---|
| `fact_alunos` | 3867999 | 2023: 1747439, 2024: 2120560 |
| `fact_meta_resultado_municipio` | 10698 | 2023: 5352, 2024: 5346 |
| `fact_indicador_municipio` | 23995 | 2023: 11547, 2024: 12448 |
| `fact_alfabetizacao_municipio` | 23995 | 2023: 11547, 2024: 12448 |
| `dim_municipio` | 5571 | sem coluna `ano` |
| `dim_rede` | 5 | sem coluna `ano` |
| `dim_serie` | 1 | sem coluna `ano` |
| `dim_tempo` | 8 | 2023: 1, 2024: 1, 2025: 1, 2026: 1, 2027: 1, 2028: 1, 2029: 1, 2030: 1 |
| `dim_uf` | 27 | sem coluna `ano` |

## Anos fora dos dados reais (eventos sintéticos)

Nenhuma tabela de fato tem linhas fora de 2023 e 2024. A Gold foi recriada sem o streaming, então não há eventos sintéticos.

## Colunas totalmente ausentes

- `fact_alfabetizacao_municipio.meta_indicador`
- `fact_alfabetizacao_municipio.percentual_participacao`
- `fact_alfabetizacao_municipio.nivel_alfabetizacao`
- `fact_alfabetizacao_municipio.gap_pontos`
- `fact_alfabetizacao_municipio.atingiu_meta`

## Colunas

### `fact_alunos`

| Coluna | Tipo | Ausentes | % ausentes | Papel provisório |
|---|---|---|---|---|
| `ano` | int64 | 0 | 0,00 | atributo candidato |
| `id_municipio` | str | 0 | 0,00 | identificador; nunca feature |
| `id_escola` | str | 0 | 0,00 | identificador; nunca feature |
| `caderno` | str | 0 | 0,00 | atributo candidato |
| `serie` | str | 0 | 0,00 | atributo candidato |
| `rede` | str | 0 | 0,00 | atributo candidato |
| `presenca` | bool | 0 | 0,00 | proibida: caderno em branco implica alfabetizado falso |
| `preenchimento_caderno` | bool | 0 | 0,00 | proibida: caderno em branco implica alfabetizado falso |
| `alfabetizado` | bool | 0 | 0,00 | alvo do aluno (T1) |
| `proficiencia` | float64 | 513338 | 13,27 | proibida: define alfabetizado (corte 743) |
| `peso_aluno` | float64 | 513338 | 13,27 | peso amostral; não é feature |

### `fact_meta_resultado_municipio`

| Coluna | Tipo | Ausentes | % ausentes | Papel provisório |
|---|---|---|---|---|
| `id_municipio` | str | 0 | 0,00 | identificador; nunca feature |
| `rede` | str | 0 | 0,00 | atributo candidato |
| `ano` | int64 | 0 | 0,00 | atributo candidato |
| `taxa_alfabetizacao` | float64 | 120 | 1,12 | proibida no mesmo ano do rótulo; só defasada |
| `meta_indicador` | float64 | 5472 | 51,15 | conhecida de antemão; uso a decidir (define atingiu_meta com a taxa) |
| `gap_pontos` | float64 | 5472 | 51,15 | proibida: diferença entre taxa e meta do mesmo ano |
| `atingiu_meta` | object | 5472 | 51,15 | alvo do município (T2); nulo quando não há meta vigente |
| `percentual_participacao` | float64 | 120 | 1,12 | atributo candidato |
| `valid_from` | int64 | 0 | 0,00 | controle SCD2 |
| `valid_to` | float64 | 5352 | 50,03 | controle SCD2 |
| `is_current` | bool | 0 | 0,00 | controle SCD2 |

### `fact_indicador_municipio`

| Coluna | Tipo | Ausentes | % ausentes | Papel provisório |
|---|---|---|---|---|
| `ano` | int64 | 0 | 0,00 | atributo candidato |
| `id_municipio` | str | 0 | 0,00 | identificador; nunca feature |
| `serie` | str | 0 | 0,00 | atributo candidato |
| `rede` | str | 0 | 0,00 | atributo candidato |
| `taxa_alfabetizacao` | float64 | 0 | 0,00 | proibida no mesmo ano do rótulo; só defasada |
| `media_portugues` | float64 | 0 | 0,00 | resultado do mesmo ano; só defasada |
| `proporcao_aluno_nivel_0` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |
| `proporcao_aluno_nivel_1` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |
| `proporcao_aluno_nivel_2` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |
| `proporcao_aluno_nivel_3` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |
| `proporcao_aluno_nivel_4` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |
| `proporcao_aluno_nivel_5` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |
| `proporcao_aluno_nivel_6` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |
| `proporcao_aluno_nivel_7` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |
| `proporcao_aluno_nivel_8` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |

### `fact_alfabetizacao_municipio`

| Coluna | Tipo | Ausentes | % ausentes | Papel provisório |
|---|---|---|---|---|
| `ano` | int64 | 0 | 0,00 | atributo candidato |
| `id_municipio` | str | 0 | 0,00 | identificador; nunca feature |
| `rede` | str | 0 | 0,00 | atributo candidato |
| `serie` | str | 0 | 0,00 | atributo candidato |
| `taxa_alfabetizacao` | float64 | 0 | 0,00 | proibida no mesmo ano do rótulo; só defasada |
| `media_portugues` | float64 | 0 | 0,00 | resultado do mesmo ano; só defasada |
| `proporcao_aluno_nivel_0` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |
| `proporcao_aluno_nivel_1` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |
| `proporcao_aluno_nivel_2` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |
| `proporcao_aluno_nivel_3` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |
| `proporcao_aluno_nivel_4` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |
| `proporcao_aluno_nivel_5` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |
| `proporcao_aluno_nivel_6` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |
| `proporcao_aluno_nivel_7` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |
| `proporcao_aluno_nivel_8` | float64 | 11547 | 48,12 | resultado do mesmo ano (distribuição de proficiência); só defasada |
| `meta_indicador` | float64 | 23995 | 100,00 | conhecida de antemão; uso a decidir (define atingiu_meta com a taxa) |
| `percentual_participacao` | float64 | 23995 | 100,00 | atributo candidato |
| `nivel_alfabetizacao` | float64 | 23995 | 100,00 | resultado do mesmo ano; só defasada |
| `gap_pontos` | float64 | 23995 | 100,00 | proibida: diferença entre taxa e meta do mesmo ano |
| `atingiu_meta` | object | 23995 | 100,00 | alvo do município (T2); nulo quando não há meta vigente |

### `dim_municipio`

| Coluna | Tipo | Ausentes | % ausentes | Papel provisório |
|---|---|---|---|---|
| `id_municipio` | str | 0 | 0,00 | identificador; nunca feature |
| `nome` | str | 0 | 0,00 | atributo candidato |
| `sigla_uf` | str | 0 | 0,00 | atributo candidato |
| `nome_regiao` | str | 0 | 0,00 | atributo candidato |
| `capital_uf` | float64 | 1 | 0,02 | atributo candidato |
| `idhm` | float64 | 6 | 0,11 | atributo candidato |
| `idhm_educacao` | float64 | 6 | 0,11 | atributo candidato |
| `idhm_renda` | float64 | 6 | 0,11 | atributo candidato |
| `idhm_longevidade` | float64 | 6 | 0,11 | atributo candidato |

### `dim_rede`

| Coluna | Tipo | Ausentes | % ausentes | Papel provisório |
|---|---|---|---|---|
| `rede` | str | 0 | 0,00 | atributo candidato |
| `rede_desc` | str | 0 | 0,00 | atributo candidato |

### `dim_serie`

| Coluna | Tipo | Ausentes | % ausentes | Papel provisório |
|---|---|---|---|---|
| `serie` | str | 0 | 0,00 | atributo candidato |
| `serie_desc` | str | 0 | 0,00 | atributo candidato |

### `dim_tempo`

| Coluna | Tipo | Ausentes | % ausentes | Papel provisório |
|---|---|---|---|---|
| `ano` | int64 | 0 | 0,00 | atributo candidato |
| `decada` | int64 | 0 | 0,00 | atributo candidato |
| `ano_tem_meta` | bool | 0 | 0,00 | atributo candidato |
| `anos_para_meta_final` | int64 | 0 | 0,00 | atributo candidato |

### `dim_uf`

| Coluna | Tipo | Ausentes | % ausentes | Papel provisório |
|---|---|---|---|---|
| `sigla_uf` | str | 0 | 0,00 | atributo candidato |
| `nome` | str | 0 | 0,00 | atributo candidato |
