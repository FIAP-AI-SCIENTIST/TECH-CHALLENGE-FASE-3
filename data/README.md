# Snapshot da camada Gold (Fase 2)

Cópia em Parquet das tabelas da Gold usadas neste projeto. Quem tem acesso ao projeto GCP a regenera; o restante do projeto lê estes arquivos sem credencial (`src/preprocessing/loader.py`).

## Origem

- **Fonte:** dataset BigQuery `alfabetizacao_analytics` do projeto GCP do grupo, criado pelo pipeline da Fase 2 (Bronze, Silver e Gold sobre `basedosdados.br_inep_avaliacao_alfabetizacao`, Pesquisa Alfabetiza Brasil, INEP).
- **Extração:** 2026-10-07, com `python -m src.preprocessing.export_gold` (consulta `SELECT <colunas> FROM <projeto>.alfabetizacao_analytics.<tabela>`, Parquet com compressão zstd).
- **Gold recriada** com o pipeline batch da Fase 2 (`make bronze silver gold`), sem o streaming: não há eventos sintéticos nos dados (`fact_alunos` tem 3.867.999 linhas, o mesmo total da Bronze).
- **Credencial:** Application Default Credentials, somente leitura (`gcloud auth application-default login`); nenhuma chave de serviço é usada ou versionada.

## Regeneração

```bash
export GCP_PROJECT_ID=<id-do-projeto>
python -m src.preprocessing.export_gold
```

## Colunas removidas

- Todas as chaves substitutas `sk_*` (servem só para joins no BigQuery; o snapshot mantém `id_municipio` e `rede`).
- `id_aluno` de `fact_alunos` (identificador individual, sem uso como feature).

## Tabelas

| Arquivo | Grão | Linhas | Tamanho |
|---|---|---|---|
| `fact_alunos.parquet` | aluno | 3.867.999 | 29,2 MiB |
| `fact_meta_resultado_municipio.parquet` | município, rede, ano (versão SCD2) | 10.698 | 0,12 MiB |
| `fact_indicador_municipio.parquet` | ano, município, série, rede | 23.995 | 0,40 MiB |
| `fact_alfabetizacao_municipio.parquet` | ano, município, rede, série | 23.995 | 0,40 MiB |
| `dim_municipio.parquet` | município | 5.571 | 0,10 MiB |
| `dim_rede.parquet` | rede | 5 | < 0,01 MiB |
| `dim_serie.parquet` | série | 1 | < 0,01 MiB |
| `dim_tempo.parquet` | ano | 8 | < 0,01 MiB |
| `dim_uf.parquet` | UF | 27 | < 0,01 MiB |

Total aproximado: 30 MiB.

## O que a verificação do snapshot mostrou

- **Anos reais:** 2023 e 2024 (não há 2025). `fact_alunos` tem 1.747.439 alunos em 2023 e 2.120.560 em 2024.
- **Rótulo do município (`atingiu_meta`):** só existe para 2024, que na tabela `fact_meta_resultado_municipio` tem 5.346 linhas: 2.782 atingiram, 2.444 não atingiram e 120 são nulas. Em 2023 todas as 5.352 linhas são nulas.
- **Rótulo do aluno (`alfabetizado`):** é exatamente `proficiencia >= 743` entre os alunos com proficiência (verificado nos 3.354.661 casos sem nulo).
- **Ausentes e caderno em branco:** 513.338 alunos (13,3%) têm `preenchimento_caderno` falso, `proficiencia` e `peso_aluno` nulos e `alfabetizado` falso. Entre os que preencheram, 59,2% são alfabetizados. Portanto `presenca` e `preenchimento_caderno` determinam o rótulo para esses alunos e não servem como feature.
