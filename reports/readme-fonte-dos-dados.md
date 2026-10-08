# Fonte dos dados: snapshot local em vez de leitura direta do GCP

Nota para o README, a consolidar na unidade de documentação. Cabe nas seções "Descrição da base utilizada", "Limitações do projeto" e "Possíveis evoluções futuras".

## O que foi feito

O projeto lê tabelas da camada Gold da Fase 2 gravadas em Parquet em `data/` (9 tabelas, cerca de 30 MiB), e não consulta o BigQuery em cada execução. O código de leitura está em `src/preprocessing/loader.py` e o de exportação em `src/preprocessing/export_gold.py`.

## Por que

- **A Gold da Fase 2 é efêmera.** A infraestrutura foi desenhada para subir, ser usada e ser destruída, para não gerar custo. Hoje o dataset não existe mais no projeto GCP. Ler direto do GCP exigiria recriar a Gold a cada uso: Terraform, Bronze, Silver e Gold levaram de 25 a 30 minutos na recriação desta fase.
- **Reprodutibilidade (requisito do enunciado).** Quem clona o repositório roda o projeto sem projeto GCP, faturamento nem credencial.
- **Dados fixos.** As métricas do README ficam atreladas a um dado que não muda entre execuções.

## O que isso custa

- O repositório cresce: o maior arquivo, `fact_alunos.parquet`, tem 29,1 MiB (o GitHub avisa a partir de 50 MiB), e cada regeneração acrescenta outros ~30 MiB ao histórico. Por isso o snapshot só deve ser regenerado se o dado mudar de fato.
- É uma cópia: se a Gold da Fase 2 mudar, o snapshot fica desatualizado até alguém rodar o exportador.

## Como voltar a ler do GCP

Foi uma escolha, e não uma limitação técnica. `carregar_tabela`, em `src/preprocessing/loader.py`, é o único ponto de leitura dos dados. Para ler do GCP:

1. Recriar a Gold com o pipeline do trabalho da Fase 2 (repositório `TECH-CHALLENGE-FASE-2`: `make infra-apply`, depois `make bronze silver gold`), no dataset `alfabetizacao_analytics`.
2. Fazer `carregar_tabela` devolver o DataFrame de uma consulta a esse dataset, em vez de ler o arquivo. `export_gold.py` já mostra as consultas e as colunas de cada tabela.
3. Conferir os tipos dos DataFrames nos testes: as duas fontes podem diferir um pouco.

Nenhuma outra parte do código chama a leitura de arquivos diretamente.
