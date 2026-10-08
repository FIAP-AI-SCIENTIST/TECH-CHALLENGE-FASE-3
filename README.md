# Predição e inteligência analítica para alfabetização no Brasil

Tech Challenge da Fase 3 da pós-graduação em AI Scientist da FIAP (Pós Tech). A partir da camada Gold construída na Fase 2, o projeto analisa e modela o Indicador Criança Alfabetizada para apoiar decisões de políticas públicas educacionais.

> **Estado do projeto.** Este README descreve o que já está no repositório. A modelagem supervisionada final, a interpretação com SHAP, o agrupamento de municípios e as respostas às perguntas de negócio ainda não foram entregues aqui, e as seções delas dizem isso. **Nenhum número abaixo é resultado de um modelo final.** O único modelo no repositório é um baseline provisório, descartável (ver [Baseline provisório](#baseline-provisório)).

## Sumário

- [Estado das etapas](#estado-das-etapas)
- [Contexto do problema](#contexto-do-problema)
- [Objetivo analítico](#objetivo-analítico)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como rodar](#como-rodar)
- [Descrição da base utilizada](#descrição-da-base-utilizada)
- [Etapas de modelagem](#etapas-de-modelagem)
- [Escolha do algoritmo](#escolha-do-algoritmo)
- [Métricas de avaliação](#métricas-de-avaliação)
- [Interpretação dos resultados](#interpretação-dos-resultados)
- [Insights encontrados](#insights-encontrados)
- [Limitações do projeto](#limitações-do-projeto)
- [Aplicação prática para políticas públicas](#aplicação-prática-para-políticas-públicas)
- [Possíveis evoluções futuras](#possíveis-evoluções-futuras)
- [Decisões e exceções ao que foi ensinado](#decisões-e-exceções-ao-que-foi-ensinado)
- [Práticas de desenvolvimento](#práticas-de-desenvolvimento)

## Estado das etapas

| Etapa | Estado |
|---|---|
| Dados: snapshot da Gold, exportador e leitor | Pronto |
| Análise exploratória e dicionário de dados | Pronta |
| Pré-processamento: tabelas de modelagem, colunas proibidas e pré-processador | Pronto |
| Baseline provisório (caminho ponta a ponta) | Pronto, descartável |
| Separação por município, validação cruzada, busca de hiperparâmetros e comparação de modelos | Ainda não entregue aqui |
| Interpretação (Feature Importance e SHAP) | Ainda não entregue |
| Agrupamento de municípios | Ainda não entregue |
| Respostas às perguntas de negócio, documentação técnica e vídeo executivo | Ainda não entregues |

## Contexto do problema

O Compromisso Nacional Criança Alfabetizada é uma política pública (União, estados, Distrito Federal e municípios) que busca garantir que toda criança brasileira esteja alfabetizada até o fim do 2º ano do ensino fundamental, com meta de 100% até 2030. O Indicador Criança Alfabetizada mede o percentual de estudantes que atingem o corte de 743 pontos na escala Saeb (Pesquisa Alfabetiza Brasil, INEP).

Conhecer só os dados atuais não basta: gestores públicos precisam antecipar riscos, identificar regiões vulneráveis e entender quais fatores mais pesam nos indicadores. A Fase 2 entregou o pipeline de engenharia de dados e a camada Gold; esta fase usa essa camada para construir análises e modelos de Machine Learning.

## Objetivo analítico

Desenvolver um modelo supervisionado que preveja se um aluno será considerado alfabetizado ou não, usando variáveis educacionais, territoriais e socioeconômicas, e transformar isso em inteligência aplicável. O projeto responde às cinco perguntas de negócio do enunciado:

1. Quais fatores mais impactam a alfabetização?
2. Quais municípios apresentam maior risco educacional?
3. Quais regiões possuem padrões semelhantes?
4. Como prever municípios que podem não atingir metas futuras?
5. Quais variáveis têm maior influência nos modelos?

O trabalho se organiza em quatro trilhas: **T1**, modelo supervisionado do aluno (`alfabetizado`); **T2**, modelo supervisionado do município (`atingiu_meta`, só 2024); **T3**, agrupamento de municípios; **T4**, interpretação (Feature Importance e SHAP). A classe de risco é sempre a positiva: *não alfabetizado* e *não atingiu a meta*.

O projeto segue apenas o que foi ensinado nas aulas; o que o enunciado exige e as aulas não cobrem está marcado em [Decisões e exceções](#decisões-e-exceções-ao-que-foi-ensinado).

## Estrutura do repositório

```
data/                     snapshot da Gold em Parquet (9 tabelas) e data/README.md com a origem
notebooks/                01_esqueleto_baseline.ipynb e 02_eda.ipynb
src/
  config.py               semente única, caminhos, tabelas da Gold e colunas por trilha
  preprocessing/          leitor e exportador do snapshot, dicionário de dados, tabelas, colunas proibidas e pré-processador
  modeling/               baseline provisório
  evaluation/             (ainda vazio)
  visualization/          histograma, barras do alvo e mapa de correlação
tests/                    testes com pytest
reports/                  dicionário de dados, resultado do baseline e fragmentos de documentação
images/                   figuras da análise exploratória
requirements.txt          dependências com versões fixas
```

## Como rodar

Testado com **Python 3.12**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q                       # suíte de testes
python -m src.preprocessing.qualidade     # regenera reports/dicionario_dados.md
python -m src.modeling.baseline           # baseline provisório (grava reports/baseline_provisorio.json)
```

Os notebooks em `notebooks/` abrem em qualquer editor com suporte a Jupyter (o ambiente já traz o `ipykernel`) e leem o snapshot de `data/`, sem credencial.

Para regenerar o snapshot a partir do BigQuery é preciso ter a Gold da Fase 2 criada no seu projeto GCP e credencial de aplicação somente leitura (`gcloud auth application-default login`):

```bash
GCP_PROJECT_ID=<id-do-projeto> python -m src.preprocessing.export_gold
```

Os detalhes da origem dos dados e da regeneração estão em [data/README.md](data/README.md).
