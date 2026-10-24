# Ranking dos municípios: metas, taxa observada e perfil

Os arquivos são gerados por `python -m src.modeling.ranking_municipios`. O tempo depende do computador e do ambiente.

## Como é feito

As probabilidades são calculadas fora da dobra: cada registro é previsto por um modelo ajustado sem os registros do seu município. São cinco dobras por município. O algoritmo e os hiperparâmetros permanecem os escolhidos no treinamento; os modelos salvos não são sobrescritos. Em T1, cada ajuste usa até 100.000 registros e prevê todas as linhas da dobra.

A avaliação descreve os dados disponíveis. Ela não é uma validação temporal nem uma avaliação independente de todo o processo de seleção de hiperparâmetros.

## Três arquivos, três perguntas

| Arquivo em reports/ | Pergunta | Limite de interpretação |
|---|---|---|
| `ranking_risco_meta_municipios.csv` | Quais municípios apresentam maior probabilidade estimada de não atingir a meta de 2024? | Não demonstra previsão para anos futuros |
| `ranking_taxa_nao_alfabetizados_municipios.csv` | Quais municípios apresentam maior proporção observada de registros de alunos não alfabetizados? | É descrição do resultado observado, não previsão individual |
| `ranking_abaixo_do_perfil_municipios.csv` | Onde a taxa observada de não alfabetização supera mais a taxa estimada pelo modelo? | Mede excesso em relação ao perfil, não o nível absoluto de necessidade educacional |

O ranking de taxa observada e o de excesso incluem municípios com pelo menos 50 registros. Esse limite reduz instabilidade, mas não elimina incerteza. As colunas de contagem permitem considerar o tamanho da base antes de priorizar ações. O excesso pode refletir atributos ausentes, erros do modelo ou variação amostral; não identifica causa nem prova falha de gestão.

## Período e cobertura

T1 agrega os registros com caderno preenchido de 2023 e 2024, conforme a tabela utilizada no treinamento. A contagem representa registros, não alunos únicos acompanhados entre anos. Os rankings de taxa observada e excesso incluem 5.134 municípios, com mediana de 236 registros por município. A taxa não deve ser apresentada como resultado exclusivo de 2024. Alterar o período produz outra análise e exige recalcular os arquivos.

T2 avalia a meta de 2024 usando atributos disponíveis para essa tarefa, incluindo informações de 2023. São 5.226 municípios. AC, DF e RR não estão representados no ranking T2. A cobertura é limitada aos municípios com dados necessários e rótulo disponível; a lista de UFs presentes e ausentes está no JSON. Ausência de um município ou estado no arquivo não significa baixo risco.

O ranking usa informações compartilhadas por municípios do mesmo estado. Não foi demonstrado que a posição dentro da UF seja mais confiável para prever anos futuros. A posição estadual é somente uma forma de organizar os resultados desta análise.

## Evidências e uso

`ranking_municipios.json` registra as AUCs fora da dobra, cobertura, quantidade de municípios, mediana de registros incluídos e os maiores valores de cada ranking. As probabilidades estimadas, taxas observadas e excessos devem permanecer identificados separadamente.

![Municípios no quintil de maior risco de meta por UF](../images/ranking_alto_risco_por_uf.png)

As comparações devem apoiar investigação e revisão humana, considerando população atendida, cobertura e contexto local. Nenhum desses rankings determina sozinho uma ação pública ou demonstra o efeito de uma política.

## Organização técnica

O módulo em modeling coordena o cálculo e a geração dos artefatos, incluindo uma função de visualization. Essa responsabilidade de orquestração é uma exceção documentada à separação das camadas. A reorganização pode ser feita posteriormente, preservando o comando atual.
