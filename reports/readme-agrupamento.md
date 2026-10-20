# Agrupamento de municípios (trilha T3)

Fragmento do README (requisitos FR7.1, FR7.2 e a tabela usada em FR9.3), consolidado na unidade de documentação. Os números vêm de `reports/agrupamento_municipios.json`, `reports/perfil_clusters.csv`, `reports/clusters_por_regiao.csv`, `reports/perfil_clusters_k3.csv`, `reports/clusters_por_regiao_k3.csv` e `reports/clusters_municipios.csv`, gerados por `python -m src.modeling.agrupamento_municipios`; os percentuais por região e por UF e a concordância entre algoritmos são contas sobre esses arquivos.

## O que foi agrupado

Os municípios da rede municipal, por seis atributos numéricos: o IDHM e seus três componentes (educação, renda e longevidade) e a taxa de alfabetização de 2023 e a de 2024. Dos 5.352 municípios do perfil, **5.222 têm os seis atributos** e foram agrupados; os 130 restantes (4 sem IDHM, 120 sem a taxa de 2023 e 6 sem a de 2024, com sobreposição) ficam de fora, e seus ids estão no JSON. `capital_uf`, UF e região **não entram na distância**: só descrevem os clusters. A capital é binária, com poucos casos, e padronizada dominaria a distância.

## Como foi feito

`StandardScaler`, depois PCA com 95% da variância acumulada, depois os três algoritmos. O PCA ficou com **4 componentes**, que explicam 97,9% da variância (3 componentes explicam 92,8%, abaixo do alvo); o primeiro sozinho explica 63,1%. Não há separação treino/teste, porque não há alvo. As regras de escolha foram fixadas no plano antes de ver qualquer resultado e estão na seção de decisões.

## Escolha de K (K-Means)

| K | Inércia (WCSS) | Silhouette | Davies-Bouldin |
|---|---|---|---|
| 2 | 15.773 | 0,4177 | 0,9609 |
| 3 | 12.121 | 0,3642 | 1,0699 |
| 4 | 10.264 | 0,2724 | 1,2546 |
| 5 | 8.860 | 0,2720 | 1,1322 |
| 6 | 8.141 | 0,2410 | 1,2852 |
| 7 | 7.506 | 0,2265 | 1,3003 |
| 8 | 6.956 | 0,2243 | 1,2351 |
| 9 | 6.581 | 0,2259 | 1,2538 |
| 10 | 6.263 | 0,2195 | 1,3009 |

Figura: `images/agrupamento_escolha_k.png`. A regra (maior silhouette médio) escolheu **K=2**. O cotovelo é uma curva suave, sem dobra nítida, e o silhouette só cai conforme K cresce.

## Comparação dos três algoritmos

| Algoritmo | Clusters | Ruído | Silhouette | Davies-Bouldin |
|---|---|---|---|---|
| K-Means (K=2) | 2 | 0,0% | 0,4177 | 0,9609 |
| Ward (K=2) | 2 | 0,0% | 0,4122 | 0,9782 |
| DBSCAN (ε=0,5598, minPts=8) | 2 | 8,6% | 0,2688 | 0,6212 |

Silhouette: quanto mais perto de 1, mais bem agrupado; perto de 0, sobreposição. Davies-Bouldin: menor é melhor. No DBSCAN as duas métricas são calculadas só nos pontos sem ruído. O K-Means e o Ward concordam em 95,2% dos municípios (4.972 de 5.222; 250 diferem).

## Perfil dos clusters (K-Means, K=2)

Os clusters são numerados do menor para o maior IDHM médio. Médias na escala original (`reports/perfil_clusters.csv`):

| Cluster | Municípios | IDHM | IDHM educação | IDHM renda | IDHM longevidade | Taxa 2023 | Taxa 2024 | Capitais |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.246 | 0,589 | 0,479 | 0,565 | 0,760 | 52,2 | 56,5 | 0,0% |
| 1 | 2.976 | 0,712 | 0,621 | 0,701 | 0,832 | 66,6 | 68,3 | 0,7% |

Figura: `images/agrupamento_pca_2d.png`. A taxa média sobe 4,2 pontos no cluster 0 e 1,7 no cluster 1 de 2023 para 2024, o que é só uma descrição, sem explicação causal.

## Regiões em cada cluster (FR9.3)

`reports/clusters_por_regiao.csv` e, por município, `reports/clusters_municipios.csv`. Percentual de cada região em cada cluster:

| Região | Municípios | Cluster 0 (menor IDHM) | Cluster 1 (maior IDHM) |
|---|---|---|---|
| Nordeste | 1.735 | 1.608 (92,7%) | 127 (7,3%) |
| Norte | 388 | 305 (78,6%) | 83 (21,4%) |
| Sudeste | 1.589 | 235 (14,8%) | 1.354 (85,2%) |
| Centro-Oeste | 459 | 47 (10,2%) | 412 (89,8%) |
| Sul | 1.051 | 51 (4,9%) | 1.000 (95,1%) |

O cluster 0 é 71,6% Nordeste e 13,6% Norte; o cluster 1 é 45,5% Sudeste, 33,6% Sul e 13,8% Centro-Oeste. **Nordeste e Norte têm o mesmo padrão, de IDHM e alfabetização mais baixos; Sul, Sudeste e Centro-Oeste têm o outro.** A divisão acompanha as regiões, mas não coincide com elas: 7,3% dos municípios do Nordeste caem no cluster de maior IDHM e 14,8% dos do Sudeste, no de menor.

## O que o agrupamento mostra, e o que não

- **Os municípios formam um contínuo, não grupos bem separados.** Indícios: o silhouette do melhor K é 0,42 e cai de K=2 para K=3 (0,36) e depois para perto de 0,22; o cotovelo é suave; na figura do PCA 2D os dois clusters são uma nuvem única cortada perto do centro do primeiro componente; e o DBSCAN só acha dois clusters em um dos valores de ε testados com ruído abaixo de 10%. O K=2 é um recorte útil, de "menor" e "maior" perfil, e não dois tipos distintos de município.
- **O DBSCAN é frágil.** Dos 7 valores de ε varridos, 3 dão dois ou mais clusters (com 29%, 22% e 8,6% de ruído) e os demais dão um só; só o de 0,5598 respeita o limite de 10% de ruído, com silhouette de 0,2688. Os 447 municípios marcados como outliers (8,6%) são os de baixa densidade nesse ε, em todas as regiões (Nordeste 218, Sul 81, Sudeste 75, Norte 53, Centro-Oeste 20) e nos dois clusters (306 no 0 e 141 no 1). Não são necessariamente anomalias nem casos a investigar.
- **Há uma leitura complementar com K=3**, descrita na seção seguinte: ela não é um refinamento do K=2 e destaca um grupo que o K=2 esconde, mas é menos firme (silhouette de 0,36) e foi escolhida depois de ver os resultados.
- **Descrição, não causa.** Os clusters resumem o que os municípios têm em comum nos seis atributos; não dizem por que são assim nem o efeito de qualquer política.
- **Dois anos só** (2023 e 2024) e 24 UFs; os 130 municípios sem dados completos não têm cluster.

## Leitura complementar com K=3 (exploratória)

O K=3 foi calculado como complemento, a pedido do usuário, **depois de ver o resultado**. Ele **não vem da regra do K**, que escolheu K=2, e por isso o JSON o marca com `escolhido_pela_regra: false`: é uma leitura exploratória, e não uma descoberta de tipos de município. Arquivos: a coluna `cluster_k3` de `reports/clusters_municipios.csv`, `reports/perfil_clusters_k3.csv`, `reports/clusters_por_regiao_k3.csv` e `images/agrupamento_pca_2d_k3.png`.

### Quanto o K=3 é firme, comparado ao K=2

| | K=2 (regra) | K=3 (complementar) |
|---|---|---|
| Silhouette | 0,4177 | 0,3642 |
| Davies-Bouldin (menor é melhor) | 0,9609 | 1,0699 |
| Concordância entre K-Means e Ward | 95,2% | 80,6% |
| Municípios na fronteira (silhouette abaixo de 0,1) | 6,2% | 9,7% |
| Silhouette médio de cada cluster | 0,352 e 0,467 | 0,363, 0,257 e 0,410 |

O K=3 é menos firme em todas as medidas, e o cluster 1 é o mais difuso (silhouette médio de 0,257).

### O que o K=3 mostra, e o K=2 esconde

| Cluster | Municípios | IDHM | IDHM educação | IDHM renda | IDHM longevidade | Taxa 2023 | Taxa 2024 | Capitais |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.423 | 0,587 | 0,474 | 0,565 | 0,759 | 39,7 | 43,7 | 0,0% |
| 1 | 1.121 | 0,608 | 0,502 | 0,583 | 0,773 | 74,4 | 78,9 | 0,0% |
| 2 | 2.678 | 0,719 | 0,629 | 0,708 | 0,835 | 65,6 | 67,0 | 0,8% |

O K=3 **não é o K=2 com mais um corte**: ele separa o grupo de IDHM baixo pela alfabetização. O cluster 1 reúne 1.121 municípios com **IDHM baixo** (0,608, perto dos 0,587 do cluster 0) e **taxa de alfabetização alta** (74,4% em 2023, acima dos 65,6% do cluster de IDHM alto). Os grupos do K=2 não se aninham nos do K=3:

| | K=3 cluster 0 | K=3 cluster 1 | K=3 cluster 2 |
|---|---|---|---|
| K=2 cluster 0 (menor IDHM) | 1.407 | 832 | 7 |
| K=2 cluster 1 (maior IDHM) | 16 | 289 | 2.671 |

Percentual de cada região em cada cluster do K=3 (`reports/clusters_por_regiao_k3.csv`):

| Região | Municípios | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|---|
| Nordeste | 1.735 | 1.054 (60,7%) | 600 (34,6%) | 81 (4,7%) |
| Norte | 388 | 254 (65,5%) | 66 (17,0%) | 68 (17,5%) |
| Sudeste | 1.589 | 66 (4,2%) | 305 (19,2%) | 1.218 (76,7%) |
| Centro-Oeste | 459 | 30 (6,5%) | 66 (14,4%) | 363 (79,1%) |
| Sul | 1.051 | 19 (1,8%) | 84 (8,0%) | 948 (90,2%) |

O cluster 1 está concentrado em poucos estados, e dá para conferir agrupando `clusters_municipios.csv` por `sigla_uf`: o **Ceará** tem 173 dos seus 183 municípios nele (94,5%); Minas Gerais tem 274 (34,2% dos 801), o Piauí 109 (49,1%), o Maranhão 100 (46,3%) e Pernambuco 95 (51,6%). Esses cinco estados juntam 751 dos 1.121 municípios do cluster (67,0%). O Ceará é também o estado com a menor proporção de municípios que não atingiram a meta de 2024 no modelo do município (8,7%), o que combina com a ideia de que, nesses municípios, a alfabetização acompanha o estado. Isto descreve onde estão e não explica por quê.

### O que o K=3 não é

- **Não é uma tipologia.** Os municípios continuam formando um contínuo; o K=3 é outro recorte dele, que destaca o grupo de IDHM baixo e alfabetização alta.
- **Não substitui o K=2.** A atribuição oficial, a do K da regra, continua sendo a do K=2; o K=3 entra como complemento exploratório na resposta de negócio.
- **Descrição, não causa.**

## Decisões e fontes

| Decisão | Fonte |
|---|---|
| Padronizar com `StandardScaler` antes do PCA e da distância | Aulas: Não supervisionados 1 (pipeline padronizar, PCA e clusterizar) e Supervisionados 6 |
| PCA com variância acumulada de 95% (`n_components=0,95`) | Aula: Não supervisionados 6 |
| K-Means com K-Means++, K pelo maior silhouette médio e cotovelo registrado | Aula: Não supervisionados 3 |
| Ward como comparação, com o mesmo K | Aula: Não supervisionados 5 |
| DBSCAN: `minPts = max(4, 2×d)` e ε pela distância ao k-ésimo vizinho (gráfico k-distância) | Aula: Não supervisionados 4 |
| Silhouette do DBSCAN só sem o ruído; Davies-Bouldin | Aula: Não supervisionados 4 |
| `n_init=10` e a semente 42 no K-Means | Decisão do time |
| `capital_uf` fora da distância, só na descrição | Decisão do time |
| Grade de ε nos quantis 0,5 a 0,99 da distância k; ε com 2 a 10 clusters e até 10% de ruído | Decisão do time, fixada no plano antes dos resultados |
| Clusters numerados do menor para o maior IDHM médio | Decisão do time |
| Sem dendrograma | Decisão do time (5.222 pontos) |
| K=3 como leitura complementar, marcada como exploratória | Decisão do time, a pedido do usuário depois de ver os resultados; não vem da regra do K |
| Silhouette por município (média por cluster e percentual na fronteira, com silhouette abaixo de 0,1) | Aula: Não supervisionados 2 (definição do silhouette por ponto); o limite de 0,1 é decisão do time |

Custo da execução (máquina de 10 núcleos e 16 GiB): cerca de 5 s.
