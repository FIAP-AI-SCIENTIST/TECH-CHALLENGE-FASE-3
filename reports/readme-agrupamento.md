# Agrupamento de municípios (trilha T3)

Fragmento do README (requisitos FR7.1, FR7.2 e a tabela usada em FR9.3), consolidado na unidade de documentação. Os números vêm de `reports/agrupamento_municipios.json`, `reports/perfil_clusters.csv`, `reports/clusters_por_regiao.csv` e `reports/clusters_municipios.csv`, gerados por `python -m src.modeling.agrupamento_municipios`; os percentuais por região e a concordância entre algoritmos são contas sobre esses arquivos.

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
- **Quem precisar de mais de dois perfis** pode usar K=3, mas os dados não sustentam grupos bem separados (silhouette de 0,36).
- **Descrição, não causa.** Os clusters resumem o que os municípios têm em comum nos seis atributos; não dizem por que são assim nem o efeito de qualquer política.
- **Dois anos só** (2023 e 2024) e 24 UFs; os 130 municípios sem dados completos não têm cluster.

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

Custo da execução (máquina de 10 núcleos e 16 GiB): cerca de 5 s.
