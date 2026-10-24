# Interpretação dos modelos (trilha T4): Feature Importance e SHAP

Fragmento do README, a consolidar na unidade de documentação. Gerado por `python -m src.evaluation.interpretacao_modelos` a partir dos modelos em `models/`. Tabelas em `reports/interpretacao_{nativa,permutacao,shap}_{t1,t2}.csv` e gráficos em `images/interpretacao_*`.

## Como é feito

Três leituras do mesmo modelo, todas sobre o **conjunto de teste** (dados que o modelo não viu; a importância medida no treino premiaria o que ele decorou):

| Leitura | O que mede | Onde fica |
|---|---|---|
| Importância nativa | Coeficiente da Regressão Logística (com sinal, na escala padronizada) ou `feature_importances_` das árvores | Uma linha por coluna do One-Hot (`sigla_uf_BA`) |
| Permutação | Queda da AUC-ROC quando a coluna original é embaralhada (10 repetições) | Uma linha por atributo original (`sigla_uf` inteira) |
| SHAP | Contribuição de cada atributo para cada previsão, na escala do algoritmo; o gráfico de barras soma as colunas One-Hot de cada atributo antes do módulo | As duas formas |

A permutação mede a queda de AUC-ROC de cada atributo original. Os coeficientes da Regressão Logística preservam o sinal; a importância nativa das árvores não informa direção. O SHAP informa contribuições por previsão: em log-odds na Regressão Logística e no Gradient Boosting, e em probabilidade na Random Forest. As unidades não devem ser comparadas diretamente entre algoritmos.

No teste de T1, a permutação usa 20 mil alunos e o SHAP 2 mil; em T2 entram os 1.046 municípios. A semente controla a amostragem do fundo e o desenho do resumo SHAP. A análise descreve os modelos já escolhidos; não é usada para selecionar atributos ou ajustar modelos depois de abrir o teste. As contribuições descrevem associações, não efeitos de políticas públicas.

## T2 (município): o que pesa em não atingir a meta

![Permutação de T2](../images/interpretacao_permutacao_t2.png)
![SHAP de T2](../images/interpretacao_shap_resumo_t2.png)

| Atributo | Queda da AUC-ROC (permutação) | Média do \|SHAP\| |
|---|---|---|
| `sigla_uf` | **0,195** | 0,94 |
| `idhm` | 0,133 | 1,35 |
| `meta_indicador_2024` | 0,111 | 0,87 |
| `idhm_educacao` | 0,097 | 0,78 |
| `media_portugues_2023` | 0,081 | 0,67 |
| `idhm_renda` | 0,069 | 0,60 |
| `taxa_alfabetizacao_2023` | 0,063 | 0,56 |
| `nome_regiao` | 0,040 | 0,41 |
| `folga_2023` | 0,016 | 0,25 |
| `idhm_longevidade` | 0,008 | 0,13 |
| `capital_uf` | 0,000 | 0,00 |

Leitura:

- A UF apresenta a maior queda de AUC-ROC por permutação (0,195), seguida do IDHM (0,133). No SHAP médio absoluto, a ordem desses dois atributos se inverte. As técnicas respondem a perguntas diferentes.
- A folga é calculada por `taxa_alfabetizacao_2023 - meta_indicador_2024`. Alterar a taxa ou a meta exige atualizar a folga de forma consistente. O sinal de um coeficiente isolado não determina o efeito líquido dessa alteração.
- IDHM e seus componentes são correlacionados. Suas contribuições devem ser interpretadas em conjunto; os coeficientes não demonstram que maior desenvolvimento aumente o risco.
- A importância da UF não identifica a causa da diferença entre estados e não constitui um ranking estadual de risco.

## T1 (aluno): o que pesa em não ser alfabetizado

![Permutação de T1](../images/interpretacao_permutacao_t1.png)
![SHAP de T1](../images/interpretacao_shap_resumo_t1.png)

| Atributo | Queda da AUC-ROC (permutação) | Média do \|SHAP\| |
|---|---|---|
| `sigla_uf` | **0,110** | 0,30 |
| `nome_regiao` | 0,009 | 0,08 |
| `idhm_renda` | 0,003 | 0,03 |
| `idhm_longevidade` | 0,002 | 0,03 |
| `capital_uf` | 0,001 | 0,01 |
| `caderno` | 0,000 | 0,00 |
| `rede` | 0,000 | 0,03 |
| `idhm` | −0,001 | 0,02 |
| `idhm_educacao` | −0,001 | 0,02 |

Leitura:

- A UF apresenta a maior queda de AUC-ROC por permutação (0,110), seguida da região (0,009). Essas importâncias não são parcelas aditivas de uma explicação causal.
- Os demais atributos apresentam quedas pequenas nesta amostra e neste modelo. Isso não demonstra ausência de relevância educacional nem que a UF substitua suas causas.
- O resumo SHAP mistura categorias e frequências. Não permite concluir, sozinho, quais estados têm maior risco médio. Essa comparação exige agrupar previsões por UF e explicitar cobertura e período.
- Rede e caderno apresentam importância pequena nas métricas registradas; a conclusão se limita ao modelo e à amostra avaliados.

## O que isso responde

| Pergunta | T1: aluno | T2: município |
|---|---|---|
| Quais variáveis influenciam o modelo? | UF tem a maior queda por permutação; região vem depois | UF e IDHM se destacam; meta e resultados de 2023 também contribuem |
| Isso comprova fatores que causam alfabetização? | Não; são associações preditivas | Não; atributos correlacionados e derivados dificultam leituras isoladas |

## Limitações

- Os atributos correlacionados (IDHM e componentes; taxa e média de Português de 2023) dividem a importância entre si, e o sinal de cada um isoladamente não tem leitura direta.
- A permutação de um atributo correlacionado cria combinações irreais (IDHM alto com IDHM educação baixo), o que pode inflar a sua importância.
- A UF concentra o efeito de tudo o que é comum ao estado: política estadual, contexto socioeconômico e o desempenho do estado no ano. O modelo não separa essas causas.

A importância calculada no teste descreve esta avaliação. Ela não comprova desempenho em anos futuros.
