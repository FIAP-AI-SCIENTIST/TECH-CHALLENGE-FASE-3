# Treino da trilha T2: o município vai atingir a meta?

Fragmento do README, a consolidar na unidade de documentação. Resultado de `python -m src.modeling.treino_municipio` (cerca de 1 minuto), gravado em [treino_t2.json](treino_t2.json) e [comparacao_modelos_t2.csv](comparacao_modelos_t2.csv).

## Protocolo

O treino segue `src/modeling/treino.py`, o mesmo de T1:

1. Treino, validação e teste (60/20/20) sorteados **por município**, estratificados pela região. Nenhum município aparece em dois conjuntos.
2. Baselines (`DummyClassifier` e Regressão Logística padrão) com validação cruzada agrupada (`GroupKFold`, 5 dobras), só no treino.
3. Busca de hiperparâmetros pela AUC-ROC de validação cruzada, somente no treino: em grade para Regressão Logística e Random Forest (24 combinações), e aleatória para Gradient Boosting.
4. Escolha do candidato de maior AUC-ROC de validação cruzada, nunca pelo teste.
5. Conferência do escolhido na validação, com o diagnóstico de overfitting (razão de erro `1 - AUC-ROC` validação/treino ≥ 2; Aula: Otimização 5).
6. Reajuste em treino + validação e avaliação do teste uma única vez.

## Resultado

| Conjunto | Municípios | Não atingiram a meta |
|---|---|---|
| Treino | 3.135 | 47,1% |
| Validação | 1.045 | 46,9% |
| Teste | 1.046 | 45,7% |

Comparação na validação cruzada do treino:

| Modelo | AUC-ROC (CV) | Desvio | AUC-ROC treino | Overfitting |
|---|---|---|---|---|
| **Regressão Logística** (`C=10`) | **0,780** | 0,015 | 0,791 | Não |
| Regressão Logística padrão | 0,780 | 0,015 | 0,790 | Não |
| Random Forest (300 árvores, profundidade 10, mínimo de 5 registros por folha) | 0,773 | 0,014 | 0,890 | **Sim** (razão 2,07) |
| Gradient Boosting (200 árvores, profundidade 2, taxa 0,1) | 0,773 | 0,014 | 0,848 | Não (razão 1,49) |
| Dummy | 0,500 | 0,000 | 0,500 | Não |

**Modelo escolhido: Regressão Logística.** Apresentou a maior AUC-ROC média de validação cruzada entre os candidatos, considerando os valores não arredondados. O diagnóstico adotado não sinalizou overfitting para ela.

| Métrica (classe positiva = não atingiu a meta, limiar 0,5) | Validação | Teste |
|---|---|---|
| AUC-ROC | 0,752 | 0,796 |
| AUC-PR | 0,730 | 0,767 |
| Precisão | 0,675 | 0,690 |
| Recall | 0,657 | 0,665 |
| F1 | 0,666 | 0,677 |

Matriz de confusão no teste: 425 verdadeiros negativos, 143 falsos positivos, 160 falsos negativos e 318 verdadeiros positivos.

## Ressalva: associação com a UF e limites da avaliação

A análise territorial usa somente os 3.135 municípios do treino,
com Regressão Logística padrão e cinco dobras. Os resultados são
gerados pelo script e registrados no bloco `analise_uf` de
`reports/treino_t2.json`.

| Verificação | AUC-ROC média | Desvio entre dobras |
|---|---|---|
| Somente UF e região; agrupamento por município | 0,731 | 0,010 |
| Demais atributos, sem UF e região; agrupamento por município | 0,640 | 0,027 |
| Todos os atributos; agrupamento por UF | 0,587 | 0,074 |

Os resultados sugerem uma associação relevante entre UF e cumprimento
das metas nesta base. Não demonstram um efeito causal do estado nem
quantificam uma contribuição exclusiva da UF.

Na avaliação agrupada por UF, as AUCs variaram de 0,445 a 0,660.
O desvio apresentado mede a variação entre dobras; não é um intervalo
de confiança.

A avaliação final por município apresentou AUC-ROC de 0,7955 no teste.
Ela avalia municípios separados do treino, mas pertencentes ao mesmo
ano dos rótulos, 2024. Não comprova desempenho em anos futuros.

Separar estados inteiros avalia transferência entre UFs, não entre
anos. Portanto, a AUC média de 0,587 não deve ser apresentada como
estimativa de desempenho futuro.

Os atributos incluem a meta de 2024, considerada conhecida antes do
resultado, e a folga calculada a partir dela e da taxa de 2023.
A taxa de alfabetização de 2024 não é usada como atributo.

Como há apenas um ano com rótulo municipal disponível, a generalização
temporal permanece sem validação.

## Ambiente de reprodução

Os relatórios desta revisão foram regenerados em:

- Windows 11, arquitetura AMD64;
- Python 3.13.13;
- scikit-learn 1.9.1;
- NumPy 2.5.3;
- pandas 3.0.6;
- SciPy 1.18.1.

Esse registro descreve a execução atual, não o ambiente que gerou
a primeira versão do relatório.

A semente controla as fontes de aleatoriedade configuradas no código,
mas não garante resultados idênticos entre sistemas, arquiteturas
e implementações numéricas diferentes. O arredondamento também não
garante igualdade entre ambientes.

O revisor relatou pequenas diferenças em métricas das árvores ao
executar em macOS ARM64. Nesta revisão, a busca da Random Forest
também mudou de aleatória para completa; portanto, a mudança dos
resultados da floresta não deve ser atribuída apenas ao ambiente.

Os números das tabelas correspondem aos relatórios regenerados
no ambiente registrado acima.

## Organização do protocolo

`src/modeling/treino.py` atua como orquestrador do experimento:
coordena a preparação dos conjuntos, o treinamento, a avaliação
e a gravação dos relatórios.

Por essa responsabilidade, ele importa funções de `evaluation`,
como exceção documentada à direção de dependências adotada no projeto.
Os cálculos de métricas e diagnóstico permanecem em `evaluation`.

Essa escolha preserva o ponto de entrada compartilhado pelas trilhas
supervisionadas, sem mover módulos usados pelas PRs dependentes.