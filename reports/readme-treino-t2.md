# Treino da trilha T2: o município vai atingir a meta?

Fragmento do README, a consolidar na unidade de documentação. Resultado de `python -m src.modeling.treino_municipio` (cerca de 1 minuto), gravado em [treino_t2.json](treino_t2.json) e [comparacao_modelos_t2.csv](comparacao_modelos_t2.csv).

## Protocolo

O treino segue `src/modeling/treino.py`, o mesmo de T1:

1. Treino, validação e teste (60/20/20) sorteados **por município**, estratificados pela região. Nenhum município aparece em dois conjuntos.
2. Baselines (`DummyClassifier` e Regressão Logística padrão) com validação cruzada agrupada (`GroupKFold`, 5 dobras), só no treino.
3. Busca de hiperparâmetros de cada candidato pela AUC-ROC de validação cruzada, só no treino: em grade para a Regressão Logística e aleatória para as árvores (Aula: Otimização 2).
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
| Gradient Boosting (200 árvores, profundidade 2, taxa 0,1) | 0,773 | 0,014 | 0,848 | Não (razão 1,49) |
| Random Forest (300 árvores, profundidade 10) | 0,771 | 0,013 | 0,953 | **Sim** (razão 4,85) |
| Dummy | 0,500 | 0,000 | 0,500 | Não |

**Modelo escolhido: Regressão Logística.** Ela empata com as árvores na validação cruzada, não sofre overfitting e é a mais simples de interpretar.

| Métrica (classe positiva = não atingiu a meta, limiar 0,5) | Validação | Teste |
|---|---|---|
| AUC-ROC | 0,752 | 0,796 |
| AUC-PR | 0,730 | 0,767 |
| Precisão | 0,675 | 0,690 |
| Recall | 0,657 | 0,665 |
| F1 | 0,666 | 0,677 |

Matriz de confusão no teste: 425 verdadeiros negativos, 143 falsos positivos, 160 falsos negativos e 318 verdadeiros positivos.

## Ressalva: a maior parte do sinal é o estado

A EDA mostrou que nenhuma variável isolada separa `atingiu_meta` (AUC-ROC bruta entre 0,48 e 0,52), e a Fase 2 obteve 0,57. A AUC-ROC de 0,78 vem quase toda da UF: a proporção de municípios que não atingiram a meta vai de 9% no Ceará a 91% no Rio Grande do Sul.

| Verificação (Regressão Logística, validação cruzada) | AUC-ROC |
|---|---|
| Só UF e região | 0,732 |
| Todos os atributos, sem UF e região | 0,634 |
| Todos os atributos, dobras agrupadas por UF (estado inteiro fora do treino) | **0,580** |

Não há vazamento de linha: nenhuma coluna de 2024 do próprio município entra, e os municípios do teste nunca foram vistos. Mas, como o sorteio é por município, o treino traz municípios de 2024 do mesmo estado que o teste, e o modelo aprende **como cada estado se saiu em 2024**. Isso tem duas leituras:

- **Para ranquear municípios dentro de 2024** (quais estão mais em risco, sabendo como o estado foi), 0,80 é uma estimativa válida.
- **Para prever metas futuras** (pergunta de negócio 4), o desempenho do estado no ano ainda não é conhecido, e a estimativa honesta fica perto de **0,58**. Com só um ano de rótulo, não dá para validar no ano seguinte (ver Limitações).

A decisão de qual número usar no README, e se a separação deve mudar para agrupar por UF, é do time.
