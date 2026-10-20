# Treino da trilha T1: o aluno é alfabetizado?

Fragmento do README, a consolidar na unidade de documentação. Resultado de `python -m src.modeling.treino_aluno` (cerca de 7 minutos), gravado em [treino_t1.json](treino_t1.json) e [comparacao_modelos_t1.csv](comparacao_modelos_t1.csv).

## Protocolo

É o mesmo de T2 (`src/modeling/treino.py`, descrito em [readme-treino-t2.md](readme-treino-t2.md)), com uma diferença: a tabela tem 3,35 milhões de alunos, grande demais para a busca de hiperparâmetros com validação cruzada. Depois da separação por município, o treino é reduzido a uma amostra de 100 mil alunos e a validação a uma de 50 mil, por sorteio simples (semente 42). **O teste é avaliado inteiro.**

A amostragem limita o custo computacional. Em comparações relatadas
pelo revisor, o Gradient Boosting passou de AUC 0,6183 com 100 mil
registros para 0,6206 com 600 mil, na validação inteira.
A Regressão Logística passou de 0,6128 para 0,6146 usando o treino inteiro.

Essas verificações indicaram ganhos pequenos nos cenários avaliados,
mas ainda não foram reproduzidas nesta branch nem automatizadas no pipeline.

São usadas grades menores para limitar o custo da busca de
hiperparâmetros. O tempo de execução depende do computador e do ambiente.

Antes da amostragem, as partições contêm 1.971.880 registros no treino,
755.768 na validação e 627.013 no teste.

A busca e o treinamento inicial usam a amostra de 100.000 registros.
A avaliação de validação usa a amostra de 50.000 registros.
O modelo final é reajustado com essas duas amostras reunidas,
totalizando 150.000 registros, e não com todas as linhas disponíveis
no treino e na validação. O teste é avaliado inteiro.

## Resultado

| Conjunto | Alunos | Municípios | Não alfabetizados |
|---|---|---|---|
| Treino (amostra) | 100.000 | 3.266 | 40,4% |
| Validação (amostra) | 50.000 | 1.095 | 41,8% |
| Teste (inteiro) | 627.013 | 1.110 | 40,7% |

Comparação na validação cruzada do treino:

| Modelo | AUC-ROC (CV) | Desvio | AUC-ROC treino | Overfitting |
|---|---|---|---|---|
| **Gradient Boosting** (100 árvores, profundidade 3, taxa 0,1) | **0,628** | 0,011 | 0,648 | Não |
| Random Forest (100 árvores, profundidade 20, folha mínima 100) | 0,627 | 0,015 | 0,654 | Não |
| Regressão Logística (`C=10`) | 0,627 | 0,014 | 0,634 | Não |
| Regressão Logística padrão | 0,627 | 0,014 | 0,634 | Não |
| Dummy | 0,500 | 0,000 | 0,500 | Não |

**Modelo escolhido pela regra: Gradient Boosting.** Os três candidatos, porém, estão empatados: a diferença entre eles (0,001) é dez vezes menor que o desvio entre as dobras (cerca de 0,014). Trocar pela Regressão Logística, mais simples de explicar, não muda o desempenho; a decisão é do time.

| Métrica (classe positiva = não alfabetizado, limiar 0,5) | Validação | Teste |
|---|---|---|
| AUC-ROC | 0,618 | 0,632 |
| AUC-PR | 0,523 | 0,525 |
| Precisão | 0,585 | 0,590 |
| Recall | 0,287 | 0,195 |
| F1 | 0,385 | 0,293 |

Matriz de confusão no teste: 337.632 verdadeiros negativos, 34.505 falsos positivos, 205.133 falsos negativos e 49.743 verdadeiros positivos.

## Leitura

- **O teto é o território.** Todos os atributos de T1 são do município, da rede e do caderno: dois alunos do mesmo município e caderno recebem a mesma probabilidade. O modelo separa municípios mais e menos vulneráveis, não alunos. A AUC-ROC de cerca de 0,63 confirma o gradiente modesto que a EDA mostrou (de 50,9% a 64,5% de alfabetizados entre as regiões).
- **O recall é baixo no limiar de 0,5.** Como só 40% dos alunos são não alfabetizados e as probabilidades ficam concentradas perto da média, poucos passam de 0,5, e o modelo deixa de apontar 80% dos alunos em risco. Para uso como triagem, o limiar precisaria ser escolhido pelo custo de cada erro; as aulas não ensinam como, então o projeto mantém 0,5 e mostra a AUC-ROC, que não depende do limiar.
- **Sem overfitting.** A razão de erro validação/treino fica perto de 1 em todos os candidatos.
- **O mesmo efeito de estado de T2 vale aqui**, com peso menor: o treino traz alunos de outros municípios do mesmo estado e ano.
