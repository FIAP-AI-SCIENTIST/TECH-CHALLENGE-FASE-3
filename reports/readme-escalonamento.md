# Pré-processamento: imputação, escalonamento e codificação

Fragmento do README (requisito FR4.2), consolidado na unidade de documentação. Toda decisão cita a aula ou está marcada como "exigida pelo enunciado, não coberta nas aulas" ou "decisão do time".

## Escalonamento por modelo

| Modelo | Escalonamento | Motivo |
|---|---|---|
| Regressão Logística | `StandardScaler` | Otimização iterativa e regularização dependem da escala; as aulas indicam padronização para regressão logística, SVM e PCA (Aula: Supervisionados 6) |
| Random Forest e Gradient Boosting | Nenhum | Árvores dividem por limiar e não dependem da distância entre valores (Aula: Supervisionados 5 diz que árvores não dependem de distância, ao tratar Label Encoding). Que a escala é dispensável para árvores é inferência do time; as aulas não a dizem textualmente |
| Agrupamento (K-Means, Ward, DBSCAN) e PCA | `StandardScaler`, obrigatório | Modelos de distância e PCA são sensíveis à escala (Aula: Não supervisionados 1, 3 e 6) |

O escalonador é ajustado só com o treino: a média e o desvio vêm do treino e são aplicados à validação e ao teste (Aula: Otimização 1 mostra `fit_transform` no treino e `transform` no teste).

## Imputação

| Tipo | Estratégia | Motivo |
|---|---|---|
| Numéricas | Mediana do treino (`SimpleImputer`) | Exigida pelo enunciado e não coberta nas aulas (0 menções a imputação no graphify). A mediana é decisão do time: o IDHM falta em 6 de 5.571 municípios e a mediana não se desloca com valores extremos |
| Categóricas | Valor ausente tratado como categoria própria pelo One-Hot | As colunas categóricas das duas trilhas vêm de `dim_municipio` e dos fatos, e a EDA não mostra ausentes relevantes nelas |

A mediana vem só do treino: um dado novo ausente recebe a mediana do treino, nunca a dos dados novos (testado em `tests/test_transformadores.py`).

## Codificação das categóricas

| Coluna | Codificação | Motivo |
|---|---|---|
| `rede`, `caderno`, `sigla_uf`, `nome_regiao` | One-Hot | Nominais de poucas categorias; as aulas indicam One-Hot para regressão logística e SVM (Aula: Supervisionados 5) |
| `id_municipio`, `id_escola`, `id_aluno` | Nenhuma: não entram como atributo | Identificadores de alta cardinalidade; codificá-los (inclusive com Target Encoding) reintroduziria o rótulo |

Categorias inéditas nos dados novos viram uma linha de zeros, sem erro.

## Seleção de atributos

Não é usada. Cada trilha tem menos de 20 atributos; a redundância entre o IDHM e seus componentes (correlação de 0,71 a 0,95, ver `notebooks/02_eda.ipynb`) é tratada pelo modelo e pela interpretação. O requisito FR4.5 só a exige "se usada".

## Ferramentas exigidas pelo enunciado e não cobertas nas aulas

`SimpleImputer`, `ColumnTransformer` e `Pipeline`. Além delas, o `OneHotEncoder` implementa uma técnica ensinada (One-Hot), mas a classe não aparece no código das aulas; o `StandardScaler` aparece.
