# Avaliação final: matriz de confusão, curva ROC e probabilidades

Fragmento do README, a consolidar na unidade de documentação. Gerado por `python -m src.evaluation.avaliacao_final` (cerca de 1 minuto) a partir dos modelos em `models/`.

## Como é feito

Os scripts de treino gravam o modelo final de cada trilha em `models/modelo_t1.joblib` e `models/modelo_t2.joblib` (150 KB e 7 KB, versionados para que ninguém precise retreinar). A avaliação carrega o modelo, reconstrói o conjunto de teste pela mesma semente da separação e, **antes de desenhar, confere que a AUC-ROC, a AUC-PR e a matriz de confusão batem com `reports/treino_<trilha>.json`**. Se não baterem, o modelo gravado não é o do relatório, e o script falha pedindo para rodar o treino de novo. Nada é reajustado e nenhum limiar é escolhido aqui: o teste já foi aberto pelo treino.

Rodar os dois treinos de novo produziu relatórios idênticos aos versionados, o que confirma que o pipeline é reproduzível com a semente 42.

## T2 (município): não atingiu a meta

![Matriz de confusão de T2](../images/avaliacao_matriz_confusao_t2.png)
![Curva ROC de T2](../images/avaliacao_curva_roc_t2.png)
![Probabilidades de T2](../images/avaliacao_probabilidades_t2.png)

- O modelo acerta 66,5% dos municípios que não atingiram a meta (recall) e 74,8% dos que atingiram.
- As probabilidades das duas classes se separam: a maioria dos que atingiram fica abaixo de 0,4, e muitos dos que não atingiram ficam acima de 0,8. A sobreposição entre 0,4 e 0,6 é onde estão os erros.
- A AUC-ROC de 0,796 vem em boa parte do efeito de estado; a ressalva está em [readme-treino-t2.md](readme-treino-t2.md).

## T1 (aluno): não alfabetizado

![Matriz de confusão de T1](../images/avaliacao_matriz_confusao_t1.png)
![Curva ROC de T1](../images/avaliacao_curva_roc_t1.png)
![Probabilidades de T1](../images/avaliacao_probabilidades_t1.png)

- **O recall é de só 19,5%:** o modelo deixa de apontar 80,5% dos alunos não alfabetizados no limiar de 0,5.
- A distribuição das probabilidades explica por quê: quase todas ficam entre 0,2 e 0,6, concentradas abaixo de 0,5, e as duas classes se sobrepõem. Como os atributos são do município, da rede e do caderno, alunos do mesmo município recebem a mesma probabilidade; o modelo separa territórios mais e menos vulneráveis, não alunos.
- A curva ROC (AUC-ROC de 0,632) mostra que um limiar mais baixo aumentaria o recall à custa de mais falsos positivos. Escolher esse limiar depende do custo de cada erro para a política pública, e as aulas não ensinam como fazê-lo, então o projeto mantém 0,5.
