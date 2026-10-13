import pytest

from src.evaluation.metricas import calcular_metricas

# Exemplo clássico com valores conferidos à mão.
Y = [0, 0, 1, 1]
P = [0.1, 0.4, 0.35, 0.8]


def test_auc_roc_do_exemplo_classico_e_0_75() -> None:
    # Dos 4 pares (positivo, negativo), 3 estão ordenados certo.
    assert calcular_metricas(Y, P)["auc_roc"] == pytest.approx(0.75)


def test_auc_pr_e_a_precisao_media_do_exemplo() -> None:
    # Positivos nas posições 1 (precisão 1/1) e 3 (precisão 2/3): (1 + 2/3) / 2.
    assert calcular_metricas(Y, P)["auc_pr"] == pytest.approx(5 / 6)


def test_matriz_de_confusao_e_metricas_no_limiar_de_meio() -> None:
    metricas = calcular_metricas(Y, P)  # previsões no limiar 0,5: 0, 0, 0, 1
    assert metricas["matriz_confusao"] == {"verdadeiro_negativo": 2, "falso_positivo": 0, "falso_negativo": 1, "verdadeiro_positivo": 1}
    assert metricas["precisao"] == pytest.approx(1.0)
    assert metricas["recall"] == pytest.approx(0.5)
    assert metricas["f1"] == pytest.approx(2 / 3)


def test_limiar_mais_baixo_troca_precisao_por_recall() -> None:
    metricas = calcular_metricas(Y, P, limiar=0.3)  # previsões: 0, 1, 1, 1
    assert metricas["recall"] == pytest.approx(1.0)
    assert metricas["precisao"] == pytest.approx(2 / 3)


def test_sem_nenhuma_previsao_positiva_a_precisao_e_zero_e_nao_ha_erro() -> None:
    metricas = calcular_metricas(Y, [0.1, 0.1, 0.1, 0.2], limiar=0.9)
    assert metricas["precisao"] == 0.0 and metricas["recall"] == 0.0 and metricas["f1"] == 0.0


def test_auc_roc_nao_depende_do_limiar() -> None:
    assert calcular_metricas(Y, P, limiar=0.1)["auc_roc"] == calcular_metricas(Y, P, limiar=0.9)["auc_roc"]


def test_alvo_com_uma_so_classe_e_recusado_com_mensagem_clara() -> None:
    with pytest.raises(ValueError, match="duas classes"):
        calcular_metricas([1, 1, 1], [0.2, 0.5, 0.9])


def test_tamanhos_diferentes_sao_recusados() -> None:
    with pytest.raises(ValueError, match="mesmo tamanho"):
        calcular_metricas([0, 1], [0.2, 0.5, 0.9])
