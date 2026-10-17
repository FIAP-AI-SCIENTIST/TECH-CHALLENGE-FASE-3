"""Ranking de risco dos municípios a partir dos modelos de T2 e T1.

`python -m src.modeling.ranking_municipios` usa a configuração dos modelos gravados
pelo treino (algoritmo e hiperparâmetros) e prevê cada município fora da dobra. Grava:

- `reports/ranking_risco_meta_municipios.csv`: a probabilidade de não atingir a meta
  de 2024 (T2), do maior para o menor risco, com a posição geral e na UF, o quintil e
  as taxas de não alfabetizados real e esperada pelo perfil (T1);
- `reports/ranking_abaixo_do_perfil_municipios.csv`: os municípios com mais não
  alfabetizados do que o perfil (UF, região, IDHM, rede e caderno) prevê;
- `reports/ranking_municipios.json`: o resumo e a AUC-ROC das previsões fora da dobra;
- `images/ranking_alto_risco_por_uf.png`: a proporção de municípios no quintil de maior risco por UF.

Leva cerca de 4 minutos, quase todos da T1.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from src.config import DIRETORIO_IMAGENS, DIRETORIO_RELATORIOS
from src.modeling.ranking import montar_ranking, probabilidades_fora_da_dobra, taxa_por_municipio
from src.modeling.treino import carregar_modelo
from src.modeling.treino_aluno import MAX_LINHAS_TREINO, carregar_tabela_t1
from src.modeling.treino_municipio import carregar_tabela_t2
from src.preprocessing.loader import carregar_tabela
from src.visualization.graficos_interpretacao import barras_importancia

# A taxa municipal de poucos alunos é ruidosa (a mediana é de 122 alunos em 2024).
# Municípios com menos alunos que isto ficam fora do ranking abaixo do perfil.
MINIMO_ALUNOS: int = 50
CASAS_DECIMAIS: int = 4


def abaixo_do_perfil(perfil: pd.DataFrame, minimo_alunos: int = MINIMO_ALUNOS) -> pd.DataFrame:
    """Municípios com pelo menos `minimo_alunos`, do maior para o menor excesso de não alfabetizados.

    O excesso é a taxa real menos a esperada pelo modelo de T1; positivo quer dizer pior
    do que o perfil do município prevê.
    """
    tabela = perfil[perfil["n_alunos"] >= minimo_alunos].copy()
    tabela["excesso_nao_alfabetizados"] = (
        tabela["taxa_nao_alfabetizados_real"] - tabela["taxa_nao_alfabetizados_esperada"]
    )
    tabela = tabela.sort_values(["excesso_nao_alfabetizados", "id_municipio"], ascending=[False, True]).reset_index(drop=True)
    tabela.insert(0, "posicao", range(1, len(tabela) + 1))
    return tabela


def proporcao_alto_risco_por_uf(ranking: pd.DataFrame) -> pd.DataFrame:
    """Municípios e proporção no quintil 5 (maior risco) em cada UF, da maior para a menor."""
    return (
        ranking.assign(no_quintil_5=ranking["quintil_risco"] == 5)
        .groupby("sigla_uf")
        .agg(municipios=("no_quintil_5", "size"), proporcao_quintil_5=("no_quintil_5", "mean"))
        .reset_index()
        .sort_values("proporcao_quintil_5", ascending=False)
        .reset_index(drop=True)
    )


def _perfil_t1() -> tuple[pd.DataFrame, float]:
    tabela = carregar_tabela_t1()
    probabilidades = probabilidades_fora_da_dobra(
        clone(carregar_modelo("t1")), tabela, max_linhas_treino=MAX_LINHAS_TREINO
    )
    perfil = pd.DataFrame(
        {
            "n_alunos": tabela.grupos.value_counts(),
            "taxa_nao_alfabetizados_real": taxa_por_municipio(tabela.alvo, tabela.grupos),
            "taxa_nao_alfabetizados_esperada": taxa_por_municipio(probabilidades, tabela.grupos),
        }
    ).rename_axis("id_municipio")
    return perfil.reset_index(), float(roc_auc_score(tabela.alvo, probabilidades))


def executar() -> dict[str, object]:
    """Monta os dois rankings, grava os arquivos e devolve o resumo."""
    municipios = carregar_tabela("dim_municipio")[["id_municipio", "nome", "sigla_uf", "nome_regiao"]]

    tabela_t2 = carregar_tabela_t2()
    probabilidades_t2 = probabilidades_fora_da_dobra(clone(carregar_modelo("t2")), tabela_t2)
    risco = pd.Series(probabilidades_t2.to_numpy(), index=tabela_t2.grupos.to_numpy())
    ranking = montar_ranking(risco, municipios, nome_risco="prob_nao_atingir_meta")
    observados = pd.DataFrame(
        {
            "id_municipio": tabela_t2.grupos,
            "nao_atingiu_meta_real": tabela_t2.alvo,
            "taxa_alfabetizacao_2023": tabela_t2.atributos["taxa_alfabetizacao_2023"],
            "meta_indicador_2024": tabela_t2.atributos["meta_indicador_2024"],
        }
    )

    perfil, auc_t1 = _perfil_t1()
    perfil = perfil.merge(municipios, on="id_municipio", how="left", validate="one_to_one")
    ranking = ranking.merge(observados, on="id_municipio", how="left", validate="one_to_one").merge(
        perfil[["id_municipio", "n_alunos", "taxa_nao_alfabetizados_real", "taxa_nao_alfabetizados_esperada"]],
        on="id_municipio",
        how="left",
        validate="one_to_one",
    )
    abaixo = abaixo_do_perfil(perfil)
    por_uf = proporcao_alto_risco_por_uf(ranking)

    DIRETORIO_RELATORIOS.mkdir(parents=True, exist_ok=True)
    ranking.round(CASAS_DECIMAIS).to_csv(DIRETORIO_RELATORIOS / "ranking_risco_meta_municipios.csv", index=False)
    abaixo.round(CASAS_DECIMAIS).to_csv(DIRETORIO_RELATORIOS / "ranking_abaixo_do_perfil_municipios.csv", index=False)
    barras_importancia(
        por_uf,
        "sigla_uf",
        "proporcao_quintil_5",
        "Municípios no quintil de maior risco de não atingir a meta, por UF",
        "Proporção dos municípios da UF no quintil 5",
        "ranking_alto_risco_por_uf.png",
        DIRETORIO_IMAGENS,
        maximo=len(por_uf),
    )

    resumo = {
        "municipios_no_ranking_de_meta": int(len(ranking)),
        "auc_roc_fora_da_dobra_t2": round(float(roc_auc_score(tabela_t2.alvo, probabilidades_t2)), CASAS_DECIMAIS),
        "auc_roc_fora_da_dobra_t1": round(auc_t1, CASAS_DECIMAIS),
        "municipios_no_ranking_abaixo_do_perfil": int(len(abaixo)),
        "minimo_alunos_abaixo_do_perfil": MINIMO_ALUNOS,
        "dez_maiores_riscos_de_meta": ranking.head(10)[["nome", "sigla_uf", "prob_nao_atingir_meta"]]
        .round(CASAS_DECIMAIS)
        .to_dict(orient="records"),
        "dez_mais_abaixo_do_perfil": abaixo.head(10)[["nome", "sigla_uf", "n_alunos", "excesso_nao_alfabetizados"]]
        .round(CASAS_DECIMAIS)
        .to_dict(orient="records"),
        "proporcao_quintil_5_por_uf": por_uf.round(CASAS_DECIMAIS).to_dict(orient="records"),
    }
    caminho: Path = DIRETORIO_RELATORIOS / "ranking_municipios.json"
    caminho.write_text(json.dumps(resumo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({chave: resumo[chave] for chave in list(resumo)[:5]}, indent=2, ensure_ascii=False))
    return resumo


if __name__ == "__main__":
    executar()
