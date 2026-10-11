"""Pré-processador das trilhas supervisionadas.

Devolve um `ColumnTransformer` NÃO ajustado. Quem o usa o encaixa num `Pipeline`
junto do estimador e o ajusta só com o treino, de modo que mediana, média, desvio
e categorias vêm do treino e nunca dos dados de validação ou de teste.
`ColumnTransformer` e `Pipeline` são exigidos pelo enunciado e não são cobertos
nas aulas.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ColunasTrilha


def construir_preprocessador(trilha: ColunasTrilha, escalonar: bool) -> ColumnTransformer:
    """Imputa pela mediana, escalona se `escalonar` e codifica as categóricas em One-Hot.

    Colunas fora da trilha são descartadas. Categorias inéditas nos dados novos viram
    uma linha de zeros em vez de erro. A saída é um DataFrame com nomes legíveis
    (`numericos__idhm`, `categoricos__sigla_uf_SP`), que a interpretação com SHAP usa.
    """
    passos_numericos: list[tuple[str, object]] = [("imputar", SimpleImputer(strategy="median"))]
    if escalonar:
        passos_numericos.append(("escalar", StandardScaler()))
    preprocessador = ColumnTransformer(
        [
            ("numericos", Pipeline(passos_numericos), list(trilha.numericas)),
            ("categoricos", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(trilha.categoricas)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessador.set_output(transform="pandas")
