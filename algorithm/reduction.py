"""
reduction.py
Módulo de Selección y Reducción de Características:
- Selección de atributos: Filtros (Chi-cuadrado, Mutual Info), Empaquetados (Random Forest).
- Transformación de atributos: Proyección lineal (TruncatedSVD / PCA para matrices dispersas).
- Varianza explicada acumulada y análisis del codo (scree plot).
- Prevención de fuga de información (data leakage).
"""

from typing import List, Tuple, Dict, Any
import numpy as np


def apply_chi2_selection(X_train_tfidf, y_train, feature_names: List[str], k: int = 20) -> List[Tuple[str, float]]:
    """
    Método de Filtro: Prueba Chi-Cuadrado (χ²) para variables categóricas o frecuencias no negativas.
    Evalúa la dependencia estadística entre cada término TF-IDF y la clase objetivo.
    """
    from sklearn.feature_selection import chi2, SelectKBest

    chi2_selector = SelectKBest(chi2, k=min(k, X_train_tfidf.shape[1]))
    chi2_selector.fit(X_train_tfidf, y_train)

    scores = chi2_selector.scores_
    sorted_indices = np.argsort(scores)[::-1][:k]

    ranking = [(feature_names[i], float(scores[i])) for i in sorted_indices]
    return ranking


def apply_svd_reduction(X_train_tfidf, X_test_tfidf, n_components: int = 50, random_state: int = 42):
    """
    Transformación de Atributos: TruncatedSVD (Latent Semantic Analysis - LSA).
    Equivalente a PCA para matrices de términos dispersas.
    Ajusta (fit) ÚNICAMENTE con los datos de entrenamiento y proyecta (transform) train y test
    para prevenir data leakage.
    """
    from sklearn.decomposition import TruncatedSVD

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_train_reduced = svd.fit_transform(X_train_tfidf)
    X_test_reduced = svd.transform(X_test_tfidf)

    explained_variance = float(svd.explained_variance_ratio_.sum())
    return X_train_reduced, X_test_reduced, svd, explained_variance


def compute_svd_cumulative_variance(X_train_tfidf, max_components: int = 100, random_state: int = 42) -> List[Tuple[int, float]]:
    """
    Calcula la curva de varianza explicada acumulada en función del número de componentes latentes (k).
    Sirve para identificar el 'codo' de la curva y justificar la elección de k.
    """
    from sklearn.decomposition import TruncatedSVD

    n_comp = min(max_components, X_train_tfidf.shape[1] - 1, X_train_tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    svd.fit(X_train_tfidf)

    cum_var = np.cumsum(svd.explained_variance_ratio_)
    return [(k + 1, float(v)) for k, v in enumerate(cum_var)]


def get_random_forest_importance(rf_model, feature_names: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
    """
    Método Empaquetado (Embedded): Importancia de variables extraída de árboles de decisión ensamble.
    Integra la selección de atributos dentro del propio proceso de entrenamiento.
    """
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], float(importances[i])) for i in sorted_idx]
