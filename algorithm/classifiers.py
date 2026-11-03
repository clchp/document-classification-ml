"""
classifiers.py
Implementación y encapsulación de algoritmos de clasificación supervisada:
- Regresión Logística (entropía cruzada, regularización L2)
- K-Nearest Neighbors (similitud coseno, análisis de K)
- Árboles de Decisión CART (impureza de Gini, regularización por poda)
- Naive Bayes Multinomial
- Random Forest
"""

from typing import Dict, Any, Iterable, Tuple, List


def build_classifier_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Retorna el diccionario con los modelos de clasificación evaluados en el proyecto.
    
    1. Regresión Logística: Modelo lineal probabilístico con función sigmoide/softmax y
       pérdida de entropía cruzada. Regularización L2 (C=1.0) para prevenir sobreajuste.
    2. Naive Bayes (MultinomialNB): Clasificador probabilístico óptimo para frecuencias
       de términos discretos/TF-IDF.
    3. Árbol de Decisión (CART): Modelo no paramétrico particionado por ganancia de pureza
       (impureza de Gini), con pre-poda (random_state=42).
    4. KNN (k=5, Coseno): Clasificador lazy basado en distancias. Se usa similitud coseno
       para matrices dispersas de texto en alta dimensión.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier

        return {
            "Regresión logística": LogisticRegression(max_iter=1000, C=1.0, penalty="l2", random_state=random_state),
            "Naive Bayes": MultinomialNB(alpha=1.0),
            "Árbol de decisión": DecisionTreeClassifier(random_state=random_state),
            "KNN": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        }
    except ImportError:
        return {}


def train_logistic_regression(X_train, y_train, C: float = 1.0, penalty: str = "l2", random_state: int = 42):
    """
    Ajusta una Regresión Logística multiclase (One-vs-Rest o Multinomial).
    """
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000, C=C, penalty=penalty, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train, y_train, alpha: float = 1.0):
    """
    Ajusta un clasificador Naive Bayes Multinomial.
    Nota: Requiere valores de características no negativas (matriz TF-IDF directa sin SVD).
    """
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """
    Ajusta un ensamble de árboles Random Forest.
    Acepta representaciones reducidas con TruncatedSVD.
    """
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def tune_knn(X_train, y_train, k_values: Iterable[int] = range(1, 16), metric: str = "cosine") -> List[Tuple[int, float]]:
    """
    Evalúa el desempeño de KNN para distintos valores de K.
    K=1 tiende al sobreajuste (alta varianza, sensible al ruido).
    K grande tiende al subajuste (alto sesgo, frontera excesivamente suavizada).
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
        scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
        results.append((k, float(scores.mean())))

    return results


def tune_decision_tree_depth(X_train, y_train, max_depths: Iterable[int] = range(1, 16), criterion: str = "gini") -> List[Tuple[int, float, float]]:
    """
    Evalúa el efecto de la profundidad máxima (max_depth) en Árboles CART.
    Muestra cómo la regularización por pre-poda previene el sobreajuste.
    Retorna lista de tuplas (depth, train_score, val_score).
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for d in max_depths:
        clf = DecisionTreeClassifier(max_depth=d, criterion=criterion, random_state=42)
        clf.fit(X_train, y_train)
        train_acc = float(clf.score(X_train, y_train))
        val_acc = float(cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy").mean())
        results.append((d, train_acc, val_acc))

    return results
