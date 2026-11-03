"""
evaluation.py
Módulo de Métricas y Evaluación Experimental:
- Métricas de clasificación multiclase (Matriz de confusión, Accuracy, Precision, Recall, F1-Score, Balanced Accuracy).
- Validación cruzada estratificada K-fold.
- Pipeline reproducible y prevención de Fuga de Datos (Data Leakage).
"""

from typing import Dict, Any, List, Optional
import numpy as np


def compute_classification_metrics(y_true, y_pred, labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Calcula las métricas estándar de clasificación multiclase:
    - Matriz de confusión (m x m)
    - Accuracy global = (TP + TN) / N
    - Macro-Precision: promedio no ponderado de precisión por clase
    - Macro-Recall: promedio no ponderado de exhaustividad por clase
    - Macro-F1: media armónica de Macro-Precision y Macro-Recall
    - Balanced Accuracy: promedio de recall individual por clase
    """
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        balanced_accuracy_score,
        classification_report
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    return {
        "confusion_matrix": cm,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "balanced_accuracy": bal_acc,
        "report_dict": report_dict
    }


def compute_per_class_metrics(report_dict: Dict[str, Any], classes: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Extrae la precisión, recall y F1-score individuales para cada clase administrativa
    (Form, Letter, Memo, Report) a partir del reporte de clasificación.
    """
    per_class = {}
    for cls in classes:
        if cls in report_dict:
            per_class[cls] = {
                "precision": report_dict[cls]["precision"],
                "recall": report_dict[cls]["recall"],
                "f1-score": report_dict[cls]["f1-score"],
                "support": report_dict[cls]["support"],
            }
    return per_class


def run_stratified_cv_evaluation(model_dict: Dict[str, Any], X_text, y, n_splits: int = 5, random_state: int = 42) -> Dict[str, Dict[str, str]]:
    """
    Ejecuta Validación Cruzada Estratificada (5 Folds) usando Pipelines (TF-IDF + Clasificador).
    Garantiza que el vocabulario y pesos TF-IDF se ajustan ÚNICAMENTE dentro de cada fold
    de entrenamiento, eliminando completamente el data leakage.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import StratifiedKFold, cross_validate

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metricas = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    summary = {}

    for name, clf in model_dict.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=1000)),
            ("clf", clf)
        ])
        cv_results = cross_validate(pipe, X_text, y, cv=cv, scoring=metricas)

        model_summary = {}
        for m in metricas:
            mean_score = float(cv_results["test_" + m].mean())
            std_score = float(cv_results["test_" + m].std())
            model_summary[m] = f"{mean_score:.2f} ± {std_score:.2f}"
            model_summary[m + "_mean"] = mean_score
            model_summary[m + "_std"] = std_score

        summary[name] = model_summary

    return summary
