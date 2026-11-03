"""
evaluation.py
Módulo de Métricas y Evaluación Experimental Bilingüe:
- Métricas de clasificación multiclase (Matriz de confusión, Accuracy, Precision, Recall, F1-Score, Balanced Accuracy).
- Validación cruzada estratificada K-fold.
- Evaluación monolingüe (Inglés, Español) y bilingüe consolidada.
- Transferencia entre lenguas (Cross-lingual Zero-Shot Transfer: Train EN -> Test ES / Train ES -> Test EN).
- Pipeline reproducible y prevención de Fuga de Datos (Data Leakage).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


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


def run_stratified_cv_evaluation(
    model_dict: Dict[str, Any],
    X_text,
    y,
    n_splits: int = 5,
    random_state: int = 42,
    max_features: int = 1500
) -> Dict[str, Dict[str, Any]]:
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
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))),
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


def run_cross_lingual_experiment(
    clf_builder,
    df_bilingue: pd.DataFrame,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Ejecuta un experimento factorial de 4 escenarios lingüísticos:
    1. Monolingüe Inglés: Train EN (80%) -> Test EN (20%)
    2. Monolingüe Español: Train ES (80%) -> Test ES (20%)
    3. Bilingüe Conjunto: Train Bilingüe (80%) -> Test Bilingüe (20%)
    4. Transferencia Cruzada (Cross-lingual Zero-Shot): Train EN (100%) -> Test ES (100%)
    """
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    df_en = df_bilingue[df_bilingue["idioma"] == "en"].reset_index(drop=True)
    df_es = df_bilingue[df_bilingue["idioma"] == "es"].reset_index(drop=True)

    # 1. Monolingüe Inglés
    X_tr_en, X_te_en, y_tr_en, y_te_en = train_test_split(
        df_en["texto_limpio"], df_en["categoria"], test_size=0.20, random_state=random_state, stratify=df_en["categoria"]
    )
    # 2. Monolingüe Español
    X_tr_es, X_te_es, y_tr_es, y_te_es = train_test_split(
        df_es["texto_limpio"], df_es["categoria"], test_size=0.20, random_state=random_state, stratify=df_es["categoria"]
    )
    # 3. Bilingüe Conjunto
    estrato_bil = df_bilingue["categoria"] + "_" + df_bilingue["idioma"]
    X_tr_bi, X_te_bi, y_tr_bi, y_te_bi = train_test_split(
        df_bilingue["texto_limpio"], df_bilingue["categoria"], test_size=0.20, random_state=random_state, stratify=estrato_bil
    )

    scenarios = [
        ("Monolingüe Inglés (300 docs)", X_tr_en, y_tr_en, X_te_en, y_te_en),
        ("Monolingüe Español (300 docs)", X_tr_es, y_tr_es, X_te_es, y_te_es),
        ("Bilingüe Consolidado (600 docs)", X_tr_bi, y_tr_bi, X_te_bi, y_te_bi),
        ("Transferencia Cero Disparo (Train EN -> Test ES)", df_en["texto_limpio"], df_en["categoria"], df_es["texto_limpio"], df_es["categoria"])
    ]

    records = []
    models = clf_builder(random_state=random_state)

    for esc_nombre, X_train, y_train, X_test, y_test in scenarios:
        for model_name, model in models.items():
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=1500, ngram_range=(1, 2))),
                ("clf", model)
            ])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro", zero_division=0)
            records.append({
                "Escenario": esc_nombre,
                "Modelo": model_name,
                "Accuracy": round(acc, 4),
                "Macro-F1": round(f1, 4)
            })

    return pd.DataFrame(records)
