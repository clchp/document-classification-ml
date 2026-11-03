"""
Módulo de algoritmos y modelos de clasificación (S10 Algoritmos de Clasificación y S7 Regresión/Regularización).
"""

from .classifiers import (
    build_classifier_models,
    tune_knn,
    tune_decision_tree_depth,
    train_logistic_regression,
    train_naive_bayes,
    train_random_forest
)
from .reduction import (
    apply_chi2_selection,
    apply_svd_reduction,
    compute_svd_cumulative_variance,
    get_random_forest_importance
)
from .evaluation import (
    compute_classification_metrics,
    compute_per_class_metrics,
    run_stratified_cv_evaluation
)

__all__ = [
    "build_classifier_models",
    "tune_knn",
    "tune_decision_tree_depth",
    "train_logistic_regression",
    "train_naive_bayes",
    "train_random_forest",
    "apply_chi2_selection",
    "apply_svd_reduction",
    "compute_svd_cumulative_variance",
    "get_random_forest_importance",
    "compute_classification_metrics",
    "compute_per_class_metrics",
    "run_stratified_cv_evaluation",
]

