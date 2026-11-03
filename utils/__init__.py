"""
Módulo de utilidades: preprocesamiento, carga de datos y visualización (S5 y S6).
"""

from .preprocessing import limpiar_texto, analizar_longitud_documentos
from .data_loader import cargar_dataset_preprocesado, dividir_train_test_estratificado
from .visualizer import generar_todos_los_graficos

__all__ = [
    "limpiar_texto",
    "analizar_longitud_documentos",
    "cargar_dataset_preprocesado",
    "dividir_train_test_estratificado",
    "generar_todos_los_graficos",
]

