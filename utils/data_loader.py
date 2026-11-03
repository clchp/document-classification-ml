"""
data_loader.py
Módulo para carga, verificación y partición de datos.
Alineado con:
- S5: Diagnóstico inicial de calidad de datos, nulos y desbalance de clases (slide 4, 12, 17).
- S5/S7: Partición Train/Test estratificada (stratify=y) ANTES de transformaciones (S5 slide 15, S7 slide 4-5).
"""

import os
from typing import Tuple
import pandas as pd


def cargar_dataset_preprocesado(ruta_csv: str = "dataset_preprocesado.csv") -> pd.DataFrame:
    """
    Carga el archivo CSV con las 300 muestras balanceadas.
    Verifica la existencia del archivo y reporta el conteo de nulos y distribución por clase.
    """
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta_csv}")

    df = pd.read_csv(ruta_csv)
    print(f"[DataLoader] Dataset cargado: {len(df)} registros.")
    print(f"[DataLoader] Valores nulos:\n{df.isnull().sum()}")
    print(f"[DataLoader] Distribución de clases (S5 slide 12):\n{df['categoria'].value_counts()}")
    return df


def dividir_train_test_estratificado(
    df: pd.DataFrame,
    columna_x: str = "texto_limpio",
    columna_y: str = "categoria",
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Partición estratificada Train / Test (Hold-Out 80% / 20%).
    
    Regla fundamental para evitar Fuga de Datos (Data Leakage - S5 slide 15, S7 slide 4):
    La partición se ejecuta de forma rigurosa ANTES de ajustar el vectorizador TF-IDF
    o cualquier técnica de reducción dimensional. La muestra de test se reserva únicamente
    para la evaluación final de generalización.
    """
    from sklearn.model_selection import train_test_split

    X = df[columna_x]
    y = df[columna_y]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[DataLoader] Train size: {len(X_train)} (80%) | Test size: {len(X_test)} (20%)")
    print(f"[DataLoader] Distribución en test:\n{y_test.value_counts()}")

    return X_train, X_test, y_train, y_test

