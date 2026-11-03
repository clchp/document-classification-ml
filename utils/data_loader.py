"""
data_loader.py
Módulo para carga, verificación y partición de datos (monolingüe y bilingüe):
- Diagnóstico inicial de calidad de datos, nulos y balance de clases.
- Partición Train/Test estratificada (stratify=y o estratificación multivariable categoria+idioma)
  antes de cualquier vectorización para evitar fuga de información (data leakage).
"""

import os
from typing import Tuple, Optional
import pandas as pd


def cargar_dataset(
    ruta_csv: str = "dataset_bilingue.csv",
    idioma: Optional[str] = None
) -> pd.DataFrame:
    """
    Carga el dataset documental.
    Permite filtrar por idioma ('es', 'en' o None para ambos).
    """
    if not os.path.exists(ruta_csv):
        # Fallback a dataset_preprocesado.csv si existe
        if os.path.exists("dataset_preprocesado.csv"):
            ruta_csv = "dataset_preprocesado.csv"
        else:
            raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta_csv}")

    df = pd.read_csv(ruta_csv)
    if idioma is not None and "idioma" in df.columns:
        df = df[df["idioma"] == idioma].copy().reset_index(drop=True)
        print(f"[DataLoader] Filtrado por idioma '{idioma}': {len(df)} registros.")
    else:
        print(f"[DataLoader] Dataset total cargado: {len(df)} registros.")

    print(f"[DataLoader] Distribución de clases:\n{df['categoria'].value_counts()}")
    if "idioma" in df.columns:
        print(f"[DataLoader] Distribución por idioma:\n{df['idioma'].value_counts()}")
    return df


def cargar_dataset_preprocesado(ruta_csv: str = "dataset_preprocesado.csv") -> pd.DataFrame:
    """Función de compatibilidad para cargar el dataset."""
    return cargar_dataset(ruta_csv=ruta_csv)


def dividir_train_test_estratificado(
    df: pd.DataFrame,
    columna_x: str = "texto_limpio",
    columna_y: str = "categoria",
    test_size: float = 0.20,
    random_state: int = 42,
    estratificar_bilingue: bool = True
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Partición estratificada Train / Test (Hold-Out 80% / 20%).
    
    Regla fundamental contra Fuga de Datos (Data Leakage):
    La partición se ejecuta de forma rigurosa ANTES de ajustar el vectorizador TF-IDF
    o cualquier técnica de reducción dimensional. La muestra de test se reserva únicamente
    para la evaluación final de generalización.
    
    Si 'estratificar_bilingue' es True y existe la columna 'idioma', estratifica
    conjuntamente por clase e idioma para garantizar representatividad perfecta.
    """
    from sklearn.model_selection import train_test_split

    X = df[columna_x]
    y = df[columna_y]

    if estratificar_bilingue and "idioma" in df.columns:
        estrato = df[columna_y].astype(str) + "_" + df["idioma"].astype(str)
    else:
        estrato = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=estrato
    )

    print(f"[DataLoader] Train size: {len(X_train)} ({int((1-test_size)*100)}%) | Test size: {len(X_test)} ({int(test_size*100)}%)")
    print(f"[DataLoader] Distribución de clases en test:\n{y_test.value_counts()}")

    return X_train, X_test, y_train, y_test
