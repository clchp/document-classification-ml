"""
preprocessing.py
Módulo de Preprocesamiento y Limpieza Textual (NLP):
- Limpieza de datos (vacíos, ruido de caracteres, minúsculas, stopwords).
- Lematización léxica.
- Análisis Exploratorio de Datos (EDA): Estadísticas descriptivas, media, mediana, asimetría.
"""

import re
from typing import Optional, Dict, Any
import pandas as pd


# Lista estándar de stopwords en inglés de respaldo en caso de que nltk no esté descargado
DEFAULT_ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
}


def _obtener_lematizador_y_stopwords():
    """Intenta cargar recursos NLTK; de lo contrario utiliza alternativas ligeras."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            stop_words = set(stopwords.words("english"))

        try:
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize("testing")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            lemmatizer = WordNetLemmatizer()

        return lemmatizer, stop_words
    except Exception:
        class DummyLemmatizer:
            def lemmatize(self, word):
                return word
        return DummyLemmatizer(), DEFAULT_ENGLISH_STOPWORDS


_lemmatizer, _stopwords = _obtener_lematizador_y_stopwords()


def limpiar_texto(texto: Optional[str]) -> str:
    """
    Pipeline de limpieza y normalización lingüística:
    1. Manejo de valores nulos o no string.
    2. Conversión total a minúsculas (lower casing).
    3. Eliminación de saltos de línea (\r, \n) y tabulaciones (\t).
    4. Sustitución de caracteres especiales y dígitos por espacios (solo alfabeto [a-z]).
    5. Tokenización por secuencias de espacios.
    6. Remoción de stopwords en inglés y palabras de longitud <= 2 caracteres.
    7. Lematización con WordNetLemmatizer para consolidar formas canónicas.
    """
    if not isinstance(texto, str):
        return ""

    # 1. Minúsculas
    texto = texto.lower()
    # 2. Saltos de línea y tabs
    texto = re.sub(r"[\r\n\t]+", " ", texto)
    # 3. Solo caracteres alfabéticos
    texto = re.sub(r"[^a-z\s]", " ", texto)
    # 4. Tokenización
    palabras = texto.split()
    # 5. Filtrado de stopwords y longitud mínima + lematización
    palabras_limpias = [
        _lemmatizer.lemmatize(p)
        for p in palabras
        if p not in _stopwords and len(p) > 2
    ]

    return " ".join(palabras_limpias)


def analizar_longitud_documentos(df: pd.DataFrame, columna_texto: str = "texto_limpio", columna_target: str = "categoria") -> pd.DataFrame:
    """
    Análisis exploratorio de longitud de textos por categoría.
    Calcula: conteo, media, desviación estándar, cuartiles (Q1, Mediana, Q3), mínimo y máximo.
    Permite diagnosticar asimetría positiva (media > mediana) y valores extremos (outliers).
    """
    df_temp = df.copy()
    df_temp["n_palabras"] = df_temp[columna_texto].astype(str).str.split().str.len()
    stats = df_temp.groupby(columna_target)["n_palabras"].describe().round(1)
    return stats
