"""
preprocessing.py
Módulo de Preprocesamiento y Limpieza Textual Bilingüe (NLP):
- Limpieza de datos (vacíos, ruido de caracteres tipográficos, minúsculas).
- Manejo adaptativo de caracteres latinos (á, é, í, ó, ú, ü, ñ) y anglosajones.
- Remoción de stopwords en inglés y español.
- Lematización léxica y normalización morfológica.
- Análisis Exploratorio de Datos (EDA): Estadísticas descriptivas, media, mediana, asimetría.
"""

import re
from typing import Optional, Dict, Any, List
import pandas as pd


# Lista estándar de stopwords en inglés
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

# Lista estándar de stopwords en español
DEFAULT_SPANISH_STOPWORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con",
    "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí",
    "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta", "hay",
    "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra",
    "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos", "qué", "unos",
    "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada",
    "muchos", "cual", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros", "mi", "mis",
    "tú", "te", "ti", "tu", "tus", "ellas", "nosotras", "vosotros", "vosotras", "os", "mío", "mía",
    "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas", "nuestro",
    "nuestra", "nuestros", "nuestras", "vuestro", "vuestra", "vuestros", "vuestras", "esos", "esas",
    "estoy", "estás", "está", "estamos", "estáis", "están", "esté", "estés", "estemos", "estéis",
    "estén", "estaré", "estarás", "estará", "estaremos", "estaréis", "estarán", "estaría", "estarías",
    "estaríamos", "estaríais", "estarían", "estaba", "estabas", "estábamos", "estabais", "estaban",
    "estuve", "estuviste", "estuvo", "estuvimos", "estuvisteis", "estuvieron", "hubiera", "hubiese",
    "habiendo", "habido", "han", "ha", "he", "has", "hemos", "habéis", "había", "habías", "habíamos",
    "habíais", "habían", "fue", "fueron", "fui", "fuiste", "fuimos", "fuisteis", "era", "eras",
    "éramos", "erais", "eran", "sea", "seas", "seamos", "seáis", "sean", "ser", "son", "somos",
    "sois", "soy", "eres", "es"
}

STOPWORDS_BILINGUE = DEFAULT_ENGLISH_STOPWORDS.union(DEFAULT_SPANISH_STOPWORDS)


def _obtener_recursos_nlp():
    """Intenta cargar lematizador NLTK; fallback a tokenización ligera."""
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        try:
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize("testing")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            lemmatizer = WordNetLemmatizer()
        return lemmatizer
    except Exception:
        class DummyLemmatizer:
            def lemmatize(self, word):
                return word
        return DummyLemmatizer()


_lemmatizer = _obtener_recursos_nlp()


def limpiar_texto(texto: Optional[str], idioma: str = "auto") -> str:
    """
    Pipeline de limpieza y normalización lingüística bilingüe:
    1. Manejo de valores nulos o no string.
    2. Conversión total a minúsculas (lower casing).
    3. Eliminación de saltos de línea (\r, \n) y tabulaciones (\t).
    4. Sustitución de caracteres especiales y dígitos por espacios.
       - Si es español o auto: preserva [a-záéíóúüñ].
       - Si es inglés puro: preserva [a-z].
    5. Tokenización por secuencias de espacios en blanco.
    6. Remoción de stopwords (español, inglés o bilingüe) y tokens con longitud <= 2 caracteres.
    7. Normalización y lematización morfológica.
    """
    if not isinstance(texto, str):
        return ""

    texto = texto.lower()
    texto = re.sub(r"[\r\n\t]+", " ", texto)

    if idioma == "en":
        texto = re.sub(r"[^a-z\s]", " ", texto)
        stopwords_set = DEFAULT_ENGLISH_STOPWORDS
    elif idioma == "es":
        texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
        stopwords_set = DEFAULT_SPANISH_STOPWORDS
    else:  # auto / bilingüe
        texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
        stopwords_set = STOPWORDS_BILINGUE

    palabras = texto.split()
    palabras_limpias = [
        _lemmatizer.lemmatize(p) if idioma == "en" else p
        for p in palabras
        if p not in stopwords_set and len(p) > 2
    ]

    return " ".join(palabras_limpias)


def analizar_longitud_documentos(
    df: pd.DataFrame,
    columna_texto: str = "texto_limpio",
    columna_target: str = "categoria"
) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas de longitud textual (conteo de palabras):
    - Media, Mediana, Desviación Estándar.
    - Cuartil 25 (Q1), Cuartil 75 (Q3), Rango Intercuartílico (IQR).
    - Asimetría estadística (Skewness).
    """
    longitudes = df[columna_texto].fillna("").apply(lambda x: len(x.split()))
    df_temp = df.copy()
    df_temp["longitud_palabras"] = longitudes

    resumen = df_temp.groupby(columna_target)["longitud_palabras"].agg(
        conteo="count",
        media="mean",
        desv_est="std",
        mediana="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        asimetria="skew"
    ).reset_index()

    return resumen
