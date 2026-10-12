# Clasificación Automática de Documentos Administrativos Mediante Aprendizaje Automático Supervisado

Proyecto del curso **Inteligencia Artificial (1INF24)** – PUCP · Grupo 7 · Horario H0781

**Integrantes:** Chauca Polanco, C.; Mayta Palacios, D.; Ludeña Talavera, A.; Cáceres Endo, D.

## 1. Descripción

Pipeline de aprendizaje automático supervisado para clasificar documentos administrativos escaneados en cuatro categorías: **Memo, Letter, Form y Report**. El flujo completo es:

`Imagen escaneada → OCR (Tesseract) → Limpieza y normalización → TF-IDF → (TruncatedSVD) → Clasificador → Clase predicha`

## 2. Dataset

- **Corpus:** [Tobacco-3482](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg) (3482 imágenes de documentos en 10 clases), descargado con `kagglehub`.
- **Subconjunto usado:** 300 documentos balanceados, **75 por clase** (Memo, Letter, Form, Report), muestreados aleatoriamente con semilla 42.

## 3. Preprocesamiento

**Extracción de texto (OCR):** `pytesseract` con Tesseract (`lang='eng'`) sobre cada imagen. El texto se guarda en la columna `texto_crudo`. Ninguna imagen devolvió texto vacío.

**Limpieza y normalización** (columna `texto_limpio`), con la función `limpiar_texto`:
1. Conversión a minúsculas.
2. Eliminación de saltos de línea y tabulaciones.
3. Eliminación de números y símbolos (se conservan solo letras).
4. Tokenización por espacios.
5. Eliminación de stopwords en inglés (NLTK) y de tokens de 2 letras o menos.
6. Lematización con `WordNetLemmatizer`.

**Longitud de los documentos limpios (en palabras, media por clase):** Form 91.6 · Letter 112.0 · Memo 117.0 · Report 174.6.

## 4. Representación TF-IDF y partición de datos

- **Partición hold-out:** 80 % entrenamiento (240 documentos) y 20 % prueba (60 documentos, 15 por clase), estratificada y con `random_state=42`. La partición se hace **antes** de vectorizar para evitar fuga de información.
- **TF-IDF:** `TfidfVectorizer(max_features=1000)`, ajustado (`fit`) solo con entrenamiento y aplicado (`transform`) al conjunto de prueba.
- **Matrices resultantes:** entrenamiento `(240, 1000)` y prueba `(60, 1000)`.

## 5. Reducción de dimensionalidad (TruncatedSVD)

Se proyectó la matriz TF-IDF a **50 componentes latentes** (LSA) con `TruncatedSVD(n_components=50, random_state=42)`, ajustado solo con entrenamiento.

- Varianza explicada acumulada: **43.48 %**.
- Se reduce la complejidad del modelo a costa de descartar más de la mitad de la varianza.
- Esta representación se usó con Random Forest. Naive Bayes usa la matriz TF-IDF original, porque `MultinomialNB` no admite valores negativos como los que genera SVD.

## 6. Modelos y evaluación

### 6.1 Validación cruzada estratificada (5 folds) sobre los 300 documentos

Cada modelo se evalúa en un `Pipeline` (TF-IDF + clasificador), de modo que el TF-IDF se reajusta en cada fold.

| Modelo | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
|---|---|---|---|---|
| Regresión logística | 0.73 ± 0.03 | 0.75 ± 0.04 | 0.73 ± 0.03 | 0.73 ± 0.03 |
| Naive Bayes | 0.72 ± 0.03 | 0.73 ± 0.03 | 0.72 ± 0.03 | 0.72 ± 0.03 |
| Árbol de decisión | 0.69 ± 0.05 | 0.72 ± 0.04 | 0.69 ± 0.05 | 0.69 ± 0.05 |
| KNN (k=5, coseno) | 0.60 ± 0.04 | 0.61 ± 0.04 | 0.60 ± 0.04 | 0.60 ± 0.04 |

### 6.2 Evaluación en el conjunto de prueba (hold-out, 60 documentos)

| Modelo | Representación | Accuracy | Macro-F1 |
|---|---|---|---|
| Naive Bayes | TF-IDF (1000) | 0.83 | 0.83 |
| Random Forest | TF-IDF + SVD (50) | 0.70 | 0.70 |

**Naive Bayes** — F1 por clase: Form 0.83, Letter 0.87, Memo 0.86, Report 0.77. Memo tiene recall 1.00 pero precisión 0.75; Report tiene precisión 0.91 pero recall 0.67.

**Random Forest (SVD)** — F1 por clase: Form 0.76, Letter 0.67, Memo 0.67, Report 0.68. Report recibe más predicciones erróneas (10 documentos de otras clases), por lo que su precisión es baja (0.57) aunque su recall es alto (0.87). Letter y Memo tienen precisión 0.89 pero recall 0.53.

> Las matrices de confusión de ambos modelos están en el notebook.

## 7. Análisis y conclusiones

- Con validación cruzada, **regresión logística (0.73) y Naive Bayes (0.72) rinden de forma equivalente**. El árbol de decisión (0.69) queda ligeramente por debajo y KNN (0.60) tiene el peor desempeño.
- En el hold-out, Naive Bayes alcanzó 0.83, más alto que su promedio en validación cruzada (0.72). El resultado **depende de la partición** y no debe interpretarse como el rendimiento esperado.
- Según la validación cruzada, **no se alcanza el Macro-F1 ≥ 0.82 planteado en la hipótesis**. El 0.83 del hold-out es un resultado favorable pero no estable.
- No hay evidencia suficiente para declarar un único mejor modelo entre regresión logística y Naive Bayes. Se propone **regresión logística como modelo base** por tener la media más alta en las cuatro métricas, aunque la diferencia no es significativa.
- Posibles causas del límite de rendimiento: muestra pequeña (300 documentos), ruido del OCR y pérdida de varianza al aplicar SVD.

## 8. Limitaciones y trabajo futuro

- Muestra reducida (300 documentos). Usar más documentos de Tobacco-3482 o de RVL-CDIP.
- OCR solo en inglés (`lang='eng'`) y sin preprocesamiento de la imagen. Probar binarización o corrección de inclinación.
- Hiperparámetros sin optimizar (se usaron valores por defecto). Aplicar `GridSearchCV`.
- Probar n-gramas (1,2) y más términos en el TF-IDF.

## 9. Estructura del repositorio

```
├── TA_GRUPO07_H0781.ipynb      # Notebook con todo el flujo (Google Colab)
├── dataset_preprocesado.csv    # Archivo, categoría, texto crudo y texto limpio
└── README.md
```

## 10. Cómo ejecutar

1. Abrir el notebook en Google Colab.
2. Ejecutar las celdas en orden. El dataset se descarga con `kagglehub` (`patrickaudriaz/tobacco3482jpg`) y se instalan `tesseract-ocr` y `pytesseract`.
3. Al final se exporta `dataset_preprocesado.csv`.

**Requisitos:** `kagglehub`, `pytesseract`, `nltk`, `scikit-learn`, `pandas`, `seaborn`, `matplotlib`.
