"""
probar_clasificador.py
Script interactivo para probar el clasificador bilingüe de documentos administrativos
(Form, Letter, Memo, Report) en tiempo real con cualquier texto en español o inglés,
archivos de texto (.txt) o documentos oficiales en formato PDF (.pdf).

Uso:
    python probar_clasificador.py                               # Modo interactivo en consola (acepta texto o rutas de PDF/TXT)
    python probar_clasificador.py --demo                        # Ejecuta 4 casos de prueba reales
    python probar_clasificador.py --texto "..."                 # Clasifica un texto directo
    python probar_clasificador.py --archivo ruta/documento.pdf  # Clasifica un PDF oficial directamente
"""

import os
import sys
import math
import argparse
from collections import Counter
import pandas as pd
import numpy as np

# Configurar salida estándar a UTF-8 para Windows PowerShell / CMD
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from utils.preprocessing import limpiar_texto


def extraer_texto_de_archivo(ruta_archivo: str, max_paginas: int = 5) -> str:
    """
    Extrae texto crudo desde un archivo PDF o TXT.
    Soporta extracción nativa de PDFs administrativos mediante pdfplumber.
    """
    ruta_limpia = ruta_archivo.strip().strip("'").strip('"')
    if not os.path.exists(ruta_limpia):
        raise FileNotFoundError(f"No se encontró el archivo: '{ruta_limpia}'")

    ext = os.path.splitext(ruta_limpia)[1].lower()

    if ext == ".pdf":
        try:
            import pdfplumber
            texto_total = []
            with pdfplumber.open(ruta_limpia) as pdf:
                num_paginas = len(pdf.pages)
                paginas_a_leer = min(num_paginas, max_paginas)
                print(f"[PDF Extractor] Leyendo '{os.path.basename(ruta_limpia)}' ({paginas_a_leer} de {num_paginas} páginas)...")
                for i in range(paginas_a_leer):
                    txt_pagina = pdf.pages[i].extract_text()
                    if txt_pagina:
                        texto_total.append(txt_pagina)
            resultado = "\n".join(texto_total).strip()
            if not resultado:
                raise ValueError("El archivo PDF no contiene capa de texto digital (es un escaneo de solo imagen). Requiere OCR.")
            return resultado
        except ImportError:
            raise ImportError("La librería 'pdfplumber' no está disponible para leer PDFs.")

    elif ext in (".jpg", ".jpeg", ".png", ".tiff", ".bmp"):
        # Verificamos si existe pytesseract disponible
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(ruta_limpia)
            print(f"[OCR Extractor] Procesando imagen '{os.path.basename(ruta_limpia)}' con OCR...")
            return pytesseract.image_to_string(img, lang="spa+eng").strip()
        except ImportError:
            raise ValueError(
                f"El archivo '{os.path.basename(ruta_limpia)}' es una imagen/foto.\n"
                "Para extraer texto de imágenes se requiere un motor de OCR (como Tesseract / pytesseract).\n"
                "Los archivos .PDF y .TXT con texto digital sí son procesados directamente."
            )

    else:
        # Archivo de texto plano (.txt, .csv, etc.)
        with open(ruta_limpia, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()


class ClasificadorDocumental:
    """Clasificador bilingüe con soporte híbrido (scikit-learn o NumPy puro)."""

    def __init__(self, dataset_path: str = "dataset_bilingue.csv"):
        self.dataset_path = dataset_path
        self.classes = ["Form", "Letter", "Memo", "Report"]
        self.model_sklearn = None
        self.vocab = {}
        self.idf = {}
        self.class_priors = {}
        self.class_term_weights = {}
        self._entrenar()

    def _entrenar(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"No se encontró el dataset en '{self.dataset_path}'.")

        df = pd.read_csv(self.dataset_path)
        X_raw = df["texto_limpio"].fillna("").astype(str).tolist()
        y_raw = df["categoria"].tolist()

        try:
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression

            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=1500, ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=1000, C=1.0, penalty="l2", random_state=42))
            ])
            pipeline.fit(X_raw, y_raw)
            self.model_sklearn = pipeline
            self.classes = list(pipeline.classes_)
        except ImportError:
            # Fallback a clasificador probabilístico en NumPy puro
            doc_tokens = [t.split() for t in X_raw]
            df_counts = Counter()
            for tokens in doc_tokens:
                df_counts.update(set(tokens))

            top_terms = [t for t, _ in df_counts.most_common(1500)]
            self.vocab = {t: i for i, t in enumerate(top_terms)}
            N = len(doc_tokens)
            self.idf = {t: math.log((1 + N) / (1 + df_counts[t])) + 1 for t in top_terms}

            self.classes = sorted(list(set(y_raw)))
            self.class_priors = {c: math.log(y_raw.count(c) / N) for c in self.classes}
            self.class_term_weights = {c: np.zeros(len(self.vocab)) for c in self.classes}

            for tokens, c in zip(doc_tokens, y_raw):
                counts = Counter(tokens)
                for term, cnt in counts.items():
                    if term in self.vocab:
                        idx = self.vocab[term]
                        self.class_term_weights[c][idx] += cnt * self.idf[term]

            for c in self.classes:
                self.class_term_weights[c] = (self.class_term_weights[c] + 1.0) / (
                    self.class_term_weights[c].sum() + len(self.vocab)
                )

    def predecir(self, texto: str):
        texto_limpio = limpiar_texto(texto, idioma="auto")

        if self.model_sklearn is not None:
            pred = self.model_sklearn.predict([texto_limpio])[0]
            probs_arr = self.model_sklearn.predict_proba([texto_limpio])[0]
            probs = {c: float(p) for c, p in zip(self.model_sklearn.classes_, probs_arr)}
        else:
            tokens = texto_limpio.split()
            tokens_in_v = [t for t in tokens if t in self.vocab]
            scores = {}
            for c in self.classes:
                score = self.class_priors[c]
                for t in tokens_in_v:
                    score += math.log(self.class_term_weights[c][self.vocab[t]])
                scores[c] = score

            max_s = max(scores.values()) if scores else 0
            exp_scores = {c: math.exp(s - max_s) for c, s in scores.items()}
            tot = sum(exp_scores.values())
            probs = {c: exp_scores[c] / tot for c in self.classes}
            pred = max(probs, key=probs.get)

        confianza = probs[pred] * 100.0
        return pred, confianza, probs, texto_limpio


def mostrar_resultado(texto: str, pred: str, conf: float, probs: dict, texto_limpio: str, origen: str = ""):
    labels_es = {
        "Form": "Formulario / Ficha de Inspección",
        "Letter": "Carta / Oficio Notificatorio",
        "Memo": "Memorándum Interno",
        "Report": "Informe Técnico / Reporte"
    }

    print("\n" + "=" * 68)
    if origen:
        print(f"ORIGEN DEL DOCUMENTO: {origen}")
    extracto = (texto.strip().replace("\n", " "))[:110]
    print(f"DOCUMENTO (Extracto):  \"{extracto}...\"")
    print("-" * 68)
    print(f"CATEGORÍA PREDICHA:    [{pred.upper()}] -> {labels_es.get(pred, pred)}")
    print(f"NIVEL DE CONFIANZA:    {conf:.1f}%")
    print("-" * 68)
    print("DESGLOSE DE PROBABILIDADES POR CLASE:")
    for c in sorted(probs.keys()):
        p = probs[c] * 100
        barra = "█" * int(p / 5) + "░" * (20 - int(p / 5))
        nombre = f"{c} ({labels_es.get(c, '').split('/')[0].strip()}):"
        print(f"  {nombre:28s} {p:5.1f}%  |{barra}|")
    print("=" * 68)


CASOS_DEMO = [
    (
        "Memorándum Interno MINSA (Gestión y Turnos de Personal)",
        """MEMORÁNDUM N° 045-2024-MINSA/DIGESA
PARA: Coordinador del Equipo de Vigilancia Sanitaria
DE: Dirección Ejecutiva de Salud Ambiental
ASUNTO: Disposiciones de coordinación de turnos y personal comisionado para fiscalización inopinada en zonas comerciales.
FECHA: 14 de marzo de 2024.
Por medio del presente memorando se dispone que el personal a su cargo ejecute las rondas inspectivas en los distritos asignados, remitiendo a la brevedad la nómina de inspectores comisionados y las actas levantadas durante la última semana."""
    ),
    (
        "Oficio / Carta CENARES-MINSA (Notificación a Proveedor)",
        """Lima, 20 de mayo de 2024.
OFICIO N° 1289-2024-CENARES/MINSA
Señores: Distribuidora Farmacéutica del Centro S.A.C.
Dirección: Av. República de Panamá N° 3400, San Isidro.
De nuestra consideración:
Tengo el agrado de dirigirme a ustedes para poner en su conocimiento la notificación de requerimiento formal sobre entregas pendientes de insumos médicos. Solicitamos se sirvan remitir sus descargos técnicos en un plazo improrrogable de cinco días hábiles.
Sin otro particular, quedo de ustedes. Atentamente, Dirección de Adquisiciones."""
    ),
    (
        "Ficha / Formulario de Inspección Sanitaria (Acta OEFA/Sunat)",
        """ACTA DE INSPECCIÓN Y FISCALIZACIÓN SANITARIA REG-2024-089
Razón Social del Establecimiento: Inversiones Comerciales Altamar E.I.R.L.
RUC: 20601458921 | Licencia Municipal: LIC-2024-8902
CHECKLIST DE VERIFICACIÓN NORMATIVA:
- Señalética de ambiente 100% libre de humo en accesos: [X] CONFORME  [ ] NO CONFORME
- Ceniceros o colillas en ambientes techados: [X] NO PRESENTA  [ ] PRESENTA
- Muestras incautadas: 5 cajetillas de cigarrillos Lote LOT-2024-45.
Firma del Inspector Sanitario / Firma, Nombre y DNI del Administrado."""
    ),
    (
        "Technical Progress Report (Investigación en Inglés / Tobacco)",
        """TECHNICAL PROGRESS REPORT: EVALUATION OF GAS CHROMATOGRAPHY TESTING
To: Research and Development Committee
Objective: This report presents analytical sampling findings regarding smoke emissions, nicotine yields, and tar condensate levels across commercial cigarette brands.
Methodology: Multistage probabilistic sampling following ISO standards with gas chromatography laboratory analysis.
Results: Gas chromatography assays revealed mean nicotine concentration of 1.42 mg per cigarette rod with standard deviation of 0.18 mg.
Conclusions: Compliance was verified in 92% of evaluated batches."""
    )
]


def ejecutar_demo(clasificador: ClasificadorDocumental):
    print("\n" + "#" * 68)
    print("### DEMOSTRACIÓN DE CLASIFICACIÓN BILINGÜE EN TIEMPO REAL ###")
    print("### Corpus Entrenado: 600 Documentos (Tobacco-3482 + Gob.pe)   ###")
    print("#" * 68)

    for i, (titulo, texto) in enumerate(CASOS_DEMO, 1):
        print(f"\n[CASO {i}/4] {titulo}")
        pred, conf, probs, limpio = clasificador.predecir(texto)
        mostrar_resultado(texto, pred, conf, probs, limpio, origen="Caso de prueba predefinido")


def modo_interactivo(clasificador: ClasificadorDocumental):
    print("\n" + "=" * 68)
    print("  SISTEMA BILINGÜE DE CLASIFICACIÓN DE DOCUMENTOS ADMINISTRATIVOS")
    print("  Categorías: Form (Formulario) | Letter (Carta/Oficio)")
    print("              Memo (Memorándum) | Report (Informe Técnico)")
    print("=" * 68)
    print("Puedes:")
    print(" 1. Escribir o pegar el texto de cualquier documento.")
    print(" 2. Ingresar la ruta de un archivo .PDF o .TXT (ej. arrastrar y soltar el archivo).")
    print("(Escribe 'salir' para terminar o 'demo' para ver los 4 casos de prueba)\n")

    while True:
        try:
            entrada = input("\n>>> Ingresa texto o ruta de archivo (.pdf / .txt):\n").strip()
            if not entrada:
                continue
            if entrada.lower() in ("salir", "exit", "quit", "q"):
                print("¡Hasta luego!")
                break
            if entrada.lower() == "demo":
                ejecutar_demo(clasificador)
                continue

            ruta_potencial = entrada.strip("'").strip('"')
            if os.path.exists(ruta_potencial) and os.path.isfile(ruta_potencial):
                try:
                    texto_extraido = extraer_texto_de_archivo(ruta_potencial)
                    pred, conf, probs, limpio = clasificador.predecir(texto_extraido)
                    mostrar_resultado(texto_extraido, pred, conf, probs, limpio, origen=f"Archivo: {ruta_potencial}")
                except Exception as err:
                    print(f"[Error al leer archivo]: {err}")
                continue

            pred, conf, probs, limpio = clasificador.predecir(entrada)
            mostrar_resultado(entrada, pred, conf, probs, limpio, origen="Texto ingresado por consola")
        except (KeyboardInterrupt, EOFError):
            print("\n¡Sesión finalizada!")
            break


def main():
    parser = argparse.ArgumentParser(description="Clasificador bilingüe de documentos administrativos")
    parser.add_argument("--demo", action="store_true", help="Ejecutar los 4 casos de prueba predefinidos")
    parser.add_argument("--texto", type=str, help="Clasificar un fragmento de texto pasado como argumento")
    parser.add_argument("--archivo", type=str, help="Clasificar un archivo .PDF o .TXT directamente")
    args = parser.parse_args()

    print("[Iniciando] Cargando corpus documental bilingüe y entrenando modelo...")
    clasificador = ClasificadorDocumental()
    print("[Listo] Modelo ajustado exitosamente sobre 600 documentos.")

    if args.demo:
        ejecutar_demo(clasificador)
    elif args.texto:
        pred, conf, probs, limpio = clasificador.predecir(args.texto)
        mostrar_resultado(args.texto, pred, conf, probs, limpio, origen="Argumento --texto")
    elif args.archivo:
        try:
            texto = extraer_texto_de_archivo(args.archivo)
            pred, conf, probs, limpio = clasificador.predecir(texto)
            mostrar_resultado(texto, pred, conf, probs, limpio, origen=f"Archivo: {args.archivo}")
        except Exception as err:
            print(f"[Error]: {err}")
    else:
        modo_interactivo(clasificador)


if __name__ == "__main__":
    main()
