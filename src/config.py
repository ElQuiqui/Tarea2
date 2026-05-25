"""
Configuración central del pipeline de inspección.

Cada constante está en este archivo (no se permiten "números mágicos" en
los módulos de procesamiento) para que un evaluador o ingeniero de
calibración pueda ajustar el sistema sin tocar la lógica.

Los valores actuales son razonables para un ROI rectificado de ~512 px
de lado, pieza de PLA negro mate sobre fondo blanco (hoja o tela).
"""

# ────────────────────────────────────────────────────────────────────────────
# Bloque 2 — Preprocesamiento
# ────────────────────────────────────────────────────────────────────────────
HIST_BINS = 256

# Umbrales para diagnóstico de iluminación.
# Si más del EXPOSURE_*_PCT % de los píxeles cae bajo/encima del nivel
# correspondiente, se considera sub/sobreexpuesta.
EXPOSURE_LOW_PCT = 5.0
EXPOSURE_HIGH_PCT = 5.0
EXPOSURE_DARK_THRESH = 25
EXPOSURE_BRIGHT_THRESH = 230

# Rango de media global aceptable; fuera de esto se marca contraste pobre
# y se aplica CLAHE.
MEAN_OK_MIN = 60
MEAN_OK_MAX = 200
STD_OK_MIN = 25  # desviación estándar mínima para considerar contraste útil

# CLAHE: ecualización adaptativa con limitación de contraste.
# Se aplica sobre el canal L de Lab para no introducir desbalance cromático.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

# Estimación de ruido por MAD del Laplaciano (Immerkaer-like).
# Si el estimador supera el umbral se aplica filtro mediana.
NOISE_MAD_THRESHOLD = 4.0
MEDIAN_KERNEL = 3   # impar; el módulo eleva a 5 si ruido muy alto
MEDIAN_KERNEL_HIGH = 5
NOISE_HIGH_THRESHOLD = 8.0

# ────────────────────────────────────────────────────────────────────────────
# Bloque 3 — Segmentación
# ────────────────────────────────────────────────────────────────────────────
# Otsu se aplica sobre L invertido (pieza oscura → blob blanco).
MORPH_OPEN_KERNEL = 5
MORPH_CLOSE_KERNEL = 7

# Filtros de validez del blob.
MIN_PIECE_AREA_RATIO = 0.02   # fracción mínima del área de la imagen
CENTER_TOLERANCE_RATIO = 0.35 # bbox debe tocar el cuadrado central de este tamaño

# Rectificación afín.
RECTIFIED_PADDING_PX = 10
RECTIFIED_SIZE_MAX = 512      # lado mayor del ROI rectificado

# ────────────────────────────────────────────────────────────────────────────
# Bloque 4 — Detección de defectos
# ────────────────────────────────────────────────────────────────────────────
# Ignorar n px junto al borde del ROI: el warp puede dejar sombras o el borde
# del propio cuadrado puede confundirse con un rayado.
ROI_EDGE_MARGIN_PX = 20

# (Constantes de rayado, contaminación y abrasión definidas arriba con
#  el nuevo modelo físico por clase.)

# Contaminación = región MUCHO MÁS CLARA que la pieza.
# Modelo físico: la pieza PLA negra mate tiene L bajo (~30-40). Una
# contaminación real (pegamento, polvo, etiqueta, marcas) es claramente
# más brillante. Umbral relativo a la mediana del interior para ser
# robusto a la iluminación absoluta.
CONTAM_L_DELTA = 60.0           # ΔL sobre la mediana del interior → "claro"
CONTAM_MIN_AREA = 200           # px: contaminación debe ser un parche grande
CONTAM_MORPH_KERNEL = 7

# Abrasión = región con ALTA DENSIDAD DE BORDES (microrayones).
# Modelo: la abrasión es un parche que contiene muchos arañazos finos
# distribuidos. Detectarla como densidad local de gradiente alto.
# Filtros morfológicos:
#   - eccentricity < MAX_ECCENTRICITY  → si es muy elongado, es un rayado
#   - solidity     > MIN_SOLIDITY      → la abrasión es compacta; gradientes
#                                          de iluminación dan máscaras fragmentadas
ABRASION_GRAD_THRESH = 30       # magnitud Sobel para pixel "con borde" (subido para descartar matte)
ABRASION_DENSITY_WIN = 31       # tamaño de ventana para densidad local (impar)
ABRASION_DENSITY_THRESH = 0.10  # fracción de píxeles con borde en la ventana
ABRASION_MIN_AREA = 400
ABRASION_MORPH_KERNEL = 9
ABRASION_MAX_ECCENTRICITY = 0.92  # blobs muy alargados → rayado, no abrasión
ABRASION_MIN_SOLIDITY = 0.50      # ratio area/convex_hull; gradientes de luz dan <0.5

# Rayado = LÍNEA ÚNICA larga y delgada (no patrón texturado).
# Frangi multiescala. Se valida que el blob sea elongado y no parte de
# un patrón abrasivo.
FRANGI_SIGMAS = (1, 2, 3)
FRANGI_BETA = 0.5
FRANGI_GAMMA = 15
SCRATCH_BIN_THRESH = 0.08
SCRATCH_LINK_KERNEL = 7          # close kernel para puentear gaps periódicos del Frangi
SCRATCH_MIN_LENGTH = 30          # px: rayado debe ser claramente largo
SCRATCH_MIN_ECCENTRICITY = 0.92
SCRATCH_MAX_WIDTH = 18           # tolera engrosamiento por el close
SCRATCH_MAX_BLOBS = 3            # ≤ este número de blobs ⇒ rayado sin más chequeos
SCRATCH_ALIGN_R_MIN = 0.85       # con > MAX_BLOBS blobs, exige R≥esto (orientaciones paralelas) para rayado

# Scoring: cada clase produce un score = fracción del área de la pieza
# ocupada por su máscara validada (en [0,1]). Los rangos esperados
# difieren por clase (contaminación grande, rayado pequeño), por lo que
# se usan umbrales independientes por clase en el clasificador.

# (Abrasión: ver constantes arriba — usa densidad de bordes, no std local.)

# ────────────────────────────────────────────────────────────────────────────
# Bloque 5 — Clasificación
# ────────────────────────────────────────────────────────────────────────────
# Scores por clase: fracción de área de la pieza ocupada por la máscara
# validada de cada defecto. Umbrales independientes por clase porque
# los tamaños típicos difieren.
SCORE_CONTAM_PRESENT = 0.005    # ≥0.5% del área → contaminación presente
SCORE_ABRASION_PRESENT = 0.015  # ≥1.5% del área → abrasión presente (filtra ruido de gradiente)
SCORE_RAYADO_PRESENT = 0.0001   # rayado es delgado: ≥0.01% basta (líneas finas dan poco área)
SCORE_CLEAR_MARGIN = 1.5
CONFIDENCE_FLOOR = 0.35

# ────────────────────────────────────────────────────────────────────────────
# Bloque 6 — Salida visual
# ────────────────────────────────────────────────────────────────────────────
OVERLAY_CONTOUR_COLOR = (0, 255, 0)  # verde, BGR
OVERLAY_BBOX_COLOR = (0, 0, 255)     # rojo, BGR
OVERLAY_TEXT_COLOR = (255, 255, 255) # blanco
OVERLAY_FONT_SCALE = 0.6
OVERLAY_THICKNESS = 2
JSON_INDENT = 2

# ────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%H:%M:%S"
