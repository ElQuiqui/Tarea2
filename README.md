# Sistema de Inspección Superficial — SVI 2026-I, Tarea 2

Pipeline clásico de visión por computador (sin deep learning) para
inspeccionar piezas planas (cuadrado de PLA negro mate impreso en 3D)
sobre fondo blanco (hoja o tela). Decide **OK / defectuoso** y, si es
defectuoso, clasifica el defecto en **rayado / contaminación / abrasión**,
localizándolo con bounding box y centroide.

---

## 1. Descripción por bloques

El sistema sigue el modelo canónico por bloques del curso:

```
Adquisición → Preprocesamiento → Segmentación → Detección → Clasificación → Salida
   (1)            (2)               (3)            (4)           (5)           (6)
```

| Bloque | Módulo | Entradas → Salidas | Técnica |
|---|---|---|---|
| 1 | `Adquisicion.py` | `path` → `BGR uint8 + metadatos` | `cv2.imread` + validaciones de forma/canal |
| 2 | `Preprocesamiento.py` | `BGR` → `BGR corregida + diagnóstico` | Histograma del canal L → diagnóstico de iluminación → CLAHE si requiere; estimación de ruido por MAD del Laplaciano → mediana si es impulsivo |
| 3 | `Segmentacion.py` | `BGR` → `ROI rectificado + máscara + warp` | Multi-Otsu sobre L (3 clases, pieza = la más oscura), apertura/cierre morfológico, componente conectado central, **convex hull** para rellenar concavidades (la pieza es convexa por diseño), `minAreaRect` → warp afín que alinea la pieza al eje |
| 4 | `Deteccion.py` | `ROI` → `mapas + scores + blobs` | **Cascada por clase**: 1) **Contaminación** = región con L muy por encima de la mediana de la pieza. 2) **Rayado** = blob(s) alargado(s) del filtro de Frangi multiescala; si hay > N blobs se exige que tengan orientación alineada (R circular ≥ 0.85), si no se considera patrón abrasivo. 3) **Abrasión** = parche compacto (alta solidity, eccentricidad acotada) con densidad local de gradiente Sobel alta, excluidas las zonas de contaminación y rayado |
| 5 | `Clasificador.py` | `scores + blobs` → `{estado, clase, confianza, bbox}` | Umbral de presencia **por clase** (porque los tamaños típicos difieren). Jerarquía coherente con la cascada del bloque 4: si varias clases activas, gana contaminación > rayado > abrasión |
| 6 | `Salida.py` | `→` overlay PNG, JSON por imagen, CSV batch | Bbox del ROI reproyectado a la imagen original con la matriz inversa del warp |
| 7 | `main.py` | CLI orquestador | `argparse` con `--input`, `--batch`, `--debug`, `--out` |

Cada función pública tiene docstring indicando qué bloque implementa,
qué hace y el fundamento teórico que la respalda. Todos los umbrales y
tamaños de kernel viven en `src/config.py`.

---

## 2. Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dependencias: `opencv-python`, `numpy`, `scikit-image`, `scikit-learn`,
`matplotlib`, `scipy`, `pytest`.

---

## 3. Estructura del proyecto

```
src/
  config.py            # umbrales y kernels (única fuente)
  utils_logging.py     # logging + dump de intermedios --debug
  Adquisicion.py       # bloque 1
  Preprocesamiento.py  # bloque 2
  Segmentacion.py      # bloque 3
  Deteccion.py         # bloque 4
  Clasificador.py      # bloque 5
  Salida.py            # bloque 6
  main.py              # bloque 7 (CLI)
tests/
  fixtures/labels.csv  # ← el usuario etiqueta aquí
  test_*.py            # un test por bloque + e2e
data/                  # provista por el usuario
  Hoja/{Iluminacion45,Iluminacion60,Iluminacionlateral}/*.jpg
  Tela/{Iluminacion45,Iluminacion60,Iluminacionlateral}/*.jpg
output/
  debug/<stem>/        # intermedios cuando --debug
  reports/             # JSON por imagen + batch.csv
  <stem>_overlay.png
```

---

## 4. Uso

### Procesar una imagen

```powershell
python -m src.main --input data/Hoja/Iluminacion45/img001.jpg --out output/
```

Genera `output/img001_overlay.png` y `output/reports/img001.json`.

### Procesar una carpeta (modo batch)

```powershell
python -m src.main --batch data/Hoja/Iluminacion45/ --out output/
```

Genera un overlay y un JSON por imagen, más `output/reports/batch.csv`.

### Modo `--debug`

```powershell
python -m src.main --input data/Hoja/Iluminacion45/img001.jpg --debug --out output/
```

Vuelca en `output/debug/img001/` los intermedios por bloque:

```
02a_imagen_corregida_clahe.png
02b_imagen_filtrada.png
02c_histogramas.png
03a_binarizacion_otsu.png
03b_mascara_pieza.png
03c_contorno_y_rect.png
03d_roi_rectificado.png
03e_roi_mascara.png
04a_interior_pieza.png
04b_mapa_rayado_float.png
04b_mapa_rayado_mask.png
04c_mapa_contaminacion_dE.png
04c_mapa_contaminacion_mask.png
04d_mapa_abrasion_std.png
04d_mapa_abrasion_mask.png
```

### Etiquetar imágenes para tests / calibración

Editar `tests/fixtures/labels.csv` con filas:

```
filepath,clase,iluminacion,fondo
data/Hoja/Iluminacion45/img001.jpg,ok,45,hoja
data/Tela/Iluminacionlateral/img007.jpg,rayado,lateral,tela
data/Hoja/Iluminacion60/img011.jpg,contaminacion,60,hoja
data/Hoja/Iluminacion45/img023.jpg,abrasion,45,hoja
```

Luego:

```powershell
pytest tests/ -v
```

Si los tests por clase fallan, revisar los umbrales correspondientes en
`src/config.py` (sección del bloque 4) y re-ejecutar.

---

## 5. Salida JSON — ejemplo

```json
{
  "imagen": {"filename": "img007.jpg", "shape": [480, 640, 3]},
  "preprocesamiento": {
    "iluminacion": {"diagnostico": "contraste_bajo", "media": 132.4, "std": 18.9,
                    "pct_oscuros": 1.2, "pct_claros": 0.3, "requiere_correccion": true},
    "ruido": {"mad_laplaciano": 5.3, "sigma_estimado": 7.85, "kurtosis": 4.1, "es_impulsivo": true},
    "filtro_aplicado": {"filtro": "mediana", "kernel": 3}
  },
  "segmentacion": {"centroide_xy": [320.4, 240.1], "angulo_grados": 12.3,
                    "rect_rotado": {"centro": [320.4, 240.1], "tamaño": [180.0, 178.5], "angulo": 12.3},
                    "roi_size_wh": [532, 530]},
  "deteccion": {"scores": {"rayado": 0.087, "contaminacion": 0.004, "abrasion": 0.002},
                 "blobs": {...}},
  "clasificacion": {"estado": "defectuoso", "clase": "rayado", "confianza": 0.84,
                     "bbox_roi_xywh": [212, 180, 96, 6],
                     "centroide_roi_xy": [260.1, 183.0],
                     "scores_por_clase": {"rayado": 0.087, "contaminacion": 0.004, "abrasion": 0.002},
                     "justificacion": "Score ganador rayado=0.087; 2º=contaminacion=0.004; margen=21.75x."},
  "salida": {"overlay_path": "C:/.../img007_overlay.png"}
}
```

---

## 6. Decisiones de diseño y alternativas consideradas

### Bloque 2 — diagnóstico de iluminación

- **Elegida**: percentiles de píxeles oscuros/claros + media/std del canal L
  → cuatro categorías interpretables (`ok`, `subexpuesta`, `sobreexpuesta`,
  `contraste_bajo`). CLAHE solo si se diagnostica problema.
- **Alternativas**:
  - *Ecualización global de histograma*: descartada porque amplifica ruido
    y satura las zonas oscuras de la pieza, donde están las pistas de
    abrasión y rayado.
  - *Gamma correction*: requiere un γ específico por escena que es difícil
    de inferir automáticamente; CLAHE es no paramétrico en ese sentido.
  - *Histograma sobre BGR conjunto*: descartado porque mezcla color y brillo,
    haciendo el diagnóstico menos claro.

### Bloque 2 — reducción de ruido

- **Elegida**: mediana 3×3 solo cuando MAD del Laplaciano > umbral **y**
  kurtosis > 3 (ruido impulsivo). Aplicar mediana siempre suaviza rayados
  finos.
- **Alternativas**:
  - *Filtro bilateral*: preserva bordes pero costoso y no soluciona ruido
    impulsivo tan bien como la mediana.
  - *Gaussiano*: descartado por suavizar estructuras tipo línea (rayado).
  - *Non-local means*: excelente calidad pero overkill para este escenario
    controlado.

### Bloque 3 — segmentación

- **Elegida**: multi-Otsu (3 clases) sobre L (pieza = clase más oscura)
  + apertura (elimina textura de tela) + cierre (rellena huecos) +
  selección del blob central + **convex hull** de la silueta. La pieza
  es un cuadrado, convexo por diseño, así que cualquier concavidad o
  hueco en la binarización es un defecto (típicamente contaminación
  brillante que Otsu metió en la clase "fondo"). El hull restaura la
  forma correcta antes de rectificar, garantizando que la contaminación
  quede DENTRO del ROI y sea detectable. Rectificación con `minAreaRect`
  + warp afín.
- **Por qué multi-Otsu**: bajo iluminación lateral aparece un tercer
  modo en el histograma (sombra rasante intermedia entre pieza y fondo).
  Otsu de 2 clases agrupa pieza+sombra; multi-Otsu de 3 clases las
  separa y deja sólo la pieza en la clase más oscura.
- **Por qué convex hull**: bug encontrado durante la calibración —
  contaminaciones brillantes (e.g. pegamento) tienen L > umbrales[0] y
  Otsu las clasifica como fondo. Cuando esa contaminación toca el borde
  de la silueta produce una concavidad, no un hueco cerrado, así que
  `findContours+drawContours(FILLED)` no la rellena. El hull sí. Fix:
  +12 pp de accuracy en la calibración.
- **Alternativas**:
  - *Umbralización adaptativa* (`cv2.adaptiveThreshold`): considerada para
    iluminación lateral; multi-Otsu+CLAHE suele ser suficiente y produce
    máscara más limpia (la adaptativa fragmenta la pieza interior).
  - *Segmentación por color en HSV*: la pieza negra tiene saturación
    indefinida (S ≈ 0), poco discriminativa. L (luminancia) es la pista
    natural.
  - *GrabCut*: muy preciso pero requiere semilla manual; descartado por
    automatización.
  - *Detección de cuatro esquinas + homografía*: más preciso para
    rectificación, pero `minAreaRect` es suficiente cuando la pieza es
    plana y casi cuadrada — y no requiere detectar esquinas explícitamente.

### Bloque 4 — Mapa de contaminación

- **Elegida**: detección de región brillante (L > mediana_L + Δ) sobre
  la pieza, seguida de morfología y filtro de área mínima.
- **Por qué**: tras inspeccionar el dataset etiquetado se observó que la
  contaminación real en estas piezas son parches GRANDES y muy CLAROS
  (pegamento, residuos blancos, marcas). La pieza PLA negra mate tiene
  L ≈ 20-40; la contaminación tiene L > 100. Un umbral simple sobre L
  relativo a la mediana captura esto con altísima precisión y recall.
- **Alternativas descartadas tras la calibración**:
  - *ΔE CIE76 completo (L+a+b)*: super-sensible al gradiente de
    iluminación → marca todo como contaminación. **Probado: 0/3 piezas
    OK clasificadas correctamente, 30.8% global.**
  - *Croma sólo (a*, b*)*: subestima contaminación acromática (polvo
    blanco), que es la observada en este dataset. **Probado: 30.8%.**
  - *Residuo Lab vs baseline Gaussiano local*: técnicamente robusto a
    iluminación pero introduce demasiados falsos positivos por textura
    matte. **Probado: 0%.**

### Bloque 4 — Mapa de abrasión

- **Elegida**: densidad local de píxeles con magnitud de Sobel alta,
  con filtros morfológicos por área mínima, eccentricidad máxima (un
  blob muy elongado es rayado, no abrasión) y solidity mínima
  (gradientes de iluminación dan máscaras fragmentadas con poca solidity).
  Se excluye la zona ya marcada como contaminación.
- **Por qué**: el dataset muestra que la abrasión real es un parche con
  muchos micro-rayones distribuidos (no un cambio uniforme de color).
  Eso eleva la densidad de gradiente en una región compacta. Los filtros
  morfológicos posteriores son críticos para no confundirla con rayados
  individuales ni con gradientes de iluminación.
- **Alternativas descartadas**:
  - *std local de L*: capta abrasión pero también la matte del PLA y la
    transición pieza-borde. Genera falsos positivos en OK.
  - *Entropía local*, *LBP*: similar problema, sin mejoras significativas.

### Bloque 4 — Mapa de rayado

- **Elegida**: filtro de Frangi multiescala (vesselness), umbral,
  cierre morfológico (para puentear gaps periódicos del Frangi),
  filtros por longitud / eccentricidad / ancho. Si hay > N blobs se
  exige **alineación angular**: se calcula la concentración circular
  R = |mean(exp(2iθ))| de las orientaciones (mod π, porque un segmento
  es equivalente a su opuesto). R ≥ 0.85 ⇒ blobs paralelos = rayado
  múltiple. R bajo ⇒ orientaciones aleatorias = patrón abrasivo.
- **Por qué**: Frangi (1998) responde fuertemente a estructuras tipo
  línea, incluso con bajo contraste. La respuesta tiende a ser
  "encadenada" — bright nodes con necks finos. Por eso el cierre
  morfológico es importante para que los segmentos vecinos cuenten
  como un único blob largo. La métrica de alineación angular resuelve
  el caso real visto en el dataset de "varios rayones paralelos"
  (etiquetado por el usuario como `rayado`, no `abrasion`).
- **Por qué rayado va ANTES que abrasión en la cascada**: un patrón
  de rayones paralelos también dispara densidad de bordes en la zona
  vecina (la ventana 31×31 absorbe los gradientes), así que el detector
  de abrasión se activa "envolviéndolo". Detectando rayado primero y
  excluyendo su máscara de abrasión evitamos la clasificación errónea.
- **Limitación conocida**: rayados extremadamente tenues sobre tela
  pueden caer por debajo del umbral de presencia incluso después de
  bajarlo a 0.01 % del área.

### Bloque 5 — Clasificación por reglas

- **Elegida**: regla con **umbral de presencia por clase** + jerarquía
  fija (contaminación > rayado > abrasión). Coherente con la cascada
  del bloque 4: si varios detectores fueron disparados, gana el que
  corre primero en la cascada. Rayado tiene precedencia sobre abrasión
  porque la ventana de densidad de abrasión absorbe los gradientes de
  los rayones vecinos, dando un score alto "colateral".
- **Por qué umbrales por clase**: los scores son fracciones de área de
  la pieza ocupadas por la máscara validada. Una contaminación típica
  es 5-20% del área; una abrasión 1-10%; un rayado <1%. Un único
  `SCORE_OK_MAX` no puede discriminar bien entre estos tres rangos
  sin sacrificar uno.
- **Alternativas descartadas**:
  - *k-NN / SVM entrenado*: con 13 imágenes etiquetadas heterogéneas
    el clasificador sobreajusta. La regla explícita es más estable.
  - *Score único combinado (max o suma ponderada)*: ofusca la
    contribución de cada defecto y dificulta calibrar.

### Bloque 6 — overlay sobre imagen original

- **Elegida**: el bbox detectado en el ROI rectificado se reproyecta a la
  imagen original con la matriz inversa del warp afín. El operador ve la
  detección en el espacio de la cámara, no en uno transformado.
- **Alternativas**:
  - *Mostrar el ROI rectificado*: pedagógico (se usa en `--debug`) pero
    rompe la correspondencia visual con la imagen capturada.

---

## 7. Resultado de calibración

Con 24 imágenes etiquetadas (6 OK, 6 rayado, 6 contaminación, 6 abrasión,
cubriendo las 6 combinaciones de fondo × iluminación):

```
                  abrasion  contaminacion   ok    rayado
  abrasion      |    5            0          1      0       (5/6 ✓)
  contaminacion |    0            6          0      0       (6/6 ✓)
  ok            |    0            0          6      0       (6/6 ✓)
  rayado        |    0            0          1      5       (5/6 ✓)

Aciertos: 23/24 = 95.8 %
```

Por bucket (priorización del operador):

| Bucket | Aciertos |
|---|---|
| Hoja con iluminación lateral | **4/4 (100 %)** |
| Hoja (todas las iluminaciones) | **12/12 (100 %)** |
| Iluminación lateral (Hoja+Tela) | **8/8 (100 %)** |
| Tela no-lateral | 7/8 (87.5 %) |

### Historial de calibración

| Iteración | Cambio | Aciertos |
|---|---|---|
| Baseline | Multi-Otsu + detectores físicos | 17/24 (70.8 %) |
| #1 | + Convex hull en segmentación + umbral rayado 0.0005 → 0.0002 | 20/24 (83.3 %) |
| #2 | + Cascada flip (rayado antes que abrasión) + jerarquía contam>rayado>abras | 22/24 (91.7 %) |
| #3 | + Density 0.15 → 0.10 + umbral abrasión 0.005 → 0.015 (limpia ruido) | 22/24 (91.7 %, mejor cualitativamente) |
| #4 | + Umbral rayado 0.0002 → 0.0001 | **23/24 (95.8 %)** |

La única falla restante (`Tela/Iluminacion45/tela45_12`) es una
abrasión circular extremadamente sutil que el detector de densidad de
bordes no consigue separar de la textura de la tela; bajar más el
umbral introduce falsos positivos en piezas OK.

Reproducir: `python -m tools.eval_calibracion`. Para iterar, modificar
umbrales en `src/config.py` y volver a correr.

## 8. Degradación elegante

`main.py` envuelve cada bloque en `try/except`. Si un bloque falla
(p. ej. imagen totalmente blanca → la segmentación no encuentra blob
central), el sistema produce un reporte con `estado="error_<bloque>"`
y un mensaje, sin abortar el batch ni mostrar trazas crudas al usuario.

---

## 9. Tests

```powershell
pytest tests/ -v
```

- `test_adquisicion.py`: forma y dtype de la imagen cargada.
- `test_preprocesamiento.py`: diagnóstico genera claves esperadas;
  CLAHE preserva dimensiones.
- `test_segmentacion.py`: centroide cae en la zona central; warp inverso
  es realmente inverso.
- `test_deteccion.py`: para cada clase etiquetada en `labels.csv`, el
  score correspondiente es el ganador en al menos la mitad de las imágenes
  (tolerancia para calibrar iterativamente).
- `test_clasificador.py`: unitarios sin imágenes (mock de
  `ResultadoDeteccion`).
- `test_pipeline_e2e.py`: el pipeline corre completo sin excepciones y
  genera overlay + JSON.

Los tests que dependen de `labels.csv` se *skipean* si está vacío,
permitiendo correr el resto del suite mientras el dataset crece.
