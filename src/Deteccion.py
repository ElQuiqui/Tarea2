"""
Bloque 4 — Extracción de características y detección de defectos.

El diseño de cada detector está guiado por la *fenomenología observada*
en el dataset etiquetado:

- **Contaminación** (pegamento, polvo, marcas): aparece como un PARCHE
  GRANDE Y CLARO sobre la pieza. La pieza PLA negra mate tiene L bajo;
  cualquier contaminación tiene L mucho mayor. Detector: pixeles con
  L > mediana_L + Δ → morfología → blob grande.

- **Abrasión**: zona compacta con MUCHOS MICRO-RAYONES, alta densidad de
  gradiente sobre una región acromática. Detector: magnitud Sobel →
  filtro local de densidad → blob compacto. La región se descarta como
  abrasión si está dentro de un parche de contaminación (overlap).

- **Rayado**: UNA SOLA línea fina, larga y elongada. Detector: filtro de
  Frangi multiescala → blobs alargados → exigir que sean pocos (un
  patrón con muchos blobs es abrasión, no rayado).

Jerarquía: contaminación se evalúa primero y enmascara la zona; abrasión
luego sobre el resto; rayado al final. Esto evita doble conteo.

Score por clase = fracción del área de la pieza ocupada por la máscara
validada. Cada clase tiene su propio umbral de presencia en config.py
(los rangos típicos difieren por clase).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List, Optional

import cv2
import numpy as np
from skimage.filters import frangi
from skimage.measure import label as sk_label, regionprops

from . import config
from .utils_logging import get_logger, dump_debug_image, is_debug
from .Segmentacion import ResultadoSegmentacion

_log = get_logger("deteccion")


@dataclass
class BlobInfo:
    bbox: Tuple[int, int, int, int]
    centroide: Tuple[float, float]
    area: int
    eccentricidad: float
    delta_e_medio: float = 0.0


@dataclass
class MapaDefecto:
    clase: str
    mapa_float: np.ndarray
    mascara: np.ndarray
    score: float
    blobs: List[BlobInfo] = field(default_factory=list)


@dataclass
class ResultadoDeteccion:
    rayado: MapaDefecto
    contaminacion: MapaDefecto
    abrasion: MapaDefecto

    def scores(self) -> dict:
        return {
            "rayado": self.rayado.score,
            "contaminacion": self.contaminacion.score,
            "abrasion": self.abrasion.score,
        }


# ─── Utilidades ─────────────────────────────────────────────────────────────

def _mascara_interior_pieza(roi_mascara: np.ndarray) -> np.ndarray:
    """Erosiona la máscara de la pieza para descartar `ROI_EDGE_MARGIN_PX`
    cerca del borde — el warp y el highlight del canto introducen
    artefactos que producirían falsos positivos en todos los detectores.
    """
    if config.ROI_EDGE_MARGIN_PX <= 0:
        return (roi_mascara > 0).astype(np.uint8) * 255
    k = 2 * config.ROI_EDGE_MARGIN_PX + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(roi_mascara, kernel)


def _extraer_blobs(
    mascara: np.ndarray,
    min_area: int,
    extra_mapa: Optional[np.ndarray] = None,
) -> List[BlobInfo]:
    """Devuelve componentes conectados que superen `min_area`."""
    labels = sk_label(mascara > 0, connectivity=2)
    blobs: List[BlobInfo] = []
    for prop in regionprops(labels):
        if prop.area < min_area:
            continue
        minr, minc, maxr, maxc = prop.bbox
        bbox = (int(minc), int(minr), int(maxc - minc), int(maxr - minr))
        cy, cx = prop.centroid
        ecc = float(prop.eccentricity)
        de_mean = 0.0
        if extra_mapa is not None:
            region_mask = (labels == prop.label)
            vals = extra_mapa[region_mask]
            if vals.size:
                de_mean = float(vals.mean())
        blobs.append(BlobInfo(
            bbox=bbox,
            centroide=(float(cx), float(cy)),
            area=int(prop.area),
            eccentricidad=ecc,
            delta_e_medio=de_mean,
        ))
    return blobs


def _score_area(mascara_defecto: np.ndarray, mascara_pieza: np.ndarray) -> float:
    area_pieza = float((mascara_pieza > 0).sum())
    if area_pieza <= 0:
        return 0.0
    return float((mascara_defecto > 0).sum()) / area_pieza


# ─── Contaminación: parche grande y claro ───────────────────────────────────

def mapa_contaminacion(roi_bgr: np.ndarray, mascara_pieza_interior: np.ndarray) -> MapaDefecto:
    """Detecta contaminación como región significativamente más clara
    que la mediana de la pieza.

    Bloque: 4. Fundamento: la pieza es PLA negro mate (L bajo, típicamente
    20-40 en escala OpenCV). Una contaminación material (pegamento, polvo,
    pintura, marca) tiene L mucho mayor (>100). El umbral relativo a la
    mediana de la pieza adapta automáticamente la decisión a la
    exposición global.
    """
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    interior = mascara_pieza_interior > 0
    if not interior.any():
        return MapaDefecto("contaminacion", L, np.zeros_like(L, dtype=np.uint8), 0.0, [])

    mediana_L = float(np.median(L[interior]))
    umbral = mediana_L + config.CONTAM_L_DELTA
    mapa = np.clip(L - mediana_L, 0.0, None)  # excedente positivo de L
    mapa[~interior] = 0.0

    mascara = ((L > umbral) & interior).astype(np.uint8) * 255

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.CONTAM_MORPH_KERNEL, config.CONTAM_MORPH_KERNEL)
    )
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k)

    blobs = _extraer_blobs(mascara, config.CONTAM_MIN_AREA, extra_mapa=mapa)
    out_mask = np.zeros_like(mascara)
    for b in blobs:
        x, y, w, h = b.bbox
        out_mask[y:y + h, x:x + w] = np.where(
            mascara[y:y + h, x:x + w] > 0, 255, out_mask[y:y + h, x:x + w]
        )

    score = _score_area(out_mask, mascara_pieza_interior)
    _log.info("mapa contaminación: score=%.4f blobs=%d (mediana_L=%.1f umbral=%.1f)",
              score, len(blobs), mediana_L, umbral)
    return MapaDefecto(
        clase="contaminacion",
        mapa_float=mapa,
        mascara=out_mask,
        score=score,
        blobs=blobs,
    )


# ─── Abrasión: parche con alta densidad de bordes ───────────────────────────

def mapa_abrasion(
    roi_bgr: np.ndarray,
    mascara_pieza_interior: np.ndarray,
    mascara_contaminacion: np.ndarray,
) -> MapaDefecto:
    """Detecta abrasión como región con alta densidad local de bordes.

    Bloque: 4. Fundamento: la abrasión observada en el dataset es un
    patrón de muchos micro-rayones agrupados (no un cambio de color
    uniforme). Eso produce alta densidad de gradiente (magnitud Sobel
    elevada) en una región compacta. Para no contar pegamento texturado
    como abrasión, se excluye de antemano la máscara de contaminación.
    """
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    interior = (mascara_pieza_interior > 0) & (mascara_contaminacion == 0)
    if not interior.any():
        return MapaDefecto("abrasion", np.zeros_like(L, dtype=np.float32),
                            np.zeros_like(L, dtype=np.uint8), 0.0, [])

    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    es_borde = (grad > config.ABRASION_GRAD_THRESH).astype(np.float32)
    es_borde[~interior] = 0.0

    # Densidad local de "píxeles con borde" en una ventana cuadrada.
    densidad = cv2.boxFilter(
        es_borde, ddepth=cv2.CV_32F,
        ksize=(config.ABRASION_DENSITY_WIN, config.ABRASION_DENSITY_WIN),
        normalize=True, borderType=cv2.BORDER_REFLECT,
    )
    densidad[~interior] = 0.0

    mascara = ((densidad > config.ABRASION_DENSITY_THRESH) & interior).astype(np.uint8) * 255
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.ABRASION_MORPH_KERNEL, config.ABRASION_MORPH_KERNEL)
    )
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k)

    # Filtrar blobs: área mínima, no demasiado elongados (eso es rayado),
    # razonablemente compactos (gradiente de iluminación da máscaras
    # fragmentadas con solidity baja).
    labels = sk_label(mascara > 0, connectivity=2)
    out_mask = np.zeros_like(mascara)
    blobs: List[BlobInfo] = []
    for prop in regionprops(labels):
        if prop.area < config.ABRASION_MIN_AREA:
            continue
        if prop.eccentricity > config.ABRASION_MAX_ECCENTRICITY:
            continue  # demasiado elongado → es un rayado
        if prop.solidity < config.ABRASION_MIN_SOLIDITY:
            continue  # fragmentado → probablemente gradiente de luz, no abrasión
        coords = prop.coords
        out_mask[coords[:, 0], coords[:, 1]] = 255
        minr, minc, maxr, maxc = prop.bbox
        cy, cx = prop.centroid
        blobs.append(BlobInfo(
            bbox=(int(minc), int(minr), int(maxc - minc), int(maxr - minr)),
            centroide=(float(cx), float(cy)),
            area=int(prop.area),
            eccentricidad=float(prop.eccentricity),
        ))

    score = _score_area(out_mask, mascara_pieza_interior)
    _log.info("mapa abrasión: score=%.4f blobs=%d", score, len(blobs))
    return MapaDefecto(
        clase="abrasion",
        mapa_float=densidad,
        mascara=out_mask,
        score=score,
        blobs=blobs,
    )


# ─── Rayado: línea única elongada ───────────────────────────────────────────

def mapa_rayado(
    roi_bgr: np.ndarray,
    mascara_pieza_interior: np.ndarray,
    mascara_contaminacion: np.ndarray,
    mascara_abrasion: np.ndarray,
) -> MapaDefecto:
    """Detecta rayado como una o pocas líneas elongadas (Frangi).

    Bloque: 4. Fundamento: Frangi (1998) responde a estructuras tipo
    línea. Para distinguir un rayado (1-2 líneas) de un patrón abrasivo
    (muchas líneas en una región), se excluyen las zonas ya marcadas
    como abrasión / contaminación y se rechaza el caso si quedan
    demasiados blobs alargados (eso indica abrasión, no rayado).
    """
    interior = ((mascara_pieza_interior > 0)
                & (mascara_contaminacion == 0)
                & (mascara_abrasion == 0))
    if not interior.any():
        z = np.zeros(mascara_pieza_interior.shape, dtype=np.float32)
        return MapaDefecto("rayado", z, z.astype(np.uint8), 0.0, [])

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    f_bright = frangi(gray, sigmas=config.FRANGI_SIGMAS,
                      beta=config.FRANGI_BETA, gamma=config.FRANGI_GAMMA,
                      black_ridges=False)
    f_dark = frangi(gray, sigmas=config.FRANGI_SIGMAS,
                    beta=config.FRANGI_BETA, gamma=config.FRANGI_GAMMA,
                    black_ridges=True)
    mapa = np.maximum(f_bright, f_dark)
    if interior.any():
        m_max = float(mapa[interior].max())
        if m_max > 1e-9:
            mapa = mapa / m_max
    mapa[~interior] = 0.0

    mascara = ((mapa > config.SCRATCH_BIN_THRESH) & interior).astype(np.uint8) * 255
    # Cierre isotrópico moderado: bridgea los pequeños gaps periódicos de la
    # respuesta de Frangi sin ensanchar demasiado el rayado.
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.SCRATCH_LINK_KERNEL, config.SCRATCH_LINK_KERNEL)
    )
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k)

    labels = sk_label(mascara > 0, connectivity=2)
    blobs_valid: List[BlobInfo] = []
    out_mask = np.zeros_like(mascara)
    orientaciones: List[float] = []
    for prop in regionprops(labels):
        if prop.area < config.SCRATCH_MIN_LENGTH:
            continue
        if prop.eccentricity < config.SCRATCH_MIN_ECCENTRICITY:
            continue
        if prop.axis_minor_length > config.SCRATCH_MAX_WIDTH:
            continue
        coords = prop.coords
        out_mask[coords[:, 0], coords[:, 1]] = 255
        minr, minc, maxr, maxc = prop.bbox
        cy, cx = prop.centroid
        blobs_valid.append(BlobInfo(
            bbox=(int(minc), int(minr), int(maxc - minc), int(maxr - minr)),
            centroide=(float(cx), float(cy)),
            area=int(prop.area),
            eccentricidad=float(prop.eccentricity),
        ))
        orientaciones.append(float(prop.orientation))

    # Discriminar rayado (paralelos) vs abrasión (orientaciones dispersas).
    # Si hay muchos blobs PERO comparten orientación → rayado múltiple.
    # Si están en orientaciones aleatorias → patrón abrasivo.
    # Métrica circular (mod π porque la orientación de un segmento es
    # equivalente con su opuesto): R = |mean(exp(2iθ))|; R≈1 = paralelos,
    # R≈0 = uniformes en ángulo.
    if len(blobs_valid) > config.SCRATCH_MAX_BLOBS:
        if len(orientaciones) >= 2:
            sin2 = float(np.mean(np.sin(2.0 * np.array(orientaciones))))
            cos2 = float(np.mean(np.cos(2.0 * np.array(orientaciones))))
            R = float(np.sqrt(sin2 * sin2 + cos2 * cos2))
        else:
            R = 1.0
        if R >= config.SCRATCH_ALIGN_R_MIN:
            _log.info("mapa rayado: %d blobs alineados (R=%.2f) → rayado múltiple",
                      len(blobs_valid), R)
        else:
            _log.info("mapa rayado: %d blobs > %d con R=%.2f → patrón abrasivo, descarto",
                      len(blobs_valid), config.SCRATCH_MAX_BLOBS, R)
            out_mask[:] = 0
            blobs_valid = []

    score = _score_area(out_mask, mascara_pieza_interior)
    _log.info("mapa rayado: score=%.5f blobs=%d", score, len(blobs_valid))
    return MapaDefecto(
        clase="rayado",
        mapa_float=mapa,
        mascara=out_mask,
        score=score,
        blobs=blobs_valid,
    )


# ─── Pipeline compuesto ─────────────────────────────────────────────────────

def detectar_defectos(seg: ResultadoSegmentacion, image_stem: str = "frame") -> ResultadoDeteccion:
    """Aplica los tres detectores en jerarquía sobre el ROI rectificado.

    Bloque: 4. Orden: contaminación → rayado → abrasión. Rayado va antes
    que abrasión porque un patrón de rayones paralelos también dispara
    densidad de bordes; al detectar rayado primero y excluir su máscara
    de abrasión evitamos que la clasifique erróneamente como abrasiva.
    """
    interior = _mascara_interior_pieza(seg.roi_mascara)
    cero = np.zeros_like(interior)

    map_c = mapa_contaminacion(seg.roi_rectificado_bgr, interior)
    map_r = mapa_rayado(seg.roi_rectificado_bgr, interior, map_c.mascara, cero)
    excluir_abras = cv2.bitwise_or(map_c.mascara, map_r.mascara)
    map_a = mapa_abrasion(seg.roi_rectificado_bgr, interior, excluir_abras)

    if is_debug():
        dump_debug_image(image_stem, "04a_interior_pieza", interior)
        dump_debug_image(image_stem, "04b_contaminacion_L_excedente", map_c.mapa_float)
        dump_debug_image(image_stem, "04b_contaminacion_mask", map_c.mascara)
        dump_debug_image(image_stem, "04c_abrasion_densidad_bordes", map_a.mapa_float)
        dump_debug_image(image_stem, "04c_abrasion_mask", map_a.mascara)
        dump_debug_image(image_stem, "04d_rayado_frangi", map_r.mapa_float)
        dump_debug_image(image_stem, "04d_rayado_mask", map_r.mascara)

    return ResultadoDeteccion(rayado=map_r, contaminacion=map_c, abrasion=map_a)
