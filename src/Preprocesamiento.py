"""
Bloque 2 — Preprocesamiento.

Responsabilidades:
1) Diagnosticar las condiciones de iluminación a partir del histograma
   de la imagen y, si son inadecuadas, corregirlas con CLAHE.
2) Estimar el ruido y aplicar filtro de mediana si el ruido es de tipo
   impulsivo (a lo que la mediana es óptima).

Fundamento teórico:
- Histograma: distribución de niveles de gris/luminancia. Una imagen
  subexpuesta concentra masa en niveles bajos; sobreexpuesta en altos;
  contraste pobre tiene baja desviación estándar. Estos diagnósticos
  son numéricamente simples e interpretables.
- CLAHE (Contrast Limited Adaptive Histogram Equalization, Zuiderveld
  1994): ecualización por tiles que limita la amplificación de ruido.
  Se aplica sobre el canal L de Lab para preservar la cromaticidad.
- Filtro mediana: óptimo bajo ruido impulsivo (sal y pimienta).
  Aplicar mediana cuando el ruido es gaussiano suave es contraproducente
  porque también suaviza estructuras finas (rayados). Por eso se gatilla
  sólo si el estimador (MAD del Laplaciano) supera un umbral.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from scipy.stats import kurtosis

from . import config
from .utils_logging import get_logger, dump_debug_image, dump_debug_plot, is_debug

_log = get_logger("preprocesamiento")


# ─── Análisis de iluminación ────────────────────────────────────────────────

def analizar_iluminacion(img_bgr: np.ndarray) -> dict:
    """Calcula histograma y diagnostica iluminación.

    Bloque: 2 (Preprocesamiento).

    Fundamento: el histograma del canal de luminancia resume globalmente
    la distribución tonal de la escena. Estadísticos sencillos (media,
    desviación, % de píxeles oscuros/claros) bastan para discriminar
    sub/sobreexposición y bajo contraste.

    Parameters
    ----------
    img_bgr : np.ndarray
        Imagen BGR uint8.

    Returns
    -------
    dict
        {diagnostico, media, std, pct_oscuros, pct_claros, requiere_correccion,
         histograma_L (lista de 256 enteros)}.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    hist = cv2.calcHist([L], [0], None, [config.HIST_BINS], [0, 256]).flatten()
    total = float(L.size)

    media = float(L.mean())
    std = float(L.std())
    pct_oscuros = float((L <= config.EXPOSURE_DARK_THRESH).sum()) / total * 100.0
    pct_claros = float((L >= config.EXPOSURE_BRIGHT_THRESH).sum()) / total * 100.0

    diagnostico = "ok"
    if pct_oscuros > config.EXPOSURE_LOW_PCT and media < config.MEAN_OK_MIN:
        diagnostico = "subexpuesta"
    elif pct_claros > config.EXPOSURE_HIGH_PCT and media > config.MEAN_OK_MAX:
        diagnostico = "sobreexpuesta"
    elif std < config.STD_OK_MIN:
        diagnostico = "contraste_bajo"

    requiere_correccion = diagnostico != "ok"

    out = {
        "diagnostico": diagnostico,
        "media": round(media, 2),
        "std": round(std, 2),
        "pct_oscuros": round(pct_oscuros, 2),
        "pct_claros": round(pct_claros, 2),
        "requiere_correccion": requiere_correccion,
        "histograma_L": hist.astype(int).tolist(),
    }
    _log.info(
        "iluminación: %s (media=%.1f std=%.1f oscuros=%.1f%% claros=%.1f%%)",
        diagnostico, media, std, pct_oscuros, pct_claros,
    )
    return out


def corregir_iluminacion(img_bgr: np.ndarray, diag: dict) -> np.ndarray:
    """Aplica CLAHE sobre el canal L de Lab si el diagnóstico lo indica.

    Bloque: 2. Fundamento: CLAHE evita amplificar ruido típica de una
    ecualización global; trabajar en Lab preserva la cromaticidad.

    Si `diag["requiere_correccion"]` es False, devuelve la imagen original.
    """
    if not diag.get("requiere_correccion", False):
        return img_bgr.copy()

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID,
    )
    L_eq = clahe.apply(L)
    lab_eq = cv2.merge((L_eq, a, b))
    out = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    _log.info("CLAHE aplicado (clip=%.1f tile=%s)",
              config.CLAHE_CLIP_LIMIT, config.CLAHE_TILE_GRID)
    return out


# ─── Estimación y reducción de ruido ────────────────────────────────────────

def estimar_ruido(img_bgr: np.ndarray) -> dict:
    """Estima sigma de ruido y detecta carácter impulsivo.

    Bloque: 2. Fundamento: la magnitud del Laplaciano de una imagen sin
    ruido es pequeña en zonas planas; el MAD (Median Absolute Deviation)
    del Laplaciano es un estimador robusto del ruido. Una kurtosis alta
    sugiere distribución con colas pesadas, típica de ruido impulsivo
    (sal y pimienta).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    mad = float(np.median(np.abs(lap - np.median(lap))))
    # convertir MAD a sigma equivalente normal: sigma ≈ MAD / 0.6745
    sigma_est = mad / 0.6745
    kurt = float(kurtosis(lap.ravel()))
    # Heurística: impulsivo si MAD alto y kurtosis alta.
    impulsivo = (mad > config.NOISE_MAD_THRESHOLD) and (kurt > 3.0)

    out = {
        "mad_laplaciano": round(mad, 3),
        "sigma_estimado": round(sigma_est, 3),
        "kurtosis": round(kurt, 3),
        "es_impulsivo": bool(impulsivo),
    }
    _log.info("ruido: MAD=%.2f sigma≈%.2f kurt=%.2f impulsivo=%s",
              mad, sigma_est, kurt, impulsivo)
    return out


def reducir_ruido(img_bgr: np.ndarray, diag_ruido: dict) -> Tuple[np.ndarray, dict]:
    """Aplica filtro de mediana si el ruido detectado es impulsivo.

    Bloque: 2. Fundamento: la mediana es un estimador robusto, óptimo
    bajo ruido impulsivo, y preserva bordes mejor que un gaussiano.
    Para ruido gaussiano leve no se aplica nada (evitamos suavizar
    rayados finos que detectaremos luego).
    """
    info = {"filtro": "ninguno", "kernel": 0}
    if not diag_ruido.get("es_impulsivo", False):
        _log.info("no se aplica filtro (ruido no impulsivo)")
        return img_bgr.copy(), info

    k = config.MEDIAN_KERNEL
    if diag_ruido.get("mad_laplaciano", 0) > config.NOISE_HIGH_THRESHOLD:
        k = config.MEDIAN_KERNEL_HIGH
    out = cv2.medianBlur(img_bgr, k)
    info = {"filtro": "mediana", "kernel": k}
    _log.info("mediana aplicada (kernel=%d)", k)
    return out, info


# ─── Pipeline compuesto ─────────────────────────────────────────────────────

def preprocesar(img_bgr: np.ndarray, image_stem: str = "frame") -> Tuple[np.ndarray, dict]:
    """Compone los pasos del bloque 2 y devuelve imagen + diagnóstico unificado.

    Bloque: 2 (Preprocesamiento).
    """
    diag_il = analizar_iluminacion(img_bgr)
    img_il = corregir_iluminacion(img_bgr, diag_il)
    diag_n = estimar_ruido(img_il)
    img_out, info_filtro = reducir_ruido(img_il, diag_n)

    if is_debug():
        dump_debug_image(image_stem, "02a_imagen_corregida_clahe", img_il)
        dump_debug_image(image_stem, "02b_imagen_filtrada", img_out)
        _dump_histograma(image_stem, img_bgr, img_out)

    diag = {
        "iluminacion": diag_il,
        "ruido": diag_n,
        "filtro_aplicado": info_filtro,
    }
    return img_out, diag


def _dump_histograma(image_stem: str, antes_bgr: np.ndarray, despues_bgr: np.ndarray) -> None:
    """Guarda comparación de histogramas antes/después en modo debug."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    L_antes = cv2.cvtColor(antes_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    L_desp = cv2.cvtColor(despues_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].hist(L_antes.ravel(), bins=256, range=(0, 256), color="steelblue")
    axes[0].set_title("Histograma L (entrada)")
    axes[1].hist(L_desp.ravel(), bins=256, range=(0, 256), color="seagreen")
    axes[1].set_title("Histograma L (preprocesada)")
    for ax in axes:
        ax.set_xlabel("nivel")
        ax.set_ylabel("frecuencia")
    fig.tight_layout()
    dump_debug_plot(image_stem, "02c_histogramas", fig)
    plt.close(fig)
