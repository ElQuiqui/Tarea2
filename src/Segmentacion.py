"""
Bloque 3 — Segmentación pieza / fondo + rectificación.

Responsabilidades:
1) Separar la pieza (cuadrado de PLA negro) del fondo (hoja o tela blanca),
   robusto a posición y rotación arbitrarias.
2) Calcular contorno, rectángulo rotado mínimo y ángulo.
3) Rectificar: warp afín que alinea la pieza al eje. Todas las etapas
   posteriores trabajan sobre este ROI, ganando invariancia a posición
   y rotación.

Fundamento teórico:
- Espacio Lab: separa luminancia (L) de cromaticidad (a*, b*). Como la
  pieza es oscura y el fondo claro, el canal L invertido seguido de Otsu
  produce una binarización limpia incluso con iluminación no uniforme.
- Otsu (1979): umbral que minimiza la varianza intra-clase. Funciona muy
  bien cuando el histograma es bimodal, como pieza-oscura/fondo-claro.
- Apertura morfológica (erosión + dilatación): elimina ruido de textura
  (granos de la tela) sin alterar el contorno principal.
- Cierre morfológico: rellena huecos pequeños dentro del blob de la pieza.
- minAreaRect: rectángulo de área mínima orientado, da el ángulo de la
  pieza necesario para la rectificación.
- Warp afín: rotación + traslación que mapea el rect orientado a un rect
  alineado con los ejes (invariancia geométrica).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu

from . import config
from .utils_logging import get_logger, dump_debug_image, is_debug

_log = get_logger("segmentacion")


class SegmentacionError(Exception):
    """No se pudo segmentar una pieza válida."""


@dataclass
class ResultadoSegmentacion:
    """Resultado del bloque 3.

    Attributes
    ----------
    mascara_pieza : np.ndarray
        Máscara binaria uint8 (0/255) en coordenadas de la imagen original.
    contorno : np.ndarray
        Contorno (Nx1x2) en coordenadas de la imagen original.
    centroide : tuple[float, float]
        (cx, cy) en la imagen original.
    rect_rotado : tuple
        ((cx, cy), (w, h), angle) — salida de cv2.minAreaRect.
    angulo_grados : float
        Ángulo de rotación de la pieza (de minAreaRect, normalizado).
    roi_rectificado_bgr : np.ndarray
        BGR uint8, pieza alineada al eje.
    roi_mascara : np.ndarray
        Máscara binaria de la pieza dentro del ROI rectificado.
    M_warp : np.ndarray
        Matriz afín 2x3 que mapea imagen original → ROI rectificado.
    M_warp_inv : np.ndarray
        Matriz inversa: ROI rectificado → imagen original (para overlay).
    roi_size : tuple[int, int]
        (W, H) del ROI rectificado.
    """
    mascara_pieza: np.ndarray
    contorno: np.ndarray
    centroide: Tuple[float, float]
    rect_rotado: Tuple
    angulo_grados: float
    roi_rectificado_bgr: np.ndarray
    roi_mascara: np.ndarray
    M_warp: np.ndarray
    M_warp_inv: np.ndarray
    roi_size: Tuple[int, int]


def _binarizar_pieza(img_bgr: np.ndarray) -> np.ndarray:
    """Binarización por multi-Otsu sobre L: pieza = clase más oscura.

    Bloque: 3. Fundamento: bajo iluminación lateral, el histograma de L
    tiene tres modos — pieza muy oscura, sombra rasante intermedia, y
    fondo claro. Otsu de 2 clases agrupa pieza+sombra; multi-Otsu de 3
    clases (Liao et al. 2001) las separa correctamente. Si la imagen
    sólo tiene dos modos reales, multi-Otsu degenera a Otsu y aún así
    deja a la pieza en la clase más oscura.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    try:
        umbrales = threshold_multiotsu(L, classes=3)
        bin_otsu = (L < umbrales[0]).astype(np.uint8) * 255
    except ValueError:
        # Histograma sin suficiente variación para 3 clases — fallback a Otsu 2-clases.
        _, bin_otsu = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.MORPH_OPEN_KERNEL, config.MORPH_OPEN_KERNEL)
    )
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.MORPH_CLOSE_KERNEL, config.MORPH_CLOSE_KERNEL)
    )
    abierto = cv2.morphologyEx(bin_otsu, cv2.MORPH_OPEN, k_open)
    cerrado = cv2.morphologyEx(abierto, cv2.MORPH_CLOSE, k_close)
    return cerrado


def _seleccionar_componente_central(bin_img: np.ndarray) -> Optional[np.ndarray]:
    """Elige el componente conectado más grande cuya bbox toca el cuadrante central.

    Devuelve una máscara binaria sólo con ese componente, o None si no hay.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    H, W = bin_img.shape
    area_total = float(H * W)
    cx_img, cy_img = W / 2.0, H / 2.0
    tol_x = W * config.CENTER_TOLERANCE_RATIO
    tol_y = H * config.CENTER_TOLERANCE_RATIO

    mejor_id, mejor_area = -1, 0
    for i in range(1, n):  # 0 = fondo
        x, y, w, h, area = stats[i]
        if area / area_total < config.MIN_PIECE_AREA_RATIO:
            continue
        # bbox debe contener al cuadrado central definido por tol_x, tol_y
        toca_centro = (
            x <= cx_img + tol_x and x + w >= cx_img - tol_x
            and y <= cy_img + tol_y and y + h >= cy_img - tol_y
        )
        if not toca_centro:
            continue
        if area > mejor_area:
            mejor_area = area
            mejor_id = i

    if mejor_id < 0:
        return None
    mask = (labels == mejor_id).astype(np.uint8) * 255
    return mask


def _rectificar_por_rect_rotado(
    img_bgr: np.ndarray,
    mascara: np.ndarray,
    rect: Tuple,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int], float]:
    """Aplica warp afín que alinea el rect rotado a los ejes.

    Returns
    -------
    roi_bgr, roi_mascara, M, M_inv, (W, H), angle_normalizado
    """
    (cx, cy), (w, h), angle = rect
    # cv2.minAreaRect entrega ángulo en (-90, 0]. Normalizamos para que el
    # lado largo quede horizontal.
    if w < h:
        w, h = h, w
        angle += 90.0
    angle_norm = float(angle)

    # Tamaño del ROI: escalar al lado mayor RECTIFIED_SIZE_MAX.
    long_side = max(w, h)
    scale = config.RECTIFIED_SIZE_MAX / max(1.0, long_side)
    out_w = int(round(w * scale)) + 2 * config.RECTIFIED_PADDING_PX
    out_h = int(round(h * scale)) + 2 * config.RECTIFIED_PADDING_PX

    # Matriz afín: rotación alrededor de (cx, cy) por -angle,
    # luego escala, luego traslación para centrar en (out_w/2, out_h/2).
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    # Trasladar el centro rotado al centro del ROI de salida:
    M_rot[0, 2] += (out_w / 2.0) - cx
    M_rot[1, 2] += (out_h / 2.0) - cy

    roi_bgr = cv2.warpAffine(
        img_bgr, M_rot, (out_w, out_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    roi_mascara = cv2.warpAffine(
        mascara, M_rot, (out_w, out_h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Inversa de la afín (2x3 → 3x3 → invertir → 2x3).
    M_inv = cv2.invertAffineTransform(M_rot)
    return roi_bgr, roi_mascara, M_rot, M_inv, (out_w, out_h), angle_norm


def segmentar_pieza(img_bgr: np.ndarray, image_stem: str = "frame") -> ResultadoSegmentacion:
    """Segmenta la pieza y la rectifica.

    Bloque: 3 (Segmentación). Junta binarización, morfología, selección
    del blob central y warp afín en un único punto de entrada.

    Raises
    ------
    SegmentacionError
        Si no se encuentra un blob válido.
    """
    bin_img = _binarizar_pieza(img_bgr)
    if is_debug():
        dump_debug_image(image_stem, "03a_binarizacion_otsu", bin_img)

    mascara = _seleccionar_componente_central(bin_img)
    if mascara is None:
        raise SegmentacionError(
            "No se encontró un componente conectado central con área mínima."
        )

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        raise SegmentacionError("Máscara sin contorno externo.")
    contorno = max(contornos, key=cv2.contourArea)
    # Convex hull: la pieza es un cuadrado (convexo por diseño). Esto cubre
    # tanto huecos cerrados como concavidades que la binarización pueda dejar
    # cuando una contaminación muy brillante (pegamento, marca) cae en la
    # clase "fondo" de multi-Otsu y "muerde" la silueta. Sin esto la
    # contaminación queda fuera del ROI y los detectores no la ven.
    hull = cv2.convexHull(contorno)
    mascara_llena = np.zeros_like(mascara)
    cv2.drawContours(mascara_llena, [hull], -1, 255, thickness=cv2.FILLED)
    # Usamos el hull también para minAreaRect: si la silueta original tiene
    # una concavidad grande, el rect del hull es más estable.
    contorno = hull

    M = cv2.moments(mascara_llena, binaryImage=True)
    if M["m00"] == 0:
        raise SegmentacionError("Momento m00 cero — máscara vacía.")
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    rect = cv2.minAreaRect(contorno)  # ((cx,cy),(w,h),angle)
    roi_bgr, roi_mask, M_warp, M_inv, roi_size, angle_norm = _rectificar_por_rect_rotado(
        img_bgr, mascara_llena, rect
    )

    if is_debug():
        dump_debug_image(image_stem, "03b_mascara_pieza", mascara_llena)
        dbg = img_bgr.copy()
        cv2.drawContours(dbg, [contorno], -1, (0, 255, 0), 2)
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(dbg, [box], -1, (0, 0, 255), 2)
        dump_debug_image(image_stem, "03c_contorno_y_rect", dbg)
        dump_debug_image(image_stem, "03d_roi_rectificado", roi_bgr)
        dump_debug_image(image_stem, "03e_roi_mascara", roi_mask)

    _log.info(
        "pieza segmentada: centroide=(%.1f,%.1f) angulo=%.1f° ROI=%dx%d",
        cx, cy, angle_norm, roi_size[0], roi_size[1],
    )
    return ResultadoSegmentacion(
        mascara_pieza=mascara_llena,
        contorno=contorno,
        centroide=(float(cx), float(cy)),
        rect_rotado=rect,
        angulo_grados=angle_norm,
        roi_rectificado_bgr=roi_bgr,
        roi_mascara=roi_mask,
        M_warp=M_warp,
        M_warp_inv=M_inv,
        roi_size=roi_size,
    )
