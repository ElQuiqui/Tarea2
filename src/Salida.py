"""
Bloque 6 — Salida y reporte.

Produce:
- Imagen anotada (overlay del contorno de la pieza, bbox del defecto, etiqueta).
- Log JSON estructurado por imagen, con todas las decisiones intermedias.
- CSV consolidado para el modo batch.

El overlay se dibuja sobre la imagen original (no el ROI rectificado).
El bbox detectado en el ROI se proyecta de regreso usando la matriz
inversa del warp afín, así el reporte visual coincide con lo que un
operador vería en la cámara.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

from . import config
from .utils_logging import get_logger
from .Segmentacion import ResultadoSegmentacion
from .Clasificador import ResultadoClasificacion

_log = get_logger("salida")


# ─── Overlay ────────────────────────────────────────────────────────────────

def _proyectar_bbox_a_original(
    bbox_roi: Tuple[int, int, int, int],
    M_inv: np.ndarray,
) -> np.ndarray:
    """Mapea los 4 vértices del bbox del ROI a la imagen original.

    Devuelve un polígono (4,2) int32 para drawContours.
    """
    x, y, w, h = bbox_roi
    pts = np.array([
        [x,     y    ],
        [x + w, y    ],
        [x + w, y + h],
        [x,     y + h],
    ], dtype=np.float32).reshape(-1, 1, 2)
    pts_t = cv2.transform(pts, M_inv)
    return pts_t.reshape(-1, 2).astype(np.int32)


def dibujar_overlay(
    img_bgr: np.ndarray,
    seg: ResultadoSegmentacion,
    clasif: ResultadoClasificacion,
) -> np.ndarray:
    """Genera la imagen anotada con contorno + bbox + etiqueta + confianza.

    Bloque: 6.
    """
    out = img_bgr.copy()
    # Contorno de la pieza (verde).
    cv2.drawContours(out, [seg.contorno], -1,
                     config.OVERLAY_CONTOUR_COLOR, config.OVERLAY_THICKNESS)

    etiqueta = clasif.estado
    if clasif.clase:
        etiqueta = f"{clasif.clase.upper()} ({clasif.confianza:.2f})"
    elif clasif.estado == "OK":
        etiqueta = "OK"
    else:
        etiqueta = f"{clasif.estado.upper()} ({clasif.confianza:.2f})"

    # Bbox del defecto (rojo) proyectado a la imagen original.
    if clasif.bbox is not None:
        poly = _proyectar_bbox_a_original(clasif.bbox, seg.M_warp_inv)
        cv2.polylines(out, [poly], isClosed=True,
                      color=config.OVERLAY_BBOX_COLOR,
                      thickness=config.OVERLAY_THICKNESS)

    # Etiqueta de texto cerca del centroide de la pieza.
    cx, cy = int(seg.centroide[0]), int(seg.centroide[1])
    _texto_con_fondo(out, etiqueta, (cx - 80, max(20, cy - 20)))
    return out


def _texto_con_fondo(img: np.ndarray, text: str, org: Tuple[int, int]) -> None:
    """Dibuja texto sobre un rectángulo opaco para mejor legibilidad."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, config.OVERLAY_FONT_SCALE,
                                         config.OVERLAY_THICKNESS)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 2, y + baseline + 2),
                  (0, 0, 0), thickness=-1)
    cv2.putText(img, text, (x, y), font, config.OVERLAY_FONT_SCALE,
                config.OVERLAY_TEXT_COLOR, config.OVERLAY_THICKNESS, cv2.LINE_AA)


# ─── JSON por imagen ────────────────────────────────────────────────────────

def construir_reporte_json(
    meta: dict,
    diag_pre: dict,
    seg: ResultadoSegmentacion,
    det,                      # ResultadoDeteccion (importado debajo en main)
    clasif: ResultadoClasificacion,
) -> dict:
    """Reúne todas las decisiones intermedias en un dict JSON-serializable.

    Bloque: 6. Todas las claves se documentan sólo aquí; los otros bloques
    devuelven dicts ya en este formato.
    """
    return {
        "imagen": meta,
        "preprocesamiento": diag_pre,
        "segmentacion": {
            "centroide_xy": [round(seg.centroide[0], 1), round(seg.centroide[1], 1)],
            "angulo_grados": round(seg.angulo_grados, 2),
            "rect_rotado": _rect_to_dict(seg.rect_rotado),
            "roi_size_wh": list(seg.roi_size),
        },
        "deteccion": {
            "scores": det.scores(),
            "blobs": {
                "rayado":        [_blob_to_dict(b) for b in det.rayado.blobs],
                "contaminacion": [_blob_to_dict(b) for b in det.contaminacion.blobs],
                "abrasion":      [_blob_to_dict(b) for b in det.abrasion.blobs],
            },
        },
        "clasificacion": {
            "estado": clasif.estado,
            "clase": clasif.clase,
            "confianza": clasif.confianza,
            "bbox_roi_xywh": list(clasif.bbox) if clasif.bbox else None,
            "centroide_roi_xy": (
                [round(clasif.centroide[0], 1), round(clasif.centroide[1], 1)]
                if clasif.centroide else None
            ),
            "scores_por_clase": {k: round(v, 4) for k, v in clasif.scores_por_clase.items()},
            "justificacion": clasif.justificacion,
        },
    }


def _rect_to_dict(rect: Tuple) -> dict:
    (cx, cy), (w, h), ang = rect
    return {
        "centro": [round(cx, 1), round(cy, 1)],
        "tamaño": [round(w, 1), round(h, 1)],
        "angulo": round(ang, 2),
    }


def _blob_to_dict(b) -> dict:
    return {
        "bbox_xywh": list(b.bbox),
        "centroide": [round(b.centroide[0], 1), round(b.centroide[1], 1)],
        "area": b.area,
        "excentricidad": round(b.eccentricidad, 3),
        "delta_e_medio": round(b.delta_e_medio, 2),
    }


def escribir_json(reporte: dict, out_path: Path) -> None:
    """Bloque: 6. Vuelca el reporte como JSON indentado."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=config.JSON_INDENT, ensure_ascii=False)
    _log.info("reporte JSON: %s", out_path)


# ─── CSV batch ──────────────────────────────────────────────────────────────

CSV_HEADER = [
    "filename", "estado", "clase", "confianza",
    "score_rayado", "score_contaminacion", "score_abrasion",
    "iluminacion_diag", "ruido_es_impulsivo", "filtro",
    "angulo_pieza", "bbox_roi", "centroide_roi",
]


def escribir_csv_batch(reportes: Iterable[dict], out_path: Path) -> None:
    """Bloque: 6. Genera un CSV con una fila por imagen procesada."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in reportes:
            cls = r["clasificacion"]
            scores = cls["scores_por_clase"]
            pre = r["preprocesamiento"]
            seg = r["segmentacion"]
            w.writerow([
                r["imagen"]["filename"],
                cls["estado"],
                cls["clase"] or "",
                cls["confianza"],
                scores.get("rayado", 0),
                scores.get("contaminacion", 0),
                scores.get("abrasion", 0),
                pre["iluminacion"]["diagnostico"],
                pre["ruido"]["es_impulsivo"],
                pre["filtro_aplicado"]["filtro"],
                seg["angulo_grados"],
                cls["bbox_roi_xywh"] if cls["bbox_roi_xywh"] else "",
                cls["centroide_roi_xy"] if cls["centroide_roi_xy"] else "",
            ])
    _log.info("CSV batch: %s (%d filas)", out_path, sum(1 for _ in reportes) if False else "?")
