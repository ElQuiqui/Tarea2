"""Tests del bloque 3 — Segmentación."""
from __future__ import annotations

import numpy as np
import pytest

from src.Adquisicion import cargar_imagen
from src.Preprocesamiento import preprocesar
from src.Segmentacion import segmentar_pieza, SegmentacionError


def _segmentar(path):
    img, _ = cargar_imagen(path)
    img_pre, _ = preprocesar(img, image_stem="t")
    return img, segmentar_pieza(img_pre, image_stem="t")


def test_segmenta_blob_central(labels):
    """La pieza segmentada debe tener centroide cerca del centro de la imagen."""
    for r in labels[:3]:
        img, seg = _segmentar(r["filepath"])
        H, W = img.shape[:2]
        cx, cy = seg.centroide
        # Debe estar en el 50% central.
        assert W * 0.25 <= cx <= W * 0.75, f"cx fuera de rango en {r['filepath'].name}"
        assert H * 0.25 <= cy <= H * 0.75, f"cy fuera de rango en {r['filepath'].name}"


def test_roi_rectificado_no_vacio(labels):
    _, seg = _segmentar(labels[0]["filepath"])
    assert seg.roi_rectificado_bgr.ndim == 3
    assert seg.roi_rectificado_bgr.shape[2] == 3
    assert seg.roi_mascara.max() > 0


def test_warp_inverso_es_inverso(labels):
    """Aplicar M_warp y luego M_warp_inv a un punto debe devolverlo."""
    import cv2
    _, seg = _segmentar(labels[0]["filepath"])
    pts = np.array([[100, 100]], dtype=np.float32).reshape(-1, 1, 2)
    pts_warp = cv2.transform(pts, seg.M_warp)
    pts_back = cv2.transform(pts_warp, seg.M_warp_inv)
    assert np.allclose(pts.reshape(-1, 2), pts_back.reshape(-1, 2), atol=0.5)
