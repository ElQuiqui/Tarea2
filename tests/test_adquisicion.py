"""Tests del bloque 1 — Adquisición."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.Adquisicion import cargar_imagen, AdquisicionError


def test_cargar_devuelve_bgr_uint8(labels):
    img, meta = cargar_imagen(labels[0]["filepath"])
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
    assert meta["channels"] == 3
    assert meta["filename"].endswith((".jpg", ".jpeg", ".png", ".bmp"))


def test_archivo_inexistente_lanza(tmp_path: Path):
    with pytest.raises(AdquisicionError):
        cargar_imagen(tmp_path / "no_existe.jpg")
