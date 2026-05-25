"""Tests del bloque 2 — Preprocesamiento."""
from __future__ import annotations

import numpy as np

from src.Adquisicion import cargar_imagen
from src.Preprocesamiento import (
    analizar_iluminacion,
    corregir_iluminacion,
    estimar_ruido,
    preprocesar,
)


def test_analizar_iluminacion_emite_diagnostico(labels):
    img, _ = cargar_imagen(labels[0]["filepath"])
    diag = analizar_iluminacion(img)
    assert diag["diagnostico"] in {"ok", "subexpuesta", "sobreexpuesta", "contraste_bajo"}
    assert 0 <= diag["pct_oscuros"] <= 100
    assert 0 <= diag["pct_claros"] <= 100
    assert len(diag["histograma_L"]) == 256


def test_clahe_no_cambia_dimensiones(labels):
    img, _ = cargar_imagen(labels[0]["filepath"])
    diag = analizar_iluminacion(img)
    # Forzamos la corrección para verificar.
    diag["requiere_correccion"] = True
    out = corregir_iluminacion(img, diag)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_pipeline_preprocesar_devuelve_misma_forma(labels):
    img, _ = cargar_imagen(labels[0]["filepath"])
    out, diag = preprocesar(img, image_stem="t")
    assert out.shape == img.shape
    assert "iluminacion" in diag and "ruido" in diag and "filtro_aplicado" in diag


def test_estimar_ruido_keys(labels):
    img, _ = cargar_imagen(labels[0]["filepath"])
    diag = estimar_ruido(img)
    for k in ("mad_laplaciano", "sigma_estimado", "kurtosis", "es_impulsivo"):
        assert k in diag
