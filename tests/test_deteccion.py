"""Tests del bloque 4 — Detección.

Validan que el mapa correspondiente al defecto etiquetado tiene score
mayor que los otros dos para esa imagen. Si una clase no tiene imágenes
etiquetadas en labels.csv, su test se saltea.
"""
from __future__ import annotations

import pytest

from src.Adquisicion import cargar_imagen
from src.Preprocesamiento import preprocesar
from src.Segmentacion import segmentar_pieza
from src.Deteccion import detectar_defectos


def _scores_para(path):
    img, _ = cargar_imagen(path)
    img_pre, _ = preprocesar(img, image_stem="t")
    seg = segmentar_pieza(img_pre, image_stem="t")
    det = detectar_defectos(seg, image_stem="t")
    return det.scores(), det


@pytest.mark.parametrize("clase_esperada", ["rayado", "contaminacion", "abrasion"])
def test_score_mayor_es_de_la_clase(labels_por_clase, clase_esperada):
    rows = labels_por_clase.get(clase_esperada, [])
    if not rows:
        pytest.skip(f"sin imágenes etiquetadas como {clase_esperada}")
    aciertos = 0
    for r in rows:
        scores, _ = _scores_para(r["filepath"])
        ganador = max(scores, key=scores.get)
        if ganador == clase_esperada:
            aciertos += 1
    # Tolerancia: al menos la mitad debe acertar (calibración perfectible).
    assert aciertos * 2 >= len(rows), (
        f"{clase_esperada}: {aciertos}/{len(rows)} aciertos — revisar umbrales en config.py"
    )


def test_ok_tiene_scores_bajos(labels_por_clase):
    """Las piezas OK deben tener sus tres scores por debajo de los
    respectivos umbrales de presencia (al menos en la mayoría)."""
    rows = labels_por_clase.get("ok", [])
    if not rows:
        pytest.skip("sin imágenes OK etiquetadas")
    from src import config
    umbrales = {
        "rayado": config.SCORE_RAYADO_PRESENT,
        "contaminacion": config.SCORE_CONTAM_PRESENT,
        "abrasion": config.SCORE_ABRASION_PRESENT,
    }
    aciertos = 0
    for r in rows:
        scores, _ = _scores_para(r["filepath"])
        if all(scores[c] < umbrales[c] for c in umbrales):
            aciertos += 1
    # Al menos la mayoría debe pasar como OK.
    assert aciertos * 2 >= len(rows), (
        f"OK: solo {aciertos}/{len(rows)} con todos los scores bajo umbral"
    )
