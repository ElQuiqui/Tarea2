"""Tests del bloque 5 — Clasificador (unitarios sin dependencia de imágenes).

Validan la nueva lógica con umbrales por clase y jerarquía:
contaminación > abrasión > rayado.
"""
from __future__ import annotations

import numpy as np

from src import config
from src.Clasificador import clasificar
from src.Deteccion import ResultadoDeteccion, MapaDefecto, BlobInfo


def _mk_mapa(clase, score, blobs=None):
    return MapaDefecto(
        clase=clase,
        mapa_float=np.zeros((10, 10), dtype=np.float32),
        mascara=np.zeros((10, 10), dtype=np.uint8),
        score=score,
        blobs=blobs or [],
    )


def test_ok_cuando_ningun_score_supera_su_umbral():
    """Todos los scores muy por debajo del umbral de presencia → OK."""
    det = ResultadoDeteccion(
        rayado=_mk_mapa("rayado", 0.0),
        contaminacion=_mk_mapa("contaminacion", 0.0),
        abrasion=_mk_mapa("abrasion", 0.0),
    )
    r = clasificar(det)
    assert r.estado == "OK"
    assert r.clase is None


def test_rayado_solo_si_es_la_unica_clase_activa():
    """Score rayado claramente sobre su umbral, contam/abrasión por debajo
    de los suyos → clase = rayado."""
    blob = BlobInfo(bbox=(1, 1, 20, 2), centroide=(11, 2), area=40, eccentricidad=0.98)
    det = ResultadoDeteccion(
        rayado=_mk_mapa("rayado", config.SCORE_RAYADO_PRESENT * 5, [blob]),
        contaminacion=_mk_mapa("contaminacion", config.SCORE_CONTAM_PRESENT * 0.1),
        abrasion=_mk_mapa("abrasion", config.SCORE_ABRASION_PRESENT * 0.1),
    )
    r = clasificar(det)
    assert r.estado == "defectuoso"
    assert r.clase == "rayado"
    assert r.bbox == (1, 1, 20, 2)


def test_contaminacion_gana_a_otros_por_jerarquia():
    """Si contaminación y rayado están activos, gana contaminación
    (jerarquía: contam > abrasión > rayado)."""
    blob_c = BlobInfo(bbox=(5, 5, 30, 30), centroide=(20, 20), area=900, eccentricidad=0.3)
    blob_r = BlobInfo(bbox=(1, 1, 20, 2), centroide=(11, 2), area=40, eccentricidad=0.98)
    det = ResultadoDeteccion(
        rayado=_mk_mapa("rayado", config.SCORE_RAYADO_PRESENT * 5, [blob_r]),
        contaminacion=_mk_mapa("contaminacion", config.SCORE_CONTAM_PRESENT * 5, [blob_c]),
        abrasion=_mk_mapa("abrasion", 0.0),
    )
    r = clasificar(det)
    assert r.estado == "defectuoso"
    assert r.clase == "contaminacion"


def test_confianza_aumenta_con_score():
    """Score muy por encima del umbral → confianza alta."""
    blob = BlobInfo(bbox=(5, 5, 50, 50), centroide=(30, 30), area=2500, eccentricidad=0.3)
    det = ResultadoDeteccion(
        rayado=_mk_mapa("rayado", 0.0),
        contaminacion=_mk_mapa("contaminacion", config.SCORE_CONTAM_PRESENT * 10, [blob]),
        abrasion=_mk_mapa("abrasion", 0.0),
    )
    r = clasificar(det)
    assert r.estado == "defectuoso"
    assert r.clase == "contaminacion"
    assert r.confianza >= 0.8
