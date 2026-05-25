"""Test end-to-end: imagen → reporte JSON sin excepciones."""
from __future__ import annotations

from pathlib import Path

from src.main import procesar_una


def test_pipeline_no_falla_y_genera_overlay(labels, tmp_path: Path):
    rep = procesar_una(labels[0]["filepath"], tmp_path)
    assert "clasificacion" in rep
    assert rep["clasificacion"]["estado"] in {
        "OK", "defectuoso", "desconocido",
        "error_segmentacion", "error_preprocesamiento",
        "error_deteccion", "error_clasificacion",
    }
    # Overlay generado salvo en error temprano de adquisición.
    if not rep["clasificacion"]["estado"].startswith("error_"):
        overlays = list(tmp_path.glob("*_overlay.png"))
        assert overlays, "no se generó overlay"
        jsons = list((tmp_path / "reports").glob("*.json"))
        assert jsons, "no se generó JSON"
