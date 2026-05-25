"""
Bloque 7 — Orquestador y CLI.

Encadena los bloques 1–6 y expone el pipeline como herramienta de
línea de comandos:

    python -m src.main --input data/Hoja/Iluminacion45/img001.jpg --out output/
    python -m src.main --batch data/Hoja/Iluminacion45/         --out output/
    python -m src.main --input ... --debug                       --out output/

En modo --debug, cada bloque vuelca sus intermedios en
`<out>/debug/<image_stem>/` para inspección visual.

Filosofía: ningún bloque debe lanzar excepciones no controladas hacia
arriba. Si un bloque falla (p. ej. la pieza no se segmenta), el
orquestador captura la excepción y produce un reporte con
`estado="error_<bloque>"`, para que el sistema degrade elegantemente
incluso ante imágenes adversas.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import List

from . import config  # noqa: F401  (asegura carga temprana de constantes)
from .utils_logging import get_logger, set_debug
from .Adquisicion import cargar_imagen, AdquisicionError
from .Preprocesamiento import preprocesar
from .Segmentacion import segmentar_pieza, SegmentacionError
from .Deteccion import detectar_defectos
from .Clasificador import clasificar, ResultadoClasificacion
from .Salida import (
    construir_reporte_json,
    dibujar_overlay,
    escribir_csv_batch,
    escribir_json,
)

_log = get_logger("main")


EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def procesar_una(path_imagen: Path, out_dir: Path) -> dict:
    """Procesa una imagen end-to-end y devuelve el reporte JSON.

    Si algún bloque falla, devuelve un reporte mínimo con `estado="error_*"`.
    """
    stem = path_imagen.stem
    reporte: dict = {}
    try:
        img, meta = cargar_imagen(path_imagen)
    except AdquisicionError as e:
        _log.error("adquisición: %s", e)
        return _reporte_error(str(path_imagen), "error_adquisicion", str(e))

    try:
        img_pre, diag_pre = preprocesar(img, image_stem=stem)
    except Exception as e:  # degradar elegantemente
        _log.error("preprocesamiento: %s\n%s", e, traceback.format_exc())
        return _reporte_error(meta["filename"], "error_preprocesamiento", str(e))

    try:
        seg = segmentar_pieza(img_pre, image_stem=stem)
    except SegmentacionError as e:
        _log.error("segmentación: %s", e)
        rep = _reporte_error(meta["filename"], "error_segmentacion", str(e))
        rep["preprocesamiento"] = diag_pre
        return rep
    except Exception as e:
        _log.error("segmentación (no esperado): %s\n%s", e, traceback.format_exc())
        rep = _reporte_error(meta["filename"], "error_segmentacion", str(e))
        rep["preprocesamiento"] = diag_pre
        return rep

    try:
        det = detectar_defectos(seg, image_stem=stem)
    except Exception as e:
        _log.error("detección: %s\n%s", e, traceback.format_exc())
        rep = _reporte_error(meta["filename"], "error_deteccion", str(e))
        rep["preprocesamiento"] = diag_pre
        return rep

    try:
        clasif = clasificar(det)
    except Exception as e:
        _log.error("clasificación: %s\n%s", e, traceback.format_exc())
        clasif = ResultadoClasificacion(
            estado="error_clasificacion", clase=None, confianza=0.0,
            bbox=None, centroide=None, scores_por_clase=det.scores(),
            justificacion=str(e),
        )

    # Salida — overlay + JSON.
    overlay = dibujar_overlay(img, seg, clasif)
    overlay_path = out_dir / f"{stem}_overlay.png"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    cv2.imwrite(str(overlay_path), overlay)
    _log.info("overlay: %s", overlay_path)

    reporte = construir_reporte_json(meta, diag_pre, seg, det, clasif)
    reporte["salida"] = {"overlay_path": str(overlay_path.resolve())}
    json_path = out_dir / "reports" / f"{stem}.json"
    escribir_json(reporte, json_path)

    return reporte


def _reporte_error(filename: str, estado: str, mensaje: str) -> dict:
    return {
        "imagen": {"filename": filename},
        "clasificacion": {
            "estado": estado,
            "clase": None,
            "confianza": 0.0,
            "bbox_roi_xywh": None,
            "centroide_roi_xy": None,
            "scores_por_clase": {"rayado": 0.0, "contaminacion": 0.0, "abrasion": 0.0},
            "justificacion": mensaje,
        },
        "preprocesamiento": {
            "iluminacion": {"diagnostico": "n/a"},
            "ruido": {"es_impulsivo": False},
            "filtro_aplicado": {"filtro": "ninguno"},
        },
        "segmentacion": {"angulo_grados": 0.0},
    }


def procesar_batch(carpeta: Path, out_dir: Path) -> List[dict]:
    """Procesa todas las imágenes válidas de `carpeta` (recursivo)."""
    archivos = [p for p in sorted(carpeta.rglob("*"))
                if p.is_file() and p.suffix.lower() in EXTENSIONES_VALIDAS]
    _log.info("batch: %d imágenes en %s", len(archivos), carpeta)
    reportes: List[dict] = []
    for p in archivos:
        _log.info("─── %s ───", p.name)
        rep = procesar_una(p, out_dir)
        reportes.append(rep)
    csv_path = out_dir / "reports" / "batch.csv"
    escribir_csv_batch(reportes, csv_path)
    _log.info("CSV batch escrito: %s", csv_path)
    return reportes


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sistema de inspección superficial (SVI 2026-I, Tarea 2)."
    )
    grupo = p.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--input", type=str, help="Ruta a una imagen individual.")
    grupo.add_argument("--batch", type=str, help="Carpeta a procesar (recursivo).")
    p.add_argument("--out", type=str, default="output",
                   help="Carpeta de salida (por defecto: ./output).")
    p.add_argument("--debug", action="store_true",
                   help="Volcar imágenes intermedias por bloque.")
    return p


def main(argv: List[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        set_debug(True, base_dir=debug_dir)
        logging.getLogger().setLevel(logging.DEBUG)
        _log.info("modo --debug activo, intermedios en %s", debug_dir)
    else:
        set_debug(False)

    if args.input:
        path = Path(args.input)
        if not path.exists():
            _log.error("no existe: %s", path)
            return 2
        procesar_una(path, out_dir)
        return 0

    if args.batch:
        carpeta = Path(args.batch)
        if not carpeta.is_dir():
            _log.error("no es carpeta: %s", carpeta)
            return 2
        procesar_batch(carpeta, out_dir)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
