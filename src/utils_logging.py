"""
Utilidades de logging y dump de imágenes intermedias para modo --debug.

Centraliza la creación de loggers por bloque y permite escribir imágenes
de depuración en una carpeta por imagen procesada, sin contaminar los
módulos de procesamiento con código de I/O.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config


_LOGGERS: dict[str, logging.Logger] = {}
_DEBUG_DIR: Optional[Path] = None
_DEBUG_ENABLED: bool = False


def get_logger(name: str) -> logging.Logger:
    """Devuelve (o crea) un logger por nombre con formato uniforme.

    Bloque: transversal. Usado por todos los módulos para evitar `print`.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATEFMT)
        )
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)
    _LOGGERS[name] = logger
    return logger


def set_debug(enabled: bool, base_dir: Optional[Path] = None) -> None:
    """Activa el dump de imágenes intermedias y eleva loggers a DEBUG.

    `base_dir` apunta a la carpeta donde se crearán las subcarpetas por
    imagen (típicamente `output/debug/`).
    """
    global _DEBUG_ENABLED, _DEBUG_DIR
    _DEBUG_ENABLED = enabled
    _DEBUG_DIR = Path(base_dir) if base_dir else None
    level = logging.DEBUG if enabled else logging.INFO
    for lg in _LOGGERS.values():
        lg.setLevel(level)


def is_debug() -> bool:
    return _DEBUG_ENABLED


def _debug_subdir(image_stem: str) -> Optional[Path]:
    if not _DEBUG_ENABLED or _DEBUG_DIR is None:
        return None
    sub = _DEBUG_DIR / image_stem
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def dump_debug_image(image_stem: str, stage: str, img: np.ndarray) -> None:
    """Guarda una imagen intermedia en `output/debug/<stem>/<stage>.png`.

    Acepta imágenes uint8 BGR/gris o float [0,1]: las normaliza para
    visualización sin afectar la imagen en memoria.
    """
    sub = _debug_subdir(image_stem)
    if sub is None:
        return
    if img.dtype != np.uint8:
        finite = img[np.isfinite(img)]
        if finite.size == 0:
            return
        lo, hi = float(finite.min()), float(finite.max())
        if hi - lo < 1e-9:
            img_u8 = np.zeros(img.shape, dtype=np.uint8)
        else:
            img_u8 = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    else:
        img_u8 = img
    out = sub / f"{stage}.png"
    cv2.imwrite(str(out), img_u8)


def dump_debug_plot(image_stem: str, stage: str, fig) -> None:
    """Guarda una figura de matplotlib como PNG (para histogramas, etc.)."""
    sub = _debug_subdir(image_stem)
    if sub is None:
        return
    out = sub / f"{stage}.png"
    fig.savefig(str(out), dpi=100, bbox_inches="tight")
