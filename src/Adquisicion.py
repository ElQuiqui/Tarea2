"""
Bloque 1 — Adquisición.

Responsabilidad: leer una imagen desde disco y devolverla en formato
BGR uint8, junto a un diccionario de metadatos elementales.

Fundamento teórico: este bloque concentra la interfaz con el "sensor"
del sistema (en esta tarea, un archivo .jpg). Aislándolo, el resto del
pipeline opera siempre sobre un arreglo NumPy con forma y tipo conocidos,
lo que permite intercambiar la fuente (cámara live, batch en disco,
stream de red) sin tocar los bloques posteriores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .utils_logging import get_logger

_log = get_logger("adquisicion")


class AdquisicionError(Exception):
    """Error al leer o validar la imagen de entrada."""


def cargar_imagen(path: str | Path) -> Tuple[np.ndarray, dict]:
    """Carga una imagen desde `path` y valida su forma/tipo.

    Bloque: 1 (Adquisición).

    Fundamento: la imagen digital se modela como una matriz de muestras
    espaciales (filas × columnas) con uno o tres canales en uint8 [0,255].
    Aquí garantizamos esa forma para todo el resto del pipeline.

    Parameters
    ----------
    path : str | Path
        Ruta a un archivo de imagen legible por OpenCV (.jpg, .png, ...).

    Returns
    -------
    img : np.ndarray
        Imagen BGR uint8 con shape (H, W, 3).
    meta : dict
        {filename, path, shape, dtype, channels}.

    Raises
    ------
    AdquisicionError
        Si el archivo no existe, no se puede decodificar, o el tipo/canales
        no son los esperados.
    """
    p = Path(path)
    if not p.exists():
        raise AdquisicionError(f"Archivo no encontrado: {p}")
    if not p.is_file():
        raise AdquisicionError(f"No es un archivo: {p}")

    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise AdquisicionError(f"OpenCV no pudo decodificar la imagen: {p}")
    if img.dtype != np.uint8:
        raise AdquisicionError(
            f"Se esperaba imagen uint8, se obtuvo dtype={img.dtype}"
        )
    if img.ndim != 3 or img.shape[2] != 3:
        raise AdquisicionError(
            f"Se esperaba imagen BGR de 3 canales, shape={img.shape}"
        )

    meta = {
        "filename": p.name,
        "path": str(p.resolve()),
        "shape": tuple(int(x) for x in img.shape),
        "dtype": str(img.dtype),
        "channels": int(img.shape[2]),
    }
    _log.info("imagen cargada: %s shape=%s", p.name, img.shape)
    return img, meta
