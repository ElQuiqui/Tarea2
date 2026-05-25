"""
Fixtures pytest compartidas.

Estrategia: el archivo `tests/fixtures/labels.csv` lo rellena el usuario
con un subconjunto etiquetado de imágenes reales. Los tests leen ese
CSV y validan por bloque. Si el CSV está vacío (o todas las filas son
comentarios), los tests se *skipean* en vez de fallar — así el código
puede correr antes de que el dataset esté completo, y los tests
empiezan a validar automáticamente cuando se añaden filas.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
LABELS_CSV = FIXTURES / "labels.csv"


def _parse_labels() -> List[Dict]:
    if not LABELS_CSV.exists():
        return []
    filas: List[Dict] = []
    with LABELS_CSV.open(encoding="utf-8") as f:
        # Eliminamos líneas de comentario que comienzan con '#'.
        contenido = [ln for ln in f
                     if ln.strip() and not ln.lstrip().startswith("#")]
    if not contenido:
        return []
    reader = csv.DictReader(contenido)
    for row in reader:
        filas.append({
            "filepath": PROJECT_ROOT / row["filepath"].strip(),
            "clase": row["clase"].strip().lower(),
            "iluminacion": row.get("iluminacion", "").strip(),
            "fondo": row.get("fondo", "").strip(),
        })
    # Filtramos las que no existen en disco para no fallar por ausencia.
    return [r for r in filas if r["filepath"].exists()]


@pytest.fixture(scope="session")
def labels() -> List[Dict]:
    rows = _parse_labels()
    if not rows:
        pytest.skip(
            "labels.csv vacío o sus imágenes no existen — rellenar para activar tests."
        )
    return rows


@pytest.fixture(scope="session")
def labels_por_clase(labels) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for r in labels:
        out.setdefault(r["clase"], []).append(r)
    return out


def primera_por_clase(rows: List[Dict], clase: str) -> Optional[Dict]:
    for r in rows:
        if r["clase"] == clase:
            return r
    return None
