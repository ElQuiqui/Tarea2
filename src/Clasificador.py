"""
Bloque 5 — Clasificación e interpretación.

Toma los scores y blobs del bloque 4 y aplica una **regla de decisión
explícita** (no entrenada) para producir:
    {estado, clase, confianza, bbox, centroide, scores_por_clase}.

Justificación de usar reglas en vez de un clasificador entrenado:
- El conjunto disponible (subconjunto etiquetado por el usuario) es
  pequeño y heterogéneo en iluminación y fondo. Un k-NN/SVM con tan
  pocos datos generaliza mal.
- Las reglas son interpretables: el evaluador puede leer la decisión
  ("el score de rayado supera al segundo mejor por X×") sin abrir un
  modelo opaco.
- Los pesos teóricos vienen de la física: un rayado es alargado, una
  contaminación es un blob compacto con ΔE alto, una abrasión es una
  región difusa con ΔE bajo. Esto se traduce directamente en reglas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

from . import config
from .utils_logging import get_logger
from .Deteccion import ResultadoDeteccion, BlobInfo

_log = get_logger("clasificador")


@dataclass
class ResultadoClasificacion:
    estado: str                      # "OK" | "defectuoso" | "desconocido" | "error"
    clase: Optional[str]             # "rayado" | "contaminacion" | "abrasion" | None
    confianza: float                 # [0,1]
    bbox: Optional[Tuple[int, int, int, int]]   # (x,y,w,h) en ROI rectificado
    centroide: Optional[Tuple[float, float]]    # (cx,cy) en ROI rectificado
    scores_por_clase: dict
    justificacion: str               # explicación breve de la decisión


def _mejor_blob(blobs: List[BlobInfo]) -> Optional[BlobInfo]:
    if not blobs:
        return None
    return max(blobs, key=lambda b: b.area)


def clasificar(det: ResultadoDeteccion) -> ResultadoClasificacion:
    """Aplica las reglas de decisión.

    Bloque: 5.

    Reglas (jerarquía coherente con la cascada del bloque 4):
    1. Si NINGUNA clase supera su umbral de presencia → OK.
    2. Si EXACTAMENTE UNA clase está presente → esa clase.
    3. Si MÁS DE UNA → se da prioridad a contaminación (defecto más
       severo y de menor ambigüedad), luego abrasión, luego rayado.
       Esta jerarquía se justifica porque los detectores ya enmascaran
       hacia abajo (contaminación enmascara abrasión, ambos enmascaran
       rayado), por lo que solapamientos son raros y la jerarquía
       coincide con el "tamaño físico" del defecto.
    """
    s = det.scores()
    presente_contam = s["contaminacion"] >= config.SCORE_CONTAM_PRESENT
    presente_abras = s["abrasion"] >= config.SCORE_ABRASION_PRESENT
    presente_rayado = s["rayado"] >= config.SCORE_RAYADO_PRESENT

    # Prioridad coherente con la cascada de detección:
    # contaminación > rayado > abrasión. Rayado tiene precedencia sobre
    # abrasión porque un patrón de rayones paralelos también dispara
    # densidad de bordes en la zona vecina (la ventana 31×31 absorbe los
    # gradientes), y si rayado se activó primero significa que las
    # estructuras lineales son la causa primaria.
    activos = [c for c, p in (("contaminacion", presente_contam),
                              ("rayado", presente_rayado),
                              ("abrasion", presente_abras)) if p]

    if not activos:
        max_score = max(s.values())
        # Confianza OK: cuán por debajo de los umbrales están los scores.
        umbrales = [config.SCORE_CONTAM_PRESENT, config.SCORE_ABRASION_PRESENT,
                    config.SCORE_RAYADO_PRESENT]
        margen_min = min(u - sc for u, sc in zip(umbrales, [s["contaminacion"],
                                                              s["abrasion"],
                                                              s["rayado"]]))
        conf = min(1.0, max(0.3, margen_min / max(umbrales[0], 1e-6)))
        _log.info("OK — ningún score supera su umbral (scores=%s)",
                  {k: round(v, 4) for k, v in s.items()})
        return ResultadoClasificacion(
            estado="OK", clase=None, confianza=round(conf, 3),
            bbox=None, centroide=None, scores_por_clase=s,
            justificacion="Ninguno de los detectores supera su umbral de presencia.",
        )

    # Prioridad fija: contaminación > abrasión > rayado.
    clase = activos[0]
    mapa = getattr(det, clase)
    mejor = _mejor_blob(mapa.blobs)

    # Confianza: cuántas veces por encima del umbral está el score ganador.
    umbral_clase = {
        "contaminacion": config.SCORE_CONTAM_PRESENT,
        "abrasion": config.SCORE_ABRASION_PRESENT,
        "rayado": config.SCORE_RAYADO_PRESENT,
    }[clase]
    conf = min(1.0, s[clase] / (3.0 * umbral_clase))  # ~1.0 cuando score = 3× umbral
    conf = max(0.3, conf)

    bbox = mejor.bbox if mejor else None
    centro = mejor.centroide if mejor else None

    if conf < config.CONFIDENCE_FLOOR:
        _log.warning("desconocido — conf=%.2f < %.2f", conf, config.CONFIDENCE_FLOOR)
        return ResultadoClasificacion(
            estado="desconocido", clase=None, confianza=conf,
            bbox=bbox, centroide=centro, scores_por_clase=s,
            justificacion=(
                f"Confianza {conf:.2f} insuficiente. Top={clase} score={s[clase]:.4f}. "
                f"Revisar manualmente."
            ),
        )

    activos_str = "+".join(activos) if len(activos) > 1 else activos[0]
    _log.info("defectuoso — clase=%s conf=%.2f score=%.4f (activos: %s)",
              clase, conf, s[clase], activos_str)
    return ResultadoClasificacion(
        estado="defectuoso", clase=clase, confianza=round(conf, 3),
        bbox=bbox, centroide=centro, scores_por_clase=s,
        justificacion=(
            f"Score {clase}={s[clase]:.4f} ≥ umbral {umbral_clase:.4f}. "
            f"Clases activas: {activos_str}. Prioridad: contaminación>rayado>abrasión."
        ),
    )
