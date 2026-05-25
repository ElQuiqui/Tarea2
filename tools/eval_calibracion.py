"""Corre el pipeline sobre todas las imágenes etiquetadas y reporta
matriz de confusión + tabla detallada con scores. Útil para iterar
sobre la calibración de umbrales en src/config.py.

Uso:
    python -m tools.eval_calibracion
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from src.main import procesar_una
from src.utils_logging import set_debug

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "tests" / "fixtures" / "labels.csv"
OUT_DIR = ROOT / "output" / "calibracion"


def cargar_labels():
    filas = []
    with CSV_PATH.open(encoding="utf-8") as f:
        contenido = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(contenido)
    for row in reader:
        filas.append({
            "filepath": ROOT / row["filepath"].strip(),
            "clase": row["clase"].strip().lower(),
            "iluminacion": row.get("iluminacion", "").strip(),
            "fondo": row.get("fondo", "").strip(),
        })
    return [r for r in filas if r["filepath"].exists()]


def main():
    set_debug(False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = cargar_labels()
    print(f"Imágenes etiquetadas: {len(labels)}\n")

    resultados = []
    for r in labels:
        rep = procesar_una(r["filepath"], OUT_DIR)
        cl = rep["clasificacion"]
        predicha = cl["clase"] if cl["estado"] == "defectuoso" else cl["estado"]
        if cl["estado"] == "OK":
            predicha = "ok"
        scores = cl["scores_por_clase"]
        resultados.append({
            "filename": r["filepath"].name,
            "iluminacion": r["iluminacion"],
            "fondo": r["fondo"],
            "esperada": r["clase"],
            "predicha": predicha,
            "estado": cl["estado"],
            "confianza": cl["confianza"],
            "score_rayado": round(scores.get("rayado", 0), 4),
            "score_contaminacion": round(scores.get("contaminacion", 0), 4),
            "score_abrasion": round(scores.get("abrasion", 0), 4),
            "acierto": predicha == r["clase"],
        })

    # Tabla detallada
    print("=" * 110)
    print(f"{'archivo':32}{'fondo':6}{'ilum':9}{'esperada':16}{'predicha':16}{'conf':6}  r/c/a")
    print("-" * 110)
    for r in resultados:
        ok = "OK " if r["acierto"] else "** "
        scores = f"{r['score_rayado']:.3f}/{r['score_contaminacion']:.3f}/{r['score_abrasion']:.3f}"
        print(f"{ok}{r['filename']:30}{r['fondo']:6}{r['iluminacion']:9}"
              f"{r['esperada']:16}{r['predicha']:16}{r['confianza']:.2f}  {scores}")

    # Matriz de confusión
    print("\nMATRIZ DE CONFUSIÓN")
    print("=" * 110)
    clases = sorted(set([r["esperada"] for r in resultados]) |
                    set([r["predicha"] for r in resultados]))
    M = defaultdict(lambda: defaultdict(int))
    for r in resultados:
        M[r["esperada"]][r["predicha"]] += 1

    header = " " * 18 + "".join(f"{c[:13]:>14}" for c in clases)
    print(header)
    for esp in sorted(set(r["esperada"] for r in resultados)):
        row = f"  {esp:14} | "
        for pre in clases:
            row += f"{M[esp][pre]:>14}"
        print(row)

    aciertos = sum(1 for r in resultados if r["acierto"])
    print(f"\nAciertos: {aciertos}/{len(resultados)} = {100.0*aciertos/len(resultados):.1f}%")


if __name__ == "__main__":
    main()
