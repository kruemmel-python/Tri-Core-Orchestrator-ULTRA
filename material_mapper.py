#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
material_mapper.py
==================

Mapping eines dimensionslosen Parametervektors x \in [0,1]^d (z. B. "X5" aus materials_run.json)
auf physikalisch interpretierbare Eigenschaften. Unterstützt:
- Basismapping mit heuristischen Default-Gewichten (ohne externe Abhängigkeiten)
- Optionales Laden/Speichern kalibrierter Gewichte (mapper_schema.json)
- CSV-Augmentierung (fügt prop_* Spalten zu deiner Top-K CSV hinzu; in-place)
- Kalibrierung via Ridge-Regression (pro Eigenschaft) aus Referenzdaten
- Erzeugung synthetischer Referenzdaten (generate-refs), optional auf Basis vorhandener Läufe

Python: 3.12
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import argparse
import csv
import json
import math
import random
import statistics
import sys


# --------------------------- Utility & numerics ---------------------------

def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _features_from_x(x: List[float]) -> List[float]:
    r"""
    Kompakte, robuste Feature-Zusammenfassung aus x \in [0,1]^d.
    - Bias = 1
    - m = Mittelwert
    - v = Varianz
    - l1 = mittlere absolute Abweichung von m
    - l2 = mittlere quadratische Größe
    - xmin, xmax
    - s = Schiefe-Schätzer (robust begrenzt)
    """
    if not x:
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    n = len(x)
    m = sum(x)/n
    v = sum((xi - m)**2 for xi in x)/n
    l1 = sum(abs(xi - m) for xi in x)/n
    l2 = sum(xi*xi for xi in x)/n
    xmin, xmax = min(x), max(x)
    # Schiefe (zentral dritter Moment / v^(3/2)); robust clampen
    if v > 1e-12:
        s = sum((xi - m)**3 for xi in x)/n / (v**1.5)
        s = max(-5.0, min(5.0, s))
    else:
        s = 0.0
    return [1.0, m, v, l1, l2, xmin, xmax, s]

# Namen der Eigenschaften (Reihenfolge fix)
PROP_NAMES = [
    "density",
    "conductivity",
    "permittivity",
    "bandgap",
    "hardness",
    "thermal_conductivity",
    "magnetic_moment",
]


# --------------------------- Default-Schema ---------------------------

@dataclass
class MapperSchema:
    r"""
    Lineares Mapping: prop_j = dot(weights[j], features(x)), optional mit Nach-Transform.
    Jede Eigenschaft hat:
      - weights: Liste von Koeffizienten (gleiche Länge wie Feature-Vektor)
      - post: {"kind": "linear"/"exp"/"softplus"/"clamp", ...} (einfache physikalische Plausibilisierung)
    """
    weights: Dict[str, List[float]]
    post: Dict[str, Dict[str, float]]

    @staticmethod
    def default() -> "MapperSchema":
        # Feature-Länge = 8 (siehe _features_from_x)
        # Heuristische Gewichte (Größenordnungen an frühere Outputs angelehnt)
        W = {
            #        bias   m      v      l1     l2     xmin   xmax   skew
            "density":                [  8.0,  6.0,  1.2,  0.0,  1.0,  0.3,  0.6,  0.0 ],
            "conductivity":           [ -4.0, -6.0, -1.0, -1.2, -3.5,  0.0,  0.5, -0.2 ],
            "permittivity":           [ 40.0, 90.0, 20.0,  0.0, 25.0,  5.0, 10.0,  0.0 ],
            "bandgap":                [  0.8,  3.0,  0.2, -0.2,  0.4,  0.0,  0.1,  0.0 ],
            "hardness":               [  2.0,  6.0,  0.8, -0.4,  0.6,  0.3,  0.3,  0.0 ],
            "thermal_conductivity":   [ 30.0, 90.0, 15.0, -5.0, 50.0,  0.0, 10.0,  0.0 ],
            "magnetic_moment":        [ -1.0,  5.0,  0.0,  0.0,  2.5,  0.0,  1.5,  0.4 ],
        }
        P = {
            "density":               {"kind": "clamp", "lo": 0.5, "hi": 25.0},
            "conductivity":          {"kind": "softplus", "shift": 0.0},  # >=0
            "permittivity":          {"kind": "softplus", "shift": 0.0},
            "bandgap":               {"kind": "clamp", "lo": 0.0, "hi": 8.0},
            "hardness":              {"kind": "clamp", "lo": 0.5, "hi": 10.0},
            "thermal_conductivity":  {"kind": "softplus", "shift": 0.0},
            "magnetic_moment":       {"kind": "softplus", "shift": 0.0},
        }
        return MapperSchema(weights=W, post=P)

    def apply(self, x: List[float]) -> Dict[str, float]:
        feats = _features_from_x(x)
        out: Dict[str, float] = {}
        for p in PROP_NAMES:
            w = self.weights[p]
            y = sum(wi * fi for wi, fi in zip(w, feats))
            out[p] = self._postprocess(p, y)
        return out

    def _postprocess(self, name: str, y: float) -> float:
        spec = self.post.get(name, {"kind": "linear"})
        kind = spec.get("kind", "linear")
        if kind == "linear":
            return float(y)
        if kind == "exp":
            return float(math.exp(max(-50.0, min(50.0, y))))
        if kind == "softplus":
            # softplus(y) = ln(1+exp(y)) + shift; numerisch stabil
            t = max(-50.0, min(50.0, y))
            val = math.log1p(math.exp(t)) + float(spec.get("shift", 0.0))
            return float(val)
        if kind == "clamp":
            lo = float(spec.get("lo", -1e9))
            hi = float(spec.get("hi", +1e9))
            return float(max(lo, min(hi, y)))
        return float(y)


# --------------------------- Core functions ---------------------------

def load_schema(schema_path: str | Path | None) -> MapperSchema:
    if schema_path and Path(schema_path).exists():
        raw = _read_json(schema_path)
        return MapperSchema(weights=raw["weights"], post=raw["post"])
    # fallback: mapper_schema.json neben Script
    default_path = Path("mapper_schema.json")
    if default_path.exists():
        raw = _read_json(default_path)
        return MapperSchema(weights=raw["weights"], post=raw["post"])
    return MapperSchema.default()

def save_schema(schema: MapperSchema, path: str | Path) -> None:
    _write_json(path, {"weights": schema.weights, "post": schema.post})

def map_single_material(in_json: str | Path,
                        out_json: str | Path,
                        schema_path: str | None = None) -> Dict[str, Any]:
    data = _read_json(in_json)
    comp = data.get("best_material", {}).get("composition", "X?")
    x = data.get("best_material", {}).get("x", [])
    driver = data.get("driver", None)
    config = data.get("config", {})

    schema = load_schema(schema_path)
    props = schema.apply(x)

    out = {
        "composition": comp,
        "x": x,
        "properties": props,
        "driver": driver,
        "config": config,
    }
    _write_json(out_json, out)
    print("[Mapper] Eigenschaften →", out_json)
    return out

def augment_csv(csv_path: str | Path,
                schema_path: str | None = None) -> None:
    r"""
    Erwartet eine Semikolon-CSV mit einer Spalte 'x_json' (JSON-Liste).
    Schreibt in-place zurück und hängt Spalten 'prop_<name>' an.
    """
    csv_path = Path(csv_path)
    rows: List[Dict[str, str]] = []
    schema = load_schema(schema_path)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        header = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    # Welche Spalte liefert x?
    col_x = None
    for c in ["x_json", "x", "best_x"]:
        if rows and c in rows[0]:
            col_x = c
            break
    if col_x is None:
        raise ValueError("CSV enthält keine Spalte 'x_json' (oder 'x'/'best_x').")

    # augmentieren
    for row in rows:
        try:
            x = json.loads(row[col_x])
        except Exception:
            # evtl. Kommas / Whitespaces fixen
            x = [_safe_float(tok) for tok in row[col_x].strip("[]").split(",") if tok.strip()]
        props = schema.apply(list(map(float, x)))
        for p in PROP_NAMES:
            row[f"prop_{p}"] = f"{props[p]:.6f}"

    # Header erweitern (Reihenfolge: bestehend + prop_*)
    prop_cols = [f"prop_{p}" for p in PROP_NAMES]
    header_out = header + [c for c in prop_cols if c not in header]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header_out, delimiter=";")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[Mapper] CSV angereichert: {csv_path}")


# --------------------------- Calibration ---------------------------

def _ridge_fit(X: List[List[float]], y: List[float], l2: float = 1e-3) -> List[float]:
    r"""
    Kleines Ridge-Regression Fit ohne NumPy:
      w = argmin ||X w - y||^2 + l2 ||w||^2
    Lösung über normal equations: (X^T X + l2 I) w = X^T y
    Wir lösen per einfachem Gauß-Jordan (kleine Dimension ~ 8).
    """
    m = len(X)
    if m == 0:
        raise ValueError("Keine Daten für Ridge-Fit.")

    d = len(X[0])
    # build XtX and Xty
    XtX = [[0.0]*d for _ in range(d)]
    Xty = [0.0]*d
    for i in range(m):
        xi = X[i]
        yi = y[i]
        for a in range(d):
            Xty[a] += xi[a]*yi
            va = xi[a]
            for b in range(d):
                XtX[a][b] += va*xi[b]
    # add l2*I
    for a in range(d):
        XtX[a][a] += l2

    # Solve XtX w = Xty
    # Augment [XtX | Xty]
    A = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]
    # Gauss-Jordan
    for col in range(d):
        # pivot
        piv = col
        for r in range(col, d):
            if abs(A[r][col]) > abs(A[piv][col]):
                piv = r
        if abs(A[piv][col]) < 1e-12:
            # schlecht konditioniert; diagonale fallback
            for i in range(d):
                if i == col:
                    A[i][col] = 1.0
                else:
                    A[i][col] = 0.0
            continue
        if piv != col:
            A[col], A[piv] = A[piv], A[col]
        # norm pivot row
        pv = A[col][col]
        for c in range(col, d+1):
            A[col][c] /= pv
        # eliminate others
        for r in range(d):
            if r == col:
                continue
            factor = A[r][col]
            if factor == 0.0:
                continue
            for c in range(col, d+1):
                A[r][c] -= factor * A[col][c]

    w = [A[i][d] for i in range(d)]
    return w

def calibrate(schema_in: str | None,
              schema_out: str | None,
              csv_ref: str,
              x_column: str = "x_json",
              l2: float = 1e-3) -> None:
    r"""
    Erwartet eine Semikolon-CSV mit:
      - Spalte x_column (Standard: x_json) als JSON-Liste
      - beliebige Teilmenge der Zielspalten: PROP_NAMES (density, conductivity, ...)
    Es wird pro Eigenschaft separat eine Ridge-Regression auf dem Feature-Vektor gefittet.
    """
    schema = load_schema(schema_in)
    path = Path(csv_ref)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)

    if not rows:
        raise ValueError("Referenz-CSV leer.")

    # Eingangsfeatures
    X_all: List[List[float]] = []
    targets: Dict[str, List[float]] = {p: [] for p in PROP_NAMES}
    has_target: Dict[str, bool] = {p: False for p in PROP_NAMES}

    for row in rows:
        # x laden
        try:
            x = json.loads(row.get(x_column, "[]"))
        except Exception:
            x = [_safe_float(tok) for tok in row.get(x_column, "[]").strip("[]").split(",") if tok.strip()]
        feats = _features_from_x(list(map(float, x)))
        X_all.append(feats)

        # targets
        for p in PROP_NAMES:
            key = p
            if key in row and row[key] not in (None, "", "NA"):
                targets[p].append(_safe_float(row[key]))
                has_target[p] = True
            else:
                # alternativ: prop_* Spalten als Ziel (falls vorhanden)
                key2 = f"prop_{p}"
                if key2 in row and row[key2] not in (None, "", "NA"):
                    targets[p].append(_safe_float(row[key2]))
                    has_target[p] = True
                else:
                    targets[p].append(float("nan"))

    # Für jede Eigenschaft separat fitten (nur Zeilen mit gültigen Zielen)
    for p in PROP_NAMES:
        if not has_target[p]:
            # nichts zu tun – alte Gewichte bleiben
            continue
        X: List[List[float]] = []
        y: List[float] = []
        for feats, yv in zip(X_all, targets[p]):
            if math.isfinite(yv):
                X.append(feats)
                y.append(yv)
        if len(X) >= 2:
            w = _ridge_fit(X, y, l2=l2)
            schema.weights[p] = w
        # Post-Processing unverändert

    save_to = schema_out or "mapper_schema.json"
    save_schema(schema, save_to)
    print(f"[Calibrate] Schema gespeichert: {save_to}")


# --------------------------- Generate refs ---------------------------

def _collect_x_from_materials_run(path: str | Path) -> List[List[float]]:
    xs: List[List[float]] = []
    try:
        data = _read_json(path)
        best = data.get("best_material", {})
        if isinstance(best, dict) and "x" in best:
            xs.append(list(map(float, best["x"])))
        # history best_x
        hist = data.get("history", [])
        for h in hist:
            if "best_x" in h and isinstance(h["best_x"], list):
                xs.append(list(map(float, h["best_x"])))
    except Exception:
        pass
    return xs

def _collect_x_from_csv(path: str | Path) -> List[List[float]]:
    xs: List[List[float]] = []
    p = Path(path)
    if not p.exists():
        return xs
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        # Spaltenkandidaten
        candidates = ["x_json", "x", "best_x"]
        first_row = None
        for row in reader:
            if first_row is None:
                first_row = row
            col = None
            for c in candidates:
                if c in row:
                    col = c
                    break
            if col is None:
                continue
            try:
                x = json.loads(row[col])
            except Exception:
                x = [_safe_float(tok) for tok in row[col].strip("[]").split(",") if tok.strip()]
            try:
                xs.append(list(map(float, x)))
            except Exception:
                continue
    return xs

def _add_noise_clip01(x: List[float], sigma: float, rng: random.Random) -> List[float]:
    if sigma <= 0.0:
        return list(x)
    y: List[float] = []
    for v in x:
        y.append(min(1.0, max(0.0, v + rng.gauss(0.0, sigma))))
    return y

def generate_refs(count: int,
                  out_csv: str | Path,
                  schema_path: str | None = None,
                  from_json: str | None = None,
                  from_csv: str | None = None,
                  noise: float = 0.0,
                  seed: int = 1337) -> None:
    r"""
    Erzeugt eine Semikolon-CSV mit Spalten:
      x_json; density; conductivity; ...; magnetic_moment; (zusätzlich prop_* Duplikate)
    Quellen:
      - --from materials_run.json (liest best_material.x + history.best_x)
      - --from-csv topk.csv (liest x_json / x / best_x)
      - Falls keine Quelle verfügbar: synthetisch mit d=8 (gleichverteilte x)
    Eigenschaften werden über das (ggf. kalibrierte) Schema geschätzt.
    """
    rng = random.Random(seed)
    schema = load_schema(schema_path)

    seeds: List[List[float]] = []
    if from_json and Path(from_json).exists():
        seeds += _collect_x_from_materials_run(from_json)
    if from_csv and Path(from_csv).exists():
        seeds += _collect_x_from_csv(from_csv)
    # dedupliziere & sanity
    normed = []
    seen = set()
    for x in seeds:
        tup = tuple(round(float(v), 6) for v in x)
        if tup not in seen and len(x) > 0:
            seen.add(tup)
            normed.append(list(map(float, x)))
    seeds = normed

    # Fallback: synthetisch
    if not seeds:
        dim = 8
        for _ in range(max(4, min(16, count))):
            seeds.append([rng.random() for _ in range(dim)])

    # Schreibkopf
    outp = Path(out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["x_json"] + PROP_NAMES + [f"prop_{p}" for p in PROP_NAMES]

    rows: List[Dict[str, str]] = []
    for i in range(count):
        base = seeds[i % len(seeds)]
        x_noisy = _add_noise_clip01(base, noise, rng)
        props = schema.apply(x_noisy)
        row: Dict[str, str] = {}
        row["x_json"] = json.dumps([float(f"{v:.8f}") for v in x_noisy])
        for p in PROP_NAMES:
            row[p] = f"{props[p]:.6f}"
            row[f"prop_{p}"] = f"{props[p]:.6f}"
        rows.append(row)

    with outp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[Generate-Refs] {len(rows)} Zeilen → {outp}")


# --------------------------- CLI ---------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Material-Mapper (Mapping, CSV-Augment, Kalibrierung, Generate-Refs)")
    sub = ap.add_subparsers(dest="cmd", required=False)

    # map (default)
    ap.add_argument("--in", dest="in_json", type=str, help="materials_run.json")
    ap.add_argument("--out", dest="out_json", type=str, help="materials_props.json (Ziel)")
    ap.add_argument("--strategy", choices=["hybrid", "default"], default="hybrid",
                    help="(reserviert – aktuell ohne Einfluss)")
    ap.add_argument("--schema", type=str, default=None, help="Pfad zu mapper_schema.json (optional)")
    ap.add_argument("--augment-csv", type=str, default=None, help="Top-K CSV (in-place augment)")

    # calibrate
    sp = sub.add_parser("calibrate", help="Kalibriere Gewichte aus Referenz-CSV (Ridge Regression).")
    sp.add_argument("--csv", type=str, required=True, help="Referenz-CSV (Semikolon)")
    sp.add_argument("--schema-in", type=str, default=None, help="Eingangs-Schema (optional)")
    sp.add_argument("--schema-out", type=str, default=None, help="Ziel-Schema-Datei (Default: mapper_schema.json)")
    sp.add_argument("--x-column", type=str, default="x_json", help="Spaltenname mit X-Vektor (Default: x_json)")
    sp.add_argument("--l2", type=float, default=1e-3, help="Ridge-Regularisierung")

    # generate-refs
    sg = sub.add_parser("generate-refs", help="Erzeuge synthetische/abgeleitete Referenzdaten.")
    sg.add_argument("--count", type=int, default=20, help="Anzahl Zeilen")
    sg.add_argument("--out", type=str, required=True, help="Ziel-CSV (Semikolon)")
    sg.add_argument("--schema", type=str, default=None, help="Schema für Eigenschaftsschätzung")
    sg.add_argument("--from", dest="from_json", type=str, default=None, help="materials_run.json als Quelle")
    sg.add_argument("--from-csv", dest="from_csv", type=str, default=None, help="Top-K CSV als Quelle")
    sg.add_argument("--noise", type=float, default=0.0, help="Rausch-Stddev auf x (clamp [0,1])")
    sg.add_argument("--seed", type=int, default=1337, help="Zufalls-Seed")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.cmd == "calibrate":
        calibrate(schema_in=args.schema_in,
                  schema_out=args.schema_out,
                  csv_ref=args.csv,
                  x_column=args.x_column,
                  l2=float(args.l2))
        return

    if args.cmd == "generate-refs":
        generate_refs(count=int(args.count),
                      out_csv=args.out,
                      schema_path=args.schema,
                      from_json=args.from_json,
                      from_csv=args.from_csv,
                      noise=float(args.noise),
                      seed=int(args.seed))
        return

    # Normaler map/augment Pfad
    if not args.in_json:
        print("Fehler: --in materials_run.json fehlt.", file=sys.stderr)
        sys.exit(2)
    if not args.out_json:
        # sinnvolles Default neben input
        outp = Path(args.in_json).with_name("materials_props.json")
        args.out_json = str(outp)

    result = map_single_material(args.in_json, args.out_json, schema_path=args.schema)

    if args.augment_csv:
        augment_csv(args.augment_csv, schema_path=args.schema)
        # friendly echo des zuletzt gemappten Materials
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
