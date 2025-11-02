#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_gemini_packet.py
Baut aus:
- materials_props.json (Einzel-Materialkarte X*)
- runs/topk_gpu1.csv    (Top-K Kandidaten mit x_json)
- mapper_schema.json    (aktuelle Mapping-Gewichte)
eine kompakte Prompt-Datei (Markdown) + Kontext (JSON) für Gemini.
Optional: ZIP mit allen Artefakten.

Python 3.12
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import zipfile

def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def try_read(path: Optional[str | Path]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None

def load_topk(csv_path: Optional[str | Path]) -> List[Dict[str, Any]]:
    if not csv_path:
        return []
    p = Path(csv_path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter=";")
        for row in r:
            # JSON-Parsing für x_json (falls vorhanden)
            x = row.get("x_json", "") or row.get("x", "") or row.get("best_x", "")
            x_list: List[float] = []
            try:
                x_list = list(map(float, json.loads(x)))
            except Exception:
                x_list = [float(tok) for tok in x.strip("[]").split(",") if tok.strip()]
            row["_x_list"] = x_list
            rows.append(row)
    return rows

def summarize_topk(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"count": 0}
    # einfache Statistik über Score, Energy, Feature-Mittelwerte
    def fnums(col: str) -> List[float]:
        out = []
        for r in rows:
            v = r.get(col)
            if v is None or v == "" or v == "NA":
                continue
            try:
                out.append(float(v))
            except Exception:
                pass
        return out

    scores = fnums("score")
    energies = fnums("energy")
    dim = max((len(r.get("_x_list", [])) for r in rows), default=0)

    feat_means = [0.0] * dim
    feat_vars = [0.0] * dim
    if dim > 0:
        for i in range(dim):
            vals = [r["_x_list"][i] for r in rows if len(r["_x_list"]) > i]
            if vals:
                m = sum(vals)/len(vals)
                v = sum((t - m)**2 for t in vals)/len(vals)
            else:
                m = 0.0; v = 0.0
            feat_means[i] = m
            feat_vars[i] = v

    def stats(vs: List[float]) -> Dict[str, float]:
        if not vs:
            return {}
        return {
            "mean": sum(vs)/len(vs),
            "median": float(statistics.median(vs)),
            "min": min(vs),
            "max": max(vs),
            "std": float(math.sqrt(sum((x - sum(vs)/len(vs))**2 for x in vs)/len(vs))) if len(vs) > 1 else 0.0,
            "n": len(vs),
        }

    return {
        "count": len(rows),
        "score_stats": stats(scores),
        "energy_stats": stats(energies),
        "x_dim": dim,
        "x_means": feat_means,
        "x_vars": feat_vars,
    }

def build_prompt(props_json: Dict[str, Any],
                 topk_rows: List[Dict[str, Any]],
                 schema_json: Optional[Dict[str, Any]],
                 driver_name: Optional[str],
                 cfg: Dict[str, Any]) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    comp = props_json.get("composition", "X?")
    x = props_json.get("x", [])
    properties = props_json.get("properties", {})
    topk_summary = summarize_topk(topk_rows)

    # kompakter Top-K-Ausschnitt (max 10)
    def compact_rows(rows: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
        cols_keep = ["rank","epoch","composition","score","energy"]
        out = []
        for r in rows[:k]:
            out.append({c: r.get(c) for c in cols_keep})
        return out

    topk_compact = compact_rows(topk_rows, k=10)

    # kurze Hinweise zur Unsicherheit/Kalibrierung
    cal_note = "Kalibrierte Mapper-Gewichte wurden verwendet." if schema_json else "Heuristische Mapper-Gewichte (nicht kalibriert)."

    # Prompt
    md = []
    md.append(f"# Materials Assessment Request – Tri-Core ULTRA → Gemini  \n_Generiert: {now}_")
    md.append("")
    md.append("## 1) Ziel")
    md.append("Bitte gib eine fachliche **Einschätzung**, welches **Materialsystem** zu den unten stehenden Kennwerten passt "
              "(z. B. oxidische Perovskite, nitridische Halbleiter, Komposit), und mache **konkrete Vorschläge für eine plausible Zusammensetzung** "
              "(Elemente und grobe Molen-/Volumenanteile).")
    md.append("")
    md.append("**Erwünschte Ausgabe (beides!):**")
    md.append("1. **Kurzbegründung** (Text) mit Referenzmaterialien/Familien.")
    md.append("2. **Strukturiertes JSON** mit Feldern: `family`, `candidate_compositions` (Liste), "
              "`process_window` (Temperaturbereich, Atmosphäre, Sinterdauer), `risks`, `next_experiments`.")
    md.append("")
    md.append("## 2) Einzelkandidat (bestes Material)")
    md.append(f"- Bezeichnung: **{comp}**")
    md.append(f"- x (normiert, d={len(x)}): `{[round(v, 6) for v in x]}`")
    md.append("- Abgeleitete Eigenschaften (gemappt, kalibriert falls verfügbar):")
    for k, v in properties.items():
        md.append(f"  - **{k}**: {v}")
    md.append("")
    md.append("## 3) Top-K-Kandidaten (Ausschnitt)")
    if topk_compact:
        md.append("```json")
        md.append(json.dumps(topk_compact, indent=2))
        md.append("```")
    else:
        md.append("_Keine Top-K CSV angegeben oder leer._")
    md.append("")
    md.append("## 4) Orchestrator/Driver (Kontext)")
    md.append(f"- Driver/DLL: `{driver_name or props_json.get('driver','n/a')}`")
    md.append("- Relevante Konfiguration:")
    md.append("```json")
    md.append(json.dumps(cfg, indent=2))
    md.append("```")
    md.append("")
    md.append("## 5) Mapping/Schema (Kurzinfo)")
    md.append(f"- {cal_note}")
    if schema_json:
        md.append("  - Feature-basiertes lineares Mapping je Eigenschaft (Ridge-fit).")
        md.append("  - Post-Processing: clamp/softplus je nach Eigenschaft.")
    else:
        md.append("  - Default-Heuristiken ohne Referenz-Kalibrierung.")
    md.append("")
    md.append("## 6) Leitfragen")
    md.append("- Welche **Materialfamilie** passt am besten?")
    md.append("- Welche **Stöchiometrie/Dotierung** ist plausibel (mit grobem Anteilsvorschlag)?")
    md.append("- Welches **Prozessfenster** (Temp., Atmosphäre, Dauer) würdest du zuerst testen?")
    md.append("- Welche **Risiken** (Phasenstabilität, Korngrenzen, Defekte, Oxidationszustände) und welche **nächsten Experimente**?")
    md.append("")
    md.append("## 7) Ausgabeformat (bitte exakt einhalten)")
    md.append("```json")
    md.append(json.dumps({
        "family": "string",
        "candidate_compositions": [
            {"formula": "e.g. (Ba,Ti)O3:Fe", "fractions": {"Ba": 0.50, "Ti": 0.49, "Fe": 0.01, "O": 3.00}},
        ],
        "process_window": {"temperature_C": [900, 1250], "atmosphere": "air or N2/H2", "duration_h": [2, 12]},
        "risks": ["string", "string"],
        "next_experiments": ["string", "string"]
    }, indent=2))
    md.append("```")
    md.append("")
    return "\n".join(md)

def main():
    ap = argparse.ArgumentParser(description="Baut Prompt + Kontext für Gemini aus Orchestrator-Ergebnissen.")
    ap.add_argument("--props", default="materials_props.json", help="Materialkarte (aus material_mapper.py)")
    ap.add_argument("--topk", default=None, help="Top-K CSV (Semikolon)")
    ap.add_argument("--schema", default="mapper_schema.json", help="Mapper-Schema (optional)")
    ap.add_argument("--prompt-out", default="gemini_prompt.md", help="Zieldatei für Prompt (Markdown)")
    ap.add_argument("--context-out", default="gemini_context.json", help="Zieldatei für Kontext (JSON)")
    ap.add_argument("--zip-out", default=None, help="Optionaler ZIP-Ausgabepfad (bündelt alles)")
    args = ap.parse_args()

    props = read_json(args.props)
    schema_json = None
    if args.schema and Path(args.schema).exists():
        schema_json = read_json(args.schema)

    # topk laden
    rows = load_topk(args.topk) if args.topk else []

    # Prompt bauen
    driver_name = props.get("driver")
    cfg = props.get("config", {})
    prompt = build_prompt(props, rows, schema_json, driver_name, cfg)

    Path(args.prompt_out).write_text(prompt, encoding="utf-8")
    # Kontext JSON zusätzlich: vollständige Infos für API-Nutzung
    context = {
        "props": props,
        "topk_sample": rows[:50],  # begrenzen
        "schema": schema_json,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    Path(args.context_out).write_text(json.dumps(context, indent=2), encoding="utf-8")

    # optional ZIP
    if args.zip_out:
        with zipfile.ZipFile(args.zip_out, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(args.prompt_out)
            z.write(args.context_out)
            # Best-effort: auch die Quellen beilegen, falls vorhanden
            for p in [args.props, args.topk, args.schema]:
                if p and Path(p).exists():
                    z.write(p, arcname=Path(p).name)

    print(f"[OK] Prompt → {args.prompt_out}")
    print(f"[OK] Kontext → {args.context_out}")
    if args.zip_out:
        print(f"[OK] ZIP → {args.zip_out}")

if __name__ == "__main__":
    main()
