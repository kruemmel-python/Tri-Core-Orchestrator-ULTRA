# Bericht: Vollständiger Durchlauf mit Tri-Core-Orchestrator-ULTRA

**Datum:** 2025-11-02  
**Ziel:** End-to-End-Lauf (Optimierung → Mapping → KI-Auswertungspaket) und fachliche Einordnung des besten Kandidaten **X5**.

---

## 1) Kontext & Setup

- **Orchestrator:** `material.py` (GPU-beschleunigt über `CipherCore_OpenCl.dll`)
- **Lern/Optimierungs-Konfiguration (Auszug):**
  - `--epochs 100`, `--pop 96`, `--dim 8`, `--strategy mix`
  - VQE: `--vqe auto`, `--vqe-steps 12`, `--num-qubits 4`, `--ansatz-layers 2`, `--num-h-terms 3`
  - Feldkopplung: `--field-p1 12`, `--field-p2 0.35`
  - Reproduzierbarkeit: `--seed 2025`
  - GPU: `--dll ...\CipherCore_OpenCl.dll`, `--gpu 1`
- **Outputs des Hauptlaufs:**
  - `materials_run.json` (Best-Kandidat + Historie)
  - `runs/topk_gpu1.csv` (Top-K Kandidaten)

---

## 2) Post-Processing (Mapping & Kalibrierung)

1. **Referenzdatensatz synthetisieren**  
   `material_mapper.py generate-refs --count 40 --out refs.csv --from materials_run.json --from-csv runs/topk_gpu1.csv --schema mapper_schema.json --noise 0.02`

2. **Schema kalibrieren (Ridge-Regression, pro Property):**  
   `material_mapper.py calibrate --csv refs.csv --schema-in mapper_schema.json --schema-out mapper_schema.json`

3. **Mapping anwenden & CSV anreichern:**  
   `material_mapper.py --in materials_run.json --out materials_props.json --augment-csv runs/topk_gpu1.csv`

> Ergebnis: **`materials_props.json`** mit physikalischen Kenngrößen zu **X5**, **Top-K CSV** um `prop_*`-Spalten erweitert.

---

## 3) Best-Kandidat X5 – abgeleitete Materialkennwerte

Aus `materials_props.json`:

- **Permittivität** \(\varepsilon_r\): ~ **50**
- **Bandlücke** \(E_g\): ~ **3.0 eV**
- **Magnetisches Moment** \(\mu\): ~ **2.6**
- **Dichte** \(\rho\): ~ **5.64 g/cm³**
- **Leitfähigkeit** \(\sigma\): ~ **1.1 (Ω·cm)⁻¹**
- **(weitere abgeleitete Kennwerte in der JSON enthalten; CSV mit `prop_*`)**

**Interpretation (Kurz):**
- \(\varepsilon_r \approx 50\) + \(E_g \approx 3.0\,\mathrm{eV}\) → stark polarisierbares, (para)elektrisches Oxid, Halbleiterbereich.  
- \(\mu \approx 2.6\) → Übergangsmetall-Anteile (Fe/Mn/Co/Ni), paramagnetisch bis schwach ferro/antiferromagnetisch.  
- \(\rho \approx 5.6\,\mathrm{g/cm^3}\) → typisch für Oxid-/Nitrid-Keramiken.

**Schlussfolgerung:** Hohe Plausibilität für **oxidische Perowskite (ABO\(_3\))** oder **Spinelle (AB\(_2\)O\(_4\))**, mit leicht magnetisch aktiver Dotierung auf der B-Seite.

---

## 4) Materialfamilien-Vorschlag & Begründung

- **Perowskit-Basis** (z. B. SrTiO\(_3\), BaTiO\(_3\)): stimmen bei \(E_g\) und \(\varepsilon_r\); \(\mu\) verlangt Dotierung (Fe/Mn).  
- **Spinell-Ferrite**: liefern \(\mu\), liegen aber bei \(\varepsilon_r\) teils niedriger; Perowskit bleibt primär.

> **Favorit:** **oxidische Perowskite**, **A-Site:** Sr/Ba (optional La zur Ladungskompensation), **B-Site:** Ti mit Fe/Mn-Dotierung.

---

## 5) Kandidatenvorschläge (strukturiert)

### 5.1 JSON (für Tools/Notebooks)

```json
{
  "family": "Oxidische Perowskite (ABO3) oder Spinelle (AB2O4)",
  "candidate_compositions": [
    {
      "formula": "(Sr,La)(Ti,Fe)O3",
      "fractions": {
        "Sr": 0.9,
        "La": 0.1,
        "Ti": 0.95,
        "Fe": 0.05,
        "O": 3.0
      }
    },
    {
      "formula": "Ba(Ti,Mn)O3 (leicht defekt)",
      "fractions": {
        "Ba": 1.0,
        "Ti": 0.9,
        "Mn": 0.1,
        "O": 3.0
      }
    }
  ],
  "process_window": {
    "temperature_C": [1000, 1300],
    "atmosphere": "Oxygen-rich (Air or O2)",
    "duration_h": [4, 16]
  },
  "risks": [
    "Phasenreinheit: Sekundärphasen bei zu hohem Fe/Mn.",
    "Oxidationszustände: Fe2+/Fe3+ bzw. Mn3+/Mn4+ steuern µ und σ.",
    "Defektdoping: O-Leerstellen erhöhen σ, mindern dielektrische Qualität."
  ],
  "next_experiments": [
    "Synthese & XRD-Phasenanalyse für (Sr0.9La0.1)Ti0.95Fe0.05O3.",
    "VSM-Messung des magnetischen Moments, dielektrische Spektroskopie, UV-Vis für Eg."
  ]
}
````

---

## 6) Vollständiger Durchlauf – Chronologie

1. **GPU/DLL-Init:** `initialize_gpu(gpu=1)`; VQE-Pfad wahlweise CPU/GPU (`--vqe auto`).
2. **Evolutionärer Lauf:** `epochs=100`, `pop=96`, `dim=8`, `strategy=mix`; feldgekoppelte LR-Modulation, adaptiver Noise-Regler.
3. **Selektion:** Top-K Kandidaten in `runs/topk_gpu1.csv`.
4. **Mapping:** `materials_props.json` generiert (kalibriertes `mapper_schema.json`).
5. **KI-Paket (optional):** `prepare_gemini_packet.py` → `gemini_prompt.md`, `gemini_context.json`, `gemini_packet.zip`.
6. **Bewertung:** Fachliche Einordnung → Perowskit/Spinell, Kandidatenformeln, Prozessfenster.

---

## 7) Risiken, Annahmen & Grenzen

* **Mapper ist kalibriert**, jedoch **modellinduziert** (Heuristiken + Ridge-Fit):

  * Werte sind **Screening-Indikatoren**, **keine** exakten Messgrößen.
  * (\varepsilon_r) und (\kappa_{th}) können korreliert/artefaktisch erscheinen → experimentell validieren.
* **VQE-Energien**: vergleichend nützlich, nicht absolut (Ansatz/Terms begrenzt).
* **Magnetismus**: stark abhängig von Oxidationszuständen, Sauerstoffstöchiometrie und Mikrostruktur.

---

## 8) Konkrete Nächste Schritte (Labor-Ready)

1. **Pulversynthese** (Festkörperreaktion) für
   ((\mathrm{Sr}*{0.9}\mathrm{La}*{0.1})\mathrm{Ti}*{0.95}\mathrm{Fe}*{0.05}\mathrm{O}_3)
   1000–1300 °C, 4–16 h, O(_2)-reiche Atmosphäre; zwischenkalzinieren & mahlen.
2. **XRD (Rietveld)**: Phasenreinheit, Gitterparameter, Perowskit-Bestätigung.
3. **VSM/MPMS**: (\mu(T,B)) zur Magnetismus-Einordnung.
4. **Dielektrik/Impedanz**: (\varepsilon_r(f,T)), Verlustfaktor; Korrelation mit Mikrostruktur.
5. **UV-Vis**: Tauc-Plot → (E_g)-Validierung.
6. **EDS/WDS/XPS**: Stöchiometrie, Oxidationszustände (Fe, Mn), O-Leerstellen.

---

## 9) Reproduzierbare Kommandos (Kurzlog)

```bash
# 1) Orchestrator-Lauf (Beispiel)
python materials_orchestrator_v4.py --epochs 100 --pop 96 --dim 8 --strategy mix \
  --dll "G:\Tri-Core-Orchestrator-ULTRA\CipherCore_OpenCl.dll" --gpu 1 \
  --vqe auto --vqe-steps 12 --num-qubits 4 --ansatz-layers 2 --num-h-terms 3 \
  --lr0 0.1 --field-p1 12 --field-p2 0.35 --seed 2025 --log info \
  --export-csv "G:\Tri-Core-Orchestrator-ULTRA\runs\topk_gpu1.csv" --topk 20

# 2) Referenz-Synthese + Kalibrierung
python material_mapper.py generate-refs --count 40 --out refs.csv \
  --from materials_run.json --from-csv G:\Tri-Core-Orchestrator-ULTRA\runs\topk_gpu1.csv \
  --schema mapper_schema.json --noise 0.02

python material_mapper.py calibrate --csv refs.csv \
  --schema-in mapper_schema.json --schema-out mapper_schema.json

# 3) Mapping & CSV-Augment
python material_mapper.py --in materials_run.json --out materials_props.json \
  --augment-csv G:\Tri-Core-Orchestrator-ULTRA\runs\topk_gpu1.csv
```

---

## 10) Artefakte des Laufs

* **Ergebnisse:**

  * `materials_run.json` (Best: **X5**)
  * `runs/topk_gpu1.csv` (Top-K inkl. Scores/Energien, plus `prop_*`)
  * `materials_props.json` (abgeleitete Eigenschaften zu X5)

* **Kalibrierung/Schema:**

  * `refs.csv` (synthetische Referenzen)
  * `mapper_schema.json` (aktualisierte Gewichte)

* **KI-Paket (optional):**

  * `gemini_prompt.md`, `gemini_context.json`, `gemini_packet.zip`

---

## 11) Fazit

Der vollständige Durchlauf identifizierte **X5** als vielversprechenden Kandidaten mit (\varepsilon_r \approx 50), (E_g \approx 3.0,\mathrm{eV}), (\mu \approx 2.6) und (\rho \approx 5.64,\mathrm{g/cm^3}).
Die Merkmalskombination stützt die Einordnung in **oxidische Perowskite** mit **magnetisch aktiver B-Site-Dotierung** (Fe/Mn).
Das vorgeschlagene **Perowskit-System** ((\mathrm{Sr}*{0.9}\mathrm{La}*{0.1})\mathrm{Ti}*{0.95}\mathrm{Fe}*{0.05}\mathrm{O}_3) ist ein geeigneter Startpunkt für Labor-Validierung und weitere Optimierung.

> **Empfehlung:** Zügig in die Synthese-/Charakterisierungsphase überführen und die Mapper-Gewichte iterativ mit Messdaten nachkalibrieren (Closed-Loop).

---


