# Materials Assessment Request – Tri-Core ULTRA → Gemini  
_Generiert: 2025-11-02 17:51 UTC_

## 1) Ziel
Bitte gib eine fachliche **Einschätzung**, welches **Materialsystem** zu den unten stehenden Kennwerten passt (z. B. oxidische Perovskite, nitridische Halbleiter, Komposit), und mache **konkrete Vorschläge für eine plausible Zusammensetzung** (Elemente und grobe Molen-/Volumenanteile).

**Erwünschte Ausgabe (beides!):**
1. **Kurzbegründung** (Text) mit Referenzmaterialien/Familien.
2. **Strukturiertes JSON** mit Feldern: `family`, `candidate_compositions` (Liste), `process_window` (Temperaturbereich, Atmosphäre, Sinterdauer), `risks`, `next_experiments`.

## 2) Einzelkandidat (bestes Material)
- Bezeichnung: **X5**
- x (normiert, d=8): `[0.324113, 0.446659, 0.799103, 0.456478, 0.690363, 0.757019, 0.562693, 0.654794]`
- Abgeleitete Eigenschaften (gemappt, kalibriert falls verfügbar):
  - **density**: 5.63488299693762
  - **conductivity**: 1.0952366779614313
  - **permittivity**: 49.968417163002805
  - **bandgap**: 3.001157286040631
  - **hardness**: 6.190367541127907
  - **thermal_conductivity**: 49.968417163002805
  - **magnetic_moment**: 2.5994728233691244

## 3) Top-K-Kandidaten (Ausschnitt)
```json
[
  {
    "rank": "1",
    "epoch": "92",
    "composition": "X5-v0",
    "score": "0.454692",
    "energy": "2.151722"
  },
  {
    "rank": "2",
    "epoch": "53",
    "composition": "X5-v68",
    "score": "0.449224",
    "energy": "2.142532"
  },
  {
    "rank": "3",
    "epoch": "26",
    "composition": "X5-v44",
    "score": "0.448799",
    "energy": "2.151288"
  },
  {
    "rank": "4",
    "epoch": "7",
    "composition": "X5-v48",
    "score": "0.447884",
    "energy": "2.150065"
  },
  {
    "rank": "5",
    "epoch": "74",
    "composition": "X5-v8",
    "score": "0.447562",
    "energy": "2.146121"
  },
  {
    "rank": "6",
    "epoch": "17",
    "composition": "X5-v60",
    "score": "0.446046",
    "energy": "2.153247"
  },
  {
    "rank": "7",
    "epoch": "96",
    "composition": "X5-v80",
    "score": "0.445299",
    "energy": "2.141627"
  },
  {
    "rank": "8",
    "epoch": "33",
    "composition": "X5-v8",
    "score": "0.444862",
    "energy": "2.139271"
  },
  {
    "rank": "9",
    "epoch": "15",
    "composition": "X5-v80",
    "score": "0.444777",
    "energy": "2.151092"
  },
  {
    "rank": "10",
    "epoch": "36",
    "composition": "X5-v0",
    "score": "0.443650",
    "energy": "2.132535"
  }
]
```

## 4) Orchestrator/Driver (Kontext)
- Driver/DLL: `CipherCore_OpenCl.dll`
- Relevante Konfiguration:
```json
{
  "epochs": 100,
  "pop": 96,
  "dim": 8,
  "strategy": "mix",
  "lr0": 0.1,
  "vqe_steps": 12,
  "vqe_mode": "auto",
  "field_p1": 12.0,
  "field_p2": 0.35,
  "seed": 2025,
  "num_qubits": 4,
  "ansatz_layers": 2,
  "num_h_terms": 3,
  "gpu": 1,
  "w_tol": 0.5,
  "w_con": 0.3,
  "w_surr": 0.2,
  "tol_sigma_lo": 0.05,
  "tol_sigma_hi": 0.2,
  "export_csv": "G:\\Tri-Core-Orchestrator-ULTRA\\runs\\topk_gpu1.csv",
  "topk": 20
}
```

## 5) Mapping/Schema (Kurzinfo)
- Kalibrierte Mapper-Gewichte wurden verwendet.
  - Feature-basiertes lineares Mapping je Eigenschaft (Ridge-fit).
  - Post-Processing: clamp/softplus je nach Eigenschaft.

## 6) Leitfragen
- Welche **Materialfamilie** passt am besten?
- Welche **Stöchiometrie/Dotierung** ist plausibel (mit grobem Anteilsvorschlag)?
- Welches **Prozessfenster** (Temp., Atmosphäre, Dauer) würdest du zuerst testen?
- Welche **Risiken** (Phasenstabilität, Korngrenzen, Defekte, Oxidationszustände) und welche **nächsten Experimente**?

## 7) Ausgabeformat (bitte exakt einhalten)
```json
{
  "family": "string",
  "candidate_compositions": [
    {
      "formula": "e.g. (Ba,Ti)O3:Fe",
      "fractions": {
        "Ba": 0.5,
        "Ti": 0.49,
        "Fe": 0.01,
        "O": 3.0
      }
    }
  ],
  "process_window": {
    "temperature_C": [
      900,
      1250
    ],
    "atmosphere": "air or N2/H2",
    "duration_h": [
      2,
      12
    ]
  },
  "risks": [
    "string",
    "string"
  ],
  "next_experiments": [
    "string",
    "string"
  ]
}
```
