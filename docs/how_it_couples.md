# How It Couples: A ↔ B ↔ C Overview

```
[Proto Kernels] A  ──field score──▶  B SubQG Field
        ▲                           │
        │                           ▼
 adaptive LR mask ◀──proto map──────┘
        │                           ▲
        └── VQE noise & energy ◀─── C VQE Optimizer
```

| Signal | Quelle | Ziel | Wirkung |
|--------|--------|------|---------|
| Feld-Score | SubQG-Simulation (B) | LR-Mischer (A) | Steuert globale Lernrate | 
| Proto-LR-Maske | SubQG-Feldkarte (B) | Proto-Update (A) | Skaliert Update je Prototyp |
| Best Energy & Stabilität | VQE Optimizer (C) | Noise-Level & KPIs | Modifiziert Rauschpegel, liefert KPI-Anker |
| Gate-Auswahl | UI → VQE (C) | SPSA/Analytic Pfad | Bestimmt Param-Anzahl & Gate-Penalty |
| KPIs | Alle Pfade | Bench/History | Visualisiert ms/Energie-Delta |
