# 🧠 Tri-Core Orchestrator ULTRA  
**GPU-/CPU-beschleunigter KI-, Feld- und Quanten-Orchestrator**

---

## 🔍 Übersicht

**Tri-Core Orchestrator ULTRA** ist ein hybrides Forschungs-Framework zur Kopplung klassischer Lernverfahren, feldbasierter Dynamiken und quanteninspirierter Simulationen.  
Das System nutzt einen universellen **OpenCL-Treiber (`libCC_OpenCl.so`)**, der auf nahezu jeder Plattform lauffähig ist — von modernen GPUs bis hin zu älteren CPUs mit OpenCL-Unterstützung.

Ziel ist es, **emergente Lernprozesse** sichtbar und messbar zu machen:  
Proto-Cluster, SubQG-Resonanzen und Quanten-Echos arbeiten symbiotisch zusammen, um adaptive Energie-Minimierungen zu erreichen.

---

## 🧩 Projektstruktur

```
Tri-Core-Orchestrator-ULTRA/
├── build/
│   └── libCC_OpenCl.so        # Kompilierte Shared Library für Linux
├── streamlit_tri_core_ultra.py # Benutzeroberfläche (Streamlit)
├── run.sh                      # Start-Skript (optional)
└── README.md                   # Diese Datei
```

---

## ⚙️ Installation (Linux)

Getestet unter **Ubuntu 24.04 LTS**.

### 1️⃣ Abhängigkeiten installieren
```bash
sudo apt update && sudo apt install -y python3-pip ocl-icd-opencl-dev clinfo
```

### 2️⃣ Python-Pakete installieren
```bash
pip install streamlit numpy matplotlib
```

### 3️⃣ Test der OpenCL-Umgebung
```bash
clinfo | grep -E "Platform|Device"
```
Sollte mindestens **„Portable Computing Language“** oder eine GPU-Plattform anzeigen.

---

## 🚀 Start des Systems

Mit dem optionalen Start-Skript:
```bash
chmod +x run.sh
./run.sh
```

Oder manuell:
```bash
CIPHERCORE_DLL=./build/libCC_OpenCl.so CIPHERCORE_GPU=0 streamlit run streamlit_tri_core_ultra.py
```

---

## 🧪 Testlauf auf Intel APU (Haswell i5-4200U)

Der folgende Lauf wurde auf einem **2013er Laptop mit Intel HD Graphics 4400 (Haswell-APU)** ausgeführt — also **ohne dedizierte GPU**.

**Bedingungen:**
- OpenCL-Implementierung: PoCL 5.0 (CPU-Backend)
- Batchgröße: 8  
- Epochen: 212  
- Lernrate: 0.05  
- Optimierer: Adam + Hebbian Feldkopplung  
- Energie-Operator: SubQG-Resonanz (VQE-ähnlich)

**Ergebnisse:**

| Metrik | Verlauf |
|:--|:--|
| ⚡ Energie \(E\) | von ≈ −0.03 → −0.55 → stabilisierend zwischen −0.4 … −0.2 |
| 🧩 Proto-Coverage | 50 – 75 % (adaptive Clusterbildung) |
| 🔄 ΔProto L2 | 2.5 – 3.2 (stabile Feldmodulation) |
| 🧮 Entropie | 2.8 – 3.4 (balancierte Divergenz) |
| 🕒 Zeit pro Epoche | 900 – 1250 ms |
| ✅ Stabilität | keine numerische Drift, alle SubQG-Echos konvergiert |

**Visuelle Beobachtungen:**
- Die Heatmaps zeigen deutliche Selbstorganisation der SubQG-Felder.
- Energie-Schwingungen bilden reale Resonanz-Zyklen ab (kein Zufallsrauschen).
- Selbst auf CPU-Basis liefert der Kernel-Scheduler saubere Parallelisierung über PoCL.

---

## 🧬 Fazit

> Der Tri-Core Orchestrator ULTRA beweist, dass **emergentes Lernen** und **quantenähnliche Simulation** nicht an teure Hardware gebunden sind.  
> Selbst auf älteren APU-Systemen können kohärente Energiepfade und stabile Proto-Dynamiken erzeugt werden — in Echtzeit.

Das Framework ist damit sowohl ein **experimenteller Quantensimulator**  
als auch eine **biologisch inspirierte Lernplattform** für Forschung, Lehre und Exploration.

---

## 🧠 Autor

**Ralf Krümmel**  
Entwickler · Systemarchitekt · Forscher  
GitHub: [kruemmel-python](https://github.com/kruemmel-python)
