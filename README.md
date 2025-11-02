# Tri-Core Orchestrator ULTRA

Die Tri-Core-Orchestrator-ULTRA-Suite bündelt einen GPU-basierten CipherCore-Treiber, eine Streamlit-Steuerzentrale sowie automatisierte Tests und Golden-Run-Fixuren. Dieses Dokument fasst alle Schritte zusammen, die für Aufbau, Validierung und Bedienung der Plattform notwendig sind.

## Inhaltsverzeichnis
1. [Architekturüberblick](#architekturüberblick)
2. [Voraussetzungen](#voraussetzungen)
3. [Treiber-Build und Bereitstellung](#treiber-build-und-bereitstellung)
4. [Testautomatisierung](#testautomatisierung)
5. [Streamlit-App ausführen](#streamlit-app-ausführen)
6. [Datenexport und Golden Runs](#datenexport-und-golden-runs)
7. [Troubleshooting & Tipps](#troubleshooting--tipps)
8. [Weiterführende Dokumente](#weiterführende-dokumente)

## Architekturüberblick
Der Orchestrator koppelt drei Pfade, die über einen gemeinsamen Export und Metrikfluss verbunden sind:

- **Pfad A – Prototyping & klassische Deep-Learning-Kernels**: Führt dynamische Token-Zuweisungen, segmentierte Proto-Summen und Proto-Updates direkt auf der GPU aus. Anpassbare Lernraten-Koppler ermöglichen eine adaptive Mischung aus exponentiellem, sigmoidalem und anderen Modulatoren.
- **Pfad B – SubQG/Mycel-Feld**: Simuliert ein batched Feld mit energieabhängigen Protomasken. Die lokale Feldenergie steuert Per-Prototype-LR-Skalen und fließt in Stabilitätsmetriken ein.
- **Pfad C – VQE (Variational Quantum Eigensolver)**: Bietet einen konfigurierbaren Ansatz mit Gate-Familien (u. a. RX, RY, CRX/CRY/CRZ), Mess-Shots sowie optionalen analytischen Gradientenpfaden.

Alle Pfade werden durch die Streamlit-App orchestriert. Export-Routinen sammeln Trainingshistorien, Stabilitätsindizes, Entropie der Assignments sowie KPI-Kennzahlen (z. B. Energie-Delta pro Sekunde). Weitere Details zur Kopplung liefert die Datei [`docs/how_it_couples.md`](docs/how_it_couples.md).

## Voraussetzungen
### Toolchain & Bibliotheken
- **Betriebssystem**: Windows 10/11 (Treiber-Build) oder Linux/macOS (App & Tests); Streamlit selbst ist plattformunabhängig.
- **Compiler**: `g++` (MinGW-w64 empfohlen) mit C++17-Unterstützung.
- **OpenCL SDK**: Header und Libraries (z. B. aus dem Intel, NVIDIA oder AMD SDK). Die Datei `CL/opencl.h` wird beim Build eingebunden.
- **Python**: Version 3.10 oder höher mit `pip`.

### Python-Abhängigkeiten
Die App und die Tests setzen u. a. folgende Pakete voraus:

```bash
pip install -r requirements.txt  # falls vorhanden
# oder manuell
pip install streamlit numpy matplotlib plotly imageio pillow pytest
```

> **Hinweis:** Für die optionalen Golden-Run-Vergleiche müssen `numpy` und `pytest` verfügbar sein. Streamlit benötigt zusätzlich eine lauffähige Browser-Umgebung.

## Treiber-Build und Bereitstellung
Die GPU-Treiberbibliothek `CipherCore_OpenCl.dll` entsteht aus den C-Quellen im Wurzelverzeichnis (`CipherCore_OpenCl.c`, `CipherCore_NoiseCtrl.c`) und wird im Ordner `build/` abgelegt.

### Schneller Build (Windows, MinGW-w64)
1. Eingabeaufforderung mit passenden Pfaden zum MinGW-Toolchain und OpenCL-SDK öffnen.
2. Skript `driver_build\build_driver_min.bat` aufrufen:
   ```cmd
   cd driver_build
   build_driver_min.bat
   ```
3. Nach erfolgreichem Build liegen die Artefakte `build\CipherCore_OpenCl.dll` und `build\libCipherCore_OpenCl.a` vor.

Das Skript kompiliert optional die Ressourcen-Datei `CipherCore_OpenCl.rc` (falls `windres` vorhanden ist) und linkt anschließend alle Quellen mit folgenden Kerneigenschaften:
- Optimierungen (`-O3`, `-ffast-math`, `-funroll-loops`) und C++17-Modus
- Statisches Linken der Runtime (`-static-libstdc++`, `-static-libgcc`)
- Export einer Import-Library für den statischen Link in Testumgebungen

### Anpassungen & Erweiterte Builds
- **Exports konfigurieren**: Die PowerShell-Datei `driver_build\exports_list.ps1` generiert Exportlisten für alternative Buildsysteme.
- **Release-Pipeline**: Die Skripte `release.bat` und `post_build.bat` automatisieren Verpackungsschritte nach erfolgreichem Build.
- **Cross-Plattform**: Unter Linux kann `g++` mit analoger Befehlszeile verwendet werden, die resultierende Bibliothek wird jedoch als `.so` erzeugt; aktualisieren Sie in diesem Fall die Ladepfade innerhalb der Streamlit-App (`load_dll`).

> **Tipp:** Behalten Sie die `build/`-Artefakte unter Versionskontrolle ignoriert (z. B. via `.gitignore`), damit lokale Builds die Repository-Historie nicht beeinflussen.

## Testautomatisierung
Automatisierte Regressionstests sichern die Exportpfade sowie Metriken.

### Pytest ausführen
```bash
pytest
```

Die Test-Suite umfasst:
- Validierung der Lernraten-Mischung (`lr_modulated`)
- Monotonie und Wertebereiche der Proto-LR-Masken
- Entropie- und Coverage-Berechnung für Assignments
- Gate-Parametrisierungslogik für VQE-Ansätze
- Golden-Run-Snapshot-Vergleich (`tests/data/golden_export.json`)
- Roundtrip-Prüfung des Export-Payload-Builders

> **Hinweis:** Das Modul lädt standardmäßig `numpy` und `streamlit`. Falls `pytest` in einem Kopf-losen CI-Umfeld ausgeführt wird, empfiehlt sich das Setzen der Umgebungsvariablen `STREAMLIT_HEADLESS=true`, damit Streamlit keine Browserfenster öffnet.

### Golden Runs aktualisieren
Bei API-Änderungen kann das Golden-Run-Fixture über die Streamlit-App (Export-Funktion) oder ein dediziertes Skript aktualisiert werden. Prüfen Sie anschließend `tests/data/golden_export.json` in die Versionskontrolle ein, um Snapshot-Divergenzen zu vermeiden.

## Streamlit-App ausführen
Die zentrale Steuerung erfolgt über `streamlit_tri_core_ultra.py`.

### Start im Entwicklungsmodus
```bash
streamlit run streamlit_tri_core_ultra.py
```

Standardmäßig lädt die App keine DLL automatisch. Auf der Startseite wählen Sie:
1. **Treiber-DLL**: Pfad zur kompilierten `CipherCore_OpenCl.dll` (oder `.so` auf Linux). Die App validiert Signaturen und initialisiert den GPU-Kontext über `initialize_gpu`.
2. **GPU-Index**: Auswahl des Geräts; kann optional über den Query-Parameter `gpu` vorbelegt werden.
3. **Run-Setup**: Konfigurieren Sie Epochen, Batchgrößen, adaptive LR-Mixer (Gewichte für `exp`/`sigmoid`/`tanh`/`linear`) sowie SubQG-Feldparameter.
4. **VQE-Sektion**: Wählen Sie Ansatz, Gate-Familien, Layers, Shots und aktivieren Sie den analytischen Gradient-Pfad, sofern Ihr Treiber diesen unterstützt.
5. **Monitoring**: Metriken wie Stabilitätsindex (Δ pro Iteration), Assignment-Entropie, Proto-Coverage und KPI-Kennzahlen werden live visualisiert (Plotly/Matplotlib). Exportierte Läufe können als `.npz` gespeichert oder wieder eingespielt werden.

### Export & Persistenz
- **Run-Historie**: Über den Export-Button erhalten Sie eine JSON-Datei mit vollständiger Historie (`epoch`, `field_score`, `vqe_best_E`, KPI etc.).
- **Snapshots**: Golden-Run-Snapshots lassen sich über das Menü "Golden Runs" starten; die Ausgabe dient als Referenz für automatisierte Tests.
- **Multimedia**: GIF-Exports und Heatmaps werden im Unterordner `dist/` abgelegt, sofern aktiviert.

> **Sicherheitshinweis:** Die DLL wird ausschließlich im Session State geführt und nicht in Query-Parametern persistiert. Bei Änderungen am Treiber empfiehlt sich ein Neustart der App, damit `ctypes` die neue Version lädt.

## Datenexport und Golden Runs
Die Funktion `build_export_payload` sammelt die wichtigsten Kenngrößen pro Epoche, u. a. Stabilität, Entropie, Coverage sowie die End-to-End-KPI „Millisekunden pro Epoche" und "Energie-Delta pro Sekunde". Golden Runs werden über `golden_run_snapshot(seed=...)` reproduzierbar erzeugt und mit `tests/data/golden_export.json` verglichen.

Für die Archivierung empfiehlt sich folgender Workflow:
1. Streamlit-Run durchführen und Export auslösen.
2. Exportierte JSON in `tests/data/golden_export.json` ersetzen (bei beabsichtigten Änderungen).
3. `pytest` ausführen, um neue Basislinie zu validieren.

## Troubleshooting & Tipps
- **DLL-Ladefehler**: Prüfen, ob alle OpenCL-Laufzeitbibliotheken im `PATH` verfügbar sind. Nutzen Sie `Dependency Walker` oder `ldd`, um fehlende Symbole zu identifizieren.
- **GPU-Initialisierung schlägt fehl**: Der Rückgabewert von `initialize_gpu` wird geprüft. In der App erscheint eine Fehlermeldung mit dem Aufrufkontext. Stellen Sie sicher, dass keine anderen Prozesse die GPU belegen.
- **Unstete Metriken/Stabilitätsindex**: Justieren Sie die Mix-Gewichte im LR-Koppler oder senken Sie die Feldrausch-Amplitude im SubQG-Panel.
- **Analytischer Gradient deaktiviert**: Wird nur angezeigt, wenn Ihr Treiber die entsprechende Schnittstelle exportiert. Die App blendet den Schalter ansonsten aus.

## Weiterführende Dokumente
- [`docs/how_it_couples.md`](docs/how_it_couples.md): Übersicht über die Kopplungsflüsse zwischen den drei Pfaden.
- `Tri-Core Orchestrator ULTRA.pdf`: Präsentation mit Architekturdiagrammen.
- `function_test.py`: Beispielskript zum direkten Treibertest ohne UI.

Bei Fragen oder zur Integration in CI/CD-Pipelines wenden Sie sich bitte an das Engineering-Team. Diese README soll als zentrale Referenz für Onboarding, Betrieb und Weiterentwicklung dienen.
