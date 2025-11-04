# 🧠 Tri-Core Orchestrator ULTRA  
### Universal GPU-Driver Framework for Classical, Field-Based, and Quantum-Inspired Computation  
**Author / Entwickler:** Ralf Krümmel  

---

## 🇩🇪 Überblick
Der **Tri-Core Orchestrator ULTRA** verbindet drei komplementäre Rechenpfade auf GPU-Ebene:  
1. **Klassische Deep-Learning- und Tensor-Kerne** (Matrixmultiplikation, Softmax, GELU, LayerNorm)  
2. **SubQG / Feldbasierte Simulation** (energie-gekoppelte Resonanzfelder, agentenbasierte Energieverteilung)  
3. **Quanteninspirierte VQE- und Gate-Kerne** (RX, RZ, CRX, CNOT, OTOC-Echo-Sequenzen)  

Dieses Framework nutzt eine einheitliche C/OpenCL-Treiberarchitektur, die sowohl unter **Windows (.dll)**  
als auch unter **Linux (.so)** vollständig funktionsfähig ist.

---

## 🇬🇧 Overview
The **Tri-Core Orchestrator ULTRA** integrates three complementary GPU computation paths:  
1. **Classical deep-learning and tensor kernels** (matrix multiplication, softmax, GELU, layer norm)  
2. **SubQG field-based simulation** (energy-coupled resonance fields and distributed agents)  
3. **Quantum-inspired VQE & gate kernels** (RX, RZ, CRX, CNOT, and OTOC echo operations)  

This unified **C/OpenCL driver** runs seamlessly on both **Windows (.dll)** and **Linux (.so)** systems.

---

## 🚀 Installation

### Linux
```bash
sudo apt update && sudo apt install -y build-essential pkg-config ocl-icd-opencl-dev
g++ -std=c++17 -O3 -march=native -ffast-math -funroll-loops -fstrict-aliasing -DNDEBUG -DCL_TARGET_OPENCL_VERSION=120 -DCL_FAST_OPTS -shared CC_OpenCL.c CipherCore_NoiseCtrl.c -o build/libCC_OpenCl.so -I"./" -I"./CL" -L"./CL" -lOpenCL -static-libstdc++ -static-libgcc
```

### Windows (PowerShell)
```bash
g++ -std=c++17 -O3 -march=native -ffast-math -funroll-loops -fstrict-aliasing -DNDEBUG -DCL_TARGET_OPENCL_VERSION=120 -DCL_FAST_OPTS -shared CC_OpenCl.c CipherCore_NoiseCtrl.c -o build/CC_OpenCl.dll -I"./" -I"./CL" -L"./CL" -lOpenCL "-Wl,--out-implib,build/libCC_OpenCl.a" -static-libstdc++ -static-libgcc
```

### Start der UI (Streamlit)
```bash
streamlit run streamlit_tri_core_ultra.py
```

---

## ⚙️ Kernel-Kategorien und Funktionen

| Kategorie | Kernel / Funktion | Beschreibung (🇩🇪) | Description (🇬🇧) |
|------------|-------------------|--------------------|-------------------|
| **Tensor / DL** | `matrix_multiply`, `softmax_rowwise`, `layer_norm`, `gelu_elementwise`, `add_elementwise`, `mul_elementwise` | Klassische Operatoren für neuronale Netze und Transformer-Modelle | Core operators for neural networks and transformer backends |
| **Optimierung** | `adam_update`, `proto_update_step`, `proto_segmented_sum_atomic`, `hebbian_update_local_reduce` | Lernkernels für Gewichtsaktualisierung und Hebb’sche Anpassung | Learning kernels for weight updates and Hebbian-style adaptation |
| **SubQG / Feld** | `shape_loss_reward_penalty`, `shape_loss_reward_penalty_list`, `subqg_initialize_state_batched`, `subqg_simulation_step_batched` | Simulation lokaler Energie- und Phasenfelder (Myzel-ähnlich) | Local energy/phase field simulation (mycelial-inspired) |
| **Quantum / VQE** | `execute_vqe_gpu`, `quantum_apply_single_qubit`, `quantum_apply_controlled_not`, `execute_quantum_echoes_otoc_gpu` | Quanteninspirierte Gatter und Echo-Sequenzen (OTOC) | Quantum-inspired gates and out-of-time-order correlations |

---

## 🧩 Architekturüberblick

```text
┌────────────────────────┐
│ Python Streamlit UI    │ ← Benutzerinteraktion, Visualisierung
└────────────┬───────────┘
             │ ctypes / FFI
┌────────────▼───────────┐
│ CipherCore_OpenCL.c    │ ← GPU-Management, Kernel-Komposition
├────────────────────────┤
│ Klassische Tensor-Kerne│ ← MatMul, Softmax, LayerNorm, GELU
│ Feldbasierte SubQG-Kerne│ ← Energie/Phasen-Simulation
│ Quantenkernels (VQE)   │ ← RX, RZ, CNOT, OTOC
└────────────────────────┘
```

---

## 🔬 Benchmark-Ergebnisse (Intel Haswell APU)

| Metrik | Wert | Kommentar |
|--------|------|-----------|
| Plattform | Intel OpenCL (pocl / CPU) | Läuft auch ohne dedizierte GPU |
| Epoche (Durchschnitt) | ~0.91 s | 212 Epochen, stabiler Lauf |
| VQE Energie-Minimum | ≈ -0.187 | Hohe Präzision trotz Software-Treiber |
| Feldkonvergenz | < 0.04 RMS | Stabile Resonanzbildung |
| CPU-Auslastung | ~85 % | Vollständig parallelisiert |
| Speicherbedarf | < 300 MB | Inklusive PCA + Feldkarten |

---

## 🧠 Fazit (🇩🇪)
Der Treiber beweist, dass **OpenCL** – richtig eingesetzt – selbst auf älteren APUs eine Plattform für hybride klassische und quanteninspirierte Lernprozesse bieten kann.  
Er kombiniert Tensoroperationen, Feldsimulation und quantenlogische Prozesse in einer **einheitlichen GPU-Schicht**.

## 🧠 Conclusion (🇬🇧)
This driver demonstrates that **OpenCL**, when properly utilized, can form a unified platform for hybrid classical and quantum-inspired learning — even on older APUs.  
It bridges tensor computation, field simulation, and quantum logic in a single cohesive GPU layer.

---

**Repository:** [Tri-Core Orchestrator ULTRA](https://github.com/kruemmel-python/Tri-Core-Orchestrator-ULTRA)  
**Author:** Ralf Krümmel  
**License:** MIT  

---

## ⚙️ Erweiterte Liste der GPU-Kernel

| Kategorie | Kernel / Funktion | Beschreibung (🇩🇪) | Description (🇬🇧) |
| :--- | :--- | :--- | :--- |
| **Tensor / DL (Core)** | `matrix_multiply`, `softmax_rowwise`, `layer_norm`, `gelu_elementwise`, `add_elementwise`, `mul_elementwise`, `log_softmax_stable_rowwise`, `add_broadcast_pe`, `add_bias_mn`, `pairwise_similarity_dot`, `embedding_lookup` | Kernoperatoren für allgemeine lineare Algebra, Aktivierungen und Normalisierung in neuronalen Netzen. | Core operators for general linear algebra, activations, and normalization in neural networks. |
| **Tensor / DL (Backward)** | `gelu_backward_elementwise`, `matmul_backward_da`, `matmul_backward_db`, `layer_norm_backward`, `softmax_backward`, `mul_backward`, `transpose_backward`, `reduce_sum_axis01`, `embedding_backward_calc_delta_local` | Berechnung der Gradienten (dA, dB, dX) für die Kernoperatoren und Bias-Gradienten. | Gradient computation (dA, dB, dX) for core operators and bias gradients. |
| **Tensor / DL (Batched/Reshape)** | `transpose`, `transpose_batched_last_two`, `transpose_12_batched`, `matmul_batched`, `matmul_batched_backward_da`, `matmul_batched_backward_db` | Operatoren für das Umschichten und die Matrixmultiplikation von Batched-Tensoren (z. B. in Multi-Head-Attention). | Operators for reshaping and matrix multiplication of batched tensors (e.g., in multi-head attention). |
| **Optimierung / Learning** | `adam_update`, `hebbian_update_local_reduce`, `proto_segmented_sum_atomic`, `proto_update_step` | Lernkerne für Gewichtsaktualisierung (Adam, Hebbian) und Prototyp-basierte Modellaktualisierungen. | Learning kernels for weight updates (Adam, Hebbian) and prototype-based model updates. |
| **Specialized NN / Hybrid** | `threshold_spike`, `dynamic_token_assignment` | Kernel für die Erzeugung binärer Spikes und die Zuweisung von Aktivierungen zu Prototypen/Tokens. | Kernels for generating binary spikes and assigning activations to prototypes/tokens. |
| **Loss / Metrics** | `cross_entropy_loss_grad`, `shape_loss_reward_penalty`, `shape_loss_reward_penalty_list` | Berechnung von Cross-Entropy-Verlust und Gradienten sowie spezialisierte Verlustformung (Shaping). | Calculation of cross-entropy loss and gradients, and specialized loss shaping. |
| **SubQG / Feld** | `subqg_simulation_step`, `subqg_inject_agents` | Simulation der lokalen Energie- und Phasenfelder (Myzel-inspiriert) und Agenten-Interaktion. | Simulation of local energy and phase fields (mycelial-inspired) and agent interaction. |
| **Quantum (Low-Level Gates)** | `quantum_apply_single_qubit`, `quantum_apply_controlled_phase`, `quantum_apply_controlled_not`, `quantum_swap_qubits` | Elementare Operationen (Gatter) auf Qubit-Zustandsvektoren. | Elementary operations (gates) on qubit state vectors. |
| **Quantum (Algorithm Components)**| `quantum_phase_oracle`, `quantum_phase_flip_except_zero`, `quantum_modular_exponentiation` | Spezifische Phasen- und Arithmetik-Kernel für Algorithmen (z.B. Grover, Shor). | Specific phase and arithmetic kernels for algorithms (e.g., Grover, Shor). |
| **Quantum (Measurement/VQE)**| `quantum_compute_probabilities`, `quantum_expectation_pauli_z` | Berechung von Wahrscheinlichkeiten und Erwartungswerten (Pauli Z-Basis) für Messungen. | Calculation of probabilities and expectation values (Pauli Z-basis) for measurements. |
| **Cryptography** | `sqse_encrypt`, `sqse_decrypt` | Symmetrische Sequenz-Entropie (SQSE) Verschlüsselungs-/Entschlüsselungskerne. | Symmetric Sequence Entropy (SQSE) encryption/decryption kernels. |
