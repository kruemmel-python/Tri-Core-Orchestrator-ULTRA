# Tri-Core ULTRA v4.1 – Korrektes PauliZTerm-Binding (uint64 z_mask)

**Header-Definition (deine DLL):**
```c
typedef struct {
    uint64_t z_mask;
    float coefficient;
} PauliZTerm;
```

Diese Version bindet `execute_vqe_gpu` exakt inkl. `PauliZTerm*` (Masken). Die Dummy-Hamiltonian-Erzeugung setzt
`z_mask = 1 << q` mit `coefficient = 1.0` und verteilt die Terme über `num_qubits`.

## Beispielaufruf (wie bei dir)
```powershell
python materials_orchestrator_v4_1.py --epochs 50 --pop 64 --dim 8 --strategy mix --dll G:\Tri-Core-Orchestrator-ULTRA\CipherCore_OpenCl.dll --field-p1 12 --field-p2 0.35 --lr0 0.1 --vqe-steps 8 --seed 42 --log info --vqe auto --num-qubits 4 --ansatz-layers 2 --num-h-terms 2
```

> Debug-Tipp: Wechsel testweise auf `--vqe cpu`, um GPU-VQE auszuschließen. Mit korrektem Binding sollte `auto` sauber laufen.
