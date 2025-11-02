# tests/conftest.py
import os
from pathlib import Path
import pytest

# Wir nutzen deine bereits korrekte ctypes-Signaturierung aus dem App-Modul
mod = pytest.importorskip("streamlit_tri_core_ultra")

@pytest.fixture(scope="session")
def ctx():
    """
    Session-weite Treiber-Initialisierung für function_test.py.
    Nutzt entweder $CIPHERCORE_DLL oder build/CipherCore_OpenCl.dll.
    Wählt GPU über $CIPHERCORE_GPU (default: 0).
    Skipped sauber, wenn DLL/GPU nicht verfügbar.
    """
    # DLL finden
    cand = [
        os.getenv("CIPHERCORE_DLL", ""),
        str(Path("build") / "CipherCore_OpenCl.dll"),
        "CipherCore_OpenCl.dll",
    ]
    dll_path = next((Path(p) for p in cand if p and Path(p).exists()), None)
    if dll_path is None:
        pytest.skip(
            "CipherCore DLL nicht gefunden. "
            "Setze CIPHERCORE_DLL oder lege build/CipherCore_OpenCl.dll ab."
        )

    # DLL mit deinen argtypes/restypes laden
    dll = mod.load_dll(dll_path)

    # GPU wählen
    gpu = int(os.getenv("CIPHERCORE_GPU", "0"))

    # GPU initialisieren (bei Misserfolg: Skip statt Error)
    ok = dll.initialize_gpu(gpu)
    if ok != 1:
        pytest.skip(f"GPU {gpu} konnte nicht initialisiert werden (initialize_gpu != 1).")

    # Headless für Streamlit-Imports (falls Tests dieses Modul berühren)
    os.environ.setdefault("STREAMLIT_HEADLESS", "true")

    try:
        yield mod.DriverCtx(dll=dll, gpu=gpu)
    finally:
        try:
            dll.finish_gpu(gpu)
        finally:
            dll.shutdown_gpu(gpu)
