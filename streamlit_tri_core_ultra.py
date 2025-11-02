#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
streamlit_tri_core_ultra.py
===========================
Interaktive Streamlit-App für deinen CipherCore-Treiber mit drei gekoppelten Pfaden:
(A) Prototypen/klassische DL-Kernels, (B) SubQG/Mycel-Feld, (C) VQE (quanteninspiriert).

ULTRA-Features:
- Persistenz (.npz): Läufe speichern & laden (History + Epoche-Data vollständig)
- Parametrische LR-Kopplung: exp / sigmoid / tanh / linear (UI-Parameter)
- Flexibler VQE-Hamiltonian-Editor (Pauli-Z-Terme als JSON, Validierung, Vorschau)
- PRO-Extras (PCA, GIF-Export, Per-Proto-Metriken, Konfidenz-Heatmaps)
- Stabil: DLL-Pfad nur im Session State (kein Query-Param), GPU-Index in Query-Param
- Robust: Slider-Fallbacks bei nur 1 Epoche (kein Streamlit-Crash), Singleton-Plot

Python: 3.12
"""

from __future__ import annotations

import ctypes as ct
import json
import math
import os
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import imageio.v2 as imageio
from PIL import Image

# =============================================================================
# Hyperparameter (klein, reproduzierbar)
# =============================================================================

# (A) Proto/DL
B: Final[int] = 2     # Batch
S: Final[int] = 8     # Sequenzlänge
E: Final[int] = 32    # Embedding-Dimension
T: Final[int] = 16    # Anzahl Prototypen

BASE_LR: Final[float] = 0.1
LR_MIN: Final[float] = 0.01
LR_MAX: Final[float] = 0.5

# (B) SubQG
SUBQG_INIT_ENERGY: Final[float] = 0.2
SUBQG_INIT_PHASE: Final[float] = 0.0
SUBQG_NOISE: Final[float] = 0.05
SUBQG_THR: Final[float] = 0.1
SUBQG_BATCH_CELLS: Final[int] = 64  # 8x8

# (C) VQE Defaults (können in UI überschrieben werden)
VQE_QUBITS_DEFAULT: Final[int] = 10
VQE_LAYERS_DEFAULT: Final[int] = 2

# SPSA (für VQE-Parameteroptimierung)
SPSA_ITERS_DEFAULT: Final[int] = 60
SPSA_A: Final[float] = 0.05
SPSA_C: Final[float] = 0.1
SPSA_ALPHA: Final[float] = 0.602
SPSA_GAMMA: Final[float] = 0.101

# =============================================================================
# ctypes-Bindings & Kontext
# =============================================================================

class PauliZTerm(ct.Structure):
    _fields_ = [("z_mask", ct.c_uint64), ("coefficient", ct.c_float)]


@dataclass(frozen=True)
class DriverCtx:
    dll: ct.CDLL
    gpu: int


def _chk(ok: int, where: str) -> None:
    if ok != 1:
        raise RuntimeError(f"Treiber-Call fehlgeschlagen: {where}")


def load_dll(dll_path: Path) -> ct.CDLL:
    dll = ct.CDLL(str(dll_path.resolve()))

    # Core / Memory / Sync
    dll.initialize_gpu.argtypes = [ct.c_int]
    dll.initialize_gpu.restype = ct.c_int
    dll.shutdown_gpu.argtypes = [ct.c_int]
    dll.shutdown_gpu.restype = None
    dll.finish_gpu.argtypes = [ct.c_int]
    dll.finish_gpu.restype = ct.c_int
    dll.allocate_gpu_memory.argtypes = [ct.c_int, ct.c_size_t]
    dll.allocate_gpu_memory.restype = ct.c_void_p
    dll.free_gpu_memory.argtypes = [ct.c_int, ct.c_void_p]
    dll.free_gpu_memory.restype = None
    dll.write_host_to_gpu_blocking.argtypes = [
        ct.c_int, ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_void_p
    ]
    dll.write_host_to_gpu_blocking.restype = ct.c_int
    dll.read_gpu_to_host_blocking.argtypes = [
        ct.c_int, ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_void_p
    ]
    dll.read_gpu_to_host_blocking.restype = ct.c_int

    # A) Proto-/DL
    dll.execute_dynamic_token_assignment_gpu.argtypes = [
        ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p,
        ct.c_int, ct.c_int, ct.c_int, ct.c_int
    ]
    dll.execute_dynamic_token_assignment_gpu.restype = ct.c_int

    dll.execute_proto_segmented_sum_gpu.argtypes = [
        ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,
        ct.c_int, ct.c_int, ct.c_int
    ]
    dll.execute_proto_segmented_sum_gpu.restype = ct.c_int

    dll.execute_proto_update_step_gpu.argtypes = [
        ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p,
        ct.c_float, ct.c_int, ct.c_int
    ]
    dll.execute_proto_update_step_gpu.restype = ct.c_int

    dll.execute_shape_loss_with_reward_penalty_list_gpu.argtypes = [
        ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,
        ct.c_int, ct.c_int, ct.c_int, ct.c_float, ct.c_float, ct.c_float
    ]
    dll.execute_shape_loss_with_reward_penalty_list_gpu.restype = ct.c_int

    # B) SubQG
    dll.subqg_initialize_state_batched.argtypes = [
        ct.c_int, ct.c_int,
        ct.POINTER(ct.c_float), ct.POINTER(ct.c_float),
        ct.c_float, ct.c_float
    ]
    dll.subqg_initialize_state_batched.restype = ct.c_int

    dll.subqg_simulation_step_batched.argtypes = [
        ct.c_int,
        ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.POINTER(ct.c_float),
        ct.c_int,
        ct.POINTER(ct.c_float), ct.POINTER(ct.c_float), ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.POINTER(ct.c_int),
        ct.POINTER(ct.c_float), ct.c_int
    ]
    dll.subqg_simulation_step_batched.restype = ct.c_int

    # C) VQE
    dll.execute_vqe_gpu.argtypes = [
        ct.c_int, ct.c_int, ct.c_int,
        ct.POINTER(ct.c_float), ct.c_int,
        ct.POINTER(PauliZTerm), ct.c_int,
        ct.POINTER(ct.c_float),
        ct.POINTER(ct.c_float)
    ]
    dll.execute_vqe_gpu.restype = ct.c_int

    dll.set_noise_level.argtypes = [ct.c_int, ct.c_float]
    dll.set_noise_level.restype = None

    return dll

# =============================================================================
# GPU-Buffer Utilities
# =============================================================================

def gpu_upload(ctx: DriverCtx, arr: np.ndarray) -> ct.c_void_p:
    buf = ctx.dll.allocate_gpu_memory(ctx.gpu, arr.nbytes)
    if not buf:
        raise MemoryError("allocate_gpu_memory")
    ok = ctx.dll.write_host_to_gpu_blocking(
        ctx.gpu, buf, 0, arr.nbytes, arr.ctypes.data_as(ct.c_void_p)
    )
    _chk(ok, "write_host_to_gpu_blocking")
    return buf


def gpu_download(ctx: DriverCtx, buf: ct.c_void_p,
                 shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    out = np.empty(shape, dtype=dtype)
    ok = ctx.dll.read_gpu_to_host_blocking(
        ctx.gpu, buf, 0, out.nbytes, out.ctypes.data_as(ct.c_void_p)
    )
    _chk(ok, "read_gpu_to_host_blocking")
    return out

# =============================================================================
# Mathe-Utilities
# =============================================================================

def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2].T
    proj = Xc @ comps
    return proj, mu.squeeze(0), comps


def square_reshape(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    N = vec.size
    n = int(np.sqrt(N))
    if n * n == N:
        return vec.reshape(n, n)
    if n >= 1:
        return vec[: n * n].reshape(n, n)
    return vec.reshape(1, -1)

# =============================================================================
# LR-Kopplungen
# =============================================================================

def lr_modulated(base: float, score: float, mode: str, p1: float, p2: float) -> float:
    """Moduliere base-LR anhand score mit gewähltem Modus + Parametern."""
    x = max(0.0, float(score))
    match mode:
        case "exp":
            s = 1.0 - math.exp(-p1 * x)         # p1: Steilheit
            lr = base * (0.5 + p2 * s)          # p2: Gain
        case "sigmoid":
            s = 1.0 / (1.0 + math.exp(-p1 * (x - p2)))  # p1: Steilheit, p2: Center
            lr = base * (0.5 + s)
        case "tanh":
            s = 0.5 * (1.0 + math.tanh(p1 * (x - p2)))  # p1: Steilheit, p2: Center
            lr = base * (0.5 + s)
        case "linear":
            lr = base * (0.5 + p1 * x)          # p1: Steigung
        case _:
            s = 1.0 - math.exp(-x)
            lr = base * (0.5 + s)
    return max(LR_MIN, min(LR_MAX, lr))

# =============================================================================
# SubQG & VQE (mit flexiblem Hamiltonian)
# =============================================================================

def subqg_single(ctx: DriverCtx, rng: np.random.Generator) -> np.ndarray:
    init_energy = np.full((SUBQG_BATCH_CELLS,), SUBQG_INIT_ENERGY, dtype=np.float32)
    init_phase  = np.full((SUBQG_BATCH_CELLS,), SUBQG_INIT_PHASE,  dtype=np.float32)

    _chk(ctx.dll.subqg_initialize_state_batched(
        ctx.gpu, SUBQG_BATCH_CELLS,
        init_energy.ctypes.data_as(ct.POINTER(ct.c_float)),
        init_phase.ctypes.data_as(ct.POINTER(ct.c_float)),
        SUBQG_NOISE, SUBQG_THR
    ), "subqg_initialize_state_batched")

    rng_energy = np.abs(rng.standard_normal((SUBQG_BATCH_CELLS,), dtype=np.float32))
    rng_phase  = np.abs(rng.standard_normal((SUBQG_BATCH_CELLS,), dtype=np.float32))
    rng_spin   = np.abs(rng.standard_normal((SUBQG_BATCH_CELLS,), dtype=np.float32))

    out_energy = np.empty_like(init_energy)
    out_phase  = np.empty_like(init_phase)
    out_interf = np.empty_like(init_energy)
    out_node   = np.empty((SUBQG_BATCH_CELLS,), dtype=np.int32)
    out_spin   = np.empty((SUBQG_BATCH_CELLS,), dtype=np.int32)
    out_topo   = np.empty((SUBQG_BATCH_CELLS,), dtype=np.int32)
    field_map  = np.empty((SUBQG_BATCH_CELLS,), dtype=np.float32)

    _chk(ctx.dll.subqg_simulation_step_batched(
        ctx.gpu,
        rng_energy.ctypes.data_as(ct.POINTER(ct.c_float)),
        rng_phase.ctypes.data_as(ct.POINTER(ct.c_float)),
        rng_spin.ctypes.data_as(ct.POINTER(ct.c_float)),
        SUBQG_BATCH_CELLS,
        out_energy.ctypes.data_as(ct.POINTER(ct.c_float)),
        out_phase.ctypes.data_as(ct.POINTER(ct.c_float)),
        out_interf.ctypes.data_as(ct.POINTER(ct.c_float)),
        out_node.ctypes.data_as(ct.POINTER(ct.c_int)),
        out_spin.ctypes.data_as(ct.POINTER(ct.c_int)),
        out_topo.ctypes.data_as(ct.POINTER(ct.c_int)),
        field_map.ctypes.data_as(ct.POINTER(ct.c_float)),
        SUBQG_BATCH_CELLS
    ), "subqg_simulation_step_batched")

    return field_map.copy()


def run_subqg_with_confidence(ctx: DriverCtx,
                              rng: np.random.Generator,
                              n_samples: int) -> tuple[float, np.ndarray, np.ndarray]:
    maps = [subqg_single(ctx, rng) for _ in range(max(1, n_samples))]
    maps = np.stack(maps, axis=0)
    mean_map = maps.mean(axis=0)
    std_map  = maps.std(axis=0, ddof=1) if n_samples > 1 else np.zeros_like(mean_map)
    field_score = float(np.maximum(0.0, mean_map).mean())
    return field_score, mean_map, std_map


def parse_pauli_z_terms(json_text: str) -> list[tuple[int, float]]:
    """
    Erwartet JSON-Liste von Objekten { "z_mask": int, "c": float }.
    Beispiel:
      [
        {"z_mask": 1, "c": 1.0},   # Z0
        {"z_mask": 2, "c": 1.0},   # Z1
        {"z_mask": 3, "c": 1.0}    # Z0Z1
      ]
    """
    data = json.loads(json_text)
    terms: list[tuple[int, float]] = []
    if not isinstance(data, list):
        raise ValueError("JSON muss eine Liste sein.")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "z_mask" not in item or "c" not in item:
            raise ValueError(f"Eintrag {i} ungültig. Erwartet {{'z_mask': int, 'c': float}}.")
        z = int(item["z_mask"])
        c = float(item["c"])
        if z < 0:
            raise ValueError(f"Eintrag {i}: z_mask < 0.")
        terms.append((z, c))
    if not terms:
        raise ValueError("Keine Terme definiert.")
    return terms


def vqe_energy(ctx: DriverCtx, params: np.ndarray,
               qubits: int, layers: int,
               pauli_terms: list[tuple[int, float]]) -> float:
    c_terms = (PauliZTerm * len(pauli_terms))(
        *[PauliZTerm(int(z), float(c)) for (z, c) in pauli_terms]
    )
    out_E = ct.c_float(0.0)
    _chk(ctx.dll.execute_vqe_gpu(
        ctx.gpu, qubits, layers,
        params.ctypes.data_as(ct.POINTER(ct.c_float)), params.size,
        c_terms, len(pauli_terms),
        ct.byref(out_E), ct.POINTER(ct.c_float)()
    ), "execute_vqe_gpu")
    return float(out_E.value)


def vqe_spsa(ctx: DriverCtx,
             qubits: int, layers: int,
             iters: int,
             pauli_terms: list[tuple[int, float]],
             seed: int = 1234,
             progress_cb: Optional[callable] = None) -> tuple[list[float], np.ndarray, float]:
    rng = np.random.default_rng(seed)
    num_params = qubits * 2 * layers
    theta = rng.uniform(-math.pi, math.pi, size=(num_params,)).astype(np.float32)

    best_E = float("inf")
    best_theta = theta.copy()
    energies: list[float] = []

    for k in range(1, iters + 1):
        a_k = SPSA_A / ((k + 10) ** SPSA_ALPHA)
        c_k = SPSA_C / (k ** SPSA_GAMMA)
        delta = rng.choice([-1.0, 1.0], size=theta.shape).astype(np.float32)

        thetap = theta + c_k * delta
        thetam = theta - c_k * delta

        Ep = vqe_energy(ctx, thetap, qubits, layers, pauli_terms)
        Em = vqe_energy(ctx, thetam, qubits, layers, pauli_terms)

        ghat = (Ep - Em) / (2.0 * c_k * delta)
        theta = theta - a_k * ghat.astype(np.float32)

        E = vqe_energy(ctx, theta, qubits, layers, pauli_terms)
        energies.append(E)
        if E < best_E:
            best_E = E
            best_theta = theta.copy()

        if progress_cb:
            progress_cb(k, E, best_E)

    return energies, best_theta, best_E

# =============================================================================
# Pipeline: ein Durchlauf (liefert Mess-Struktur inkl. PCA & Feldkarte)
# =============================================================================

def run_pipeline_once(ctx: DriverCtx,
                      qubits: int,
                      layers: int,
                      vqe_iters: int,
                      pauli_terms: list[tuple[int, float]],
                      lr_mode: str,
                      lr_p1: float,
                      lr_p2: float,
                      subqg_samples: int,
                      rng_seed: int = 42,
                      spsa_seed: int = 123) -> dict:
    rng = np.random.default_rng(rng_seed)

    # A) Demo-Daten
    activations = rng.standard_normal((B, S, E), dtype=np.float32)
    prototypes  = rng.standard_normal((T, E), dtype=np.float32)
    proto_before = prototypes.copy()

    buf_A  = gpu_upload(ctx, activations)
    buf_Te = gpu_upload(ctx, prototypes)

    idx_host = np.empty((B, S), dtype=np.int32)
    buf_idx = ctx.dll.allocate_gpu_memory(ctx.gpu, idx_host.nbytes)
    if not buf_idx:
        raise MemoryError("allocate idx buffer")

    # A1) Assignment
    _chk(ctx.dll.execute_dynamic_token_assignment_gpu(
        ctx.gpu, buf_A, buf_Te, buf_idx, B, S, E, T
    ), "execute_dynamic_token_assignment_gpu")

    # Indizes lesen
    idx_flat = np.empty((B * S,), dtype=np.int32)
    _chk(ctx.dll.read_gpu_to_host_blocking(
        ctx.gpu, buf_idx, 0, idx_flat.nbytes, idx_flat.ctypes.data_as(ct.c_void_p)
    ), "read idx_flat")

    # A2) Segmented Sum
    A_flat = activations.reshape(B * S, E).astype(np.float32, copy=False)
    buf_Aflat   = gpu_upload(ctx, A_flat)
    buf_idxflat = gpu_upload(ctx, idx_flat)

    proto_sums = np.zeros((T, E), dtype=np.float32)
    proto_cnts = np.zeros((T,), dtype=np.int32)
    buf_ps = gpu_upload(ctx, proto_sums)
    buf_pc = gpu_upload(ctx, proto_cnts)

    _chk(ctx.dll.execute_proto_segmented_sum_gpu(
        ctx.gpu, buf_Aflat, buf_idxflat, buf_ps, buf_pc, B * S, E, T
    ), "execute_proto_segmented_sum_gpu")

    # A3) Erster Update (konst. LR)
    _chk(ctx.dll.execute_proto_update_step_gpu(
        ctx.gpu, buf_Te, buf_ps, buf_pc, BASE_LR, E, T
    ), "execute_proto_update_step_gpu [1]")

    # B) SubQG → Feld-Score/-Karte (Konfidenz über mehrere Samples)
    field_score, mean_map, std_map = run_subqg_with_confidence(ctx, rng, n_samples=subqg_samples)
    lr_mod = lr_modulated(BASE_LR, field_score, lr_mode, lr_p1, lr_p2)

    # C) VQE (SPSA) → best_E → set_noise_level
    def _progress(k, E, bestE):
        st.write(f"[VQE] iter={k:03d} E={E:.6f} best={bestE:.6f}")

    energies, best_theta, best_E = vqe_spsa(
        ctx, qubits=qubits, layers=layers, iters=vqe_iters,
        pauli_terms=pauli_terms, seed=spsa_seed, progress_cb=_progress
    )
    new_noise = float(np.clip(SUBQG_NOISE * (1.0 + 0.25 * best_E), 0.0, 1.0))
    ctx.dll.set_noise_level(ctx.gpu, new_noise)

    # A4) Zweiter Update (modulierte LR)
    _chk(ctx.dll.execute_proto_update_step_gpu(
        ctx.gpu, buf_Te, buf_ps, buf_pc, ct.c_float(lr_mod), E, T
    ), "execute_proto_update_step_gpu [2]")

    prototypes_updated = gpu_download(ctx, buf_Te, (T, E), np.float32)
    delta = float(np.linalg.norm(prototypes_updated - proto_before))

    # PCA fit auf concat -> stabile Achsen
    X = np.vstack([proto_before, prototypes_updated])
    proj, mu, comps = pca_2d(X)
    proj_before = proj[:T]
    proj_after  = proj[T:]

    # Per-Proto-Metriken
    d_emb = np.linalg.norm(prototypes_updated - proto_before, axis=1)   # im Embedding
    d_pca = np.linalg.norm(proj_after - proj_before, axis=1)            # in PCA

    for buf in (buf_A, buf_Te, buf_idx, buf_Aflat, buf_idxflat, buf_ps, buf_pc):
        ctx.dll.free_gpu_memory(ctx.gpu, buf)

    return {
        "field_score": field_score,
        "field_map_mean": mean_map,
        "field_map_std":  std_map,
        "lr_mod": lr_mod,
        "vqe_energies": energies,
        "vqe_best_E": best_E,
        "noise_set": new_noise,
        "delta_proto_l2_total": delta,
        "pca_before": proj_before,      # (T,2)
        "pca_after":  proj_after,       # (T,2)
        "delta_per_proto_emb": d_emb,   # (T,)
        "delta_per_proto_pca": d_pca,   # (T,)
        "qubits": qubits,
        "layers": layers,
        "pauli_terms": pauli_terms,
        "lr_mode": lr_mode,
        "lr_p1": lr_p1,
        "lr_p2": lr_p2,
    }

# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Tri-Core Orchestrator ULTRA", layout="wide")
st.title("🧠 Tri-Core Orchestrator ULTRA – A (Proto) + B (SubQG) + C (VQE)")

# -------------------------------------------
# Sidebar: DLL & GPU & VQE & SubQG & LR
# -------------------------------------------
with st.sidebar:
    st.header("⚙️ Einstellungen")

    # --- Stabiler DLL-Pfad NUR im Session State ---
    if "dll_path" not in st.session_state:
        qp_dll = st.query_params.get("dll", ["CipherCore_OpenCl.dll"])[0]
        st.session_state["dll_path"] = qp_dll if (":" not in qp_dll and "\\" not in qp_dll) else "CipherCore_OpenCl.dll"

    dll_path_str = st.text_input("DLL-Pfad", value=st.session_state["dll_path"])
    st.session_state["dll_path"] = dll_path_str
    dll_path = Path(dll_path_str)

    dll_loaded = False
    dll_handle: Optional[ct.CDLL] = None
    try:
        dll_handle = load_dll(dll_path)
        dll_loaded = True
        st.caption(f"✅ DLL geladen: {dll_path.resolve()}")
    except Exception as e:
        st.error(f"DLL konnte nicht geladen werden: {e}")

    colA, colB = st.columns(2)
    with colA:
        list_btn = st.button("🧭 GPUs auflisten", use_container_width=True, disabled=not dll_loaded)
    with colB:
        bench_btn = st.button("📊 GPU-Benchmark", use_container_width=True, disabled=not dll_loaded)
    auto_btn = st.button("🏎️ Schnellste GPU wählen (Auto)", use_container_width=True, disabled=not dll_loaded)

    # GPU-Index (Session State + Query nur für gpu)
    if "gpu_index" not in st.session_state:
        st.session_state["gpu_index"] = int(st.query_params.get("gpu", [0])[0])

    gpu_index = st.number_input("🎛️ GPU-Index", min_value=0, value=st.session_state["gpu_index"], step=1)
    st.session_state["gpu_index"] = int(gpu_index)
    st.query_params["gpu"] = str(st.session_state["gpu_index"])  # Nur GPU in URL, nicht DLL!

    st.divider()
    st.subheader("VQE / SPSA")
    vqe_qubits = st.number_input("Qubits", min_value=2, max_value=64, value=VQE_QUBITS_DEFAULT, step=1)
    vqe_layers = st.number_input("Layers", min_value=1, max_value=16, value=VQE_LAYERS_DEFAULT, step=1)
    vqe_iters  = st.number_input("SPSA-Iterationen", min_value=1, max_value=500, value=SPSA_ITERS_DEFAULT, step=1)

    st.subheader("SubQG Konfidenz")
    subqg_samples = st.number_input("Samples pro Epoche (Konfidenz)", min_value=1, max_value=64, value=5, step=1)

    st.subheader("LR-Kopplung")
    lr_mode = st.selectbox("Modus", ["exp", "sigmoid", "tanh", "linear"], index=0)
    if lr_mode == "exp":
        lr_p1 = st.number_input("p1 (Steilheit)", value=1.0, step=0.1)
        lr_p2 = st.number_input("p2 (Gain)", value=1.0, step=0.1)
    elif lr_mode == "sigmoid":
        lr_p1 = st.number_input("p1 (Steilheit)", value=10.0, step=0.5)
        lr_p2 = st.number_input("p2 (Center)", value=0.1, step=0.05)
    elif lr_mode == "tanh":
        lr_p1 = st.number_input("p1 (Steilheit)", value=5.0, step=0.5)
        lr_p2 = st.number_input("p2 (Center)", value=0.1, step=0.05)
    else:  # linear
        lr_p1 = st.number_input("p1 (Steigung)", value=1.0, step=0.1)
        lr_p2 = st.number_input("p2 (reserviert)", value=0.0, step=0.1)

    st.divider()
    st.subheader("Pauli-Z Hamiltonian (JSON)")
    default_terms = json.dumps([
        {"z_mask": 1, "c": 1.0},  # Z0
        {"z_mask": 2, "c": 1.0},  # Z1
        {"z_mask": 3, "c": 1.0},  # Z0Z1
    ], indent=2)
    if "hamilton_json" not in st.session_state:
        st.session_state["hamilton_json"] = default_terms
    hjson = st.text_area("Terme (Liste von Objekte mit z_mask und c)",
                         value=st.session_state["hamilton_json"], height=160)
    st.session_state["hamilton_json"] = hjson

    st.divider()
    run_once = st.button("▶️ Run once", use_container_width=True, disabled=not dll_loaded)
    epochs = st.number_input("Epochen (Mehrfachläufe)", min_value=1, max_value=200, value=1, step=1)
    run_epochs = st.button("🚀 Run epochs", use_container_width=True, disabled=not dll_loaded)

# Layout-Spalten
left, mid, right = st.columns([1.7, 1.5, 1.1])

# Session-Init
if "history" not in st.session_state:
    st.session_state["history"] = {
        "epoch": [],
        "field_score": [],
        "vqe_best_E": [],
        "delta_proto_l2_total": [],
        "noise_set": [],
    }

if "bench" not in st.session_state:
    st.session_state["bench"] = []  # (gpu, ms)

if "epoch_data" not in st.session_state:
    st.session_state["epoch_data"] = []  # dicts mit PCA, Feld & Proto-Deltas

# GPU Werkzeuge
if 'dll_path' in st.session_state and dll_loaded and list_btn:
    try:
        def try_initialize(dll: ct.CDLL, gpu_i: int) -> bool:
            ok = dll.initialize_gpu(gpu_i)
            if ok == 1:
                try:
                    dll.finish_gpu(gpu_i)
                finally:
                    dll.shutdown_gpu(gpu_i)
                return True
            return False

        found: list[int] = []
        i = 0
        while True:
            ok = False
            try:
                ok = try_initialize(dll_handle, i)
            except Exception:
                ok = False
            if ok:
                found.append(i)
                i += 1
                continue
            if found:
                break
            i += 1
            if i >= 8:
                break

        if found:
            st.info("🧭 Verfügbare GPUs: " + ", ".join(map(str, found)))
        else:
            st.warning("Keine initialisierbaren GPUs gefunden.")
    except Exception as e:
        st.error(f"Auflisten fehlgeschlagen: {e}")

if 'dll_path' in st.session_state and dll_loaded and bench_btn:
    try:
        scores: list[tuple[int, float]] = []
        for g in range(8):
            try:
                rng = np.random.default_rng(123)
                A = rng.standard_normal((B, S, E), dtype=np.float32)
                Tm = rng.standard_normal((T, E), dtype=np.float32)
                if dll_handle.initialize_gpu(g) != 1:
                    continue
                try:
                    def up(x: np.ndarray) -> ct.c_void_p:
                        buf = dll_handle.allocate_gpu_memory(g, x.nbytes)
                        if not buf:
                            raise MemoryError("alloc")
                        ok = dll_handle.write_host_to_gpu_blocking(g, buf, 0, x.nbytes, x.ctypes.data_as(ct.c_void_p))
                        if ok != 1:
                            raise RuntimeError("H2D")
                        return buf

                    buf_A = up(A)
                    buf_Te = up(Tm)
                    buf_idx = dll_handle.allocate_gpu_memory(g, np.empty((B, S), np.int32).nbytes)
                    if not buf_idx:
                        raise MemoryError("alloc idx")
                    Aflat = A.reshape(B*S, E)
                    idxflat = np.empty((B*S,), dtype=np.int32)
                    buf_Af = up(Aflat)
                    buf_If = up(idxflat)
                    buf_ps = up(np.zeros((T, E), dtype=np.float32))
                    buf_pc = up(np.zeros((T,), dtype=np.int32))

                    import time
                    t0 = time.perf_counter()
                    if dll_handle.execute_dynamic_token_assignment_gpu(g, buf_A, buf_Te, buf_idx, B, S, E, T) != 1:
                        raise RuntimeError("assign")
                    if dll_handle.read_gpu_to_host_blocking(g, buf_idx, 0, idxflat.nbytes, idxflat.ctypes.data_as(ct.c_void_p)) != 1:
                        raise RuntimeError("D2H idx")
                    if dll_handle.execute_proto_segmented_sum_gpu(g, buf_Af, buf_If, buf_ps, buf_pc, B*S, E, T) != 1:
                        raise RuntimeError("segsum")
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    scores.append((g, dt_ms))

                    for buf in (buf_A, buf_Te, buf_idx, buf_Af, buf_If, buf_ps, buf_pc):
                        dll_handle.free_gpu_memory(g, buf)
                finally:
                    try:
                        dll_handle.finish_gpu(g)
                    finally:
                        dll_handle.shutdown_gpu(g)
            except Exception:
                pass

        st.session_state["bench"] = scores
        if scores:
            with right:
                fig_b, ax_b = plt.subplots()
                xs = [str(g) for g, _ in scores]
                ys = [ms for _, ms in scores]
                ax_b.bar(xs, ys)
                ax_b.set_title("GPU Microbenchmark (ms) – kleiner ist besser")
                ax_b.set_xlabel("GPU-Index")
                ax_b.set_ylabel("Zeit [ms]")
                st.pyplot(fig_b, clear_figure=True)
                st.caption("Benchmark: Assignment + Segmented Sum.")
        else:
            st.info("Keine Benchmarks erstellt.")
    except Exception as e:
        st.error(f"Benchmark fehlgeschlagen: {e}")

if 'dll_path' in st.session_state and dll_loaded and auto_btn:
    try:
        candidates = []
        for g in range(8):
            try:
                ok = dll_handle.initialize_gpu(g)
                if ok == 1:
                    dll_handle.finish_gpu(g); dll_handle.shutdown_gpu(g)
                    candidates.append(g)
            except Exception:
                pass
        if not candidates:
            st.warning("Für Auto wurde keine GPU gefunden.")
        else:
            scores: list[tuple[float, int]] = []
            for g in candidates:
                try:
                    rng = np.random.default_rng(123)
                    A = rng.standard_normal((B, S, E), dtype=np.float32)
                    Tm = rng.standard_normal((T, E), dtype=np.float32)
                    if dll_handle.initialize_gpu(g) != 1:
                        continue
                    try:
                        def up(x: np.ndarray) -> ct.c_void_p:
                            buf = dll_handle.allocate_gpu_memory(g, x.nbytes)
                            if not buf:
                                raise MemoryError("alloc")
                            ok = dll_handle.write_host_to_gpu_blocking(g, buf, 0, x.nbytes, x.ctypes.data_as(ct.c_void_p))
                            if ok != 1:
                                raise RuntimeError("H2D")
                            return buf
                        buf_A = up(A); buf_Te = up(Tm)
                        buf_idx = dll_handle.allocate_gpu_memory(g, np.empty((B, S), np.int32).nbytes)
                        Aflat = A.reshape(B*S, E); idxflat = np.empty((B*S,), dtype=np.int32)
                        buf_Af = up(Aflat); buf_If = up(idxflat)
                        buf_ps = up(np.zeros((T, E), dtype=np.float32))
                        buf_pc = up(np.zeros((T,), dtype=np.int32))

                        import time
                        t0 = time.perf_counter()
                        if dll_handle.execute_dynamic_token_assignment_gpu(g, buf_A, buf_Te, buf_idx, B, S, E, T) != 1:
                            raise RuntimeError("assign")
                        if dll_handle.read_gpu_to_host_blocking(g, buf_idx, 0, idxflat.nbytes, idxflat.ctypes.data_as(ct.c_void_p)) != 1:
                            raise RuntimeError("D2H idx")
                        if dll_handle.execute_proto_segmented_sum_gpu(g, buf_Af, buf_If, buf_ps, buf_pc, B*S, E, T) != 1:
                            raise RuntimeError("segsum")
                        dt_ms = (time.perf_counter() - t0) * 1000.0
                        scores.append((dt_ms, g))

                        for buf in (buf_A, buf_Te, buf_idx, buf_Af, buf_If, buf_ps, buf_pc):
                            dll_handle.free_gpu_memory(g, buf)
                    finally:
                        dll_handle.finish_gpu(g); dll_handle.shutdown_gpu(g)
                except Exception:
                    pass
            if scores:
                scores.sort()
                best_ms, best_gpu = scores[0]
                st.success(f"🏆 Schnellste GPU: {best_gpu} ({best_ms:.3f} ms)")
                st.session_state["gpu_index"] = int(best_gpu)
                st.query_params["gpu"] = str(best_gpu)
                st.session_state["bench"] = [(g, ms) for (ms, g) in scores]
            else:
                st.warning("Auto-Auswahl: keine messbaren Kandidaten.")
    except Exception as e:
        st.error(f"Auto-Auswahl fehlgeschlagen: {e}")

# -----------------------------------------------------------------------------
# Pipeline-Steuerung + Persistenz
# -----------------------------------------------------------------------------

with st.expander("💾 Persistenz – Läufe speichern & laden", expanded=False):
    col1, col2 = st.columns([1,1])
    with col1:
        fname = st.text_input("Dateiname (.npz)", value="tri_core_session.npz")
        if st.button("💿 Speichern (.npz)"):
            h = st.session_state["history"]
            ed = st.session_state["epoch_data"]
            pack = {
                "history_epoch": np.array(h["epoch"], dtype=np.int32),
                "history_field_score": np.array(h["field_score"], dtype=np.float32),
                "history_vqe_best_E": np.array(h["vqe_best_E"], dtype=np.float32),
                "history_delta_proto": np.array(h["delta_proto_l2_total"], dtype=np.float32),
                "history_noise_set": np.array(h["noise_set"], dtype=np.float32),
            }
            ed_json = []
            for item in ed:
                ed_json.append({
                    "pca_before": item["pca_before"].tolist(),
                    "pca_after": item["pca_after"].tolist(),
                    "field_map_mean": item["field_map_mean"].tolist(),
                    "field_map_std": item["field_map_std"].tolist(),
                    "delta_per_proto_emb": item["delta_per_proto_emb"].tolist(),
                    "delta_per_proto_pca": item["delta_per_proto_pca"].tolist(),
                })
            pack_bytes = json.dumps(ed_json).encode("utf-8")

            params = {
                "qubits": int(vqe_qubits),
                "layers": int(vqe_layers),
                "spsa_iters": int(vqe_iters),
                "lr_mode": lr_mode,
                "lr_p1": float(lr_p1),
                "lr_p2": float(lr_p2),
                "subqg_samples": int(subqg_samples),
                "dll": str(st.session_state["dll_path"]),
                "gpu": int(st.session_state["gpu_index"]),
                "pauli_terms": st.session_state["hamilton_json"],
            }
            params_bytes = json.dumps(params, ensure_ascii=False).encode("utf-8")

            bio = io.BytesIO()
            np.savez_compressed(
                bio,
                **pack,
                epoch_data_json=pack_bytes,
                params_json=params_bytes,
            )
            st.download_button(
                "⬇️ Sitzung herunterladen",
                data=bio.getvalue(),
                file_name=fname,
                mime="application/octet-stream"
            )
            st.success("Sitzung gepackt.")

    with col2:
        up = st.file_uploader("📥 Sitzung laden (.npz)", type=["npz"])
        if up is not None and st.button("🔁 Laden"):
            try:
                data = np.load(up, allow_pickle=False)
                st.session_state["history"] = {
                    "epoch": data["history_epoch"].astype(int).tolist(),
                    "field_score": data["history_field_score"].astype(float).tolist(),
                    "vqe_best_E": data["history_vqe_best_E"].astype(float).tolist(),
                    "delta_proto_l2_total": data["history_delta_proto"].astype(float).tolist(),
                    "noise_set": data["history_noise_set"].astype(float).tolist(),
                }
                ed_json = json.loads(bytes(data["epoch_data_json"]).decode("utf-8"))
                st.session_state["epoch_data"] = []
                for item in ed_json:
                    st.session_state["epoch_data"].append({
                        "pca_before": np.array(item["pca_before"], dtype=np.float32),
                        "pca_after": np.array(item["pca_after"], dtype=np.float32),
                        "field_map_mean": np.array(item["field_map_mean"], dtype=np.float32),
                        "field_map_std": np.array(item["field_map_std"], dtype=np.float32),
                        "delta_per_proto_emb": np.array(item["delta_per_proto_emb"], dtype=np.float32),
                        "delta_per_proto_pca": np.array(item["delta_per_proto_pca"], dtype=np.float32),
                    })
                params = json.loads(bytes(data["params_json"]).decode("utf-8"))
                st.session_state["gpu_index"] = int(params.get("gpu", st.session_state["gpu_index"]))
                st.query_params["gpu"] = str(st.session_state["gpu_index"])
                st.session_state["dll_path"] = params.get("dll", st.session_state["dll_path"])
                st.session_state["hamilton_json"] = params.get("pauli_terms", st.session_state["hamilton_json"])
                st.success("Sitzung geladen.")
                st.info(f"Geladene Parameter: {params}")
            except Exception as e:
                st.error(f"Laden fehlgeschlagen: {e}")

# Ausführung
def run_and_store(n_epochs: int) -> None:
    assert dll_handle is not None
    _chk(dll_handle.initialize_gpu(int(st.session_state["gpu_index"])), "initialize_gpu")
    ctx = DriverCtx(dll=dll_handle, gpu=int(st.session_state["gpu_index"]))
    try:
        # Hamiltonian parsen
        try:
            terms = parse_pauli_z_terms(st.session_state["hamilton_json"])
        except Exception as e:
            st.error(f"Hamiltonian-JSON ungültig: {e}")
            return

        for _ in range(n_epochs):
            ep_no = len(st.session_state["history"]["epoch"]) + 1
            st.write(f"### 🧪 Epoche {ep_no}")
            res = run_pipeline_once(
                ctx,
                qubits=int(vqe_qubits),
                layers=int(vqe_layers),
                vqe_iters=int(vqe_iters),
                pauli_terms=terms,
                lr_mode=lr_mode, lr_p1=float(lr_p1), lr_p2=float(lr_p2),
                subqg_samples=int(subqg_samples),
                rng_seed=42 + ep_no, spsa_seed=123 + ep_no,
            )

            # History
            h = st.session_state["history"]
            h["epoch"].append(ep_no)
            h["field_score"].append(res["field_score"])
            h["vqe_best_E"].append(res["vqe_best_E"])
            h["delta_proto_l2_total"].append(res["delta_proto_l2_total"])
            h["noise_set"].append(res["noise_set"])

            # Epoche-Daten
            st.session_state["epoch_data"].append({
                "pca_before": res["pca_before"],
                "pca_after":  res["pca_after"],
                "field_map_mean": res["field_map_mean"],
                "field_map_std":  res["field_map_std"],
                "delta_per_proto_emb": res["delta_per_proto_emb"],
                "delta_per_proto_pca": res["delta_per_proto_pca"],
            })

            # Live-Plots
            with left:
                fig1, ax1 = plt.subplots()
                ax1.plot(res["vqe_energies"])
                ax1.set_title(f"VQE Energie pro Iteration (Epoche {ep_no})")
                ax1.set_xlabel("Iteration")
                ax1.set_ylabel("Energie")
                st.pyplot(fig1, clear_figure=True)

            with right:
                st.metric("Feld-Score (mean)", f"{res['field_score']:.4f}")
                st.metric("VQE best E", f"{res['vqe_best_E']:.6f}")
                st.metric("LR (moduliert)", f"{res['lr_mod']:.4f}")
                st.metric("Noise gesetzt", f"{res['noise_set']:.4f}")
                st.metric("ΔProto L2 (gesamt)", f"{res['delta_proto_l2_total']:.6f}")

        st.success("✅ Läufe abgeschlossen.")
    finally:
        try:
            dll_handle.finish_gpu(int(st.session_state["gpu_index"]))
        finally:
            dll_handle.shutdown_gpu(int(st.session_state["gpu_index"]))

if dll_loaded and run_once:
    run_and_store(1)

if dll_loaded and run_epochs:
    run_and_store(int(epochs))

# -----------------------------------------------------------------------------
# Zeitverlauf-Kurven (global) – robust auch für nur 1 Epoche
# -----------------------------------------------------------------------------
h = st.session_state["history"]

def _plot_singleton(ax, x_val: float, y_val: float, title: str, xlab: str, ylab: str):
    ax.scatter([x_val], [y_val], s=60)
    pad_x = 0.5
    pad_y = max(1e-4, abs(float(y_val)) * 0.15)
    ax.set_xlim(x_val - pad_x, x_val + pad_x)
    ax.set_ylim(float(y_val) - pad_y, float(y_val) + pad_y)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

with left:
    epochs_arr = np.asarray(h["epoch"], dtype=float)
    fs = np.asarray(h["field_score"], dtype=float)
    dp = np.asarray(h["delta_proto_l2_total"], dtype=float)

    if epochs_arr.size == 0:
        st.info("Noch keine Epochen gelaufen.")
    else:
        # Feld-Score
        fig_fs, ax_fs = plt.subplots()
        if epochs_arr.size == 1:
            _plot_singleton(ax_fs, epochs_arr[0], fs[0],
                            "Feld-Score (mean) über Epochen",
                            "Epoche", "Feld-Score")
        else:
            ax_fs.plot(epochs_arr, fs, marker="o")
            ax_fs.set_title("Feld-Score (mean) über Epochen")
            ax_fs.set_xlabel("Epoche"); ax_fs.set_ylabel("Feld-Score")
            ax_fs.grid(True, alpha=0.3)
        st.pyplot(fig_fs, clear_figure=True)

        # ΔPrototypen L2
        fig_dp, ax_dp = plt.subplots()
        if epochs_arr.size == 1:
            _plot_singleton(ax_dp, epochs_arr[0], dp[0],
                            "ΔPrototypen L2 gesamt über Epochen",
                            "Epoche", "||Δ||₂")
        else:
            ax_dp.plot(epochs_arr, dp, marker="o")
            ax_dp.set_title("ΔPrototypen L2 gesamt über Epochen")
            ax_dp.set_xlabel("Epoche"); ax_dp.set_ylabel("||Δ||₂")
            ax_dp.grid(True, alpha=0.3)
        st.pyplot(fig_dp, clear_figure=True)

# -----------------------------------------------------------------------------
# PCA: Interaktiv + Zeitverlauf + GIF-Export (Slider-Fallback bei 1 Epoche)
# -----------------------------------------------------------------------------
ed = st.session_state["epoch_data"]

with mid:
    st.subheader("🧭 PCA – Interaktive Auswahl, Zeitverlauf & Export")

    if not ed:
        st.info("Noch keine Epochen gelaufen. Starte einen Run.")
    else:
        max_ep = len(ed)
        if max_ep <= 1:
            sel_ep = 1
            idx = 0
            st.caption("ℹ️ Nur eine Epoche vorhanden – Slider deaktiviert.")
        else:
            sel_ep = st.slider("Epoche (PCA/Heatmap anzeigen)",
                               min_value=1, max_value=max_ep, value=max_ep, step=1)
            idx = sel_ep - 1

        sel_proto = st.number_input("Prototyp-ID für Hervorhebung", min_value=0, max_value=T-1, value=0, step=1)

        p_before = ed[idx]["pca_before"]
        p_after  = ed[idx]["pca_after"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=p_before[:, 0], y=p_before[:, 1],
            mode="markers", name="Vorher",
            hovertemplate="Proto %{customdata}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}",
            customdata=np.arange(T),
            marker=dict(size=8, opacity=0.7)
        ))
        fig.add_trace(go.Scatter(
            x=p_after[:, 0], y=p_after[:, 1],
            mode="markers", name="Nachher",
            hovertemplate="Proto %{customdata}<br>PC1=%{x:.3f}<br>PC2=%{y:.3f}",
            customdata=np.arange(T),
            marker=dict(size=8, symbol="x", opacity=0.9)
        ))
        for t in range(T):
            fig.add_trace(go.Scatter(
                x=[p_before[t, 0], p_after[t, 0]],
                y=[p_before[t, 1], p_after[t, 1]],
                mode="lines",
                line=dict(width=1),
                name=f"Δ Proto {t}",
                showlegend=False,
                hoverinfo="skip"
            ))

        traj_x, traj_y = [], []
        for j in range(sel_ep):
            traj_x.append(ed[j]["pca_after"][sel_proto, 0])
            traj_y.append(ed[j]["pca_after"][sel_proto, 1])
        fig.add_trace(go.Scatter(
            x=traj_x, y=traj_y, mode="lines+markers",
            name=f"Trajektorie Proto {sel_proto}",
            marker=dict(size=10),
        ))

        fig.update_layout(
            title=f"PCA Vorher/Nachher – Epoche {sel_ep}",
            xaxis_title="PC1",
            yaxis_title="PC2",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Export des PCA-Zeitverlaufs als GIF
        st.markdown("**🎬 Export PCA-Zeitverlauf**")
        fps = st.slider("FPS", min_value=1, max_value=30, value=6, step=1)
        width_px = st.number_input("Breite (px)", min_value=400, max_value=1920, value=800, step=50)
        export_btn = st.button("📤 Export (GIF)")

        def render_pca_frame(ep_idx: int, highlight: int | None = None, size=(800, 600)) -> Image.Image:
            pb = ed[ep_idx]["pca_before"]; pa = ed[ep_idx]["pca_after"]
            figf, axf = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)
            axf.scatter(pb[:, 0], pb[:, 1], s=40, alpha=0.6, label="Vorher")
            axf.scatter(pa[:, 0], pa[:, 1], s=40, marker="x", label="Nachher")
            for t in range(T):
                axf.plot([pb[t, 0], pa[t, 0]], [pb[t, 1], pa[t, 1]], linewidth=1, alpha=0.8, color="black")
            if highlight is not None:
                tx = [ed[j]["pca_after"][highlight, 0] for j in range(ep_idx + 1)]
                ty = [ed[j]["pca_after"][highlight, 1] for j in range(ep_idx + 1)]
                axf.plot(tx, ty, linewidth=2)
                axf.scatter([tx[-1]], [ty[-1]], s=80)
            axf.set_title(f"PCA Vorher/Nachher – Epoche {ep_idx+1}")
            axf.set_xlabel("PC1"); axf.set_ylabel("PC2")
            axf.legend(); figf.tight_layout()
            buf = io.BytesIO(); figf.savefig(buf, format="png"); plt.close(figf); buf.seek(0)
            return Image.open(buf).convert("RGB")

        if export_btn:
            frames = [render_pca_frame(i, highlight=sel_proto, size=(int(width_px), int(width_px*0.75)))
                      for i in range(len(ed))]
            gif_buf = io.BytesIO()
            imageio.mimsave(gif_buf, frames, format="GIF", fps=fps)
            st.download_button("⬇️ GIF herunterladen", data=gif_buf.getvalue(),
                               file_name="pca_trajectory.gif", mime="image/gif")

# -----------------------------------------------------------------------------
# Heatmap: aktuelle + Historie + Konfidenz (mean/std/Sigma)
# -----------------------------------------------------------------------------
with right:
    st.subheader("🌡️ SubQG-Heatmap (Konfidenz)")
    if not ed:
        st.info("Keine Feldkarten vorhanden.")
    else:
        # sel_ep ist oben robust gesetzt
        fmap_mean = square_reshape(ed[sel_ep-1]["field_map_mean"])
        fmap_std  = square_reshape(ed[sel_ep-1]["field_map_std"])
        sigma = np.zeros_like(fmap_mean)
        nz = fmap_std > 1e-8
        sigma[nz] = np.abs(fmap_mean[nz]) / fmap_std[nz]

        tabs = st.tabs(["Mean", "Std", "Sigma (|mean|/std)"])
        with tabs[0]:
            fig_h, ax_h = plt.subplots()
            im = ax_h.imshow(fmap_mean, aspect="equal")
            ax_h.set_title(f"Feldkarte (mean) Epoche {sel_ep}")
            fig_h.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
            st.pyplot(fig_h, clear_figure=True)
        with tabs[1]:
            fig_s, ax_s = plt.subplots()
            im = ax_s.imshow(fmap_std, aspect="equal")
            ax_s.set_title(f"Feldkarte (std) Epoche {sel_ep}")
            fig_s.colorbar(im, ax=ax_s, fraction=0.046, pad=0.04)
            st.pyplot(fig_s, clear_figure=True)
        with tabs[2]:
            fig_sig, ax_sig = plt.subplots()
            im = ax_sig.imshow(sigma, aspect="equal")
            ax_sig.set_title(f"Feldkarte (Sigma) Epoche {sel_ep}")
            fig_sig.colorbar(im, ax=ax_sig, fraction=0.046, pad=0.04)
            st.pyplot(fig_sig, clear_figure=True)

# Historie-Raster (unterhalb) – robust bei 1 Epoche (kein Slider)
st.divider()
st.subheader("🗂️ Heatmap-Historie (Mean, letzte N)")
if not ed:
    st.info("Noch keine Feldkarten in der Historie.")
else:
    if len(ed) <= 1:
        n_show = 1
        st.caption("ℹ️ Nur eine Heatmap vorhanden – Anzahl-Auswahl deaktiviert.")
    else:
        n_show = st.slider("Anzahl letzter Heatmaps (Mean)",
                           min_value=1, max_value=len(ed), value=min(6, len(ed)))
    cols = 3 if n_show >= 3 else 2
    rows = int(np.ceil(n_show / cols))
    idxs = list(range(len(ed)-n_show, len(ed)))
    fig_grid, axes = plt.subplots(rows, cols, figsize=(cols*3.6, rows*3.2))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    k = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if k < len(idxs):
                eidx = idxs[k]
                fmap = square_reshape(ed[eidx]["field_map_mean"])
                im = ax.imshow(fmap, aspect="equal")
                ax.set_title(f"Ep {eidx+1}")
                ax.set_xticks([]); ax.set_yticks([])
                k += 1
            else:
                ax.axis("off")
    fig_grid.tight_layout()
    st.pyplot(fig_grid, clear_figure=True)

# -----------------------------------------------------------------------------
# Per-Proto-Metriken (Δ im Embedding & Δ in PCA)
# -----------------------------------------------------------------------------
st.divider()
st.subheader("📏 Per-Proto-Metriken")
if not ed:
    st.info("Noch keine Per-Proto-Metriken vorhanden.")
else:
    # 🔧 Robust gegen len(ed) == 1 (kein Slider mit min == max)
    if len(ed) <= 1:
        sel_ep_metrics = 1
        st.caption("ℹ️ Nur eine Epoche vorhanden – Slider deaktiviert.")
    else:
        sel_ep_metrics = st.slider(
            "Epoche für Metrik-Ansicht",
            min_value=1,
            max_value=len(ed),
            value=len(ed),
            step=1,
            key="metric_ep",
        )

    mi = sel_ep_metrics - 1
    dm_emb = ed[mi]["delta_per_proto_emb"]
    dm_pca = ed[mi]["delta_per_proto_pca"]

    table = np.stack([np.arange(T), dm_emb, dm_pca], axis=1)
    sort_by = st.selectbox("Sortieren nach", ["Δ Embedding", "Δ PCA"])
    order = np.argsort(-dm_emb) if sort_by == "Δ Embedding" else np.argsort(-dm_pca)
    table_sorted = table[order]
    st.write("Top-Prototypen (absteigend):")
    st.dataframe(
        {
            "Proto": table_sorted[:, 0].astype(int),
            "Δ Embedding": np.round(table_sorted[:, 1].astype(float), 6),
            "Δ PCA": np.round(table_sorted[:, 2].astype(float), 6),
        },
        use_container_width=True,
    )

    fig_be, ax_be = plt.subplots()
    ax_be.bar(np.arange(T), dm_emb)
    ax_be.set_title(f"Δ pro Proto (Embedding) – Epoche {sel_ep_metrics}")
    ax_be.set_xlabel("Proto-ID"); ax_be.set_ylabel("||Δ||₂ (Embedding)")
    st.pyplot(fig_be, clear_figure=True)

    fig_bp, ax_bp = plt.subplots()
    ax_bp.bar(np.arange(T), dm_pca)
    ax_bp.set_title(f"Δ pro Proto (PCA) – Epoche {sel_ep_metrics}")
    ax_bp.set_xlabel("Proto-ID"); ax_bp.set_ylabel("||Δ||₂ (PCA)")
    st.pyplot(fig_bp, clear_figure=True)


# -----------------------------------------------------------------------------
# Export JSON (kompakte Zusammenfassung)
# -----------------------------------------------------------------------------
export = {
    "epoch": h["epoch"],
    "field_score_mean": h["field_score"],
    "vqe_best_E": h["vqe_best_E"],
    "delta_proto_l2_total": h["delta_proto_l2_total"],
    "noise_set": h["noise_set"],
    "params": {
        "qubits": int(vqe_qubits),
        "layers": int(vqe_layers),
        "spsa_iters": int(vqe_iters),
        "B": B, "S": S, "E": E, "T": T,
        "SUBQG_BATCH_CELLS": SUBQG_BATCH_CELLS,
        "subqg_samples": int(subqg_samples),
        "lr_mode": lr_mode, "lr_p1": float(lr_p1), "lr_p2": float(lr_p2),
        "pauli_terms": st.session_state["hamilton_json"],
    }
}
st.download_button(
    "📥 Ergebnisse als JSON herunterladen",
    data=json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name="tri_core_results.json",
    mime="application/json",
)

# Persistierte Benchmarks (falls vorhanden)
if st.session_state["bench"]:
    with right:
        fig_b2, ax_b2 = plt.subplots()
        xs = [str(g) for (g, _) in st.session_state["bench"]]
        ys = [ms for (_, ms) in st.session_state["bench"]]
        ax_b2.bar(xs, ys)
        ax_b2.set_title("GPU Microbenchmark (ms) – kleiner ist besser")
        ax_b2.set_xlabel("GPU-Index")
        ax_b2.set_ylabel("Zeit [ms]")
        st.pyplot(fig_b2, clear_figure=True)
        st.caption("Persistierte Benchmarks (letzte Auto- oder Bench-Messung).")
