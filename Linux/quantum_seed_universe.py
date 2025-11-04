#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
quantum_seed_universe_gpu_first.py
==================================
GPU-first Mini-Universum (deterministisch, quanteninspiriert) für deine libCC_OpenCl.so.

Kernidee:
- Komplexes y = U @ x wird *ausschließlich auf der GPU* gerechnet.
- Da die DLL reelle Matmul-Exporte bereitstellt (execute_matmul_on_gpu, ...),
  zerlegen wir (U_r + i U_i)(x_r + i x_i) in vier reelle GPU-Multiplikationen:
    y_r = U_r @ x_r - U_i @ x_i
    y_i = U_r @ x_i + U_i @ x_r
- Datenlayout: float32 (reell). Komplexe numpy-Arrays (complex128) werden
  in reelle/imag-Teile (float32) gesplittet.

Voraussetzungen:
- Python 3.12
- numpy, matplotlib
- deine libCC_OpenCl.so (LD_LIBRARY_PATH auf build/ setzen oder vollständigen Pfad mit --dll übergeben)

CLI-Beispiele:
- Reine Evolution (n=256, 120 Schritte, GPU über DLL):
  python3 quantum_seed_universe_gpu_first.py --backend dll --dll ./build/libCC_OpenCl.so --n 256 --steps 120
- OTOC-Modus:
  python3 quantum_seed_universe_gpu_first.py otoc --backend dll --dll ./build/libCC_OpenCl.so --n 128 --steps 80
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable, Optional

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


# -----------------------------
# Fehlerklassen
# -----------------------------
class BackendError(RuntimeError):
    pass


class DLLBindingError(BackendError):
    pass


class DimensionError(ValueError):
    pass


# -----------------------------------------
# Konfiguration
# -----------------------------------------
@dataclass(slots=True)
class Config:
    n: int = 128
    steps: int = 200
    seed: str = "Ursprung"
    backend: Literal["dll"] = "dll"
    dll_path: Optional[Path] = None
    gpu_index: int = 0
    plot_every: int = 10
    otoc: bool = False


# ---------------------------------------
# Backend-Protokoll
# ---------------------------------------
@runtime_checkable
class Backend(Protocol):
    def matvec(self, U: np.ndarray, x: np.ndarray) -> np.ndarray:
        ...


# ------------------------------------------------------
# DLL-Backend (GPU-first, kein CPU-Fallback)
# ------------------------------------------------------
class DLLBackend:
    """
    Bindet die folgenden Exporte (aus 'nm -D' Liste):
      - initialize_gpu(int) -> int
      - finish_gpu(int) -> int
      - shutdown_gpu(int) -> void
      - allocate_gpu_memory(int, size_t) -> void*
      - free_gpu_memory(int, void*)
      - write_host_to_gpu_blocking(int, void*, size_t, size_t, const void*) -> int
      - read_gpu_to_host_blocking (dito) -> int
      - execute_matmul_on_gpu(int, void* A, void* B, void* C, int M, int N, int K) -> int
      - execute_matmul_batched_on_gpu(int, void* A, void* B, void* C, int BATCH, int M, int N, int K) -> int
        (Batched-Fallback mit BATCH=1, falls single-API mal nicht verfügbar/anders signiert ist)

    Annahmen:
      * Row-major float32.
      * A: (M,N), B: (N,K), C: (M,K). Für Vektor setzt K=1 und B ist (N,1).
    """

    def __init__(self, dll_path: Path, gpu_index: int = 0) -> None:
        if not dll_path or not dll_path.exists():
            raise DLLBindingError(f"Pfad nicht gefunden: {dll_path}")
        try:
            self.lib = ct.CDLL(str(dll_path))
        except OSError as e:
            raise DLLBindingError(f".so konnte nicht geladen werden: {e}") from e

        self.gpu = int(gpu_index)

        # ——— Signaturen binden ———
        self._bind_core()
        self._bind_matmul_variants()

        # ——— GPU initialisieren ———
        rc = self.lib.initialize_gpu(self.gpu)
        if rc != 1 and rc != 0:
            # dein initialize_gpu gibt 1 bei Erfolg aus; PoCL-Pfade geben manchmal 1/0
            raise DLLBindingError(f"initialize_gpu({self.gpu}) rc={rc}")

    # ---------- Bindings ----------
    def _bind_core(self) -> None:
        self.lib.initialize_gpu.argtypes = [ct.c_int]
        self.lib.initialize_gpu.restype = ct.c_int
        self.lib.finish_gpu.argtypes = [ct.c_int]
        self.lib.finish_gpu.restype = ct.c_int
        self.lib.shutdown_gpu.argtypes = [ct.c_int]
        self.lib.shutdown_gpu.restype = None

        self.lib.allocate_gpu_memory.argtypes = [ct.c_int, ct.c_size_t]
        self.lib.allocate_gpu_memory.restype = ct.c_void_p
        self.lib.free_gpu_memory.argtypes = [ct.c_int, ct.c_void_p]
        self.lib.free_gpu_memory.restype = None

        self.lib.write_host_to_gpu_blocking.argtypes = [
            ct.c_int, ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_void_p
        ]
        self.lib.write_host_to_gpu_blocking.restype = ct.c_int
        self.lib.read_gpu_to_host_blocking.argtypes = [
            ct.c_int, ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_void_p
        ]
        self.lib.read_gpu_to_host_blocking.restype = ct.c_int

    def _bind_matmul_variants(self) -> None:
        # Versuch 1: single GEMM (M,N,K)
        try:
            fn = getattr(self.lib, "execute_matmul_on_gpu")
            fn.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                           ct.c_int, ct.c_int, ct.c_int]
            fn.restype = ct.c_int
            self._mm_single = fn
        except AttributeError:
            self._mm_single = None

        # Versuch 2: batched GEMM – wir nutzen BATCH=1
        try:
            fnb = getattr(self.lib, "execute_matmul_batched_on_gpu")
            fnb.argtypes = [ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                            ct.c_int, ct.c_int, ct.c_int, ct.c_int]
            fnb.restype = ct.c_int
            self._mm_batched = fnb
        except AttributeError:
            self._mm_batched = None

        if self._mm_single is None and self._mm_batched is None:
            raise DLLBindingError(
                "Weder execute_matmul_on_gpu noch execute_matmul_batched_on_gpu vorhanden – "
                "bitte Signaturen/Exportnamen prüfen."
            )

    # ---------- Utils ----------
    def _gpu_alloc(self, nbytes: int) -> ct.c_void_p:
        buf = self.lib.allocate_gpu_memory(self.gpu, ct.c_size_t(nbytes))
        if not buf:
            raise BackendError("allocate_gpu_memory fehlgeschlagen")
        return buf

    def _h2d(self, buf: ct.c_void_p, arr: np.ndarray) -> None:
        ok = self.lib.write_host_to_gpu_blocking(
            self.gpu, buf, 0, arr.nbytes, arr.ctypes.data_as(ct.c_void_p)
        )
        if ok != 1:
            raise BackendError("write_host_to_gpu_blocking fehlgeschlagen")

    def _d2h(self, buf: ct.c_void_p, out: np.ndarray) -> None:
        ok = self.lib.read_gpu_to_host_blocking(
            self.gpu, buf, 0, out.nbytes, out.ctypes.data_as(ct.c_void_p)
        )
        if ok != 1:
            raise BackendError("read_gpu_to_host_blocking fehlgeschlagen")

    def _gpu_free(self, *bufs: ct.c_void_p) -> None:
        for b in bufs:
            if b:
                self.lib.free_gpu_memory(self.gpu, b)

    # ---------- öffentlich ----------
    def close(self) -> None:
        try:
            self.lib.finish_gpu(self.gpu)
        finally:
            self.lib.shutdown_gpu(self.gpu)

    # y = U @ x (komplex), via 4 reelle GPU-Multiplikationen (float32)
    def matvec(self, U: np.ndarray, x: np.ndarray) -> np.ndarray:
        if U.ndim != 2 or U.shape[0] != U.shape[1]:
            raise DimensionError("U muss quadratisch sein.")
        n = U.shape[0]
        if x.shape != (n,):
            raise DimensionError("x muss (n,) sein.")

        # Vorbereitung: float32, row-major
        Ur = np.ascontiguousarray(U.real, dtype=np.float32)
        Ui = np.ascontiguousarray(U.imag, dtype=np.float32)
        xr = np.ascontiguousarray(x.real, dtype=np.float32).reshape(n, 1)
        xi = np.ascontiguousarray(x.imag, dtype=np.float32).reshape(n, 1)

        # GPU-Buffer
        A_r = self._gpu_alloc(Ur.nbytes)
        A_i = self._gpu_alloc(Ui.nbytes)
        X_r = self._gpu_alloc(xr.nbytes)
        X_i = self._gpu_alloc(xi.nbytes)
        Y_rr = self._gpu_alloc(xr.nbytes)
        Y_ii = self._gpu_alloc(xr.nbytes)
        Y_ri = self._gpu_alloc(xr.nbytes)
        Y_ir = self._gpu_alloc(xr.nbytes)

        # Host-Ausgaben
        y_rr = np.empty_like(xr)
        y_ii = np.empty_like(xr)
        y_ri = np.empty_like(xr)
        y_ir = np.empty_like(xr)

        try:
            # H2D
            self._h2d(A_r, Ur); self._h2d(A_i, Ui)
            self._h2d(X_r, xr); self._h2d(X_i, xi)

            # Aufruf

            def gemm(A, B, C, M, N, K) -> None:
                if self._mm_single is not None:
                    # Der C-Funktion die Batch-Größe B=1 übergeben
                    rc = self._mm_single(self.gpu, A, B, C, 1, int(M), int(N), int(K))
                else:
                    # batched: (BATCH=1, M, N, K)
                    rc = self._mm_batched(self.gpu, A, B, C, 1, int(M), int(N), int(K))
                if rc != 1:
                    raise BackendError(f"GPU GEMM rc={rc}")
                
            # M=n, N=1, K=n für A(n,n) @ B(n,1)
            # y_rr = U_r @ x_r
            gemm(A_r, X_r, Y_rr, n, 1, n)
            # y_ii = U_i @ x_i
            gemm(A_i, X_i, Y_ii, n, 1, n)
            # y_ri = U_r @ x_i
            gemm(A_r, X_i, Y_ri, n, 1, n)
            # y_ir = U_i @ x_r
            gemm(A_i, X_r, Y_ir, n, 1, n)

            # D2H
            self._d2h(Y_rr, y_rr); self._d2h(Y_ii, y_ii)
            self._d2h(Y_ri, y_ri); self._d2h(Y_ir, y_ir)

        finally:
            self._gpu_free(A_r, A_i, X_r, X_i, Y_rr, Y_ii, Y_ri, Y_ir)

        y_real = (y_rr - y_ii).reshape(n)
        y_imag = (y_ri + y_ir).reshape(n)
        return y_real.astype(np.float64) + 1j * y_imag.astype(np.float64)


# ------------------------------------------
# Deterministischer RNG aus Textseed
# ------------------------------------------
def seed_to_rng(seed: str) -> np.random.Generator:
    h = hashlib.sha256(seed.encode("utf-8")).digest()
    s64 = int.from_bytes(h[:8], "little", signed=False)
    return np.random.default_rng(s64)


def make_seed_state(n: int, seed: str) -> np.ndarray:
    if n <= 0:
        raise DimensionError("n muss > 0 sein.")
    rng = seed_to_rng(seed + "::psi0")
    psi = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    return psi


def make_seed_unitary(n: int, seed: str) -> np.ndarray:
    if n <= 0:
        raise DimensionError("n muss > 0 sein.")
    rng = seed_to_rng(seed + "::U")
    M = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))).astype(np.complex128)
    Q, R = np.linalg.qr(M)
    diag = np.diag(R)
    phase = diag / np.abs(diag)
    Q = Q * phase
    return Q


def make_seed_operators(n: int, seed: str) -> tuple[np.ndarray, np.ndarray]:
    rng = seed_to_rng(seed + "::ops")
    i = int(rng.integers(0, n))
    j = int(rng.integers(0, n))
    W = np.eye(n, dtype=np.complex128)
    W[i, i] = -1.0 + 0.0j
    v = np.zeros((n, 1), dtype=np.complex128); v[j, 0] = 1.0
    V = np.eye(n, dtype=np.complex128) - 2.0 * (v @ v.conj().T)
    return W, V


# ------------------------------------------
# OTOC-Serie (Zustandsbild, GPU-Matvec)
# ------------------------------------------
def otoc_series(
    U: np.ndarray,
    psi0: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    steps: int,
    backend: Backend,
) -> np.ndarray:
    n = len(psi0)
    Udag = U.conj().T

    phi = psi0.copy()
    vpsi = V @ psi0

    out = np.zeros(steps, dtype=np.float64)
    for t in range(steps):
        tmp = W @ phi
        back = tmp.copy()
        for _ in range(t):
            back = backend.matvec(Udag, back)
        K = np.vdot(vpsi, back)
        out[t] = 1.0 - float(np.real(K))
        phi = backend.matvec(U, phi)
        phi /= np.linalg.norm(phi)
    return out


# ------------------------------------------
# Visualisierung (Blätterbild)
# ------------------------------------------
def visualize_state_as_leaves(psi: np.ndarray, step: int) -> None:
    amp = np.abs(psi) ** 2
    n = amp.shape[0]
    side = int(math.ceil(math.sqrt(n)))
    canvas = np.zeros((side * side,), dtype=np.float64)
    canvas[:n] = amp
    canvas = canvas.reshape(side, side)
    plt.clf()
    plt.title(f"Deterministisches Blätterbild – Schritt {step}")
    plt.imshow(canvas, interpolation="nearest")
    plt.colorbar(label="Wahrscheinlichkeitsdichte")
    plt.pause(0.001)


@dataclass(slots=True)
class StepStats:
    steps: int
    total_s: float
    mean_ms: float
# ------------------------------------------
# Evolutions-Loop
# ------------------------------------------
def evolve(cfg: Config, backend: Backend) -> None:
    U = make_seed_unitary(cfg.n, cfg.seed)
    psi = make_seed_state(cfg.n, cfg.seed)

    t0 = perf_counter()
    if cfg.plot_every <= cfg.steps:
        plt.figure()

    for t in range(cfg.steps + 1):
        if cfg.plot_every <= cfg.steps and t % cfg.plot_every == 0:
            visualize_state_as_leaves(psi, t)
        psi = backend.matvec(U, psi)
        psi /= np.linalg.norm(psi)

    total = perf_counter() - t0
    mean_ms = (total / (cfg.steps + 1)) * 1e3
    print(f"[Profil] steps={cfg.steps+1} total={total:.6f}s mean_step={mean_ms:.3f}ms")
    if cfg.plot_every <= cfg.steps:
        plt.show()



# ------------------------------------------
# OTOC-Run
# ------------------------------------------
def run_otoc(cfg: Config, backend: Backend) -> None:
    U = make_seed_unitary(cfg.n, cfg.seed)
    psi0 = make_seed_state(cfg.n, cfg.seed)
    W, V = make_seed_operators(cfg.n, cfg.seed)
    series = otoc_series(U, psi0, W, V, cfg.steps, backend)
    plt.figure()
    plt.title("OTOC – deterministischer Chaos-Indikator (kleiner = stärker)")
    plt.xlabel("t")
    plt.ylabel("OTOC(t)")
    plt.plot(np.arange(cfg.steps), series)
    plt.grid(True, alpha=0.3)
    plt.show()


# ------------------------------------------
# CLI
# ------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GPU-first deterministische Mini-Welt (komplexes MatVec via DLL).")
    sub = p.add_subparsers(dest="cmd")

    p.add_argument("--n", type=int, default=128)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=str, default="Ursprung")
    p.add_argument("--backend", type=str, choices=["dll"], default="dll")
    p.add_argument("--dll", type=str, required=True, help="Pfad zur libCC_OpenCl.so")
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--plot-every", type=int, default=10)

    sp = sub.add_parser("otoc", help="OTOC-Zeitreihe plotten")
    sp.add_argument("--n", type=int, default=128)
    sp.add_argument("--steps", type=int, default=200)
    sp.add_argument("--seed", type=str, default="Ursprung")
    sp.add_argument("--backend", type=str, choices=["dll"], default="dll")
    sp.add_argument("--dll", type=str, required=True)
    sp.add_argument("--gpu-index", type=int, default=0)
    return p


def make_backend(cfg: Config) -> Backend:
    # Wir unterstützen hier bewusst *nur* DLL/GPU.
    return DLLBackend(cfg.dll_path, cfg.gpu_index)


def main() -> None:
    args = build_argparser().parse_args()

    if args.cmd == "otoc":
        cfg = Config(
            n=args.n, steps=args.steps, seed=args.seed,
            backend="dll", dll_path=Path(args.dll),
            gpu_index=args.gpu_index, otoc=True
        )
    else:
        cfg = Config(
            n=args.n, steps=args.steps, seed=args.seed,
            backend="dll", dll_path=Path(args.dll),
            gpu_index=args.gpu_index, plot_every=args.plot_every, otoc=False
        )

    backend = make_backend(cfg)
    try:
        if cfg.otoc:
            run_otoc(cfg, backend)
        else:
            evolve(cfg, backend)
    finally:
        if isinstance(backend, DLLBackend):
            backend.close()


if __name__ == "__main__":
    main()
