#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tri-Core ULTRA (v4.4.3) – Materials Orchestrator

Fix:
- Verhindert "subqg_simulation_step_batched: State not initialized." durch
  lazy Init + Ready-Flag. SubQG wird nur bei erfolgreichem Init genutzt,
  sonst *leise* Fallback (keine C-Logs).

Features:
- GPU-Index per CLI (--gpu)
- Korrektes ctypes-Binding inkl. PauliZTerm (uint64_t, float)
- Generator-Kopplung (SubQG-gestützt; mit Fallback)
- Zusammengesetztes Scoring (Toleranz + Energie-Kontrast + Surrogat)
- CSV-Export der Top-K Kandidaten (--export-csv, --topk)

Python 3.12
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

import argparse
import ctypes
import csv
import json
import math
import random
import re
import statistics


# ----------------------------- Hilfsfunktionen -----------------------------

def _as_float(x) -> float:
    if hasattr(x, "value"):
        try:
            return float(x.value)
        except Exception:
            pass
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = re.search(r"[-+]?\d*\.\d+|\d+", x)
        if m:
            return float(m.group(0))
        return float(x)
    raise TypeError(f"Kann Wert nicht in float casten: {x!r}")


# ----------------------------- Datenmodelle -----------------------------

@dataclass
class Material:
    composition: str
    x: list[float]


@dataclass
class OrchestratorConfig:
    max_epochs: int = 50
    population: int = 32
    strategy: Literal["explore", "exploit", "mix"] = "mix"
    lr0: float = 0.10
    seed: int = 42

    field_p1: float = 6.0
    field_p2: float = 0.40

    vqe_steps: int = 8
    vqe_mode: Literal["auto", "cpu", "gpu"] = "auto"
    num_qubits: int = 4
    ansatz_layers: int = 2
    num_h_terms: int = 2

    dll_path: str | None = None
    gpu_index: int = 0
    log_verbosity: Literal["quiet", "info", "debug"] = "info"

    w_tol: float = 0.5
    w_con: float = 0.3
    w_surr: float = 0.2

    tol_sigma_lo: float = 0.05
    tol_sigma_hi: float = 0.20

    export_csv: str | None = None
    topk: int = 10


@dataclass
class RunState:
    epoch: int = 0
    best_score: float = float("-inf")
    best_material: Material | None = None
    history: list[dict] = field(default_factory=list)


# ----------------------------- ctypes Strukturen -----------------------------

class PauliZTerm(ctypes.Structure):
    _fields_ = [
        ("z_mask", ctypes.c_uint64),
        ("coefficient", ctypes.c_float),
    ]


# ----------------------------- Treiber/Bindings -----------------------------

class Driver:
    def _stub(self, name: str) -> Callable:
        def _fallback(*args, **kwargs):
            if name == "initialize_gpu":
                print("[Stub] initialize_gpu(...) -> OK (CPU)")
                return 0
            if name == "subqg_simulation_step":
                _, rng_e, rng_p, rng_s, out_E, out_P, out_I, out_node, out_spin, out_topo, out_map, map_len = args
                E = _as_float(rng_e) - 0.5
                P = math.sin(_as_float(rng_p) * math.pi * 0.5)
                I = (E + P) * 0.5
                out_E[0] = E; out_P[0] = P; out_I[0] = I
                out_node[0] = 1 if I > 0.3 else 0
                out_spin[0] = 1 if _as_float(rng_s) > 0.5 else -1
                out_topo[0] = 0
                if int(map_len) > 0: out_map[0] = I
                return 1
            if name == "proto_step":
                arr_ptr, n, lr = args
                n = int(n); lr = _as_float(lr)
                xs = [arr_ptr[i] for i in range(n)]
                mean = sum(xs) / max(n, 1)
                for i in range(n):
                    xs[i] += lr * (mean - xs[i])
                    arr_ptr[i] = xs[i]
                return 0
            if name == "vqe_spsa":
                theta_ptr, n, steps, out_energy_best = args
                n = int(n); steps = int(steps)
                energy = 0.0
                for _ in range(steps):
                    energy = sum((theta_ptr[i] ** 2) for i in range(n)) / max(n, 1)
                    for i in range(n):
                        theta_ptr[i] *= 0.9
                out_energy_best[0] = float(energy)
                return 0
            print(f"[Stub] {name}(...) ausgeführt.")
            return 0
        return _fallback

    def __init__(self, dll_path: str | None, gpu_index: int = 0, verbosity: str = "info"):
        self.lib: ctypes.CDLL | None = None
        self.loaded: bool = False
        self.name: str = "CPU-Stub"
        self.gpu_index: int = int(gpu_index)
        self.verbosity = verbosity

        self._vqe_gpu = None
        self._proto_gpu = None
        self.subqg_simulation_step = None
        self._set_noise_level = None
        self._set_noise_accepts_gpu = False

        # SubQG-Init-Optionen
        self._subqg_init = None          # Funktionszeiger auf subqg_initialize_state
        self._subqg_inited = False       # interner Flag
        self._subqg_set_determ = None    # subqg_set_deterministic_mode

        self.proto_step = self._stub("proto_step")
        self.vqe_spsa = self._stub("vqe_spsa")
        self.initialize_gpu = self._stub("initialize_gpu")

        if dll_path:
            p = Path(dll_path)
            if p.exists():
                try:
                    self.lib = ctypes.CDLL(str(p))
                    self._bind_symbols()
                    self.loaded = True
                    self.name = p.name
                    print(f"[Driver] ✅ DLL geladen: {self.name}")
                except OSError as e:
                    print(f"[Driver] ⚠️ DLL konnte nicht geladen werden: {e}")
            else:
                print(f"[Driver] ⚠️ DLL-Pfad nicht gefunden: {p}")

    def _get_symbol(self, names: Sequence[str]):
        if self.lib is None:
            return None
        for nm in names:
            try:
                return getattr(self.lib, nm)
            except AttributeError:
                continue
        return None

    def _bind_symbols(self) -> None:
        fn = self._get_symbol(["initialize_gpu"])
        if fn is not None:
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_int]
            self.initialize_gpu = fn

        fn = self._get_symbol(["proto_update_step", "proto_step_gpu"])
        if fn is not None:
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_int,
                           ctypes.POINTER(ctypes.c_float),
                           ctypes.c_int,
                           ctypes.c_float]
            self._proto_gpu = fn

        # SubQG init/probe
        fn = self._get_symbol(["subqg_initialize_state"])
        if fn is not None:
            fn.restype  = ctypes.c_int
            fn.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
            self._subqg_init = fn

        fn = self._get_symbol(["subqg_set_deterministic_mode"])
        if fn is not None:
            fn.restype  = None
            fn.argtypes = [ctypes.c_int, ctypes.c_uint64]
            self._subqg_set_determ = fn

        fn = self._get_symbol(["subqg_simulation_step"])
        if fn is not None:
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_int,
                           ctypes.c_float, ctypes.c_float, ctypes.c_float,
                           ctypes.POINTER(ctypes.c_float),
                           ctypes.POINTER(ctypes.c_float),
                           ctypes.POINTER(ctypes.c_float),
                           ctypes.POINTER(ctypes.c_int),
                           ctypes.POINTER(ctypes.c_int),
                           ctypes.POINTER(ctypes.c_int),
                           ctypes.POINTER(ctypes.c_float),
                           ctypes.c_int]
            self.subqg_simulation_step = fn
        else:
            self.subqg_simulation_step = self._stub("subqg_simulation_step")

        fn = self._get_symbol(["execute_vqe_gpu"])
        if fn is not None:
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                           ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                           ctypes.POINTER(PauliZTerm), ctypes.c_int,
                           ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
            self._vqe_gpu = fn

        fn = self._get_symbol(["set_noise_level"])
        if fn is not None:
            try:
                fn.argtypes = [ctypes.c_int, ctypes.c_float]
                fn.restype = None
                self._set_noise_accepts_gpu = True
            except Exception:
                fn.argtypes = [ctypes.c_float]
                fn.restype = None
                self._set_noise_accepts_gpu = False
            self._set_noise_level = fn

    # ---------- Öffentliche API ----------

    def initialize(self) -> None:
        rc = self.initialize_gpu(self.gpu_index)
        if rc not in (0, 1):
            raise RuntimeError("initialize_gpu(...) meldet Fehler.")
        # SubQG-Zustand initialisieren, sofern API vorhanden (empfohlen)
        if self._subqg_init is not None and not self._subqg_inited:
            ok = self._subqg_init(
                ctypes.c_int(self.gpu_index),
                ctypes.c_float(0.0),  # initial_energy
                ctypes.c_float(0.0),  # initial_phase
                ctypes.c_float(0.0),  # noise_level
                ctypes.c_float(0.0),  # threshold
            )
            if ok not in (0, 1):
                raise RuntimeError("subqg_initialize_state(...) meldet Fehler.")
            self._subqg_inited = True
        # Optional: deterministischen Modus setzen
        if self._subqg_set_determ is not None:
            self._subqg_set_determ(ctypes.c_int(0), ctypes.c_uint64(0))

    def _ensure_subqg(self) -> None:
        if self._subqg_init is not None and not self._subqg_inited:
            ok = self._subqg_init(
                ctypes.c_int(self.gpu_index),
                ctypes.c_float(0.0),
                ctypes.c_float(0.0),
                ctypes.c_float(0.0),
                ctypes.c_float(0.0),
            )
            if ok in (0, 1):
                self._subqg_inited = True
            else:
                raise RuntimeError("subqg_initialize_state(...) meldet Fehler (Lazy-Init).")

    def initialize_gpu(self, *args, **kwargs):  # Dummy – wird durch _bind_symbols überschrieben
        return self._stub("initialize_gpu")(*args, **kwargs)

    def proto_cpu_update(self, x: list[float], lr: float) -> list[float]:
        arr = (ctypes.c_float * len(x))(*x)
        is_ctypes_fn = hasattr(self.proto_step, "__call__") and hasattr(self.proto_step, "argtypes")
        if is_ctypes_fn:
            self.proto_step(arr, len(x), ctypes.c_float(float(lr)))
        else:
            self.proto_step(arr, len(x), float(lr))
        return [float(arr[i]) for i in range(len(x))]

    def proto_gpu_update(self, x: list[float], lr: float) -> list[float]:
        if self._proto_gpu is not None:
            n = len(x)
            arr = (ctypes.c_float * n)(*x)
            rc = self._proto_gpu(ctypes.c_int(self.gpu_index), arr, ctypes.c_int(n), ctypes.c_float(float(lr)))
            if rc in (0, 1):
                return [float(arr[i]) for i in range(n)]
            print(f"[Python] ⚠️ proto_update_step() returned {rc}, CPU-Fallback.")
        return self.proto_cpu_update(x, lr)

    def vqe_cpu(self, theta: list[float], steps: int) -> tuple[list[float], float]:
        arr = (ctypes.c_float * len(theta))(*theta)
        outE = (ctypes.c_float * 1)(0.0)
        self.vqe_spsa(arr, len(theta), int(steps), outE)
        return [float(arr[i]) for i in range(len(theta))], float(outE[0])

    def vqe_gpu(self, theta: list[float], steps: int, *, num_qubits: int, ansatz_layers: int,
                h_terms: list[PauliZTerm]) -> tuple[list[float], float]:
        if not self._vqe_gpu:
            return self.vqe_cpu(theta, steps)
        num_params = len(theta)
        num_terms = len(h_terms)
        params = (ctypes.c_float * num_params)(*theta)
        outE = (ctypes.c_float * 1)(0.0)
        grads = (ctypes.c_float * num_params)()
        h_arr = (PauliZTerm * num_terms)(*h_terms)
        rc = self._vqe_gpu(self.gpu_index, num_qubits, ansatz_layers,
                           params, num_params, h_arr, num_terms, outE, grads)
        if rc != 1:
            print(f"[Python] ⚠️ execute_vqe_gpu() returned {rc}")
            return theta, 0.0
        return [float(params[i]) for i in range(num_params)], float(outE[0])

    def _subqg_read(self, phi: float) -> tuple[float, float, float]:
        self._ensure_subqg()
        try:
            out_E  = (ctypes.c_float * 1)(0.0)
            out_P  = (ctypes.c_float * 1)(0.0)
            out_I  = (ctypes.c_float * 1)(0.0)
            out_n  = (ctypes.c_int   * 1)(0)
            out_sp = (ctypes.c_int   * 1)(0)
            out_to = (ctypes.c_int   * 1)(-1)
            out_map = (ctypes.c_float * 1)(0.0)
            rc = self.subqg_simulation_step(
                ctypes.c_int(self.gpu_index),
                ctypes.c_float(phi),
                ctypes.c_float(phi),
                ctypes.c_float(0.5),
                out_E, out_P, out_I, out_n, out_sp, out_to, out_map, ctypes.c_int(0)
            )
            if rc in (0, 1):
                return float(out_E[0]), float(out_P[0]), float(out_I[0])
        except Exception:
            pass
        # Fallback (identisch zur Stub-Logik)
        E = phi - 0.5
        P = math.sin(phi * math.pi * 0.5)
        I = 0.5*(E + P)
        return E, P, I

    def field_lr_mod(self, mean_signal: float, p1: float, p2: float) -> float:
        phi = max(0.0, min(1.0, float(mean_signal)))
        E, P, I = self._subqg_read(phi)
        z = 0.5 * (I + 1.0)
        return 0.5 + 1.0 / (1.0 + math.exp(-float(p1) * (z - float(p2))))

    def set_noise(self, value: float) -> None:
        if not self._set_noise_level:
            return
        try:
            self._set_noise_level(ctypes.c_int(self.gpu_index), ctypes.c_float(float(value)))
        except Exception:
            pass


# ----------------------------- Utility: Ziele, Toleranz, Surrogat -----------------------------

def _target_profile(i: int) -> float:
    return (i / (i + 1)) * 0.75

def _sigma_profile(i: int, sigma_lo: float, sigma_hi: float) -> float:
    w = 1.0 - math.exp(-0.2 * i)
    return sigma_lo * (1.0 - w) + sigma_hi * w


class OnlineLinearSurrogate:
    def __init__(self, l2: float = 1e-3):
        self.count = 0
        self.mean_y = 0.0
        self.mean_norm2 = 0.0
        self.mean_avg = 0.0
        self.l2 = float(l2)

    def update(self, x: list[float], y: float) -> None:
        n2 = sum(v * v for v in x)
        av = sum(x) / max(1, len(x))
        self.count += 1
        alpha = 1.0 / min(self.count, 50)
        self.mean_y = (1 - alpha) * self.mean_y + alpha * y
        self.mean_norm2 = (1 - alpha) * self.mean_norm2 + alpha * n2
        self.mean_avg = (1 - alpha) * self.mean_avg + alpha * av

    def predict(self, x: list[float]) -> float:
        if self.count < 5:
            return self.mean_y
        n2 = sum(v * v for v in x)
        av = sum(x) / max(1, len(x))
        num = (self.mean_y * (1.0 + self.l2)
               + 0.15 * (self.mean_norm2 - abs(self.mean_norm2 - n2))
               + 0.25 * (self.mean_avg + av))
        den = (1.0 + self.l2 + 0.15 + 0.25)
        return num / den


# ----------------------------- Orchestrator -----------------------------

class MaterialsOrchestrator:
    def __init__(self, cfg: OrchestratorConfig):
        self.cfg = cfg
        self.state = RunState()
        self.driver = Driver(cfg.dll_path, gpu_index=cfg.gpu_index, verbosity=cfg.log_verbosity)
        random.seed(cfg.seed)

        self.driver.initialize()

        self.surrogate = OnlineLinearSurrogate(l2=1e-3)
        self.energy_history: list[float] = []
        self._epoch_top: list[dict] = []

    def _compute_generator_coupling(self, baseline_lr: float, proto_vec: list[float], vqe_energy: float) -> tuple[float, float]:
        proto_mean = sum(proto_vec) / max(1, len(proto_vec))
        best_s = (0.0 if self.state.best_score == float("-inf") else self.state.best_score)
        phi = max(0.0, min(1.0, 0.5 * proto_mean + 0.5 * max(0.0, min(1.0, best_s))))

        E, P, I = self.driver._subqg_read(phi)

        z = 0.5 * (I + 1.0)
        k_lr = 0.5 + 1.0 / (1.0 + math.exp(-self.cfg.field_p1 * (z - self.cfg.field_p2)))

        self.driver.set_noise(max(0.0, min(1.0, vqe_energy)) - 0.5)  # Noise setzt selbst auf 0..1

        sig = lambda u: 1.0 / (1.0 + math.exp(-float(u)))
        k_pert = 0.35 + 0.65 * sig(2.0 * abs(P) + 1.5 * abs(E))

        lr = float(baseline_lr * k_lr)
        pert = float(0.08 * k_pert)
        return lr, pert

    def _tolerance_factor(self, x: list[float]) -> float:
        s = 0.0
        for i, v in enumerate(x, start=1):
            T = _target_profile(i)
            sigma = _sigma_profile(i, self.cfg.tol_sigma_lo, self.cfg.tol_sigma_hi)
            s += math.exp(-((v - T) / max(1e-6, sigma)) ** 2)
        return s / max(1, len(x))

    def _energy_contrast(self, e: float) -> float:
        self.energy_history.append(e)
        baseline = statistics.median(self.energy_history[-15:]) if len(self.energy_history) >= 3 else e
        return 1.0 / (1.0 + math.exp(3.0 * (e - baseline)))

    def _make_dummy_hamiltonian(self, num_qubits: int, num_terms: int) -> list[PauliZTerm]:
        terms: list[PauliZTerm] = []
        for i in range(num_terms):
            q = i % max(1, num_qubits)
            mask = (1 << q) & ((1 << 64) - 1)
            terms.append(PauliZTerm(z_mask=ctypes.c_uint64(mask).value, coefficient=1.0))
        return terms

    def _vqe_refine(self, theta: list[float], steps: int) -> tuple[list[float], float]:
        expected = 2 * self.cfg.num_qubits * self.cfg.ansatz_layers
        if len(theta) < expected:
            pad = [random.uniform(-math.pi * 0.01, math.pi * 0.01) for _ in range(expected - len(theta))]
            theta = theta + pad

        if self.cfg.vqe_mode == "cpu" or not self.driver.loaded or self.driver._vqe_gpu is None:
            return self.driver.vqe_cpu(theta, steps)

        h_terms = self._make_dummy_hamiltonian(self.cfg.num_qubits, self.cfg.num_h_terms)
        return self.driver.vqe_gpu(theta, steps,
                                   num_qubits=self.cfg.num_qubits,
                                   ansatz_layers=self.cfg.ansatz_layers,
                                   h_terms=h_terms)

    def _perturb(self, x: list[float], scale: float) -> list[float]:
        return [min(1.0, max(0.0, v + random.uniform(-scale, scale))) for v in x]

    def _sample_candidates(self, base: Material, pert_base: float) -> list[Material]:
        match self.cfg.strategy:
            case "explore":
                scales = [1.50 * pert_base] * self.cfg.population
            case "exploit":
                scales = [0.50 * pert_base] * self.cfg.population
            case "mix":
                scales = [1.50 * pert_base if i % 4 == 0 else 0.65 * pert_base for i in range(self.cfg.population)]
            case _:
                scales = [pert_base] * self.cfg.population
        return [Material(composition=f"{base.composition}-v{i}", x=self._perturb(base.x, sc))
                for i, sc in enumerate(scales)]

    def _push_epoch_top(self, epoch: int, mat: Material, score: float, energy: float) -> None:
        self._epoch_top.append({
            "epoch": epoch,
            "composition": mat.composition,
            "score": float(score),
            "energy": float(energy),
            "x": mat.x[:],
        })

    def _flush_csv(self) -> None:
        if not self.cfg.export_csv:
            return
        path = Path(self.cfg.export_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        sorted_all = sorted(self._epoch_top, key=lambda d: d["score"], reverse=True)[:max(1, self.cfg.topk)]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["rank", "epoch", "composition", "score", "energy", "x_json"])
            for r, row in enumerate(sorted_all, start=1):
                w.writerow([r, row["epoch"], row["composition"], f"{row['score']:.6f}", f"{row['energy']:.6f}", json.dumps(row["x"])])

    def run(self, dim: int = 6) -> RunState:
        rng = random.Random(self.cfg.seed)
        pop = [Material(composition=f"X{k}", x=[rng.random() for _ in range(dim)]) for k in range(self.cfg.population)]

        scores0 = []
        for m in pop:
            s = self._tolerance_factor(m.x)
            self.surrogate.update(m.x, s)
            scores0.append(s)
        best_idx = max(range(len(pop)), key=lambda i: scores0[i])
        best = pop[best_idx]; best_score = scores0[best_idx]
        self.state.best_material = best; self.state.best_score = best_score

        _, e0 = self._vqe_refine(best.x[:], steps=max(1, self.cfg.vqe_steps // 2))
        self.energy_history.append(e0)

        for epoch in range(1, self.cfg.max_epochs + 1):
            self.state.epoch = epoch

            x_proto = self.driver.proto_gpu_update(best.x, lr=self.cfg.lr0)
            theta_refined, e_best = self._vqe_refine(theta=x_proto, steps=self.cfg.vqe_steps)

            lr_coupled, pert_scale = self._compute_generator_coupling(self.cfg.lr0, theta_refined, e_best)
            x_proto2 = self.driver.proto_gpu_update(theta_refined, lr=lr_coupled)

            base = Material(best.composition, x_proto2)
            candidates = self._sample_candidates(base, pert_base=pert_scale)

            cand_scores: list[float] = []
            for c in candidates:
                s_tol = self._tolerance_factor(c.x)
                s_con = self._energy_contrast(e_best)
                s_surr_pred = self.surrogate.predict(c.x)
                score_tmp = self.cfg.w_tol * s_tol + self.cfg.w_con * s_con + self.cfg.w_surr * (1.0 / (1.0 + math.exp(-(s_tol - s_surr_pred))))
                self.surrogate.update(c.x, score_tmp)
                cand_scores.append(score_tmp)

            epoch_best_idx = max(range(len(candidates)), key=lambda i: cand_scores[i])
            epoch_best = candidates[epoch_best_idx]
            epoch_best_score = cand_scores[epoch_best_idx]

            if epoch_best_score > best_score:
                best, best_score = epoch_best, epoch_best_score

            self.state.best_material = best
            self.state.best_score = best_score

            self._push_epoch_top(epoch, epoch_best, epoch_best_score, e_best)
            self.state.history.append({
                "epoch": epoch,
                "lr0": self.cfg.lr0,
                "lr_coupled": lr_coupled,
                "pert_scale": pert_scale,
                "e_best": e_best,
                "best_score": best_score,
                "best_x": best.x[:],
            })
            print(f"[Epoch {epoch:03d}] score={best_score:.4f}  lr*={lr_coupled:.4f}  pert={pert_scale:.4f}  e_best={e_best:.4f}")

        self._flush_csv()
        return self.state


# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tri-Core ULTRA v4.4.3 – Materials Orchestrator")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--strategy", choices=["explore", "exploit", "mix"], default="mix")

    ap.add_argument("--dll", type=str, default=None)
    ap.add_argument("--gpu", type=int, default=0)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", choices=["quiet", "info", "debug"], default="info")

    ap.add_argument("--lr0", type=float, default=0.10)
    ap.add_argument("--field-p1", type=float, default=6.0)
    ap.add_argument("--field-p2", type=float, default=0.40)

    ap.add_argument("--vqe-steps", type=int, default=8)
    ap.add_argument("--vqe", choices=["auto", "cpu", "gpu"], default="auto")
    ap.add_argument("--num-qubits", type=int, default=4)
    ap.add_argument("--ansatz-layers", type=int, default=2)
    ap.add_argument("--num-h-terms", type=int, default=2)

    ap.add_argument("--w-tol", type=float, default=0.5)
    ap.add_argument("--w-con", type=float, default=0.3)
    ap.add_argument("--w-surr", type=float, default=0.2)

    ap.add_argument("--tol-sigma-lo", type=float, default=0.05)
    ap.add_argument("--tol-sigma-hi", type=float, default=0.20)

    ap.add_argument("--export-csv", type=str, default=None)
    ap.add_argument("--topk", type=int, default=10)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OrchestratorConfig(
        max_epochs=args.epochs,
        population=args.pop,
        strategy=args.strategy,
        dll_path=args.dll,
        seed=args.seed,
        lr0=args.lr0,
        vqe_steps=args.vqe_steps,
        vqe_mode=args.vqe,
        field_p1=args.field_p1,
        field_p2=args.field_p2,
        log_verbosity=args.log,
        num_qubits=args.num_qubits,
        ansatz_layers=args.ansatz_layers,
        num_h_terms=args.num_h_terms,
        gpu_index=args.gpu,
        w_tol=args.w_tol, w_con=args.w_con, w_surr=args.w_surr,
        tol_sigma_lo=args.tol_sigma_lo, tol_sigma_hi=args.tol_sigma_hi,
        export_csv=args.export_csv, topk=args.topk,
    )

    orch = MaterialsOrchestrator(cfg)
    state = orch.run(dim=args.dim)

    out = {
        "best_score": state.best_score,
        "best_material": {
            "composition": state.best_material.composition if state.best_material else None,
            "x": state.best_material.x if state.best_material else None,
        },
        "history": state.history,
        "driver": orch.driver.name,
        "config": {
            "epochs": cfg.max_epochs, "pop": cfg.population, "dim": args.dim,
            "strategy": cfg.strategy, "lr0": cfg.lr0, "vqe_steps": cfg.vqe_steps,
            "vqe_mode": cfg.vqe_mode, "field_p1": cfg.field_p1, "field_p2": cfg.field_p2,
            "seed": cfg.seed, "num_qubits": cfg.num_qubits, "ansatz_layers": cfg.ansatz_layers,
            "num_h_terms": cfg.num_h_terms, "gpu": cfg.gpu_index,
            "w_tol": cfg.w_tol, "w_con": cfg.w_con, "w_surr": cfg.w_surr,
            "tol_sigma_lo": cfg.tol_sigma_lo, "tol_sigma_hi": cfg.tol_sigma_hi,
            "export_csv": cfg.export_csv, "topk": cfg.topk,
        }
    }
    Path("materials_run.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n✅ Fertig. Ergebnisse in 'materials_run.json' gespeichert.")
    if cfg.export_csv:
        print(f"📄 CSV exportiert nach: {cfg.export_csv}")
    # --- nach dem Speichern von materials_run.json ---
    try:
        from material_mapper import MaterialMapper, make_default_schema
        mapper = MaterialMapper(schema=make_default_schema("hybrid", affine_scale=0.2))
        best_x = out["best_material"]["x"] or []
        props = mapper.map_vector(best_x)
        mapped = {
            "composition": out["best_material"]["composition"],
            "x": best_x,
            "properties": props,
            "config": out["config"],
            "driver": out["driver"],
        }
        Path("materials_props.json").write_text(json.dumps(mapped, indent=2), encoding="utf-8")
        print("🧪 Material-Eigenschaften → materials_props.json")
    except Exception as e:
        print(f"[Warn] Konnte Properties nicht mappen: {e}")


if __name__ == "__main__":
    main()
