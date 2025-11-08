#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eon_ocr.py – Natives, GPU-residentes OCR-Framework mit OpenCL

- PyTorch wird nur noch zur Parameter-Verwaltung und für CPU-Operationen genutzt.
- Der gesamte Forward- und Backward-Pass für das Training läuft über den OpenCL-Treiber.
- Daten bleiben während der Berechnung auf der GPU, um Transfer-Overhead zu eliminieren.
- Manuelle Implementierung der Backpropagation und des AdamW-Updates.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Literal, Sequence, Tuple, Dict, Any, List, Union, Iterable
import os
import atexit
import math
import random
import platform
import ctypes
import time
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

# PyTorch wird nur noch minimal genutzt (z.B. für den initialen Loss)
import torch
import torch.nn.functional as F

# Optionale Imports
try:
    import kenlm
    _KENLM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _KENLM_AVAILABLE = False
try:
    from pdf2image import convert_from_path
    _PDF2IMG_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _PDF2IMG_AVAILABLE = False

# =============================================================================
# 0) OpenCL C-Treiber Interface (angepasst für natives Training)
# =============================================================================

_DEFAULT_DRIVER_NAMES: Tuple[str, ...] = (
    "CipherCore_OpenCl.so",
    "CipherCore_OpenCl.dll",
    "libCipherCore_OpenCl.so",
)


def _resolve_driver_path(explicit: Optional[Union[str, os.PathLike]]) -> str:
    if explicit:
        candidate = Path(explicit)
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"Angegebener OpenCL-Treiber nicht gefunden: {explicit}")

    env_path = os.getenv("TRI_CORE_OPENCL_DRIVER")
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            return str(candidate)

    base_dir = Path(__file__).resolve().parent
    for name in _DEFAULT_DRIVER_NAMES:
        candidate = base_dir / name
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "Kein kompatibler OpenCL-Treiber gefunden. Setzen Sie TRI_CORE_OPENCL_DRIVER oder"
        " platzieren Sie die kompilierten Bibliotheken neben eon.py."
    )


class _Driver:
    """Thin ctypes Wrapper um die C-API des CC_OpenCL-Treibers."""

    def __init__(self, lib_path: Optional[Union[str, os.PathLike]] = None, gpu_index: int = 0):
        self._gpu_index = gpu_index
        resolved = _resolve_driver_path(lib_path)
        self._lib = ctypes.CDLL(resolved)
        self._configure_signatures()
        self._closed = True
        if not self._lib.initialize_gpu(self._gpu_index):
            detail = self.last_error()
            raise RuntimeError(f"OpenCL-Initialisierung fehlgeschlagen: {detail}")
        self._closed = False

    # -- Lifecycle ------------------------------------------------------------
    def close(self) -> None:
        if not getattr(self, "_closed", True):
            try:
                self._lib.shutdown_gpu(self._gpu_index)
            finally:
                self._closed = True

    def __enter__(self) -> "_Driver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - cleanup best effort
        try:
            self.close()
        except Exception:
            pass

    # -- Signaturkonfiguration -------------------------------------------------
    def _configure_signatures(self) -> None:
        lib = self._lib
        c_int = ctypes.c_int
        c_size_t = ctypes.c_size_t
        c_float = ctypes.c_float
        c_void_p = ctypes.c_void_p
        c_char_p = ctypes.c_char_p

        lib.initialize_gpu.argtypes = [c_int]
        lib.initialize_gpu.restype = c_int
        lib.shutdown_gpu.argtypes = [c_int]
        lib.shutdown_gpu.restype = None

        lib.allocate_gpu_memory.argtypes = [c_int, c_size_t]
        lib.allocate_gpu_memory.restype = c_void_p
        lib.free_gpu_memory.argtypes = [c_int, c_void_p]
        lib.free_gpu_memory.restype = None

        lib.write_host_to_gpu_blocking.argtypes = [c_int, c_void_p, c_size_t, c_size_t, c_void_p]
        lib.write_host_to_gpu_blocking.restype = c_int
        lib.read_gpu_to_host_blocking.argtypes = [c_int, c_void_p, c_size_t, c_size_t, c_void_p]
        lib.read_gpu_to_host_blocking.restype = c_int

        lib.execute_conv2d_forward_on_gpu.argtypes = [
            c_int, c_void_p, c_void_p, c_void_p, c_void_p,
            c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
        ]
        lib.execute_conv2d_forward_on_gpu.restype = c_int

        lib.execute_conv2d_backward_on_gpu.argtypes = [
            c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
            c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
        ]
        lib.execute_conv2d_backward_on_gpu.restype = c_int

        lib.execute_patch_permute_reshape_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        lib.execute_patch_permute_reshape_on_gpu.restype = c_int
        lib.execute_patch_permute_reshape_backward_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        lib.execute_patch_permute_reshape_backward_on_gpu.restype = c_int

        lib.execute_eon_encoder_chain_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_size_t]
        lib.execute_eon_encoder_chain_on_gpu.restype = c_int
        lib.execute_eon_encoder_backward_chain_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_size_t]
        lib.execute_eon_encoder_backward_chain_on_gpu.restype = c_int

        lib.execute_adam_update_on_gpu.argtypes = [
            c_int, c_void_p, c_void_p, c_void_p, c_void_p,
            c_int, c_int, c_float, c_float, c_float, c_float, c_float,
        ]
        lib.execute_adam_update_on_gpu.restype = c_int

        lib.execute_matmul_on_gpu.argtypes = [c_int, c_void_p, c_void_p, c_void_p, c_int, c_int, c_int, c_int]
        lib.execute_matmul_on_gpu.restype = c_int

        lib.compute_ctc_loss_cpu.argtypes = [
            ctypes.POINTER(c_float), c_int, c_int, c_int,
            ctypes.POINTER(ctypes.c_int), c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            c_int,
            ctypes.POINTER(c_float), ctypes.POINTER(c_float),
        ]
        lib.compute_ctc_loss_cpu.restype = c_int

        lib.cc_get_last_error.argtypes = []
        lib.cc_get_last_error.restype = c_char_p
        lib.cc_get_version.argtypes = []
        lib.cc_get_version.restype = c_char_p

    def _decode(self, raw: Optional[bytes]) -> str:
        if not raw:
            return ""
        return raw.decode("utf-8", errors="replace")

    def last_error(self) -> str:
        return self._decode(self._lib.cc_get_last_error())

    def version(self) -> str:
        return self._decode(self._lib.cc_get_version()) or "unknown"

    def _ensure_ok(self, ok: int, context: str) -> None:
        if not ok:
            raise RuntimeError(f"{context}: {self.last_error() or 'unbekannter Fehler'}")

    # -- Speicherverwaltung ----------------------------------------------------
    def allocate_gpu_memory(self, size: int) -> Optional[int]:
        raw = self._lib.allocate_gpu_memory(self._gpu_index, size)
        ptr = raw if isinstance(raw, int) else (raw.value if raw else None)
        if not ptr and size:
            detail = self.last_error()
            raise MemoryError(f"OpenCL-Treiber konnte keinen Speicher allokieren: {detail}")
        return ptr

    def free_gpu_memory(self, ptr: Optional[int]) -> None:
        if ptr:
            self._lib.free_gpu_memory(self._gpu_index, ctypes.c_void_p(ptr))

    def write_host_to_gpu(self, gpu_ptr: int, host_ptr: int, nbytes: int, offset: int = 0) -> None:
        ok = self._lib.write_host_to_gpu_blocking(
            self._gpu_index,
            ctypes.c_void_p(gpu_ptr),
            ctypes.c_size_t(offset),
            ctypes.c_size_t(nbytes),
            ctypes.c_void_p(host_ptr),
        )
        self._ensure_ok(ok, "GPU-Schreibvorgang fehlgeschlagen")

    def read_gpu_to_host(self, gpu_ptr: int, host_ptr: int, nbytes: int, offset: int = 0) -> None:
        ok = self._lib.read_gpu_to_host_blocking(
            self._gpu_index,
            ctypes.c_void_p(gpu_ptr),
            ctypes.c_size_t(offset),
            ctypes.c_size_t(nbytes),
            ctypes.c_void_p(host_ptr),
        )
        self._ensure_ok(ok, "GPU-Lesevorgang fehlgeschlagen")

    # -- Kernel Aufrufe --------------------------------------------------------
    def offload_conv2d_forward(
        self,
        x_ptr: int,
        w_ptr: int,
        b_ptr: Optional[int],
        out_ptr: int,
        B: int,
        C_in: int,
        H: int,
        W: int,
        C_out: int,
        K_h: int,
        K_w: int,
        stride_h: int,
        stride_w: int,
    ) -> None:
        ok = self._lib.execute_conv2d_forward_on_gpu(
            self._gpu_index,
            ctypes.c_void_p(x_ptr),
            ctypes.c_void_p(w_ptr),
            ctypes.c_void_p(b_ptr) if b_ptr else ctypes.c_void_p(),
            ctypes.c_void_p(out_ptr),
            B, C_in, H, W, C_out, K_h, K_w, stride_h, stride_w,
        )
        self._ensure_ok(ok, "Conv2D Forward fehlgeschlagen")

    def offload_conv2d_backward(
        self,
        grad_out_ptr: int,
        x_ptr: int,
        w_ptr: int,
        grad_x_ptr: Optional[int],
        grad_w_ptr: Optional[int],
        grad_b_ptr: Optional[int],
        B: int,
        C_in: int,
        H: int,
        W: int,
        C_out: int,
        K_h: int,
        K_w: int,
        stride_h: int,
        stride_w: int,
    ) -> None:
        ok = self._lib.execute_conv2d_backward_on_gpu(
            self._gpu_index,
            ctypes.c_void_p(grad_out_ptr),
            ctypes.c_void_p(x_ptr),
            ctypes.c_void_p(w_ptr),
            ctypes.c_void_p(grad_x_ptr) if grad_x_ptr else ctypes.c_void_p(),
            ctypes.c_void_p(grad_w_ptr) if grad_w_ptr else ctypes.c_void_p(),
            ctypes.c_void_p(grad_b_ptr) if grad_b_ptr else ctypes.c_void_p(),
            B, C_in, H, W, C_out, K_h, K_w, stride_h, stride_w,
        )
        self._ensure_ok(ok, "Conv2D Backward fehlgeschlagen")

    def offload_permute_and_reshape(self, in_ptr: int, out_ptr: int, B: int, C: int, H: int, W: int) -> None:
        ok = self._lib.execute_patch_permute_reshape_on_gpu(
            self._gpu_index,
            ctypes.c_void_p(in_ptr),
            ctypes.c_void_p(out_ptr),
            B, C, H, W,
        )
        self._ensure_ok(ok, "Patch-Permute fehlgeschlagen")

    def offload_reshape_and_permute_backward(self, in_ptr: int, out_ptr: int, B: int, C: int, H: int, W: int) -> None:
        ok = self._lib.execute_patch_permute_reshape_backward_on_gpu(
            self._gpu_index,
            ctypes.c_void_p(in_ptr),
            ctypes.c_void_p(out_ptr),
            B, C, H, W,
        )
        self._ensure_ok(ok, "Patch-Permute-Backward fehlgeschlagen")

    def execute_eon_encoder_chain(self, in_ptr: int, out_ptr: int, num_bytes: int) -> None:
        ok = self._lib.execute_eon_encoder_chain_on_gpu(
            self._gpu_index,
            ctypes.c_void_p(in_ptr),
            ctypes.c_void_p(out_ptr),
            ctypes.c_size_t(num_bytes),
        )
        self._ensure_ok(ok, "Encoder-Kette konnte nicht ausgeführt werden")

    def execute_eon_encoder_backward_chain(self, grad_out_ptr: int, grad_in_ptr: int, num_bytes: int) -> None:
        ok = self._lib.execute_eon_encoder_backward_chain_on_gpu(
            self._gpu_index,
            ctypes.c_void_p(grad_out_ptr),
            ctypes.c_void_p(grad_in_ptr),
            ctypes.c_size_t(num_bytes),
        )
        self._ensure_ok(ok, "Encoder-Backward-Kette konnte nicht ausgeführt werden")

    def offload_adamw_update(
        self,
        param_ptr: int,
        grad_ptr: int,
        m_ptr: int,
        v_ptr: int,
        num_elements: int,
        step: int,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        ok = self._lib.execute_adam_update_on_gpu(
            self._gpu_index,
            ctypes.c_void_p(param_ptr),
            ctypes.c_void_p(grad_ptr),
            ctypes.c_void_p(m_ptr),
            ctypes.c_void_p(v_ptr),
            num_elements,
            step,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
        )
        self._ensure_ok(ok, "AdamW-Update fehlgeschlagen")

    def offload_ctc_loss(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        target_lengths: np.ndarray,
        input_lengths: np.ndarray,
        blank_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if logits.dtype != np.float32:
            logits = logits.astype(np.float32, copy=False)
        if targets.dtype != np.int32:
            targets = targets.astype(np.int32, copy=False)
        target_lengths = np.asarray(target_lengths, dtype=np.int32)
        input_lengths = np.asarray(input_lengths, dtype=np.int32)

        B, T, V = logits.shape
        max_target_len = targets.shape[1]

        loss = np.zeros((B,), dtype=np.float32)
        grad = np.zeros_like(logits, dtype=np.float32)

        ok = self._lib.compute_ctc_loss_cpu(
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(T),
            ctypes.c_int(B),
            ctypes.c_int(V),
            targets.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(max_target_len),
            target_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            input_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(blank_index),
            loss.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        self._ensure_ok(ok, "CTC-Loss Berechnung fehlgeschlagen")
        return loss, grad


class _NullDriver:
    """Fallback, falls kein GPU-Treiber verfügbar ist."""

    def allocate_gpu_memory(self, size: int) -> Optional[int]:  # pragma: no cover - fallback
        return None

    def free_gpu_memory(self, ptr: Optional[int]) -> None:  # pragma: no cover - fallback
        pass

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - fallback
        raise RuntimeError("OpenCL-Treiber nicht geladen")


DRIVER: Any = None
USE_OPENCL: bool = False


def _load_opencl_driver(lib_path: Optional[Union[str, os.PathLike]] = None) -> Any:
    global DRIVER, USE_OPENCL
    if DRIVER is None:
        try:
            DRIVER = _Driver(lib_path=lib_path)
            atexit.register(DRIVER.close)
            USE_OPENCL = True
        except Exception as exc:  # pragma: no cover - fallback
            print(f"[EON] Warnung: OpenCL-Treiber konnte nicht geladen werden ({exc}). Fallback aktiv.")
            DRIVER = _NullDriver()
            USE_OPENCL = False
    return DRIVER


# =============================================================================
# 1) GPU-Datenstruktur und natives Modell
# =============================================================================


class GPUTensor:
    """Wrapper für einen Pointer auf GPU-Speicher, der von unserem C-Treiber verwaltet wird."""

    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32, ptr: Optional[int] = None):
        driver = _load_opencl_driver()
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(shape)) * self.dtype.itemsize
        self.ptr: Optional[int]
        if ptr is None:
            if self.nbytes > 0 and USE_OPENCL:
                raw_ptr = driver.allocate_gpu_memory(self.nbytes)
                if not raw_ptr:
                    raise MemoryError("OpenCL-Treiber konnte keinen Speicher allokieren.")
                self.ptr = raw_ptr
            else:
                self.ptr = None
        else:
            self.ptr = ptr

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> GPUTensor:
        driver = _load_opencl_driver()
        gpu_tensor = cls(arr.shape, arr.dtype)
        if gpu_tensor.ptr and USE_OPENCL:
            driver.write_host_to_gpu(gpu_tensor.ptr, arr.ctypes.data, gpu_tensor.nbytes)
        return gpu_tensor

    def to_numpy(self) -> np.ndarray:
        driver = _load_opencl_driver()
        arr = np.empty(self.shape, dtype=self.dtype)
        if self.ptr and USE_OPENCL:
            driver.read_gpu_to_host(self.ptr, arr.ctypes.data, self.nbytes)
        return arr

    def __del__(self):
        if getattr(self, "ptr", None) and USE_OPENCL:
            try:
                DRIVER.free_gpu_memory(self.ptr)
            except Exception:
                pass


def he_init(shape):
    fan_in = np.prod(shape[1:])
    std = math.sqrt(2.0 / fan_in)
    return np.random.normal(0, std, size=shape).astype(np.float32)


@dataclass
class Parameter:
    data: np.ndarray
    grad: np.ndarray = field(init=False)
    m: np.ndarray = field(init=False)
    v: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        data = self.data.astype(np.float32, copy=False)
        self.data = data
        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self.m = np.zeros_like(self.data, dtype=np.float32)
        self.v = np.zeros_like(self.data, dtype=np.float32)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)


class Conv2D:
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        self.weight = Parameter(he_init((out_ch, in_ch, kernel_size[0], kernel_size[1])))
        self.bias = Parameter(np.zeros(out_ch, dtype=np.float32))
        self.stride = stride
        self.cache: Dict[str, Any] = {}

    def forward(self, x: GPUTensor) -> GPUTensor:
        driver = _load_opencl_driver()
        B, C, H, W = x.shape
        kh, kw = self.weight.data.shape[2:]
        sh, sw = self.stride
        out_h, out_w = (H - kh) // sh + 1, (W - kw) // sw + 1

        y = GPUTensor((B, self.weight.data.shape[0], out_h, out_w))

        if USE_OPENCL and x.ptr and y.ptr:
            w_gpu = GPUTensor.from_numpy(self.weight.data)
            b_gpu = GPUTensor.from_numpy(self.bias.data)
            driver.offload_conv2d_forward(
                x.ptr,
                w_gpu.ptr if w_gpu.ptr else 0,
                b_gpu.ptr if (b_gpu and b_gpu.ptr) else 0,
                y.ptr,
                B, C, H, W,
                self.weight.data.shape[0],
                kh, kw,
                sh, sw,
            )
            self.cache['w_gpu'] = w_gpu
            self.cache['b_gpu'] = b_gpu
        else:
            # CPU Fallback
            x_np = x.to_numpy()
            w = self.weight.data
            b = self.bias.data
            out = np.zeros((B, w.shape[0], out_h, out_w), dtype=np.float32)
            for b_idx in range(B):
                for oc in range(w.shape[0]):
                    for oh in range(out_h):
                        ih = oh * sh
                        for ow in range(out_w):
                            iw = ow * sw
                            region = x_np[b_idx, :, ih:ih + kh, iw:iw + kw]
                            out[b_idx, oc, oh, ow] = np.sum(region * w[oc]) + b[oc]
            y = GPUTensor.from_numpy(out)
        self.cache['x'] = x
        return y

    def backward(self, grad_y: GPUTensor) -> GPUTensor:
        driver = _load_opencl_driver()
        x = self.cache['x']
        grad_x = GPUTensor(x.shape)
        B, C, H, W = x.shape
        kh, kw = self.weight.data.shape[2:]
        sh, sw = self.stride

        if USE_OPENCL and grad_y.ptr:
            w_gpu = self.cache.get('w_gpu')
            b_gpu = self.cache.get('b_gpu')
            grad_w_gpu = GPUTensor(self.weight.data.shape)
            grad_b_gpu = GPUTensor(self.bias.data.shape)
            driver.offload_conv2d_backward(
                grad_y.ptr,
                x.ptr if x.ptr else 0,
                w_gpu.ptr if (w_gpu and w_gpu.ptr) else 0,
                grad_x.ptr if grad_x.ptr else 0,
                grad_w_gpu.ptr if grad_w_gpu.ptr else 0,
                grad_b_gpu.ptr if grad_b_gpu.ptr else 0,
                B, C, H, W,
                self.weight.data.shape[0],
                kh, kw,
                sh, sw,
            )
            self.weight.grad[...] += grad_w_gpu.to_numpy()
            self.bias.grad[...] += grad_b_gpu.to_numpy()
        else:
            grad_y_np = grad_y.to_numpy()
            x_np = x.to_numpy()
            grad_w = np.zeros_like(self.weight.data)
            grad_b = np.zeros_like(self.bias.data)
            grad_x_np = np.zeros_like(x_np)
            out_h = grad_y_np.shape[2]
            out_w = grad_y_np.shape[3]
            for b_idx in range(B):
                for oc in range(self.weight.data.shape[0]):
                    for ic in range(C):
                        for kh_idx in range(kh):
                            for kw_idx in range(kw):
                                acc = 0.0
                                for oh in range(out_h):
                                    ih = oh * sh + kh_idx
                                    for ow in range(out_w):
                                        iw = ow * sw + kw_idx
                                        acc += grad_y_np[b_idx, oc, oh, ow] * x_np[b_idx, ic, ih, iw]
                                grad_w[oc, ic, kh_idx, kw_idx] += acc
                    grad_b[oc] += grad_y_np[b_idx, oc].sum()
                # grad_x
                for ic in range(C):
                    for ih in range(H):
                        for iw in range(W):
                            acc = 0.0
                            for oc in range(self.weight.data.shape[0]):
                                for kh_idx in range(kh):
                                    oh = ih - kh_idx
                                    if oh % sh != 0:
                                        continue
                                    oh //= sh
                                    if oh < 0 or oh >= out_h:
                                        continue
                                    for kw_idx in range(kw):
                                        ow = iw - kw_idx
                                        if ow % sw != 0:
                                            continue
                                        ow //= sw
                                        if ow < 0 or ow >= out_w:
                                            continue
                                        acc += grad_y_np[b_idx, oc, oh, ow] * self.weight.data[oc, ic, kh_idx, kw_idx]
                            grad_x_np[b_idx, ic, ih, iw] = acc
            grad_x = GPUTensor.from_numpy(grad_x_np)
            self.weight.grad[...] += grad_w
            self.bias.grad[...] += grad_b
        self.cache.pop('w_gpu', None)
        self.cache.pop('b_gpu', None)
        return grad_x

    def parameters(self) -> Iterable[Parameter]:
        yield self.weight
        yield self.bias


class Linear:
    def __init__(self, in_features: int, out_features: int):
        limit = math.sqrt(6.0 / (in_features + out_features))
        weight = np.random.uniform(-limit, limit, size=(in_features, out_features)).astype(np.float32)
        bias = np.zeros(out_features, dtype=np.float32)
        self.weight = Parameter(weight)
        self.bias = Parameter(bias)
        self.cache: Dict[str, Any] = {}

    def forward(self, x: GPUTensor) -> GPUTensor:
        x_np = x.to_numpy().astype(np.float32, copy=False)
        orig_shape = x_np.shape
        if x_np.ndim == 2:
            flat_in = x_np
        elif x_np.ndim == 3:
            flat_in = x_np.reshape(-1, x_np.shape[-1])
        else:
            raise ValueError(f"Linear erwartet Tensor mit 2 oder 3 Dimensionen, erhielt {x_np.shape}")
        flat_out = flat_in @ self.weight.data + self.bias.data
        if x_np.ndim == 2:
            out_np = flat_out
        else:
            out_np = flat_out.reshape(orig_shape[0], orig_shape[1], -1)
        y = GPUTensor.from_numpy(out_np.astype(np.float32, copy=False))
        self.cache = {
            'input_numpy': x_np,
            'orig_shape': orig_shape,
        }
        return y

    def backward(self, grad_out: GPUTensor) -> GPUTensor:
        grad_np = grad_out.to_numpy().astype(np.float32, copy=False)
        input_np = self.cache['input_numpy']
        orig_shape = self.cache['orig_shape']
        if grad_np.ndim == 2:
            flat_grad = grad_np
            flat_input = input_np
        elif grad_np.ndim == 3:
            flat_grad = grad_np.reshape(-1, grad_np.shape[-1])
            flat_input = input_np.reshape(-1, input_np.shape[-1])
        else:
            raise ValueError(f"Linear.backward erhielt unerwartete Grad-Form {grad_np.shape}")

        grad_w = flat_input.T @ flat_grad
        grad_b = flat_grad.sum(axis=0)
        grad_in = flat_grad @ self.weight.data.T

        self.weight.grad[...] += grad_w.reshape(self.weight.data.shape)
        self.bias.grad[...] += grad_b.reshape(self.bias.data.shape)

        if grad_np.ndim == 2:
            grad_in_np = grad_in
        else:
            grad_in_np = grad_in.reshape(orig_shape)
        return GPUTensor.from_numpy(grad_in_np.astype(np.float32, copy=False))

    def parameters(self) -> Iterable[Parameter]:
        yield self.weight
        yield self.bias


@dataclass
class PatchConfig:
    in_ch: int
    d_model: int
    patch_h: int
    patch_w: int
    img_h: int
    img_w: int


class PatchEmbed:
    def __init__(self, pc: PatchConfig):
        self.pc = pc
        self.conv = Conv2D(pc.in_ch, pc.d_model, (pc.patch_h, pc.patch_w), (pc.patch_h, pc.patch_w))

    def forward(self, x: GPUTensor) -> GPUTensor:
        driver = _load_opencl_driver()
        f = self.conv.forward(x)
        B, D, Hp, Wp = f.shape
        tokens = GPUTensor((B, Wp * Hp, D))
        if USE_OPENCL and f.ptr and tokens.ptr:
            driver.offload_permute_and_reshape(f.ptr, tokens.ptr, B, D, Hp, Wp)
        else:
            f_np = f.to_numpy()
            tokens_np = f_np.transpose(0, 3, 2, 1).reshape(B, Wp * Hp, D)
            tokens = GPUTensor.from_numpy(tokens_np)
        self.cache = {'f': f}
        return tokens

    def backward(self, grad_tokens: GPUTensor) -> GPUTensor:
        driver = _load_opencl_driver()
        f = self.cache['f']
        grad_f = GPUTensor(f.shape)
        B, D, Hp, Wp = f.shape
        if USE_OPENCL and grad_tokens.ptr and grad_f.ptr:
            driver.offload_reshape_and_permute_backward(grad_tokens.ptr, grad_f.ptr, B, D, Hp, Wp)
        else:
            grad_tokens_np = grad_tokens.to_numpy()
            grad_f_np = grad_tokens_np.reshape(B, Wp, Hp, D).transpose(0, 3, 2, 1)
            grad_f = GPUTensor.from_numpy(grad_f_np)
        return self.conv.backward(grad_f)

    def parameters(self) -> Iterable[Parameter]:
        yield from self.conv.parameters()


class EONEncoder:
    # ...
    def forward(self, x: GPUTensor) -> GPUTensor:
        driver = _load_opencl_driver()
        x_out = GPUTensor(x.shape)
        if USE_OPENCL and x.ptr and x_out.ptr:
            driver.execute_eon_encoder_chain(x.ptr, x_out.ptr, x.nbytes)
        else:
            x_out = GPUTensor.from_numpy(x.to_numpy())
        return x_out

    def backward(self, grad_out: GPUTensor) -> GPUTensor:
        driver = _load_opencl_driver()
        grad_in = GPUTensor(grad_out.shape)
        if USE_OPENCL and grad_out.ptr and grad_in.ptr:
            driver.execute_eon_encoder_backward_chain(grad_out.ptr, grad_in.ptr, grad_out.nbytes)
        else:
            grad_in = GPUTensor.from_numpy(grad_out.to_numpy())
        return grad_in

class TokenClassificationHead:
    def __init__(self, hidden_dim: int, vocab_size: int):
        self.proj = Linear(hidden_dim, vocab_size)

    def forward(self, x: GPUTensor) -> GPUTensor:
        return self.proj.forward(x)

    def backward(self, grad_out: GPUTensor) -> GPUTensor:
        return self.proj.backward(grad_out)

    def parameters(self) -> Iterable[Parameter]:
        yield from self.proj.parameters()


@dataclass
class OCRSample:
    image: np.ndarray
    targets: Sequence[int]


@dataclass
class ModelConfig:
    patch: PatchConfig
    vocab_size: int


@dataclass
class TrainCfg:
    epochs: int
    batch_size: int
    lr: float
    blank_index: int
    train_samples: Sequence[OCRSample]
    model: ModelConfig
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    shuffle: bool = True
    seed: int = 42


class EONOCRModel:
    def __init__(self, cfg: ModelConfig):
        self.patch = PatchEmbed(cfg.patch)
        self.encoder = EONEncoder()
        self.head = TokenClassificationHead(cfg.patch.d_model, cfg.vocab_size)

    def parameters(self) -> Iterable[Parameter]:
        for module in (self.patch, self.head):
            yield from module.parameters()

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.zero_grad()


class AdamW_NumPy:
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.params = [p for p in params]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        beta1_t = 1.0 - self.beta1 ** self.step_count
        beta2_t = 1.0 - self.beta2 ** self.step_count
        for param in self.params:
            grad = param.grad
            if grad is None:
                continue
            param.m = self.beta1 * param.m + (1.0 - self.beta1) * grad
            param.v = self.beta2 * param.v + (1.0 - self.beta2) * (grad * grad)
            m_hat = param.m / beta1_t
            v_hat = param.v / beta2_t
            if self.weight_decay != 0.0:
                param.data *= (1.0 - self.lr * self.weight_decay)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _iterate_minibatches(
    samples: Sequence[OCRSample],
    batch_size: int,
    blank_index: int,
    shuffle: bool = True,
) -> List[Dict[str, np.ndarray]]:
    if not samples:
        return []
    indices = list(range(len(samples)))
    if shuffle:
        random.shuffle(indices)
    batches: List[Dict[str, np.ndarray]] = []
    for start in range(0, len(indices), batch_size):
        excerpt = [samples[idx] for idx in indices[start:start + batch_size]]
        if not excerpt:
            continue
        images = np.stack([np.asarray(sample.image, dtype=np.float32) for sample in excerpt], axis=0)
        max_target = max((len(sample.targets) for sample in excerpt), default=0)
        if max_target > 0:
            targets = np.full((len(excerpt), max_target), blank_index, dtype=np.int32)
        else:
            targets = np.zeros((len(excerpt), 0), dtype=np.int32)
        target_lengths = np.zeros((len(excerpt),), dtype=np.int32)
        for i, sample in enumerate(excerpt):
            seq = np.asarray(sample.targets, dtype=np.int32)
            length = min(len(seq), max_target)
            if length > 0:
                targets[i, :length] = seq[:length]
            target_lengths[i] = length
        batches.append({
            'images': images,
            'targets': targets,
            'target_lengths': target_lengths,
        })
    return batches


def _ctc_loss_with_torch(
    logits: np.ndarray,
    targets: np.ndarray,
    target_lengths: np.ndarray,
    input_lengths: np.ndarray,
    blank_index: int,
) -> Tuple[float, np.ndarray]:
    logits_t = torch.tensor(logits, dtype=torch.float32, requires_grad=True)
    log_probs = logits_t.log_softmax(dim=-1).permute(1, 0, 2).contiguous()
    target_list: List[int] = []
    for b, length in enumerate(target_lengths):
        if length > 0:
            target_list.extend(targets[b, :length].tolist())
    targets_t = torch.tensor(target_list, dtype=torch.int32)
    if targets_t.numel() == 0:
        targets_t = torch.zeros((0,), dtype=torch.int32)
    target_lengths_t = torch.tensor(target_lengths, dtype=torch.int32)
    input_lengths_t = torch.tensor(input_lengths, dtype=torch.int32)
    loss_t = F.ctc_loss(
        log_probs,
        targets_t,
        input_lengths_t,
        target_lengths_t,
        blank=blank_index,
        reduction='mean',
        zero_infinity=True,
    )
    loss_t.backward()
    grad = logits_t.grad.detach().numpy().astype(np.float32)
    return float(loss_t.item()), grad

# =============================================================================
# Angepasste Trainingsschleife
# =============================================================================


def train_ocr(cfg: TrainCfg) -> Dict[str, Any]:
    if not isinstance(cfg, TrainCfg):
        raise TypeError("train_ocr erwartet eine Instanz von TrainCfg")
    if not cfg.train_samples:
        raise ValueError("Trainingsdaten dürfen nicht leer sein")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model = EONOCRModel(cfg.model)
    optimizer = AdamW_NumPy(model.parameters(), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)

    history: Dict[str, List[float]] = {"loss": []}

    for epoch in range(1, cfg.epochs + 1):
        epoch_losses: List[float] = []
        batches = _iterate_minibatches(cfg.train_samples, cfg.batch_size, cfg.blank_index, shuffle=cfg.shuffle)
        for batch in batches:
            model.zero_grad()
            images_np = batch["images"].astype(np.float32, copy=False)
            gpu_images = GPUTensor.from_numpy(images_np)

            gpu_tokens = model.patch.forward(gpu_images)
            gpu_encoded = model.encoder.forward(gpu_tokens)
            gpu_logits = model.head.forward(gpu_encoded)

            logits_np = gpu_logits.to_numpy().astype(np.float32, copy=False)
            B, T, V = logits_np.shape
            input_lengths = np.full((B,), T, dtype=np.int32)
            targets = batch["targets"]
            target_lengths = batch["target_lengths"]

            if USE_OPENCL:
                loss_vec, grad_logits_np = DRIVER.offload_ctc_loss(
                    logits_np,
                    targets,
                    target_lengths,
                    input_lengths,
                    cfg.blank_index,
                )
                loss_value = float(np.mean(loss_vec))
            else:
                loss_value, grad_logits_np = _ctc_loss_with_torch(
                    logits_np,
                    targets,
                    target_lengths,
                    input_lengths,
                    cfg.blank_index,
                )

            epoch_losses.append(loss_value)

            gpu_grad_logits = GPUTensor.from_numpy(grad_logits_np.astype(np.float32, copy=False))

            grad_encoded = model.head.backward(gpu_grad_logits)
            grad_tokens = model.encoder.backward(grad_encoded)
            model.patch.backward(grad_tokens)

            optimizer.step()

        mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        history["loss"].append(mean_epoch_loss)

    return history


if __name__ == "__main__":
    print("EON OCR-Modul geladen. Verwenden Sie train_ocr(...) mit einer TrainCfg-Konfiguration.")

