"""Functional validation harness for CipherCore_OpenCl.dll.

This script exercises a representative subset of the exported GPU
kernels exposed by ``CipherCore_OpenCl.c``.  It focuses on core tensor
operators, the Adam optimizer, embedding lookup, and the
``execute_quantum_echoes_otoc_gpu`` quantum routine.  Each test allocates
GPU buffers through the public C API, writes deterministic host data,
invokes the kernel, and validates the device results against a NumPy-free
reference implementation on the host.

The harness is intentionally lightweight so it can run on systems where
only the Microsoft Visual C++ runtime and a working OpenCL stack are
present.  No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# DLL bootstrap
# ---------------------------------------------------------------------------

DLL_PATH = Path(r"build\CipherCore_OpenCl.dll")

if not DLL_PATH.exists():
    raise FileNotFoundError(f"Expected DLL at {DLL_PATH!s} was not found.")

dll = ctypes.CDLL(str(DLL_PATH))


# ---------------------------------------------------------------------------
# ctypes structure mirrors
# ---------------------------------------------------------------------------


class ClFloat2(ctypes.Structure):
    _fields_ = [("s", ctypes.c_float * 2)]


class QuantumGate(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 8),
        ("arity", ctypes.c_uint),
        ("control", ctypes.c_uint),
        ("target", ctypes.c_uint),
        ("control2", ctypes.c_uint),
        ("params", ctypes.c_float * 4),
        ("matrix", ClFloat2 * 8 * 8),
    ]


def _assert_gate_layout_safe() -> None:
    expected = {
        "name": 8,
        "params": 16,
        "cl_float2": 8,
        "matrix": 8 * 8 * 8,  # 8x8 cl_float2
    }
    actual = {
        "name": ctypes.sizeof(ctypes.c_char * 8),
        "params": ctypes.sizeof(ctypes.c_float * 4),
        "cl_float2": ctypes.sizeof(ClFloat2),
        "matrix": ctypes.sizeof((ClFloat2 * 8) * 8),
    }
    problems = [k for k in expected if expected[k] != actual[k]]
    if problems:
        dump = ", ".join(f"{k}: exp={expected[k]} got={actual[k]}" for k in problems)
        raise RuntimeError(f"QuantumGate ctypes layout mismatch -> {dump}")



_assert_gate_layout_safe()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _set_proto(name: str, restype, argtypes: Iterable) -> None:
    func = getattr(dll, name)
    func.restype = restype
    func.argtypes = list(argtypes)


_set_proto("initialize_gpu", ctypes.c_int, [ctypes.c_int])
_set_proto("shutdown_gpu", None, [ctypes.c_int])
_set_proto("finish_gpu", ctypes.c_int, [ctypes.c_int])
_set_proto("allocate_gpu_memory", ctypes.c_void_p, [ctypes.c_int, ctypes.c_size_t])
_set_proto("free_gpu_memory", None, [ctypes.c_int, ctypes.c_void_p])
_set_proto(
    "write_host_to_gpu_blocking",
    ctypes.c_int,
    [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p],
)
_set_proto(
    "read_gpu_to_host_blocking",
    ctypes.c_int,
    [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p],
)
_set_proto(
    "execute_add_on_gpu",
    ctypes.c_int,
    [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int],
)
_set_proto(
    "execute_mul_on_gpu",
    ctypes.c_int,
    [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int],
)
_set_proto(
    "execute_matmul_on_gpu",
    ctypes.c_int,
    [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
)
_set_proto(
    "execute_softmax_on_gpu",
    ctypes.c_int,
    [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
)
_set_proto(
    "execute_gelu_on_gpu",
    ctypes.c_int,
    [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int],
)
_set_proto(
    "execute_layernorm_on_gpu",
    ctypes.c_int,
    [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float],
)
_set_proto(
    "execute_transpose_on_gpu",
    ctypes.c_int,
    [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
)
_set_proto(
    "execute_adam_update_on_gpu",
    ctypes.c_int,
    [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ],
)
_set_proto(
    "execute_embedding_lookup_gpu",
    ctypes.c_int,
    [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int],
)
_set_proto(
    "execute_quantum_echoes_otoc_gpu",
    ctypes.c_int,
    [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(QuantumGate),
        ctypes.c_int,
        ctypes.POINTER(QuantumGate),
        ctypes.POINTER(QuantumGate),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ],
)


@dataclass
class DeviceBuffer:
    handle: ctypes.c_void_p
    size_bytes: int


class DriverContext:
    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        if dll.initialize_gpu(gpu_index) != 1:
            raise RuntimeError(f"initialize_gpu({gpu_index}) failed")
        self._buffers: List[DeviceBuffer] = []

    def alloc(self, size_bytes: int) -> ctypes.c_void_p:
        handle = ctypes.c_void_p(dll.allocate_gpu_memory(self.gpu_index, size_bytes))
        if not handle.value:
            raise MemoryError(f"allocate_gpu_memory failed for {size_bytes} bytes")
        self._buffers.append(DeviceBuffer(handle, size_bytes))
        return handle

    def free_all(self) -> None:
        for buf in self._buffers:
            dll.free_gpu_memory(self.gpu_index, buf.handle)
        self._buffers.clear()

    def close(self) -> None:
        try:
            self.free_all()
        finally:
            dll.finish_gpu(self.gpu_index)
            dll.shutdown_gpu(self.gpu_index)

    def write_floats(self, handle: ctypes.c_void_p, values: Sequence[float]) -> None:
        array_type = ctypes.c_float * len(values)
        host_array = array_type(*values)
        size_bytes = ctypes.sizeof(host_array)
        if (
            dll.write_host_to_gpu_blocking(
                self.gpu_index,
                handle,
                ctypes.c_size_t(0),
                ctypes.c_size_t(size_bytes),
                ctypes.cast(host_array, ctypes.c_void_p),
            )
            != 1
        ):
            raise RuntimeError("write_host_to_gpu_blocking failed")

    def read_floats(self, handle: ctypes.c_void_p, count: int) -> List[float]:
        array_type = ctypes.c_float * count
        host_array = array_type()
        size_bytes = ctypes.sizeof(host_array)
        if (
            dll.read_gpu_to_host_blocking(
                self.gpu_index,
                handle,
                ctypes.c_size_t(0),
                ctypes.c_size_t(size_bytes),
                ctypes.cast(host_array, ctypes.c_void_p),
            )
            != 1
        ):
            raise RuntimeError("read_gpu_to_host_blocking failed")
        return list(host_array)

    def __enter__(self) -> "DriverContext":
        return self
    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        


def almost_equal(a: Sequence[float], b: Sequence[float], tol: float = 1e-4) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Host reference implementations
# ---------------------------------------------------------------------------


def softmax_rows(data: Sequence[float], rows: int, cols: int) -> List[float]:
    result: List[float] = []
    for r in range(rows):
        row = data[r * cols : (r + 1) * cols]
        max_val = max(row)
        exps = [math.exp(x - max_val) for x in row]
        denom = sum(exps)
        result.extend(val / denom for val in exps)
    return result


def gelu_vector(data: Sequence[float]) -> List[float]:
    out: List[float] = []
    for x in data:
        out.append(0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x))))
    return out


def layernorm_rows(data: Sequence[float], rows: int, cols: int, eps: float) -> List[float]:
    out: List[float] = []
    for r in range(rows):
        row = data[r * cols : (r + 1) * cols]
        mean = sum(row) / cols
        var = sum((x - mean) ** 2 for x in row) / cols
        denom = math.sqrt(var + eps)
        out.extend((x - mean) / denom for x in row)
    return out


def matmul_host(a: Sequence[float], b: Sequence[float], B: int, M: int, N: int, K: int) -> List[float]:
    out = [0.0] * (B * M * N)
    for batch in range(B):
        for i in range(M):
            for j in range(N):
                acc = 0.0
                for k in range(K):
                    a_idx = batch * M * K + i * K + k
                    b_idx = k * N + j
                    acc += a[a_idx] * b[b_idx]
                out[batch * M * N + i * N + j] = acc
    return out


def adam_update_host(
    params: List[float],
    grads: List[float],
    m: List[float],
    v: List[float],
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    t: int,
) -> Tuple[List[float], List[float], List[float]]:
    beta1_t = beta1 ** t
    beta2_t = beta2 ** t
    new_params: List[float] = []
    new_m: List[float] = []
    new_v: List[float] = []
    for p, g, m_val, v_val in zip(params, grads, m, v):
        if weight_decay > 0.0:
            g = g + weight_decay * p
        m_new = beta1 * m_val + (1.0 - beta1) * g
        v_new = beta2 * v_val + (1.0 - beta2) * (g * g)
        m_hat = m_new / (1.0 - beta1_t + 1e-9)
        v_hat = v_new / (1.0 - beta2_t + 1e-9)
        update = lr * m_hat / (math.sqrt(v_hat) + eps)
        p_new = p - update
        new_params.append(p_new)
        new_m.append(m_new)
        new_v.append(v_new)
    return new_params, new_m, new_v


# ---------------------------------------------------------------------------
# Quantum helper utilities
# ---------------------------------------------------------------------------


def make_gate(
    name: str,
    arity: int,
    control: int,
    target: int,
    angle: float = 0.0,
    control2: int = 0,
) -> QuantumGate:
    gate = QuantumGate()
    encoded = name.encode("ascii")
    name_array_type = QuantumGate._fields_[0][1]
    max_len = ctypes.sizeof(name_array_type)
    if len(encoded) >= max_len:
        raise ValueError(f"Gate name '{name}' exceeds {max_len - 1} characters")
    padded = encoded + b"\0" * (max_len - len(encoded))
    gate.name = padded
    gate.arity = arity
    gate.control = control
    gate.target = target
    gate.control2 = control2
    gate.params[0] = angle
    return gate


def make_u_sequence_random(
    num_qubits: int, depth: int, seed: int = 1234
) -> Tuple[ctypes.Array, int]:
    """Construct a deterministic pseudo-random ``U`` sequence."""

    if num_qubits < 2:
        raise ValueError("num_qubits >= 2 erforderlich")

    rnd = random.Random(seed)
    gates: List[QuantumGate] = []

    def rand_angle() -> float:
        return rnd.uniform(-math.pi, math.pi)

    for i in range(depth):
        tgt = rnd.randrange(0, num_qubits)
        if i % 2 == 0:
            gates.append(make_gate("RX", 1, 0, tgt, angle=rand_angle()))
        else:
            gates.append(make_gate("RZ", 1, 0, tgt, angle=rand_angle()))

    for control in range(num_qubits - 1):
        target = (control + 1) % num_qubits
        gates.append(make_gate("CNOT", 2, control, target))

    sequence = (QuantumGate * len(gates))(*gates)
    return sequence, len(gates)


def apply_single_qubit(state: List[complex], num_qubits: int, target: int, matrix: Sequence[complex]) -> None:
    stride = 1 << target
    for base in range(0, len(state), stride * 2):
        for offset in range(stride):
            i0 = base + offset
            i1 = i0 + stride
            a0 = state[i0]
            a1 = state[i1]
            state[i0] = matrix[0] * a0 + matrix[1] * a1
            state[i1] = matrix[2] * a0 + matrix[3] * a1


def apply_cnot(state: List[complex], num_qubits: int, control: int, target: int) -> None:
    control_mask = 1 << control
    target_mask = 1 << target
    for idx in range(len(state)):
        if idx & control_mask:
            flipped = idx ^ target_mask
            state[idx], state[flipped] = state[flipped], state[idx]


def simulate_quantum_sequence(
    num_qubits: int,
    U: Sequence[QuantumGate],
    W: QuantumGate,
    V: QuantumGate | None,
) -> Tuple[float, complex, complex]:
    def gate_to_matrix(g: QuantumGate) -> Sequence[complex]:
        name = bytes(g.name).split(b"\0", 1)[0].decode("ascii")
        match name:
            case "H":
                s = 1 / math.sqrt(2)
                return [s, s, s, -s]
            case "RZ":
                theta = float(g.params[0])
                c, s = math.cos(theta / 2), math.sin(theta / 2)
                return [complex(c, -s), 0, 0, complex(c, s)]
            case "RX":
                theta = float(g.params[0])
                c, s = math.cos(theta / 2), math.sin(theta / 2)
                return [c, -1j * s, -1j * s, c]
            case _:
                raise ValueError(f"Unsupported gate in simulator: {name!r}")

    def apply_sequence(state: List[complex], seq: Sequence[QuantumGate]) -> None:
        for gate in seq:
            name = bytes(gate.name).split(b"\0", 1)[0].decode("ascii")
            match name:
                case "H" | "RZ" | "RX":
                    apply_single_qubit(state, num_qubits, gate.target, gate_to_matrix(gate))
                case "CNOT":
                    apply_cnot(state, num_qubits, gate.control, gate.target)
                case _:
                    raise ValueError(f"Unsupported gate {name!r}")


    def dagger_sequence(seq: Sequence[QuantumGate]) -> List[QuantumGate]:
        result: List[QuantumGate] = []
        for gate in reversed(seq):
            base_name = bytes(gate.name).split(b"\0", 1)[0].decode("ascii")
            adj = make_gate(base_name, gate.arity, gate.control, gate.target, float(gate.params[0]), gate.control2)
            name = base_name
            if name in {"RX", "RY", "RZ"}:
                adj.params[0] = -float(gate.params[0])
            result.append(adj)
        return result

    zero_state = [0j] * (1 << num_qubits)
    zero_state[0] = 1.0 + 0j

    echo_state = zero_state.copy()
    apply_sequence(echo_state, U)
    apply_sequence(echo_state, [W])
    apply_sequence(echo_state, dagger_sequence(U))
    echo_amp = echo_state[0]

    otoc_state = zero_state.copy()
    apply_sequence(otoc_state, U)
    apply_sequence(otoc_state, [W])
    apply_sequence(otoc_state, dagger_sequence(U))
    if V is not None:
        apply_sequence(otoc_state, [V])
    apply_sequence(otoc_state, U)
    w_name = bytes(W.name).split(b"\0", 1)[0].decode("ascii")
    apply_sequence(
        otoc_state,
        [make_gate(w_name, W.arity, W.control, W.target, -float(W.params[0]), W.control2)],
    )
    apply_sequence(otoc_state, dagger_sequence(U))
    if V is not None:
        v_name = bytes(V.name).split(b"\0", 1)[0].decode("ascii")
        V_dagger = make_gate(v_name, V.arity, V.control, V.target, -float(V.params[0]), V.control2)
        apply_sequence(otoc_state, [V_dagger])
    otoc_amp = otoc_state[0]

    return abs(echo_amp) ** 2, echo_amp, otoc_amp


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_elementwise_ops(ctx: DriverContext) -> None:
    data_a = [0.1 * i for i in range(8)]
    data_b = [1.0 - 0.05 * i for i in range(8)]
    size_bytes = len(data_a) * ctypes.sizeof(ctypes.c_float)

    buf_a = ctx.alloc(size_bytes)
    buf_b = ctx.alloc(size_bytes)
    buf_c = ctx.alloc(size_bytes)

    ctx.write_floats(buf_a, data_a)
    ctx.write_floats(buf_b, data_b)

    if dll.execute_add_on_gpu(ctx.gpu_index, buf_a, buf_b, buf_c, len(data_a)) != 1:
        raise RuntimeError("execute_add_on_gpu failed")
    add_result = ctx.read_floats(buf_c, len(data_a))
    expected_add = [a + b for a, b in zip(data_a, data_b)]
    if not almost_equal(add_result, expected_add):
        raise AssertionError("Add kernel mismatch")

    if dll.execute_mul_on_gpu(ctx.gpu_index, buf_a, buf_b, buf_c, len(data_a)) != 1:
        raise RuntimeError("execute_mul_on_gpu failed")
    mul_result = ctx.read_floats(buf_c, len(data_a))
    expected_mul = [a * b for a, b in zip(data_a, data_b)]
    if not almost_equal(mul_result, expected_mul):
        raise AssertionError("Mul kernel mismatch")


def test_matmul_and_transpose(ctx: DriverContext) -> None:
    B, M, N, K = 1, 2, 3, 4
    a = [float(i + 1) for i in range(B * M * K)]
    b = [float((i % (K * N)) + 1) * 0.25 for i in range(K * N)]
    out_size = B * M * N * ctypes.sizeof(ctypes.c_float)

    buf_a = ctx.alloc(len(a) * ctypes.sizeof(ctypes.c_float))
    buf_b = ctx.alloc(len(b) * ctypes.sizeof(ctypes.c_float))
    buf_c = ctx.alloc(out_size)
    ctx.write_floats(buf_a, a)
    ctx.write_floats(buf_b, b)

    if (
        dll.execute_matmul_on_gpu(ctx.gpu_index, buf_a, buf_b, buf_c, B, M, N, K)
        != 1
    ):
        raise RuntimeError("execute_matmul_on_gpu failed")
    result = ctx.read_floats(buf_c, B * M * N)
    expected = matmul_host(a, b, B, M, N, K)
    if not almost_equal(result, expected):
        raise AssertionError("Matmul mismatch")

    # Transpose the result (M x N -> N x M)
    buf_t = ctx.alloc(out_size)
    if dll.execute_transpose_on_gpu(ctx.gpu_index, buf_c, buf_t, M, N) != 1:
        raise RuntimeError("execute_transpose_on_gpu failed")
    transpose = ctx.read_floats(buf_t, B * M * N)
    expected_t = []
    for col in range(N):
        for row in range(M):
            expected_t.append(result[row * N + col])
    if not almost_equal(transpose, expected_t):
        raise AssertionError("Transpose mismatch")


def test_activation_ops(ctx: DriverContext) -> None:
    rows, cols = 2, 4
    data = [0.25 * (i - 3) for i in range(rows * cols)]
    buf_in = ctx.alloc(len(data) * ctypes.sizeof(ctypes.c_float))
    buf_out = ctx.alloc(len(data) * ctypes.sizeof(ctypes.c_float))
    ctx.write_floats(buf_in, data)

    if dll.execute_softmax_on_gpu(ctx.gpu_index, buf_in, buf_out, rows, cols) != 1:
        raise RuntimeError("execute_softmax_on_gpu failed")
    softmax_gpu = ctx.read_floats(buf_out, len(data))
    if not almost_equal(softmax_gpu, softmax_rows(data, rows, cols), tol=1e-3):
        raise AssertionError("Softmax mismatch")

    if dll.execute_gelu_on_gpu(ctx.gpu_index, buf_in, buf_out, len(data)) != 1:
        raise RuntimeError("execute_gelu_on_gpu failed")
    gelu_gpu = ctx.read_floats(buf_out, len(data))
    if not almost_equal(gelu_gpu, gelu_vector(data), tol=1e-3):
        raise AssertionError("GELU mismatch")

    eps = 1e-4
    if (
        dll.execute_layernorm_on_gpu(ctx.gpu_index, buf_in, buf_out, rows, cols, eps)
        != 1
    ):
        raise RuntimeError("execute_layernorm_on_gpu failed")
    ln_gpu = ctx.read_floats(buf_out, len(data))
    if not almost_equal(ln_gpu, layernorm_rows(data, rows, cols, eps), tol=1e-3):
        raise AssertionError("LayerNorm mismatch")


def test_adam(ctx: DriverContext) -> None:
    params = [0.5, -0.25, 1.5, -1.0]
    grads = [0.1, -0.2, 0.3, -0.4]
    m = [0.0, 0.0, 0.0, 0.0]
    v = [0.0, 0.0, 0.0, 0.0]
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.01
    t = 1

    buf_params = ctx.alloc(len(params) * ctypes.sizeof(ctypes.c_float))
    buf_grads = ctx.alloc(len(grads) * ctypes.sizeof(ctypes.c_float))
    buf_m = ctx.alloc(len(m) * ctypes.sizeof(ctypes.c_float))
    buf_v = ctx.alloc(len(v) * ctypes.sizeof(ctypes.c_float))

    ctx.write_floats(buf_params, params)
    ctx.write_floats(buf_grads, grads)
    ctx.write_floats(buf_m, m)
    ctx.write_floats(buf_v, v)

    if (
        dll.execute_adam_update_on_gpu(
            ctx.gpu_index,
            buf_params,
            buf_grads,
            buf_m,
            buf_v,
            len(params),
            t,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
        )
        != 1
    ):
        raise RuntimeError("execute_adam_update_on_gpu failed")

    new_params = ctx.read_floats(buf_params, len(params))
    new_m = ctx.read_floats(buf_m, len(m))
    new_v = ctx.read_floats(buf_v, len(v))

    exp_params, exp_m, exp_v = adam_update_host(
        params, grads, m, v, lr, beta1, beta2, eps, weight_decay, t
    )

    if not almost_equal(new_params, exp_params, tol=1e-3):
        raise AssertionError("Adam params mismatch")
    if not almost_equal(new_m, exp_m, tol=1e-3):
        raise AssertionError("Adam m state mismatch")
    if not almost_equal(new_v, exp_v, tol=1e-3):
        raise AssertionError("Adam v state mismatch")


def test_embedding(ctx: DriverContext) -> None:
    batch, seq_len, embed_dim, vocab = 2, 2, 3, 5
    indices = [0, 1, 2, 3]
    weights = [
        1.0,
        0.0,
        0.5,
        -0.25,
        0.75,
        0.1,
        0.2,
        -0.4,
        0.3,
        0.9,
        -0.6,
        0.8,
        0.05,
        0.2,
        -0.15,
    ]
    expected: List[float] = []
    for idx in indices:
        row = weights[idx * embed_dim : (idx + 1) * embed_dim]
        expected.extend(row)

    buf_idx = ctx.alloc(len(indices) * ctypes.sizeof(ctypes.c_int))
    buf_w = ctx.alloc(len(weights) * ctypes.sizeof(ctypes.c_float))
    buf_out = ctx.alloc(len(expected) * ctypes.sizeof(ctypes.c_float))

    idx_array = (ctypes.c_int * len(indices))(*indices)
    if (
        dll.write_host_to_gpu_blocking(
            ctx.gpu_index,
            buf_idx,
            ctypes.c_size_t(0),
            ctypes.c_size_t(ctypes.sizeof(idx_array)),
            ctypes.cast(idx_array, ctypes.c_void_p),
        )
        != 1
    ):
        raise RuntimeError("write indices failed")
    ctx.write_floats(buf_w, weights)

    if (
        dll.execute_embedding_lookup_gpu(
            ctx.gpu_index,
            buf_idx,
            buf_w,
            buf_out,
            batch,
            seq_len,
            embed_dim,
            vocab,
        )
        != 1
    ):
        raise RuntimeError("execute_embedding_lookup_gpu failed")

    result = ctx.read_floats(buf_out, len(expected))
    if not almost_equal(result, expected):
        raise AssertionError("Embedding lookup mismatch")


def test_quantum_echo(ctx: DriverContext) -> None:
    num_qubits = 2
    H0 = make_gate("H", 1, 0, 0)
    CNOT01 = make_gate("CNOT", 2, 0, 1)
    W_gate = make_gate("RZ", 1, 0, 0, angle=0.3)
    V_gate = make_gate("RZ", 1, 0, 1, angle=0.17)

    U_seq = (QuantumGate * 2)(H0, CNOT01)

    L_out = ctypes.c_float()
    otoc_re = ctypes.c_float()
    otoc_im = ctypes.c_float()

    if (
        dll.execute_quantum_echoes_otoc_gpu(
            ctx.gpu_index,
            num_qubits,
            U_seq,
            len(U_seq),
            ctypes.byref(W_gate),
            ctypes.byref(V_gate),
            1,
            ctypes.byref(L_out),
            ctypes.byref(otoc_re),
            ctypes.byref(otoc_im),
        )
        != 1
    ):
        raise RuntimeError("execute_quantum_echoes_otoc_gpu failed")

    L_host, echo_amp, otoc_amp = simulate_quantum_sequence(num_qubits, U_seq, W_gate, V_gate)

    if abs(L_out.value - L_host) > 1e-3:
        raise AssertionError("Echo fidelity mismatch")
    if abs(otoc_re.value - otoc_amp.real) > 1e-3:
        raise AssertionError("OTOC real mismatch")
    if abs(otoc_im.value - otoc_amp.imag) > 1e-3:
        raise AssertionError("OTOC imag mismatch")


def test_quantum_echo_without_v(ctx: DriverContext) -> None:
    num_qubits = 2
    H0 = make_gate("H", 1, 0, 0)
    CNOT01 = make_gate("CNOT", 2, 0, 1)
    U_seq = (QuantumGate * 2)(H0, CNOT01)
    W_gate = make_gate("RZ", 1, 0, 0, angle=0.125)

    L_out = ctypes.c_float()
    otoc_re = ctypes.c_float()
    otoc_im = ctypes.c_float()

    ok = dll.execute_quantum_echoes_otoc_gpu(
        ctx.gpu_index,
        num_qubits,
        U_seq,
        len(U_seq),
        ctypes.byref(W_gate),
        ctypes.POINTER(QuantumGate)(),
        0,
        ctypes.byref(L_out),
        ctypes.byref(otoc_re),
        ctypes.byref(otoc_im),
    )
    if ok != 1:
        raise RuntimeError("execute_quantum_echoes_otoc_gpu (no V) failed")

    if not (abs(otoc_re.value) < 1e-7 and abs(otoc_im.value) < 1e-7):
        raise AssertionError("OTOC outputs must be zero when measure_otoc2=0")
    if not (0.0 <= L_out.value <= 1.0):
        raise AssertionError("Echo fidelity must be a probability")


def test_quantum_echo_random(ctx: DriverContext) -> None:
    num_qubits = 3
    U_seq, u_len = make_u_sequence_random(num_qubits=num_qubits, depth=8, seed=42)

    W_gate = make_gate("RZ", 1, 0, 0, angle=0.2)
    V_gate = make_gate("RX", 1, 0, 1, angle=-0.15)

    L_out = ctypes.c_float()
    otoc_re = ctypes.c_float()
    otoc_im = ctypes.c_float()

    ok = dll.execute_quantum_echoes_otoc_gpu(
        ctx.gpu_index,
        num_qubits,
        U_seq,
        u_len,
        ctypes.byref(W_gate),
        ctypes.byref(V_gate),
        1,
        ctypes.byref(L_out),
        ctypes.byref(otoc_re),
        ctypes.byref(otoc_im),
    )
    if ok != 1:
        raise RuntimeError("execute_quantum_echoes_otoc_gpu (random U) failed")

    if not (0.0 <= L_out.value <= 1.0):
        raise AssertionError("Echo-Fidelity outside [0,1]")
    if not (math.isfinite(otoc_re.value) and math.isfinite(otoc_im.value)):
        raise AssertionError("OTOC outputs must be finite")


TESTS = [
    ("elementwise", test_elementwise_ops),
    ("matmul_transpose", test_matmul_and_transpose),
    ("activations", test_activation_ops),
    ("adam", test_adam),
    ("embedding", test_embedding),
    ("quantum_echo", test_quantum_echo),
    ("quantum_echo_noV", test_quantum_echo_without_v),
    ("quantum_echo_random_U", test_quantum_echo_random),
]


def bench_quantum(
    ctx: DriverContext,
    num_qubits: int,
    depth: int,
    seed: int,
    warmup: int,
    repeats: int,
) -> None:
    U_seq, u_len = make_u_sequence_random(num_qubits=num_qubits, depth=depth, seed=seed)
    W_gate = make_gate("RZ", 1, 0, 0, angle=0.3)
    V_gate = make_gate("RX", 1, 0, 1, angle=0.17)

    L_out = ctypes.c_float()
    otoc_re = ctypes.c_float()
    otoc_im = ctypes.c_float()

    for _ in range(max(0, warmup)):
        dll.execute_quantum_echoes_otoc_gpu(
            ctx.gpu_index,
            num_qubits,
            U_seq,
            u_len,
            ctypes.byref(W_gate),
            ctypes.byref(V_gate),
            1,
            ctypes.byref(L_out),
            ctypes.byref(otoc_re),
            ctypes.byref(otoc_im),
        )

    times: List[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        ok = dll.execute_quantum_echoes_otoc_gpu(
            ctx.gpu_index,
            num_qubits,
            U_seq,
            u_len,
            ctypes.byref(W_gate),
            ctypes.byref(V_gate),
            1,
            ctypes.byref(L_out),
            ctypes.byref(otoc_re),
            ctypes.byref(otoc_im),
        )
        elapsed = (time.perf_counter() - start) * 1e3
        if ok != 1:
            raise RuntimeError("execute_quantum_echoes_otoc_gpu failed in benchmark")
        times.append(elapsed)

    mean = sum(times) / len(times)
    variance = 0.0
    if len(times) > 1:
        variance = sum((t - mean) ** 2 for t in times) / (len(times) - 1)
    print(
        f"[BENCH] quantum_echo: mean={mean:.3f} ms, var={variance:.5f}, n={len(times)}, "
        f"num_qubits={num_qubits}, depth={depth}, seed={seed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CipherCore_OpenCl.dll harness")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument(
        "--bench",
        choices=["none", "quantum"],
        default="none",
        help="Enable benchmark mode",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations for benchmarks")
    parser.add_argument(
        "--repeats", type=int, default=10, help="Timed iterations for benchmarks"
    )
    parser.add_argument("--qubits", type=int, default=3, help="Qubit count for quantum bench")
    parser.add_argument("--depth", type=int, default=8, help="U-sequence depth for quantum bench")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random U sequences")
    args = parser.parse_args()

    ctx = DriverContext(gpu_index=args.gpu)
    failures: List[str] = []
    try:
        if args.bench == "none":
            for name, fn in TESTS:
                try:
                    fn(ctx)
                    print(f"[PASS] {name}")
                except Exception as exc:  # noqa: BLE001 - surface rich diagnostics
                    failures.append(f"{name}: {exc}")
                    print(f"[FAIL] {name}: {exc}")
            if failures:
                joined = ", ".join(failures)
                raise SystemExit(f"Test failures: {joined}")
        else:
            if args.bench == "quantum":
                bench_quantum(
                    ctx,
                    num_qubits=args.qubits,
                    depth=args.depth,
                    seed=args.seed,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            else:
                raise SystemExit(f"Unknown benchmark: {args.bench}")
    finally:
        ctx.close()


if __name__ == "__main__":
    main()