# conftest.py
# Setzt --dll/--gpu via ENV, erzwingt headless Streamlit
# und stellt den "ctx"-Fixture bereit (kompatibel zu function_test.py).
from __future__ import annotations

import os
import ctypes
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--dll", action="store", default=None, help="Pfad zur CipherCore DLL")
    parser.addoption("--gpu", action="store", default=None, help="GPU-Index")


@pytest.fixture(autouse=True)
def _apply_cmdline_overrides(request: pytest.FixtureRequest):
    """
    Function-scope (wichtig gegen ScopeMismatch).
    Setzt Streamlit/Env sauber für Bare-Mode Tests.
    """
    dll_cli = request.config.getoption("--dll")
    gpu_cli = request.config.getoption("--gpu")

    # Streamlit: Headless + still (unterdrückt Bare-Mode-Spam)
    os.environ["STREAMLIT_HEADLESS"] = "true"
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_LOG_LEVEL", "error")  # error|warning|info|debug

    # Optional: global pytest/py warnings dämpfen (keine Deprecation-Flut)
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

    if dll_cli:
        os.environ["CIPHERCORE_DLL"] = dll_cli
    if gpu_cli is not None:
        os.environ["CIPHERCORE_GPU"] = str(gpu_cli)


def _set_proto(dll: ctypes.CDLL, name: str, restype, argtypes: Iterable) -> None:
    func = getattr(dll, name)
    func.restype = restype
    func.argtypes = list(argtypes)


def _load_dll_from_env() -> ctypes.CDLL:
    dll_path_env = os.getenv("CIPHERCORE_DLL", r"build\CipherCore_OpenCl.dll")
    dll_path = Path(dll_path_env)
    if not dll_path.exists():
        raise FileNotFoundError(f"CipherCore DLL nicht gefunden: {dll_path}")
    dll = ctypes.CDLL(str(dll_path))

    # Pflichtsymbole (weitere Protos setzt function_test.py selbst)
    _set_proto(dll, "initialize_gpu", ctypes.c_int, [ctypes.c_int])
    _set_proto(dll, "shutdown_gpu", None, [ctypes.c_int])
    _set_proto(dll, "finish_gpu", ctypes.c_int, [ctypes.c_int])
    _set_proto(dll, "allocate_gpu_memory", ctypes.c_void_p, [ctypes.c_int, ctypes.c_size_t])
    _set_proto(dll, "free_gpu_memory", None, [ctypes.c_int, ctypes.c_void_p])
    _set_proto(
        dll, "write_host_to_gpu_blocking", ctypes.c_int,
        [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p],
    )
    _set_proto(
        dll, "read_gpu_to_host_blocking", ctypes.c_int,
        [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p],
    )
    return dll


@dataclass
class _DeviceBuffer:
    handle: ctypes.c_void_p
    size_bytes: int


class _DriverCtx:
    def __init__(self, dll: ctypes.CDLL, gpu_index: int):
        self._dll = dll
        self.gpu_index = int(gpu_index)
        ok = self._dll.initialize_gpu(self.gpu_index)
        if ok != 1:
            raise RuntimeError(f"initialize_gpu({self.gpu_index}) failed")
        self._buffers: List[_DeviceBuffer] = []

    def alloc(self, size_bytes: int) -> ctypes.c_void_p:
        handle = ctypes.c_void_p(self._dll.allocate_gpu_memory(self.gpu_index, size_bytes))
        if not handle.value:
            raise MemoryError(f"allocate_gpu_memory({size_bytes}) failed")
        self._buffers.append(_DeviceBuffer(handle, size_bytes))
        return handle

    def write_floats(self, handle: ctypes.c_void_p, values: Sequence[float]) -> None:
        array_type = ctypes.c_float * len(values)
        host_array = array_type(*map(float, values))
        size_bytes = ctypes.sizeof(host_array)
        ok = self._dll.write_host_to_gpu_blocking(
            self.gpu_index, handle, ctypes.c_size_t(0), ctypes.c_size_t(size_bytes),
            ctypes.cast(host_array, ctypes.c_void_p),
        )
        if ok != 1:
            raise RuntimeError("write_host_to_gpu_blocking failed")

    def read_floats(self, handle: ctypes.c_void_p, count: int) -> List[float]:
        array_type = ctypes.c_float * count
        host_array = array_type()
        size_bytes = ctypes.sizeof(host_array)
        ok = self._dll.read_gpu_to_host_blocking(
            self.gpu_index, handle, ctypes.c_size_t(0), ctypes.c_size_t(size_bytes),
            ctypes.cast(host_array, ctypes.c_void_p),
        )
        if ok != 1:
            raise RuntimeError("read_gpu_to_host_blocking failed")
        return list(host_array)

    def free_all(self) -> None:
        for buf in self._buffers:
            with contextlib.suppress(Exception):
                self._dll.free_gpu_memory(self.gpu_index, buf.handle)
        self._buffers.clear()

    def close(self) -> None:
        try:
            self.free_all()
        finally:
            with contextlib.suppress(Exception):
                self._dll.finish_gpu(self.gpu_index)
            with contextlib.suppress(Exception):
                self._dll.shutdown_gpu(self.gpu_index)

    def __enter__(self) -> "_DriverCtx":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@pytest.fixture(scope="session")
def dll() -> ctypes.CDLL:
    return _load_dll_from_env()


@pytest.fixture(scope="session")
def gpu_index() -> int:
    return int(os.getenv("CIPHERCORE_GPU", "0"))


@pytest.fixture
def ctx(dll: ctypes.CDLL, gpu_index: int):
    c = _DriverCtx(dll, gpu_index)
    try:
        yield c
    finally:
        c.close()
