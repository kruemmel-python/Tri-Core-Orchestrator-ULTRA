import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
mod = pytest.importorskip("streamlit_tri_core_ultra")

BASE_LR = mod.BASE_LR
LR_MIN = mod.LR_MIN
LR_MAX = mod.LR_MAX
assignment_metrics = mod.assignment_metrics
build_export_payload = mod.build_export_payload
golden_run_snapshot = mod.golden_run_snapshot
lr_modulated = mod.lr_modulated
num_params_for_ansatz = mod.num_params_for_ansatz
proto_lr_mask_from_field = mod.proto_lr_mask_from_field


def test_lr_modulated_mix_component():
    score = 0.75
    lr_mix = lr_modulated(BASE_LR, score, "mix", 2.0, 0.3, mix_weights=(0.7, 0.3))
    lr_exp = lr_modulated(BASE_LR, score, "exp", 2.0, 0.3)
    lr_sig = lr_modulated(BASE_LR, score, "sigmoid", 2.0, 0.3)
    expected = 0.7 * lr_exp + 0.3 * lr_sig
    assert LR_MIN <= lr_mix <= LR_MAX
    assert np.isclose(lr_mix, expected, atol=1e-6)


def test_proto_lr_mask_monotonic():
    field = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    mask = proto_lr_mask_from_field(field, 8, strength=1.0)
    assert mask.shape == (8,)
    assert np.all(mask >= 0.1) and np.all(mask <= 2.5)
    assert mask[-1] >= mask[0]


def test_assignment_metrics_entropy_and_coverage():
    idx = np.array([0, 0, 1, 2, 2, 3, 3, 3], dtype=np.int32)
    entropy, coverage = assignment_metrics(idx, 4)
    assert coverage == 1.0
    assert entropy > 0


def test_num_params_for_ansatz_gate_awareness():
    base = num_params_for_ansatz("Hardware-Efficient", 4, 2, ["RX", "RY"])
    extended = num_params_for_ansatz("Custom-CR", 4, 2, ["RX", "RY", "CRX"])
    assert extended > base


def test_golden_snapshot_matches_fixture():
    fixture_path = Path(__file__).parent / "data" / "golden_export.json"
    with fixture_path.open("r", encoding="utf-8") as fh:
        golden = json.load(fh)
    snapshot = golden_run_snapshot(seed=123)
    assert snapshot == golden


def test_build_export_payload_roundtrip():
    history = {
        "epoch": [1, 2],
        "field_score": [0.1, 0.2],
        "vqe_best_E": [0.5, 0.4],
        "delta_proto_l2_total": [0.01, 0.02],
        "noise_set": [0.03, 0.04],
        "stability_index": [0.1, 0.05],
        "assignment_entropy": [0.8, 0.9],
        "proto_coverage": [0.75, 0.8],
        "epoch_ms": [10.0, 11.0],
        "energy_delta_per_sec": [0.2, 0.25],
    }
    params = {"qubits": 4, "layers": 1}
    export = build_export_payload(history, params)
    assert export["epoch"] == history["epoch"]
    assert export["params"] == params
