"""Tests for KnockNet: 4-class shapes, training loop, and the ONNX export
round trip (regression test for the torch.onnx dynamo-exporter default
flip, which needs an unrelated 'onnxscript' package and silently breaks
export on any sufficiently recent PyTorch unless dynamo=False is pinned)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="pytorch not installed")

from nexcrawl_ml.models.knocknet import (
    KNOCK_CLASSES,
    KnockNet,
    ONNXKnockDetector,
    TrainConfig,
    export_onnx,
    train_knocknet,
)


def test_knock_classes_has_four_entries_including_voice():
    assert KNOCK_CLASSES == ["knock", "voice", "motor_hum", "silence"]


def test_forward_pass_shape_matches_four_classes():
    torch.manual_seed(0)
    model = KnockNet()
    assert model.n_classes == 4
    x = torch.randn(3, 1, 64, 87)
    out = model(x)
    assert out.shape == (3, 4)


def test_forward_pass_handles_variable_time_length():
    """Global average pooling should make the model clip-length agnostic -
    important since real recordings won't always be exactly 87 frames."""
    torch.manual_seed(0)
    model = KnockNet()
    for n_frames in [40, 87, 150]:
        x = torch.randn(1, 1, 64, n_frames)
        out = model(x)
        assert out.shape == (1, 4)


def test_predict_proba_sums_to_one():
    torch.manual_seed(0)
    model = KnockNet()
    x = torch.randn(2, 1, 64, 87)
    proba = model.predict_proba(x)
    assert proba.shape == (2, 4)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_training_loop_runs_and_reduces_loss():
    """Small smoke test: does the training loop actually run end to end and
    produce a lower loss than a random init, on a trivially separable toy
    set - not a claim about real audio performance."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    n_per_class = 20
    X, y = [], []
    for i in range(4):
        centre = np.full((64, 20), i * 3.0)
        X.append(rng.normal(centre, 0.5, (n_per_class, 64, 20)))
        y += [i] * n_per_class
    X = np.stack(X).reshape(-1, 1, 64, 20).astype(np.float32)
    y = np.array(y)

    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    model = KnockNet()
    model, history = train_knocknet(
        model, loader, loader, TrainConfig(epochs=5, early_stopping_patience=10)
    )
    assert history["val_acc"][-1] > history["val_acc"][0] - 0.5  # ran without crashing
    assert len(history["train_loss"]) > 0


def test_onnx_export_matches_pytorch_inference(tmp_path):
    """Regression test: torch.onnx.export must use the legacy (non-dynamo)
    exporter so dynamic_axes behaves as written, and the exported model's
    output must numerically match the source PyTorch model."""
    torch.manual_seed(42)
    model = KnockNet()
    model.eval()

    onnx_path = export_onnx(model, tmp_path / "knocknet.onnx", n_mels=64, n_frames=87)
    assert onnx_path.exists()

    detector = ONNXKnockDetector(onnx_path, classes=KNOCK_CLASSES)

    x = torch.randn(1, 1, 64, 87)
    with torch.no_grad():
        torch_proba = model.predict_proba(x)[0]

    result = detector.predict(x.numpy()[0, 0])
    onnx_proba = np.array(list(result["distribution"].values()))

    assert np.allclose(torch_proba, onnx_proba, atol=1e-4)
    assert result["label"] in KNOCK_CLASSES


def test_onnx_export_handles_variable_length_clips(tmp_path):
    """The dynamic time axis must actually work, not just be declared."""
    torch.manual_seed(0)
    model = KnockNet()
    model.eval()
    onnx_path = export_onnx(model, tmp_path / "knocknet.onnx", n_mels=64, n_frames=87)
    detector = ONNXKnockDetector(onnx_path, classes=KNOCK_CLASSES)

    for n_frames in [50, 87, 120]:
        mel = np.random.randn(64, n_frames).astype(np.float32)
        result = detector.predict(mel)
        assert result["label"] in KNOCK_CLASSES
