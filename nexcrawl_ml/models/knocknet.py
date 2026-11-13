"""
KnockNet - 4-layer CNN for acoustic knock detection.

Task: given a 1-second audio clip, classify it as a deliberate human knock (a
trapped miner signalling), a human voice (shouting/talking - a rescuer or a
conscious victim), the rover's own motor/machinery hum, or silence.

Input : log-mel spectrogram, shape (1, n_mels, n_frames)
Output: 4-class logits -> {knock, voice, motor_hum, silence}

Design notes:
  - 4 conv blocks, ~200k parameters. Small enough to run comfortably on a
    Pi 5 CPU via ONNX Runtime, large enough to learn spectro-temporal
    structure that a simple energy threshold misses.
  - 'voice' is its own class, not merged into 'knock'. A shout and a
    deliberate knock call for different operator responses (a knock is a
    rescue-protocol signal; a voice may carry words worth relaying), and a
    3-class model with no voice category tends to confuse the two - exactly
    the failure mode a confusion matrix review would need to catch.
  - 'motor_hum' is an explicit class, not lumped into a background class.
    The rover's own drivetrain is the single most likely false-positive
    source, so the model is given the chance to learn it directly.
  - BatchNorm + dropout because a few hundred training clips per class is a
    small dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

    class _Stub:
        Module = object

    nn = _Stub()  # type: ignore

KNOCK_CLASSES: list[str] = ["knock", "voice", "motor_hum", "silence"]


if _HAS_TORCH:

    class ConvBlock(nn.Module):
        """Conv -> BatchNorm -> ReLU -> MaxPool, the standard audio-CNN unit."""

        def __init__(self, in_ch: int, out_ch: int, pool: tuple[int, int] = (2, 2)):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(out_ch)
            self.pool = nn.MaxPool2d(pool)

        def forward(self, x):
            return self.pool(F.relu(self.bn(self.conv(x))))

    class KnockNet(nn.Module):
        """4-block CNN with global pooling for variable-length clips."""

        def __init__(self, n_classes: int = 4, dropout: float = 0.3):
            super().__init__()
            self.block1 = ConvBlock(1, 16)
            self.block2 = ConvBlock(16, 32)
            self.block3 = ConvBlock(32, 64)
            self.block4 = ConvBlock(64, 128)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(128, n_classes)
            self.n_classes = n_classes

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            # Global average pool over both freq and time -> clip-length agnostic.
            x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            return self.fc(self.dropout(x))

        @torch.no_grad()
        def predict_proba(self, x) -> np.ndarray:
            self.eval()
            return F.softmax(self.forward(x), dim=1).cpu().numpy()

else:  # pragma: no cover

    class KnockNet:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("pytorch is required: pip install torch")


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 8
    device: str = "cpu"


def train_knocknet(
    model,
    train_loader,
    val_loader,
    config: TrainConfig | None = None,
    class_weights: np.ndarray | None = None,
):
    """Standard supervised training loop with early stopping.

    class_weights matters: staged datasets almost always have far more
    silence/motor clips than genuine knocks, and an unweighted loss teaches
    the model that predicting 'never a knock' is a great strategy.
    """
    if not _HAS_TORCH:
        raise ImportError("pytorch is required: pip install torch")

    config = config or TrainConfig()
    device = torch.device(config.device)
    model = model.to(device)

    weight_tensor = (
        torch.tensor(class_weights, dtype=torch.float32, device=device)
        if class_weights is not None
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val = float("inf")
    best_state = None
    patience = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= max(len(train_loader.dataset), 1)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)
        val_loss /= max(total, 1)
        val_acc = correct / max(total, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        scheduler.step(val_loss)

        print(
            f"epoch {epoch+1:3d}/{config.epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.3f}"
        )

        if val_loss < best_val:
            best_val, patience = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                print(f"early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def export_onnx(
    model,
    output_path: str | Path,
    n_mels: int = 64,
    n_frames: int = 87,
    opset: int = 17,
) -> Path:
    """Export to ONNX for CPU inference on the Pi 5.

    Both batch and time axes are dynamic so the same exported file handles a
    single live clip and an offline batch without re-export.

    dynamo=False is explicit and load-bearing, not a stylistic choice.
    PyTorch >= 2.5 defaults torch.onnx.export to its newer dynamo-based
    exporter, which needs the separate `onnxscript` package and uses a
    different dynamic_shapes API - it silently ignores or errors on the
    dynamic_axes argument below. Pinning dynamo=False keeps this function's
    behaviour identical across torch versions instead of depending on
    whichever exporter happens to be the installed version's default.
    """
    if not _HAS_TORCH:
        raise ImportError("pytorch is required")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.randn(1, 1, n_mels, n_frames)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        dynamo=False,
        input_names=["mel_spectrogram"],
        output_names=["logits"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch", 3: "time"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
    )
    return output_path


class ONNXKnockDetector:
    """Runtime wrapper used on the rover - no PyTorch dependency needed."""

    def __init__(self, onnx_path: str | Path, classes: list[str] | None = None):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.classes = classes or KNOCK_CLASSES

    def predict(self, mel: np.ndarray) -> dict:
        """mel: (n_mels, n_frames) or (batch, 1, n_mels, n_frames)."""
        x = np.asarray(mel, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, None, :, :]
        elif x.ndim == 3:
            x = x[:, None, :, :]

        logits = self.session.run(None, {self.input_name: x})[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba = exp / exp.sum(axis=1, keepdims=True)

        idx = int(np.argmax(proba[0]))
        return {
            "label": self.classes[idx],
            "confidence": float(proba[0][idx]),
            "distribution": {
                c: float(p) for c, p in zip(self.classes, proba[0])
            },
        }
