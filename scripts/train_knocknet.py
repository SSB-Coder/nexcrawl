#!/usr/bin/env python3
"""
Train KnockNet and export to ONNX.

Expected data layout:
    data/raw/audio/
        knock/      *.wav
        voice/      *.wav
        motor_hum/  *.wav
        silence/    *.wav

Usage:
    python train_knocknet.py --data ../data/raw/audio --out ../data/models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexcrawl_ml.audio.features import (
    AudioConfig,
    augment_waveform,
    compute_log_mel,
    load_audio,
    spec_augment,
)
from nexcrawl_ml.models.knocknet import (
    KNOCK_CLASSES,
    KnockNet,
    TrainConfig,
    export_onnx,
    train_knocknet,
)


def build_dataset(
    data_dir: Path, config: AudioConfig, augment_factor: int = 3, seed: int = 42
):
    """Load clips, compute log-mels, augment the training set."""
    rng = np.random.default_rng(seed)
    features, labels = [], []

    for class_idx, class_name in enumerate(KNOCK_CLASSES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"  WARNING: {class_dir} missing - skipping {class_name}")
            continue

        wavs = sorted(class_dir.glob("*.wav"))
        print(f"  {class_name:12s} {len(wavs)} clips")

        for wav in wavs:
            y = load_audio(str(wav), config)
            features.append(compute_log_mel(y, config))
            labels.append(class_idx)

            for _ in range(augment_factor):
                y_aug = augment_waveform(y, config, rng=rng)
                mel = compute_log_mel(y_aug, config)
                features.append(spec_augment(mel, rng=rng))
                labels.append(class_idx)

    if not features:
        raise RuntimeError(f"no audio found under {data_dir}")

    return np.stack(features), np.array(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="data/models")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--augment-factor", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    torch.manual_seed(args.seed)
    config = AudioConfig()

    print("[1/4] building dataset...")
    X, y = build_dataset(
        Path(args.data), config, augment_factor=args.augment_factor, seed=args.seed
    )
    print(f"  total {X.shape[0]} samples of shape {X.shape[1:]}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )

    def to_loader(Xa, ya, shuffle):
        ds = TensorDataset(
            torch.tensor(Xa, dtype=torch.float32).unsqueeze(1),
            torch.tensor(ya, dtype=torch.long),
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, True)
    val_loader = to_loader(X_val, y_val, False)

    # Inverse-frequency class weights: staged sets are knock-poor, and an
    # unweighted loss learns that never predicting 'knock' is a fine strategy.
    counts = np.bincount(y_train, minlength=len(KNOCK_CLASSES)).astype(np.float64)
    weights = len(y_train) / (len(KNOCK_CLASSES) * np.maximum(counts, 1))
    print(f"  class weights: {dict(zip(KNOCK_CLASSES, weights.round(2)))}")

    print("\n[2/4] training...")
    model = KnockNet(n_classes=len(KNOCK_CLASSES))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters")

    model, history = train_knocknet(
        model,
        train_loader,
        val_loader,
        TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, device=args.device
        ),
        class_weights=weights,
    )

    print("\n[3/4] evaluating...")
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_val, dtype=torch.float32).unsqueeze(1))
        preds = logits.argmax(1).numpy()

    from sklearn.metrics import classification_report, confusion_matrix

    print(classification_report(y_val, preds, target_names=KNOCK_CLASSES,
                                zero_division=0))

    print("[4/4] exporting ONNX...")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "knocknet.pt")
    onnx_path = export_onnx(
        model, out / "knocknet.onnx",
        n_mels=config.n_mels, n_frames=config.n_frames,
    )

    (out / "knocknet_training_report.json").write_text(
        json.dumps(
            {
                "classes": KNOCK_CLASSES,
                "n_parameters": int(n_params),
                "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
                "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
                "confusion_matrix": confusion_matrix(y_val, preds).tolist(),
                "audio_config": config.__dict__,
            },
            indent=2,
        )
    )
    print(f"  wrote {onnx_path}")


if __name__ == "__main__":
    main()
