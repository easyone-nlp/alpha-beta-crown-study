"""Custom model and data loader for the EMNIST Digits alpha-beta-CROWN run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

INPUT_SIZE = 196
HIDDEN_SIZE = 32
OUTPUT_SIZE = 10
CLASS_NAMES = [str(i) for i in range(OUTPUT_SIZE)]


class TinyEmnistMLP(nn.Module):
    """Small ReLU MLP reused from the Marabou assignment."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


def tiny_emnist_mlp() -> TinyEmnistMLP:
    return TinyEmnistMLP()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _artifact_dir() -> Path:
    return _project_root() / "models" / "emnist_digits"


def emnist_digits_sample(spec: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the selected EMNIST sample and return an L-infinity robustness spec."""

    eps = spec["epsilon"]
    if eps is None:
        raise ValueError("Set specification.epsilon in the config.")

    artifact_dir = _artifact_dir()
    sample_path = artifact_dir / "sample.npy"
    metadata_path = artifact_dir / "metadata.json"
    if not sample_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing EMNIST artifacts in {artifact_dir}. "
            "Run `python src/prepare_emnist_digits.py` first."
        )

    sample = torch.from_numpy(np.load(sample_path).astype(np.float32)).reshape(1, INPUT_SIZE)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    label = int(metadata["sample_prediction"])
    labels = torch.tensor([label], dtype=torch.long)

    data_max = torch.ones(1, INPUT_SIZE, dtype=torch.float32)
    data_min = torch.zeros(1, INPUT_SIZE, dtype=torch.float32)
    eps_temp = torch.tensor(float(eps), dtype=torch.float32).reshape(1, 1)
    return sample, labels, data_max, data_min, eps_temp
