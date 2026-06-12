"""Run alpha-beta-CROWN on this project's verification config."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--abcrown-dir",
        default="alpha-beta-CROWN",
        help="Path to the cloned alpha-beta-CROWN repository.",
    )
    parser.add_argument(
        "--config",
        default="configs/emnist_digits_tiny_mlp.yaml",
        help="Project YAML config to pass to abcrown.py.",
    )
    parser.add_argument(
        "--log",
        default="results/abcrown_run.log",
        help="Path where stdout/stderr from alpha-beta-CROWN will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    abcrown_dir = (project_root / args.abcrown_dir).resolve()
    verifier_dir = abcrown_dir / "complete_verifier"
    abcrown_py = verifier_dir / "abcrown.py"
    config_path = (project_root / args.config).resolve()
    log_path = (project_root / args.log).resolve()

    if not abcrown_py.exists():
        raise FileNotFoundError(
            f"Could not find {abcrown_py}. Clone alpha-beta-CROWN first."
        )
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config file: {config_path}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["python", str(abcrown_py), "--config", str(config_path)]

    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=verifier_dir,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    print(f"alpha-beta-CROWN exited with code {result.returncode}")
    print(f"Log saved to {log_path}")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

