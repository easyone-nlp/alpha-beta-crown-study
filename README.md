# alpha-beta-crown-study

Assignment 4 project for Reliable and Trustworthy Artificial Intelligence.

This repository documents and reproduces a small alpha-beta-CROWN verification
experiment on the EMNIST Digits model used in the earlier Marabou assignment.

## Repository Layout

- `configs/`: YAML configuration files for alpha-beta-CROWN.
- `src/`: helper scripts for model preparation or result parsing.
- `models/`: generated or downloaded model artifacts.
- `data/`: small input samples or metadata used by the experiment.
- `results/`: verifier logs and summarized verification results.
- `test.py`: entry point demonstrating how to run alpha-beta-CROWN on the chosen model.
- `requirements.txt`: Python dependencies used by helper scripts.
- `report.pdf`: final 1-2 page report.

## Setup

Clone alpha-beta-CROWN next to the project files:

```bash
git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
cd alpha-beta-CROWN
conda env create -f complete_verifier/environment.yaml --name alpha-beta-crown
conda activate alpha-beta-crown
```

Verify the installation with one official tutorial config:

```bash
cd complete_verifier
python abcrown.py --config exp_configs/tutorial_examples/cifar_resnet_2b.yaml
```

## Run This Project

Prepare the EMNIST Digits model and selected sample:

```bash
python src/prepare_emnist_digits.py --download
```

Run alpha-beta-CROWN on the prepared model:

```bash
python test.py --abcrown-dir ./alpha-beta-CROWN --config configs/emnist_digits_tiny_mlp.yaml
```

The script writes logs to `results/`.

## Problem 1 Exploration

The alpha-beta-CROWN repository is organized around model artifacts and YAML
experiment configurations. `complete_verifier/models` includes MNIST/CIFAR SDP
models, ERAN `.pth` models, OVAL CIFAR models, Marabou CIFAR10 models, toy
MNIST MLPs, CIFAR-10 ResNets, custom operation examples, non-ReLU examples, and
VNN-COMP benchmark placeholders. `complete_verifier/exp_configs` contains
configuration families such as `beta_crown`, `GCP-CROWN`, `BICCOS`,
`bab_attack`, `tutorial_examples`, and VNN-COMP year folders.

Compared with Marabou, alpha-beta-CROWN is more YAML-driven: the config links
model loading, dataset loading, perturbation size, attack, solver, and timeout.
Marabou scripts more directly manipulate input bounds and output inequalities.

## Problem 2 Experiment

This project reuses the Assignment 3 Marabou model to make the comparison
direct:

- Dataset: EMNIST Digits.
- Preprocessing: 28x28 grayscale images are average-pooled to 14x14 and
  flattened to 196 input values in `[0, 1]`.
- Model: `TinyEmnistMLP`, a ReLU MLP with architecture `196 -> 32 -> 10`.
- Marabou format: `.nnet` in the Assignment 3 repository.
- alpha-beta-CROWN format: PyTorch `.pt`, with ONNX also exported for reference.
- Property: local L-infinity robustness around one correctly classified test
  sample.

The Marabou assignment recorded `UNSAT` for `epsilon=0.02` and a `SAT`
counterexample for `epsilon=0.2`. This repository runs alpha-beta-CROWN on the
same model family, sample, and perturbation sizes for comparison.

## Results

| Epsilon | alpha-beta-CROWN result | Runtime | Marabou reference |
| ---: | --- | ---: | --- |
| 0.02 | verified (`safe-incomplete`) | 17.44398 s | verified (`UNSAT`) |
| 0.2 | falsified (`unsafe-pgd`) | 0.43998 s | falsified (`SAT`) |

See `results/verification_summary.md` and the raw logs in `results/`.

## Assignment Checklist

- [x] Explore `complete_verifier/models` and summarize available models.
- [x] Explore `complete_verifier/exp_configs` and summarize YAML configurations.
- [x] Choose an external model and dataset.
- [x] Export or save the model in a supported PyTorch/ONNX format.
- [x] Write an alpha-beta-CROWN YAML config.
- [x] Run verification and record verified/falsified/timeout outcomes.
- [x] Write `report.pdf`.
- [ ] Commit changes incrementally and push to GitHub.

