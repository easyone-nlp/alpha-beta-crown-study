# alpha-beta-crown-study

Assignment 4 project for Reliable and Trustworthy Artificial Intelligence.

This repository documents and reproduces a small alpha-beta-CROWN verification experiment on an external model.

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

After preparing the external model and config, run:

```bash
python test.py --abcrown-dir ./alpha-beta-CROWN --config configs/mnist_tiny.yaml
```

The script writes logs to `results/`.

## Assignment Checklist

- [ ] Explore `complete_verifier/models` and summarize available models.
- [ ] Explore `complete_verifier/exp_configs` and summarize YAML configurations.
- [ ] Choose an external model and dataset.
- [ ] Export or save the model in a supported PyTorch/ONNX format.
- [ ] Write an alpha-beta-CROWN YAML config.
- [ ] Run verification and record verified/falsified/timeout outcomes.
- [ ] Write `report.pdf`.
- [ ] Commit changes incrementally and push to GitHub.

