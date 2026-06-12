# Working Notes

Use this file to record installation issues, model choices, command outputs, and interpretation notes before writing the final report.

## Problem 1: Repository Exploration

- `complete_verifier/models`:
- `complete_verifier/exp_configs`:
- Difference from Marabou:

## Problem 2: External Model Experiment

- Model: TinyEmnistMLP reused from Assignment 3 / Marabou study.
- Dataset: EMNIST Digits.
- Architecture: 196 -> 32 ReLU -> 10.
- Preprocessing: 28x28 grayscale -> average pool to 14x14 -> flatten.
- Verification property: local L-infinity robustness around one correctly classified sample.
- Epsilon: 0.02 first, then 0.2 for comparison with Marabou.
- Timeout: 30 seconds per alpha-beta-CROWN instance.
- Marabou reference results: epsilon=0.02 UNSAT, epsilon=0.2 SAT with a target digit 8 counterexample.
- alpha-beta-CROWN results: epsilon=0.02 safe-incomplete in 17.44398s; epsilon=0.2 unsafe-pgd in 0.43998s.

