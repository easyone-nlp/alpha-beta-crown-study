# Working Notes

Use this file to record installation issues, model choices, command outputs, and interpretation notes before writing the final report.

## Problem 1: Repository Exploration

- `complete_verifier/models`: contains bundled benchmark models grouped by source or experiment family. I found SDP-style MNIST/CIFAR `.model` files, ERAN MNIST and CIFAR PyTorch `.pth` models, OVAL CIFAR models, Marabou CIFAR10 models, toy MNIST MLP models, CIFAR-10 ResNet models, non-ReLU/custom-op examples, L2-norm examples, and VNN-COMP benchmark placeholders.
- `complete_verifier/exp_configs`: contains YAML files that define complete verification experiments. The major groups include `beta_crown`, `GCP-CROWN`, `BICCOS`, `bab_attack`, `tutorial_examples`, and VNN-COMP year folders such as `vnncomp21`, `vnncomp22`, `vnncomp23`, `vnncomp24`, and `vnncomp25`.
- The YAML files specify the model path or loader, dataset loader, norm and epsilon, solver options, attack settings, branch-and-bound timeout, and sometimes VNN-LIB/CSV benchmark inputs. This made alpha-beta-CROWN feel more configuration-driven than Marabou.
- Difference from Marabou: Marabou directly loads a network format such as `.nnet` or ONNX and the script usually constructs explicit variable bounds and output inequalities. alpha-beta-CROWN uses YAML to connect model, data, perturbation, attack, and solver settings, then internally constructs the robustness specifications and runs bound propagation plus branch-and-bound.

## Problem 2: External Model Experiment

- Model: TinyEmnistMLP reused from Assignment 3 / Marabou study.
- Dataset: EMNIST Digits.
- Architecture: 196 -> 32 ReLU -> 10.
- Preprocessing: 28x28 grayscale -> average pool to 14x14 -> flatten.
- Verification property: local L-infinity robustness around one correctly classified sample.
- Epsilon: 0.02 first, then 0.2 for comparison with Marabou.
- Timeout: 30 seconds per alpha-beta-CROWN instance.
- Marabou reference results: epsilon=0.02 UNSAT in 0.06094s total per-target runtime; epsilon=0.2 SAT in 11.11434s total per-target runtime with a target digit 8 counterexample.
- alpha-beta-CROWN results: epsilon=0.02 safe-incomplete in 17.44398s; epsilon=0.2 unsafe-pgd in 0.43998s.

