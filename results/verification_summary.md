# Verification Summary

The alpha-beta-CROWN experiment reuses the EMNIST Digits `TinyEmnistMLP` model
from the Marabou assignment.

| Tool | Epsilon | Result | Status | Runtime |
| --- | ---: | --- | --- | ---: |
| alpha-beta-CROWN | 0.02 | verified | safe-incomplete | 17.44398 s |
| alpha-beta-CROWN | 0.2 | falsified | unsafe-pgd | 0.43998 s |
| Marabou | 0.02 | verified | UNSAT | 0.06094 s |
| Marabou | 0.2 | falsified | SAT | 11.11434 s |

For `epsilon=0.02`, alpha-beta-CROWN first tried PGD, found no violation, and
then verified the property with initial CROWN bounds. For `epsilon=0.2`, PGD
found an unsafe adversarial example quickly, matching the Marabou observation
that the larger perturbation admits a counterexample. Marabou runtimes are sums of per-target solve times from the Assignment 3 JSON files.
