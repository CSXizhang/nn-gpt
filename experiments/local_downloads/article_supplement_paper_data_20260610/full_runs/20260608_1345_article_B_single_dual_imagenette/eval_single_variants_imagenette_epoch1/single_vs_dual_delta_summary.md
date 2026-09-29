# Single-vs-Dual Imagenette Delta Summary

- selected dual candidates: 36
- mutation success: 72 / 72
- single eval records: 72 / 72
- paired dual records: 36 / 36
- missing single eval: 0
- eval failures: 0

## Overall

| Metric | Mean | Median | Std | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| dual_acc | 99.39% | 99.36% | 0.20% | 98.96% | 99.82% |
| a_only_acc | 99.24% | 99.31% | 0.40% | 97.22% | 99.80% |
| b_only_acc | 94.37% | 97.85% | 11.82% | 52.87% | 99.62% |
| best_single_acc | 99.31% | 99.31% | 0.21% | 98.98% | 99.80% |
| dual_minus_a_only | 0.14% | 0.09% | 0.29% | -0.18% | 1.73% |
| dual_minus_b_only | 5.02% | 1.44% | 11.82% | -0.48% | 46.55% |
| dual_minus_best_single | 0.08% | 0.08% | 0.15% | -0.48% | 0.31% |

dual beats best single: 28 / 36
dual ties or beats best single: 30 / 36

## By Setting

| Setting | N | Dual Mean | Best Single Mean | Dual - Best Single Mean | Beats Best Single |
| --- | ---: | ---: | ---: | ---: | ---: |
| dscoder_cifar10 | 4 | 99.51% | 99.41% | 0.10% | 4 / 4 |
| dscoder_cifar100 | 4 | 99.07% | 99.18% | -0.11% | 2 / 4 |
| dscoder_imagenette | 4 | 99.36% | 99.29% | 0.06% | 3 / 4 |
| olympic_cifar10 | 4 | 99.33% | 99.33% | 0.00% | 1 / 4 |
| olympic_cifar100 | 4 | 99.29% | 99.03% | 0.26% | 4 / 4 |
| olympic_imagenette | 4 | 99.41% | 99.31% | 0.11% | 3 / 4 |
| qwen_cifar10 | 4 | 99.80% | 99.75% | 0.04% | 3 / 4 |
| qwen_cifar100 | 4 | 99.48% | 99.34% | 0.14% | 4 / 4 |
| qwen_imagenette | 4 | 99.25% | 99.11% | 0.14% | 4 / 4 |

## Per Candidate

| Candidate | Setting | Dual | A-only | B-only | Best Single | Dual - Best Single |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dscoder_cifar10-0002 | dscoder_cifar10 | 99.62% | 99.36% | 96.10% | 99.36% | 0.25% |
| dscoder_cifar10-0026 | dscoder_cifar10 | 99.49% | 99.46% | 92.56% | 99.46% | 0.03% |
| dscoder_cifar10-0001 | dscoder_cifar10 | 99.46% | 99.39% | 97.48% | 99.39% | 0.08% |
| dscoder_cifar10-0017 | dscoder_cifar10 | 99.46% | 99.44% | 97.43% | 99.44% | 0.03% |
| dscoder_cifar100-0008 | dscoder_cifar100 | 99.18% | 98.98% | 97.94% | 98.98% | 0.20% |
| dscoder_cifar100-0025 | dscoder_cifar100 | 99.13% | 99.11% | 97.78% | 99.11% | 0.03% |
| dscoder_cifar100-0028 | dscoder_cifar100 | 99.01% | 99.18% | 96.18% | 99.18% | -0.18% |
| dscoder_cifar100-0009 | dscoder_cifar100 | 98.96% | 97.22% | 99.44% | 99.44% | -0.48% |
| dscoder_imagenette-0016 | dscoder_imagenette | 99.36% | 99.21% | 98.47% | 99.21% | 0.15% |
| dscoder_imagenette-0021 | dscoder_imagenette | 99.36% | 99.31% | 98.68% | 99.31% | 0.05% |
| dscoder_imagenette-0028 | dscoder_imagenette | 99.36% | 99.26% | 98.93% | 99.26% | 0.10% |
| dscoder_imagenette-0022 | dscoder_imagenette | 99.34% | 99.39% | 98.88% | 99.39% | -0.05% |
| olympic_cifar10-0014 | olympic_cifar10 | 99.36% | 99.31% | 97.94% | 99.31% | 0.05% |
| olympic_cifar10-0009 | olympic_cifar10 | 99.34% | 99.34% | 97.81% | 99.34% | 0.00% |
| olympic_cifar10-0016 | olympic_cifar10 | 99.31% | 99.34% | 98.01% | 99.34% | -0.03% |
| olympic_cifar10-0021 | olympic_cifar10 | 99.31% | 99.34% | 97.63% | 99.34% | -0.03% |
| olympic_cifar100-0020 | olympic_cifar100 | 99.31% | 99.08% | 97.83% | 99.08% | 0.23% |
| olympic_cifar100-0013 | olympic_cifar100 | 99.29% | 98.98% | 98.01% | 98.98% | 0.31% |
| olympic_cifar100-0023 | olympic_cifar100 | 99.29% | 99.03% | 98.11% | 99.03% | 0.25% |
| olympic_cifar100-0002 | olympic_cifar100 | 99.26% | 99.01% | 97.81% | 99.01% | 0.25% |
| olympic_imagenette-0006 | olympic_imagenette | 99.52% | 99.36% | 97.86% | 99.36% | 0.15% |
| olympic_imagenette-0005 | olympic_imagenette | 99.41% | 99.24% | 52.87% | 99.24% | 0.18% |
| olympic_imagenette-0027 | olympic_imagenette | 99.39% | 99.29% | 58.78% | 99.29% | 0.10% |
| olympic_imagenette-0000 | olympic_imagenette | 99.34% | 99.34% | 54.78% | 99.34% | 0.00% |
| qwen_cifar10-0026 | qwen_cifar10 | 99.82% | 99.80% | 99.36% | 99.80% | 0.03% |
| qwen_cifar10-0004 | qwen_cifar10 | 99.80% | 99.69% | 99.57% | 99.69% | 0.10% |
| qwen_cifar10-0025 | qwen_cifar10 | 99.80% | 99.72% | 99.62% | 99.72% | 0.08% |
| qwen_cifar10-0010 | qwen_cifar10 | 99.77% | 99.80% | 99.59% | 99.80% | -0.03% |
| qwen_cifar100-0029 | qwen_cifar100 | 99.52% | 99.26% | 99.18% | 99.26% | 0.25% |
| qwen_cifar100-0004 | qwen_cifar100 | 99.49% | 99.44% | 98.83% | 99.44% | 0.05% |
| qwen_cifar100-0010 | qwen_cifar100 | 99.46% | 99.34% | 99.26% | 99.34% | 0.13% |
| qwen_cifar100-0002 | qwen_cifar100 | 99.44% | 99.31% | 96.15% | 99.31% | 0.13% |
| qwen_imagenette-0000 | qwen_imagenette | 99.29% | 98.98% | 97.10% | 98.98% | 0.31% |
| qwen_imagenette-0019 | qwen_imagenette | 99.29% | 99.08% | 97.10% | 99.08% | 0.20% |
| qwen_imagenette-0014 | qwen_imagenette | 99.21% | 99.18% | 97.02% | 99.18% | 0.03% |
| qwen_imagenette-0015 | qwen_imagenette | 99.21% | 99.18% | 97.17% | 99.18% | 0.03% |

## Inference

- dual_minus_best_single bootstrap 95% CI: 0.03% to 0.13%
- Wilcoxon signed-rank dual vs best single: W=84.000, p=0.000106559, nonzero pairs=34
