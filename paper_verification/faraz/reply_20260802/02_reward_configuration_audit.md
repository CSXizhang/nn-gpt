# Main 5-epoch full_reward configuration audit

## Provenance

- Main-run source commit: `c91714dbe7dad1d02a9080243945bbf8e8ec9300`.
- Archived run variant: `full_reward`.
- Stage: `stage2_formal_explore`.
- Dataset/split: CIFAR-10 `trainvaltest` = train 45k, reward-eval 5k, held-out test 10k.
- Formal evaluation epochs: 5.
- Generation budget: 8 generations x 100 optimizer steps = 800 candidates.
- KL coefficient: 0.005.

## Source-verified Stage 2 values

| Component | Source value |
|---|---:|
| previous-group scale | 0.20 |
| best-group scale | 0.20 |
| backbone previous-group scale | 0.25 |
| backbone best-group scale | 0.25 |
| global-baseline blend | 0.20 |
| formal-success signal | +0.08 |
| target-structure-match bonus | +0.20 |
| dense scale | 0.50 |
| no-progress scale | 0.50 |
| no-progress base penalty | -0.06 |
| generalization gap tolerance | 0.02 |
| generalization scale | -2.0 x excess gap |
| generalization cap | -0.20 |
| repeated-block reward cap | 2.0 |

Dense term:

`0.50 * clip(0.03 + 0.28 * T5 + 0.04 * max(0, train_acc - 0.5), 0.02, 0.35)`

Generalization term:

`max(-2.0 * max(0, train_acc - test_acc - 0.02), -0.20)`

The main scalar reward is clipped to `[-2, 2]` inside the base reward function. The SFT wrapper then applies stricter post-processing caps:

- core format violation: at most `-3.0`;
- severe hygiene violation: at most `-2.0`;
- missing dual-backbone requirement: at most `-3.5`.

Therefore the final raw reward emitted by the SFT wrapper is not globally bounded to `[-2, 2]`.

## Manuscript values requiring correction

The current `tab:reward_components` does not match the main-run commit in several places:

- Stage 2 previous/best group values should be `0.20 / 0.20`, not `0.70`.
- Backbone previous/best values should be `0.25 / 0.25`, not `0.95`.
- Repeated-block reward cap in this commit is `2.0`, not `0.20`.
- A statement that the final reward always remains in `[-2,2]` is incorrect because wrapper caps are applied after the base clip.

The attached excerpts are copied from the exact commit, not from the current working tree.
