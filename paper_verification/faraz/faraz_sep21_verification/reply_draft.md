Subject: Re: reward components, shared adapters, and external collapse-onset check

Dear Faraz,

Thank you for checking the four earlier decompositions and for pointing out that the reward table never reached you. I have pasted the corrected source-constant grid and an independently recomputed empirical grid below, in the body of this message. I agree with presenting the equation, the audited Stage 2 configuration, and then an observed distribution. The empirical numbers describe what fired in these runs; they cannot replace the source constants, branch conditions, or the post-processing that changes final reward.

I used the twelve six-seed CIFAR-10 raw trajectories (9,600 attempts, 8,030 rows with `formal_success_candidate == true`) for the empirical grid. Each of the 24 top-level `r_*` fields was numeric on all 8,030 selected rows. Minima, medians, and maxima below are rounded for display; nonzero counts use the unrounded logged values.

Corrected Stage 2 source grid (NNGPT base commit c91714dbe7dad1d02a9080243945bbf8e8ec9300):

Component / raw configuration                         | Effective or logged Stage 2 behavior
Dense: scale 0.50, inner clip [0.02, 0.35]            | 0.50 * clip(0.03 + 0.28*target + 0.04*max(0, train_acc-0.50), 0.02, 0.35)
Formal success: +0.08; target match: +0.20             | r_formal_success_signal = 0.28 only when formal success and target_structure_match is not False; otherwise the applicable portion
Previous / best group: scales 0.20 / 0.20              | Previous threshold +0.003; best threshold +0.0015; baseline-dependent clipped gaps, with a 0.20 global blend where applicable
Previous / best backbone group: scales 0.25 / 0.25    | Same respective thresholds; requires eligible backbone baseline
Goal-best refresh: +0.08, scale 0.70                 | +0.056 only on refresh
Goal-tag match: 0.12, scale 0.85                     | 0.102 * goal_tag_hit_rate; observed zero in this cohort
Generalization: tolerance 0.02, factor -2.0          | clip(-2*max(0, train_acc-test_acc-0.02), -0.20, 0)
No progress: -0.06, scale 0.50                       | -0.03 only on the applicable non-improving branch
Family repeat: -0.05, scale 1.10                     | -0.055 when its repeat condition fires
Plain fuse: -0.10 / -0.28, scale 1.10                | -0.11 / -0.308 on the respective branches
Structure group/archive: Stage 2 scale 1.40         | Conditional, capped aggregates; train-set novelty is not added a second time
Descriptor / CNN / block diversity                   | Aggregates of gated novelty and repeat branches, not single fixed bonuses; source novelty constants include descriptor +0.03/+0.02/+0.08, CNN +0.07/+0.05/+0.12, and block +0.06/+0.08
Repeated-block cap: 2.0                             | Conditional upper cap on the total, not a +0.20 component
History context                                      | `_history_context_reward` returns 0.0 at this commit

The Stage 2 primary sum also includes the logged `r_batch_elite`, `r_target_structure_penalty`, `r_template_penalty`, `r_descriptor_diversity`, `r_cnn_diversity`, `r_block_diversity`, `r_structure_group`, `r_structure_archive`, the group and accuracy terms above, and the applicable negative terms. `r_goal_match` is the additional tiebreak term. The TuneRL base sum is clipped to [-2, 2], with conditional caps, local competition, and executability/target gates around it. TuneRLSft then applies extraction and dual-backbone contract deltas, format/hygiene caps, trainability clamps, and compactness penalties. Consequently [-2, 2] is not a universal bound on the final logged reward. The earlier short constant list did not describe every contributing branch; in particular, it should not be described as a complete equation for final reward.

Empirical grid for the 8,030 formal-success candidates (nonzero = count / 8,030):

Field                          | Min     | Median  | Max     | Nonzero
r_batch_elite                  | 0       | 0       | 0.0400  | 2256 (28.09%)
r_best_backbone_group         | -0.3000 | 0       | 0.3000  | 4562 (56.81%)
r_best_group                  | -0.2400 | 0.0017  | 0.1822  | 6079 (75.70%)
r_block_diversity             | -0.1300 | -0.0200 | 0.1400  | 5800 (72.23%)
r_cnn_diversity               | -0.3500 | -0.1100 | 0.3200  | 7022 (87.45%)
r_dense                       | 0       | 0.0915  | 0.0977  | 7239 (90.15%)
r_descriptor_diversity        | -0.1550 | -0.1050 | 0.1900  | 7904 (98.43%)
r_formal_success_signal       | 0       | 0.2800  | 0.2800  | 7239 (90.15%)
r_generalization              | -0.2000 | 0       | 0       | 3288 (40.95%)
r_goal_best                   | 0       | 0       | 0.0560  | 244  (3.04%)
r_goal_match                  | 0       | 0       | 0       | 0    (0.00%)
r_history_context             | 0       | 0       | 0       | 0    (0.00%)
r_length_compactness          | -0.0347 | 0       | 0       | 248  (3.09%)
r_no_progress_penalty         | -0.0300 | 0       | 0       | 2839 (35.35%)
r_plain_fuse_penalty          | -0.3080 | -0.1100 | 0       | 5637 (70.20%)
r_prev_backbone_group         | -0.4500 | 0       | 0.4500  | 4485 (55.85%)
r_prev_group                  | -0.3600 | 0.0009  | 0.1867  | 6079 (75.70%)
r_repeat_family               | -0.0550 | -0.0550 | 0       | 5191 (64.65%)
r_repeated_line_penalty       | -0.1500 | 0       | 0       | 105  (1.31%)
r_structure_archive           | 0       | 0       | 0.0980  | 3262 (40.62%)
r_structure_group             | 0       | 0       | 0.1960  | 3262 (40.62%)
r_target_structure_penalty    | -1.0000 | 0       | 0       | 1324 (16.49%)
r_template_penalty            | -0.0500 | 0       | 0       | 815  (10.15%)
r_trainset_novelty            | 0       | 0.0200  | 0.0400  | 6210 (77.33%)

The last field, `r_trainset_novelty`, is logged provenance and is not a separate addition to the Stage 2 primary reward. The logged `r_*` values are already after their applicable stage scaling; for example, -0.055 is -0.05*1.10. `r_goal_match` is present in all 8,030 rows, and all have `prompt_goal_tags = None` and zero goal-tag hit rate, so its observed zero is explained by the recorded inputs rather than a missing field. `r_history_context` is likewise present, and the fixed-commit function returns 0.0 unconditionally. It is implementation-inert at that commit. Some run configurations declare a dirty working tree, so the fixed commit and archived logs cannot establish byte-for-byte identity with an unrecorded local diff. I can check your version line by line if you paste it; I did not have your numeric grid in the 21 September message.

On the adapter question, all six one-pattern CIFAR-10 RL configs point to the same DeepSeek A9 SFT adapter path, and all six four-pattern configs point to the same DeepSeek A18 path. The three SFT-only generation configs per condition also point to the corresponding path:

One-pattern: /home/s471802/nn-gpt/out/nngpt/llm/20260601_1pattern_three_model_sft_v3_cifar10_dscoder7b/epoch_sft/A9/deepseek-ai/deepseek-coder-6.7b-instruct
Four-pattern: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct

It is therefore correct to describe one shared starting adapter per condition across all six RL seeds. The SFT-only samples characterize output from that shared adapter, so a descriptive condition-level pre/post comparison can include all twelve RL trajectories. They do not supply a separate SFT sample matched to each of the other three RL seeds, nor does this comparison isolate a causal RL effect; the SFT-only evaluation uses held-out test, whereas formal-success gating in RL uses reward-eval. Four-pattern seed 114 also has no eligible formal-success family denominator in its final 100 attempts.

I reran the 80% family-onset analysis on the 15 runs you named, keeping the same eight non-overlapping 100-attempt windows, formal-success family denominator of at least 20, and first-threshold-crossing definition as the primary six-seed calculation. The results are:

Cohort / seed                         | First >=80% family window | Final 100 family top-1 >=80%?
Qwen CIFAR-10 42 / 123 / 777         | 1-100 / 1-100 / 1-100     | yes / yes / no (seed 777)
CIFAR-100 one-pattern 42 / 123 / 777 | 1-100 / 1-100 / 1-100     | yes / yes / yes
CIFAR-100 four-pattern 42 / 123 / 777| none / none / none        | no / no / no
No-diversity 42 / 123 / 777          | none / 101-200 / none     | no / yes / no
No-repeat 42 / 123 / 777             | 401-500 / 101-200 / none  | yes / yes / no

Using an 80% cutoff for both onset and final state gives eight onset-positive/terminally-collapsed runs, one onset-positive/terminally-diverse run (Qwen seed 777), six onset-negative/terminally-diverse runs, and no missed terminal collapses in these 15 archived trajectories. This is useful out-of-cohort evidence, but it is not a perfect classifier and should not yet be called a validated early-stopping rule. The choice of window matters: with 100-attempt windows advancing by 25, CIFAR-100 four-pattern seed 42 briefly crosses 80% in attempts 51-150, then finishes at 30.77%; the fixed non-overlapping windows do not flag it.

I also rechecked the primary-cohort figures. Nine of twelve runs ever cross the non-overlapping 80% threshold; five do so by attempt 100, seven by 200, and all nine by 400. Only eight of twelve finish above 80% in the final 100 attempts. One-pattern seed 777 crossed early but finished at 55.56%. The 458 GPU-hour number can be reproduced as 458.49 estimated avoidable allocated GPU-hours by prorating each onset-positive job's whole-job GPU hours by (800 - onset-window end)/800. It is a counterfactual under a constant GPU-hours-per-attempt assumption, not observed savings; its denominator is 907.67 allocated GPU-hours, so the modeled fraction is 50.51%. Since an early stop would also have stopped seed 777 before its later diversification, this estimate needs that qualification in the paper.

For the requested per-run accounting, the retained Slurm records support the following allocated GPU-hours (elapsed seconds multiplied by allocated GPU count):

Cohort / seed                    | Job(s)               | GPU-hours
Qwen CIFAR-10 42                | 2791927              | 43.47
Qwen CIFAR-10 123               | 2791928              | 91.19
Qwen CIFAR-10 777               | 2791929 + 2793568    | 100.30 (timeout plus continuation)
CIFAR-100 one-pattern 42        | 2866647              | 71.54
CIFAR-100 one-pattern 123 rerun | 2943044              | 48.64
CIFAR-100 one-pattern 777       | 2944149              | 90.52
CIFAR-100 four-pattern 42       | 2960907 + 2968439 + 3043789, including requeues | 90.66
CIFAR-100 four-pattern 123      | 2960909 + 2968441 + 3043791, including requeues | 78.26
CIFAR-100 four-pattern 777      | 2960911 + 2968443 + 3043793, including requeues | 74.34
No-diversity 42                 | 2697779              | 67.86
No-diversity 123                | 2751655              | 69.34 (retained result, Slurm FAILED after manual stop)
No-diversity 777                | 2760934              | 79.70
No-repeat 42                    | 2697849              | 126.07
No-repeat 123                   | 2755237              | 65.00
No-repeat 777                   | 2771489              | 85.70

The original local snapshots supported 939.3378 GPU-hours for twelve runs. I have now recovered the three CIFAR-100 four-pattern run histories from Julia2 using `sacct --duplicates`: 243.2622 additional GPU-hours, including the preempted/requeued allocations that an ordinary final-state query hides. Counting each parent allocation once and excluding `.batch`/`.extern` steps gives **1,182.60 allocated GPU-hours for these fifteen retained run IDs**. This is a retained-run scope, not a full accounting of every attempted job in the project.

Applying the same proportional-per-attempt model to the nine onset-positive external runs gives an estimated 537.99 GPU-hours of avoidable tail allocation, or 45.49% of this 1,182.60-hour retained-run total. This estimate uses whole-job times, includes Qwen seed 777 although it later diversified, assumes GPU-hours per attempt are constant, and is not observed savings. The added CIFAR-100 four-pattern runs have no onset under the fixed non-overlapping-window definition, so their recovered hours change the denominator but not this numerator.

The SFT training pool's family frequency distribution is not in the delivered raw candidate logs. Those logs expose some train-pool membership flags, but not each family's count in the training pool. The A9 and A18 server archives contain per-candidate synthesis artifacts, but I have not established which source rows and stages were actually included in the SFT input pool. I cannot responsibly provide the three-stage distribution without that training manifest/data and the same family-signature extraction.

I also cannot state that every started attempt is included in the published packages. The run archive records started, superseded or excluded partial attempts beyond the CIFAR-100 one-pattern seed-123 run, including ablation attempts stopped after 144, 168, and 88 samples, and other replacement attempts. The retained-result GPU grid above includes the Qwen seed-777 timeout and continuation and the manually stopped but retained no-diversity seed-123 result; it does not purport to be an all-attempt cost ledger. We should state the retained-cohort inclusion rule explicitly and build a separate all-attempt ledger before making an exhaustive reporting claim.

Finally, I agree that the current repository URL and commit metadata can reveal author identity. We should leave the timing of code availability to Dr. Ignatov. A mirror that preserves the exact Git commit object also preserves that object's author/committer metadata; changing those fields necessarily changes the commit hash. If submission-time anonymity is chosen, the workable option is a reviewed anonymous source snapshot or a new anonymized commit with the same audited source tree and a disclosed mapping to c91714db, after removing identifying README and repository metadata. I have not created or published a mirror.

Best regards,
Xi
