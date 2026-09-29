# 2026-06-08 Article 补充实验方案

## 目的

这篇文档记录答辩结束后，为 article 补充结果而确定的实验方案。后续切换 session 时，以这篇本地文档为准，不依赖聊天记忆。

要补的证据缺口有四个：

1. 1 epoch 的 backbone formal eval 是否能预测 10 epoch 表现。
2. 第二个 backbone 是否带来可测量的 paired accuracy 增益。
3. reward 组件中哪些项影响成功率、准确率和结构 collapse。
4. 1-pattern RL 行为是否对 seed 稳健。

## 已确认数据源

Julia2 数据库：

```text
/home/s471802/nn-gpt/db/ab.nn.db
/home/s471802/nn-dataset/db/ab.nn.db -> /home/s471802/nn-gpt/db/ab.nn.db
```

DB 中已确认的 backbone 候选 prefix：

| Prefix | NN 数量 | CIFAR-10 epoch1 | CIFAR-10 epoch10 | CIFAR-100 epoch1 | Imagenette epoch1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rl-bb-test1%` | 1988 | 1916 | 0 | 18 | 30 |
| `rl-bb-struct1%` | 1225 | 1225 | 0 | 0 | 0 |
| `rl-bb%` | 4535 | 4463 | 0 | 18 | 30 |

当前可用的 3x3 heldout-test 横评根目录：

```text
/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_1455_gen30min32_formal100_dsqwenolympic
```

这个 3x3 是：

```text
dscoder / qwen / olympic
x cifar10 / cifar100 / imagenette
```

它包含 9 组 candidate，每组 30 个生成模型，并且已经完成 CIFAR-10、CIFAR-100、Imagenette 三个数据集的 heldout-test eval。

注意：这不是更早的 DeepSeek/Qwen 6-setting 横评。除非后续明确有新结果替代，article 的 single-vs-dual 选样默认使用 `20260607_1455_gen30min32_formal100_dsqwenolympic`。

## 实验 A：1 Epoch vs 10 Epoch

### 问题

当前 RL 和 baseline comparison 用的是 frozen-backbone 1 epoch formal eval。这个 1 epoch 结果能不能预测 10 epoch 表现？

### 范围

只用 nn-dataset / DB 里已有的历史 backbone-generated models。不要让 LLM 为这个实验重新生成模型。

只使用 SFT/RL 流程中用过的 backbone prefix：

```text
rl-bb-test1%
rl-bb-struct1%
```

不要混入任意历史普通 NN 模型。

### 主数据集

主实验使用 CIFAR-10。

原因：DB 里 CIFAR-10 的 backbone epoch1 记录足够多：

```text
rl-bb-test1%:   1916 条 CIFAR-10 epoch1
rl-bb-struct1%: 1225 条 CIFAR-10 epoch1
```

这些 prefix 当前没有 epoch10 记录。因此实验不是重新生成模型，而是复用历史模型代码和 epoch1 分数，只补跑缺失的 10 epoch eval。

### 候选抽样

从 DB 中取候选，条件为：

```sql
task = 'img-classification'
dataset = 'cifar-10'
epoch = 1
nn.name LIKE 'rl-bb-test1%' OR nn.name LIKE 'rl-bb-struct1%'
```

如果同一个 NN 有重复 stat row，使用已有的最好 epoch1 accuracy。

按 epoch1 accuracy 分层抽样，总共 300 个：

| 分层 | 规则 | 数量 |
| --- | --- | ---: |
| 高 | epoch1 accuracy 前 1/3 | 100 |
| 中 | epoch1 accuracy 中间 1/3 | 100 |
| 低 | epoch1 accuracy 后 1/3 | 100 |

候选 manifest 用固定 seed `42` 生成，保证可复现。

提交 eval 前必须先写 manifest：

```text
<run_root>/selection/backbone_epoch1_epoch10_manifest.json
```

每行至少包含：

```text
nn_name
prefix_group
dataset
epoch1_accuracy
stratum
source_db
code_hash if available
```

### 评估

对选出的 300 个 CIFAR-10 模型跑 10 epoch formal eval。

评估设置：

```text
task=img-classification
dataset=cifar-10
epochs=10
使用当前 formal backbone eval 相同的 evaluator / split protocol
```

不要生成代码，不要修复代码。模型如果 build / forward / train 失败，就作为这个 10 epoch 点的 evaluation failure 记录。

### Imagenette 补充

DB 当前在 `rl-bb-test1%` 下正好有 30 条 Imagenette epoch1 记录。可以把这 30 个全部补跑 Imagenette 10 epoch，作为小规模 secondary check。

CIFAR-100 暂时不作为实验 A 主数据集，因为当前 `rl-bb-test1%` 只有 18 条 CIFAR-100 epoch1，样本太少。

### 指标

CIFAR-10 主实验报告：

```text
Pearson(acc1, acc10)
Spearman(acc1, acc10)
Kendall tau，如果方便
top-10 / top-20 / top-50 overlap
acc10 - acc1 的分层分布
10 epoch failure rate by stratum
```

图：

```text
acc1 vs acc10 scatter，加 y=x 参考线
按 stratum 分组的 delta boxplot
top-k overlap bar chart
```

Article 解释口径：

- 如果 Spearman 高，可以说 1 epoch 是合理筛选 proxy。
- 如果 Spearman 中等或低，只能说 1 epoch 衡量 early trainability / executability，不能当 final architecture ranking。

## 实验 B：Single vs Dual Backbone

### 问题

第二个 backbone 是否比最好的 single-backbone variant 带来 paired accuracy 增益？

### 来源

使用已完成的 3x3 Imagenette heldout-test eval：

```text
/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_1455_gen30min32_formal100_dsqwenolympic/eval_heldout_test/imagenette/generation_samples.jsonl
```

### 候选选择

至少选 30 个 dual-backbone 候选。默认方案选 36 个：

```text
9 settings x 每个 setting 按 Imagenette heldout-test accuracy 取 top 4
```

9 个 setting：

```text
dscoder_cifar10
dscoder_cifar100
dscoder_imagenette
qwen_cifar10
qwen_cifar100
qwen_imagenette
olympic_cifar10
olympic_cifar100
olympic_imagenette
```

每个 setting 的选择规则：

1. 只保留 formal-success 且 Imagenette heldout-test accuracy 是数字的候选。
2. 按 Imagenette accuracy 从高到低排序。
3. 取 top 4。
4. 主协议不按 signature 去重；重复结构本身就是生成器行为的一部分。
5. 同时记录 signature 和 backbone-pair duplicate，后续如果需要可做 deduplicated sensitivity analysis。

写方案时已观察到的每个 setting 当前最高候选：

| Setting | Candidate | Imagenette acc | Backbone pair |
| --- | --- | ---: | --- |
| `dscoder_cifar10` | `dscoder_cifar10-0002` | 0.9962 | `efficientnet_b3 + resnet50` |
| `dscoder_cifar100` | `dscoder_cifar100-0008` | 0.9918 | `regnet_x_3_2gf + shufflenet_v2_x2_0` |
| `dscoder_imagenette` | `dscoder_imagenette-0016` | 0.9936 | `regnet_x_1_6gf + regnet_y_3_2gf` |
| `qwen_cifar10` | `qwen_cifar10-0026` | 0.9982 | `regnet_y_16gf + resnet50` |
| `qwen_cifar100` | `qwen_cifar100-0029` | 0.9952 | `regnet_y_3_2gf + resnext50_32x4d` |
| `qwen_imagenette` | `qwen_imagenette-0000` | 0.9929 | `mobilenet_v2 + regnet_x_3_2gf` |
| `olympic_cifar10` | `olympic_cifar10-0014` | 0.9936 | `regnet_x_1_6gf + shufflenet_v2_x2_0` |
| `olympic_cifar100` | `olympic_cifar100-0020` | 0.9931 | `regnet_x_3_2gf + shufflenet_v2_x2_0` |
| `olympic_imagenette` | `olympic_imagenette-0006` | 0.9952 | `regnet_x_1_6gf + shufflenet_v2_x2_0` |

完整 36 个候选在 mutation 前写入 manifest：

```text
<run_root>/selection/single_dual_imagenette_top4_per_setting.json
```

Manifest 字段：

```text
candidate_id
setting
source_run
imagenette_acc_dual_existing
backbone_a
backbone_b
backbone_signature
cnn_signature
forward_graph_signature
completion/code reference
```

### 结构改写

对每个选中的 dual 模型，构造两个 paired single-backbone variant：

```text
A-only：保留 backbone_a，删除 backbone_b 及其 fusion contribution
B-only：保留 backbone_b，删除 backbone_a 及其 fusion contribution
```

改写必须保证 classifier head 和 tensor shape 合法。不能加自动修复、fallback 或静默降级。如果某个候选不能机械改写，就记录为 mutation failure，并保留在 denominator 里。

新增 eval candidate 数量：

```text
36 A-only + 36 B-only = 72 个 single-backbone variants
```

原始 36 个 dual 模型已经有 3x3 横评里的 Imagenette 1 epoch 结果。如果为了完全同协议比较需要，也可以重跑 dual。

### 评估

主评估：

```text
Imagenette heldout-test，1 epoch
```

补充评估：

```text
Imagenette heldout-test，10 epochs
```

如果实验 A 说明 1 epoch 不是强 ranking proxy，则 B 的 10 epoch 应完整跑 36 dual + 72 single。  
如果实验 A 说明 1 epoch proxy 很强，B 的 10 epoch 可以只跑较小 sensitivity subset。

### 指标

每个原始候选报告：

```text
dual_acc
a_only_acc
b_only_acc
best_single_acc = max(a_only_acc, b_only_acc)
dual_minus_best_single
dual_minus_a_only
dual_minus_b_only
mutation_success
eval_success
params_dual
params_a_only
params_b_only
```

汇总：

```text
paired delta 的 mean / median
dual_minus_best_single 的 bootstrap 95% CI
如果 paired eval 都成功，做 Wilcoxon signed-rank test
mutation/eval failure 单独统计，不静默丢掉
```

Article 解释口径：

- 如果 `dual_minus_best_single` 稳定为正，说明第二个 backbone 有实证价值。
- 如果 delta 接近 0，不要把 dual backbone 写成关键贡献，只能说它是 search-space choice。
- 如果 B-only 经常超过 A-only，要报告生成器可能主要使用第二条 backbone，而不是两条等价融合。

## 实验 C：Reward Ablation

### 问题

哪些 reward 组件影响 validity、accuracy 和 structural collapse？

### 分支

开新分支，避免污染当前 matrix / seed 代码：

```text
experiment/article-reward-ablation-20260608
```

建议基于当前正式 seed rerun 的提交：

```text
261bf21b7b90339f2d67e8981521e42b6ce0251b
Release reward eval dataloader workers
```

如果实际执行时用的不是这个 commit，必须在 run manifest 里记录真实 base commit。

### 条件

使用显式 env/config switch。默认行为必须和当前 full reward 完全一致。

| Condition | 行为 |
| --- | --- |
| `full_reward` | 当前 reward 不变 |
| `no_diversity_bonus` | 去掉 descriptor/CNN/block novelty bonus，保留 repeat penalty |
| `no_repeat_penalty` | 保留 novelty bonus，去掉 repeat penalty、dominant-repeat penalty 和 repeated-block cap |

不允许加 fallback、自动修复、自动补齐或静默降级。无效生成继续保持强负奖励。

### 设置

使用 4-pattern primary robustness setting：

```text
DeepSeek A18
CIFAR-10
fresh stage2_formal_explore
trainable SFT adapter
每个 condition 1000 samples
full formal reward eval
```

三组 condition 使用同一个 seed，并在 run manifest 里记录；Reward ablation 的重点是在固定随机性下只改变 reward 组件。

### 指标

报告 all-sample 和 final-100 两个窗口：

```text
formal_success_rate
mean accuracy
max accuracy
positive_reward_rate
backbone unique/effective number
module/block unique/effective number
forward graph unique/effective number
top1 backbone/module/graph share
actual_block_live_rate
dominant signature over time
```

Article 主表比较 `full_reward`、`no_diversity_bonus`、`no_repeat_penalty`。再加 success、accuracy、effective-number collapse 的 trajectory 图。

## 实验 D：1-Pattern Multi-Seed RL

### 问题

1-pattern SFT+RL 的行为是否对 RL seed 稳健？

4-pattern multi-seed 已经另行在跑。远端记录里已有 4-pattern DeepSeek A18 CIFAR-10 seed jobs，seed 为 `42/123/777`，以及后续 seed42 loader cleanup rerun。不要为了本方案取消或重命名这些 job。

### 1-Pattern 目标 Seed

使用：

```text
42
114
514
```

如果 4-pattern 已运行 seed 与这个集合不一致，保留 4-pattern 已运行事实；新的 1-pattern runs 使用 `42/114/514`。

### 设置

```text
DeepSeek selected 1-pattern SFT adapter A9
CIFAR-10
fresh stage2_formal_explore
NNGPT_SFT_LOAD_INITIAL_ADAPTER=1
NNGPT_SFT_INITIAL_ADAPTER_MODE=trainable
NNGPT_RL_FORMAL_REWARD_EPOCHS=1
NNGPT_SFT_RL_NN_PREFIXES=rl-bb-test1
NNGPT_SFT_NUM_GENERATIONS=8
NNGPT_SFT_MAX_STEPS=125
目标：每个 seed 1000 training samples
```

使用 full formal reward，不用 cheap eval。

### 指标

每个 seed 报告：

```text
all-1000 formal success
final-100 formal success
all-1000 mean accuracy
final-100 mean accuracy
positive reward rate
backbone/module/graph effective number
top1 signature share
actual_block_live_rate if available
```

汇总：

```text
mean +/- std across seeds
min/max across seeds
seed-level trajectory plots
```

Article 使用口径：

- 这组结果支撑 1-pattern RL robustness。
- 不把旧 A7/A12 single-run evidence 混进 seed mean。旧 run 只能作为 historical reference。

## 并行执行顺序

Manifest 固定后，这些实验可以并行跑：

1. A：从 DB 选 300 个 CIFAR-10 backbone 模型，提交 10 epoch eval。
2. B：从已完成 3x3 中选 36 个 Imagenette top candidates，生成 A-only/B-only variants，提交 Imagenette 1 epoch eval。
3. C：开 reward-ablation 分支，提交 3 个 1000-sample runs。
4. D：提交 1-pattern DeepSeek CIFAR-10 seeds `42/114/514`。

每次提交到集群前，必须先把 manifest 和 run root 写进本文档或链接的结果记录，避免后续 session 需要反推当时选了什么。

## 2026-06-08 执行记录

### A：CIFAR-10 backbone epoch1 vs epoch10

提交前固定 manifest：

```text
run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1345_article_A_epoch1_epoch10
selection manifest: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1345_article_A_epoch1_epoch10/selection/backbone_epoch1_epoch10_manifest.json
eval candidate file: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1345_article_A_epoch1_epoch10/selection/backbone_epoch1_epoch10_candidates_full_code.jsonl
source_db: /home/s471802/nn-gpt/db/ab.nn.db
seed: 42
selected: 300 = low 100 + mid 100 + high 100
distinct source candidates: 3141
eval: cifar-10 formal reward eval, 10 epochs, reward_eval split
```

| Job ID | 状态 | 备注 |
| --- | --- | --- |
| 2674051 | cancelled after 212/300 models | 10-epoch eval was stopped to free `jnfat06` for Article D seed123; completed rows kept in `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1345_article_A_epoch1_epoch10/eval_cifar10_epoch10/generation_samples.jsonl` |
| 2680504 | running on standard | remaining 88 models only; run root `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260609_1020_article_A_epoch1_epoch10_remaining88`; output dir `eval_cifar10_epoch10_remaining88` |

### B：Imagenette single-vs-dual top candidates

提交前固定 manifest：

```text
run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1345_article_B_single_dual_imagenette
selection manifest: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1345_article_B_single_dual_imagenette/selection/single_dual_imagenette_top4_per_setting.json
source eval: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_1455_gen30min32_formal100_dsqwenolympic/eval_heldout_test/imagenette/generation_samples.jsonl
selected: 36 = 9 settings x top 4
mutation candidate file: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1345_article_B_single_dual_imagenette/mutations/single_backbone_variants.jsonl
mutation manifest: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1345_article_B_single_dual_imagenette/mutations/single_backbone_mutation_manifest.json
single eval output: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1345_article_B_single_dual_imagenette/eval_single_variants_imagenette_epoch1
delta summary: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1345_article_B_single_dual_imagenette/eval_single_variants_imagenette_epoch1/single_vs_dual_delta_summary.json
```

机械改写 72 个 single-backbone variants，72/72 eval 成功，无 mutation/eval failure。

| Job ID | 状态 | 备注 |
| --- | --- | --- |
| 2674786 | completed | Imagenette heldout-test 1 epoch single variants eval |

初步结果：

```text
paired dual records: 36 / 36
dual mean: 99.3878%
best single mean: 99.3050%
dual - best single mean: +0.0828 pp
dual - best single median: +0.0764 pp
dual - best single bootstrap 95% CI: +0.0326 pp to +0.1281 pp
Wilcoxon signed-rank dual vs best single: W=84.0, p=0.0001066, nonzero pairs=34
dual beats best single: 28 / 36
dual ties or beats best single: 30 / 36
```

### D：1-pattern DeepSeek CIFAR-10 multi-seed

提交前固定 manifest：

```text
adapter: /home/s471802/nn-gpt/out/nngpt/llm/20260601_1pattern_three_model_sft_v3_cifar10_dscoder7b/epoch_sft/A9/deepseek-ai/deepseek-coder-6.7b-instruct
base commit: 90b868a0 on Julia2 reward-ablation branch / local 44c0b23f9
partition: h100 final; L40/L40S fallback is not reliable for this D config
finalize partition: small_cpu
gpus: 4
mem: 160G
cpus: 32
fresh stage: stage2_formal_explore
formal dataset: cifar-10
formal reward epochs: 1
prefix: rl-bb-test1
num generations: 8
max steps: 125
generation batch size: 8
max prompt length: 3000
```

L40S 4096 prompt attempts failed on train GPU OOM. 对照此前完成的 DeepSeek formal run 后，D 改为 `NNGPT_SFT_MAX_PROMPT_LENGTH=3000`，保持 `num_generations=8`、`generation_batch_size=8`、`max_steps=125` 不变，并保留 `NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU=1`、`NNGPT_REWARD_WORKERS_PER_GPU=1`。但 L40S `p3000` seed114 仍在训练 GPU0 OOM，因此最终 D 不再 fallback 到 L40/L40S，统一回 H100 排队。

后续对比 3x3 正常配置后，D 正式参数改为 `max_prompt_length=3500`、`max_completion_length=1200`、`generation_batch_size=8`，并启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。seed514 旧 commit 两次 Slurm memory OOM 后，使用 `c91714dbe` 的 reward meta memory preflight replacement 重投。

| Seed | Run ID | Run root | Job ID | 状态 |
| ---: | --- | --- | --- | --- |
| 42 | `20260608_1335_article_1pattern_dscoder_cifar10_seed42_gcvl` | `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1335_article_1pattern_dscoder_cifar10_seed42_gcvl` | 2674042 | failed: CUDA OOM on L40S train GPU0; replaced by 2674083 |
| 114 | `20260608_1335_article_1pattern_dscoder_cifar10_seed114_gcvl` | `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1335_article_1pattern_dscoder_cifar10_seed114_gcvl` | 2674044 | failed: CUDA OOM on L40S train GPU0; replaced by 2674085 |
| 514 | `20260608_1335_article_1pattern_dscoder_cifar10_seed514_gcvl` | `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1335_article_1pattern_dscoder_cifar10_seed514_gcvl` | 2674046 | failed: CUDA OOM on L40S train GPU0; replaced by 2674087 |
| 42 | `20260608_1403_article_1pattern_dscoder_cifar10_seed42_gcvl_rewgpu2` | `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1403_article_1pattern_dscoder_cifar10_seed42_gcvl_rewgpu2` | 2674249 | failed: CUDA OOM on L40S train GPU0 with prompt 4096 |
| 114 | `20260608_1403_article_1pattern_dscoder_cifar10_seed114_gcvl_rewgpu2` | `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1403_article_1pattern_dscoder_cifar10_seed114_gcvl_rewgpu2` | 2674251 | failed: CUDA OOM on L40S train GPU0 with prompt 4096 |
| 514 | `20260608_1403_article_1pattern_dscoder_cifar10_seed514_gcvl_rewgpu2` | `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1403_article_1pattern_dscoder_cifar10_seed514_gcvl_rewgpu2` | 2674253 | failed: CUDA OOM on L40S train GPU0 with prompt 4096 |
| 114 | `20260608_1526_article_1pattern_dscoder_cifar10_seed114_gcvl_p3000` | `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1526_article_1pattern_dscoder_cifar10_seed114_gcvl_p3000` | 2675601 | failed: CUDA OOM on L40S train GPU0 with prompt 3000 |
| 42 | `20260608_1535_article_1pattern_dscoder_cifar10_seed42_h100_p3000_final` | `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1535_article_1pattern_dscoder_cifar10_seed42_h100_p3000_final` | 2675730 | pending on H100; prompt 3000 |
| 114 | `20260608_1535_article_1pattern_dscoder_cifar10_seed114_h100_p3000_final` | `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1535_article_1pattern_dscoder_cifar10_seed114_h100_p3000_final` | 2675732 | pending on H100; prompt 3000 |
| 514 | `20260608_1535_article_1pattern_dscoder_cifar10_seed514_h100_p3000_final` | `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1535_article_1pattern_dscoder_cifar10_seed514_h100_p3000_final` | 2675734 | pending on H100; prompt 3000 |
| 42 | `20260608_1922_article_D_a9_cifar10_seed42_p3500_c1200_alloc` | `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1922_article_D_a9_cifar10_seed42_p3500_c1200_alloc` | 2677251 | completed on standard; prompt 3500, completion 1200 |
| 114 | `20260608_1922_article_D_a9_cifar10_seed114_p3500_c1200_alloc` | `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1922_article_D_a9_cifar10_seed114_p3500_c1200_alloc` | 2677253 | completed but collapsed; prompt 3500, completion 1200 |
| 514 | `20260608_1922_article_D_a9_cifar10_seed514_p3500_c1200_alloc` | `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1922_article_D_a9_cifar10_seed514_p3500_c1200_alloc` | 2677255 | failed: Slurm memory OOM at 160G; replaced by 2677284 |
| 514 | `20260608_1945_article_D_a9_cifar10_seed514_p3500_c1200_alloc_mem240` | `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1945_article_D_a9_cifar10_seed514_p3500_c1200_alloc_mem240` | 2677284 | failed: Slurm memory OOM at 240G on old commit; replaced by 2677791 |
| 514 | `20260608_2245_article_d_a9_cifar10_seed514_p3500_c1200_alloc_meta` | `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_2245_article_d_a9_cifar10_seed514_p3500_c1200_alloc_meta` | 2677791 | failed: Slurm memory OOM at 160G despite meta preflight; commit `c91714dbe`, prompt 3500, completion 1200 |
| 123 | `20260609_1016_article_D_a9_cifar10_seed123_gcvl_p3500_c1200_alloc` | `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260609_1016_article_D_a9_cifar10_seed123_gcvl_p3500_c1200_alloc` | 2680341 | pending on gpu_computervision_long; H100 predicted start exceeded 2h; prompt 3500, completion 1200 |
| 777 | `20260609_1019_article_D_a9_cifar10_seed777_std_p3500_c1200_alloc` | `/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260609_1019_article_D_a9_cifar10_seed777_std_p3500_c1200_alloc` | 2680345 | running on standard; H100/gpu_computervision_long predicted start exceeded 2h; prompt 3500, completion 1200 |

### C：Reward ablation

已在分支 `experiment/article-reward-ablation-20260608` 实现 `full_reward`、`no_diversity_bonus`、`no_repeat_penalty`，本地 commit `7cfd5c112`、`44c0b23f9` 已推送；Julia2 分支 cherry-pick 为 `d7435127`、`90b868a0`。

| Condition | Run ID | Job ID | 状态 |
| --- | --- | ---: | --- |
H100 pending 版未启动，已取消并从 archive 删除。第一次 L40S 版误投为 `formal_reward_epochs=10`，已取消并从 archive 删除。最终 C 按文档口径改投 `formal_reward_epochs=1`；参数核查为 `max_prompt_length=3500`、`generation_batch_size=None`，并保留 `NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU=1` 避免 reward worker 占训练卡。

| Condition | Run ID | Job ID | 状态 |
| --- | --- | ---: | --- |
| full_reward | `20260608_1547_article_reward_ablation_a18_cifar10_gcvl_e1_full_reward` | 2675832 | running on gpu_computervision_long |
| no_diversity_bonus | `20260608_1547_article_reward_ablation_a18_cifar10_gcvl_e1_no_diversity_bonus` | 2675833 | running on gpu_computervision_long |
| no_repeat_penalty | `20260608_1547_article_reward_ablation_a18_cifar10_gcvl_e1_no_repeat_penalty` | 2675834 | pending on gpu_computervision_long |
