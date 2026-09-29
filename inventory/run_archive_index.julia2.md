## 20260604_1617_rl_dscoder_cifar_10_fix

## 20260604_1617_rl_dscoder_cifar_100_fix

## 20260604_1617_rl_dscoder_imagenette_fix

## 20260604_1617_rl_mistral_cifar_10_fix

## 20260604_1617_rl_mistral_cifar_100_fix

## 20260604_1617_rl_mistral_imagenette_fix

## 20260604_1617_rl_qwen_cifar_10_fix

## 20260604_1617_rl_qwen_cifar_100_fix

## 20260604_1617_rl_qwen_imagenette_fix

## 20260604_1619_rl_dscoder_cifar_10_fix

<!-- NNGPT_RUN:20260604_1619_rl_dscoder_cifar_10_fix:START -->
- 运行 ID：`20260604_1619_rl_dscoder_cifar_10_fix`
- 标签：`rl_dscoder_cifar-10_fix`
- 状态：已提交
- 提交时间：`2026-06-04T16:19:22+02:00`
- 开始时间：-
- 结束时间：-
- Job ID：`2665508`
- 分区 / QoS：`h100`
- 节点：-
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL rerun: dscoder on cifar-10 after warmup+split-cache fixes
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_dscoder_cifar_10_fix`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_dscoder_cifar_10_fix/slurm/tunerl-rl_dscoder_cifar_10_fix-2665508.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_dscoder_cifar_10_fix/slurm/tunerl-rl_dscoder_cifar_10_fix-2665508.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：完成 800/800，Slurm COMPLETED 0:0，耗时 09:31:49。全轨迹 built/formal/positive 为 621/800、621/800、0/800，5-epoch formal mean acc 88.33%。final-100 built/formal/positive 为 100/100、100/100、0/100，mean reward -0.8030，5-epoch mean acc 88.36%，max 92.66%，Bb Eff. 10.97，Block Eff. 1.00，Family Top-1 100.0%，Graph Eff. 1.00。
- 主要缺陷：formal success 不等于有效 reward；final-100 完全没有 positive reward，且 Block/Graph Eff. 均为 1.00，属于 zero-positive structural replay。日志中有候选级 timeout 字样，但没有 OOM、Traceback、Killed 或 worker restart。
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_dscoder_cifar_10_fix`
<!-- NNGPT_RUN:20260604_1619_rl_dscoder_cifar_10_fix:END -->
## 20260604_1619_rl_dscoder_cifar_100_fix

<!-- NNGPT_RUN:20260604_1619_rl_dscoder_cifar_100_fix:START -->
- 运行 ID：`20260604_1619_rl_dscoder_cifar_100_fix`
- 标签：`rl_dscoder_cifar-100_fix`
- 状态：已提交
- 提交时间：`2026-06-04T16:19:22+02:00`
- 开始时间：-
- 结束时间：-
- Job ID：`2665510`
- 分区 / QoS：`h100`
- 节点：-
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL rerun: dscoder on cifar-100 after warmup+split-cache fixes
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_dscoder_cifar_100_fix`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_dscoder_cifar_100_fix/slurm/tunerl-rl_dscoder_cifar_100_fix-2665510.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_dscoder_cifar_100_fix/slurm/tunerl-rl_dscoder_cifar_100_fix-2665510.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：完成 800/800，Slurm COMPLETED 0:0，耗时 14:19:35。全轨迹 built/formal/positive 为 727/800、727/800、723/800，5-epoch formal mean acc 87.04%。final-100 built/formal/positive 为 100/100、100/100、100/100，mean reward 0.1822，5-epoch mean acc 87.98%，max 88.64%，Bb Eff. 1.00，Block Eff. 5.07，Family Top-1 100.0%，Graph Eff. 5.07。
- 主要缺陷：final-100 reward 与 formal gate 均正常，但 family 完全坍塌，backbone signature 也完全坍塌；Block/Graph 仍有少量变化。日志中有候选级 timeout 字样，但没有 OOM、Traceback、Killed 或 worker restart。
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_dscoder_cifar_100_fix`
<!-- NNGPT_RUN:20260604_1619_rl_dscoder_cifar_100_fix:END -->
## 20260604_1619_rl_dscoder_imagenette_fix

<!-- NNGPT_RUN:20260604_1619_rl_dscoder_imagenette_fix:START -->
- 运行 ID：`20260604_1619_rl_dscoder_imagenette_fix`
- 标签：`rl_dscoder_imagenette_fix`
- 状态：已提交
- 提交时间：`2026-06-04T16:19:22+02:00`
- 开始时间：-
- 结束时间：-
- Job ID：`2665512`
- 分区 / QoS：`h100`
- 节点：-
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL rerun: dscoder on imagenette after warmup+split-cache fixes
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_dscoder_imagenette_fix`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_dscoder_imagenette_fix/slurm/tunerl-rl_dscoder_imagenette_fix-2665512.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_dscoder_imagenette_fix/slurm/tunerl-rl_dscoder_imagenette_fix-2665512.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_dscoder_imagenette_fix`
<!-- NNGPT_RUN:20260604_1619_rl_dscoder_imagenette_fix:END -->
## 20260604_1619_rl_mistral_cifar_10_fix

<!-- NNGPT_RUN:20260604_1619_rl_mistral_cifar_10_fix:START -->
- 运行 ID：`20260604_1619_rl_mistral_cifar_10_fix`
- 标签：`rl_mistral_cifar-10_fix`
- 状态：已人工停止；正式结果完成（Slurm 因 TERM 记为 FAILED）
- 提交时间：`2026-06-04T16:19:22+02:00`
- 开始时间：`2026-06-04T16:19:29+02:00`
- 结束时间：`2026-06-04T16:20:18+02:00`
- Job ID：`2665514`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL rerun: mistral on cifar-10 after warmup+split-cache fixes
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_mistral_cifar_10_fix`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_mistral_cifar_10_fix/slurm/tunerl-rl_mistral_cifar_10_fix-2665514.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_mistral_cifar_10_fix/slurm/tunerl-rl_mistral_cifar_10_fix-2665514.err`
- 工作目录：`/tmp/s471802/20260604_1619_rl_mistral_cifar_10_fix/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_mistral_cifar_10_fix`
<!-- NNGPT_RUN:20260604_1619_rl_mistral_cifar_10_fix:END -->
## 20260604_1619_rl_mistral_cifar_100_fix

<!-- NNGPT_RUN:20260604_1619_rl_mistral_cifar_100_fix:START -->
- 运行 ID：`20260604_1619_rl_mistral_cifar_100_fix`
- 标签：`rl_mistral_cifar-100_fix`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T16:19:22+02:00`
- 开始时间：`2026-06-04T16:20:28+02:00`
- 结束时间：`2026-06-04T16:21:10+02:00`
- Job ID：`2665516`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL rerun: mistral on cifar-100 after warmup+split-cache fixes
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_mistral_cifar_100_fix`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_mistral_cifar_100_fix/slurm/tunerl-rl_mistral_cifar_100_fix-2665516.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_mistral_cifar_100_fix/slurm/tunerl-rl_mistral_cifar_100_fix-2665516.err`
- 工作目录：`/tmp/s471802/20260604_1619_rl_mistral_cifar_100_fix/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_mistral_cifar_100_fix`
<!-- NNGPT_RUN:20260604_1619_rl_mistral_cifar_100_fix:END -->
## 20260604_1619_rl_mistral_imagenette_fix

<!-- NNGPT_RUN:20260604_1619_rl_mistral_imagenette_fix:START -->
- 运行 ID：`20260604_1619_rl_mistral_imagenette_fix`
- 标签：`rl_mistral_imagenette_fix`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T16:19:23+02:00`
- 开始时间：`2026-06-04T16:21:28+02:00`
- 结束时间：`2026-06-04T16:22:08+02:00`
- Job ID：`2665518`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL rerun: mistral on imagenette after warmup+split-cache fixes
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_mistral_imagenette_fix`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_mistral_imagenette_fix/slurm/tunerl-rl_mistral_imagenette_fix-2665518.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_mistral_imagenette_fix/slurm/tunerl-rl_mistral_imagenette_fix-2665518.err`
- 工作目录：`/tmp/s471802/20260604_1619_rl_mistral_imagenette_fix/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_mistral_imagenette_fix`
<!-- NNGPT_RUN:20260604_1619_rl_mistral_imagenette_fix:END -->
## 20260604_1619_rl_qwen_cifar_10_fix

<!-- NNGPT_RUN:20260604_1619_rl_qwen_cifar_10_fix:START -->
- 运行 ID：`20260604_1619_rl_qwen_cifar_10_fix`
- 标签：`rl_qwen_cifar-10_fix`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T16:19:23+02:00`
- 开始时间：`2026-06-04T16:22:29+02:00`
- 结束时间：`2026-06-04T16:23:09+02:00`
- Job ID：`2665520`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL rerun: qwen on cifar-10 after warmup+split-cache fixes
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_qwen_cifar_10_fix`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_qwen_cifar_10_fix/slurm/tunerl-rl_qwen_cifar_10_fix-2665520.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_qwen_cifar_10_fix/slurm/tunerl-rl_qwen_cifar_10_fix-2665520.err`
- 工作目录：`/tmp/s471802/20260604_1619_rl_qwen_cifar_10_fix/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_qwen_cifar_10_fix`
<!-- NNGPT_RUN:20260604_1619_rl_qwen_cifar_10_fix:END -->
## 20260604_1619_rl_qwen_cifar_100_fix

<!-- NNGPT_RUN:20260604_1619_rl_qwen_cifar_100_fix:START -->
- 运行 ID：`20260604_1619_rl_qwen_cifar_100_fix`
- 标签：`rl_qwen_cifar-100_fix`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T16:19:23+02:00`
- 开始时间：`2026-06-04T16:23:29+02:00`
- 结束时间：`2026-06-04T16:24:10+02:00`
- Job ID：`2665522`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL rerun: qwen on cifar-100 after warmup+split-cache fixes
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_qwen_cifar_100_fix`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_qwen_cifar_100_fix/slurm/tunerl-rl_qwen_cifar_100_fix-2665522.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_qwen_cifar_100_fix/slurm/tunerl-rl_qwen_cifar_100_fix-2665522.err`
- 工作目录：`/tmp/s471802/20260604_1619_rl_qwen_cifar_100_fix/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_qwen_cifar_100_fix`
<!-- NNGPT_RUN:20260604_1619_rl_qwen_cifar_100_fix:END -->
## 20260604_1619_rl_qwen_imagenette_fix

<!-- NNGPT_RUN:20260604_1619_rl_qwen_imagenette_fix:START -->
- 运行 ID：`20260604_1619_rl_qwen_imagenette_fix`
- 标签：`rl_qwen_imagenette_fix`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T16:19:23+02:00`
- 开始时间：`2026-06-04T16:24:29+02:00`
- 结束时间：`2026-06-04T16:25:11+02:00`
- Job ID：`2665524`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL rerun: qwen on imagenette after warmup+split-cache fixes
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_qwen_imagenette_fix`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_qwen_imagenette_fix/slurm/tunerl-rl_qwen_imagenette_fix-2665524.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_qwen_imagenette_fix/slurm/tunerl-rl_qwen_imagenette_fix-2665524.err`
- 工作目录：`/tmp/s471802/20260604_1619_rl_qwen_imagenette_fix/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1619_rl_qwen_imagenette_fix`
<!-- NNGPT_RUN:20260604_1619_rl_qwen_imagenette_fix:END -->
## 20260604_1636_rl_dscoder_cifar_10_fix_v2

<!-- NNGPT_RUN:20260604_1636_rl_dscoder_cifar_10_fix_v2:START -->
- 运行 ID：`20260604_1636_rl_dscoder_cifar_10_fix_v2`
- 标签：`rl_dscoder_cifar-10_fix_v2`
- 状态：已替换(H100等候过长，被 20260604_1645_rl_dscoder_cifar_10_gcvl 替换)
- 提交时间：`2026-06-04T16:36:23+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T16:45:19+02:00`
- Job ID：`2665531`
- 分区 / QoS：`h100`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: dscoder on cifar-10 after warmup+split-cache fixes (v2 with model dir)
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_dscoder_cifar_10_fix_v2`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_dscoder_cifar_10_fix_v2/slurm/tunerl-rl_dscoder_cifar_10_fix_v2-2665531.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_dscoder_cifar_10_fix_v2/slurm/tunerl-rl_dscoder_cifar_10_fix_v2-2665531.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_dscoder_cifar_10_fix_v2`

- 替换原因：H100 StartTime 超过 2 小时，按规则 fallback 到 gpu_computervision_long<!-- NNGPT_RUN:20260604_1636_rl_dscoder_cifar_10_fix_v2:END -->
## 20260604_1636_rl_dscoder_cifar_100_fix_v2

<!-- NNGPT_RUN:20260604_1636_rl_dscoder_cifar_100_fix_v2:START -->
- 运行 ID：`20260604_1636_rl_dscoder_cifar_100_fix_v2`
- 标签：`rl_dscoder_cifar-100_fix_v2`
- 状态：已替换(H100等候过长，被 20260604_1645_rl_dscoder_cifar_100_gcvl 替换)
- 提交时间：`2026-06-04T16:36:23+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T16:45:19+02:00`
- Job ID：`2665533`
- 分区 / QoS：`h100`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: dscoder on cifar-100 after warmup+split-cache fixes (v2 with model dir)
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_dscoder_cifar_100_fix_v2`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_dscoder_cifar_100_fix_v2/slurm/tunerl-rl_dscoder_cifar_100_fix_v2-2665533.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_dscoder_cifar_100_fix_v2/slurm/tunerl-rl_dscoder_cifar_100_fix_v2-2665533.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_dscoder_cifar_100_fix_v2`

- 替换原因：H100 StartTime 超过 2 小时，按规则 fallback 到 gpu_computervision_long<!-- NNGPT_RUN:20260604_1636_rl_dscoder_cifar_100_fix_v2:END -->
## 20260604_1636_rl_dscoder_imagenette_fix_v2

<!-- NNGPT_RUN:20260604_1636_rl_dscoder_imagenette_fix_v2:START -->
- 运行 ID：`20260604_1636_rl_dscoder_imagenette_fix_v2`
- 标签：`rl_dscoder_imagenette_fix_v2`
- 状态：已替换(H100等候过长，被 20260604_1645_rl_dscoder_imagenette_gcvl 替换)
- 提交时间：`2026-06-04T16:36:23+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T16:45:19+02:00`
- Job ID：`2665535`
- 分区 / QoS：`h100`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: dscoder on imagenette after warmup+split-cache fixes (v2 with model dir)
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_dscoder_imagenette_fix_v2`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_dscoder_imagenette_fix_v2/slurm/tunerl-rl_dscoder_imagenette_fix_v2-2665535.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_dscoder_imagenette_fix_v2/slurm/tunerl-rl_dscoder_imagenette_fix_v2-2665535.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_dscoder_imagenette_fix_v2`

- 替换原因：H100 StartTime 超过 2 小时，按规则 fallback 到 gpu_computervision_long<!-- NNGPT_RUN:20260604_1636_rl_dscoder_imagenette_fix_v2:END -->
## 20260604_1636_rl_mistral_cifar_10_fix_v2

<!-- NNGPT_RUN:20260604_1636_rl_mistral_cifar_10_fix_v2:START -->
- 运行 ID：`20260604_1636_rl_mistral_cifar_10_fix_v2`
- 标签：`rl_mistral_cifar-10_fix_v2`
- 状态：已替换(被 20260604_1642_rl_mistral_cifar_10_fix_v3 替换)
- 提交时间：`2026-06-04T16:36:23+02:00`
- 开始时间：`2026-06-04T16:36:29+02:00`
- 结束时间：`2026-06-04T16:37:09+02:00`
- Job ID：`2665537`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: mistral on cifar-10 after warmup+split-cache fixes (v2 with model dir)
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_mistral_cifar_10_fix_v2`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_mistral_cifar_10_fix_v2/slurm/tunerl-rl_mistral_cifar_10_fix_v2-2665537.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_mistral_cifar_10_fix_v2/slurm/tunerl-rl_mistral_cifar_10_fix_v2-2665537.err`
- 工作目录：`/tmp/s471802/20260604_1636_rl_mistral_cifar_10_fix_v2/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_mistral_cifar_10_fix_v2`
- 替换原因：使用 NNGPT_SFT_BASE_MODEL_ID 替代无效的 NNGPT_SFT_MODEL_DIR
<!-- NNGPT_RUN:20260604_1636_rl_mistral_cifar_10_fix_v2:END -->
## 20260604_1636_rl_mistral_cifar_100_fix_v2

<!-- NNGPT_RUN:20260604_1636_rl_mistral_cifar_100_fix_v2:START -->
- 运行 ID：`20260604_1636_rl_mistral_cifar_100_fix_v2`
- 标签：`rl_mistral_cifar-100_fix_v2`
- 状态：已替换(被 20260604_1642_rl_mistral_cifar_100_fix_v3 替换)
- 提交时间：`2026-06-04T16:36:23+02:00`
- 开始时间：`2026-06-04T16:37:29+02:00`
- 结束时间：`2026-06-04T16:38:09+02:00`
- Job ID：`2665539`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: mistral on cifar-100 after warmup+split-cache fixes (v2 with model dir)
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_mistral_cifar_100_fix_v2`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_mistral_cifar_100_fix_v2/slurm/tunerl-rl_mistral_cifar_100_fix_v2-2665539.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_mistral_cifar_100_fix_v2/slurm/tunerl-rl_mistral_cifar_100_fix_v2-2665539.err`
- 工作目录：`/tmp/s471802/20260604_1636_rl_mistral_cifar_100_fix_v2/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_mistral_cifar_100_fix_v2`
- 替换原因：使用 NNGPT_SFT_BASE_MODEL_ID 替代无效的 NNGPT_SFT_MODEL_DIR
<!-- NNGPT_RUN:20260604_1636_rl_mistral_cifar_100_fix_v2:END -->
## 20260604_1636_rl_mistral_imagenette_fix_v2

<!-- NNGPT_RUN:20260604_1636_rl_mistral_imagenette_fix_v2:START -->
- 运行 ID：`20260604_1636_rl_mistral_imagenette_fix_v2`
- 标签：`rl_mistral_imagenette_fix_v2`
- 状态：已替换(被 20260604_1642_rl_mistral_imagenette_fix_v3 替换)
- 提交时间：`2026-06-04T16:36:23+02:00`
- 开始时间：`2026-06-04T16:38:29+02:00`
- 结束时间：`2026-06-04T16:39:09+02:00`
- Job ID：`2665541`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: mistral on imagenette after warmup+split-cache fixes (v2 with model dir)
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_mistral_imagenette_fix_v2`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_mistral_imagenette_fix_v2/slurm/tunerl-rl_mistral_imagenette_fix_v2-2665541.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_mistral_imagenette_fix_v2/slurm/tunerl-rl_mistral_imagenette_fix_v2-2665541.err`
- 工作目录：`/tmp/s471802/20260604_1636_rl_mistral_imagenette_fix_v2/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_mistral_imagenette_fix_v2`
- 替换原因：使用 NNGPT_SFT_BASE_MODEL_ID 替代无效的 NNGPT_SFT_MODEL_DIR
<!-- NNGPT_RUN:20260604_1636_rl_mistral_imagenette_fix_v2:END -->
## 20260604_1636_rl_qwen_cifar_10_fix_v2

<!-- NNGPT_RUN:20260604_1636_rl_qwen_cifar_10_fix_v2:START -->
- 运行 ID：`20260604_1636_rl_qwen_cifar_10_fix_v2`
- 标签：`rl_qwen_cifar-10_fix_v2`
- 状态：已替换(被 20260604_1642_rl_qwen_cifar_10_fix_v3 替换)
- 提交时间：`2026-06-04T16:36:24+02:00`
- 开始时间：`2026-06-04T16:39:29+02:00`
- 结束时间：`2026-06-04T16:40:09+02:00`
- Job ID：`2665543`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: qwen on cifar-10 after warmup+split-cache fixes (v2 with model dir)
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_qwen_cifar_10_fix_v2`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_qwen_cifar_10_fix_v2/slurm/tunerl-rl_qwen_cifar_10_fix_v2-2665543.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_qwen_cifar_10_fix_v2/slurm/tunerl-rl_qwen_cifar_10_fix_v2-2665543.err`
- 工作目录：`/tmp/s471802/20260604_1636_rl_qwen_cifar_10_fix_v2/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_qwen_cifar_10_fix_v2`
- 替换原因：使用 NNGPT_SFT_BASE_MODEL_ID 替代无效的 NNGPT_SFT_MODEL_DIR
<!-- NNGPT_RUN:20260604_1636_rl_qwen_cifar_10_fix_v2:END -->
## 20260604_1636_rl_qwen_cifar_100_fix_v2

<!-- NNGPT_RUN:20260604_1636_rl_qwen_cifar_100_fix_v2:START -->
- 运行 ID：`20260604_1636_rl_qwen_cifar_100_fix_v2`
- 标签：`rl_qwen_cifar-100_fix_v2`
- 状态：已替换(被 20260604_1642_rl_qwen_cifar_100_fix_v3 替换)
- 提交时间：`2026-06-04T16:36:24+02:00`
- 开始时间：-
- 结束时间：-
- Job ID：`2665545`
- 分区 / QoS：`gpu_computervision_long`
- 节点：-
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: qwen on cifar-100 after warmup+split-cache fixes (v2 with model dir)
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_qwen_cifar_100_fix_v2`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_qwen_cifar_100_fix_v2/slurm/tunerl-rl_qwen_cifar_100_fix_v2-2665545.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_qwen_cifar_100_fix_v2/slurm/tunerl-rl_qwen_cifar_100_fix_v2-2665545.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_qwen_cifar_100_fix_v2`
- 替换原因：使用 NNGPT_SFT_BASE_MODEL_ID 替代无效的 NNGPT_SFT_MODEL_DIR
<!-- NNGPT_RUN:20260604_1636_rl_qwen_cifar_100_fix_v2:END -->
## 20260604_1636_rl_qwen_imagenette_fix_v2

<!-- NNGPT_RUN:20260604_1636_rl_qwen_imagenette_fix_v2:START -->
- 运行 ID：`20260604_1636_rl_qwen_imagenette_fix_v2`
- 标签：`rl_qwen_imagenette_fix_v2`
- 状态：已替换(被 20260604_1642_rl_qwen_imagenette_fix_v3 替换)
- 提交时间：`2026-06-04T16:36:24+02:00`
- 开始时间：-
- 结束时间：-
- Job ID：`2665547`
- 分区 / QoS：`gpu_computervision_long`
- 节点：-
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: qwen on imagenette after warmup+split-cache fixes (v2 with model dir)
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_qwen_imagenette_fix_v2`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_qwen_imagenette_fix_v2/slurm/tunerl-rl_qwen_imagenette_fix_v2-2665547.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_qwen_imagenette_fix_v2/slurm/tunerl-rl_qwen_imagenette_fix_v2-2665547.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1636_rl_qwen_imagenette_fix_v2`
- 替换原因：使用 NNGPT_SFT_BASE_MODEL_ID 替代无效的 NNGPT_SFT_MODEL_DIR
<!-- NNGPT_RUN:20260604_1636_rl_qwen_imagenette_fix_v2:END -->
## 20260604_1642_rl_mistral_cifar_10_fix_v3

<!-- NNGPT_RUN:20260604_1642_rl_mistral_cifar_10_fix_v3:START -->
- 运行 ID：`20260604_1642_rl_mistral_cifar_10_fix_v3`
- 标签：`rl_mistral_cifar-10_fix_v3`
- 状态：已替换(H100等候过长，被 20260604_1645_rl_mistral_cifar_10_gcvl 替换)
- 提交时间：`2026-06-04T16:42:13+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T16:45:19+02:00`
- Job ID：`2665552`
- 分区 / QoS：`h100`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: mistral on cifar-10, v3 fix NNGPT_SFT_BASE_MODEL_ID
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_mistral_cifar_10_fix_v3`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_mistral_cifar_10_fix_v3/slurm/tunerl-rl_mistral_cifar_10_fix_v3-2665552.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_mistral_cifar_10_fix_v3/slurm/tunerl-rl_mistral_cifar_10_fix_v3-2665552.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_mistral_cifar_10_fix_v3`

- 替换原因：H100 StartTime 超过 2 小时，按规则 fallback 到 gpu_computervision_long<!-- NNGPT_RUN:20260604_1642_rl_mistral_cifar_10_fix_v3:END -->
## 20260604_1642_rl_mistral_cifar_100_fix_v3

<!-- NNGPT_RUN:20260604_1642_rl_mistral_cifar_100_fix_v3:START -->
- 运行 ID：`20260604_1642_rl_mistral_cifar_100_fix_v3`
- 标签：`rl_mistral_cifar_100_fix_v3`
- 状态：已替换(H100等候过长，被 20260604_1645_rl_mistral_cifar_100_gcvl 替换)
- 提交时间：`2026-06-04T16:42:26+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T16:45:19+02:00`
- Job ID：`2665554`
- 分区 / QoS：`h100`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: mistral on cifar-100, v3 fix NNGPT_SFT_BASE_MODEL_ID
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_mistral_cifar_100_fix_v3`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_mistral_cifar_100_fix_v3/slurm/tunerl-rl_mistral_cifar_100_fix_v3-2665554.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_mistral_cifar_100_fix_v3/slurm/tunerl-rl_mistral_cifar_100_fix_v3-2665554.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_mistral_cifar_100_fix_v3`

- 替换原因：H100 StartTime 超过 2 小时，按规则 fallback 到 gpu_computervision_long<!-- NNGPT_RUN:20260604_1642_rl_mistral_cifar_100_fix_v3:END -->
## 20260604_1642_rl_mistral_imagenette_fix_v3

<!-- NNGPT_RUN:20260604_1642_rl_mistral_imagenette_fix_v3:START -->
- 运行 ID：`20260604_1642_rl_mistral_imagenette_fix_v3`
- 标签：`rl_mistral_imagenette_fix_v3`
- 状态：已替换(H100等候过长，被 20260604_1645_rl_mistral_imagenette_gcvl 替换)
- 提交时间：`2026-06-04T16:42:26+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T16:45:20+02:00`
- Job ID：`2665556`
- 分区 / QoS：`h100`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: mistral on imagenette, v3 fix NNGPT_SFT_BASE_MODEL_ID
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_mistral_imagenette_fix_v3`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_mistral_imagenette_fix_v3/slurm/tunerl-rl_mistral_imagenette_fix_v3-2665556.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_mistral_imagenette_fix_v3/slurm/tunerl-rl_mistral_imagenette_fix_v3-2665556.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_mistral_imagenette_fix_v3`

- 替换原因：H100 StartTime 超过 2 小时，按规则 fallback 到 gpu_computervision_long<!-- NNGPT_RUN:20260604_1642_rl_mistral_imagenette_fix_v3:END -->
## 20260604_1642_rl_qwen_cifar_10_fix_v3

<!-- NNGPT_RUN:20260604_1642_rl_qwen_cifar_10_fix_v3:START -->
- 运行 ID：`20260604_1642_rl_qwen_cifar_10_fix_v3`
- 标签：`rl_qwen_cifar_10_fix_v3`
- 状态：已替换(H100等候过长，被 20260604_1645_rl_qwen_cifar_10_gcvl 替换)
- 提交时间：`2026-06-04T16:42:26+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T16:45:20+02:00`
- Job ID：`2665558`
- 分区 / QoS：`h100`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: qwen on cifar-10, v3 fix NNGPT_SFT_BASE_MODEL_ID
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_qwen_cifar_10_fix_v3`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_qwen_cifar_10_fix_v3/slurm/tunerl-rl_qwen_cifar_10_fix_v3-2665558.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_qwen_cifar_10_fix_v3/slurm/tunerl-rl_qwen_cifar_10_fix_v3-2665558.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_qwen_cifar_10_fix_v3`

- 替换原因：H100 StartTime 超过 2 小时，按规则 fallback 到 gpu_computervision_long<!-- NNGPT_RUN:20260604_1642_rl_qwen_cifar_10_fix_v3:END -->
## 20260604_1642_rl_qwen_cifar_100_fix_v3

<!-- NNGPT_RUN:20260604_1642_rl_qwen_cifar_100_fix_v3:START -->
- 运行 ID：`20260604_1642_rl_qwen_cifar_100_fix_v3`
- 标签：`rl_qwen_cifar_100_fix_v3`
- 状态：已替换(H100等候过长，被 20260604_1645_rl_qwen_cifar_100_gcvl 替换)
- 提交时间：`2026-06-04T16:42:26+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T16:45:20+02:00`
- Job ID：`2665560`
- 分区 / QoS：`h100`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: qwen on cifar-100, v3 fix NNGPT_SFT_BASE_MODEL_ID
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_qwen_cifar_100_fix_v3`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_qwen_cifar_100_fix_v3/slurm/tunerl-rl_qwen_cifar_100_fix_v3-2665560.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_qwen_cifar_100_fix_v3/slurm/tunerl-rl_qwen_cifar_100_fix_v3-2665560.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_qwen_cifar_100_fix_v3`

- 替换原因：H100 StartTime 超过 2 小时，按规则 fallback 到 gpu_computervision_long<!-- NNGPT_RUN:20260604_1642_rl_qwen_cifar_100_fix_v3:END -->
## 20260604_1642_rl_qwen_imagenette_fix_v3

<!-- NNGPT_RUN:20260604_1642_rl_qwen_imagenette_fix_v3:START -->
- 运行 ID：`20260604_1642_rl_qwen_imagenette_fix_v3`
- 标签：`rl_qwen_imagenette_fix_v3`
- 状态：已替换(H100等候过长，被 20260604_1645_rl_qwen_imagenette_gcvl 替换)
- 提交时间：`2026-06-04T16:42:26+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T16:45:19+02:00`
- Job ID：`2665562`
- 分区 / QoS：`h100`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: qwen on imagenette, v3 fix NNGPT_SFT_BASE_MODEL_ID
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_qwen_imagenette_fix_v3`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_qwen_imagenette_fix_v3/slurm/tunerl-rl_qwen_imagenette_fix_v3-2665562.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_qwen_imagenette_fix_v3/slurm/tunerl-rl_qwen_imagenette_fix_v3-2665562.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1642_rl_qwen_imagenette_fix_v3`

- 替换原因：H100 StartTime 超过 2 小时，按规则 fallback 到 gpu_computervision_long<!-- NNGPT_RUN:20260604_1642_rl_qwen_imagenette_fix_v3:END -->
## 20260604_1645_rl_dscoder_cifar_10_gcvl

<!-- NNGPT_RUN:20260604_1645_rl_dscoder_cifar_10_gcvl:START -->
- 运行 ID：`20260604_1645_rl_dscoder_cifar_10_gcvl`
- 标签：`rl_dscoder_cifar_10_gcvl`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T16:45:38+02:00`
- 开始时间：`2026-06-04T16:45:59+02:00`
- 结束时间：`2026-06-04T16:46:12+02:00`
- Job ID：`2665564`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: dscoder on cifar-10 to gpu_computervision_long
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_dscoder_cifar_10_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_dscoder_cifar_10_gcvl/slurm/tunerl-rl_dscoder_cifar_10_gcvl-2665564.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_dscoder_cifar_10_gcvl/slurm/tunerl-rl_dscoder_cifar_10_gcvl-2665564.err`
- 工作目录：`/tmp/s471802/20260604_1645_rl_dscoder_cifar_10_gcvl/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_dscoder_cifar_10_gcvl`
<!-- NNGPT_RUN:20260604_1645_rl_dscoder_cifar_10_gcvl:END -->
## 20260604_1645_rl_dscoder_cifar_100_gcvl

<!-- NNGPT_RUN:20260604_1645_rl_dscoder_cifar_100_gcvl:START -->
- 运行 ID：`20260604_1645_rl_dscoder_cifar_100_gcvl`
- 标签：`rl_dscoder_cifar_100_gcvl`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T16:45:38+02:00`
- 开始时间：`2026-06-04T16:46:30+02:00`
- 结束时间：`2026-06-04T16:46:44+02:00`
- Job ID：`2665566`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: dscoder on cifar-100 to gpu_computervision_long
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_dscoder_cifar_100_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_dscoder_cifar_100_gcvl/slurm/tunerl-rl_dscoder_cifar_100_gcvl-2665566.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_dscoder_cifar_100_gcvl/slurm/tunerl-rl_dscoder_cifar_100_gcvl-2665566.err`
- 工作目录：`/tmp/s471802/20260604_1645_rl_dscoder_cifar_100_gcvl/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_dscoder_cifar_100_gcvl`
<!-- NNGPT_RUN:20260604_1645_rl_dscoder_cifar_100_gcvl:END -->
## 20260604_1645_rl_dscoder_imagenette_gcvl

<!-- NNGPT_RUN:20260604_1645_rl_dscoder_imagenette_gcvl:START -->
- 运行 ID：`20260604_1645_rl_dscoder_imagenette_gcvl`
- 标签：`rl_dscoder_imagenette_gcvl`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T16:45:38+02:00`
- 开始时间：`2026-06-04T16:46:59+02:00`
- 结束时间：`2026-06-04T16:47:13+02:00`
- Job ID：`2665568`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: dscoder on imagenette to gpu_computervision_long
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_dscoder_imagenette_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_dscoder_imagenette_gcvl/slurm/tunerl-rl_dscoder_imagenette_gcvl-2665568.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_dscoder_imagenette_gcvl/slurm/tunerl-rl_dscoder_imagenette_gcvl-2665568.err`
- 工作目录：`/tmp/s471802/20260604_1645_rl_dscoder_imagenette_gcvl/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_dscoder_imagenette_gcvl`
<!-- NNGPT_RUN:20260604_1645_rl_dscoder_imagenette_gcvl:END -->
## 20260604_1645_rl_mistral_cifar_10_gcvl

<!-- NNGPT_RUN:20260604_1645_rl_mistral_cifar_10_gcvl:START -->
- 运行 ID：`20260604_1645_rl_mistral_cifar_10_gcvl`
- 标签：`rl_mistral_cifar_10_gcvl`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T16:45:39+02:00`
- 开始时间：`2026-06-04T16:47:29+02:00`
- 结束时间：`2026-06-04T16:47:41+02:00`
- Job ID：`2665570`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: mistral on cifar-10 to gpu_computervision_long
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_mistral_cifar_10_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_mistral_cifar_10_gcvl/slurm/tunerl-rl_mistral_cifar_10_gcvl-2665570.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_mistral_cifar_10_gcvl/slurm/tunerl-rl_mistral_cifar_10_gcvl-2665570.err`
- 工作目录：`/tmp/s471802/20260604_1645_rl_mistral_cifar_10_gcvl/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_mistral_cifar_10_gcvl`
<!-- NNGPT_RUN:20260604_1645_rl_mistral_cifar_10_gcvl:END -->
## 20260604_1645_rl_mistral_cifar_100_gcvl

<!-- NNGPT_RUN:20260604_1645_rl_mistral_cifar_100_gcvl:START -->
- 运行 ID：`20260604_1645_rl_mistral_cifar_100_gcvl`
- 标签：`rl_mistral_cifar_100_gcvl`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T16:45:39+02:00`
- 开始时间：`2026-06-04T16:47:59+02:00`
- 结束时间：`2026-06-04T16:48:10+02:00`
- Job ID：`2665572`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: mistral on cifar-100 to gpu_computervision_long
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_mistral_cifar_100_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_mistral_cifar_100_gcvl/slurm/tunerl-rl_mistral_cifar_100_gcvl-2665572.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_mistral_cifar_100_gcvl/slurm/tunerl-rl_mistral_cifar_100_gcvl-2665572.err`
- 工作目录：`/tmp/s471802/20260604_1645_rl_mistral_cifar_100_gcvl/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_mistral_cifar_100_gcvl`
<!-- NNGPT_RUN:20260604_1645_rl_mistral_cifar_100_gcvl:END -->
## 20260604_1645_rl_mistral_imagenette_gcvl

<!-- NNGPT_RUN:20260604_1645_rl_mistral_imagenette_gcvl:START -->
- 运行 ID：`20260604_1645_rl_mistral_imagenette_gcvl`
- 标签：`rl_mistral_imagenette_gcvl`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T16:45:39+02:00`
- 开始时间：`2026-06-04T16:48:29+02:00`
- 结束时间：`2026-06-04T16:48:41+02:00`
- Job ID：`2665574`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: mistral on imagenette to gpu_computervision_long
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_mistral_imagenette_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_mistral_imagenette_gcvl/slurm/tunerl-rl_mistral_imagenette_gcvl-2665574.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_mistral_imagenette_gcvl/slurm/tunerl-rl_mistral_imagenette_gcvl-2665574.err`
- 工作目录：`/tmp/s471802/20260604_1645_rl_mistral_imagenette_gcvl/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_mistral_imagenette_gcvl`
<!-- NNGPT_RUN:20260604_1645_rl_mistral_imagenette_gcvl:END -->
## 20260604_1645_rl_qwen_cifar_10_gcvl

<!-- NNGPT_RUN:20260604_1645_rl_qwen_cifar_10_gcvl:START -->
- 运行 ID：`20260604_1645_rl_qwen_cifar_10_gcvl`
- 标签：`rl_qwen_cifar_10_gcvl`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T16:45:39+02:00`
- 开始时间：`2026-06-04T16:48:59+02:00`
- 结束时间：`2026-06-04T16:49:12+02:00`
- Job ID：`2665576`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: qwen on cifar-10 to gpu_computervision_long
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_qwen_cifar_10_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_qwen_cifar_10_gcvl/slurm/tunerl-rl_qwen_cifar_10_gcvl-2665576.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_qwen_cifar_10_gcvl/slurm/tunerl-rl_qwen_cifar_10_gcvl-2665576.err`
- 工作目录：`/tmp/s471802/20260604_1645_rl_qwen_cifar_10_gcvl/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_qwen_cifar_10_gcvl`
<!-- NNGPT_RUN:20260604_1645_rl_qwen_cifar_10_gcvl:END -->
## 20260604_1645_rl_qwen_cifar_100_gcvl

<!-- NNGPT_RUN:20260604_1645_rl_qwen_cifar_100_gcvl:START -->
- 运行 ID：`20260604_1645_rl_qwen_cifar_100_gcvl`
- 标签：`rl_qwen_cifar_100_gcvl`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T16:45:39+02:00`
- 开始时间：`2026-06-04T16:49:29+02:00`
- 结束时间：`2026-06-04T16:49:41+02:00`
- Job ID：`2665578`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: qwen on cifar-100 to gpu_computervision_long
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_qwen_cifar_100_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_qwen_cifar_100_gcvl/slurm/tunerl-rl_qwen_cifar_100_gcvl-2665578.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_qwen_cifar_100_gcvl/slurm/tunerl-rl_qwen_cifar_100_gcvl-2665578.err`
- 工作目录：`/tmp/s471802/20260604_1645_rl_qwen_cifar_100_gcvl/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_qwen_cifar_100_gcvl`
<!-- NNGPT_RUN:20260604_1645_rl_qwen_cifar_100_gcvl:END -->
## 20260604_1645_rl_qwen_imagenette_gcvl

<!-- NNGPT_RUN:20260604_1645_rl_qwen_imagenette_gcvl:START -->
- 运行 ID：`20260604_1645_rl_qwen_imagenette_gcvl`
- 标签：`rl_qwen_imagenette_gcvl`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T16:45:39+02:00`
- 开始时间：`2026-06-04T16:50:00+02:00`
- 结束时间：`2026-06-04T16:50:12+02:00`
- Job ID：`2665580`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern three-model matrix RL: qwen on imagenette to gpu_computervision_long
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_qwen_imagenette_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_qwen_imagenette_gcvl/slurm/tunerl-rl_qwen_imagenette_gcvl-2665580.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_qwen_imagenette_gcvl/slurm/tunerl-rl_qwen_imagenette_gcvl-2665580.err`
- 工作目录：`/tmp/s471802/20260604_1645_rl_qwen_imagenette_gcvl/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1645_rl_qwen_imagenette_gcvl`
<!-- NNGPT_RUN:20260604_1645_rl_qwen_imagenette_gcvl:END -->
## 20260604_1712_rl_dscoder_cifar_10_gcvl

<!-- NNGPT_RUN:20260604_1712_rl_dscoder_cifar_10_gcvl:START -->
- 运行 ID：`20260604_1712_rl_dscoder_cifar_10_gcvl`
- 标签：`rl_dscoder_cifar_10_gcvl`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T17:12:00+02:00`
- 开始时间：`2026-06-04T17:12:11+02:00`
- 结束时间：`2026-06-04T17:24:23+02:00`
- Job ID：`2665591`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on cifar-10, gcvl with RL_NN_PREFIXES and correct base model id
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_dscoder_cifar_10_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_dscoder_cifar_10_gcvl/slurm/tunerl-rl_dscoder_cifar_10_gcvl-2665591.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_dscoder_cifar_10_gcvl/slurm/tunerl-rl_dscoder_cifar_10_gcvl-2665591.err`
- 工作目录：`/tmp/s471802/20260604_1712_rl_dscoder_cifar_10_gcvl/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_dscoder_cifar_10_gcvl`
<!-- NNGPT_RUN:20260604_1712_rl_dscoder_cifar_10_gcvl:END -->
## 20260604_1712_rl_dscoder_cifar_100_gcvl

<!-- NNGPT_RUN:20260604_1712_rl_dscoder_cifar_100_gcvl:START -->
- 运行 ID：`20260604_1712_rl_dscoder_cifar_100_gcvl`
- 标签：`rl_dscoder_cifar_100_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:12:00+02:00`
- 开始时间：`2026-06-04T17:24:24+02:00`
- 结束时间：`2026-06-04T17:31:21+02:00`
- Job ID：`2665593`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on cifar-100, gcvl with RL_NN_PREFIXES and correct base model id
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_dscoder_cifar_100_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_dscoder_cifar_100_gcvl/slurm/tunerl-rl_dscoder_cifar_100_gcvl-2665593.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_dscoder_cifar_100_gcvl/slurm/tunerl-rl_dscoder_cifar_100_gcvl-2665593.err`
- 工作目录：`/tmp/s471802/20260604_1712_rl_dscoder_cifar_100_gcvl/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_dscoder_cifar_100_gcvl`
<!-- NNGPT_RUN:20260604_1712_rl_dscoder_cifar_100_gcvl:END -->
## 20260604_1712_rl_dscoder_imagenette_gcvl

<!-- NNGPT_RUN:20260604_1712_rl_dscoder_imagenette_gcvl:START -->
- 运行 ID：`20260604_1712_rl_dscoder_imagenette_gcvl`
- 标签：`rl_dscoder_imagenette_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:12:00+02:00`
- 开始时间：`2026-06-04T17:31:22+02:00`
- 结束时间：`2026-06-04T17:35:27+02:00`
- Job ID：`2665595`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on imagenette, gcvl with RL_NN_PREFIXES and correct base model id
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_dscoder_imagenette_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_dscoder_imagenette_gcvl/slurm/tunerl-rl_dscoder_imagenette_gcvl-2665595.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_dscoder_imagenette_gcvl/slurm/tunerl-rl_dscoder_imagenette_gcvl-2665595.err`
- 工作目录：`/tmp/s471802/20260604_1712_rl_dscoder_imagenette_gcvl/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_dscoder_imagenette_gcvl`
<!-- NNGPT_RUN:20260604_1712_rl_dscoder_imagenette_gcvl:END -->
## 20260604_1712_rl_mistral_cifar_10_gcvl

<!-- NNGPT_RUN:20260604_1712_rl_mistral_cifar_10_gcvl:START -->
- 运行 ID：`20260604_1712_rl_mistral_cifar_10_gcvl`
- 标签：`rl_mistral_cifar_10_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:12:01+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T17:35:23+02:00`
- Job ID：`2665597`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: mistral on cifar-10, gcvl with RL_NN_PREFIXES and correct base model id
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_mistral_cifar_10_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_mistral_cifar_10_gcvl/slurm/tunerl-rl_mistral_cifar_10_gcvl-2665597.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_mistral_cifar_10_gcvl/slurm/tunerl-rl_mistral_cifar_10_gcvl-2665597.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_mistral_cifar_10_gcvl`
<!-- NNGPT_RUN:20260604_1712_rl_mistral_cifar_10_gcvl:END -->
## 20260604_1712_rl_mistral_cifar_100_gcvl

<!-- NNGPT_RUN:20260604_1712_rl_mistral_cifar_100_gcvl:START -->
- 运行 ID：`20260604_1712_rl_mistral_cifar_100_gcvl`
- 标签：`rl_mistral_cifar_100_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:12:01+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T17:35:24+02:00`
- Job ID：`2665599`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: mistral on cifar-100, gcvl with RL_NN_PREFIXES and correct base model id
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_mistral_cifar_100_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_mistral_cifar_100_gcvl/slurm/tunerl-rl_mistral_cifar_100_gcvl-2665599.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_mistral_cifar_100_gcvl/slurm/tunerl-rl_mistral_cifar_100_gcvl-2665599.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_mistral_cifar_100_gcvl`
<!-- NNGPT_RUN:20260604_1712_rl_mistral_cifar_100_gcvl:END -->
## 20260604_1712_rl_mistral_imagenette_gcvl

<!-- NNGPT_RUN:20260604_1712_rl_mistral_imagenette_gcvl:START -->
- 运行 ID：`20260604_1712_rl_mistral_imagenette_gcvl`
- 标签：`rl_mistral_imagenette_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:12:01+02:00`
- 开始时间：`2026-06-04T17:35:30+02:00`
- 结束时间：`2026-06-04T17:40:09+02:00`
- Job ID：`2665601`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: mistral on imagenette, gcvl with RL_NN_PREFIXES and correct base model id
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_mistral_imagenette_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_mistral_imagenette_gcvl/slurm/tunerl-rl_mistral_imagenette_gcvl-2665601.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_mistral_imagenette_gcvl/slurm/tunerl-rl_mistral_imagenette_gcvl-2665601.err`
- 工作目录：`/tmp/s471802/20260604_1712_rl_mistral_imagenette_gcvl/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_mistral_imagenette_gcvl`
<!-- NNGPT_RUN:20260604_1712_rl_mistral_imagenette_gcvl:END -->
## 20260604_1712_rl_qwen_cifar_10_gcvl

<!-- NNGPT_RUN:20260604_1712_rl_qwen_cifar_10_gcvl:START -->
- 运行 ID：`20260604_1712_rl_qwen_cifar_10_gcvl`
- 标签：`rl_qwen_cifar_10_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:12:01+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T17:40:07+02:00`
- Job ID：`2665603`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: qwen on cifar-10, gcvl with RL_NN_PREFIXES and correct base model id
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_qwen_cifar_10_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_qwen_cifar_10_gcvl/slurm/tunerl-rl_qwen_cifar_10_gcvl-2665603.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_qwen_cifar_10_gcvl/slurm/tunerl-rl_qwen_cifar_10_gcvl-2665603.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_qwen_cifar_10_gcvl`
<!-- NNGPT_RUN:20260604_1712_rl_qwen_cifar_10_gcvl:END -->
## 20260604_1712_rl_qwen_cifar_100_gcvl

<!-- NNGPT_RUN:20260604_1712_rl_qwen_cifar_100_gcvl:START -->
- 运行 ID：`20260604_1712_rl_qwen_cifar_100_gcvl`
- 标签：`rl_qwen_cifar_100_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:12:01+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T17:40:07+02:00`
- Job ID：`2665605`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: qwen on cifar-100, gcvl with RL_NN_PREFIXES and correct base model id
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_qwen_cifar_100_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_qwen_cifar_100_gcvl/slurm/tunerl-rl_qwen_cifar_100_gcvl-2665605.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_qwen_cifar_100_gcvl/slurm/tunerl-rl_qwen_cifar_100_gcvl-2665605.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_qwen_cifar_100_gcvl`
<!-- NNGPT_RUN:20260604_1712_rl_qwen_cifar_100_gcvl:END -->
## 20260604_1712_rl_qwen_imagenette_gcvl

<!-- NNGPT_RUN:20260604_1712_rl_qwen_imagenette_gcvl:START -->
- 运行 ID：`20260604_1712_rl_qwen_imagenette_gcvl`
- 标签：`rl_qwen_imagenette_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:12:01+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T17:40:07+02:00`
- Job ID：`2665607`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: qwen on imagenette, gcvl with RL_NN_PREFIXES and correct base model id
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_qwen_imagenette_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_qwen_imagenette_gcvl/slurm/tunerl-rl_qwen_imagenette_gcvl-2665607.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_qwen_imagenette_gcvl/slurm/tunerl-rl_qwen_imagenette_gcvl-2665607.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1712_rl_qwen_imagenette_gcvl`
<!-- NNGPT_RUN:20260604_1712_rl_qwen_imagenette_gcvl:END -->
## 20260604_1731_rl_dscoder_cifar_100_h100

<!-- NNGPT_RUN:20260604_1731_rl_dscoder_cifar_100_h100:START -->
- 运行 ID：`20260604_1731_rl_dscoder_cifar_100_h100`
- 标签：`rl_dscoder_cifar_100_h100`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:31:15+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T17:31:57+02:00`
- Job ID：`2665675`
- 分区 / QoS：`h100`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on cifar-100, try H100
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1731_rl_dscoder_cifar_100_h100`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1731_rl_dscoder_cifar_100_h100/slurm/tunerl-rl_dscoder_cifar_100_h100-2665675.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1731_rl_dscoder_cifar_100_h100/slurm/tunerl-rl_dscoder_cifar_100_h100-2665675.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1731_rl_dscoder_cifar_100_h100`
<!-- NNGPT_RUN:20260604_1731_rl_dscoder_cifar_100_h100:END -->
## 20260604_1731_rl_dscoder_cifar_100_gcvl

<!-- NNGPT_RUN:20260604_1731_rl_dscoder_cifar_100_gcvl:START -->
- 运行 ID：`20260604_1731_rl_dscoder_cifar_100_gcvl`
- 标签：`rl_dscoder_cifar_100_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:31:56+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T17:36:28+02:00`
- Job ID：`2665677`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on cifar-100, back to gcvl
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1731_rl_dscoder_cifar_100_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1731_rl_dscoder_cifar_100_gcvl/slurm/tunerl-rl_dscoder_cifar_100_gcvl-2665677.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1731_rl_dscoder_cifar_100_gcvl/slurm/tunerl-rl_dscoder_cifar_100_gcvl-2665677.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1731_rl_dscoder_cifar_100_gcvl`
<!-- NNGPT_RUN:20260604_1731_rl_dscoder_cifar_100_gcvl:END -->
## 20260604_1735_rl_dscoder_imagenette_std

<!-- NNGPT_RUN:20260604_1735_rl_dscoder_imagenette_std:START -->
- 运行 ID：`20260604_1735_rl_dscoder_imagenette_std`
- 标签：`rl_dscoder_imagenette_std`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:35:21+02:00`
- 开始时间：`2026-06-04T17:35:32+02:00`
- 结束时间：`2026-06-04T17:40:08+02:00`
- Job ID：`2665689`
- 分区 / QoS：`standard`
- 节点：`jn002`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on imagenette, fallback to standard partition
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1735_rl_dscoder_imagenette_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1735_rl_dscoder_imagenette_std/slurm/tunerl-rl_dscoder_imagenette_std-2665689.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1735_rl_dscoder_imagenette_std/slurm/tunerl-rl_dscoder_imagenette_std-2665689.err`
- 工作目录：`/tmp/s471802/20260604_1735_rl_dscoder_imagenette_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1735_rl_dscoder_imagenette_std`
<!-- NNGPT_RUN:20260604_1735_rl_dscoder_imagenette_std:END -->
## 20260604_1735_rl_mistral_cifar_10_std

<!-- NNGPT_RUN:20260604_1735_rl_mistral_cifar_10_std:START -->
- 运行 ID：`20260604_1735_rl_mistral_cifar_10_std`
- 标签：`rl_mistral_cifar_10_std`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:35:21+02:00`
- 开始时间：`2026-06-04T17:35:34+02:00`
- 结束时间：`2026-06-04T17:40:07+02:00`
- Job ID：`2665691`
- 分区 / QoS：`standard`
- 节点：`jn102`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: mistral on cifar-10, fallback to standard partition
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1735_rl_mistral_cifar_10_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1735_rl_mistral_cifar_10_std/slurm/tunerl-rl_mistral_cifar_10_std-2665691.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1735_rl_mistral_cifar_10_std/slurm/tunerl-rl_mistral_cifar_10_std-2665691.err`
- 工作目录：`/tmp/s471802/20260604_1735_rl_mistral_cifar_10_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1735_rl_mistral_cifar_10_std`
<!-- NNGPT_RUN:20260604_1735_rl_mistral_cifar_10_std:END -->
## 20260604_1735_rl_mistral_cifar_100_std

<!-- NNGPT_RUN:20260604_1735_rl_mistral_cifar_100_std:START -->
- 运行 ID：`20260604_1735_rl_mistral_cifar_100_std`
- 标签：`rl_mistral_cifar_100_std`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:35:22+02:00`
- 开始时间：`2026-06-04T17:35:32+02:00`
- 结束时间：`2026-06-04T17:40:06+02:00`
- Job ID：`2665693`
- 分区 / QoS：`standard`
- 节点：`jnfat01`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: mistral on cifar-100, fallback to standard partition
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1735_rl_mistral_cifar_100_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1735_rl_mistral_cifar_100_std/slurm/tunerl-rl_mistral_cifar_100_std-2665693.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1735_rl_mistral_cifar_100_std/slurm/tunerl-rl_mistral_cifar_100_std-2665693.err`
- 工作目录：`/tmp/s471802/20260604_1735_rl_mistral_cifar_100_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1735_rl_mistral_cifar_100_std`
<!-- NNGPT_RUN:20260604_1735_rl_mistral_cifar_100_std:END -->
## 20260604_1736_rl_dscoder_cifar_100_std

<!-- NNGPT_RUN:20260604_1736_rl_dscoder_cifar_100_std:START -->
- 运行 ID：`20260604_1736_rl_dscoder_cifar_100_std`
- 标签：`rl_dscoder_cifar_100_std`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:36:27+02:00`
- 开始时间：`2026-06-04T17:36:37+02:00`
- 结束时间：`2026-06-04T17:40:06+02:00`
- Job ID：`2665698`
- 分区 / QoS：`standard`
- 节点：`jnfat09`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on cifar-100, standard 4GPU
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1736_rl_dscoder_cifar_100_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1736_rl_dscoder_cifar_100_std/slurm/tunerl-rl_dscoder_cifar_100_std-2665698.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1736_rl_dscoder_cifar_100_std/slurm/tunerl-rl_dscoder_cifar_100_std-2665698.err`
- 工作目录：`/tmp/s471802/20260604_1736_rl_dscoder_cifar_100_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1736_rl_dscoder_cifar_100_std`
<!-- NNGPT_RUN:20260604_1736_rl_dscoder_cifar_100_std:END -->
## 20260604_1738_rl_dscoder_cifar_10_gcvl

<!-- NNGPT_RUN:20260604_1738_rl_dscoder_cifar_10_gcvl:START -->
- 运行 ID：`20260604_1738_rl_dscoder_cifar_10_gcvl`
- 标签：`rl_dscoder_cifar_10_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:38:46+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T17:40:07+02:00`
- Job ID：`2665701`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on cifar-10, exclude train GPU from reward
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1738_rl_dscoder_cifar_10_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1738_rl_dscoder_cifar_10_gcvl/slurm/tunerl-rl_dscoder_cifar_10_gcvl-2665701.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1738_rl_dscoder_cifar_10_gcvl/slurm/tunerl-rl_dscoder_cifar_10_gcvl-2665701.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1738_rl_dscoder_cifar_10_gcvl`
<!-- NNGPT_RUN:20260604_1738_rl_dscoder_cifar_10_gcvl:END -->
## 20260604_1740_rl_dscoder_cifar_10_gcvl

<!-- NNGPT_RUN:20260604_1740_rl_dscoder_cifar_10_gcvl:START -->
- 运行 ID：`20260604_1740_rl_dscoder_cifar_10_gcvl`
- 标签：`rl_dscoder_cifar_10_gcvl`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T17:40:21+02:00`
- 开始时间：`2026-06-04T17:40:24+02:00`
- 结束时间：`2026-06-04T17:57:16+02:00`
- Job ID：`2665704`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on cifar-10, exclude train GPU from reward
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_dscoder_cifar_10_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_dscoder_cifar_10_gcvl/slurm/tunerl-rl_dscoder_cifar_10_gcvl-2665704.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_dscoder_cifar_10_gcvl/slurm/tunerl-rl_dscoder_cifar_10_gcvl-2665704.err`
- 工作目录：`/tmp/s471802/20260604_1740_rl_dscoder_cifar_10_gcvl/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_dscoder_cifar_10_gcvl`
<!-- NNGPT_RUN:20260604_1740_rl_dscoder_cifar_10_gcvl:END -->
## 20260604_1740_rl_dscoder_cifar_100_gcvl

<!-- NNGPT_RUN:20260604_1740_rl_dscoder_cifar_100_gcvl:START -->
- 运行 ID：`20260604_1740_rl_dscoder_cifar_100_gcvl`
- 标签：`rl_dscoder_cifar_100_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:40:22+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T17:46:45+02:00`
- Job ID：`2665706`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on cifar-100, exclude train GPU from reward
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_dscoder_cifar_100_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_dscoder_cifar_100_gcvl/slurm/tunerl-rl_dscoder_cifar_100_gcvl-2665706.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_dscoder_cifar_100_gcvl/slurm/tunerl-rl_dscoder_cifar_100_gcvl-2665706.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_dscoder_cifar_100_gcvl`
<!-- NNGPT_RUN:20260604_1740_rl_dscoder_cifar_100_gcvl:END -->
## 20260604_1740_rl_dscoder_imagenette_gcvl

<!-- NNGPT_RUN:20260604_1740_rl_dscoder_imagenette_gcvl:START -->
- 运行 ID：`20260604_1740_rl_dscoder_imagenette_gcvl`
- 标签：`rl_dscoder_imagenette_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:40:22+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T17:46:45+02:00`
- Job ID：`2665708`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on imagenette, exclude train GPU from reward
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_dscoder_imagenette_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_dscoder_imagenette_gcvl/slurm/tunerl-rl_dscoder_imagenette_gcvl-2665708.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_dscoder_imagenette_gcvl/slurm/tunerl-rl_dscoder_imagenette_gcvl-2665708.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_dscoder_imagenette_gcvl`
<!-- NNGPT_RUN:20260604_1740_rl_dscoder_imagenette_gcvl:END -->
## 20260604_1740_rl_mistral_cifar_10_gcvl

<!-- NNGPT_RUN:20260604_1740_rl_mistral_cifar_10_gcvl:START -->
- 运行 ID：`20260604_1740_rl_mistral_cifar_10_gcvl`
- 标签：`rl_mistral_cifar_10_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:40:22+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T17:48:41+02:00`
- Job ID：`2665710`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: mistral on cifar-10, exclude train GPU from reward
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_mistral_cifar_10_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_mistral_cifar_10_gcvl/slurm/tunerl-rl_mistral_cifar_10_gcvl-2665710.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_mistral_cifar_10_gcvl/slurm/tunerl-rl_mistral_cifar_10_gcvl-2665710.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_mistral_cifar_10_gcvl`
<!-- NNGPT_RUN:20260604_1740_rl_mistral_cifar_10_gcvl:END -->
## 20260604_1740_rl_mistral_cifar_100_gcvl

<!-- NNGPT_RUN:20260604_1740_rl_mistral_cifar_100_gcvl:START -->
- 运行 ID：`20260604_1740_rl_mistral_cifar_100_gcvl`
- 标签：`rl_mistral_cifar_100_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:40:22+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T17:48:40+02:00`
- Job ID：`2665712`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: mistral on cifar-100, exclude train GPU from reward
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_mistral_cifar_100_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_mistral_cifar_100_gcvl/slurm/tunerl-rl_mistral_cifar_100_gcvl-2665712.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_mistral_cifar_100_gcvl/slurm/tunerl-rl_mistral_cifar_100_gcvl-2665712.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_mistral_cifar_100_gcvl`
<!-- NNGPT_RUN:20260604_1740_rl_mistral_cifar_100_gcvl:END -->
## 20260604_1740_rl_mistral_imagenette_gcvl

<!-- NNGPT_RUN:20260604_1740_rl_mistral_imagenette_gcvl:START -->
- 运行 ID：`20260604_1740_rl_mistral_imagenette_gcvl`
- 标签：`rl_mistral_imagenette_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:40:22+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T17:48:41+02:00`
- Job ID：`2665714`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: mistral on imagenette, exclude train GPU from reward
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_mistral_imagenette_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_mistral_imagenette_gcvl/slurm/tunerl-rl_mistral_imagenette_gcvl-2665714.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_mistral_imagenette_gcvl/slurm/tunerl-rl_mistral_imagenette_gcvl-2665714.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_mistral_imagenette_gcvl`
<!-- NNGPT_RUN:20260604_1740_rl_mistral_imagenette_gcvl:END -->
## 20260604_1740_rl_qwen_cifar_10_gcvl

<!-- NNGPT_RUN:20260604_1740_rl_qwen_cifar_10_gcvl:START -->
- 运行 ID：`20260604_1740_rl_qwen_cifar_10_gcvl`
- 标签：`rl_qwen_cifar_10_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:40:22+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T17:48:40+02:00`
- Job ID：`2665716`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: qwen on cifar-10, exclude train GPU from reward
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_qwen_cifar_10_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_qwen_cifar_10_gcvl/slurm/tunerl-rl_qwen_cifar_10_gcvl-2665716.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_qwen_cifar_10_gcvl/slurm/tunerl-rl_qwen_cifar_10_gcvl-2665716.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_qwen_cifar_10_gcvl`
<!-- NNGPT_RUN:20260604_1740_rl_qwen_cifar_10_gcvl:END -->
## 20260604_1740_rl_qwen_cifar_100_gcvl

<!-- NNGPT_RUN:20260604_1740_rl_qwen_cifar_100_gcvl:START -->
- 运行 ID：`20260604_1740_rl_qwen_cifar_100_gcvl`
- 标签：`rl_qwen_cifar_100_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:40:23+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T17:48:40+02:00`
- Job ID：`2665718`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: qwen on cifar-100, exclude train GPU from reward
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_qwen_cifar_100_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_qwen_cifar_100_gcvl/slurm/tunerl-rl_qwen_cifar_100_gcvl-2665718.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_qwen_cifar_100_gcvl/slurm/tunerl-rl_qwen_cifar_100_gcvl-2665718.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_qwen_cifar_100_gcvl`
<!-- NNGPT_RUN:20260604_1740_rl_qwen_cifar_100_gcvl:END -->
## 20260604_1740_rl_qwen_imagenette_gcvl

<!-- NNGPT_RUN:20260604_1740_rl_qwen_imagenette_gcvl:START -->
- 运行 ID：`20260604_1740_rl_qwen_imagenette_gcvl`
- 标签：`rl_qwen_imagenette_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:40:23+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T17:48:40+02:00`
- Job ID：`2665720`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: qwen on imagenette, exclude train GPU from reward
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_qwen_imagenette_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_qwen_imagenette_gcvl/slurm/tunerl-rl_qwen_imagenette_gcvl-2665720.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_qwen_imagenette_gcvl/slurm/tunerl-rl_qwen_imagenette_gcvl-2665720.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1740_rl_qwen_imagenette_gcvl`
<!-- NNGPT_RUN:20260604_1740_rl_qwen_imagenette_gcvl:END -->
## 20260604_1746_rl_dscoder_cifar_100_std

<!-- NNGPT_RUN:20260604_1746_rl_dscoder_cifar_100_std:START -->
- 运行 ID：`20260604_1746_rl_dscoder_cifar_100_std`
- 标签：`rl_dscoder_cifar_100_std`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T17:46:43+02:00`
- 开始时间：`2026-06-04T17:46:47+02:00`
- 结束时间：`2026-06-04T18:02:35+02:00`
- Job ID：`2665730`
- 分区 / QoS：`standard`
- 节点：`jnfat09`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on cifar-100, standard 4GPU with exclude train GPU
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1746_rl_dscoder_cifar_100_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1746_rl_dscoder_cifar_100_std/slurm/tunerl-rl_dscoder_cifar_100_std-2665730.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1746_rl_dscoder_cifar_100_std/slurm/tunerl-rl_dscoder_cifar_100_std-2665730.err`
- 工作目录：`/tmp/s471802/20260604_1746_rl_dscoder_cifar_100_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1746_rl_dscoder_cifar_100_std`
<!-- NNGPT_RUN:20260604_1746_rl_dscoder_cifar_100_std:END -->
## 20260604_1746_rl_dscoder_imagenette_std

<!-- NNGPT_RUN:20260604_1746_rl_dscoder_imagenette_std:START -->
- 运行 ID：`20260604_1746_rl_dscoder_imagenette_std`
- 标签：`rl_dscoder_imagenette_std`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T17:46:43+02:00`
- 开始时间：`2026-06-04T17:46:53+02:00`
- 结束时间：`2026-06-04T18:09:40+02:00`
- Job ID：`2665732`
- 分区 / QoS：`standard`
- 节点：`jnfat09`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on imagenette, standard 4GPU with exclude train GPU
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1746_rl_dscoder_imagenette_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1746_rl_dscoder_imagenette_std/slurm/tunerl-rl_dscoder_imagenette_std-2665732.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1746_rl_dscoder_imagenette_std/slurm/tunerl-rl_dscoder_imagenette_std-2665732.err`
- 工作目录：`/tmp/s471802/20260604_1746_rl_dscoder_imagenette_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1746_rl_dscoder_imagenette_std`
<!-- NNGPT_RUN:20260604_1746_rl_dscoder_imagenette_std:END -->
## 20260604_1748_rl_mistral_cifar_10_std

<!-- NNGPT_RUN:20260604_1748_rl_mistral_cifar_10_std:START -->
- 运行 ID：`20260604_1748_rl_mistral_cifar_10_std`
- 标签：`rl_mistral_cifar_10_std`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:48:39+02:00`
- 开始时间：`2026-06-04T17:48:43+02:00`
- 结束时间：`2026-06-04T21:40:27+02:00`
- Job ID：`2665735`
- 分区 / QoS：`standard`
- 节点：`jn002`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: mistral on cifar-10, standard 3GPU
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_mistral_cifar_10_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_mistral_cifar_10_std/slurm/tunerl-rl_mistral_cifar_10_std-2665735.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_mistral_cifar_10_std/slurm/tunerl-rl_mistral_cifar_10_std-2665735.err`
- 工作目录：`/tmp/s471802/20260604_1748_rl_mistral_cifar_10_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_mistral_cifar_10_std`
<!-- NNGPT_RUN:20260604_1748_rl_mistral_cifar_10_std:END -->
## 20260604_1748_rl_mistral_cifar_100_std

<!-- NNGPT_RUN:20260604_1748_rl_mistral_cifar_100_std:START -->
- 运行 ID：`20260604_1748_rl_mistral_cifar_100_std`
- 标签：`rl_mistral_cifar_100_std`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:48:39+02:00`
- 开始时间：`2026-06-04T17:48:49+02:00`
- 结束时间：`2026-06-04T21:40:27+02:00`
- Job ID：`2665737`
- 分区 / QoS：`standard`
- 节点：`jn004`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: mistral on cifar-100, standard 3GPU
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_mistral_cifar_100_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_mistral_cifar_100_std/slurm/tunerl-rl_mistral_cifar_100_std-2665737.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_mistral_cifar_100_std/slurm/tunerl-rl_mistral_cifar_100_std-2665737.err`
- 工作目录：`/tmp/s471802/20260604_1748_rl_mistral_cifar_100_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_mistral_cifar_100_std`
<!-- NNGPT_RUN:20260604_1748_rl_mistral_cifar_100_std:END -->
## 20260604_1748_rl_mistral_imagenette_std

<!-- NNGPT_RUN:20260604_1748_rl_mistral_imagenette_std:START -->
- 运行 ID：`20260604_1748_rl_mistral_imagenette_std`
- 标签：`rl_mistral_imagenette_std`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T17:48:39+02:00`
- 开始时间：`2026-06-04T17:48:49+02:00`
- 结束时间：`2026-06-04T21:40:25+02:00`
- Job ID：`2665739`
- 分区 / QoS：`standard`
- 节点：`jn011`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: mistral on imagenette, standard 3GPU
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_mistral_imagenette_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_mistral_imagenette_std/slurm/tunerl-rl_mistral_imagenette_std-2665739.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_mistral_imagenette_std/slurm/tunerl-rl_mistral_imagenette_std-2665739.err`
- 工作目录：`/tmp/s471802/20260604_1748_rl_mistral_imagenette_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_mistral_imagenette_std`
<!-- NNGPT_RUN:20260604_1748_rl_mistral_imagenette_std:END -->
## 20260604_1748_rl_qwen_cifar_10_std

<!-- NNGPT_RUN:20260604_1748_rl_qwen_cifar_10_std:START -->
- 运行 ID：`20260604_1748_rl_qwen_cifar_10_std`
- 标签：`rl_qwen_cifar_10_std`
- 状态：已手动停止(超过1000样本)
- 提交时间：`2026-06-04T17:48:39+02:00`
- 开始时间：`2026-06-04T17:48:44+02:00`
- 结束时间：`2026-06-05T09:37:43+02:00`
- Job ID：`2665741`
- 分区 / QoS：`standard`
- 节点：`jn102`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: qwen on cifar-10, standard 3GPU
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_qwen_cifar_10_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_qwen_cifar_10_std/slurm/tunerl-rl_qwen_cifar_10_std-2665741.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_qwen_cifar_10_std/slurm/tunerl-rl_qwen_cifar_10_std-2665741.err`
- 工作目录：`/tmp/s471802/20260604_1748_rl_qwen_cifar_10_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：停止时总样本 1192；当前停止时保存的 adapter/checkpoint 作为目标 1000 样本 adapter 使用。前 1000 formal1/test_acc 0.9071，全量 formal1 0.9083，差异很小。
- 主要缺陷：提交时漏加 max step/样本上限，实际训练超过目标 1000 样本；本次不回滚 checkpoint，直接把停止时 adapter 记作 1000 样本口径。
- 分析：超量段未造成 formal accuracy 大偏差；后续生成/横评直接使用该停止 checkpoint，不再单独截断或回找第 1000 条时刻。
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_qwen_cifar_10_std`
<!-- NNGPT_RUN:20260604_1748_rl_qwen_cifar_10_std:END -->
## 20260604_1748_rl_qwen_cifar_100_std

<!-- NNGPT_RUN:20260604_1748_rl_qwen_cifar_100_std:START -->
- 运行 ID：`20260604_1748_rl_qwen_cifar_100_std`
- 标签：`rl_qwen_cifar_100_std`
- 状态：已手动停止(超过1000样本)
- 提交时间：`2026-06-04T17:48:39+02:00`
- 开始时间：`2026-06-04T17:48:56+02:00`
- 结束时间：`2026-06-05T09:37:43+02:00`
- Job ID：`2665743`
- 分区 / QoS：`standard`
- 节点：`jn116`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: qwen on cifar-100, standard 3GPU
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_qwen_cifar_100_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_qwen_cifar_100_std/slurm/tunerl-rl_qwen_cifar_100_std-2665743.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_qwen_cifar_100_std/slurm/tunerl-rl_qwen_cifar_100_std-2665743.err`
- 工作目录：`/tmp/s471802/20260604_1748_rl_qwen_cifar_100_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：停止时总样本 1600；当前停止时保存的 adapter/checkpoint 作为目标 1000 样本 adapter 使用。前 1000 formal1 0.6178，全量 formal1 0.6666。
- 主要缺陷：提交时漏加 max step/样本上限，实际训练超过目标 1000 样本；本次不回滚 checkpoint，直接把停止时 adapter 记作 1000 样本口径。
- 分析：超量段指标更高，但本轮不再区分 1000 后训练影响；后续生成/横评直接使用该停止 checkpoint。
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_qwen_cifar_100_std`
<!-- NNGPT_RUN:20260604_1748_rl_qwen_cifar_100_std:END -->
## 20260604_1748_rl_qwen_imagenette_std

<!-- NNGPT_RUN:20260604_1748_rl_qwen_imagenette_std:START -->
- 运行 ID：`20260604_1748_rl_qwen_imagenette_std`
- 标签：`rl_qwen_imagenette_std`
- 状态：已手动停止(超过1000样本)
- 提交时间：`2026-06-04T17:48:39+02:00`
- 开始时间：`2026-06-04T17:48:55+02:00`
- 结束时间：`2026-06-05T09:37:48+02:00`
- Job ID：`2665745`
- 分区 / QoS：`standard`
- 节点：`jn118`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: qwen on imagenette, standard 3GPU
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_qwen_imagenette_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_qwen_imagenette_std/slurm/tunerl-rl_qwen_imagenette_std-2665745.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_qwen_imagenette_std/slurm/tunerl-rl_qwen_imagenette_std-2665745.err`
- 工作目录：`/tmp/s471802/20260604_1748_rl_qwen_imagenette_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：停止时总样本 1872；当前停止时保存的 adapter/checkpoint 作为目标 1000 样本 adapter 使用。前 1000 formal1 0.9907，全量 formal1 0.9930，差异很小。
- 主要缺陷：提交时漏加 max step/样本上限，实际训练超过目标 1000 样本；本次不回滚 checkpoint，直接把停止时 adapter 记作 1000 样本口径。
- 分析：超量段未造成 formal accuracy 大偏差；后续生成/横评直接使用该停止 checkpoint，不再单独截断或回找第 1000 条时刻。
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1748_rl_qwen_imagenette_std`
<!-- NNGPT_RUN:20260604_1748_rl_qwen_imagenette_std:END -->
## 20260604_1814_rl_dscoder_cifar_10_gcvl

<!-- NNGPT_RUN:20260604_1814_rl_dscoder_cifar_10_gcvl:START -->
- 运行 ID：`20260604_1814_rl_dscoder_cifar_10_gcvl`
- 标签：`rl_dscoder_cifar_10_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T18:14:18+02:00`
- 开始时间：`2026-06-04T18:14:31+02:00`
- 结束时间：`2026-06-04T20:57:55+02:00`
- Job ID：`2665757`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on cifar-10, expandable_segments to fix frag OOM
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1814_rl_dscoder_cifar_10_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1814_rl_dscoder_cifar_10_gcvl/slurm/tunerl-rl_dscoder_cifar_10_gcvl-2665757.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1814_rl_dscoder_cifar_10_gcvl/slurm/tunerl-rl_dscoder_cifar_10_gcvl-2665757.err`
- 工作目录：`/tmp/s471802/20260604_1814_rl_dscoder_cifar_10_gcvl/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1814_rl_dscoder_cifar_10_gcvl`
<!-- NNGPT_RUN:20260604_1814_rl_dscoder_cifar_10_gcvl:END -->
## 20260604_1814_rl_dscoder_cifar_100_gcvl

<!-- NNGPT_RUN:20260604_1814_rl_dscoder_cifar_100_gcvl:START -->
- 运行 ID：`20260604_1814_rl_dscoder_cifar_100_gcvl`
- 标签：`rl_dscoder_cifar_100_gcvl`
- 状态：已提交
- 提交时间：`2026-06-04T18:14:18+02:00`
- 开始时间：-
- 结束时间：-
- Job ID：`2665759`
- 分区 / QoS：`gpu_computervision_long`
- 节点：-
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on cifar-100, expandable_segments to fix frag OOM
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1814_rl_dscoder_cifar_100_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1814_rl_dscoder_cifar_100_gcvl/slurm/tunerl-rl_dscoder_cifar_100_gcvl-2665759.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1814_rl_dscoder_cifar_100_gcvl/slurm/tunerl-rl_dscoder_cifar_100_gcvl-2665759.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1814_rl_dscoder_cifar_100_gcvl`
<!-- NNGPT_RUN:20260604_1814_rl_dscoder_cifar_100_gcvl:END -->
## 20260604_1814_rl_dscoder_imagenette_gcvl

<!-- NNGPT_RUN:20260604_1814_rl_dscoder_imagenette_gcvl:START -->
- 运行 ID：`20260604_1814_rl_dscoder_imagenette_gcvl`
- 标签：`rl_dscoder_imagenette_gcvl`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T18:14:18+02:00`
- 开始时间：-
- 结束时间：`2026-06-04T18:29:13+02:00`
- Job ID：`2665761`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`None assigned`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on imagenette, expandable_segments to fix frag OOM
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1814_rl_dscoder_imagenette_gcvl`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1814_rl_dscoder_imagenette_gcvl/slurm/tunerl-rl_dscoder_imagenette_gcvl-2665761.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1814_rl_dscoder_imagenette_gcvl/slurm/tunerl-rl_dscoder_imagenette_gcvl-2665761.err`
- 工作目录：-
- 初始恢复来源：-
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1814_rl_dscoder_imagenette_gcvl`
<!-- NNGPT_RUN:20260604_1814_rl_dscoder_imagenette_gcvl:END -->
## 20260604_1829_rl_dscoder_cifar_100_std

<!-- NNGPT_RUN:20260604_1829_rl_dscoder_cifar_100_std:START -->
- 运行 ID：`20260604_1829_rl_dscoder_cifar_100_std`
- 标签：`rl_dscoder_cifar_100_std`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T18:29:11+02:00`
- 开始时间：`2026-06-04T18:29:22+02:00`
- 结束时间：`2026-06-04T20:58:00+02:00`
- Job ID：`2665770`
- 分区 / QoS：`standard`
- 节点：`jn019`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on cifar-100, standard L40 with expandable_segments
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1829_rl_dscoder_cifar_100_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1829_rl_dscoder_cifar_100_std/slurm/tunerl-rl_dscoder_cifar_100_std-2665770.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1829_rl_dscoder_cifar_100_std/slurm/tunerl-rl_dscoder_cifar_100_std-2665770.err`
- 工作目录：`/tmp/s471802/20260604_1829_rl_dscoder_cifar_100_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1829_rl_dscoder_cifar_100_std`
<!-- NNGPT_RUN:20260604_1829_rl_dscoder_cifar_100_std:END -->
## 20260604_1829_rl_dscoder_imagenette_std

<!-- NNGPT_RUN:20260604_1829_rl_dscoder_imagenette_std:START -->
- 运行 ID：`20260604_1829_rl_dscoder_imagenette_std`
- 标签：`rl_dscoder_imagenette_std`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T18:29:11+02:00`
- 开始时间：`2026-06-04T18:29:29+02:00`
- 结束时间：`2026-06-04T20:57:58+02:00`
- Job ID：`2665772`
- 分区 / QoS：`standard`
- 节点：`jn101`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on imagenette, standard L40 with expandable_segments
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1829_rl_dscoder_imagenette_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1829_rl_dscoder_imagenette_std/slurm/tunerl-rl_dscoder_imagenette_std-2665772.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1829_rl_dscoder_imagenette_std/slurm/tunerl-rl_dscoder_imagenette_std-2665772.err`
- 工作目录：`/tmp/s471802/20260604_1829_rl_dscoder_imagenette_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_1829_rl_dscoder_imagenette_std`
<!-- NNGPT_RUN:20260604_1829_rl_dscoder_imagenette_std:END -->
## 20260604_2057_rl_dscoder_cifar_10_std

<!-- NNGPT_RUN:20260604_2057_rl_dscoder_cifar_10_std:START -->
- 运行 ID：`20260604_2057_rl_dscoder_cifar_10_std`
- 标签：`rl_dscoder_cifar_10_std`
- 状态：已手动停止(超过1000样本)
- 提交时间：`2026-06-04T20:57:51+02:00`
- 开始时间：`2026-06-04T20:58:02+02:00`
- 结束时间：`2026-06-05T09:37:44+02:00`
- Job ID：`2665839`
- 分区 / QoS：`standard`
- 节点：`jn009`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on cifar-10, fix to original deepseek-coder model
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2057_rl_dscoder_cifar_10_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2057_rl_dscoder_cifar_10_std/slurm/tunerl-rl_dscoder_cifar_10_std-2665839.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2057_rl_dscoder_cifar_10_std/slurm/tunerl-rl_dscoder_cifar_10_std-2665839.err`
- 工作目录：`/tmp/s471802/20260604_2057_rl_dscoder_cifar_10_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：停止时总样本 1312；当前停止时保存的 adapter/checkpoint 作为目标 1000 样本 adapter 使用。前 1000 formal1 0.9042，全量 formal1 0.9065，差异很小。
- 主要缺陷：提交时漏加 max step/样本上限，实际训练超过目标 1000 样本；本次不回滚 checkpoint，直接把停止时 adapter 记作 1000 样本口径。
- 分析：超量段未造成 formal accuracy 大偏差；后续生成/横评直接使用该停止 checkpoint，不再单独截断或回找第 1000 条时刻。
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2057_rl_dscoder_cifar_10_std`
<!-- NNGPT_RUN:20260604_2057_rl_dscoder_cifar_10_std:END -->
## 20260604_2057_rl_dscoder_cifar_100_std

<!-- NNGPT_RUN:20260604_2057_rl_dscoder_cifar_100_std:START -->
- 运行 ID：`20260604_2057_rl_dscoder_cifar_100_std`
- 标签：`rl_dscoder_cifar_100_std`
- 状态：已手动停止(超过1000样本)
- 提交时间：`2026-06-04T20:57:51+02:00`
- 开始时间：`2026-06-04T20:58:04+02:00`
- 结束时间：`2026-06-05T09:37:47+02:00`
- Job ID：`2665841`
- 分区 / QoS：`standard`
- 节点：`jn117`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on cifar-100, fix to original deepseek-coder model
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2057_rl_dscoder_cifar_100_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2057_rl_dscoder_cifar_100_std/slurm/tunerl-rl_dscoder_cifar_100_std-2665841.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2057_rl_dscoder_cifar_100_std/slurm/tunerl-rl_dscoder_cifar_100_std-2665841.err`
- 工作目录：`/tmp/s471802/20260604_2057_rl_dscoder_cifar_100_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：停止时总样本 1112；当前停止时保存的 adapter/checkpoint 作为目标 1000 样本 adapter 使用。前 1000 formal1/test_acc 0.7463，全量 formal1 0.7473，差异很小。
- 主要缺陷：提交时漏加 max step/样本上限，实际训练超过目标 1000 样本；本次不回滚 checkpoint，直接把停止时 adapter 记作 1000 样本口径。
- 分析：超量段未造成 formal accuracy 大偏差；后续生成/横评直接使用该停止 checkpoint，不再单独截断或回找第 1000 条时刻。
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2057_rl_dscoder_cifar_100_std`
<!-- NNGPT_RUN:20260604_2057_rl_dscoder_cifar_100_std:END -->
## 20260604_2057_rl_dscoder_imagenette_std

<!-- NNGPT_RUN:20260604_2057_rl_dscoder_imagenette_std:START -->
- 运行 ID：`20260604_2057_rl_dscoder_imagenette_std`
- 标签：`rl_dscoder_imagenette_std`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T20:57:51+02:00`
- 开始时间：`2026-06-04T20:58:02+02:00`
- 结束时间：`2026-06-04T23:39:48+02:00`
- Job ID：`2665843`
- 分区 / QoS：`standard`
- 节点：`jnfat01`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: dscoder on imagenette, fix to original deepseek-coder model
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2057_rl_dscoder_imagenette_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2057_rl_dscoder_imagenette_std/slurm/tunerl-rl_dscoder_imagenette_std-2665843.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2057_rl_dscoder_imagenette_std/slurm/tunerl-rl_dscoder_imagenette_std-2665843.err`
- 工作目录：`/tmp/s471802/20260604_2057_rl_dscoder_imagenette_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2057_rl_dscoder_imagenette_std`
<!-- NNGPT_RUN:20260604_2057_rl_dscoder_imagenette_std:END -->
## 20260604_2140_rl_mistral_a7_cifar_10_std

<!-- NNGPT_RUN:20260604_2140_rl_mistral_a7_cifar_10_std:START -->
- 运行 ID：`20260604_2140_rl_mistral_a7_cifar_10_std`
- 标签：`rl_mistral_a7_cifar_10_std`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T21:40:22+02:00`
- 开始时间：`2026-06-04T21:40:39+02:00`
- 结束时间：`2026-06-04T23:00:56+02:00`
- Job ID：`2665858`
- 分区 / QoS：`standard`
- 节点：`jn101`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: mistral A7 on cifar-10, formal 4gen
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2140_rl_mistral_a7_cifar_10_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2140_rl_mistral_a7_cifar_10_std/slurm/tunerl-rl_mistral_a7_cifar_10_std-2665858.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2140_rl_mistral_a7_cifar_10_std/slurm/tunerl-rl_mistral_a7_cifar_10_std-2665858.err`
- 工作目录：`/tmp/s471802/20260604_2140_rl_mistral_a7_cifar_10_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2140_rl_mistral_a7_cifar_10_std`
<!-- NNGPT_RUN:20260604_2140_rl_mistral_a7_cifar_10_std:END -->
## 20260604_2140_rl_mistral_a7_cifar_100_std

<!-- NNGPT_RUN:20260604_2140_rl_mistral_a7_cifar_100_std:START -->
- 运行 ID：`20260604_2140_rl_mistral_a7_cifar_100_std`
- 标签：`rl_mistral_a7_cifar_100_std`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T21:40:22+02:00`
- 开始时间：`2026-06-04T21:40:38+02:00`
- 结束时间：`2026-06-04T23:00:56+02:00`
- Job ID：`2665860`
- 分区 / QoS：`standard`
- 节点：`jn119`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: mistral A7 on cifar-100, formal 4gen
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2140_rl_mistral_a7_cifar_100_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2140_rl_mistral_a7_cifar_100_std/slurm/tunerl-rl_mistral_a7_cifar_100_std-2665860.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2140_rl_mistral_a7_cifar_100_std/slurm/tunerl-rl_mistral_a7_cifar_100_std-2665860.err`
- 工作目录：`/tmp/s471802/20260604_2140_rl_mistral_a7_cifar_100_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2140_rl_mistral_a7_cifar_100_std`
<!-- NNGPT_RUN:20260604_2140_rl_mistral_a7_cifar_100_std:END -->
## 20260604_2140_rl_mistral_a7_imagenette_std

<!-- NNGPT_RUN:20260604_2140_rl_mistral_a7_imagenette_std:START -->
- 运行 ID：`20260604_2140_rl_mistral_a7_imagenette_std`
- 标签：`rl_mistral_a7_imagenette_std`
- 状态：已结束(CANCELLED by 221940)
- 提交时间：`2026-06-04T21:40:22+02:00`
- 开始时间：`2026-06-04T21:40:39+02:00`
- 结束时间：`2026-06-04T23:00:55+02:00`
- Job ID：`2665862`
- 分区 / QoS：`standard`
- 节点：`jn120`
- 提交 commit：`f3c397f005de18a8514e58d08b53eacfd3687a5e Symmetrize warmup reward for non-trainable candidates`
- 本次改动：1pattern matrix RL: mistral A7 on imagenette, formal 4gen
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2140_rl_mistral_a7_imagenette_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2140_rl_mistral_a7_imagenette_std/slurm/tunerl-rl_mistral_a7_imagenette_std-2665862.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2140_rl_mistral_a7_imagenette_std/slurm/tunerl-rl_mistral_a7_imagenette_std-2665862.err`
- 工作目录：`/tmp/s471802/20260604_2140_rl_mistral_a7_imagenette_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2140_rl_mistral_a7_imagenette_std`
<!-- NNGPT_RUN:20260604_2140_rl_mistral_a7_imagenette_std:END -->
## 20260604_2300_rl_mistral_a7_cifar_10_std

<!-- NNGPT_RUN:20260604_2300_rl_mistral_a7_cifar_10_std:START -->
- 运行 ID：`20260604_2300_rl_mistral_a7_cifar_10_std`
- 标签：`rl_mistral_a7_cifar_10_std`
- 状态：运行中
- 提交时间：`2026-06-04T23:00:50+02:00`
- 开始时间：`2026-06-04T23:01:01+02:00`
- 结束时间：-
- Job ID：`2665890`
- 分区 / QoS：`standard`
- 节点：`jn002`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：1pattern RL: mistral A7 on cifar-10, warmup penalty scaled 0.2x
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2300_rl_mistral_a7_cifar_10_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2300_rl_mistral_a7_cifar_10_std/slurm/tunerl-rl_mistral_a7_cifar_10_std-2665890.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2300_rl_mistral_a7_cifar_10_std/slurm/tunerl-rl_mistral_a7_cifar_10_std-2665890.err`
- 工作目录：`/tmp/s471802/20260604_2300_rl_mistral_a7_cifar_10_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2300_rl_mistral_a7_cifar_10_std`
<!-- NNGPT_RUN:20260604_2300_rl_mistral_a7_cifar_10_std:END -->
## 20260604_2300_rl_mistral_a7_cifar_100_std

<!-- NNGPT_RUN:20260604_2300_rl_mistral_a7_cifar_100_std:START -->
- 运行 ID：`20260604_2300_rl_mistral_a7_cifar_100_std`
- 标签：`rl_mistral_a7_cifar_100_std`
- 状态：运行中
- 提交时间：`2026-06-04T23:00:50+02:00`
- 开始时间：`2026-06-04T23:01:01+02:00`
- 结束时间：-
- Job ID：`2665892`
- 分区 / QoS：`standard`
- 节点：`jn011`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：1pattern RL: mistral A7 on cifar-100, warmup penalty scaled 0.2x
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2300_rl_mistral_a7_cifar_100_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2300_rl_mistral_a7_cifar_100_std/slurm/tunerl-rl_mistral_a7_cifar_100_std-2665892.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2300_rl_mistral_a7_cifar_100_std/slurm/tunerl-rl_mistral_a7_cifar_100_std-2665892.err`
- 工作目录：`/tmp/s471802/20260604_2300_rl_mistral_a7_cifar_100_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2300_rl_mistral_a7_cifar_100_std`
<!-- NNGPT_RUN:20260604_2300_rl_mistral_a7_cifar_100_std:END -->
## 20260604_2300_rl_mistral_a7_imagenette_std

<!-- NNGPT_RUN:20260604_2300_rl_mistral_a7_imagenette_std:START -->
- 运行 ID：`20260604_2300_rl_mistral_a7_imagenette_std`
- 标签：`rl_mistral_a7_imagenette_std`
- 状态：运行中
- 提交时间：`2026-06-04T23:00:51+02:00`
- 开始时间：`2026-06-04T23:01:01+02:00`
- 结束时间：-
- Job ID：`2665894`
- 分区 / QoS：`standard`
- 节点：`jn019`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：1pattern RL: mistral A7 on imagenette, warmup penalty scaled 0.2x
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2300_rl_mistral_a7_imagenette_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2300_rl_mistral_a7_imagenette_std/slurm/tunerl-rl_mistral_a7_imagenette_std-2665894.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2300_rl_mistral_a7_imagenette_std/slurm/tunerl-rl_mistral_a7_imagenette_std-2665894.err`
- 工作目录：`/tmp/s471802/20260604_2300_rl_mistral_a7_imagenette_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2300_rl_mistral_a7_imagenette_std`
<!-- NNGPT_RUN:20260604_2300_rl_mistral_a7_imagenette_std:END -->
## 20260604_2349_mistral_a7_cifar_10_kl04

<!-- NNGPT_RUN:20260604_2349_mistral_a7_cifar_10_kl04:START -->
- 运行 ID：`20260604_2349_mistral_a7_cifar_10_kl04`
- 标签：`mistral_a7_cifar_10_kl04`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T23:49:49+02:00`
- 开始时间：`2026-06-04T23:49:54+02:00`
- 结束时间：`2026-06-04T23:50:07+02:00`
- Job ID：`2665917`
- 分区 / QoS：`standard`
- 节点：`jn002`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：MS A7 RL on cifar_10, KL beta=0.04 to prevent format collapse
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2349_mistral_a7_cifar_10_kl04`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2349_mistral_a7_cifar_10_kl04/slurm/tunerl-mistral_a7_cifar_10_kl04-2665917.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2349_mistral_a7_cifar_10_kl04/slurm/tunerl-mistral_a7_cifar_10_kl04-2665917.err`
- 工作目录：`/tmp/s471802/20260604_2349_mistral_a7_cifar_10_kl04/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2349_mistral_a7_cifar_10_kl04`
<!-- NNGPT_RUN:20260604_2349_mistral_a7_cifar_10_kl04:END -->
## 20260604_2349_mistral_a7_cifar_100_kl04

<!-- NNGPT_RUN:20260604_2349_mistral_a7_cifar_100_kl04:START -->
- 运行 ID：`20260604_2349_mistral_a7_cifar_100_kl04`
- 标签：`mistral_a7_cifar_100_kl04`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-04T23:49:49+02:00`
- 开始时间：`2026-06-04T23:49:53+02:00`
- 结束时间：`2026-06-04T23:50:07+02:00`
- Job ID：`2665919`
- 分区 / QoS：`standard`
- 节点：`jn019`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：MS A7 RL on cifar_100, KL beta=0.04 to prevent format collapse
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2349_mistral_a7_cifar_100_kl04`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2349_mistral_a7_cifar_100_kl04/slurm/tunerl-mistral_a7_cifar_100_kl04-2665919.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2349_mistral_a7_cifar_100_kl04/slurm/tunerl-mistral_a7_cifar_100_kl04-2665919.err`
- 工作目录：`/tmp/s471802/20260604_2349_mistral_a7_cifar_100_kl04/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2349_mistral_a7_cifar_100_kl04`
<!-- NNGPT_RUN:20260604_2349_mistral_a7_cifar_100_kl04:END -->
## 20260604_2349_mistral_a7_imagenette_kl04

<!-- NNGPT_RUN:20260604_2349_mistral_a7_imagenette_kl04:START -->
- 运行 ID：`20260604_2349_mistral_a7_imagenette_kl04`
- 标签：`mistral_a7_imagenette_kl04`
- 状态：运行中
- 提交时间：`2026-06-04T23:49:49+02:00`
- 开始时间：`2026-06-04T23:50:06+02:00`
- 结束时间：-
- Job ID：`2665921`
- 分区 / QoS：`standard`
- 节点：`jn101`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：MS A7 RL on imagenette, KL beta=0.04 to prevent format collapse
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2349_mistral_a7_imagenette_kl04`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2349_mistral_a7_imagenette_kl04/slurm/tunerl-mistral_a7_imagenette_kl04-2665921.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2349_mistral_a7_imagenette_kl04/slurm/tunerl-mistral_a7_imagenette_kl04-2665921.err`
- 工作目录：`/tmp/s471802/20260604_2349_mistral_a7_imagenette_kl04/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2349_mistral_a7_imagenette_kl04`
<!-- NNGPT_RUN:20260604_2349_mistral_a7_imagenette_kl04:END -->
## 20260604_2350_dscoder_imagenette_std

<!-- NNGPT_RUN:20260604_2350_dscoder_imagenette_std:START -->
- 运行 ID：`20260604_2350_dscoder_imagenette_std`
- 标签：`dscoder_imagenette_std`
- 状态：已结束(OUT_OF_MEMORY)
- 提交时间：`2026-06-04T23:50:34+02:00`
- 开始时间：`2026-06-04T23:50:38+02:00`
- 结束时间：`2026-06-05T01:38:15+02:00`
- Job ID：`2665923`
- 分区 / QoS：`standard`
- 节点：`jn002`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：DS A9 RL on imagenette, resume after crash
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2350_dscoder_imagenette_std`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2350_dscoder_imagenette_std/slurm/tunerl-dscoder_imagenette_std-2665923.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2350_dscoder_imagenette_std/slurm/tunerl-dscoder_imagenette_std-2665923.err`
- 工作目录：`/tmp/s471802/20260604_2350_dscoder_imagenette_std/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2350_dscoder_imagenette_std`
<!-- NNGPT_RUN:20260604_2350_dscoder_imagenette_std:END -->
## 20260604_2355_mistral_a7_cifar_10_kl04

<!-- NNGPT_RUN:20260604_2355_mistral_a7_cifar_10_kl04:START -->
- 运行 ID：`20260604_2355_mistral_a7_cifar_10_kl04`
- 标签：`mistral_a7_cifar_10_kl04`
- 状态：运行中
- 提交时间：`2026-06-04T23:55:38+02:00`
- 开始时间：`2026-06-04T23:55:43+02:00`
- 结束时间：-
- Job ID：`2665929`
- 分区 / QoS：`standard`
- 节点：`jn019`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：MS A7 RL on cifar-10, KL beta=0.04
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2355_mistral_a7_cifar_10_kl04`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2355_mistral_a7_cifar_10_kl04/slurm/tunerl-mistral_a7_cifar_10_kl04-2665929.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2355_mistral_a7_cifar_10_kl04/slurm/tunerl-mistral_a7_cifar_10_kl04-2665929.err`
- 工作目录：`/tmp/s471802/20260604_2355_mistral_a7_cifar_10_kl04/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2355_mistral_a7_cifar_10_kl04`
<!-- NNGPT_RUN:20260604_2355_mistral_a7_cifar_10_kl04:END -->
## 20260604_2355_mistral_a7_cifar_100_kl04

<!-- NNGPT_RUN:20260604_2355_mistral_a7_cifar_100_kl04:START -->
- 运行 ID：`20260604_2355_mistral_a7_cifar_100_kl04`
- 标签：`mistral_a7_cifar_100_kl04`
- 状态：运行中
- 提交时间：`2026-06-04T23:55:38+02:00`
- 开始时间：`2026-06-04T23:55:49+02:00`
- 结束时间：-
- Job ID：`2665931`
- 分区 / QoS：`standard`
- 节点：`jnfat01`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：MS A7 RL on cifar-100, KL beta=0.04
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2355_mistral_a7_cifar_100_kl04`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2355_mistral_a7_cifar_100_kl04/slurm/tunerl-mistral_a7_cifar_100_kl04-2665931.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2355_mistral_a7_cifar_100_kl04/slurm/tunerl-mistral_a7_cifar_100_kl04-2665931.err`
- 工作目录：`/tmp/s471802/20260604_2355_mistral_a7_cifar_100_kl04/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260604_2355_mistral_a7_cifar_100_kl04`
<!-- NNGPT_RUN:20260604_2355_mistral_a7_cifar_100_kl04:END -->
## 20260605_0033_mistral_a7_cifar_10_kl08

<!-- NNGPT_RUN:20260605_0033_mistral_a7_cifar_10_kl08:START -->
- 运行 ID：`20260605_0033_mistral_a7_cifar_10_kl08`
- 标签：`mistral_a7_cifar_10_kl08`
- 状态：已结束(FAILED)
- 提交时间：`2026-06-05T00:33:41+02:00`
- 开始时间：`2026-06-05T00:33:52+02:00`
- 结束时间：`2026-06-05T13:35:37+02:00`
- Job ID：`2665950`
- 分区 / QoS：`standard`
- 节点：`jn019`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：MS A7 RL on cifar-10, KL beta=0.08
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_0033_mistral_a7_cifar_10_kl08`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_0033_mistral_a7_cifar_10_kl08/slurm/tunerl-mistral_a7_cifar_10_kl08-2665950.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_0033_mistral_a7_cifar_10_kl08/slurm/tunerl-mistral_a7_cifar_10_kl08-2665950.err`
- 工作目录：`/tmp/s471802/20260605_0033_mistral_a7_cifar_10_kl08/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：人为按 1000 样本口径停止；实际 generation_samples.jsonl 为 1112 条。last-100 reward 0.3293，formal/executable/forward/backward 成功率均约 0.90，但 declared_pattern_matches_prompt 仅 0.01，actual_block_live 为 0.0。
- 主要缺陷：Slurm 状态 FAILED 是 2026-06-05 手动 SIGTERM 停止造成，不是训练崩溃；结构上已经退化为无 live block，不能按健康 1-pattern adapter 解读。
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_0033_mistral_a7_cifar_10_kl08`
<!-- NNGPT_RUN:20260605_0033_mistral_a7_cifar_10_kl08:END -->
## 20260605_0033_mistral_a7_cifar_100_kl08

<!-- NNGPT_RUN:20260605_0033_mistral_a7_cifar_100_kl08:START -->
- 运行 ID：`20260605_0033_mistral_a7_cifar_100_kl08`
- 标签：`mistral_a7_cifar_100_kl08`
- 状态：已结束(OUT_OF_MEMORY)
- 提交时间：`2026-06-05T00:33:41+02:00`
- 开始时间：`2026-06-05T00:33:46+02:00`
- 结束时间：`2026-06-05T07:30:55+02:00`
- Job ID：`2665952`
- 分区 / QoS：`standard`
- 节点：`jn101`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：MS A7 RL on cifar-100, KL beta=0.08
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_0033_mistral_a7_cifar_100_kl08`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_0033_mistral_a7_cifar_100_kl08/slurm/tunerl-mistral_a7_cifar_100_kl08-2665952.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_0033_mistral_a7_cifar_100_kl08/slurm/tunerl-mistral_a7_cifar_100_kl08-2665952.err`
- 工作目录：`/tmp/s471802/20260605_0033_mistral_a7_cifar_100_kl08/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_0033_mistral_a7_cifar_100_kl08`
<!-- NNGPT_RUN:20260605_0033_mistral_a7_cifar_100_kl08:END -->
## 20260605_0033_mistral_a7_imagenette_kl08

<!-- NNGPT_RUN:20260605_0033_mistral_a7_imagenette_kl08:START -->
- 运行 ID：`20260605_0033_mistral_a7_imagenette_kl08`
- 标签：`mistral_a7_imagenette_kl08`
- 状态：已结束(OUT_OF_MEMORY)
- 提交时间：`2026-06-05T00:33:41+02:00`
- 开始时间：`2026-06-05T00:33:45+02:00`
- 结束时间：`2026-06-05T08:08:30+02:00`
- Job ID：`2665954`
- 分区 / QoS：`standard`
- 节点：`jnfat01`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：MS A7 RL on imagenette, KL beta=0.08
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_0033_mistral_a7_imagenette_kl08`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_0033_mistral_a7_imagenette_kl08/slurm/tunerl-mistral_a7_imagenette_kl08-2665954.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_0033_mistral_a7_imagenette_kl08/slurm/tunerl-mistral_a7_imagenette_kl08-2665954.err`
- 工作目录：`/tmp/s471802/20260605_0033_mistral_a7_imagenette_kl08/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_0033_mistral_a7_imagenette_kl08`
<!-- NNGPT_RUN:20260605_0033_mistral_a7_imagenette_kl08:END -->
## 20260605_0925_dscoder_imagenette_h100

<!-- NNGPT_RUN:20260605_0925_dscoder_imagenette_h100:START -->
- 运行 ID：`20260605_0925_dscoder_imagenette_h100`
- 标签：`dscoder_imagenette_h100`
- 状态：已结束
- 提交时间：`2026-06-05T09:25:20+02:00`
- 开始时间：`2026-06-05T11:40:38+02:00`
- 结束时间：`2026-06-05T17:08:50+02:00`
- Job ID：`2666209`
- 分区 / QoS：`h100`
- 节点：`jnultra01`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：DS A9 RL on imagenette, retry on H100 after L40 OOM
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_0925_dscoder_imagenette_h100`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_0925_dscoder_imagenette_h100/slurm/tunerl-dscoder_imagenette_h100-2666209.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_0925_dscoder_imagenette_h100/slurm/tunerl-dscoder_imagenette_h100-2666209.err`
- 工作目录：`/tmp/s471802/20260605_0925_dscoder_imagenette_h100/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_0925_dscoder_imagenette_h100`
<!-- NNGPT_RUN:20260605_0925_dscoder_imagenette_h100:END -->

## 20260605_granite_1pattern_sft10_pipe0_granite4_1_8b

<!-- NNGPT_RUN:20260605_granite_1pattern_sft10_pipe0_granite4_1_8b:START -->
- 运行 ID：`20260605_granite_1pattern_sft10_pipe0_granite4_1_8b`
- 标签：`granite_1pattern_sft10_pipe0_granite4_1_8b`
- 状态：运行中
- 提交时间：`2026-06-05T21:04:48+02:00`
- 开始时间：`2026-06-05T21:04:48+02:00`
- 结束时间：-
- Job ID：`2666708`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat04`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：Granite 4.1 8B 1-pattern formal SFT, `NNGPT_FORCE_DIRECT_GENERATE=0`, 10 cycles, 30 generations per cycle, SFT dataset cap 1000.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_granite_1pattern_sft10_pipe0_granite4_1_8b`
- Epoch root：`/home/s471802/nn-gpt/out/nngpt/llm/20260605_granite_1pattern_sft10_pipe0_granite4_1_8b/epoch_sft`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_granite_1pattern_sft10_pipe0_granite4_1_8b/slurm/sft10-granite-pipe0-2666708.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_granite_1pattern_sft10_pipe0_granite4_1_8b/slurm/sft10-granite-pipe0-2666708.err`
- 训练结果：待手写
- 主要缺陷：待手写
- 备注：首次 H100 job `2666707` 因资源预计启动时间为 `2026-06-06T11:27:23+02:00`，超过 2 小时，未开始训练即取消，不作为正式 run 保留。
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_granite_1pattern_sft10_pipe0_granite4_1_8b`
<!-- NNGPT_RUN:20260605_granite_1pattern_sft10_pipe0_granite4_1_8b:END -->

## 20260605_olympiccoder_1pattern_sft10_pipe0_no_think_template

<!-- NNGPT_RUN:20260605_olympiccoder_1pattern_sft10_pipe0_no_think_template:START -->
- 运行 ID：`20260605_olympiccoder_1pattern_sft10_pipe0_no_think_template`
- 标签：`olympiccoder_1pattern_sft10_pipe0_no_think_template`
- 状态：运行中
- 提交时间：`2026-06-05T22:14:35+02:00`
- 开始时间：-
- 结束时间：-
- Job ID：`2666897`
- 分区 / QoS：`gpu_computervision_long`
- 节点：-
- 提交 commit：`c660a1e0e57747063321cbb98285e9b4ac926ef6 Support configured chat template overrides`
- 本次改动：OlympicCoder-7B 1-pattern formal SFT, config-level no-think chat template, `NNGPT_FORCE_DIRECT_GENERATE=0`, 10 cycles, 30 generations per cycle, SFT dataset cap 1000.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_olympiccoder_1pattern_sft10_pipe0_no_think_template`
- Epoch root：`/home/s471802/nn-gpt/out/nngpt/llm/20260605_olympiccoder_1pattern_sft10_pipe0_no_think_template/epoch_sft`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_olympiccoder_1pattern_sft10_pipe0_no_think_template/slurm/sft10-olym-template-2666897.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_olympiccoder_1pattern_sft10_pipe0_no_think_template/slurm/sft10-olym-template-2666897.err`
- 训练结果：待手写
- 主要缺陷：待手写
- 备注：正式 run 先启动，后续若早期 cycle 明确异常再取消。
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260605_olympiccoder_1pattern_sft10_pipe0_no_think_template`
<!-- NNGPT_RUN:20260605_olympiccoder_1pattern_sft10_pipe0_no_think_template:END -->
## 20260606_1606_formal100_dscoder_cifar10

<!-- NNGPT_RUN:20260606_1606_formal100_dscoder_cifar10:START -->
- 运行 ID：`20260606_1606_formal100_dscoder_cifar10`
- 标签：`formal100_dscoder_cifar10`
- 状态：已结束
- 提交时间：`2026-06-06T16:06:50+02:00`
- 开始时间：`2026-06-06T16:07:02+02:00`
- 结束时间：`2026-06-06T23:06:47+02:00`
- Job ID：`2667942`
- 分区 / QoS：`h100`
- 节点：`jnultra02`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：Formal DS/Qwen 1-pattern RL rerun: dscoder on cifar-10, max_steps=100, num_generations=8, KL=0.04, reward/live fixes, output on /data.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1606_formal100_dscoder_cifar10`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1606_formal100_dscoder_cifar10/slurm/tunerl-formal100_dscoder_cifar10-2667942.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1606_formal100_dscoder_cifar10/slurm/tunerl-formal100_dscoder_cifar10-2667942.err`
- 工作目录：`/tmp/s471802/20260606_1606_formal100_dscoder_cifar10/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1606_formal100_dscoder_cifar10`
<!-- NNGPT_RUN:20260606_1606_formal100_dscoder_cifar10:END -->
## 20260606_1611_formal100_dscoder_cifar100_l40s

<!-- NNGPT_RUN:20260606_1611_formal100_dscoder_cifar100_l40s:START -->
- 运行 ID：`20260606_1611_formal100_dscoder_cifar100_l40s`
- 标签：`formal100_dscoder_cifar100_l40s`
- 状态：已结束
- 提交时间：`2026-06-06T16:11:23+02:00`
- 开始时间：`2026-06-06T16:11:33+02:00`
- 结束时间：`2026-06-07T02:39:51+02:00`
- Job ID：`2667956`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat05`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：Formal DS/Qwen 1-pattern RL rerun: dscoder on cifar-100, max_steps=100, num_generations=8, KL=0.04, reward/live fixes, L40S fallback from long H100 queue, output on /data.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1611_formal100_dscoder_cifar100_l40s`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1611_formal100_dscoder_cifar100_l40s/slurm/tunerl-formal100_dscoder_cifar100_l40s-2667956.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1611_formal100_dscoder_cifar100_l40s/slurm/tunerl-formal100_dscoder_cifar100_l40s-2667956.err`
- 工作目录：`/tmp/s471802/20260606_1611_formal100_dscoder_cifar100_l40s/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1611_formal100_dscoder_cifar100_l40s`
<!-- NNGPT_RUN:20260606_1611_formal100_dscoder_cifar100_l40s:END -->
## 20260606_1611_formal100_dscoder_imagenette_l40s

<!-- NNGPT_RUN:20260606_1611_formal100_dscoder_imagenette_l40s:START -->
- 运行 ID：`20260606_1611_formal100_dscoder_imagenette_l40s`
- 标签：`formal100_dscoder_imagenette_l40s`
- 状态：已结束
- 提交时间：`2026-06-06T16:11:23+02:00`
- 开始时间：`2026-06-06T16:11:33+02:00`
- 结束时间：`2026-06-07T02:55:30+02:00`
- Job ID：`2667958`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat05`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：Formal DS/Qwen 1-pattern RL rerun: dscoder on imagenette, max_steps=100, num_generations=8, KL=0.04, reward/live fixes, L40S fallback from long H100 queue, output on /data.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1611_formal100_dscoder_imagenette_l40s`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1611_formal100_dscoder_imagenette_l40s/slurm/tunerl-formal100_dscoder_imagenette_l40s-2667958.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1611_formal100_dscoder_imagenette_l40s/slurm/tunerl-formal100_dscoder_imagenette_l40s-2667958.err`
- 工作目录：`/tmp/s471802/20260606_1611_formal100_dscoder_imagenette_l40s/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1611_formal100_dscoder_imagenette_l40s`
<!-- NNGPT_RUN:20260606_1611_formal100_dscoder_imagenette_l40s:END -->
## 20260606_1611_formal100_qwen_cifar10_l40s

<!-- NNGPT_RUN:20260606_1611_formal100_qwen_cifar10_l40s:START -->
- 运行 ID：`20260606_1611_formal100_qwen_cifar10_l40s`
- 标签：`formal100_qwen_cifar10_l40s`
- 状态：已结束(OUT_OF_MEMORY)
- 提交时间：`2026-06-06T16:11:23+02:00`
- 开始时间：`2026-06-06T16:11:33+02:00`
- 结束时间：`2026-06-06T18:24:14+02:00`
- Job ID：`2667960`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：Formal DS/Qwen 1-pattern RL rerun: qwen on cifar-10, max_steps=100, num_generations=8, KL=0.04, reward/live fixes, L40S fallback from long H100 queue, output on /data.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1611_formal100_qwen_cifar10_l40s`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1611_formal100_qwen_cifar10_l40s/slurm/tunerl-formal100_qwen_cifar10_l40s-2667960.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1611_formal100_qwen_cifar10_l40s/slurm/tunerl-formal100_qwen_cifar10_l40s-2667960.err`
- 工作目录：`/tmp/s471802/20260606_1611_formal100_qwen_cifar10_l40s/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1611_formal100_qwen_cifar10_l40s`
<!-- NNGPT_RUN:20260606_1611_formal100_qwen_cifar10_l40s:END -->
## 20260606_1612_formal100_qwen_cifar100_std3

<!-- NNGPT_RUN:20260606_1612_formal100_qwen_cifar100_std3:START -->
- 运行 ID：`20260606_1612_formal100_qwen_cifar100_std3`
- 标签：`formal100_qwen_cifar100_std3`
- 状态：已结束
- 提交时间：`2026-06-06T16:12:28+02:00`
- 开始时间：`2026-06-06T16:12:36+02:00`
- 结束时间：`2026-06-06T22:17:05+02:00`
- Job ID：`2667967`
- 分区 / QoS：`standard`
- 节点：`jn001`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：Formal Qwen 1-pattern RL rerun: qwen on cifar-100, max_steps=100, num_generations=8, KL=0.04, reward/live fixes, standard 3-GPU fallback because L40S/H100 queue exceeded 2h, output on /data.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1612_formal100_qwen_cifar100_std3`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1612_formal100_qwen_cifar100_std3/slurm/tunerl-formal100_qwen_cifar100_std3-2667967.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1612_formal100_qwen_cifar100_std3/slurm/tunerl-formal100_qwen_cifar100_std3-2667967.err`
- 工作目录：`/tmp/s471802/20260606_1612_formal100_qwen_cifar100_std3/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1612_formal100_qwen_cifar100_std3`
<!-- NNGPT_RUN:20260606_1612_formal100_qwen_cifar100_std3:END -->
## 20260606_1612_formal100_qwen_imagenette_std3

<!-- NNGPT_RUN:20260606_1612_formal100_qwen_imagenette_std3:START -->
- 运行 ID：`20260606_1612_formal100_qwen_imagenette_std3`
- 标签：`formal100_qwen_imagenette_std3`
- 状态：已结束(OUT_OF_MEMORY)
- 提交时间：`2026-06-06T16:12:28+02:00`
- 开始时间：`2026-06-06T16:12:36+02:00`
- 结束时间：`2026-06-06T17:11:53+02:00`
- Job ID：`2667969`
- 分区 / QoS：`standard`
- 节点：`jn006`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：Formal Qwen 1-pattern RL rerun: qwen on imagenette, max_steps=100, num_generations=8, KL=0.04, reward/live fixes, standard 3-GPU fallback because L40S/H100 queue exceeded 2h, output on /data.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1612_formal100_qwen_imagenette_std3`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1612_formal100_qwen_imagenette_std3/slurm/tunerl-formal100_qwen_imagenette_std3-2667969.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1612_formal100_qwen_imagenette_std3/slurm/tunerl-formal100_qwen_imagenette_std3-2667969.err`
- 工作目录：`/tmp/s471802/20260606_1612_formal100_qwen_imagenette_std3/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260606_1612_formal100_qwen_imagenette_std3`
<!-- NNGPT_RUN:20260606_1612_formal100_qwen_imagenette_std3:END -->
## 20260607_0120_formal100_qwen_cifar10_kl0005_l40s

<!-- NNGPT_RUN:20260607_0120_formal100_qwen_cifar10_kl0005_l40s:START -->
- 运行 ID：`20260607_0120_formal100_qwen_cifar10_kl0005_l40s`
- 标签：`formal100_qwen_cifar10_kl0005_l40s`
- 状态：已结束
- 提交时间：`2026-06-07T01:20:26+02:00`
- 开始时间：`2026-06-07T01:20:38+02:00`
- 结束时间：`2026-06-07T13:04:46+02:00`
- Job ID：`2668872`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：Formal Qwen 1-pattern RL rerun: qwen on cifar-10, max_steps=100, num_generations=8, KL=0.005, CUDA-OOM reward worker restart fix, output on /data.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0120_formal100_qwen_cifar10_kl0005_l40s`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0120_formal100_qwen_cifar10_kl0005_l40s/slurm/tunerl-formal100_qwen_cifar10_kl0005_l40s-2668872.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0120_formal100_qwen_cifar10_kl0005_l40s/slurm/tunerl-formal100_qwen_cifar10_kl0005_l40s-2668872.err`
- 工作目录：`/tmp/s471802/20260607_0120_formal100_qwen_cifar10_kl0005_l40s/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0120_formal100_qwen_cifar10_kl0005_l40s`
<!-- NNGPT_RUN:20260607_0120_formal100_qwen_cifar10_kl0005_l40s:END -->
## 20260607_0120_formal100_qwen_cifar100_kl0005_l40s

<!-- NNGPT_RUN:20260607_0120_formal100_qwen_cifar100_kl0005_l40s:START -->
- 运行 ID：`20260607_0120_formal100_qwen_cifar100_kl0005_l40s`
- 标签：`formal100_qwen_cifar100_kl0005_l40s`
- 状态：已结束
- 提交时间：`2026-06-07T01:20:26+02:00`
- 开始时间：`2026-06-07T01:20:38+02:00`
- 结束时间：`2026-06-07T13:50:24+02:00`
- Job ID：`2668874`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：Formal Qwen 1-pattern RL rerun: qwen on cifar-100, max_steps=100, num_generations=8, KL=0.005, CUDA-OOM reward worker restart fix, output on /data.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0120_formal100_qwen_cifar100_kl0005_l40s`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0120_formal100_qwen_cifar100_kl0005_l40s/slurm/tunerl-formal100_qwen_cifar100_kl0005_l40s-2668874.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0120_formal100_qwen_cifar100_kl0005_l40s/slurm/tunerl-formal100_qwen_cifar100_kl0005_l40s-2668874.err`
- 工作目录：`/tmp/s471802/20260607_0120_formal100_qwen_cifar100_kl0005_l40s/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0120_formal100_qwen_cifar100_kl0005_l40s`
<!-- NNGPT_RUN:20260607_0120_formal100_qwen_cifar100_kl0005_l40s:END -->
## 20260607_0120_formal100_qwen_imagenette_kl0005_h100

<!-- NNGPT_RUN:20260607_0120_formal100_qwen_imagenette_kl0005_h100:START -->
- 运行 ID：`20260607_0120_formal100_qwen_imagenette_kl0005_h100`
- 标签：`formal100_qwen_imagenette_kl0005_h100`
- 状态：已结束
- 提交时间：`2026-06-07T01:20:26+02:00`
- 开始时间：`2026-06-07T01:20:38+02:00`
- 结束时间：`2026-06-07T08:45:13+02:00`
- Job ID：`2668876`
- 分区 / QoS：`h100`
- 节点：`jnultra01`
- 提交 commit：`c9276599c8dd1df4def862b21dd3a766638bccd5 Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse`
- 本次改动：Formal Qwen 1-pattern RL rerun: qwen on imagenette, max_steps=100, num_generations=8, KL=0.005, CUDA-OOM reward worker restart fix, output on /data.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0120_formal100_qwen_imagenette_kl0005_h100`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0120_formal100_qwen_imagenette_kl0005_h100/slurm/tunerl-formal100_qwen_imagenette_kl0005_h100-2668876.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0120_formal100_qwen_imagenette_kl0005_h100/slurm/tunerl-formal100_qwen_imagenette_kl0005_h100-2668876.err`
- 工作目录：`/tmp/s471802/20260607_0120_formal100_qwen_imagenette_kl0005_h100/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0120_formal100_qwen_imagenette_kl0005_h100`
<!-- NNGPT_RUN:20260607_0120_formal100_qwen_imagenette_kl0005_h100:END -->
## 20260607_0214_formal100_olympic_a7_cifar10_kl0005_nothink_l40s

<!-- NNGPT_RUN:20260607_0214_formal100_olympic_a7_cifar10_kl0005_nothink_l40s:START -->
- 运行 ID：`20260607_0214_formal100_olympic_a7_cifar10_kl0005_nothink_l40s`
- 标签：`formal100_olympic_a7_cifar10_kl0005_nothink_l40s`
- 状态：已结束
- 提交时间：`2026-06-07T02:01:22+02:00`
- 开始时间：`2026-06-07T02:01:25+02:00`
- 结束时间：`2026-06-07T09:21:41+02:00`
- Job ID：`2668922`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat06`
- 提交 commit：`78aadac6b5e01cfa9fcb390676d96166b68c3ba3 Add RL seed control for SFT reward runs`
- 本次改动：Formal100 OlympicCoder A7 cifar-10 RL with copied no-think tokenizer, KL=0.005 max_steps=100 gen=8
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0214_formal100_olympic_a7_cifar10_kl0005_nothink_l40s`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0214_formal100_olympic_a7_cifar10_kl0005_nothink_l40s/slurm/tunerl-formal100_olympic_a7_cifar10_kl0005_nothink_l40s-2668922.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0214_formal100_olympic_a7_cifar10_kl0005_nothink_l40s/slurm/tunerl-formal100_olympic_a7_cifar10_kl0005_nothink_l40s-2668922.err`
- 工作目录：`/tmp/s471802/20260607_0214_formal100_olympic_a7_cifar10_kl0005_nothink_l40s/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0214_formal100_olympic_a7_cifar10_kl0005_nothink_l40s`
<!-- NNGPT_RUN:20260607_0214_formal100_olympic_a7_cifar10_kl0005_nothink_l40s:END -->
## 20260607_0214_formal100_olympic_a7_cifar100_kl0005_nothink_l40s

<!-- NNGPT_RUN:20260607_0214_formal100_olympic_a7_cifar100_kl0005_nothink_l40s:START -->
- 运行 ID：`20260607_0214_formal100_olympic_a7_cifar100_kl0005_nothink_l40s`
- 标签：`formal100_olympic_a7_cifar100_kl0005_nothink_l40s`
- 状态：已结束
- 提交时间：`2026-06-07T02:01:22+02:00`
- 开始时间：`2026-06-07T02:01:26+02:00`
- 结束时间：`2026-06-07T09:46:38+02:00`
- Job ID：`2668924`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat06`
- 提交 commit：`78aadac6b5e01cfa9fcb390676d96166b68c3ba3 Add RL seed control for SFT reward runs`
- 本次改动：Formal100 OlympicCoder A7 cifar-100 RL with copied no-think tokenizer, KL=0.005 max_steps=100 gen=8
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0214_formal100_olympic_a7_cifar100_kl0005_nothink_l40s`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0214_formal100_olympic_a7_cifar100_kl0005_nothink_l40s/slurm/tunerl-formal100_olympic_a7_cifar100_kl0005_nothink_l40s-2668924.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0214_formal100_olympic_a7_cifar100_kl0005_nothink_l40s/slurm/tunerl-formal100_olympic_a7_cifar100_kl0005_nothink_l40s-2668924.err`
- 工作目录：`/tmp/s471802/20260607_0214_formal100_olympic_a7_cifar100_kl0005_nothink_l40s/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0214_formal100_olympic_a7_cifar100_kl0005_nothink_l40s`
<!-- NNGPT_RUN:20260607_0214_formal100_olympic_a7_cifar100_kl0005_nothink_l40s:END -->
## 20260607_0214_formal100_olympic_a7_imagenette_kl0005_nothink_std3

<!-- NNGPT_RUN:20260607_0214_formal100_olympic_a7_imagenette_kl0005_nothink_std3:START -->
- 运行 ID：`20260607_0214_formal100_olympic_a7_imagenette_kl0005_nothink_std3`
- 标签：`formal100_olympic_a7_imagenette_kl0005_nothink_std3`
- 状态：已结束
- 提交时间：`2026-06-07T02:01:23+02:00`
- 开始时间：`2026-06-07T02:01:33+02:00`
- 结束时间：`2026-06-07T11:05:25+02:00`
- Job ID：`2668926`
- 分区 / QoS：`standard`
- 节点：`jn001`
- 提交 commit：`78aadac6b5e01cfa9fcb390676d96166b68c3ba3 Add RL seed control for SFT reward runs`
- 本次改动：Formal100 OlympicCoder A7 imagenette RL with copied no-think tokenizer, KL=0.005 max_steps=100 gen=8
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0214_formal100_olympic_a7_imagenette_kl0005_nothink_std3`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0214_formal100_olympic_a7_imagenette_kl0005_nothink_std3/slurm/tunerl-formal100_olympic_a7_imagenette_kl0005_nothink_std3-2668926.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0214_formal100_olympic_a7_imagenette_kl0005_nothink_std3/slurm/tunerl-formal100_olympic_a7_imagenette_kl0005_nothink_std3-2668926.err`
- 工作目录：`/tmp/s471802/20260607_0214_formal100_olympic_a7_imagenette_kl0005_nothink_std3/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_0214_formal100_olympic_a7_imagenette_kl0005_nothink_std3`
<!-- NNGPT_RUN:20260607_0214_formal100_olympic_a7_imagenette_kl0005_nothink_std3:END -->
## 20260607_1153_formal100_granite_a7_cifar10_kl0005_l40s

<!-- NNGPT_RUN:20260607_1153_formal100_granite_a7_cifar10_kl0005_l40s:START -->
- 运行 ID：`20260607_1153_formal100_granite_a7_cifar10_kl0005_l40s`
- 标签：`formal100_granite_a7_cifar10_kl0005_l40s`
- 状态：运行中
- 提交时间：`2026-06-07T11:12:05+02:00`
- 开始时间：`2026-06-07T11:12:18+02:00`
- 结束时间：-
- Job ID：`2669499`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat04`
- 提交 commit：`78aadac6b5e01cfa9fcb390676d96166b68c3ba3 Add RL seed control for SFT reward runs`
- 本次改动：Formal100 Granite 4.1 8B A7 CIFAR10 RL pilot, KL=0.005 max_steps=100 gen=8
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_1153_formal100_granite_a7_cifar10_kl0005_l40s`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_1153_formal100_granite_a7_cifar10_kl0005_l40s/slurm/tunerl-formal100_granite_a7_cifar10_kl0005_l40s-2669499.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_1153_formal100_granite_a7_cifar10_kl0005_l40s/slurm/tunerl-formal100_granite_a7_cifar10_kl0005_l40s-2669499.err`
- 工作目录：`/tmp/s471802/20260607_1153_formal100_granite_a7_cifar10_kl0005_l40s/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_1153_formal100_granite_a7_cifar10_kl0005_l40s`
<!-- NNGPT_RUN:20260607_1153_formal100_granite_a7_cifar10_kl0005_l40s:END -->

人工更新：2026-06-07 13:05 左右，旧 Granite A7 CIFAR10 KL=0.005 job 2669499 因 early RL 样本大量 target_structure_match=False / actual_block_live=False 且出现下采样导致 output size too small，已手动取消；后续改投 KL=0.06 probe job 2669987，run_id 20260607_1305_formal100_granite_a7_cifar10_kl006_l40s。

## 20260607_1305_formal100_granite_a7_cifar10_kl006_l40s

<!-- NNGPT_RUN:20260607_1305_formal100_granite_a7_cifar10_kl006_l40s:START -->
- 运行 ID：`20260607_1305_formal100_granite_a7_cifar10_kl006_l40s`
- 标签：`formal100_granite_a7_cifar10_kl006_l40s`
- 状态：运行中
- 提交时间：`2026-06-07T13:03:12+02:00`
- 开始时间：`2026-06-07T13:03:15+02:00`
- 结束时间：-
- Job ID：`2669987`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat04`
- 提交 commit：`78aadac6b5e01cfa9fcb390676d96166b68c3ba3 Add RL seed control for SFT reward runs`
- 本次改动：Formal100 Granite 4.1 8B A7 CIFAR10 RL probe, KL=0.06 max_steps=100 gen=8
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_1305_formal100_granite_a7_cifar10_kl006_l40s`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_1305_formal100_granite_a7_cifar10_kl006_l40s/slurm/tunerl-formal100_granite_a7_cifar10_kl006_l40s-2669987.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_1305_formal100_granite_a7_cifar10_kl006_l40s/slurm/tunerl-formal100_granite_a7_cifar10_kl006_l40s-2669987.err`
- 工作目录：`/tmp/s471802/20260607_1305_formal100_granite_a7_cifar10_kl006_l40s/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260607_1305_formal100_granite_a7_cifar10_kl006_l40s`
<!-- NNGPT_RUN:20260607_1305_formal100_granite_a7_cifar10_kl006_l40s:END -->
## 20260608_0109_struct1_v2_a18_cifar10_seed42

<!-- NNGPT_RUN:20260608_0109_struct1_v2_a18_cifar10_seed42:START -->
- 运行 ID：`20260608_0109_struct1_v2_a18_cifar10_seed42`
- 标签：`struct1_v2_a18_cifar10_s42`
- 状态：运行中
- 提交时间：`2026-06-08T01:09:45+02:00`
- 开始时间：`2026-06-08T01:09:56+02:00`
- 结束时间：-
- Job ID：`2671655`
- 分区 / QoS：`h100`
- 节点：`jnultra01`
- 提交 commit：`78aadac6b5e01cfa9fcb390676d96166b68c3ba3 Add RL seed control for SFT reward runs`
- 本次改动：Current reward 4-pattern DeepSeek A18 CIFAR-10 RL seed 42; 1000-sample target, fresh stage2, trainable SFT adapter
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_seed42`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_seed42/slurm/tunerl-struct1_v2_a18_cifar10_s42-2671655.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_seed42/slurm/tunerl-struct1_v2_a18_cifar10_s42-2671655.err`
- 工作目录：`/tmp/s471802/20260608_0109_struct1_v2_a18_cifar10_seed42/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_seed42`
<!-- NNGPT_RUN:20260608_0109_struct1_v2_a18_cifar10_seed42:END -->
## 20260608_0109_struct1_v2_a18_cifar10_seed123

<!-- NNGPT_RUN:20260608_0109_struct1_v2_a18_cifar10_seed123:START -->
- 运行 ID：`20260608_0109_struct1_v2_a18_cifar10_seed123`
- 标签：`struct1_v2_a18_cifar10_s123`
- 状态：运行中
- 提交时间：`2026-06-08T01:09:45+02:00`
- 开始时间：`2026-06-08T01:09:56+02:00`
- 结束时间：-
- Job ID：`2671657`
- 分区 / QoS：`h100`
- 节点：`jnultra02`
- 提交 commit：`78aadac6b5e01cfa9fcb390676d96166b68c3ba3 Add RL seed control for SFT reward runs`
- 本次改动：Current reward 4-pattern DeepSeek A18 CIFAR-10 RL seed 123; 1000-sample target, fresh stage2, trainable SFT adapter
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_seed123`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_seed123/slurm/tunerl-struct1_v2_a18_cifar10_s123-2671657.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_seed123/slurm/tunerl-struct1_v2_a18_cifar10_s123-2671657.err`
- 工作目录：`/tmp/s471802/20260608_0109_struct1_v2_a18_cifar10_seed123/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_seed123`
<!-- NNGPT_RUN:20260608_0109_struct1_v2_a18_cifar10_seed123:END -->
## 20260608_0109_struct1_v2_a18_cifar10_seed777

<!-- NNGPT_RUN:20260608_0109_struct1_v2_a18_cifar10_seed777:START -->
- 运行 ID：`20260608_0109_struct1_v2_a18_cifar10_seed777`
- 标签：`struct1_v2_a18_cifar10_s777`
- 状态：运行中
- 提交时间：`2026-06-08T01:09:45+02:00`
- 开始时间：`2026-06-08T01:09:56+02:00`
- 结束时间：-
- Job ID：`2671659`
- 分区 / QoS：`h100`
- 节点：`jnultra02`
- 提交 commit：`78aadac6b5e01cfa9fcb390676d96166b68c3ba3 Add RL seed control for SFT reward runs`
- 本次改动：Current reward 4-pattern DeepSeek A18 CIFAR-10 RL seed 777; 1000-sample target, fresh stage2, trainable SFT adapter
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_seed777`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_seed777/slurm/tunerl-struct1_v2_a18_cifar10_s777-2671659.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_seed777/slurm/tunerl-struct1_v2_a18_cifar10_s777-2671659.err`
- 工作目录：`/tmp/s471802/20260608_0109_struct1_v2_a18_cifar10_seed777/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_seed777`
<!-- NNGPT_RUN:20260608_0109_struct1_v2_a18_cifar10_seed777:END -->
## 20260608_0109_struct1_v2_a18_cifar10_dirty_seed42

<!-- NNGPT_RUN:20260608_0109_struct1_v2_a18_cifar10_dirty_seed42:START -->
- 运行 ID：`20260608_0109_struct1_v2_a18_cifar10_dirty_seed42`
- 标签：`struct1_v2_a18_cifar10_dirty_s42`
- 状态：已结束(OUT_OF_MEMORY)
- 提交时间：`2026-06-08T01:09:45+02:00`
- 开始时间：`2026-06-08T01:09:56+02:00`
- 结束时间：`2026-06-08T06:12:39+02:00`
- Job ID：`2671655`
- 分区 / QoS：`h100`
- 节点：`jnultra01`
- 提交 commit：`c9276599-dirty Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse + current dirty reward/runtime fixes + RL seed plumbing`
- 本次改动：Current dirty reward 4-pattern DeepSeek A18 CIFAR-10 RL seed 42; 1000-sample target, fresh stage2, trainable SFT adapter
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_dirty_seed42`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_dirty_seed42/slurm/tunerl-struct1_v2_a18_cifar10_dirty_s42-2671655.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_dirty_seed42/slurm/tunerl-struct1_v2_a18_cifar10_dirty_s42-2671655.err`
- 工作目录：`/tmp/s471802/20260608_0109_struct1_v2_a18_cifar10_dirty_seed42/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_dirty_seed42`
<!-- NNGPT_RUN:20260608_0109_struct1_v2_a18_cifar10_dirty_seed42:END -->
## 20260608_0846_struct1_v2_a18_cifar10_seed42_mem240_loaderfix

<!-- NNGPT_RUN:20260608_0846_struct1_v2_a18_cifar10_seed42_mem240_loaderfix:START -->
- 运行 ID：`20260608_0846_struct1_v2_a18_cifar10_seed42_mem240_loaderfix`
- 标签：`struct1_v2_a18_cifar10_seed42_mem240_loaderfix`
- 状态：已结束
- 提交时间：`2026-06-08T08:46:47+02:00`
- 开始时间：`2026-06-08T08:46:51+02:00`
- 结束时间：`2026-06-08T19:26:46+02:00`
- Job ID：`2671917`
- 分区 / QoS：`h100`
- 节点：`jnultra01`
- 提交 commit：`18e1fa29f7000b9cff4d3fb4ce2e2dfcdbaf964f Release reward eval dataloader workers`
- 本次改动：Replacement for cancelled seed42 CIFAR10 RL after reward eval DataLoader worker cleanup; old job 2671907 was stopped to avoid CPU RAM OOM accumulation.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0846_struct1_v2_a18_cifar10_seed42_mem240_loaderfix`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0846_struct1_v2_a18_cifar10_seed42_mem240_loaderfix/slurm/tunerl-struct1_v2_a18_cifar10_seed42_mem240_loaderfix-2671917.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0846_struct1_v2_a18_cifar10_seed42_mem240_loaderfix/slurm/tunerl-struct1_v2_a18_cifar10_seed42_mem240_loaderfix-2671917.err`
- 工作目录：`/tmp/s471802/20260608_0846_struct1_v2_a18_cifar10_seed42_mem240_loaderfix/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0846_struct1_v2_a18_cifar10_seed42_mem240_loaderfix`
<!-- NNGPT_RUN:20260608_0846_struct1_v2_a18_cifar10_seed42_mem240_loaderfix:END -->
## 20260608_0109_struct1_v2_a18_cifar10_dirty_seed123

<!-- NNGPT_RUN:20260608_0109_struct1_v2_a18_cifar10_dirty_seed123:START -->
- 运行 ID：`20260608_0109_struct1_v2_a18_cifar10_dirty_seed123`
- 标签：`struct1_v2_a18_cifar10_dirty_s123`
- 状态：已结束
- 提交时间：`2026-06-08T01:09:45+02:00`
- 开始时间：`2026-06-08T01:09:56+02:00`
- 结束时间：`2026-06-08T12:19:37+02:00`
- Job ID：`2671657`
- 分区 / QoS：`h100`
- 节点：`jnultra02`
- 提交 commit：`c9276599-dirty Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse + current dirty reward/runtime fixes + RL seed plumbing`
- 本次改动：Current dirty reward 4-pattern DeepSeek A18 CIFAR-10 RL seed 123; 1000-sample target, fresh stage2, trainable SFT adapter
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_dirty_seed123`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_dirty_seed123/slurm/tunerl-struct1_v2_a18_cifar10_dirty_s123-2671657.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_dirty_seed123/slurm/tunerl-struct1_v2_a18_cifar10_dirty_s123-2671657.err`
- 工作目录：`/tmp/s471802/20260608_0109_struct1_v2_a18_cifar10_dirty_seed123/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_dirty_seed123`
<!-- NNGPT_RUN:20260608_0109_struct1_v2_a18_cifar10_dirty_seed123:END -->
## 20260608_0109_struct1_v2_a18_cifar10_dirty_seed777

<!-- NNGPT_RUN:20260608_0109_struct1_v2_a18_cifar10_dirty_seed777:START -->
- 运行 ID：`20260608_0109_struct1_v2_a18_cifar10_dirty_seed777`
- 标签：`struct1_v2_a18_cifar10_dirty_s777`
- 状态：已结束
- 提交时间：`2026-06-08T01:09:45+02:00`
- 开始时间：`2026-06-08T01:09:56+02:00`
- 结束时间：`2026-06-08T12:38:07+02:00`
- Job ID：`2671659`
- 分区 / QoS：`h100`
- 节点：`jnultra02`
- 提交 commit：`c9276599-dirty Scale non-trainable warmup penalty by 0.2x to prevent Mistral collapse + current dirty reward/runtime fixes + RL seed plumbing`
- 本次改动：Current dirty reward 4-pattern DeepSeek A18 CIFAR-10 RL seed 777; 1000-sample target, fresh stage2, trainable SFT adapter
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_dirty_seed777`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_dirty_seed777/slurm/tunerl-struct1_v2_a18_cifar10_dirty_s777-2671659.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_dirty_seed777/slurm/tunerl-struct1_v2_a18_cifar10_dirty_s777-2671659.err`
- 工作目录：`/tmp/s471802/20260608_0109_struct1_v2_a18_cifar10_dirty_seed777/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_0109_struct1_v2_a18_cifar10_dirty_seed777`
<!-- NNGPT_RUN:20260608_0109_struct1_v2_a18_cifar10_dirty_seed777:END -->
## 20260608_1547_article_reward_ablation_a18_cifar10_gcvl_e1_full_reward
- 提交时间: 2026-06-08T15:47:33+0200
- 状态: failed; superseded by 20260608_1922_article_reward_ablation_a18_cifar10_gcvl_e1_p3500_c1200_alloc_full_reward
- main job: 2675832
- 分区/GPU: gpu_computervision_long, 4 GPU
- mem/cpus: 160G, 32
- commit: 90b868a049a4acc154559011f688196b594395de (Record reward ablation seed)
- reward variant: full_reward
- seed: 42
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1
- formal reward epochs: 1
- max prompt length: 3500 (sbatch default)
- max completion length: 1536
- generation batch size: default/unset
- reward workers: NNGPT_REWARD_WORKERS_PER_GPU=1, NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU=1
- samples target: 1000 (8 generations x 125 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1547_article_reward_ablation_a18_cifar10_gcvl_e1_full_reward
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1547_article_reward_ablation_a18_cifar10_gcvl_e1_full_reward/slurm/abl-e1-full_reward-2675832.out / .err
- 训练结果: 144/1000 samples 后训练进程 CUDA OOM；原参数 prompt=3500, completion=1536, generation_batch_size=None（TRL 实际默认 8）, formal_reward_epochs=1。
- 主要缺陷: OOM 日志显示 allocated 34.02GiB, reserved but unallocated 8.97GiB，符合 allocator fragmentation；未作为最终 C 结果使用。
- 分析: replacement 保留 prompt=3500 和其它正式口径，只改 completion=1200 并启用 expandable_segments allocator。

## 20260608_1547_article_reward_ablation_a18_cifar10_gcvl_e1_no_diversity_bonus
- 提交时间: 2026-06-08T15:47:33+0200
- 状态: cancelled; superseded by 20260608_1928_article_reward_ablation_a18_cifar10_std_e1_p3500_c1200_alloc_no_diversity_bonus
- main job: 2675833
- 分区/GPU: gpu_computervision_long, 4 GPU
- mem/cpus: 160G, 32
- commit: 90b868a049a4acc154559011f688196b594395de (Record reward ablation seed)
- reward variant: no_diversity_bonus
- seed: 42
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1
- formal reward epochs: 1
- max prompt length: 3500 (sbatch default)
- max completion length: 1536
- generation batch size: default/unset
- reward workers: NNGPT_REWARD_WORKERS_PER_GPU=1, NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU=1
- samples target: 1000 (8 generations x 125 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1547_article_reward_ablation_a18_cifar10_gcvl_e1_no_diversity_bonus
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1547_article_reward_ablation_a18_cifar10_gcvl_e1_no_diversity_bonus/slurm/abl-e1-no_diversity_bonus-2675833.out / .err
- 训练结果: 旧参数任务运行中被主动停止；停止时约 168 samples，原参数 prompt=3500, completion=1536, generation_batch_size=None（TRL 实际默认 8）, formal_reward_epochs=1。
- 主要缺陷: 参数口径已被 replacement 更新；未作为最终 C 结果使用。
- 分析: replacement 保留 prompt=3500 和其它正式口径，只改 completion=1200 并启用 expandable_segments allocator。


## 20260608_1646_article_reward_ablation_a18_cifar10_std_e1_no_repeat_penalty
- 提交时间: 2026-06-08T16:46:28+0200
- 状态: cancelled; superseded by 20260608_1928_article_reward_ablation_a18_cifar10_std_e1_p3500_c1200_alloc_no_repeat_penalty
- main job: 2676023
- 分区/GPU: standard, 3 GPU
- mem/cpus: 160G, 32
- commit: 90b868a049a4acc154559011f688196b594395de (Record reward ablation seed)
- reward variant: no_repeat_penalty
- seed: 42
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1
- formal reward epochs: 1
- max prompt length: 3500 (sbatch default)
- max completion length: 1536
- generation batch size: default/unset
- reward workers: NNGPT_REWARD_WORKERS_PER_GPU=1, NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU=1
- samples target: 1000 (8 generations x 125 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1646_article_reward_ablation_a18_cifar10_std_e1_no_repeat_penalty
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1646_article_reward_ablation_a18_cifar10_std_e1_no_repeat_penalty/slurm/abl-e1-std-no_repeat_penalty-2676023.out / .err
- 训练结果: 旧参数任务运行中被主动停止；停止时约 88 samples，原参数 prompt=3500, completion=1536, generation_batch_size=None（TRL 实际默认 8）, formal_reward_epochs=1。
- 主要缺陷: 参数口径已被 replacement 更新；未作为最终 C 结果使用。
- 分析: replacement 保留 prompt=3500 和其它正式口径，只改 completion=1200 并启用 expandable_segments allocator。

## 20260608_1922_article_reward_ablation_a18_cifar10_gcvl_e1_p3500_c1200_alloc_full_reward
- 提交时间: 2026-06-08T19:22:02+0200
- 状态: submitted
- main job: 2677248
- 分区/GPU: gpu_computervision_long, 4 GPU
- mem/cpus: 160G, 32
- commit: 90b868a049a4acc154559011f688196b594395de (Record reward ablation seed)
- reward variant: full_reward
- seed: 42
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1
- formal reward epochs: 1
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- allocator: PYTORCH_ALLOC_CONF=expandable_segments:True, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- reward workers: NNGPT_REWARD_WORKERS_PER_GPU=1, NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU=1
- samples target: 1000 (8 generations x 125 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1922_article_reward_ablation_a18_cifar10_gcvl_e1_p3500_c1200_alloc_full_reward
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1922_article_reward_ablation_a18_cifar10_gcvl_e1_p3500_c1200_alloc_full_reward/slurm/abl-full_reward-2677248.out / .err
- 训练结果: TODO
- 主要缺陷: TODO
- 分析: TODO

## 20260608_1922_article_D_a9_cifar10_seed42_p3500_c1200_alloc

<!-- NNGPT_RUN:20260608_1922_article_D_a9_cifar10_seed42_p3500_c1200_alloc:START -->
- 运行 ID：`20260608_1922_article_D_a9_cifar10_seed42_p3500_c1200_alloc`
- 标签：`article_D_a9_cifar10_seed42_p3500_c1200_alloc`
- 状态：已结束
- 提交时间：`2026-06-08T19:22:02+02:00`
- 开始时间：`2026-06-08T19:22:15+02:00`
- 结束时间：`2026-06-09T06:25:29+02:00`
- Job ID：`2677251`
- 分区 / QoS：`standard`
- 节点：`jn006`
- 提交 commit：`90b868a049a4acc154559011f688196b594395de Record reward ablation seed`
- 本次改动：Article D 1-pattern DeepSeek A9 CIFAR-10 seed 42; formal reward epoch 1; prompt=3500 completion=1200 allocator expandable_segments
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1922_article_D_a9_cifar10_seed42_p3500_c1200_alloc`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1922_article_D_a9_cifar10_seed42_p3500_c1200_alloc/slurm/tunerl-article_d_a9_cifar10_seed42_p3500_c1200_alloc-2677251.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1922_article_D_a9_cifar10_seed42_p3500_c1200_alloc/slurm/tunerl-article_d_a9_cifar10_seed42_p3500_c1200_alloc-2677251.err`
- 工作目录：`/tmp/s471802/20260608_1922_article_D_a9_cifar10_seed42_p3500_c1200_alloc/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1922_article_D_a9_cifar10_seed42_p3500_c1200_alloc`
<!-- NNGPT_RUN:20260608_1922_article_D_a9_cifar10_seed42_p3500_c1200_alloc:END -->
## 20260608_1922_article_D_a9_cifar10_seed114_p3500_c1200_alloc

<!-- NNGPT_RUN:20260608_1922_article_D_a9_cifar10_seed114_p3500_c1200_alloc:START -->
- 运行 ID：`20260608_1922_article_D_a9_cifar10_seed114_p3500_c1200_alloc`
- 标签：`article_D_a9_cifar10_seed114_p3500_c1200_alloc`
- 状态：已结束
- 提交时间：`2026-06-08T19:22:02+02:00`
- 开始时间：`2026-06-08T19:22:15+02:00`
- 结束时间：`2026-06-09T01:11:31+02:00`
- Job ID：`2677253`
- 分区 / QoS：`standard`
- 节点：`jn020`
- 提交 commit：`90b868a049a4acc154559011f688196b594395de Record reward ablation seed`
- 本次改动：Article D 1-pattern DeepSeek A9 CIFAR-10 seed 114; formal reward epoch 1; prompt=3500 completion=1200 allocator expandable_segments
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1922_article_D_a9_cifar10_seed114_p3500_c1200_alloc`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1922_article_D_a9_cifar10_seed114_p3500_c1200_alloc/slurm/tunerl-article_d_a9_cifar10_seed114_p3500_c1200_alloc-2677253.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1922_article_D_a9_cifar10_seed114_p3500_c1200_alloc/slurm/tunerl-article_d_a9_cifar10_seed114_p3500_c1200_alloc-2677253.err`
- 工作目录：`/tmp/s471802/20260608_1922_article_D_a9_cifar10_seed114_p3500_c1200_alloc/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1922_article_D_a9_cifar10_seed114_p3500_c1200_alloc`
<!-- NNGPT_RUN:20260608_1922_article_D_a9_cifar10_seed114_p3500_c1200_alloc:END -->
## 20260608_1922_article_D_a9_cifar10_seed514_p3500_c1200_alloc

<!-- NNGPT_RUN:20260608_1922_article_D_a9_cifar10_seed514_p3500_c1200_alloc:START -->
- 运行 ID：`20260608_1922_article_D_a9_cifar10_seed514_p3500_c1200_alloc`
- 标签：`article_D_a9_cifar10_seed514_p3500_c1200_alloc`
- 状态：失败；Slurm memory OOM，已用 mem=240G replacement 重投
- 提交时间：`2026-06-08T19:22:03+02:00`
- 开始时间：`2026-06-08T19:22:09+02:00`
- 结束时间：`2026-06-08T19:41:23+02:00`
- Job ID：`2677255`
- 分区 / QoS：`standard`
- 节点：`jn102`
- 提交 commit：`90b868a049a4acc154559011f688196b594395de Record reward ablation seed`
- 本次改动：Article D 1-pattern DeepSeek A9 CIFAR-10 seed 514; formal reward epoch 1; prompt=3500 completion=1200 allocator expandable_segments
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1922_article_D_a9_cifar10_seed514_p3500_c1200_alloc`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1922_article_D_a9_cifar10_seed514_p3500_c1200_alloc/slurm/tunerl-article_d_a9_cifar10_seed514_p3500_c1200_alloc-2677255.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1922_article_D_a9_cifar10_seed514_p3500_c1200_alloc/slurm/tunerl-article_d_a9_cifar10_seed514_p3500_c1200_alloc-2677255.err`
- 工作目录：`/tmp/s471802/20260608_1922_article_D_a9_cifar10_seed514_p3500_c1200_alloc/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：运行 16/1000 samples 后 Slurm 标记 OUT_OF_MEMORY；MaxRSS 约 167761112K，超过本次 --mem 160G。训练参数为 prompt=3500, completion=1200, generation_batch_size=8, formal_reward_epochs=1。
- 主要缺陷：这是作业内存 cgroup OOM，不是 CUDA allocator OOM；同参数仅提高 sbatch mem 到 240G 后重投 seed514。
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1922_article_D_a9_cifar10_seed514_p3500_c1200_alloc`
<!-- NNGPT_RUN:20260608_1922_article_D_a9_cifar10_seed514_p3500_c1200_alloc:END -->

## 20260608_1928_article_reward_ablation_a18_cifar10_std_e1_p3500_c1200_alloc_no_diversity_bonus
- 提交时间: 2026-06-08T19:28:09+0200
- 状态: submitted
- main job: 2677276
- 分区/GPU: standard, 3 GPU
- mem/cpus: 160G, 32
- commit: 90b868a049a4acc154559011f688196b594395de (Record reward ablation seed)
- reward variant: no_diversity_bonus
- seed: 42
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1
- formal reward epochs: 1
- max completion length: 1200
- samples target: 1000 (8 generations x 125 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1928_article_reward_ablation_a18_cifar10_std_e1_p3500_c1200_alloc_no_diversity_bonus
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1928_article_reward_ablation_a18_cifar10_std_e1_p3500_c1200_alloc_no_diversity_bonus/slurm/abl-no_diversity_bonus-2677276.out / .err
- 训练结果: TODO
- 主要缺陷: TODO
- 分析: TODO

## 20260608_1928_article_reward_ablation_a18_cifar10_std_e1_p3500_c1200_alloc_no_repeat_penalty
- 提交时间: 2026-06-08T19:28:09+0200
- 状态: submitted
- main job: 2677277
- 分区/GPU: standard, 3 GPU
- mem/cpus: 160G, 32
- commit: 90b868a049a4acc154559011f688196b594395de (Record reward ablation seed)
- reward variant: no_repeat_penalty
- seed: 42
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1
- formal reward epochs: 1
- max completion length: 1200
- samples target: 1000 (8 generations x 125 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1928_article_reward_ablation_a18_cifar10_std_e1_p3500_c1200_alloc_no_repeat_penalty
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1928_article_reward_ablation_a18_cifar10_std_e1_p3500_c1200_alloc_no_repeat_penalty/slurm/abl-no_repeat_penalty-2677277.out / .err
- 训练结果: TODO
- 主要缺陷: TODO
- 分析: TODO
## 20260608_1945_article_D_a9_cifar10_seed514_p3500_c1200_alloc_mem240

<!-- NNGPT_RUN:20260608_1945_article_D_a9_cifar10_seed514_p3500_c1200_alloc_mem240:START -->
- 运行 ID：`20260608_1945_article_D_a9_cifar10_seed514_p3500_c1200_alloc_mem240`
- 标签：`article_D_a9_cifar10_seed514_p3500_c1200_alloc_mem240`
- 状态：已结束(OUT_OF_MEMORY)
- 提交时间：`2026-06-08T19:45:07+02:00`
- 开始时间：`2026-06-08T19:45:23+02:00`
- 结束时间：`2026-06-08T20:11:54+02:00`
- Job ID：`2677284`
- 分区 / QoS：`standard`
- 节点：`jn019`
- 提交 commit：`90b868a049a4acc154559011f688196b594395de Record reward ablation seed`
- 本次改动：Article D 1-pattern DeepSeek A9 CIFAR-10 seed 514 retry after Slurm mem OOM; same training params, mem=240G
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1945_article_D_a9_cifar10_seed514_p3500_c1200_alloc_mem240`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1945_article_D_a9_cifar10_seed514_p3500_c1200_alloc_mem240/slurm/tunerl-article_d_a9_cifar10_seed514_p3500_c1200_alloc_mem240-2677284.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1945_article_D_a9_cifar10_seed514_p3500_c1200_alloc_mem240/slurm/tunerl-article_d_a9_cifar10_seed514_p3500_c1200_alloc_mem240-2677284.err`
- 工作目录：`/tmp/s471802/20260608_1945_article_D_a9_cifar10_seed514_p3500_c1200_alloc_mem240/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260608_1945_article_D_a9_cifar10_seed514_p3500_c1200_alloc_mem240`
<!-- NNGPT_RUN:20260608_1945_article_D_a9_cifar10_seed514_p3500_c1200_alloc_mem240:END -->
## 20260609_1016_article_D_a9_cifar10_seed123_gcvl_p3500_c1200_alloc

<!-- NNGPT_RUN:20260609_1016_article_D_a9_cifar10_seed123_gcvl_p3500_c1200_alloc:START -->
- 运行 ID：`20260609_1016_article_D_a9_cifar10_seed123_gcvl_p3500_c1200_alloc`
- 标签：`article_D_a9_cifar10_seed123_gcvl_p3500_c1200_alloc`
- 状态：已结束
- 提交时间：`2026-06-09T10:13:34+02:00`
- 开始时间：`2026-06-09T10:17:26+02:00`
- 结束时间：`2026-06-09T15:57:53+02:00`
- Job ID：`2680341`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat06`
- 提交 commit：`c91714dbe7dad1d02a9080243945bbf8e8ec9300 Simplify reward memory preflight`
- 本次改动：Article D 1-pattern DeepSeek A9 CIFAR-10 seed123; aligned multi-seed with 4-pattern; formal reward epoch 1; prompt=3500 completion=1200 allocator expandable_segments; fallback from H100 because predicted start exceeded 2h
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260609_1016_article_D_a9_cifar10_seed123_gcvl_p3500_c1200_alloc`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260609_1016_article_D_a9_cifar10_seed123_gcvl_p3500_c1200_alloc/slurm/tunerl-article_d_a9_cifar10_seed123_gcvl_p3500_c1200_alloc-2680341.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260609_1016_article_D_a9_cifar10_seed123_gcvl_p3500_c1200_alloc/slurm/tunerl-article_d_a9_cifar10_seed123_gcvl_p3500_c1200_alloc-2680341.err`
- 工作目录：`/tmp/s471802/20260609_1016_article_D_a9_cifar10_seed123_gcvl_p3500_c1200_alloc/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260609_1016_article_D_a9_cifar10_seed123_gcvl_p3500_c1200_alloc`
<!-- NNGPT_RUN:20260609_1016_article_D_a9_cifar10_seed123_gcvl_p3500_c1200_alloc:END -->
## 20260609_1019_article_D_a9_cifar10_seed777_std_p3500_c1200_alloc

<!-- NNGPT_RUN:20260609_1019_article_D_a9_cifar10_seed777_std_p3500_c1200_alloc:START -->
- 运行 ID：`20260609_1019_article_D_a9_cifar10_seed777_std_p3500_c1200_alloc`
- 标签：`article_D_a9_cifar10_seed777_std_p3500_c1200_alloc`
- 状态：已结束
- 提交时间：`2026-06-09T10:14:45+02:00`
- 开始时间：`2026-06-09T10:15:06+02:00`
- 结束时间：`2026-06-09T20:58:53+02:00`
- Job ID：`2680345`
- 分区 / QoS：`standard`
- 节点：`jn002`
- 提交 commit：`c91714dbe7dad1d02a9080243945bbf8e8ec9300 Simplify reward memory preflight`
- 本次改动：Article D 1-pattern DeepSeek A9 CIFAR-10 seed777; aligned multi-seed with 4-pattern; formal reward epoch 1; prompt=3500 completion=1200 allocator expandable_segments; fallback to standard because gpu_computervision_long predicted start exceeded 2h
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260609_1019_article_D_a9_cifar10_seed777_std_p3500_c1200_alloc`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260609_1019_article_D_a9_cifar10_seed777_std_p3500_c1200_alloc/slurm/tunerl-article_d_a9_cifar10_seed777_std_p3500_c1200_alloc-2680345.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260609_1019_article_D_a9_cifar10_seed777_std_p3500_c1200_alloc/slurm/tunerl-article_d_a9_cifar10_seed777_std_p3500_c1200_alloc-2680345.err`
- 工作目录：`/tmp/s471802/20260609_1019_article_D_a9_cifar10_seed777_std_p3500_c1200_alloc/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260609_1019_article_D_a9_cifar10_seed777_std_p3500_c1200_alloc`
<!-- NNGPT_RUN:20260609_1019_article_D_a9_cifar10_seed777_std_p3500_c1200_alloc:END -->

## 20260610_4pattern_a18_cifar10_e10_p3500_c1200_s100_full_reward_seed42
- 提交时间: 2026-06-10T19:54:14+0200
- 开始时间: 2026-06-10T19:54:14+0200
- 结束时间: 2026-06-10T22:46:29+0200
- 状态: failed
- main job: 2690384
- 分区/GPU: gpu_computervision_long, 4 GPU
- 节点: jnfat04
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: full_reward
- seed: 42
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 10
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260610_4pattern_a18_cifar10_e10_p3500_c1200_s100_full_reward_seed42
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260610_4pattern_a18_cifar10_e10_p3500_c1200_s100_full_reward_seed42/slurm/abl-full_reward-s42-2690384.out / .err
- 训练结果: 失败；仅写出 24/800 samples，未形成完整正式结果。
- 主要缺陷: trainer update 阶段 GPU0 OOM；本次 reward worker 未排除训练 GPU，pool_size=4，GPU0 同时有训练进程和 reward worker；日志显示 GPU0 reserved 42.99 GiB、free 0.01 GiB。
- 分析: 不作为正式结果；重投必须显式设置 NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU=1，并使用 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。

## 20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100_full_reward_seed42
- 提交时间: 2026-06-12T11:23:35+0200
- 开始时间: 2026-06-12T11:23:35+0200
- 结束时间: 2026-06-12T11:31:01+0200
- 状态: cancelled
- main job: 2697697
- 分区/GPU: h100, 3 GPU
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: full_reward
- seed: 42
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100_full_reward_seed42
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100_full_reward_seed42/slurm/abl-full_reward-s42-2697697.out / .err
- 训练结果: 已取消；0/800 samples，未形成结果。
- 主要缺陷: 资源配置偏保守，reward worker 排除训练 GPU，用户要求改为 3×H100 训练/eval 共用后取消重投。
- 分析: 不作为结果；replacement run 为 2697701 / 20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100_shared_full_reward_seed42。

## 20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100_shared_full_reward_seed42
- 提交时间: 2026-06-12T11:31:38+0200
- 开始时间: 2026-06-12T11:31:38+0200
- 结束时间: 2026-06-12T12:55:29+0200
- 状态: cancelled
- main job: 2697701
- 分区/GPU: h100, 3 GPU
- 节点: jnultra01
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: full_reward
- seed: 42
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100_shared_full_reward_seed42
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100_shared_full_reward_seed42/slurm/abl-full_reward-s42-2697701.out / .err
- 训练结果: 已取消；写出约 32/800 samples，未形成结果。
- 主要缺陷: 3×H100 预计无法在 h100 24h wall time 内完成，且当时尚无 checkpoint；为释放 H100 并改投 4×L40S long 取消。
- 分析: 不作为结果；replacement run 为 2697785 / 20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s_shared_full_full_reward_seed42。

## 20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_ablation_no_diversity_bonus_seed42
- 提交时间: 2026-06-12T12:50:50+0200
- 开始时间: 2026-06-12T12:58:25+0200
- 状态: running
- main job: 2697779
- 分区/GPU: h100, 4 GPU
- 节点: jnultra02
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: no_diversity_bonus
- seed: 42
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_ablation_no_diversity_bonus_seed42
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_ablation_no_diversity_bonus_seed42/slurm/abl-no_diversity_bonus-s42-2697779.out / .err
- 训练结果: TODO
- 主要缺陷: TODO
- 分析: TODO

## 20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s_shared_full_full_reward_seed42
- 提交时间: 2026-06-12T12:56:01+0200
- 开始时间: 2026-06-12T12:56:07+0200
- 结束时间: 2026-06-12T13:19:17+0200
- 状态: cancelled
- main job: 2697785
- 分区/GPU: gpu_computervision_long, 4 GPU
- 节点: jnfat05
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: full_reward
- seed: 42
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s_shared_full_full_reward_seed42
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s_shared_full_full_reward_seed42/slurm/abl-full_reward-s42-2697785.out / .err
- 训练结果: 已取消；0/800 samples，未形成结果。
- 主要缺陷: L40S 误用训练/eval shared 配置；用户指出 L40S 应排除训练 GPU 后取消重投。
- 分析: 不作为结果；replacement run 为 2697817 / 20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s_excl_full_full_reward_seed42。

## 20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s_excl_full_full_reward_seed42
- 提交时间: 2026-06-12T13:19:51+0200
- 开始时间: 2026-06-12T13:19:55+0200
- 状态: running
- main job: 2697817
- 分区/GPU: gpu_computervision_long, 4 GPU
- 节点: jnfat05
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: full_reward
- seed: 42
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s_excl_full_full_reward_seed42
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s_excl_full_full_reward_seed42/slurm/abl-full_reward-s42-2697817.out / .err
- 训练结果: TODO
- 主要缺陷: TODO
- 分析: TODO

## 20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed123
- 提交时间: 2026-06-12T13:40:12+0200
- 状态: submitted
- main job: 2697840
- 分区/GPU: h100, 4 GPU
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: full_reward
- seed: 123
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed123
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed123/slurm/abl-full_reward-s123-2697840.out / .err
- 训练结果: TODO
- 主要缺陷: TODO
- 分析: TODO

## 20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s_excl_full_full_reward_seed777
- 提交时间: 2026-06-12T13:40:57+0200
- 状态: submitted
- main job: 2697842
- 分区/GPU: gpu_computervision_long, 4 GPU
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: full_reward
- seed: 777
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s_excl_full_full_reward_seed777
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s_excl_full_full_reward_seed777/slurm/abl-full_reward-s777-2697842.out / .err
- 训练结果: TODO
- 主要缺陷: TODO
- 分析: TODO

## 20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s_excl_ablation_no_repeat_penalty_seed42
- 提交时间: 2026-06-12T13:45:35+0200
- 状态: submitted
- main job: 2697849
- 分区/GPU: gpu_computervision_long, 4 GPU
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: no_repeat_penalty
- seed: 42
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s_excl_ablation_no_repeat_penalty_seed42
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260612_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s_excl_ablation_no_repeat_penalty_seed42/slurm/abl-no_repeat_penalty-s42-2697849.out / .err
- 训练结果: TODO
- 主要缺陷: TODO
- 分析: TODO

## 20260618_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed42
- 提交时间: 2026-06-18T06:53:23+0200
- 状态: submitted
- main job: 2723084
- 分区/GPU: h100, 4 GPU
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: full_reward
- seed: 42
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/20260601_1pattern_three_model_sft_v3_cifar10_dscoder7b/epoch_sft/A9/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-test1
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260618_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed42
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260618_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed42/slurm/abl-full_reward-s42-2723084.out / .err
- 训练结果: TODO
- 主要缺陷: TODO
- 分析: TODO

## 20260618_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed123
- 提交时间: 2026-06-18T06:53:23+0200
- 状态: submitted
- main job: 2723085
- 分区/GPU: h100, 4 GPU
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: full_reward
- seed: 123
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/20260601_1pattern_three_model_sft_v3_cifar10_dscoder7b/epoch_sft/A9/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-test1
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260618_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed123
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260618_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed123/slurm/abl-full_reward-s123-2723085.out / .err
- 训练结果: TODO
- 主要缺陷: TODO
- 分析: TODO

## 20260618_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed777
- 提交时间: 2026-06-18T06:53:23+0200
- 状态: submitted
- main job: 2723086
- 分区/GPU: h100, 4 GPU
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: full_reward
- seed: 777
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/20260601_1pattern_three_model_sft_v3_cifar10_dscoder7b/epoch_sft/A9/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-test1
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260618_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed777
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260618_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed777/slurm/abl-full_reward-s777-2723086.out / .err
- 训练结果: TODO
- 主要缺陷: TODO
- 分析: TODO

## 20260623_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_ablation_extra_no_diversity_bonus_seed123
- 提交时间: 2026-06-22T19:51:54+0200
- 状态: completed after manual TERM at aligned 800-sample ablation target
- main job: 2751655
- 分区/GPU: h100, 4 GPU
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: no_diversity_bonus
- seed: 123
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (submitted as 8 generations x 125 steps; stopped after 800 rows for aligned ablation analysis)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260623_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_ablation_extra_no_diversity_bonus_seed123
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260623_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_ablation_extra_no_diversity_bonus_seed123/slurm/abl-no_diversity_bonus-s123-2751655.out / .err
- 训练结果: 800/800 samples; positive reward 769/800, formal_success_candidate 778/800, built_ok 786/800, 5-epoch mean test_acc 90.68%, max 94.02%, reward mean 0.4618. Last 100 rows: positive reward 98/100, formal_success_candidate 99/100, built_ok 100/100, mean test_acc 92.94%, max 94.02%, reward mean 0.4876.
- 主要缺陷: job was stopped by TERM for aligned analysis; sacct reports FAILED 1:0 because the stop was intentional, not because of a runtime error.
- 分析: seed123 no-diversity-bonus rerun is the planned second-seed check for the reward ablation; use the full 800 rows only.

## 20260623_rc_sampler_cifar10_e5_heldout_test_l40s_reuse100
- 提交时间: 2026-06-22T19:55:58+0200
- 状态: completed
- main job: 2751692
- 分区/GPU: gpu_computervision_long, 8 L40S GPUs requested
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- 任务: reused 100 rule-constrained sampler candidates; 5-epoch heldout-test evaluation on CIFAR-10
- candidate file: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/home_migrated_20260527/parallel_runs/20260522_baseline_100_v4_audited/baseline/candidates/brute_constrained_random/candidates.jsonl
- eval split: trainvaltest heldout_test, seed 42
- formal reward epochs: 5
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260623_rc_sampler_cifar10_e5_heldout_test_l40s_reuse100
- eval dir: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260623_rc_sampler_cifar10_e5_heldout_test_l40s_reuse100/baseline/eval_rc_sampler_heldout_test_e5
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260623_rc_sampler_cifar10_e5_heldout_test_l40s_reuse100/slurm/rc-e5-test-8g-2751692.out / .err
- 训练结果: 100/100 candidates evaluated; formal_success_candidate 94/100, built_ok 100/100, positive reward 100/100, 5-epoch mean test_acc 93.41%, max 96.92%, backbone effective number 83.08, family effective number 2.54, graph effective number 94.00.
- 主要缺陷: 6 candidates carried formal-evaluation error flags. This is a 5-epoch calibration baseline, not the protocol-matched 1-epoch independent-audit baseline.
- 分析: useful for main-protocol calibration against the 5-epoch RL horizon; keep separate from the 1-epoch SFT-only/RL-after audit comparison.

## 20260623_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s5_ablation_extra_no_repeat_penalty_seed123
- 提交时间: 2026-06-22T20:46:15+0200
- 状态: failed
- main job: 2752062
- 分区/GPU: standard, jnfat01, 5 x L40S
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: no_repeat_penalty
- seed: 123
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- reward/eval GPU rule: NNGPT_SFT_REWARD_EXCLUDE_TRAIN_GPU=1
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (submitted as 8 generations x 125 steps; failed before watcher could stop it)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260623_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s5_ablation_extra_no_repeat_penalty_seed123
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260623_4pattern_a18_cifar10_e5_p3500_c1200_s100_l40s5_ablation_extra_no_repeat_penalty_seed123/slurm/abl-no_repeat_penalty-s123-l40s5-2752062.out / .err
- 训练结果: failed at 112/800 samples. Partial rows: positive reward 109/112, formal_success_candidate 109/112, built_ok 111/112, 5-epoch mean test_acc 88.00%, max 93.28%, reward mean 0.276.
- 主要缺陷: CUDA OOM in trainer training on GPU 0 at 2026-06-23 04:21:56+0200 (`torch.OutOfMemoryError`, requested 470 MiB with only 33 MiB free on a 44.39 GiB L40S).
- 分析: L40S is not reliable for this ablation variant under the current memory profile; a valid second-seed no-repeat-penalty result needs an H100 rerun or should be omitted rather than using this partial failed run.

### Update seed123 reward ablation sample target
- 更新时间: 2026-06-22T21:22:14+0200
- 状态: correction watcher submitted
- watcher job: 2752278
- 原因: Existing seed42 ablation runs contain 800 samples. Seed123 no-diversity/no-repeat reruns were submitted with 8 x 125 = 1000 by mistake; watcher will stop both at the aligned 800-row target and analysis should use 800 rows.

## 20260623_rc_sampler_cifar10_e1_heldout_test_h100x3_reuse100
- 提交时间: 2026-06-22T21:53:48+0200
- 状态: completed
- main job: 2752475
- 分区/GPU: h100, jnultra02, 3 x H100
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- 任务: reused the same 100 rule-constrained sampler candidates; 1-epoch heldout-test evaluation on CIFAR-10 to match the SFT-only/RL-after independent audit protocol
- candidate file: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/home_migrated_20260527/parallel_runs/20260522_baseline_100_v4_audited/baseline/candidates/brute_constrained_random/candidates.jsonl
- eval split: trainvaltest heldout_test, seed 42
- formal reward epochs: 1
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260623_rc_sampler_cifar10_e1_heldout_test_h100x3_reuse100
- eval dir: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260623_rc_sampler_cifar10_e1_heldout_test_h100x3_reuse100/baseline/eval_rc_sampler_heldout_test_e1
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260623_rc_sampler_cifar10_e1_heldout_test_h100x3_reuse100/slurm/rc-e1-test-h100x3-2752475.out / .err
- 训练结果: 100/100 candidates evaluated; formal_success_candidate 100/100, built_ok 100/100, positive reward 100/100, 1-epoch mean test_acc 87.34%, backbone effective number 87.81, family effective number 2.59, graph effective number 100.00.
- 主要缺陷: no flagged runtime or formal-evaluation errors.
- 分析: this is the protocol-matched rule-constrained sampler baseline for the 1-epoch independent audit of SFT-only and RL-after generations.

## 20260623_4pattern_a18_cifar10_e5_p3500_c1200_s800_h100x4_ablation_extra_no_repeat_penalty_seed123
- 提交时间: 2026-06-23T13:09:39+0200
- 状态: submitted after replacing pending 5-GPU job 2754544 before it started
- main job: 2755237
- 分区/GPU: h100, 4 GPU
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: no_repeat_penalty
- seed: 123
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260623_4pattern_a18_cifar10_e5_p3500_c1200_s800_h100x4_ablation_extra_no_repeat_penalty_seed123
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260623_4pattern_a18_cifar10_e5_p3500_c1200_s800_h100x4_ablation_extra_no_repeat_penalty_seed123/slurm/abl-no_repeat_penalty-s123-2755237.out / .err
- 训练结果: submitted; no samples yet.
- 主要缺陷: none observed yet; runtime checks pending until the job starts.
- 分析: same experiment parameters as the canceled pending 5-GPU no-repeat job, except GPU count reduced from 5 to 4 to avoid unnecessary queue delay.

## 20260624_4pattern_a18_cifar10_e5_p3500_c1200_s800_standard_l40s5_ablation_extra_no_diversity_bonus_seed777
- 提交时间: 2026-06-24T11:17:22+0200
- 状态: submitted
- main job: 2759372
- 分区/GPU: standard, 5 GPU
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: no_diversity_bonus
- seed: 777
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260624_4pattern_a18_cifar10_e5_p3500_c1200_s800_standard_l40s5_ablation_extra_no_diversity_bonus_seed777
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260624_4pattern_a18_cifar10_e5_p3500_c1200_s800_standard_l40s5_ablation_extra_no_diversity_bonus_seed777/slurm/abl-no_diversity_bonus-s777-2759372.out / .err
- 训练结果: TODO
- 主要缺陷: TODO
- 分析: TODO
- 状态更新: 2026-06-24T17:06:00+0200 手动取消旧 standard 副本，job 2759372，已运行 05:45:40，已产出 88 samples。原因: L40S standard 24h 内无法完成 e5/s800，改为 H100 replacement job 2760894。

## 20260624_4pattern_a18_cifar10_e5_p3500_c1200_s800_standard_l40s5_ablation_extra_no_repeat_penalty_seed777
- 提交时间: 2026-06-24T11:17:22+0200
- 状态: submitted
- main job: 2759373
- 分区/GPU: standard, 5 GPU
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: no_repeat_penalty
- seed: 777
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260624_4pattern_a18_cifar10_e5_p3500_c1200_s800_standard_l40s5_ablation_extra_no_repeat_penalty_seed777
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260624_4pattern_a18_cifar10_e5_p3500_c1200_s800_standard_l40s5_ablation_extra_no_repeat_penalty_seed777/slurm/abl-no_repeat_penalty-s777-2759373.out / .err
- 训练结果: TODO
- 主要缺陷: TODO
- 分析: TODO
- 状态更新: 2026-06-24T17:06:00+0200 手动取消旧 standard 副本，job 2759373，已运行 05:45:07，已产出 96 samples。原因: L40S standard 24h 内无法完成 e5/s800，改为 gpu_computervision_long replacement job 2760893。

## 20260624_1700_4pattern_a18_cifar10_e5_p3500_c1200_s800_long_l40s5_ablation_extra_no_repeat_penalty_seed777
- 提交时间: 2026-06-24T17:00:44+0200
- 状态: submitted
- main job: 2760893
- 分区/GPU: gpu_computervision_long, 5 GPU
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: no_repeat_penalty
- seed: 777
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260624_1700_4pattern_a18_cifar10_e5_p3500_c1200_s800_long_l40s5_ablation_extra_no_repeat_penalty_seed777
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260624_1700_4pattern_a18_cifar10_e5_p3500_c1200_s800_long_l40s5_ablation_extra_no_repeat_penalty_seed777/slurm/abl-no_repeat_penalty-s777-2760893.out / .err
- 训练结果: TODO
- 主要缺陷: TODO
- 分析: TODO
- 状态更新: 2026-06-24T17:06:00+0200 replacement for cancelled standard job 2759373；优先跑 no_repeat_penalty，因为它对应 seed 间结论冲突更明显的消融项。
- 失败/替换: 2026-06-25，原 job 2760893 在 L40S 上 CUDA OOM，停止在 168/800；旧 partial 不纳入正式结果。已用同参数重投到 4×H100 replacement job 2771080。
- replacement run: 20260625_1448_4pattern_a18_cifar10_e5_p3500_c1200_s800_h100x4_ablation_extra_no_repeat_penalty_seed777

## 20260624_1700_4pattern_a18_cifar10_e5_p3500_c1200_s800_h100x5_ablation_extra_no_diversity_bonus_seed777
- 提交时间: 2026-06-24T17:00:44+0200
- 状态: submitted
- main job: 2760894
- 分区/GPU: h100, 5 GPU
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: no_diversity_bonus
- seed: 777
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260624_1700_4pattern_a18_cifar10_e5_p3500_c1200_s800_h100x5_ablation_extra_no_diversity_bonus_seed777
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260624_1700_4pattern_a18_cifar10_e5_p3500_c1200_s800_h100x5_ablation_extra_no_diversity_bonus_seed777/slurm/abl-no_diversity_bonus-s777-2760894.out / .err
- 训练结果: TODO
- 主要缺陷: TODO
- 分析: TODO
- 状态更新: 2026-06-24T17:06:00+0200 replacement for cancelled standard job 2759372；挂到 H100 等待启动。
- 取消/替换: 2026-06-24，原 job 2760894 误用 5×H100；H100 可共享训练卡，4×H100 足够，因此先提交 replacement job 2760934 后取消旧 job，旧 run 仅产生 8/800 样本，不纳入正式结果。
- replacement run: 20260624_1756_4pattern_a18_cifar10_e5_p3500_c1200_s800_h100x4_ablation_extra_no_diversity_bonus_seed777

## 20260624_1756_4pattern_a18_cifar10_e5_p3500_c1200_s800_h100x4_ablation_extra_no_diversity_bonus_seed777
- 提交时间: 2026-06-24T17:56:38+0200
- 状态: submitted
- main job: 2760934
- 分区/GPU: h100, 4 GPU
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: no_diversity_bonus
- seed: 777
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260624_1756_4pattern_a18_cifar10_e5_p3500_c1200_s800_h100x4_ablation_extra_no_diversity_bonus_seed777
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260624_1756_4pattern_a18_cifar10_e5_p3500_c1200_s800_h100x4_ablation_extra_no_diversity_bonus_seed777/slurm/abl-no_diversity_bonus-s777-2760934.out / .err
- 训练结果: TODO
- 主要缺陷: TODO
- 分析: TODO

## 20260625_1448_4pattern_a18_cifar10_e5_p3500_c1200_s800_h100x4_ablation_extra_no_repeat_penalty_seed777
- 提交时间: 2026-06-25T14:48:06+0200
- 状态: cancelled/replaced
- main job: 2771080
- 分区/GPU: h100, 4 GPU
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: no_repeat_penalty
- seed: 777
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260625_1448_4pattern_a18_cifar10_e5_p3500_c1200_s800_h100x4_ablation_extra_no_repeat_penalty_seed777
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260625_1448_4pattern_a18_cifar10_e5_p3500_c1200_s800_h100x4_ablation_extra_no_repeat_penalty_seed777/slurm/abl-no_repeat_penalty-s777-2771080.out / .err
- 训练结果: 手动取消于约 24/800，释放 H100 节点给 8-GPU replacement；partial 不纳入正式结果。
- 主要缺陷: 未完成全量 800 样本，只作为 replacement 前序记录保留。
- 分析: 正式 seed777 no-repeat 结果以后续 8-GPU job 2771489 为准。

## 20260625_1535_4pattern_a18_cifar10_e5_p3500_c1200_s800_h100x8_ablation_extra_no_repeat_penalty_seed777
- 提交时间: 2026-06-25T15:35:28+0200
- 状态: completed
- main job: 2771489
- 分区/GPU: h100, 8 GPU
- mem/cpus: 160G, 32
- commit: c91714dbe7dad1d02a9080243945bbf8e8ec9300 (Simplify reward memory preflight)
- reward variant: no_repeat_penalty
- seed: 777
- formal dataset: cifar-10
- init adapter: /home/s471802/nn-gpt/out/nngpt/llm/epoch_sft_20260527_1410_struct1_v2_sftcycle_h100x4_home/A18/deepseek-ai/deepseek-coder-6.7b-instruct
- prompt/prefix: sft_aligned, rl-bb-struct1,rl-bb-struct1-v2
- formal reward epochs: 5
- max prompt length: 3500
- max completion length: 1200
- generation batch size: 8
- samples target: 800 (8 generations x 100 steps)
- run root: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260625_1535_4pattern_a18_cifar10_e5_p3500_c1200_s800_h100x8_ablation_extra_no_repeat_penalty_seed777
- stdout/stderr: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260625_1535_4pattern_a18_cifar10_e5_p3500_c1200_s800_h100x8_ablation_extra_no_repeat_penalty_seed777/slurm/abl-no_repeat_penalty-s777-2771489.out / .err
- 训练结果: 正常完成 800/800，Slurm COMPLETED 0:0，耗时 10:42:47。final-100 formal success 96/100，mean 5-epoch reward-eval accuracy 89.6%（bootstrap CI 87.7--90.8%），Block Eff. 6.08，Family Top-1 39.6%，Graph Eff. 18.27；全轨迹 formal success 765/800。
- 主要缺陷: seed777 no-repeat 不复现 seed42/123 的单 family final-window collapse，显示 repeat ablation 的结构效应有明显 seed dependence。
- 分析: 已纳入 article 的 Table 10；结论保持保守，强调 reliability/structural-exploration tradeoff 而非简单 accuracy effect。
## 20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114

<!-- NNGPT_RUN:20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114:START -->
- 运行 ID：`20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114`
- 标签：`4p-full-s114`
- 状态：已结束
- 提交时间：`2026-06-26T18:59:46+02:00`
- 开始时间：`2026-06-26T18:59:56+02:00`
- 结束时间：`2026-06-26T20:56:53+02:00`
- Job ID：`2776904`
- 分区 / QoS：`h100`
- 节点：`jnultra01`
- 提交 commit：`c91714dbe7dad1d02a9080243945bbf8e8ec9300 Simplify reward memory preflight`
- 本次改动：Article main 5-seed expansion: 4-pattern A18 CIFAR-10 5-epoch full-reward seed 114; matched to existing seeds 42/123/777.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114/slurm/abl-full_reward-s114-2776904.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114/slurm/abl-full_reward-s114-2776904.err`
- 工作目录：`/tmp/s471802/20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114`
<!-- NNGPT_RUN:20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114:END -->
## 20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514

<!-- NNGPT_RUN:20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514:START -->
- 运行 ID：`20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514`
- 标签：`4p-full-s514`
- 状态：已结束
- 提交时间：`2026-06-26T18:59:46+02:00`
- 开始时间：`2026-06-26T18:59:56+02:00`
- 结束时间：`2026-06-27T06:43:19+02:00`
- Job ID：`2776906`
- 分区 / QoS：`h100`
- 节点：`jnultra01`
- 提交 commit：`c91714dbe7dad1d02a9080243945bbf8e8ec9300 Simplify reward memory preflight`
- 本次改动：Article main 5-seed expansion: 4-pattern A18 CIFAR-10 5-epoch full-reward seed 514; matched to existing seeds 42/123/777.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514/slurm/abl-full_reward-s514-2776906.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514/slurm/abl-full_reward-s514-2776906.err`
- 工作目录：`/tmp/s471802/20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514`
<!-- NNGPT_RUN:20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514:END -->
## 20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114

<!-- NNGPT_RUN:20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114:START -->
- 运行 ID：`20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114`
- 标签：`1p-full-s114`
- 状态：已结束
- 提交时间：`2026-06-26T19:00:12+02:00`
- 开始时间：`2026-06-26T20:57:02+02:00`
- 结束时间：`2026-06-27T06:47:46+02:00`
- Job ID：`2776908`
- 分区 / QoS：`h100`
- 节点：`jnultra01`
- 提交 commit：`c91714dbe7dad1d02a9080243945bbf8e8ec9300 Simplify reward memory preflight`
- 本次改动：Article main 5-seed expansion: 1-pattern A9 CIFAR-10 5-epoch full-reward seed 114; matched to existing seeds 42/123/777.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114/slurm/abl-full_reward-s114-2776908.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114/slurm/abl-full_reward-s114-2776908.err`
- 工作目录：`/tmp/s471802/20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114`
<!-- NNGPT_RUN:20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed114:END -->
## 20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514

<!-- NNGPT_RUN:20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514:START -->
- 运行 ID：`20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514`
- 标签：`1p-full-s514`
- 状态：已结束
- 提交时间：`2026-06-26T19:00:13+02:00`
- 开始时间：`2026-06-26T23:00:30+02:00`
- 结束时间：`2026-06-27T10:07:07+02:00`
- Job ID：`2776910`
- 分区 / QoS：`h100`
- 节点：`jnultra02`
- 提交 commit：`c91714dbe7dad1d02a9080243945bbf8e8ec9300 Simplify reward memory preflight`
- 本次改动：Article main 5-seed expansion: 1-pattern A9 CIFAR-10 5-epoch full-reward seed 514; matched to existing seeds 42/123/777.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514/slurm/abl-full_reward-s514-2776910.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514/slurm/abl-full_reward-s514-2776910.err`
- 工作目录：`/tmp/s471802/20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514`
<!-- NNGPT_RUN:20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed514:END -->
## 20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919

<!-- NNGPT_RUN:20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919:START -->
- 运行 ID：`20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919`
- 标签：`4p_full_s919`
- 状态：已结束
- 提交时间：`2026-06-27T10:51:20+02:00`
- 开始时间：`2026-06-27T10:51:32+02:00`
- 结束时间：`2026-06-27T20:23:11+02:00`
- Job ID：`2777817`
- 分区 / QoS：`h100`
- 节点：`jnultra01`
- 提交 commit：`c91714dbe7dad1d02a9080243945bbf8e8ec9300 Simplify reward memory preflight`
- 本次改动：Reserve 4-pattern full-reward 5-epoch CIFAR-10 seed 919 after seed114 syntax-template collapse; keep balanced five valid 4-pattern seeds for article seed-level variance.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919/slurm/abl-full_reward-s919-2777817.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919/slurm/abl-full_reward-s919-2777817.err`
- 工作目录：`/tmp/s471802/20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919`
<!-- NNGPT_RUN:20260627_4pattern_a18_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919:END -->
## 20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919

<!-- NNGPT_RUN:20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919:START -->
- 运行 ID：`20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919`
- 标签：`1p_full_s919`
- 状态：已结束
- 提交时间：`2026-06-27T12:17:47+02:00`
- 开始时间：`2026-06-27T12:18:00+02:00`
- 结束时间：`2026-06-28T02:37:24+02:00`
- Job ID：`2777842`
- 分区 / QoS：`h100`
- 节点：`jnultra01`
- 提交 commit：`c91714dbe7dad1d02a9080243945bbf8e8ec9300 Simplify reward memory preflight`
- 本次改动：Article main 5-seed expansion: 1-pattern A9 CIFAR-10 5-epoch full-reward seed 919; matched to existing 4-pattern seed 919 after seed514 structural-collapse discussion.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919/slurm/tunerl-1p_full_s919-2777842.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919/slurm/tunerl-1p_full_s919-2777842.err`
- 工作目录：`/tmp/s471802/20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919`
<!-- NNGPT_RUN:20260627_1pattern_a9_cifar10_e5_p3500_c1200_s100_h100x4_shared_full_full_reward_seed919:END -->

## 20260704_qwen_cifar10_e5_seed42

<!-- NNGPT_RUN:20260704_qwen_cifar10_e5_seed42:START -->
- 运行 ID：`20260704_qwen_cifar10_e5_seed42`
- 标签：`qwen-c10-e5-s42`
- 状态：已提交，等待/运行 H100
- 提交时间：`2026-07-04T04:27:21+02:00`
- 开始时间：待完成后核对
- 结束时间：待完成
- Job ID：`2791927`
- 分区 / QoS：`h100`
- 节点：待完成后核对
- 提交 commit：`3c4843f5 Use backbone data filters during generation`
- 本次改动：Qwen CIFAR-10 5-epoch full-reward 3-seed补实验；seed 42；4 GPU；不排除训练卡作为 reward/eval worker。旧 20260703 同 seed job 为 sbatch 引号错误，无训练样本。
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260704_qwen_cifar10_e5_seed42`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260704_qwen_cifar10_e5_seed42/slurm/qwen-c10-e5-s42-2791927.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260704_qwen_cifar10_e5_seed42/slurm/qwen-c10-e5-s42-2791927.err`
- 工作目录：`/tmp/s471802/20260704_qwen_cifar10_e5_seed42/nn-gpt`
- 初始恢复来源：Qwen A7 SFT adapter `/home/s471802/nn-gpt/out/nngpt/llm/20260601_1pattern_three_model_sft_v3_cifar10_qwen7b/epoch_sft/A7/Qwen/Qwen2.5-Coder-7B-Instruct`
- 训练结果：待手写
- 主要缺陷：待手写
- 分析：待完成后补充
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260704_qwen_cifar10_e5_seed42`
<!-- NNGPT_RUN:20260704_qwen_cifar10_e5_seed42:END -->

## 20260704_qwen_cifar10_e5_seed123

<!-- NNGPT_RUN:20260704_qwen_cifar10_e5_seed123:START -->
- 运行 ID：`20260704_qwen_cifar10_e5_seed123`
- 标签：`qwen-c10-e5-s123`
- 状态：已提交，等待/运行 H100
- 提交时间：`2026-07-04T04:27:21+02:00`
- 开始时间：待完成后核对
- 结束时间：待完成
- Job ID：`2791928`
- 分区 / QoS：`h100`
- 节点：待完成后核对
- 提交 commit：`3c4843f5 Use backbone data filters during generation`
- 本次改动：Qwen CIFAR-10 5-epoch full-reward 3-seed补实验；seed 123；4 GPU；不排除训练卡作为 reward/eval worker。旧 20260703 同 seed job 为 sbatch 引号错误，无训练样本。
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260704_qwen_cifar10_e5_seed123`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260704_qwen_cifar10_e5_seed123/slurm/qwen-c10-e5-s123-2791928.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260704_qwen_cifar10_e5_seed123/slurm/qwen-c10-e5-s123-2791928.err`
- 工作目录：`/tmp/s471802/20260704_qwen_cifar10_e5_seed123/nn-gpt`
- 初始恢复来源：Qwen A7 SFT adapter `/home/s471802/nn-gpt/out/nngpt/llm/20260601_1pattern_three_model_sft_v3_cifar10_qwen7b/epoch_sft/A7/Qwen/Qwen2.5-Coder-7B-Instruct`
- 训练结果：待手写
- 主要缺陷：待手写
- 分析：待完成后补充
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260704_qwen_cifar10_e5_seed123`
<!-- NNGPT_RUN:20260704_qwen_cifar10_e5_seed123:END -->

## 20260704_qwen_cifar10_e5_seed777

<!-- NNGPT_RUN:20260704_qwen_cifar10_e5_seed777:START -->
- 运行 ID：`20260704_qwen_cifar10_e5_seed777`
- 标签：`qwen-c10-e5-s777`
- 状态：已提交，等待/运行 H100
- 提交时间：`2026-07-04T04:27:21+02:00`
- 开始时间：待完成后核对
- 结束时间：待完成
- Job ID：`2791929`
- 分区 / QoS：`h100`
- 节点：待完成后核对
- 提交 commit：`3c4843f5 Use backbone data filters during generation`
- 本次改动：Qwen CIFAR-10 5-epoch full-reward 3-seed补实验；seed 777；4 GPU；不排除训练卡作为 reward/eval worker。旧 20260703 同 seed job 为 sbatch 引号错误，无训练样本。
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260704_qwen_cifar10_e5_seed777`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260704_qwen_cifar10_e5_seed777/slurm/qwen-c10-e5-s777-2791929.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260704_qwen_cifar10_e5_seed777/slurm/qwen-c10-e5-s777-2791929.err`
- 工作目录：`/tmp/s471802/20260704_qwen_cifar10_e5_seed777/nn-gpt`
- 初始恢复来源：Qwen A7 SFT adapter `/home/s471802/nn-gpt/out/nngpt/llm/20260601_1pattern_three_model_sft_v3_cifar10_qwen7b/epoch_sft/A7/Qwen/Qwen2.5-Coder-7B-Instruct`
- 训练结果：待手写
- 主要缺陷：待手写
- 分析：待完成后补充
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260704_qwen_cifar10_e5_seed777`

- 续跑记录：`2026-07-05T11:30:05+02:00` 提交补跑 Job `2793568`（`qwen-c10-e5-s777-resume`，h100/jnultra02，4 GPU），使用同一 run_id 从 `grpo_backbone_outputs/checkpoints/stage2_formal_explore` stage checkpoint 恢复，起点 `760/800`，设置 `NNGPT_SFT_MAX_STEPS=5` 补剩余 40 条样本；提交 commit `00a25549a02b8a733efbfc04093263cb0b776493 Remove temporary resume serialization test`。Job `2793568` 已于 `2026-07-05T12:34:04+02:00` 正常完成，Slurm `COMPLETED 0:0`，最终 `800/800`。
- 续跑前处理：Job `2793551` 因 resume 脚本仍是 fresh 配置被取消，未改动样本文件；Job `2793555` 因 stage resume 写 `run_config.json` 时 `Path` 不能 JSON 序列化失败，未追加样本；已在 `20260704_qwen_cifar10_e5_seed777_backup_before_resume_20260705_1117` 备份 760 条结果后继续。
<!-- NNGPT_RUN:20260704_qwen_cifar10_e5_seed777:END -->

## backbone_1536_struct_10cycle_20260709_132615

<!-- NNGPT_RUN:backbone_1536_struct_10cycle_20260709_132615:START -->
- 运行 ID：`backbone_1536_struct_10cycle_20260709_132615`
- 标签：`TuneBackbone 1536-token 10-cycle functionality check`
- 状态：运行中
- 提交时间：`2026-07-09T13:26:15+02:00`
- 开始时间：`2026-07-09T13:26:xx+02:00`（已启动，精确时间待完成后核对）
- 结束时间：待完成
- Job ID：`2803086`
- 分区 / QoS：`standard`
- 节点：`jn001`
- GPU：`3 x L40`
- 提交 commit：`d8cf7fc09 Limit backbone SFT generation tokens`
- 本次改动：TuneBackbone 10-cycle 功能验证；`max_new_tokens=1536`；SFT prefixes `rl-bb-struct1,rl-bb-struct1-v2`；generated prefix `rl-bb-struct1-v2-dscoder7b-sftcycle-10cycle1536`；每轮生成 `test_nn=9`；SFT dataset `cifar-10`；未设置 `sft_dataset_limit`；NNEval 多卡并行，每卡 1 worker。
- 输出目录：`/home/s471802/nn-gpt/out/sft_formal/backbone_1536_struct_10cycle_20260709_132615`
- 标准输出：`/home/s471802/nn-gpt/slurm_logs/backbone_1536_struct_10cycle_20260709_132615-2803086.out`
- 标准错误：`/home/s471802/nn-gpt/slurm_logs/backbone_1536_struct_10cycle_20260709_132615-2803086.err`
- 工作目录：`/home/s471802/nn-gpt`
- 训练结果：待完成后手写
- 主要缺陷：待完成后手写
- 分析：待完成后补充
<!-- NNGPT_RUN:backbone_1536_struct_10cycle_20260709_132615:END -->
## 20260718_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42

<!-- NNGPT_RUN:20260718_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42:START -->
- 运行 ID：`20260718_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42`
- 标签：`1p-c100-e5-s42`
- 状态：已结束
- 提交时间：`2026-07-17T19:07:39+02:00`
- 开始时间：`2026-07-17T19:10:12+02:00`
- 结束时间：`2026-07-18T13:01:01+02:00`
- Job ID：`2866647`
- 分区 / QoS：`h100`
- 节点：`jnultra02`
- 提交 commit：`64c54fb1f41d7ca253d319df62658c951ed112a7 Update README.md`
- 本次改动：Matched second-dataset RL run: 1-pattern DeepSeek A9 on CIFAR-100, seed 42; same 5-epoch full-reward 800-sample protocol as the CIFAR-10 six-seed runs; nn-dataset commit 6dbf65b9a3bf6ed93cb1f655d2110276312f1157.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260718_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260718_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42/slurm/rl-1p-c100-s42-2866647.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260718_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42/slurm/rl-1p-c100-s42-2866647.err`
- 工作目录：`/tmp/s471802/20260718_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：待手写
- 主要缺陷：待手写
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260718_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42`
<!-- NNGPT_RUN:20260718_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42:END -->
## 20260720_1pattern_a9_cifar100_e5_p3500_c1200_s100_l40sx4_excl_full_reward_seed123

<!-- NNGPT_RUN:20260720_1pattern_a9_cifar100_e5_p3500_c1200_s100_l40sx4_excl_full_reward_seed123:START -->
- 运行 ID：`20260720_1pattern_a9_cifar100_e5_p3500_c1200_s100_l40sx4_excl_full_reward_seed123`
- 标签：`1p-c100-e5-s123`
- 状态：已结束(FAILED)
- 提交时间：`2026-07-20T06:52:56+02:00`
- 开始时间：`2026-07-20T06:55:32+02:00`
- 结束时间：`2026-07-20T14:56:55+02:00`
- Job ID：`2878976`
- 分区 / QoS：`gpu_computervision_long`
- 节点：`jnfat07`
- 提交 commit：`64c54fb1f41d7ca253d319df62658c951ed112a7 Update README.md`
- 本次改动：Matched second-dataset RL run: 1-pattern DeepSeek A9 on CIFAR-100, seed 123; same 5-epoch full-reward 800-sample protocol as seed 42; L40S training GPU excluded from reward workers; nn-dataset commit 6dbf65b9a3bf6ed93cb1f655d2110276312f1157.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260720_1pattern_a9_cifar100_e5_p3500_c1200_s100_l40sx4_excl_full_reward_seed123`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260720_1pattern_a9_cifar100_e5_p3500_c1200_s100_l40sx4_excl_full_reward_seed123/slurm/rl-1p-c100-s123-2878976.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260720_1pattern_a9_cifar100_e5_p3500_c1200_s100_l40sx4_excl_full_reward_seed123/slurm/rl-1p-c100-s123-2878976.err`
- 工作目录：`/tmp/s471802/20260720_1pattern_a9_cifar100_e5_p3500_c1200_s100_l40sx4_excl_full_reward_seed123/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：完成 104/800 条；formal_success_candidate 80/104，built_ok 80/104，positive reward 74/104，reward 均值 -0.1948，5-epoch test_acc 均值 70.12%、最高 74.86%，timeout 2。
- 主要缺陷：job 2878976 在 reward batch 14 的并行评估中无 Python traceback、OOM、worker restart 或 Slurm kill 信息地突然终止；随后确认登录、CPU 和 GPU 节点的 /data Ceph 挂载同时缺失。run 未到 save_steps=25，未生成 trainer/stage checkpoint，不能可靠续跑；保留 104 条数据仅用于诊断，并以 fresh seed123 replacement 重跑。
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260720_1pattern_a9_cifar100_e5_p3500_c1200_s100_l40sx4_excl_full_reward_seed123`
<!-- NNGPT_RUN:20260720_1pattern_a9_cifar100_e5_p3500_c1200_s100_l40sx4_excl_full_reward_seed123:END -->
## 20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123_rerun

<!-- NNGPT_RUN:20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123_rerun:START -->
- 运行 ID：`20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123_rerun`
- 标签：`1p-c100-e5-s123-rerun`
- 状态：已结束
- 提交时间：`2026-07-23T12:46:14+02:00`
- 开始时间：`2026-07-23T12:48:52+02:00`
- 结束时间：`2026-07-24T00:56:13+02:00`
- Job ID：`2943044`
- 分区 / QoS：`h100`
- 节点：`jnultra02`
- 提交 commit：`64c54fb1f41d7ca253d319df62658c951ed112a7 Update README.md`
- 本次改动：Replacement for job 2878976 after the shared /data Ceph mount outage; matched 1-pattern DeepSeek A9 CIFAR-100 seed123 run using the same 5-epoch full-reward 800-sample protocol and H100 shared-GPU layout as completed seed42; nn-dataset commit 6dbf65b9a3bf6ed93cb1f655d2110276312f1157.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123_rerun`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123_rerun/slurm/rl-1p-c100-s123-rerun-2943044.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123_rerun/slurm/rl-1p-c100-s123-rerun-2943044.err`
- 工作目录：`/tmp/s471802/20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123_rerun/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：800/800；formal_success_candidate 772/800（96.50%），built_ok 772/800，positive reward 757/800（94.63%），reward 均值 0.3176；有效 5-epoch test_acc 772 条，均值 69.15%，最高 74.86%；0 timeout，0 worker restart；Slurm COMPLETED 0:0。
- 主要缺陷：结构严重坍缩：最终 dominant family share 99.35%，ParallelTriple_Shallow 767 条；仅 4 families / 4 skeletons。28 条生成存在错误，但未出现 OOM、进程 traceback、killed 或 reward worker 重启。
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123_rerun`
<!-- NNGPT_RUN:20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123_rerun:END -->
## 20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777

<!-- NNGPT_RUN:20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777:START -->
- 运行 ID：`20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777`
- 标签：`1p-c100-e5-s777`
- 状态：已结束
- 提交时间：`2026-07-23T12:51:06+02:00`
- 开始时间：`2026-07-23T12:56:32+02:00`
- 结束时间：`2026-07-24T11:34:05+02:00`
- Job ID：`2944149`
- 分区 / QoS：`h100`
- 节点：`jnultra02`
- 提交 commit：`64c54fb1f41d7ca253d319df62658c951ed112a7 Update README.md`
- 本次改动：Parallel CIFAR-100 replication for the 1-pattern DeepSeek A9 condition; seed 777; matched 5-epoch full-reward 800-sample protocol and H100 shared-GPU layout; nn-dataset commit 6dbf65b9a3bf6ed93cb1f655d2110276312f1157.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777/slurm/rl-1p-c100-s777-2944149.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777/slurm/rl-1p-c100-s777-2944149.err`
- 工作目录：`/tmp/s471802/20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777/nn-gpt`
- 初始恢复来源：none; fresh base model with stage override: stage2_formal_explore
- 训练结果：800/800；formal_success_candidate 735/800（91.88%），built_ok 738/800，positive reward 681/800（85.13%），reward 均值 0.1156；有效 5-epoch test_acc 737 条，均值 71.86%，最高 83.62%；0 timeout，0 worker restart；Slurm COMPLETED 0:0。
- 主要缺陷：结构明显坍缩：最终 dominant family share 96.05%，ParallelTriple_Shallow 728 条；5 families / 13 skeletons。63 条记录含生成或评估错误，但未出现 OOM、进程 traceback、killed 或 reward worker 重启。
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777`
<!-- NNGPT_RUN:20260723_1pattern_a9_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777:END -->
## 20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42

<!-- NNGPT_RUN:20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42:START -->
- 运行 ID：`20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42`
- 标签：`4p_c100_e5_s42`
- 状态：已人工停止；正式结果完成（Slurm 因 TERM 记为 FAILED）
- 提交时间：`2026-07-26T05:13:18+02:00`
- 开始时间：`2026-08-07T11:07:22+02:00`
- 结束时间：`2026-08-07T16:29:31+02:00`
- Job ID：`3043789`
- 分区 / QoS：`h100 / computervision`
- 节点：`jnultra01`
- 提交 commit：`64c54fb1f41d7ca253d319df62658c951ed112a7 Update README.md`
- 本次改动：Matched second-dataset RL run: 4-pattern DeepSeek A18 on CIFAR-100, seed 42; 5-epoch full-reward 800-sample protocol; H100 shared train/reward GPU layout; retained on H100 because long partition quota is unavailable, gpu_computervision starts later, and standard cannot reliably finish within 24 hours; nn-dataset commit 6dbf65b9a3bf6ed93cb1f655d2110276312f1157.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42/slurm/rl-4p-c100-s42-q-3043789.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42/slurm/rl-4p-c100-s42-q-3043789.err`
- 工作目录：`/tmp/s471802/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42/3043789.0.3615415/nn-gpt`
- 初始恢复来源：本 run trainer checkpoint: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42/grpo_backbone_outputs_trainer/checkpoint-50
- 训练结果：正式口径取 generation_samples.jsonl 前 800 条：formal_success_candidate 699/800，built_ok 711/800，positive reward 688/800，reward 均值 0.3588；有效 5-epoch test_acc 均值 71.67%、最高 77.76%。job 3043789 从 693 条恢复后错误地把 100 steps 当作新增步数，写到 909 条才人工 TERM；后 109 条不纳入正式结果。
- 主要缺陷：旧 replacement 2968439 使用 normal QoS，并在抢占/重排后的临时工作目录重建期间与残留 reward worker 发生路径冲突，out 链接消失后 FileNotFoundError；连续 launch failure 达到 Slurm 上限后 held。runner 已改用每次启动独立 workdir，job 3043789 成功连续恢复，但 resume 未按已有样本扣减剩余 steps，导致超过 800 目标；Slurm FAILED 仅由人工 TERM 产生，未出现 OOM、timeout、traceback、killed 或 worker restart。
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42`
<!-- NNGPT_RUN:20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed42:END -->
## 20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123

<!-- NNGPT_RUN:20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123:START -->
- 运行 ID：`20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123`
- 标签：`4p_c100_e5_s123`
- 状态：已人工停止；正式结果完成（Slurm 因 TERM 记为 FAILED）
- 提交时间：`2026-07-26T05:13:18+02:00`
- 开始时间：`2026-08-07T11:32:00+02:00`
- 结束时间：`2026-08-07T16:29:31+02:00`
- Job ID：`3043791`
- 分区 / QoS：`h100 / computervision`
- 节点：`jnultra02`
- 提交 commit：`64c54fb1f41d7ca253d319df62658c951ed112a7 Update README.md`
- 本次改动：Matched second-dataset RL run: 4-pattern DeepSeek A18 on CIFAR-100, seed 123; 5-epoch full-reward 800-sample protocol; H100 shared train/reward GPU layout; retained on H100 because long partition quota is unavailable, gpu_computervision starts later, and standard cannot reliably finish within 24 hours; nn-dataset commit 6dbf65b9a3bf6ed93cb1f655d2110276312f1157.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123/slurm/rl-4p-c100-s123-q-3043791.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123/slurm/rl-4p-c100-s123-q-3043791.err`
- 工作目录：`/tmp/s471802/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123/3043791.0.3319969/nn-gpt`
- 初始恢复来源：本 run trainer checkpoint: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123/grpo_backbone_outputs_trainer/checkpoint-50
- 训练结果：正式口径取 generation_samples.jsonl 前 800 条：formal_success_candidate 671/800，built_ok 750/800，positive reward 649/800，reward 均值 0.3421；有效 5-epoch test_acc 均值 65.87%、最高 77.48%。job 3043791 从 720 条恢复后错误地把 100 steps 当作新增步数，写到 944 条才人工 TERM；后 144 条不纳入正式结果。
- 主要缺陷：旧 replacement 2968441 在重排启动阶段退出；同系列 runner 存在临时 workdir/out 路径冲突。runner 已改用每次启动独立 workdir，job 3043791 成功连续恢复，但 resume 未按已有样本扣减剩余 steps，导致超过 800 目标；Slurm FAILED 仅由人工 TERM 产生，未出现 OOM、timeout、traceback、killed 或 worker restart。
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123`
<!-- NNGPT_RUN:20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed123:END -->
## 20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777

<!-- NNGPT_RUN:20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777:START -->
- 运行 ID：`20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777`
- 标签：`4p_c100_e5_s777`
- 状态：已人工停止；正式结果完成（Slurm 因 TERM 记为 FAILED）
- 提交时间：`2026-07-26T05:13:18+02:00`
- 开始时间：`2026-08-07T11:32:00+02:00`
- 结束时间：-
- Job ID：`3043793`
- 分区 / QoS：`h100 / computervision`
- 节点：`jnultra02`
- 提交 commit：`64c54fb1f41d7ca253d319df62658c951ed112a7 Update README.md`
- 本次改动：Matched second-dataset RL run: 4-pattern DeepSeek A18 on CIFAR-100, seed 777; 5-epoch full-reward 800-sample protocol; H100 shared train/reward GPU layout; retained on H100 because long partition quota is unavailable, gpu_computervision starts later, and standard cannot reliably finish within 24 hours; nn-dataset commit 6dbf65b9a3bf6ed93cb1f655d2110276312f1157.
- 输出目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777`
- 标准输出：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777/slurm/rl-4p-c100-s777-q-3043793.out`
- 标准错误：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777/slurm/rl-4p-c100-s777-q-3043793.err`
- 工作目录：`/tmp/s471802/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777/3043793.0.3319968/nn-gpt`
- 初始恢复来源：本 run trainer checkpoint: /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777/grpo_backbone_outputs_trainer/checkpoint-50
- 训练结果：完成 800/800；formal_success_candidate 726/800，built_ok 783/800，positive reward 723/800，reward 均值 0.4677；有效 5-epoch test_acc 均值 71.19%、最高 78.36%。watcher job 3045381 在达到 800 条后触发 TERM；Slurm FAILED 仅由人工 TERM 产生，正式结果使用全部 800 条。
- 主要缺陷：旧 replacement 2968443 使用 normal QoS，并在抢占/重排后的临时工作目录重建期间与残留 reward worker 发生路径冲突，out 链接消失后 FileNotFoundError；连续 launch failure 达到 Slurm 上限后 held。runner 已改用每次启动独立 workdir，job 3043793 以默认 computervision QoS、H100 4 卡从 checkpoint-50 连续恢复；未出现 OOM、timeout、traceback、killed 或 worker restart。
- 保留目录：`/data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs/20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777`
<!-- NNGPT_RUN:20260726_4pattern_a18_cifar100_e5_p3500_c1200_s100_h100x4_shared_full_reward_seed777:END -->
