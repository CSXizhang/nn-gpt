# Source recovery after correspondence and session review

The September manuscript correspondence identifies the June 7 Imagenette
dual-backbone evaluation as the source of the Section 5.12 dual accuracies.
That run (`20260607_1455_gen30min32_formal100_dsqwenolympic`) and the June 8
single-backbone run are already in this archive with their raw evaluations and
configs. The June 24 3x3 re-evaluation is a distinct result source.

The June project session referred to 507 serialized four-pattern SFT cycle
records, while the September verification reply stated that no row-level cycle
manifest had been recovered. A fresh read-only check of Julia2 found exactly
507 `rl-bb-struct1-v2-sftcycle-*.py` code files in
`/home/s471802/nn-gpt/out/nngpt/new_lemur/nn/` and exactly 507 matching
`img-classification_cifar-10_acc_rl-bb-struct1-v2-sftcycle-*/1.json`
evaluation records in the `train/` sibling directory. The suffixes match
one-to-one; all 507 JSON records parse. The 1,014 original files are in
`datasets/derived/sft_cycle_507/cycle_507_source.tar.zst`, with per-file
source path and SHA-256 in `cycle_507_manifest.tsv`. A credential-pattern scan
of those files found no matches. The files were pushed from Julia2 without
altering their source locations.

This recovers the 507 generated code/evaluation pairs and independently
supports the reported cycle-output count. It does **not** prove that all 507
were consumed as trainer input or reproduce a training-time filtered dataset
manifest. The earlier count must not be described as 507 confirmed trainer
rows without an additional training log or formatter audit.

The September reply to Faraz explicitly failed to recover the older Qwen
seed-777 graph value 14.33. It recomputed 14.29 using inverse Simpson over
`graph_hash` for the last 100 formal successes; the standard final-window
Shannon calculations are separately documented in the Faraz package. Treat
14.33 as an unreconciled historical value, not as a missing raw trajectory.

The six original CIFAR-10 run configs record `git.dirty=true` relative to
`c91714dbe7dad1d02a9080243945bbf8e8ec9300`. Neither the correspondence
nor the reviewed historical sessions yielded the exact patch; logged reward
fields agree with the clean added-seed runs in the discrete values audited for
the manuscript, but byte-for-byte source identity remains unproven.

The two proxy directories have their exact preparation/evaluation scripts,
candidate manifests and outputs archived. Historical sessions show the
top-20 evaluation script was revised inside its run directory during retries;
no authoritative Git commit for either proxy run was identified. Use those
archived scripts as the executable source snapshots and leave the commit field
blank.
