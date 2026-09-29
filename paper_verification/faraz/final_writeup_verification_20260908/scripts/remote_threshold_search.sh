#!/usr/bin/env bash
set -euo pipefail

# Read-only provenance scan. Supply the password through SSHPASS; this script
# neither contains nor prints credentials.
host="${1:-s471802@10.106.223.223}"
out_dir="${2:-final_writeup_verification_20260908/remote_evidence/threshold/raw_search}"
ssh_base=(sshpass -e ssh -o HostKeyAlias=julia2.hpc.uni-wuerzburg.de -o StrictHostKeyChecking=yes "$host")
mkdir -p "$out_dir"

run_remote() {
  local label="$1"
  local command="$2"
  local stdout_file="$out_dir/${label}.stdout.txt"
  local stderr_file="$out_dir/${label}.stderr.txt"
  local status_file="$out_dir/${label}.status.txt"
  printf '%s\n' "$command" >"$out_dir/${label}.command.txt"
  set +e
  "${ssh_base[@]}" "$command" >"$stdout_file" 2>"$stderr_file"
  local rc=$?
  set -e
  printf '%s\n' "$rc" >"$status_file"
}

run_remote filename_search 'find /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs -type f \( -iname "*threshold*" -o -iname "*sensitivity*" -o -iname "*canonical*" -o -iname "*granular*" -o -iname "*collapse*" \) -printf "%p\n"'

run_remote source_content_search 'timeout 60 sh -c '\''find /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs -type f -size -10M \( -name "*.py" -o -name "*.sh" -o -name "*.md" \) -print0 | xargs -0 grep -IlE "threshold[_ -]?sensitivity|70[^[:alnum:]]+80[^[:alnum:]]+90|four of twelve|4(/| of )12|collapse.*(twelve|12)|(twelve|12).*collapse"'\'''

run_remote summary_content_search 'timeout 60 sh -c '\''find /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/parallel_runs -type f -size -10M \( -name "*.json" -o -name "*.csv" -o -name "*.txt" \) ! -path "*/tokenizer/*" ! -path "*/checkpoint-*/*" ! -name "generation_samples.jsonl" -print0 | xargs -0 grep -IlE "threshold[_ -]?sensitivity|four of twelve|4(/| of )12|collapse.*(twelve|12)|(twelve|12).*collapse"'\'''

run_remote archive_search 'timeout 60 find /data/42-julia-hpc-ai-cv-students/s471802 /home/s471802 -type f \( -name "*.zip" -o -name "*.tar.gz" -o -name "*.tgz" \) -printf "%p\n"'

run_remote history_search 'for f in /home/s471802/.bash_history /home/s471802/.zsh_history; do test -r "$f" && grep -inE "threshold[_ -]?sensitivity|70[^[:alnum:]]+80[^[:alnum:]]+90|four of twelve|4(/| of )12" "$f" || true; done'

run_remote git_history_search 'cd /home/s471802/nn-gpt && git log --all --oneline --pickaxe-all -S "threshold_sensitivity_70_80_90" -- . ":(exclude)*.jsonl"'

run_remote nearby_listing 'find /data/42-julia-hpc-ai-cv-students/s471802/nn-gpt-runs/paper_followup_20260727 -maxdepth 1 -type f -printf "%p\t%s\t%TY-%Tm-%TdT%TH:%TM:%TS%Tz\n" | sort'
