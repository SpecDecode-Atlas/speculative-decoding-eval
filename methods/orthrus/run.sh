#!/usr/bin/env bash
# One shell runs one Orthrus experiment stage. Defaults target a single Ascend A2 node.
set -Ee -o pipefail
mode=${1:---prepare-only}
case "$mode" in
    --prepare-only|--check-only|--benchmark|--run|--evaluate) ;;
    *) echo "Usage: bash run.sh [--prepare-only|--check-only|--benchmark|--run|--evaluate]" >&2; exit 2 ;;
esac
shift || true
if [[ "$mode" != --evaluate && $# -gt 0 ]]; then echo "Unexpected arguments: $*" >&2; exit 2; fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "$here"
BASE=${BASE:-/home/n84449292/a84384644}
MODEL=${MODEL:-$BASE/models/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}
DATASET=${DATASET:-$BASE/datasets/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072.dedup}
if [[ -z "${CONDA_ENV:-}" && "${CONDA_PREFIX##*/}" == spec_decode_atlas ]]; then CONDA_ENV=$CONDA_PREFIX; fi
CONDA_ENV=${CONDA_ENV:-/home/n84449292/miniforge3/envs/spec_decode_atlas}
for init in "${CONDA_SH:-}" /home/n84449292/miniforge3/etc/profile.d/conda.sh /home/n84449292/m84379596/conda/miniforge3/etc/profile.d/conda.sh; do
    if [[ -n "$init" && -r "$init" ]]; then source "$init"; break; fi
done
if ! type conda >/dev/null 2>&1; then echo "Set CONDA_SH to the miniforge etc/profile.d/conda.sh on this node." >&2; exit 2; fi
conda activate "$CONDA_ENV"
# Avoid importing source trees from previous research checkouts.
unset PYTHONPATH
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
cann=
for location in "${CANN_ROOT:-}" "$BASE/CANN/9.1.0" /home/n84449292/m84379596/CANN/9.1.0 /m84379596/CANN/9.1.0 /home/n84449292/m84379596/CANN; do
    if [[ -r "$location/cann/set_env.sh" && -r "$location/nnal/atb/set_env.sh" ]]; then cann=$location; break; fi
done
[[ -n "$cann" ]] || { echo "Set CANN_ROOT to the 9.1.0 directory containing cann/ and nnal/." >&2; exit 2; }
source "$cann/cann/set_env.sh"
source "$cann/nnal/atb/set_env.sh"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
TARGET_NPUS=${TARGET_NPUS:-0}
TRAIN_NPUS=${TRAIN_NPUS:-1,2,3,4,5,6,7}
EVAL_NPU=${EVAL_NPU:-0}
TEACHER_MODE=${TEACHER_MODE:-local}
FSDP_STRATEGY=${FSDP_STRATEGY:-shard-grad-op}
MASTER_PORT=${MASTER_PORT:-29647}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-1800}
RUN_ROOT=${RUN_ROOT:-$BASE/SpecDecode-Atlas/runs/orthrus}
die() { echo "$*" >&2; exit 2; }
[[ "$TEACHER_MODE" == dedicated || "$TEACHER_MODE" == local ]] || die "TEACHER_MODE can be dedicated or local."
[[ "$FSDP_STRATEGY" == shard-grad-op || "$FSDP_STRATEGY" == full-shard ]] || die "FSDP_STRATEGY can be shard-grad-op or full-shard."
[[ "$TRAIN_NPUS" =~ ^[0-9]+(,[0-9]+)*$ && "$TARGET_NPUS" =~ ^[0-9]+$ && "$EVAL_NPU" =~ ^[0-9]+$ ]] || die "Set physical NPU IDs as integers separated by commas."
IFS=, read -r -a devices <<< "$TRAIN_NPUS"
declare -A used
for card in "${devices[@]}"; do
    [[ ! -v "used[$card]" ]] || die "Duplicate trainer NPU $card"
    used[$card]=1
done
if [[ "$TEACHER_MODE" == dedicated && -v "used[$TARGET_NPUS]" ]]; then die "Target and trainer NPU IDs overlap."; fi
if [[ "$mode" == --evaluate ]]; then
    [[ -n "${CHECKPOINT:-}" && -e "$CHECKPOINT" ]] || die "Set CHECKPOINT to the saved checkpoint directory to evaluate."
elif [[ -n "${RESUME:-}" && ( "$mode" != --run || ! -e "$RESUME" ) ]]; then
    die "Use RESUME=/path/to/checkpoint or its latest.json with --run."
fi
mkdir -p "$RUN_ROOT"
if [[ -z "${OUTPUT_DIR:-}" && -n "${RESUME:-}" ]]; then OUTPUT_DIR=$(dirname "$(dirname "$(realpath "$RESUME")")"); fi
if [[ -z "${OUTPUT_DIR:-}" ]]; then OUTPUT_DIR=$(mktemp -d "$RUN_ROOT/$(date +%Y%m%d-%H%M%S)-${mode#--}-XXXX"); fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
if [[ "$mode" == --check-only && -d "$OUTPUT_DIR/checkpoints" ]]; then die "Choose a fresh OUTPUT_DIR for the smoke test."; fi
session=$(mktemp -d "$OUTPUT_DIR/session-${mode#--}-$(date +%Y%m%d-%H%M%S)-XXXX")
exec > >(tee -a "$session/launcher.log") 2>&1
teacher_pid=
socket_dir=
start=$(date +%s)
finish() {
    status=$?
    trap - EXIT
    if [[ -n "$teacher_pid" ]]; then kill "$teacher_pid" 2>/dev/null || true; wait "$teacher_pid" 2>/dev/null || true; fi
    if [[ -n "$socket_dir" ]]; then rm -f -- "$socket_dir/teacher.sock"; rmdir -- "$socket_dir" 2>/dev/null || true; fi
    end=$(date +%s)
    printf 'exit_code\t%s\nstart_unix\t%s\nend_unix\t%s\nelapsed_seconds\t%s\n' "$status" "$start" "$end" "$((end-start))" > "$session/status.tsv"
    printf 'Stage %s exited %s after %ss. Log: %s/launcher.log\n' "$mode" "$status" "$((end-start))" "$session"
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'Stage: %s\nRun: %s\nLog: %s/launcher.log\nModel: %s\nDataset: %s\n' "$mode" "$OUTPUT_DIR" "$session" "$MODEL" "$DATASET"
if [[ "$mode" == --evaluate ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="$EVAL_NPU"
    python -u -m evaluation.run --model "$MODEL" --checkpoint "$CHECKPOINT" --device npu:0 \
        --output "$session/evaluation" "$@"
    exit 0
fi
python -u preflight.py --model "$MODEL" --dataset "$DATASET" --output "$session/provenance.json" \
    --target-npu "$TARGET_NPUS" --train-npus "$TRAIN_NPUS" --teacher-mode "$TEACHER_MODE"
[[ "$mode" != --prepare-only ]] || exit 0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
args=(--model "$MODEL" --dataset "$DATASET" --output-dir "$OUTPUT_DIR"
      --blocks-per-chunk "${BLOCKS_PER_CHUNK:-64}" --head-chunk-size "${HEAD_CHUNK_SIZE:-256}"
      --workers "${NUM_WORKERS:-4}" --log-every "${LOG_EVERY:-10}" --fsdp-strategy "$FSDP_STRATEGY")
if [[ "$mode" == --check-only ]]; then args+=(--smoke); fi
if [[ "$mode" == --benchmark ]]; then
    args+=(--benchmark-steps "${BENCHMARK_STEPS:-20}" --benchmark-warmup "${BENCHMARK_WARMUP:-10}")
fi
if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
if [[ "$TEACHER_MODE" == dedicated ]]; then
    socket_dir=$(mktemp -d /tmp/orthrus-XXXXXXXX)
    socket="$socket_dir/teacher.sock"
    ASCEND_RT_VISIBLE_DEVICES="$TARGET_NPUS" python -u -m teacher.service \
        --model "$MODEL" --socket "$socket" --device npu:0 --max-length 3072 --timeout 180 --log-every 10 > "$session/teacher.log" 2>&1 &
    teacher_pid=$!
    echo "Teacher on NPU $TARGET_NPUS. Log: $session/teacher.log"
    remaining=$BOOT_TIMEOUT
    while [[ ! -S "$socket" ]]; do
        if ! kill -0 "$teacher_pid" 2>/dev/null; then tail -40 "$session/teacher.log"; die "Teacher exited during startup."; fi
        if (( remaining <= 0 )); then die "Teacher startup timed out. See $session/teacher.log"; fi
        sleep 2
        remaining=$((remaining-2))
    done
    python -m teacher.client --socket "$socket" --model "$MODEL" --wait 30
    args+=(--teacher-socket "$socket")
fi
export ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS"
echo "Training ranks: ${#devices[@]} on physical NPUs $TRAIN_NPUS; teacher=$TEACHER_MODE; FSDP=$FSDP_STRATEGY"
python -u -m torch.distributed.run --nnodes=1 --nproc-per-node="${#devices[@]}" \
    --master-addr=127.0.0.1 --master-port="$MASTER_PORT" --module training.train "${args[@]}"
if [[ "$mode" == --benchmark && -n "${BASELINE_RUN:-}" ]]; then
    python -m training.compare_timing "$BASELINE_RUN" "$OUTPUT_DIR/benchmark.json"
fi
