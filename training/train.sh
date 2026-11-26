#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."

python=${PYTHON:-python3}
model_args=()
if [[ -n ${SDXL_BASE:-} ]]; then
    model_args=(--model "$SDXL_BASE")
fi

mkdir -p runs logs
exec 9>logs/train.lock
flock -n 9 || { echo 'Training is already running in this checkout.' >&2; exit 1; }

train_stage() {
    local name=$1 module=$2 manifest=$3 steps=$4 interval=$5 lr=$6 seed=$7
    shift 7
    local output="runs/$name" log="logs/$name.log" receipt="runs/$name/completed.sha256"
    local files=("$output/settings.json" "$output/metrics.jsonl" "$output/evaluation.json"
                 "$output/adapter.safetensors" "$manifest")
    if [[ $module == book ]]; then
        files+=("$output/conditioner.safetensors")
    fi
    if [[ $name == book_joint_01 ]]; then
        files+=(data/book/print_lines_joint.jsonl configs/fonts/DejaVuSerif.ttf runs/book/completed.sha256)
    fi
    if [[ -f $receipt ]]; then
        local script_hash expected actual
        # The first release recorded the hash and path of its root-level launcher.
        local first_recipe="a260268bfbadd6334eef2e2279984e217b25de861e97d6841ca1c95377b46618  train.sh"
        if grep -Fqx "$first_recipe" "$receipt"; then
            script_hash=$first_recipe
        else
            script_hash=$(sha256sum training/train.sh)
        fi
        expected=$(sha256sum "${files[@]}")
        actual=$(sort "$receipt")
        if [[ $actual != "$(printf '%s\n' "$expected" "$script_hash" | sort)" ]]; then
            printf 'Checkpoint or recipe mismatch for %s\n' "$name" >&2
            return 1
        fi
        printf '%s %s already complete\n' "$(date -Is)" "$name"
        return
    fi
    if [[ -e $output ]]; then
        local previous="$name.interrupted.$(date -u +%Y%m%dT%H%M%SZ).$$"
        mv -T -- "$output" "runs/$previous"
        if [[ -f $log ]]; then mv -- "$log" "logs/$previous.log"; fi
        printf '%s restarted %s from step 1; earlier files in runs/%s\n' "$(date -Is)" "$name" "$previous"
    fi

    printf '%s starting %s (%s steps); log: %s\n' "$(date -Is)" "$name" "$steps" "$log"
    if ! "$python" -Bu -m training.train --module "$module" --manifest "$manifest" --output "$output" \
         --size 768 --rank 4 --steps "$steps" --eval-every "$interval" --lr "$lr" --seed "$seed" \
         "${model_args[@]}" "$@" >"$log" 2>&1; then
        tail -n 25 "$log" >&2
        return 1
    fi
    sha256sum "${files[@]}" training/train.sh >"$receipt.tmp"
    mv -T -- "$receipt.tmp" "$receipt"
    printf '%s completed %s\n' "$(date -Is)" "$name"
}

train_stage background background data/background/manifest.jsonl 240 48 0.00006 20261125
train_stage illustration illustration data/illustration/manifest.jsonl 240 48 0.00006 20261125
train_stage book book data/book/manifest.jsonl 128 32 0.00008 20261125
train_stage book_joint_01 book data/book/manifest_joint.jsonl 128 32 0.00008 20261126 \
    --joint-lines data/book/print_lines_joint.jsonl --font configs/fonts/DejaVuSerif.ttf --init-book runs/book
