#!/usr/bin/env bash
# Install the three SpecDecode Atlas forks for a 910B3 with CANN 9.1.0.
# Run with Bash from a checkout of speculative-decoding-eval.
set -eo pipefail

branch=cann-9.1-910b3
workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
env_dir="$workspace/.envs/cann91-py311"
mode=${1:-install}
case "$mode" in install|--clone-only) ;; *) echo "Usage: bash install_910b3.sh [--clone-only]" >&2; exit 2 ;; esac

mkdir -p "$workspace/install-logs"
log=$(mktemp "$workspace/install-logs/910b3-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")
exec > >(tee -a "$log") 2>&1
trap 'printf "\nInstallation stopped at line %s. Log: %s\n" "$LINENO" "$log" >&2' ERR
echo "Workspace: $workspace"
echo "Log: $log"

fail() { echo "$*" >&2; exit 1; }

checkout_release() {
    local name=$1 tag=$2 sha=$3 dir="$workspace/$1"
    local fork="https://github.com/SpecDecode-Atlas/$name.git"
    local upstream="https://github.com/vllm-project/$name.git"
    if [[ ! -e "$dir" ]]; then
        git clone --progress --depth 1 --single-branch --no-tags --branch "$branch" "$fork" "$dir"
    fi
    [[ -e "$dir/.git" ]] || fail "$dir already exists and is not a Git checkout"
    case "$(git -C "$dir" remote get-url origin)" in
        "$fork"|"${fork%.git}"|"git@github.com:SpecDecode-Atlas/$name.git") ;;
        *) fail "Existing $dir has a different origin. Choose a fresh workspace." ;;
    esac
    # GitHub forks may omit the upstream tags. Fetch only the tag used for this build.
    git -C "$dir" fetch --no-tags --depth 1 "$upstream" "refs/tags/$tag:refs/tags/$tag"
    [[ "$(git -C "$dir" rev-parse "$tag^{commit}")" = "$sha" ]] || fail "Unexpected upstream commit for $name $tag"
    if [[ "$(git -C "$dir" rev-parse HEAD)" != "$sha" ]]; then
        [[ -z "$(git -C "$dir" status --porcelain)" ]] || fail "Save your work in $dir before switching revisions."
        if git -C "$dir" show-ref --verify --quiet "refs/heads/$branch"; then
            if [[ "$(git -C "$dir" rev-parse "$branch")" = "$sha" ]]; then
                git -C "$dir" switch "$branch"
            else
                # Keep collaborators' newer branch commits available while reproducing the release.
                git -C "$dir" switch --detach "$sha"
            fi
        else
            git -C "$dir" switch -c "$branch" "$sha"
        fi
    fi
    # Verify the source used below even when resuming a partially completed setup.
    [[ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ]] || fail "Unexpected checkout for $name"
    if [[ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]]; then
        echo "Local tracked edits in $name will be included in this build."
    fi
    printf '%-14s %s  %s\n' "$name" "$tag" "$sha"
}

echo "[1/4] Checking out the three shared forks"
checkout_release vllm v0.23.0 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
checkout_release vllm-ascend v0.23.0.post1 1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8
checkout_release speculators v0.7.0 1c2895f3b6c7b649c271ced55418e67e3ddae255
if [[ "$mode" = --clone-only ]]; then exit 0; fi

: "${CANN_ROOT:?Set CANN_ROOT to your CANN 9.1.0 directory containing cann/ and nnal/}"
for script in "$CANN_ROOT/cann/set_env.sh" "$CANN_ROOT/nnal/atb/set_env.sh"; do
    [[ -r "$script" ]] || fail "Missing environment script: $script"
done
conda_bin=${CONDA_EXE:-$(type -P conda || true)}
if [[ -z "$conda_bin" && -x "$HOME/miniforge3/bin/conda" ]]; then conda_bin="$HOME/miniforge3/bin/conda"; fi
[[ -x "$conda_bin" ]] || fail "Initialize Conda (or set CONDA_EXE to its executable) and rerun."
echo "[2/4] Preparing an isolated Python 3.11 environment"
if [[ ! -e "$env_dir" ]]; then
    "$conda_bin" create -y -p "$env_dir" python=3.11 pip
fi
[[ -d "$env_dir/conda-meta" ]] || fail "Expected a Conda environment at $env_dir"
source "$("$conda_bin" info --base)/etc/profile.d/conda.sh"
conda activate "$env_dir"
[[ "$CONDA_PREFIX" = "$env_dir" ]] || fail "Could not activate $env_dir"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
source "$CANN_ROOT/cann/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-$CANN_ROOT/cann}"
export SOC_VERSION=ascend910b3 COMPILE_CUSTOM_KERNELS=1
export MAX_JOBS="${MAX_JOBS:-2}" CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS:-2}" CARGO_BUILD_JOBS="${MAX_JOBS:-2}"
python -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
git -C "$workspace/vllm-ascend" submodule update --init --recursive

echo "[3/4] Installing the source packages and the 910B3 kernels"
cpu_index=https://download.pytorch.org/whl/cpu
ascend_index=${ASCEND_PIP_INDEX:-https://mirrors.huaweicloud.com/ascend/repos/pypi}
indices=(--extra-index-url "$ascend_index" --extra-index-url "$cpu_index")
python -m pip install --upgrade pip
python -m pip install --index-url "$cpu_index" torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

# Resolve the runtime requirements together so conflicts are reported before the native build.
spec_deps=$(mktemp "$workspace/install-logs/speculators-requirements-XXXXXX.txt")
python - "$workspace/speculators/pyproject.toml" > "$spec_deps" <<'PY'
import sys
import tomllib
with open(sys.argv[1], 'rb') as file:
    print(*tomllib.load(file)['project']['dependencies'], sep='\n')
PY
python -m pip install "${indices[@]}" \
    -r "$workspace/vllm/requirements/common.txt" -r "$workspace/vllm-ascend/requirements.txt" -r "$spec_deps" \
    'setuptools>=77.0.3,<81' 'setuptools-scm>=8' 'setuptools-rust>=1.9' 'packaging>=24.2' \
    'cmake>=3.26.1' ninja jinja2 wheel attrs googleapis-common-protos concurrent-log-handler
VLLM_TARGET_DEVICE=empty python -m pip install "${indices[@]}" --no-build-isolation -e "$workspace/vllm"
python -m pip install "${indices[@]}" --no-build-isolation -e "$workspace/vllm-ascend"
# Speculators uses a newer setuptools in its own isolated build environment.
SPECULATORS_BUILD_TYPE=release python -m pip install "${indices[@]}" \
    -e "$workspace/speculators/hs_connectors" -e "$workspace/speculators"

echo "[4/4] Checking packages, the Ascend plugin, and NPU visibility"
python -m pip check
python - <<'PY'
from importlib.metadata import version
import torch
import torch_npu
import vllm
import vllm_ascend
import speculators
import hs_connectors
from vllm.platforms import current_platform
for name in ('vllm', 'vllm-ascend', 'speculators', 'hs-connectors', 'torch', 'torch-npu', 'transformers', 'triton-ascend'):
    print(f'{name:16} {version(name)}')
print('vLLM platform:', current_platform.device_type)
print('Visible NPUs:', torch.npu.device_count())
assert current_platform.device_type == 'npu', 'vLLM did not select its Ascend plugin'
assert torch.npu.device_count() > 0, 'No Ascend device is visible to this process'
PY
echo "Install checks passed. For a new shell, activate this environment and source the same CANN and ATB scripts."
echo "conda activate $env_dir"
