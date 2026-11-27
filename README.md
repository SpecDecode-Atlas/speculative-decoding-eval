# SpecDecode Atlas

A survey and benchmark of speculative decoding for large language models, maintained by [Ali](https://github.com/alipourranjbar1) and [Mohammad](https://github.com/MohammadMahdi1375).

When does a longer accepted draft make generation faster? The planned comparison fixes the Qwen3-4B target, trains compatible drafters for one epoch on the same PerfectBlend rollouts, and measures them against matched autoregressive runs. Across the eight benchmarks, report committed tokens per verification round (including the bonus token), output tokens per second, and speedup over the autoregressive control.

## Ascend 910B3 development environment

The `cann-9.1-910b3` branch of each shared fork starts at the original project's release shown below. The vLLM Ascend release targets CANN and NNAL 9.1.0, PyTorch 2.10.0, TorchNPU 2.10.0.post4, and Triton Ascend 3.2.2. Speculators 0.7.0 declares support for this PyTorch version and for Transformers 5.5.4, which the Ascend release pins. [Upstream compatibility matrix](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html#release-compatibility-matrix)

| Shared fork | Purpose | Original release | Pinned commit |
| --- | --- | --- | --- |
| [vllm](https://github.com/SpecDecode-Atlas/vllm/tree/cann-9.1-910b3) | Inference engine | [v0.23.0](https://github.com/vllm-project/vllm/tree/v0.23.0) | [0fc695fc6](https://github.com/SpecDecode-Atlas/vllm/commit/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665) |
| [vllm-ascend](https://github.com/SpecDecode-Atlas/vllm-ascend/tree/cann-9.1-910b3) | Ascend backend | [v0.23.0.post1](https://github.com/vllm-project/vllm-ascend/tree/v0.23.0.post1) | [1cdb8c4db](https://github.com/SpecDecode-Atlas/vllm-ascend/commit/1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8) |
| [speculators](https://github.com/SpecDecode-Atlas/speculators/tree/cann-9.1-910b3) | Drafter training and evaluation | [v0.7.0](https://github.com/vllm-project/speculators/tree/v0.7.0) | [1c2895f3b](https://github.com/SpecDecode-Atlas/speculators/commit/1c2895f3b6c7b649c271ced55418e67e3ddae255) |

From the directory where you want all four checkouts, run the commands below in Bash. Set `CANN_ROOT` to the installation directory containing `cann/set_env.sh` and `nnal/atb/set_env.sh`. The installer uses Conda, a C/C++ toolchain, and the existing Ascend driver and toolkit. It installs the three local forks in editable mode. It checks Python dependencies, the vLLM platform, and visible devices. Model inference and benchmark runs are separate steps.

```bash
git clone https://github.com/SpecDecode-Atlas/speculative-decoding-eval.git
export CANN_ROOT=/path/to/CANN/9.1.0
bash speculative-decoding-eval/install_910b3.sh
```

The default install creates or reuses `.envs/cann91-py311` in the workspace. To install into an existing Python 3.11 Conda environment, activate it, set `CANN_ROOT` as above, and run `bash speculative-decoding-eval/install_910b3.sh --active-env`. Pip replaces the editable packages in that environment. The final checks print and verify the source directory for each import.

To download and verify the sources only, use `bash speculative-decoding-eval/install_910b3.sh --clone-only`. The install log is in `install-logs/` beside the four repositories. Use `serving/` for server launchers and `evaluation/` for benchmark drivers and configurations.

## Working together

For a launcher or benchmark change, open a pull request in this repository. For changes to vLLM, vLLM Ascend, or Speculators, open a pull request against the corresponding SpecDecode Atlas fork. Record the tested revisions alongside each benchmark run.
