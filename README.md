# SpecDecode Atlas

A survey and benchmark of speculative decoding for large language models, maintained by [Ali](https://github.com/alipourranjbar1) and [Mohammad](https://github.com/MohammadMahdi1375).

When does a longer accepted draft make generation faster? The planned comparison fixes the Qwen3-4B target, trains compatible drafters for one epoch on the same PerfectBlend rollouts, and measures them against matched autoregressive runs. Across the eight benchmarks, report committed tokens per verification round (including the bonus token), output tokens per second, and speedup over the autoregressive control.

## Repositories

| Repository | Purpose | Starting revision |
| --- | --- | --- |
| [vllm](https://github.com/SpecDecode-Atlas/vllm) | Inference engine | [9dad8f82c](https://github.com/SpecDecode-Atlas/vllm/commit/9dad8f82cd8657dd87b7ea0610ed0d65dd4100c6) |
| [vllm-ascend](https://github.com/SpecDecode-Atlas/vllm-ascend) | Ascend backend for vLLM | [97af565d5](https://github.com/SpecDecode-Atlas/vllm-ascend/commit/97af565d51137c5db6f9d4c70cebad9fabafaff0) |
| [speculators](https://github.com/SpecDecode-Atlas/speculators) | Drafter training and evaluation | [ee3151907](https://github.com/SpecDecode-Atlas/speculators/commit/ee31519074ab5cbe2d1ace38377b445ac68e531c) |

The three organization repositories are direct forks of the original `vllm-project` repositories. Use `serving/` for server launchers and `evaluation/` for benchmark drivers and configurations.

## Working together

For a launcher or benchmark change, open a pull request in this repository. For changes to vLLM, vLLM Ascend, or Speculators, open a pull request against the corresponding SpecDecode Atlas fork. Record the tested revisions alongside each benchmark run.
