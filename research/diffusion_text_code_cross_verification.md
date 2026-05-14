# Cross-Verification Report: Diffusion Models for Text and Code Generation

## Methodology
Findings from 12 dimension deep-dives and 6 wide exploration facets were cross-compared. Each finding was classified into one of four confidence tiers based on source independence and consistency.

---

## High Confidence Findings (Confirmed by ≥2 agents from independent sources)

### HC-1: Diffusion LLMs can match or exceed AR models on code generation benchmarks
- **Evidence**: Gemini Diffusion 89.6% HumanEval vs Seed-Coder 84.8% (dim01, dim10)
- **Evidence**: LLaDA2.0-flash 94.51 HumanEval vs Qwen3-30B 93.29 (dim02)
- **Evidence**: Stable-DiffCoder 86.6% HumanEval, surpasses AR counterpart (dim05)
- **Sources**: Official benchmark reports, arXiv papers, independent evaluation studies
- **Confidence**: HIGH — Multiple independent confirmations

### HC-2: Inference speed is the primary commercial advantage of diffusion LLMs
- **Evidence**: Seed Diffusion 2,146 tok/s, Gemini Diffusion 1,479 tok/s, Mercury 1,109 tok/s (dim09, dim11)
- **Evidence**: 2.1x speedup for LLaDA2.0 over AR baselines under controlled conditions (dim02)
- **Sources**: Official papers, LMSYS blog, vendor claims
- **Confidence**: HIGH — But hardware differences make exact comparisons difficult (see Conflict Zone CZ-1)

### HC-3: AR-to-diffusion conversion is more efficient than training from scratch
- **Evidence**: LLaDA2.0 WSD uses 3-phase conversion preserving AR knowledge (dim02)
- **Evidence**: I-DLM achieves quality matching with only ~5B tokens (dim07)
- **Evidence**: DiffuLLaMA converts LLaMA2-7B with <200B tokens (dim06)
- **Sources**: Multiple papers from different institutions
- **Confidence**: HIGH

### HC-4: RL outperforms SFT for diffusion model post-training
- **Evidence**: VRPO (LLaDA 1.5) outperforms DPO/IPO/SLiC (dim06, dim08)
- **Evidence**: Coupled-GRPO (DiffuCoder) achieves +4.4% EvalPlus (dim06, dim08)
- **Evidence**: EBPO (LLaDA2.1) enables first large-scale RL for dLLMs (dim03, dim08)
- **Sources**: Independent papers from Apple, Ant Group, academic institutions
- **Confidence**: HIGH

### HC-5: Block diffusion is the production deployment paradigm
- **Evidence**: LLaDA2.0 uses block size 32 for inference (dim02)
- **Evidence**: Stable-DiffCoder uses block diffusion CPT (dim05)
- **Evidence**: Gemini Diffusion uses block diffusion (dim01)
- **Sources**: All major production diffusion LLMs
- **Confidence**: HIGH

### HC-6: Diffusion models excel at code editing and infilling tasks
- **Evidence**: Stable-DiffCoder 60.0% CanItEdit vs Seed-Coder 50.5% (dim05, dim10)
- **Evidence**: Seed Diffusion 54.3% CanItEdit with edit-based training (dim05)
- **Sources**: Controlled comparisons from ByteDance
- **Confidence**: HIGH

### HC-7: Diffusion models show superior length extrapolation
- **Evidence**: Mercury-Coder shows ~15% degradation at 64K vs Qwen3 ~30% (dim10)
- **Evidence**: DiffuCoder maintains >30% accuracy where Llama-2 drops below 10% (dim10)
- **Sources**: RepoQA benchmark, independent study
- **Confidence**: HIGH

### HC-8: Ant Group is the leading open-source contributor to diffusion LLMs
- **Evidence**: LLaDA2.0 (100B), LLaDA2.1, LLaDA-MoE all open-sourced (dim02, dim03, dim04)
- **Evidence**: Complete toolchain (dFactory, dInfer, SGLang integration) (dim02, dim04)
- **Evidence**: CodeFuse NES serves 20,000+ developers (dim04)
- **Sources**: GitHub, HuggingFace, official publications
- **Confidence**: HIGH

---

## Medium Confidence Findings (Confirmed by 1 agent from authoritative source)

### MC-1: Google DeepMind's MD4 is the foundational framework for Gemini Diffusion
- **Source**: NeurIPS 2024 paper, open-sourced code
- **Note**: Logical inference — Gemini Diffusion team includes MD4 authors
- **Confidence**: MEDIUM

### MC-2: Mercury Coder has 25,000+ developers
- **Source**: Inception Labs marketing claims (dim11)
- **Note**: Self-reported metric, not independently verified
- **Confidence**: MEDIUM

### MC-3: Diffusion models lag on competitive programming and SWE-bench
- **Source**: Beyond Autoregression study (dim10)
- **Note**: Single comprehensive study, but well-designed
- **Confidence**: MEDIUM

### MC-4: MDM-Prime-v2 claimed 21.8x compute efficiency
- **Source**: NeurIPS 2025 paper
- **Note**: Paper was later withdrawn — claim is suspect
- **Confidence**: MEDIUM — FINDING DISPUTED

### MC-5: Stefano Ermon predicts all frontier models will be diffusion
- **Source**: Multiple interviews and talks
- **Note**: Prediction, not fact
- **Confidence**: MEDIUM

---

## Low Confidence Findings (Weak sourcing or single unverified claim)

### LC-1: Exact TPS figures for commercial models
- **Issue**: Hardware differences (H20 vs H100 vs TPU) make comparisons unreliable
- **Sources**: Vendor self-reports under different conditions
- **Confidence**: LOW

### LC-2: "Pseudo diffusion" critique from ADD paper
- **Issue**: Philosophical debate about definitions, not empirical finding
- **Sources**: Single paper with limited follow-up
- **Confidence**: LOW

---

## Conflict Zones

### CZ-1: Speed comparison fairness
- **Seed Diffusion**: 2,146 tok/s on H20 GPUs (dim05)
- **Mercury Coder**: 1,109 tok/s on H100 GPUs (dim11)
- **Gemini Diffusion**: 1,479 tok/s on unknown hardware (dim01)
- **Conflict**: Different hardware, different serving stacks, different measurement conditions
- **Resolution**: Only controlled comparison (SGLang on H20) shows ~2.1x speedup — impressive but far from 10x headline claims
- **Status**: UNRESOLVED — Need standardized benchmarking

### CZ-2: Diffusion vs AR on LiveCodeBench
- **Beyond Autoregression**: Diffusion avg 14.9% vs AR avg 18.9% (dim10)
- **Individual models**: Dream-Coder 21.4%, Mercury 22.9%, Gemini Diffusion 30.9%
- **Conflict**: Some diffusion models beat some AR models, but diffusion average is lower
- **Resolution**: Depends on model scale and training — top diffusion models can compete
- **Status**: PARTIALLY RESOLVED

### CZ-3: MDM-Prime-v2 21.8x efficiency claim
- **Claim**: 21.8x more compute-efficient than AR models (dim07)
- **Reality**: Paper was withdrawn without explanation
- **Standard MDM**: Actually performed worse than AR in the same analysis (PPL 18.94 vs 12.99)
- **Status**: WITHDRAWN — Do not cite as fact

### CZ-4: Whether SFT helps or hurts diffusion models
- **Finding 1**: Standard SFT provides marginal gains due to train-test mismatch (dim08)
- **Finding 2**: Blockwise SFT and structure-aware training (TreeDiff) can improve by 13.3% (dim08)
- **Resolution**: Standard SFT is less effective, but specialized approaches help
- **Status**: RESOLVED

### CZ-5: Discrete vs continuous diffusion superiority
- **Discrete**: Dominates at scale (LLaDA 100B, MDLM) (dim07)
- **Continuous**: LangFlow and ELF closing the gap (dim07)
- **Resolution**: Discrete leads empirically at scale, but continuous has theoretical advantages (editability, CFG compatibility)
- **Status**: UNRESOLVED — Active research area
