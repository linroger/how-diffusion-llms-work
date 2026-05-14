## Facet: Code Generation Benchmarks and Comparative Performance (Deep Dive)

### Key Findings

#### 1. "Beyond Autoregression" Paper Deep Dive

The paper "Beyond Autoregression: An Empirical Study of Diffusion Large Language Models for Code Generation" (Zhang et al., 2025) is the **most comprehensive controlled comparison** of diffusion and autoregressive code models to date [^5^](https://arxiv.org/html/2509.11252v1).

**Methodology:**
- Evaluated **9 diffusion LLMs** from 6 families: LLaDA (8B, 1.5), Dream (v0-Instruct, Coder-v0, DreamOn), DiffuCoder-7B-cpGRPO, Mercury-Coder-Small, Gemini-Diffusion, Seed-Diffusion-Preview
- Compared against **4 AR baselines**: Qwen3-8B, Seed-Coder-8B-Instruct, DeepSeek-Coder-6.7B-Instruct, CodeLlama-7B-Instruct-hf
- Used **4 benchmarks**: HumanEval, MBPP, LiveCodeBench (v1-v6, v6), and RepoQA
- Key evaluation metrics: pass@1, retrieval accuracy, FLOPs, throughput
- All experiments on 8x NVIDIA A100-40GB GPUs with zero-shot prompting

**Key quantitative findings from Table 2 [^9^](https://arxiv.org/html/2509.11252v2):**

| Benchmark | Diffusion Avg | AR Avg | Best Diffusion | Best AR |
|-----------|--------------|--------|----------------|---------|
| HumanEval | 66.7% | 71.3% | Gemini-Diffusion 89.6% | Seed-Coder 84.8% |
| MBPP | 61.2% | 60.8% | Seed-Diffusion 79.4% | Seed-Coder 70.8% |
| LiveCodeBench v1-v6 | 19.1% | 25.8% | Seed-Diffusion 33.7% | Qwen3-8B 42.3% |
| LiveCodeBench v6 | 14.9% | 18.9% | Gemini-Diffusion 30.9% | Qwen3-8B 26.0% |

The paper's four research questions:
- **RQ1**: Effectiveness of diffusion LLMs for code generation
- **RQ2**: Impact of generation length, diffusion steps, remasking strategy, block length on effectiveness
- **RQ3**: Impact of same settings on efficiency (FLOPs, throughput)
- **RQ4**: Long-context code understanding (RepoQA)

#### 2. HumanEval Comparison — Which Diffusion Models Beat Which AR Models

**Gemini Diffusion achieves the highest HumanEval score across all models (89.6%)**, surpassing the best AR baseline Seed-Coder-8B-Instruct (84.8%) by 4.8 percentage points [^9^](https://arxiv.org/html/2509.11252v2). Mercury-Coder-Small also edges out Seed-Coder at 86.0% vs. 84.8%. Several open-source diffusion models approach or match strong AR models:

- **Dream-Coder-v0-Instruct-7B**: 76.2% — approaches DeepSeek-Coder-6.7B-Instruct (77.4%)
- **DiffuCoder-7B-cpGRPO**: 69.5% — competitive with CodeLlama-7B (40.2%) but behind larger AR models
- **Stable-DiffCoder-8B-Instruct**: 86.6% — surpasses Seed-Coder-8B-Instruct on HumanEval [^83^](https://arxiv.org/html/2601.15892v1)

Notably, the closed-source diffusion models (Gemini Diffusion, Mercury, Seed-Diffusion) significantly outperform open-source diffusion models, suggesting training data quality and scale matter enormously. The "Beyond Autoregression" authors note: "Diffusion LLMs still lag behind AR LLMs in overall performance... [but] demonstrate strong competitiveness against state-of-the-art AR LLMs" [^9^](https://arxiv.org/html/2509.11252v2).

Mercury Coder Mini achieves 88.0% on HumanEval at **1,109 tokens/sec** — vastly faster than AR models of comparable quality [^85^](https://arxiv.org/html/2506.17298v1).

#### 3. LiveCodeBench Gap — Why Diffusion Models Lag on Competitive Programming

The LiveCodeBench gap is the **most consistent and concerning weakness** for diffusion code models:

- Diffusion average on LiveCodeBench v6: **14.9%** vs. AR average **18.9%**
- Best diffusion (Gemini Diffusion at 30.9%) still trails best AR (Qwen3-8B at 26.0% on v6, but on v1-v6 Qwen3 reaches 42.3% vs. Seed-Diffusion's 33.7%)
- Open-source diffusion models are particularly weak: LLaDA-8B scores only **6.9%** on v1-v6

**Why diffusion models struggle with competitive programming:**

1. **Chain-of-thought reasoning mismatch**: The NAP paper (Li et al., 2026) identifies that DLMs trained on sequential CoT data converge to autoregressive-like decoding patterns. When forced to use true parallel decoding, reasoning accuracy collapses. "Standard supervision creates a dependency on sequential stability; when forced to hurry, the reasoning collapses" [^429^](https://arxiv.org/html/2602.23225v1). On Dream-7B/GSM8K, accuracy drops from 78.0% (1024 steps) to 46.5% (256 steps).

2. **Multi-step reasoning under parallel decoding**: Competitive programming requires step-by-step algorithmic reasoning. Diffusion models generate tokens in parallel, which can disrupt the logical chain. The NAP paper shows that even with Arbitrary Order (AO) decoding, models exhibit high ARness (~0.92 for Dream), meaning their "most confident tokens are almost always the next tokens in the sequence" [^429^](https://arxiv.org/html/2602.23225v1).

3. **Lower exposure to algorithmic content**: The "Beyond Autoregression" paper notes that open-source diffusion LLMs lag behind closed-source counterparts, possibly due to training data composition [^9^](https://arxiv.org/html/2509.11252v2).

4. **LiveCodeBench is contamination-free**: Unlike HumanEval which may be in training data, LiveCodeBench v6 contains problems released after model training cutoffs, measuring true generalization — where diffusion models are weaker [^9^](https://arxiv.org/html/2509.11252v2).

**Stable-DiffCoder LiveCodeBench results**: 23.5% vs. Seed-Coder-8B-Instruct's 24.7% — essentially matching its AR counterpart, with only a 1.2pp gap [^83^](https://arxiv.org/html/2601.15892v1).

#### 4. BigCodeBench Results — Real-World Multi-Library Code Generation

BigCodeBench evaluates "challenging, real-world coding problems with rich context and tool-like function calls" [^83^](https://arxiv.org/html/2601.15892v1). This is where diffusion models show **near-parity or superiority**:

| Model | BigCodeBench Score |
|-------|-------------------|
| Gemini Diffusion | 45.4% [^272^](https://deepmind.google/models/gemini-diffusion/) |
| Flash-Lite | 45.8% [^37^](https://venturebeat.com/technology/beyond-gpt-architecture-why-googles-diffusion-approach-could-reshape-llm-deployment) |
| Seed Diffusion Preview | 45.4% [^482^](https://neurohive.io/en/state-of-the-art/seed-diffusion-new-state-of-the-art-in-speed-quality-balance-for-code-generation-models/) |
| Mercury Coder Small | 45.5% [^71^](https://www.inceptionlabs.ai/blog/introducing-mercury) |
| Stable-DiffCoder-8B-Instruct | Only surpassed by DeepSeek-Coder-V2-Instruct (21B/236B) [^83^](https://arxiv.org/html/2601.15892v1) |
| Dream-Coder 7B Instruct | 21.4% on BigCodeBench completion [^3^](https://arxiv.org/pdf/2509.01142) |

Gemini Diffusion essentially ties Flash-Lite (45.4% vs. 45.8%) [^37^](https://venturebeat.com/technology/beyond-gpt-architecture-why-googles-diffusion-approach-could-reshape-llm-deployment). The VentureBeat analysis notes: "the gap between the two techniques is essentially closed in terms of benchmark performance."

DiffuCoder with coupled-GRPO achieves 40.4% on BigCodeBench completion (from 35.7% before RL), a +4.7pp improvement from reinforcement learning [^10^](https://arxiv.org/html/2506.20639v2).

#### 5. RepoQA Length Extrapolation — Diffusion's Advantage at Long Context

RepoQA's "Searching Needle Function" task evaluates long-context code understanding across 500 problems in 50 repositories [^526^](https://arxiv.org/pdf/2509.11252). Diffusion models show **dramatically superior length extrapolation**:

**Key findings from RQ4:**
- At 4K tokens input, AR model (Llama-2-7B-CHAT-HF) retrieval accuracy drops below **10%**, while DiffuCoder-7B-cpGRPO maintains above **30%**
- When context length exceeds training window (8K→64K), Mercury-Coder-Small shows only ~15% performance decrease, while Qwen3-8B drops by nearly **30%**
- "Diffusion LLMs remain relatively robust as context length increases, whereas the performance of AR LLMs declines rapidly" [^526^](https://arxiv.org/pdf/2509.11252)
- LLaDA-2.0-flash achieves **88.3%** EvalPlus at 6B/100B active/total parameters, demonstrating strong scaling [^83^](https://arxiv.org/html/2601.15892v1)

This advantage is hypothesized to stem from diffusion models' bidirectional attention during training — they learn to attend to all positions equally rather than building left-to-right dependencies.

#### 6. SWE-Bench Gap — Why Diffusion Struggles with Real-World Software Engineering

**Gemini Diffusion scores 22.9% on SWE-Bench Verified vs. Flash-Lite's 28.5%** — a 5.6pp gap [^272^](https://deepmind.google/models/gemini-diffusion/). However, both scores are far below frontier models (Claude Opus 4.6 at 80.6%, Gemini 3.1 Pro at 80.8%) [^803^](https://exzilcalanza.info/agentic-coding-2026-agent-teams-swe-bench-enterprise/).

**Why diffusion models underperform on SWE-Bench:**

1. **Non-agentic evaluation**: The Gemini Diffusion result uses "non-agentic evaluation (single turn edit only), max prompt length of 32K" [^272^](https://deepmind.google/models/gemini-diffusion/). SWE-Bench requires multi-step reasoning: understanding codebase structure, locating relevant files, diagnosing issues, and generating patches — tasks that benefit from iterative exploration.

2. **Single-turn limitation**: Diffusion models excel at generating complete solutions in one pass but may struggle with tasks requiring sequential decision-making and tool use (e.g., running tests, examining error messages, iterating).

3. **Context window constraints**: Many open-source diffusion models have 4K context windows, far smaller than needed for SWE-Bench tasks that span entire repositories.

4. **No multi-turn capability**: SWE-Bench Verified leaderboards show agentic models (Claude, GPT-4) with iterative feedback loops vastly outperform single-turn models. Diffusion models have not yet been evaluated in agentic settings on SWE-Bench.

#### 7. CanItEdit — Diffusion's Advantage on Code Editing Tasks

**CanItEdit is where diffusion models show their most decisive advantage.** Code editing is fundamentally non-sequential — modifying existing code requires understanding the full context and making targeted changes.

| Model | CanItEdit pass@1 |
|-------|-----------------|
| Stable-DiffCoder-8B-Instruct | **60.0%** [^83^](https://arxiv.org/html/2601.15892v1) |
| Seed-Diffusion-Preview | **54.3%** [^793^](https://seed.bytedance.com/en/seed_diffusion) |
| Qwen2.5-Coder-14B-Instruct | 52.9% |
| Seed-Coder-8B-Instruct | 50.5% |
| Yi-Coder-9B-Chat | 50.5% |
| Qwen3-8B | 45.7% |
| DeepSeek-Coder-33B-Instruct | 46.2% |

Stable-DiffCoder-8B-Instruct achieves **60.0%** on CanItEdit, surpassing ALL other models including those 4x larger (DeepSeek-Coder-33B at 46.2%) and nearly matching Codestral-22B (52.4%) [^83^](https://arxiv.org/html/2601.15892v1). The authors hypothesize: "random masking and reconstruction inherently train the model on edit- and infill-like patterns, enabling it to better exploit editing supervision" [^83^](https://arxiv.org/html/2601.15892v1).

Seed Diffusion's two-stage curriculum learning (mask-based → edit-based training) boosted CanItEdit by **4.8pp over AR models** (54.3% vs. 50.5%) [^793^](https://seed.bytedance.com/en/seed_diffusion).

On Aider (multi-turn editing), Stable-DiffCoder achieves 54.9% (tries=2), slightly behind Seed-Coder (57.1%) but comparable to Qwen3-8B (55.6%) [^83^](https://arxiv.org/html/2601.15892v1).

#### 8. CRUXEval — Reasoning-Intensive Code Evaluation

CRUXEval tests code execution reasoning through 800 Python functions with input prediction (CRUXEval-I) and output prediction (CRUXEval-O) tasks [^813^](https://arxiv.org/pdf/2401.03065).

**Stable-DiffCoder CRUXEval results:**
- Stable-DiffCoder-8B-Base: Input-CoT 53.8%, Output-CoT 60.0% vs. Seed-Coder-8B-Base 52.0%/54.8% [^160^](https://arxiv.org/pdf/2601.15892)
- Stable-DiffCoder-8B-Instruct: stronger on Output-CoT, slightly better average across Input-CoT and Output-CoT vs. Seed-Coder
- Still trails Qwen3-8B on CRUXEval, "indicating that there is still considerable headroom for specialized small-code models on fine-grained reasoning tasks" [^83^](https://arxiv.org/html/2601.15892v1)

**Dream-Coder 7B** also shows competitive results on CRUXEval [^274^](https://arxiv.org/abs/2509.01142), demonstrating that diffusion models can develop strong code reasoning capabilities.

The "Beyond Autoregression" paper notes that diffusion models benefit from "any-order modeling" on CRUXEval because "the inputs and outputs are inherently structured rather than strictly following left-to-right causal logic" [^83^](https://arxiv.org/html/2601.15892v1).

#### 9. Benchmark Fairness — Are Benchmarks Biased Toward Left-to-Right Generation?

**Yes — multiple lines of evidence suggest existing benchmarks favor autoregressive generation:**

1. **HumanEval and MBPP are left-to-right function completion tasks**: The prompt provides function signature + docstring, and the model completes the body left-to-right. This aligns perfectly with AR generation order. As the "Beyond Autoregression" paper notes: "HumanEval provides function signatures and docstrings, which means the model only needs to complete the function body" [^9^](https://arxiv.org/html/2509.11252v2).

2. **The ARness-Accuracy Tradeoff**: The NAP paper demonstrates that forcing diffusion models to decode non-sequentially causes reasoning accuracy to collapse [^429^](https://arxiv.org/html/2602.23225v1). Standard benchmarks reward high-ARness behavior, creating an incentive for diffusion models to mimic AR patterns rather than leverage true parallel generation.

3. **PythonSaga critique**: Research shows "more than 80% of the problems [in HumanEval/MBPP] are perceived as easy" and "existing benchmarks lack a comprehensive evaluation of their diversity in terms of programming concepts and difficulty level" [^885^](https://aclanthology.org/2024.findings-emnlp.996.pdf). Easy problems don't stress non-sequential reasoning.

4. **LiveCodeBench requires step-by-step reasoning**: The gap between diffusion and AR is largest on LiveCodeBench — precisely because competitive programming requires sequential logical reasoning that current diffusion training doesn't optimize for.

5. **Code editing benchmarks (CanItEdit) favor diffusion**: The one benchmark type where diffusion consistently wins is code editing — a fundamentally non-sequential task. This suggests benchmark selection dramatically affects which paradigm appears superior.

6. **Diffusion-native RL struggles**: "Previous RL approaches for diffusion models rely heavily on semi-AR decoding, which deviates from diffusion's global nature" [^10^](https://arxiv.org/html/2506.20639v2). Training infrastructure itself biases toward AR-like behavior.

#### 10. New Benchmarks for Diffusion Code Models

**What would better evaluate diffusion code models?**

1. **Code editing and infilling benchmarks** (CanItEdit, Aider): Already show diffusion superiority. More benchmarks should measure non-sequential code modification.

2. **RepoQA-style long-context benchmarks**: Diffusion models excel at length extrapolation. Repository-level tasks requiring cross-file understanding play to diffusion strengths [^526^](https://arxiv.org/pdf/2509.11252).

3. **Multi-turn agentic coding tasks**: SWE-Bench in agentic mode evaluates real software engineering. Diffusion models have not been systematically tested with iterative tool use and feedback loops.

4. **Structured reasoning benchmarks**: CRUXEval-style tasks that require reasoning about code execution rather than just generating it. The NAP paper shows that restructuring supervision as "multiple independent reasoning trajectories" enables better parallel decoding [^429^](https://arxiv.org/html/2602.23225v1).

5. **Low-resource language benchmarks**: Stable-DiffCoder shows particular gains on sparse languages (C#, PHP) due to diffusion's "data augmentation" effect from multiple denoising views [^83^](https://arxiv.org/html/2601.15892v1).

6. **True parallel decoding evaluation**: Current benchmarks implicitly reward AR-like behavior. Benchmarks that explicitly measure performance under varying degrees of parallelism (as NAP does) would better assess diffusion's unique capabilities.

7. **Fill-in-the-middle (FIM) benchmarks**: Code completion in the middle of existing code is naturally non-sequential. Mercury Coder achieves **84.8%** on fill-in-the-middle tasks vs. Flash-Lite's 60.1% — a massive 24.7pp advantage [^71^](https://www.inceptionlabs.ai/blog/introducing-mercury).

8. **EndoCoT-style reasoning benchmarks**: "Vanilla denoising commits to solutions early without reasoning" [^857^](https://arxiv.org/html/2603.12252v3). Benchmarks that require explicit step-by-step reasoning chains during generation could push diffusion models to develop better reasoning mechanisms.

### Major Players & Sources

| Entity | Role/Relevance |
|--------|---------------|
| **Zhang et al. ("Beyond Autoregression")** | First systematic empirical study; 9 diffusion vs. 4 AR models; open-sourced all code/data [^5^](https://arxiv.org/html/2509.11252v1) |
| **Google DeepMind (Gemini Diffusion)** | Closed-source; HumanEval 89.6%, SWE-Bench 22.9%, LiveCodeBench 30.9%; 1,479 tok/s [^272^](https://deepmind.google/models/gemini-diffusion/) |
| **Inception Labs (Mercury Coder)** | First commercial-scale diffusion LLM; 1,109 tok/s; strong on HumanEval (86-90%), BigCodeBench (45.5%) [^85^](https://arxiv.org/html/2506.17298v1) |
| **ByteDance Seed (Seed Diffusion, Stable-DiffCoder)** | Seed Diffusion: 2,146 tok/s; Stable-DiffCoder: SOTA CanItEdit 60.0%, controlled AR vs. diffusion comparison [^83^](https://arxiv.org/html/2601.15892v1) |
| **Apple/HKU (DiffuCoder)** | Open-source 7B; coupled-GRPO RL; +4.4% EvalPlus improvement; AR-ness analysis [^10^](https://arxiv.org/html/2506.20639v2) |
| **Huawei/HKU (Dream-Coder)** | Adaptive decoding; sketch-first/left-to-right/interleaved modes; 21.4% LiveCodeBench [^274^](https://arxiv.org/abs/2509.01142) |
| **Nie et al. (LLaDA)** | First open-source 8B diffusion LLM; baseline for many comparisons; 45.1% HumanEval [^75^](https://arxiv.org/html/2605.10980v1) |
| **Li et al. (NAP paper)** | Diagnosed AR-like behavior in DLMs; data-centric solution; +14.4% GSM8K under parallel decoding [^429^](https://arxiv.org/html/2602.23225v1) |

### Trends & Signals

1. **Diffusion code models are improving rapidly**: Dream-v0 (April 2025) scored 13.3% on LiveCodeBench v1-v6; Dream-Coder (July 2025) reached 24.8% — nearly doubling in 3 months [^9^](https://arxiv.org/html/2509.11252v2). Stable-DiffCoder (January 2026) reached 23.5% LiveCodeBench and dominated CanItEdit.

2. **Closed-source > open-source, but gap is closing**: Mercury-Coder-Small and Gemini Diffusion lead, but Stable-DiffCoder-8B and Dream-Coder-7B approach their performance on many benchmarks.

3. **Speed-quality Pareto frontier is being redefined**: Mercury Coder (1,109 tok/s), Seed Diffusion (2,146 tok/s), and Gemini Diffusion (1,479 tok/s) achieve speeds 5-10x faster than AR models at comparable quality [^71^](https://www.inceptionlabs.ai/blog/introducing-mercury) [^482^](https://neurohive.io/en/state-of-the-art/seed-diffusion-new-state-of-the-art-in-speed-quality-balance-for-code-generation-models/).

4. **Code editing is diffusion's killer app**: CanItEdit results show diffusion models 10-20pp ahead of AR models at the same scale. The non-sequential nature of editing aligns with diffusion's fundamental design.

5. **ARness is the critical metric**: The concept of ARness (Global ARness@1) from Gong et al. and the NAP paper provides a quantitative framework for measuring how much diffusion models "cheat" by mimicking AR behavior [^429^](https://arxiv.org/html/2602.23225v1).

6. **Training data matters more than architecture**: Stable-DiffCoder's controlled comparison (identical architecture/data to Seed-Coder) shows diffusion training itself provides improvement, but commercial models still lead due to data quality [^83^](https://arxiv.org/html/2601.15892v1).

7. **Benchmark saturation is real**: HumanEval approaches 90%+ scores, raising questions about whether it can still differentiate model capabilities [^885^](https://aclanthology.org/2024.findings-emnlp.996.pdf).

### Controversies & Conflicting Claims

1. **"Diffusion models lag AR models" vs. "essentially closed gap"**: The "Beyond Autoregression" paper states "diffusion LLMs are not yet able to replace AR LLMs at the current stage" [^9^](https://arxiv.org/html/2509.11252v2), while Google's O'Donoghue claims "the gap... is essentially closed in terms of benchmark performance" [^37^](https://venturebeat.com/technology/beyond-gpt-architecture-why-googles-diffusion-approach-could-reshape-llm-deployment). The discrepancy arises from which benchmarks are weighted — HumanEval/BigCodeBench show parity, while LiveCodeBench/SWE-Bench show gaps.

2. **ARness trade-off**: High ARness enables competitive accuracy but defeats the purpose of parallel generation. The NAP paper shows forcing low ARness collapses reasoning accuracy [^429^](https://arxiv.org/html/2602.23225v1). Is the diffusion paradigm fundamentally limited, or is it a training data problem?

3. **Speed claims vs. practical reality**: While Mercury claims 10x speedup, the "Beyond Autoregression" paper notes "current open-source diffusion LLMs often remain substantially slower than comparable ARMs in practice" [^784^](https://arxiv.org/html/2510.04146v2) due to lack of optimized inference infrastructure.

4. **Controlled comparisons**: Stable-DiffCoder's claim to "overall outperform its AR counterpart" [^83^](https://arxiv.org/html/2601.15892v1) is from a single controlled study. Broader comparisons across more benchmarks and model families show mixed results.

5. **Benchmark bias**: CanItEdit's strong results for diffusion may reflect genuine architectural advantage, or may simply mean diffusion training happens to produce good editors — not necessarily good reasoners. The LiveCodeBench gap remains the strongest evidence against diffusion superiority.

### Recommended Deep-Dive Areas

1. **LiveCodeBench gap root causes**: Understanding why diffusion models underperform 20-40% relative to AR on competitive programming is critical. Is it training data, the parallel generation mechanism, or benchmark design? The NAP paper's data-centric approach (+14.4% under parallelism) suggests training data restructuring may help [^429^](https://arxiv.org/html/2602.23225v1).

2. **Agentic evaluation of diffusion models**: No diffusion model has been evaluated on SWE-Bench in agentic mode. Given diffusion's strength in global planning, iterative agentic workflows (where the model can refine solutions across multiple turns) may significantly improve results.

3. **Diffusion-native chain-of-thought**: EndoCoT [^857^](https://arxiv.org/html/2603.12252v3) demonstrates that diffusion models can perform genuine step-by-step reasoning through iterative latent state refinement. Extending this to code generation could close the LiveCodeBench gap.

4. **Custom benchmarks for non-sequential code tasks**: Most benchmarks assume left-to-right generation. Developing benchmarks that explicitly require non-sequential generation (e.g., editing, refactoring, cross-file changes) would better measure diffusion's unique advantages.

5. **Scaling laws for diffusion code models**: All evaluations are at 7-8B scale. How do diffusion models scale to 30B+ parameters? Do the gaps with AR models widen or narrow?

6. **Inference-time compute scaling**: Diffusion models can trade diffusion steps for quality. Systematic analysis of this trade-off across benchmarks (as in the "Beyond Autoregression" RQ2/RQ3) provides practical guidance for deployment [^9^](https://arxiv.org/html/2509.11252v2).

7. **Multi-language evaluation**: Stable-DiffCoder shows diffusion's "data augmentation" effect benefits low-resource languages [^83^](https://arxiv.org/html/2601.15892v1). A systematic multi-language benchmark comparison would validate this finding.

---

*Research compiled from 20+ independent searches across arXiv, official publications, and authoritative tech journalism. Last updated: July 2025.*
