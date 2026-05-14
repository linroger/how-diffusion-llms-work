# Dimension 06: Open-Source Diffusion LLM Ecosystem — Deep Dive

## LLaDA, Dream, DiffuCoder, SEDD, Mercury

---

## Key Findings

### 1. LLaDA-8B Training Details

- **LLaDA 8B** was pre-trained from scratch on **2.3 trillion tokens** using **0.13 million H800 GPU hours** (approximately 130,000 H800 GPU hours), followed by SFT on **4.5 million instruction pairs** [^568^](https://arxiv.org/html/2502.09992v3). The architecture closely follows LLaMA3 8B but uses vanilla multi-head attention (instead of GQA), bidirectional attention (no causal mask), and an adjusted FFN dimension of 12,288 to maintain comparable parameter count (~8.02B total) [^568^](https://arxiv.org/html/2502.09992v3).
- The training pipeline is standard: data preparation → pre-training → supervised fine-tuning (SFT) → evaluation. During pre-training, each token is masked at ratio t sampled uniformly from [0,1]. During SFT, only response tokens are masked (prompt tokens remain visible) — the natural equivalent of AR models computing loss only on the response during instruction tuning [^31^](https://www.dailydoseofds.com/diffusion-models-part-2/).
- On **MMLU**, LLaDA 8B scores **65.9** vs LLaMA3 8B's 65.4. On **HumanEval**, LLaDA reaches **33.5%** vs LLaMA3's 34.2%. Notably, LLaDA **surpasses GPT-4o on reversal poem completion**, directly demonstrating the bidirectional advantage of diffusion models [^31^](https://www.dailydoseofds.com/diffusion-models-part-2/).
- LLaDA was accepted as an **oral presentation at NeurIPS 2025** [^31^](https://www.dailydoseofds.com/diffusion-models-part-2/).
- The model was trained with AdamW optimizer using a **Warmup-Stable-Decay (WSD)** learning rate schedule: 2000-iteration warmup to 4e-4, stable at 4e-4, mid-training drop to 1e-4 after 1.2T tokens, final decay to 1e-5 over the last 0.3T tokens. Weight decay = 0.1, global batch size = 1280 [^571^](https://darkenstar.github.io/blogs/llada/).

### 2. LLaDA 1.5 VRPO (Variance-Reduced Preference Optimization)

- **LLaDA 1.5** addresses the core challenge of aligning diffusion LLMs with human preferences: the **high variance in ELBO-based likelihood estimates** required for preference optimization [^215^](https://arxiv.org/abs/2505.19223).
- The key insight: preference optimization for diffusion models requires estimating the model's log-likelihood for given outputs, but diffusion models cannot compute exact log-probability and instead rely on ELBO approximations. These ELBO estimates are noisy (high-variance), making preference-based gradient updates extremely unstable [^217^](https://arxiv.org/pdf/2506.13759).
- **VRPO introduces three principled variance-reduction techniques:**
  1. **Increased sampling budget** for ELBO estimates (using more random draws of diffusion time and mask patterns)
  2. **Optimal allocation**: sample many different diffusion timesteps but only one mask per timestep — this minimizes variance for a fixed total number of samples
  3. **Antithetic sampling**: share the same random noise (timesteps and mask patterns) between winning and losing outputs in preference comparisons, so random errors in their log-likelihood estimates tend to cancel out [^215^](https://arxiv.org/abs/2505.19223) [^66^](https://aman.ai/primers/ai/diffusion-LLMs/)
- Training consumed approximately **405 H100 GPU hours** for 8 Monte Carlo samples — less than **0.5% of pre-training cost** [^216^](https://arxiv.org/pdf/2505.19223).
- **LLaDA 1.5 results**: improvements of **GSM8K +4.7**, **HumanEval +3.0**, **MBPP +1.8**, **IFEval +4.0**, **Arena-Hard +4.3** over the SFT-only LLaDA Instruct baseline [^215^](https://arxiv.org/abs/2505.19223).
- LLaDA 1.5 was trained on **350K preference pairs** covering 35% creative writing, 18% knowledge QA, 16% NLP tasks, 14% mathematics, 7% recommendations, 5% code, 3% reasoning, plus safety tasks [^216^](https://arxiv.org/pdf/2505.19223).

### 3. Dream Adaptive Decoding & Context-Adaptive Noise Rescheduling

- **Dream 7B** is initialized from pretrained autoregressive model weights (Qwen2.5-Coder) rather than trained from scratch. It uses a **"Shift Operation"** strategy that preserves the positional relationship learned by AR models — the model continues to use hidden state h_i to generate predictions for position i+1, contrasting with conventional diffusion that predicts masked tokens at their original positions [^257^](https://arxiv.org/html/2508.15487v1).
- **Context-adaptive token-level noise rescheduling**: Dream re-decides the noise level for each masked token by measuring its contextual "informationness." It uses a **mixture of geometric distributions** to quantify the information contribution of each clean token relative to noised tokens [^257^](https://arxiv.org/html/2508.15487v1):

  > w(t, x_t, n) = (1/2) * sum_i [1[x_t^i ≠ MASK] * Geo(p, |n-i|-1)]

  where p controls the sharpness — smaller p means clean tokens contribute uniformly to all masked tokens; larger p forces nearby clean tokens to have greater influence.

- Dream supports multiple remasking strategies: `random`, `maskgit_plus` (top-1 confidence), `topk_margin` (top1-top2 margin), and `entropy` (token distribution entropy). The `entropy` strategy with `alg_temp=0` is the default and equivalent to low-confidence remasking [^703^](https://github.com/DreamLM/Dream).

### 4. Dream-Coder Open-Source Release

- **Dream-Coder 7B** was released on **July 15, 2025** by HKU + Huawei Noah's Ark Lab. It is positioned as a **"fully open"** 7B dLLM for code — trained exclusively on open-source/public data across all stages (adaptation, SFT, RL) [^171^](https://github.com/DreamLM/Dream-Coder).
- **What's released**: Dream-Coder-7B and Dream-Coder-7B-Instruct **checkpoints**, **complete training recipes**, **preprocessing pipelines**, and **inference code** [^59^](https://arxiv.org/html/2509.01142v1).
- **Training data sources**: OpenCoder, Stack-Edu, Dolmino, DCLM-Baseline [^573^](http://mp.weixin.qq.com/s?__biz=MzkyNzQ5NjA1MA==&mid=2247485004&idx=1&sn=5a5a775c4c12c96b61c391a5fe672070).
- **Post-training recipe**: (i) SFT with random truncation and padding penalty to mitigate padding pathologies; (ii) RL with verifiable rewards over curated high-quality prompts using a tailored RL recipe for diffusion LMs [^59^](https://arxiv.org/html/2509.01142v1).
- **Benchmark results**: **21.4% pass@1 on LiveCodeBench** (2410-2505) — on par with Mercury Coder Small (22.9%) and outperforming OpenCoder 8B Instruct. **HumanEval: 82.9%**, **MBPP: 79.6%**, **EvalPlus: 73.1%** [^274^](https://arxiv.org/abs/2509.01142) [^287^](https://www.emergentmind.com/topics/diffusion-style-code-models).
- **Emergent any-order generation capabilities**: Dream-Coder exhibits three distinct generation patterns: (a) **sketch-first** for complex algorithms (generates structural elements first, fills in details), (b) **left-to-right** for straightforward completions, (c) **interleaved reasoning** for logic-intensive tasks (generates key conditions first, then supporting code) [^59^](https://arxiv.org/html/2509.01142v1).

### 5. DiffuCoder Coupled-GRPO

- **DiffuCoder 7B** is Apple's open-source diffusion model for code, trained on **130B tokens** of code data. It introduces **coupled-GRPO**, a diffusion-native RL algorithm [^153^](https://arxiv.org/html/2506.20639v1).
- **Training pipeline** (4 stages):
  1. **Adaptation pre-training**: Continual pre-training on 400B-token code corpus from RefineCode and Stackv2; early stopping at 65B tokens
  2. **Mid-training**: 16B tokens of annealing code data (algorithmic corpus + synthetic data); trained on 8 nodes × 8 A100 GPUs for 90 hours
  3. **Instruction tuning**: 436K SFT samples from OpenCoder using classifier-free guidance; trained on 8 nodes × 8 H100 GPUs for ~24 hours
  4. **RL post-training**: Coupled-GRPO on 21K hard samples from Acecoder-87K [^153^](https://arxiv.org/html/2506.20639v1) [^602^](https://www.marktechpost.com/2025/07/16/apple-introduces-diffucoder-a-7b-diffusion-llm-tailored-for-code-generation/)
- **Coupled-GRPO core idea**: Constructs **complementary mask noise** for completions used in training. For each completion, two masks are generated such that **every token position is masked in exactly one of the two masks** — together they cover all completion tokens. This guarantees:
  1. Each token's log-probability is computed at least once (non-zero learning signal)
  2. Log-probability estimations are more accurate (evaluated under realistic partial-masking context)
  3. 2λ additional samples compared to baseline [^153^](https://arxiv.org/html/2506.20639v1)
- **Results**: +4.4% improvement on EvalPlus from coupled-GRPO; HumanEval 72.0%, MBPP 65.2%, EvalPlus 75.1%. Notably, RL training increases optimal sampling temperature from 0.2 to higher values, reducing reliance on strict AR causal decoding [^153^](https://arxiv.org/html/2506.20639v1).
- **Key finding**: dLLMs can decide how causal their generation should be without semi-AR decoding, and increasing temperature diversifies not only token choices but also **generation order** — creating a rich search space for RL rollouts [^153^](https://arxiv.org/html/2506.20639v1).

### 6. SEDD Score Entropy

- **SEDD (Score Entropy Discrete Diffusion)**, authored by **Aaron Lou, Chenlin Meng, and Stefano Ermon** from **Stanford University**, won **ICML 2024 Best Paper** [^518^](https://zhichai.net/topic/177168598) [^551^](https://dl.acm.org/doi/10.5555/3692070.3693403).
- **Core problem**: Standard diffusion models rely on score matching (∇ₓ log p(x)), which generalizes naturally to continuous spaces (images) but **fails for discrete data** like text — because gradients don't exist for discrete token indices [^518^](https://zhichai.net/topic/177168598).
- **Score entropy solution**: Instead of modeling gradients, SEDD parameterizes the reverse discrete diffusion process using **ratios of the data distribution** p_t(y)/p_t(x). These probability ratios are learned using a novel **score entropy loss** that naturally extends score matching to discrete spaces [^541^](https://arxiv.org/abs/2310.16834) [^544^](https://arxiv.org/pdf/2310.16834).
- The score entropy loss is:

  > D_SE(p_θ(·|x_t) || p_t(·|x_0)) = sum_y [p_t(y|x_0)/p_t(x) * (log(p_t(y|x_0)/p_t(x)) - log(s_θ(x_t)_y))]

- **Empirical results**: SEDD beats existing language diffusion paradigms (reducing perplexity by **25-75%**) and is competitive with autoregressive models, in particular **outperforming GPT-2**. SEDD achieves **6-8× better generative perplexity** than un-annealed GPT-2, and can match GPT-2 quality with **32× fewer network evaluations**. It enables **controllable infilling** — matching nucleus sampling quality while enabling non-left-to-right prompting [^541^](https://arxiv.org/abs/2310.16834) [^545^](https://arxiv.org/abs/2310.16834).
- **From SEDD to Mercury**: SEDD co-author **Stefano Ermon** went on to co-found **Inception Labs**, which commercialized the diffusion-language approach as the **Mercury** model family. The SEDD paper's fundamental insight — that probability ratios (not gradients) enable diffusion in discrete spaces — directly informs Mercury's architecture [^518^](https://zhichai.net/topic/177168598) [^583^](https://www.pymnts.com/artificial-intelligence-2/2025/silicon-valley-startup-inception-labs-creates-faster-llm/).

### 7. Mercury Coder (Inception Labs)

- **Mercury Coder**, launched February 2025, is the **world's first commercially available diffusion LLM** (dLLM), developed by **Inception Labs** [^71^](https://www.inceptionlabs.ai/blog/introducing-mercury) [^586^](https://www.inceptionlabs.ai/blog/mercury-refreshed).
- **Company**: Inception Labs was founded in 2024 in Palo Alto by **Stefano Ermon** (Stanford professor, co-inventor of diffusion methods behind Midjourney and Sora), **Aditya Grover** (UCLA professor), and **Volodymyr Kuleshov** (Cornell professor). The founding trio have worked together for over 10 years on AI research [^519^](https://nextomoro.com/inception-labs/) [^577^](https://www.mayfield.com/introducing-inception-labs/).
- **Funding**: **$50 million seed round** announced November 2025, led by **Menlo Ventures**, with participation from Mayfield, Innovation Endeavors, **NVentures (NVIDIA)**, **M12 (Microsoft)**, **Snowflake Ventures**, **Databricks Investment**, and angel investors **Andrew Ng** and **Andrej Karpathy** [^586^](https://www.inceptionlabs.ai/blog/mercury-refreshed) [^589^](https://www.businesswire.com/news/home/20251106570339/en/Inception-Raises-245M-to-Power-Diffusion-LLMs-Increasing-LLM-Speed-and-Efficiency-by-up-to-10X-and-Unlocking-Real-Time-Accessible-AI-Applications).
- **API pricing**: **$0.25 per million input tokens**, **$1.00 per million output tokens** — substantially below frontier AR model pricing [^519^](https://nextomoro.com/inception-labs/) [^552^](https://langdb.ai/app/models/mercury-coder).
- **Performance**: Mercury Coder Mini achieves **1,109 tokens/sec** on NVIDIA H100 GPUs; Mercury Coder Small achieves **737 tokens/sec**. This is **5-10× faster** than speed-optimized frontier models while maintaining comparable quality [^45^](https://arxiv.org/abs/2506.17298) [^71^](https://www.inceptionlabs.ai/blog/introducing-mercury).
- **Benchmarks**: Mercury Coder Small achieves **90.0% HumanEval**, **76.6% MBPP**, **80.4% EvalPlus**, **25.0% LiveCodeBench**, **45.5% BigCodeBench** [^71^](https://www.inceptionlabs.ai/blog/introducing-mercury).
- **Availability**: API via platform.inceptionlabs.ai, also available on Amazon Bedrock, OpenRouter, and Poe. Integrated into developer tools including ProxyAI, Buildglare, and Kilo Code. **No open weights released** as of April 2026 [^519^](https://nextomoro.com/inception-labs/).

---

## Community Benchmark Comparison

### Code Generation (pass@1)

| Model | Size | HumanEval | MBPP | LiveCodeBench v6 | EvalPlus |
|-------|------|-----------|------|------------------|----------|
| **LLaDA-8B-Instruct** | 8B | 45.1% | 39.4% | 9.2% | — |
| **LLaDA-1.5** | 8B | 43.3% | 40.4% | 9.2% | — |
| **Dream-v0-Instruct-7B** | 7B | 56.7% | 56.8% | 11.5% | — |
| **Dream-Coder-v0-Instruct-7B** | 7B | 76.2% | 65.8% | 18.3% | — |
| **DreamOn-v0-7B** | 7B | 51.2% | 53.0% | 9.2% | — |
| **DiffuCoder-7B-cfGRPO** | 7B | 69.5% | 64.2% | 8.4% | 75.1% |
| **Mercury-Coder-Small** (closed) | — | 86.0% | 76.2% | 22.1% | 80.4% |
| **Gemini-Diffusion** (closed) | — | 89.6% | 76.0% | 30.9% | — |
| **Qwen3-8B** (AR baseline) | 8B | 82.9% | 68.8% | 26.0% | — |
| **Seed-Coder-8B** (AR baseline) | 8B | 84.8% | 70.8% | 22.1% | — |

> Source: [^9^](https://arxiv.org/html/2509.11252v2) — "Beyond Autoregression: An Empirical Study of Diffusion Large Language Models for Code Generation"

### Key Observations from Benchmark Comparison

1. **Dream-Coder leads open-source diffusion models on LiveCodeBench** (18.3% v6), approaching Mercury Coder Small (22.1%). On HumanEval, Dream-Coder (76.2%) outperforms DiffuCoder (69.5%) by a wide margin [^9^](https://arxiv.org/html/2509.11252v2).
2. **All open-source diffusion LLMs lag behind AR models on average**: Diffusion LLMs average 66.7% on HumanEval vs 71.3% for AR LLMs; 14.9% vs 18.9% on LiveCodeBench v6 [^9^](https://arxiv.org/html/2509.11252v2).
3. **Closed-source diffusion models are competitive with AR models**: Gemini-Diffusion (89.6% HumanEval) surpasses Seed-Coder-8B (84.8%). Mercury Coder Small (86.0%) exceeds most AR baselines [^9^](https://arxiv.org/html/2509.11252v2).
4. **Diffusion LLMs show substantially better long-context understanding**: On RepoQA, diffusion LLMs maintain robust performance as context length increases, while AR LLMs decline rapidly. At 4k tokens, Llama-2-7B drops below 10% retrieval accuracy while DiffuCoder maintains above 30%. Mercury-Coder-Small shows only ~15% decrease when extrapolating from 8k to 64k, vs Qwen3-8B's ~30% drop [^9^](https://arxiv.org/html/2509.11252v2).
5. **Low-confidence remasking is critical**: Random remasking vs low-confidence remasking shows massive gaps — e.g., LLaDA-8B-Instruct goes from 11.6% to 43.9% on HumanEval (+32.3%) with low-confidence remasking [^9^](https://arxiv.org/html/2509.11252v2).

---

## Licensing Analysis

| Model | License | Weights | Code | Training Data | Commercial Use |
|-------|---------|---------|------|---------------|----------------|
| **LLaDA 8B / 1.5 / 2.0** | **MIT** | Open | Open | Open | Yes |
| **Dream 7B** | **Apache 2.0** | Open | Open | Open | Yes |
| **Dream-Coder 7B** | **Apache 2.0** | Open | Open | Open | Yes |
| **DiffuCoder 7B** | Apple OSS License | Open | Open | Open | Yes |
| **SEDD** | Open (GitHub) | Open | Open | Open | Yes |
| **Mercury** | **Proprietary** | Closed | Closed | Closed | API-only |
| **Gemini-Diffusion** | **Proprietary** | Closed | Closed | Closed | API-only |
| **Seed-Diffusion** | **Proprietary** | Closed | Closed | Closed | API-only |

### Key Licensing Observations

- **LLaDA uses MIT license** — one of the most permissive open-source licenses, allowing unrestricted commercial use, modification, and redistribution with minimal requirements [^169^](https://github.com/ML-GSAI/LLaDA).
- **Dream family uses Apache 2.0** — also permissive, with explicit patent grant, making it safe for commercial adoption [^714^](https://github.com/DreamLM/Dream/blob/main/LICENSE).
- **Apple's DiffuCoder** is released under Apple's standard open-source license (similar to Apache 2.0), with full code, recipes, and model weights on HuggingFace [^170^](https://github.com/apple/ml-diffucoder).
- **Mercury, Gemini-Diffusion, and Seed-Diffusion are fully proprietary/closed-source** — accessible only via API, no weights or training details released [^519^](https://nextomoro.com/inception-labs/).
- **LLaDA 2.0 (100B)** from Ant Group/InclusionAI is fully open-sourced on HuggingFace under permissive terms [^408^](https://github.com/inclusionAI/LLaDA2.0/blob/main/README.md).

---

## HuggingFace Presence & Ecosystem

### Model Availability on HuggingFace

| Model | HuggingFace Handle | Status |
|-------|-------------------|--------|
| LLaDA-8B-Base | `GSAI-ML/LLaDA-8B-Base` | Available |
| LLaDA-8B-Instruct | `GSAI-ML/LLaDA-8B-Instruct` | Available |
| LLaDA-1.5 | `GSAI-ML/LLaDA-1.5` | Available |
| LLaDA-MoE-7B | `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` | Available |
| LLaDA 2.0-mini (16B) | `inclusionAI/LLaDA2.0-mini` | Available |
| LLaDA 2.0-flash (100B) | `inclusionAI/LLaDA2.0-flash` | Available |
| LLaDA 2.1-mini | `inclusionAI/LLaDA2.1-mini` | Available |
| LLaDA 2.1-flash | `inclusionAI/LLaDA2.1-flash` | Available |
| Dream-v0-Instruct-7B | `Dream-org/Dream-v0-Instruct-7B` | Available |
| Dream-Coder-v0-Instruct-7B | `Dream-org/Dream-Coder-v0-Instruct-7B` | Available |
| DiffuCoder-7B | Apple HF page | Available |
| Mercury | **Closed API only** | `platform.inceptionlabs.ai` |

> Sources: [^169^](https://github.com/ML-GSAI/LLaDA), [^703^](https://github.com/DreamLM/Dream), [^170^](https://github.com/apple/ml-diffucoder), [^408^](https://github.com/inclusionAI/LLaDA2.0/blob/main/README.md)

### GitHub Community Metrics

| Repository | Stars | Forks | Contributors | License |
|------------|-------|-------|-------------|---------|
| ML-GSAI/LLaDA (LLaDA 8B) | **3,800** | 264 | 9 | MIT |
| DreamLM/Dream (Dream 7B) | **~1,231** | 77 | 20+ | Apache 2.0 |
| DreamLM/Dream-Coder | **~95** | 6 | — | Apache 2.0 |
| DreamLM/DreamOn | **112** | 10 | — | Apache 2.0 |
| apple/ml-diffucoder | **821** | 56 | 3 | Apple OSS |
| inclusionAI/LLaDA2.X | Growing | — | — | Open |

> Sources: [^169^](https://github.com/ML-GSAI/LLaDA), [^170^](https://github.com/apple/ml-diffucoder), [^715^](https://github.com/DreamLM), [^703^](https://github.com/DreamLM/Dream)

### Ecosystem Integrations

- **dInfer**: Efficient inference framework by InclusionAI supporting LLaDA, LLaDA-MoE, and LLaDA2 variants with batched inference [^598^](https://github.com/inclusionAI/dInfer)
- **Information-Gain Sampler**: ICML 2026 paper providing a unified decoding framework for MDMs (LLaDA, Dream, etc.) that replaces greedy local-certainty heuristics with principled information-gain objectives [^578^](https://github.com/yks23/Information-Gain-Sampler)
- **A-CFG**: Adaptive Classifier-Free Guidance for diffusion LMs that dynamically re-masks low-confidence tokens at every denoising step; plug-and-play for LLaDA and Dream [^595^](https://github.com/pixeli99/A-CFG)
- **MLX support for DiffuCoder**: Community-driven Apple Silicon (MLX) implementation of DiffuCoder in progress [^170^](https://github.com/apple/ml-diffucoder)

---

## Trends & Signals

### Trend 1: From-Scratch vs. AR-Conversion Divergence
- **LLaDA** trains from scratch (2.3T tokens, 130K H800 hours), proving diffusion LLMs can match AR models when given equivalent compute [^568^](https://arxiv.org/html/2502.09992v3).
- **Dream** converts pretrained AR checkpoints via "Shift Operation" and context-adaptive noise, leveraging AR model quality as initialization [^257^](https://arxiv.org/html/2508.15487v1).
- **LLaDA 2.0** adopts AR-to-diffusion conversion at 100B scale via the novel **WSD (Warmup-Stable-Decay)** 3-phase training, making conversion practical at frontier scale [^24^](https://arxiv.org/html/2512.15745v2).
- This divergence suggests the field is converging on **AR initialization + diffusion fine-tuning** as the practical path, though from-scratch training remains viable.

### Trend 2: RL for Diffusion is Maturing Rapidly
- **VRPO** (LLaDA 1.5): Variance-reduced DPO for general alignment — 405 H100 GPU hours [^215^](https://arxiv.org/abs/2505.19223).
- **Coupled-GRPO** (DiffuCoder): Diffusion-native RL with complementary mask sampling — +4.4% EvalPlus [^153^](https://arxiv.org/html/2506.20639v1).
- **d1-LLaDA**: Adapts GRPO to MDMs via one-step log-probability estimator with random prompt masking [^516^](https://arxiv.org/pdf/2509.21912).
- **IGPO**: Leverages inpainting ability to guide RL exploration [^150^](https://arxiv.org/html/2508.10875v2).
- This signals that **RL alignment for diffusion is moving from research curiosity to production reality**.

### Trend 3: Speed as the Killer Feature
- Mercury Coder Mini achieves **1,109 tokens/sec** — previously achievable only on specialized hardware (Groq, Cerebras) [^45^](https://arxiv.org/abs/2506.17298).
- LLaDA 2.0-flash-CAP achieves **535 tokens/s** with Confidence-Aware Parallel decoding [^408^](https://github.com/inclusionAI/LLaDA2.0/blob/main/README.md).
- Gemini Diffusion reports **5× speed improvements** over comparable AR models [^1^](https://arxiv.org/html/2603.22075v1).
- The **speed-quality frontier** is becoming the primary competitive axis for diffusion LLMs.

### Trend 4: Code as the Killer Application
- All major open-source diffusion models have **code-specialized variants**: Dream-Coder, DiffuCoder, Mercury Coder, Seed-Diffusion, Gemini-Diffusion.
- Code generation benefits uniquely from diffusion's **global planning** (sketch-first), **iterative refinement** (debugging-like), and **any-order generation** (non-sequential programming patterns) [^59^](https://arxiv.org/html/2509.01142v1).
- Dream-Coder's 21.4% on LiveCodeBench approaches Mercury Coder Small (22.9%) — suggesting **open-source diffusion code models are closing the gap with commercial systems** [^59^](https://arxiv.org/html/2509.01142v1).

### Trend 5: MoE Scaling for Diffusion
- **LLaDA 2.0-flash** (100B total, 6.1B active) is the first 100B-parameter diffusion LLM, using MoE to keep inference costs manageable [^24^](https://arxiv.org/html/2512.15745v2).
- **LLaDA-MoE-7B** uses only ~1B active parameters while surpassing LLaDA 1.5 (8B dense), demonstrating MoE's viability for diffusion [^169^](https://github.com/ML-GSAI/LLaDA).
- This shows the **entire MoE toolkit** (expert parallelism, load balancing, routing) transfers directly from AR to diffusion models.

---

## Controversies & Conflicting Claims

### Controversy 1: Can Diffusion LLMs Actually Replace AR Models?
- **Pro-diffusion**: LLaDA 8B "demonstrates comparable results to ARM baselines trained on the same data across six tasks" and "challenges the common assumption that core LLM capabilities inherently depend on ARMs" [^568^](https://arxiv.org/html/2502.09992v3). Inception Labs envisions "a future where all LLMs are going to be based on the diffusion paradigm" [^583^](https://www.pymnts.com/artificial-intelligence-2/2025/silicon-valley-startup-inception-labs-creates-faster-llm/).
- **Skeptical**: The comprehensive empirical study found that "diffusion LLMs are not yet able to replace AR LLMs at the current stage in code generation" — diffusion LLMs average 66.7% on HumanEval vs 71.3% for AR LLMs [^9^](https://arxiv.org/html/2509.11252v2). "LLaDA-Instruct" actually *degraded* from base on coding benchmarks (-18.6% on HumanEval+) [^153^](https://arxiv.org/html/2506.20639v1).
- **Resolution**: The gap is closing rapidly. Newer models (Dream-Coder, Gemini-Diffusion) match or exceed AR baselines on specific benchmarks. The reversal curse advantage and long-context robustness are genuine diffusion strengths.

### Controversy 2: Open-Source vs. Commercial Data Quality Gap
- All open-source diffusion LLMs lag behind closed-source counterparts (Mercury, Gemini-Diffusion) in code generation. The empirical study attributes this to "the availability of higher-quality training data in commercial settings" [^9^](https://arxiv.org/html/2509.11252v2).
- Dream-Coder challenges this by being "trained exclusively on public data" and matching Mercury Coder Small on LiveCodeBench (21.4% vs 22.9%) [^59^](https://arxiv.org/html/2509.01142v1).
- This suggests the gap is **not inherent** — careful data curation and RL training can close it.

### Controversy 3: Inference Speed vs. Quality Tradeoff
- Diffusion models require multiple forward passes (denoising steps) vs. AR models' single pass per token. On paper, this should make diffusion *slower*.
- In practice, **parallel token generation** across the full sequence enables massive throughput: Mercury achieves 1,109 tokens/sec vs. 59 for GPT-4o Mini on the same hardware [^45^](https://arxiv.org/abs/2506.17298).
- However, latency for short outputs may still favor AR models (fewer total forward passes). The speed advantage is most pronounced for **long-form generation**.

---

## Major Players & Sources

| Entity | Role / Relevance |
|--------|-----------------|
| **Ant Group / InclusionAI** (China) | LLaDA family (8B→100B). Largest open-source diffusion LLM effort. MIT license. |
| **HKU / Huawei Noah's Ark Lab** | Dream family (Dream 7B, Dream-Coder 7B). Apache 2.0. Pioneered AR-initialized diffusion. |
| **Apple** | DiffuCoder 7B. Coupled-GRPO for code. Released recipes and code. |
| **Inception Labs** (Stanford/UCLA/Cornell) | Mercury — first commercial dLLM. $50M funding. Closed-source but API-accessible. |
| **Google DeepMind** | Gemini-Diffusion. Closed-source commercial model with reported 5× speedup. |
| **ByteDance / Seed Team** | Seed-Diffusion. Code-oriented diffusion model. Closed-source. |
| **Stanford (SEDD authors)** | SEDD paper (ICML 2024 Best Paper). Theoretical foundation for discrete diffusion. Aaron Lou → commercialization via Ermon → Mercury. |
| **Renmin University / ML-GSAI** | LLaDA training infrastructure and research. VRPO development. |

---

## Recommended Deep-Dive Areas

1. **LLaDA 2.0's WSD training paradigm**: The 3-phase block-level WSD (Warmup-Stable-Decay) approach for converting AR to diffusion at 100B scale represents a breakthrough. Understanding the mechanics of progressive block-size scheduling could be foundational for the field. [^24^](https://arxiv.org/html/2512.15745v2)

2. **Coupled-GRPO and diffusion-native RL**: Apple's coupled-GRPO represents the most sophisticated RL algorithm designed specifically for diffusion models. Its complementary mask sampling scheme could generalize beyond code to other domains. [^153^](https://arxiv.org/html/2506.20639v1)

3. **Mercury's commercial trajectory**: At $0.25/M input tokens and 1000+ tokens/sec, Mercury is positioned as the "fast inference" option. Tracking developer adoption, enterprise partnerships, and whether the quality gap with frontier AR models persists will determine dLLM commercial viability. [^519^](https://nextomoro.com/inception-labs/)

4. **Long-context capabilities of diffusion LLMs**: The finding that diffusion LLMs exhibit substantially better length extrapolation than AR models (RepoQA results) could be a genuine differentiator for repo-level code understanding and long-document analysis. [^9^](https://arxiv.org/html/2509.11252v2)

5. **SEDD's theoretical legacy and Mercury's engineering**: Understanding how SEDD's probability-ratio formulation (ICML 2024 Best Paper) was adapted for Mercury's commercial-scale system would bridge theory and practice. The SEDD-to-Mercury pipeline (Lou → Ermon → Inception Labs) is a rare case of academic research directly translating to commercial product.

6. **The inference ecosystem**: Tools like dInfer (by InclusionAI), Information-Gain Sampler (ICML 2026), and A-CFG represent an emerging ecosystem of diffusion-specific inference optimizations that could collectively close the efficiency gap with AR models. [^598^](https://github.com/inclusionAI/dInfer) [^578^](https://github.com/yks23/Information-Gain-Sampler)

---

## Sources & References

- [^1^] Autoregressive vs. Masked Diffusion Language Models: A Controlled Comparison (arXiv 2026) — https://arxiv.org/html/2603.22075v1
- [^9^] Beyond Autoregression: An Empirical Study of Diffusion LLMs for Code Generation (arXiv 2025) — https://arxiv.org/html/2509.11252v2
- [^24^] LLaDA2.0: Scaling Up Diffusion Language Models to 100B (arXiv 2025) — https://arxiv.org/html/2512.15745v2
- [^31^] Diffusion LLMs from the Ground Up (Daily Dose of DS, 2026) — https://www.dailydoseofds.com/diffusion-models-part-2/
- [^45^] Mercury: Ultra-Fast Language Models Based on Diffusion (arXiv 2025) — https://arxiv.org/abs/2506.17298
- [^59^] Dream-Coder 7B: An Open Diffusion Language Model for Code (arXiv 2025) — https://arxiv.org/html/2509.01142v1
- [^66^] Primers: Diffusion LLMs (aman.ai, 2026) — https://aman.ai/primers/ai/diffusion-LLMs/
- [^71^] Introducing Mercury (Inception Labs Blog, 2025) — https://www.inceptionlabs.ai/blog/introducing-mercury
- [^85^] Mercury: Ultra-Fast Language Models Based on Diffusion (arXiv 2025) — https://arxiv.org/html/2506.17298v1
- [^150^] A Survey on Diffusion Language Models (arXiv 2025) — https://arxiv.org/html/2508.10875v2
- [^153^] DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation (arXiv 2025) — https://arxiv.org/html/2506.20639v1
- [^169^] LLaDA GitHub Repository (ML-GSAI, 2025) — https://github.com/ML-GSAI/LLaDA
- [^170^] DiffuCoder GitHub Repository (Apple, 2025) — https://github.com/apple/ml-diffucoder
- [^171^] Dream-Coder GitHub Repository (DreamLM, 2025) — https://github.com/DreamLM/Dream-Coder
- [^215^] LLaDA 1.5: VRPO for Large Language Diffusion Models (arXiv 2025) — https://arxiv.org/abs/2505.19223
- [^216^] LLaDA 1.5 VRPO (arXiv PDF, 2025) — https://arxiv.org/pdf/2505.19223
- [^217^] Discrete Diffusion in LLMs Survey (arXiv 2025) — https://arxiv.org/pdf/2506.13759
- [^246^] Blockwise SFT for Diffusion Language Models (arXiv 2025) — https://arxiv.org/html/2508.19529v1
- [^257^] Dream 7B: Diffusion Large Language Models (arXiv 2025) — https://arxiv.org/html/2508.15487v1
- [^274^] Dream-Coder 7B (arXiv Abstract, 2025) — https://arxiv.org/abs/2509.01142
- [^402^] Milestone: First 100B Diffusion Language Model (36kr, 2025) — https://eu.36kr.com/en/p/3592063556468736
- [^408^] LLaDA2.X README (GitHub, 2025) — https://github.com/inclusionAI/LLaDA2.0/blob/main/README.md
- [^516^] Diffusion-based and Flow-based LLMs Survey (arXiv 2025) — https://arxiv.org/pdf/2509.21912
- [^518^] SEDD: Score Entropy Discrete Diffusion (zhichai.net, 2026) — https://zhichai.net/topic/177168598
- [^519^] Inception Labs Overview (nextomoro.com, 2026) — https://nextomoro.com/inception-labs/
- [^541^] Discrete Diffusion Modeling by Estimating Ratios (arXiv 2023) — https://arxiv.org/abs/2310.16834
- [^544^] SEDD Paper PDF (arXiv 2023) — https://arxiv.org/pdf/2310.16834
- [^551^] SEDD ICML 2024 Proceedings — https://dl.acm.org/doi/10.5555/3692070.3693403
- [^568^] Large Language Diffusion Models (arXiv 2025) — https://arxiv.org/html/2502.09992v3
- [^571^] LLaDA Technical Deep Dive Blog (2025) — https://darkenstar.github.io/blogs/llada/
- [^577^] Introducing Inception Labs (Mayfield, 2025) — https://www.mayfield.com/introducing-inception-labs/
- [^578^] Information-Gain Sampler GitHub (2026) — https://github.com/yks23/Information-Gain-Sampler
- [^583^] Inception Labs Creates Faster LLM (PYMNTS, 2025) — https://www.pymnts.com/artificial-intelligence-2/2025/silicon-valley-startup-inception-labs-creates-faster-llm/
- [^584^] Race to Production-Grade Diffusion LLMs (TWIML, 2026) — https://twimlai.com/podcast/twimlai/race-production-grade-diffusion-llms
- [^586^] Mercury Refresh: Scaling Up (Inception Labs, 2026) — https://www.inceptionlabs.ai/blog/mercury-refreshed
- [^589^] Inception Raises $50M (BusinessWire, 2025) — https://www.businesswire.com/news/home/20251106570339/en/
- [^598^] dInfer GitHub (InclusionAI, 2025) — https://github.com/inclusionAI/dInfer
- [^601^] Apple Open Sources DiffuCoder (InfoQ, 2025) — https://www.infoq.com/news/2025/07/apple-diffucoder/
- [^602^] Apple Introduces DiffuCoder (MarkTechPost, 2025) — https://www.marktechpost.com/2025/07/16/apple-introduces-diffucoder/
- [^703^] Dream GitHub Repository (DreamLM, 2025) — https://github.com/DreamLM/Dream
- [^714^] Dream LICENSE (GitHub, 2025) — https://github.com/DreamLM/Dream/blob/main/LICENSE
- [^715^] DreamLM GitHub Organization (2026) — https://github.com/DreamLM

---

*Research compiled: July 2025. 22+ targeted searches conducted across arXiv, GitHub, official blogs, and tech journalism.*
