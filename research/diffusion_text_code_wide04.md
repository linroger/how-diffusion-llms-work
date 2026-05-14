## Facet: Open-Source Diffusion LLM Ecosystem for Text and Code

### Key Findings

#### 1. The LLaDA Family — The Most Comprehensive Diffusion LLM Lineage

- **LLaDA-8B** (Nie et al., February 2025) was the first open-source diffusion LLM trained from scratch at the billion-parameter scale. It uses a standard Transformer with bidirectional attention, trained on 2.3 trillion tokens using 0.13 million H800 GPU hours via masked diffusion (ELBO objective). On MMLU it scored 65.9 (vs. LLaMA3-8B's 65.4), on HumanEval 33.5 (vs. LLaMA3's 34.2) [^31^]. The GitHub repo has **3.8k stars**, 264 forks, and 9 contributors [^169^]. License: **MIT**.
- **LLaDA 1.5** (Zhu et al., May 2025) added Variance-Reduced Preference Optimization (VRPO), a diffusion-native alignment method that reduces gradient variance during preference optimization. Trained on 350K preference pairs and 4.5M SFT pairs. Improvements: +4.7 on GSM8K, +3.0 on HumanEval, +1.8 on MBPP, +4.0 on IFEval, +4.3 on Arena-Hard [^80^]. Available as `GSAI-ML/LLaDA-1.5` on HuggingFace [^73^].
- **LLaDA 2.0** (Bie et al., November 2025) was the first dLLM scaled to **100B parameters** (Mixture of Experts), with two variants: LLaDA2.0-mini (16B total, 1.4B active) and LLaDA2.0-flash (100B total, 6.1B active). Key innovation: block diffusion with a three-phase WSD (warm-up, stable, decay) training scheme that converts pre-trained AR models to diffusion, rather than training from scratch. Context window: 32,768 tokens. Vocab: 157,184. License: **Apache 2.0**. GitHub: 419 stars [^106^] [^74^].
- **LLaDA 2.1** (Bie et al., February 2026) introduced **Token Editing (T2T)**, combining Mask-to-Token (M2T) and Token-to-Token (T2T) editing for configurable speed/quality tradeoffs. Two inference modes: Speedy Mode (S-Mode) for fast drafting with retroactive correction, and Quality Mode (Q-Mode) for conservative thresholds. Achieved 892 TPS on HumanEval+, 801 TPS on BigCodeBench, and 663 TPS on LiveCodeBench on the 100B model. Also introduced the first large-scale RL framework for dLLMs [^182^] [^164^].
- **LLaDA2.0-Uni** (Bie et al., 2026) extended the LLaDA2.0 framework to multimodal understanding and generation, using SigLIP-VQ visual tokens and a unified diffusion architecture for both text and image generation [^66^].
- **LLaDA-MoE-7B** (September 2025) was the first diffusion language model pretrained from scratch with MoE architecture, using only ~1B active parameters while surpassing LLaDA 1.5 (8B dense) [^169^].
- **UltraLLaDA** (He et al., October 2025) demonstrated context length extension to **128K tokens** for diffusion LLMs via modified RoPE extension and specialized masking strategies [^118^].

#### 2. The Dream Family — Code-Focused Diffusion Innovations

- **Dream-7B** (Ye et al., April 2025): A 7B-parameter diffusion LLM initialized from Qwen2.5-7B, using adaptive decoding with context-adaptive token-level noise rescheduling. Uses AR-based LLM initialization with attention mask annealing. Available as `Dream-org/Dream-v0-Instruct-7B` [^29^] [^40^].
- **Dream-Coder-7B** (Xie et al., July 2025): The first fully open-source diffusion LLM for code with complete transparency (data processing, training recipes, model weights). Key innovations include adaptive decoding styles (sketch-first for complex algorithms, left-to-right for completions, interleaved reasoning for understanding). Post-trained with SFT using random truncation and padding penalty, then RL with verifiable rewards. Achieves **21.4% pass@1 on LiveCodeBench** (2410-2505), competitive with proprietary models [^104^] [^40^]. GitHub: **98 stars**, 6 forks. License: **Apache 2.0** [^171^].
- **DreamOn** (Wu et al., February 2026): Solves the fixed-length canvas problem for diffusion LMs by enabling dynamic expansion/contraction of mask tokens during generation. Uses two special tokens `[expand]` and `[delete]` with no architectural changes. Built on Dream-Coder-7B and DiffuCoder-7B. Achieves 26.4% average absolute performance boost over diffusion baselines on infilling tasks, matching oracle-level performance [^76^]. GitHub: **114 stars**, 10 forks. License: **Apache 2.0** [^172^].

#### 3. DiffuCoder — Apple's Coupled-GRPO Innovation

- DiffuCoder-7B (Gong et al., Apple + HKU, June 2025) is trained on **130B tokens of code** with a complete pipeline: adaptation pre-training, mid-training, instruction tuning, and coupled-GRPO reinforcement learning.
- **Coupled-GRPO** is a novel sampling scheme that constructs complementary mask noise for completions used in RL training, reducing variance of token log-likelihood estimates. Each token is sampled exactly the same number of times across complementary masks [^45^].
- Key finding: increasing sampling temperature in diffusion LMs diversifies not only token choices but also generation order, creating a rich search space for RL rollouts.
- Performance: +4.4% improvement on EvalPlus with coupled-GRPO. Reduced reliance on AR causal bias during decoding [^45^].
- GitHub: **821 stars**, 56 forks. Available from Apple at `apple/ml-diffucoder` [^170^].

#### 4. SEDD — ICML 2024 Best Paper

- **SEDD (Score Entropy Discrete Diffusion)** by Lou et al. (2023/2024) introduced **score entropy**, a novel loss that naturally extends score matching to discrete spaces.
- For comparable model sizes, SEDD beats existing language diffusion paradigms, reducing perplexity by 25-75% over prior diffusion models (e.g., D3PM). On One Billion Words, SEDD Absorb achieves perplexity of 32.79 vs. D3PM Absorb's 77.50 [^129^].
- **Key milestone**: First diffusion language model to **outperform GPT-2** on generative perplexity (around 6-8x better generative perplexity than un-annealed GPT-2). Awarded **ICML 2024 Best Paper** [^1^] [^124^].
- Enables controllable infilling (matching nucleus sampling quality while enabling strategies beyond left-to-right prompting) and can trade compute for quality (similar quality with 32x fewer network evaluations) [^131^].

#### 5. MDLM — The Foundational Masked Diffusion Framework

- **MDLM (Masked Diffusion Language Model)** by Sahoo et al. (2024) simplified discrete diffusion to a masking-based framework with a cosine schedule, using standard Transformer backbones with bidirectional attention [^1^] [^30^].
- Training: For each example, sample timestep t ~ U(0,1), mask tokens independently with probability gamma(t) = 1 - cos^2(pi*t/2), compute cross-entropy loss only at masked positions [^43^].
- Generation: Starting from fully masked sequence, iteratively unmask tokens over S=100 steps, selecting the most confident predictions at each step [^43^].
- MDLM forms the architectural foundation that LLaDA, DiffuCoder, Dream, and many other dLLMs build upon. The ELBO-based objective from MDLM has been widely adopted by subsequent large-scale dLLMs [^33^].
- Follow-up work established scaling laws for MDLM under matched training FLOPs, showing predictable relationships between model size, data, and performance comparable to AR models [^44^].

#### 6. Mercury Coder (Inception Labs) — First Commercial-Scale Diffusion LLM

- Mercury is the **world's first commercial-scale diffusion LLM**, launched February 2025 by Inception Labs (founded by researchers who pioneered diffusion models for images and co-invented DPO, Flash Attention, and Decision Transformers) [^71^].
- **Mercury Coder Mini**: 1,109 tok/s, 88.0 HumanEval, 17.0 LiveCodeBench. **Mercury Coder Small**: 737 tok/s, 90.0 HumanEval, 25.0 LiveCodeBench [^71^].
- Pricing: **$0.25/1M input tokens, $0.75/1M output tokens** — significantly cheaper than speed-optimized AR models. Context window: 128K tokens. Max output: 32K tokens [^175^] [^176^].
- Available via OpenAI-compatible API. Free tier: 10M tokens to start. Key advantage: 5-10x faster than frontier speed-optimized LLMs, running at 1000+ tokens/sec on NVIDIA H100s [^71^] [^176^].
- **Closed-source** — weights not available. Enterprise clients get API and on-premise deployments [^71^].

#### 7. Other Notable Diffusion LLMs

- **DiffuLLaMA/DiffuGPT** (Gong et al., ICLR 2025): Converted GPT-2 (127M, 355M) and LLaMA2 (7B) from AR to diffusion using less than 200B tokens. Key technique: attention mask annealing (gradually transitioning from causal to bidirectional attention). DiffuLLaMA is competitive with AR baselines on language modeling, reasoning, and infilling [^105^] [^111^]. GitHub: part of HKUNLP/DiffuLLaMA.
- **TESS 2** (Tae et al., 2025): Generalist instruction-following diffusion LM adapted from Mistral-7B via continued pretraining (~45B tokens). Introduced reward guidance for inference-time preference alignment without additional training. Outperforms contemporary instruction-tuned diffusion models [^120^] [^130^].
- **Plaid** (Gulrajani & Hashimoto, December 2023): 1B-parameter continuous diffusion language model. Analyzed scaling laws for continuous diffusion LMs. Cited as a milestone that accelerated the field [^109^] [^110^].
- **MMaDA** (Yang et al., May 2025): Multimodal diffusion foundation model with unified architecture for textual reasoning, multimodal understanding, and text-to-image generation. Proposes UniGRPO for RL across both reasoning and generation. Open-sourced at `Gen-Verse/MMaDA-8B-MixCoT` [^164^] [^166^].
- **Seed Diffusion** (ByteDance, August 2025): Discrete diffusion code model achieving **2,146 tokens/s** on H20 GPUs, faster than Mercury and Gemini. Commercial preview available [^11^] [^131^].
- **Stable-DiffCoder** (ByteDance Seed, January 2026): Open-source code diffusion LLM family built on Seed-Coder architecture with block diffusion continual pretraining. Outperforms AR counterpart under identical architecture and data [^160^] [^20^].
- **Gemini Diffusion** (Google DeepMind, 2025): Commercial diffusion model with 5x speed improvements over AR baselines [^1^]. Available via Google AI Studio.

#### 8. Training Paradigms: From Scratch vs. AR Conversion

| Approach | Representative Models | Cost | Tradeoffs |
|----------|----------------------|------|-----------|
| Train from scratch | LLaDA-8B, LLaDA-MoE-7B | ~2.3T tokens, ~130K GPU hours | Full optimization for diffusion; highest quality potential; very expensive |
| AR-to-diffusion conversion | DiffuLLaMA, Dream-7B, TESS 2 | <200B tokens | Order of magnitude cheaper; may retain AR biases; needs attention mask annealing |
| Staged conversion (block diffusion) | LLaDA 2.0, LLaDA 2.1 | Three-phase: warmup->stable->decay | Best scaling path; inherits AR knowledge; MoE-compatible |

Key insight from LLaDA 2.0: converting a pretrained AR model to diffusion via block-level WSD training is "a more data-efficient alternative" that preserves linguistic knowledge while changing the generation mechanism [^66^] [^31^]. LLaDA 2.0-flash achieves 535 tokens/s with CAP (Confidence-Aware Parallel) decoding [^106^].

#### 9. Community Adoption Metrics

| Model/Repo | GitHub Stars | Forks | License | HuggingFace |
|-----------|-------------|-------|---------|-------------|
| ML-GSAI/LLaDA | 3.8k | 264 | MIT | GSAI-ML/LLaDA-8B-* |
| apple/ml-diffucoder | 821 | 56 | Apple license | Available |
| inclusionAI/LLaDA2.X | 419 | 22 | Apache 2.0 | inclusionAI/llada-20,21 |
| DreamLM/DreamOn | 114 | 10 | Apache 2.0 | Dream-org/DreamOn-v0-7B |
| DreamLM/Dream-Coder | 98 | 6 | Apache 2.0 | Dream-org/dream-coder-7b |
| HKUNLP/DiffuLLaMA | ~500+ (estimated) | N/A | N/A | Available |
| ByteDance-Seed/Stable-DiffCoder | Growing | N/A | N/A | ByteDance-Seed/ |
| Gen-Verse/MMaDA | N/A | N/A | N/A | Gen-Verse/MMaDA-8B-MixCoT |

#### 10. Licensing and Availability Summary

- **LLaDA family**: MIT (LLaDA 8B), Apache 2.0 (LLaDA 2.0/2.1). All weights fully open on HuggingFace.
- **Dream family**: Apache 2.0 (inherits from Qwen2.5-7B's Apache 2.0 license). Checkpoints, training recipes, preprocessing pipelines, and inference code fully released [^104^].
- **DiffuCoder**: Released by Apple under Apple's open-source license. GitHub repo includes recipes, inference demo, training code [^170^].
- **SEDD**: Open-source code at `github.com/louaaron/Score-Entropy-Discrete-Diffusion` [^129^].
- **MDLM**: Open-source implementation available.
- **TESS 2**: Open-source, code and models on GitHub [^130^].
- **MMaDA**: Open-source code and trained models at `github.com/Gen-Verse/MMaDA` [^166^].
- **Mercury**: **Closed-source commercial**. API access only. Pricing: $0.25/1M input, $0.75/1M output [^175^].
- **Seed Diffusion/Stable-DiffCoder**: Open-source from ByteDance [^20^].
- **Gemini Diffusion**: Closed commercial from Google DeepMind.

### Major Players & Sources

| Entity | Role/Relevance |
|--------|---------------|
| **GSAI-ML / ML-GSAI** (Renmin University + Ant Group) | Creators of LLaDA family. Most prolific open-source diffusion LLM research group. Responsible for LLaDA 8B, 1.5, MoE, and LLaDA 2.0/2.1 via InclusionAI (Ant Group) |
| **HKU NLP Group** (University of Hong Kong) | Creators of Dream, Dream-Coder, DreamOn, DiffuLLaMA, DiffuCoder. Core hub for diffusion LLM research with 10+ papers in the space |
| **Apple (ml-diffucoder)** | Released DiffuCoder with coupled-GRPO innovation. Significant industry contribution to open-source diffusion code models |
| **Inception Labs** | Commercial pioneer with Mercury Coder. First to market with commercial-scale diffusion LLM. Founded by diffusion model pioneers |
| **ByteDance Seed** | Released Seed Diffusion (fastest inference at 2,146 tok/s) and Stable-DiffCoder (open-source). Major commercial player |
| **Google DeepMind** | Gemini Diffusion — commercial diffusion LLM with 5x speed improvements |
| **Aaron Lou (Stanford)** | SEDD author, ICML 2024 Best Paper. Core theoretical contributions to discrete diffusion |
| **Sahoo et al. (MIT/Various)** | MDLM authors. Established foundational framework for masked diffusion language models |
| **Allen Institute for AI + Yale** | TESS 2 authors. Generalist instruction-following diffusion model with reward guidance |
| **Gen-Verse team (PKU/Princeton/ByteDance)** | MMaDA creators. Unified multimodal diffusion foundation model |

### Trends & Signals

1. **Rapid scaling**: From SEDD (GPT-2 scale, 2023) to LLaDA-8B (Feb 2025) to LLaDA 2.0-100B (Nov 2025) in just 2 years. The field is scaling at a pace comparable to early AR LLM development [^31^] [^74^].

2. **Block diffusion as the deployment paradigm**: Both LLaDA 2.0 and Stable-DiffCoder adopted block diffusion (generating fixed-size blocks autoregressively while denoising within blocks) as the practical serving architecture, bridging parallel generation with efficient inference [^106^] [^160^].

3. **AR-to-diffusion conversion winning for practical deployment**: Training from scratch (LLaDA 8B) proved viability, but conversion (LLaDA 2.0, DiffuLLaMA, Dream) is becoming the dominant paradigm for production due to 10x+ lower training costs [^66^] [^31^].

4. **Code as the killer application**: Multiple code-focused diffusion models emerged (DiffuCoder, Dream-Coder, Mercury Coder, Seed Diffusion, Stable-DiffCoder), with code infilling and any-order generation being natural advantages of the diffusion paradigm [^45^] [^104^] [^11^].

5. **RL for diffusion is maturing**: From VRPO (LLaDA 1.5) to coupled-GRPO (DiffuCoder) to UniGRPO (MMaDA) — a full suite of diffusion-native RL algorithms is emerging for post-training alignment [^68^] [^45^] [^164^].

6. **Inference speed as the primary commercial advantage**: Mercury (1000+ tok/s), Seed Diffusion (2146 tok/s), and LLaDA 2.1 (892 TPS on code) all emphasize speed as the key differentiator vs. AR models [^71^] [^11^] [^182^].

7. **MoE architecture successfully adapted**: LLaDA 2.0 demonstrated that MoE scaling works for diffusion LMs, with 100B total / 6.1B active parameters, achieving 2.1x inference acceleration over comparable AR models [^106^] [^74^].

8. **Multimodal expansion**: LLaDA2.0-Uni and MMaDA extended diffusion LMs to text-to-image generation and multimodal understanding, showing the paradigm's flexibility beyond text [^66^] [^164^].

### Controversies & Conflicting Claims

1. **Diffusion vs. AR: Which is better?** A controlled comparison (same data, same compute, same hardware) found AR converges faster but MDLM generates more diverse outputs (93.4% unique 5-word openings vs. 0.2% for AR), with comparable training throughput. AR produces more fluent but more repetitive text [^1^]. However, at scale, diffusion models still lag on some benchmarks — e.g., on LiveCodeBench v6, diffusion LLMs average 14.9% vs. AR's 18.9% [^9^].

2. **Training from scratch vs. conversion**: LLaDA 8B showed competitive performance training from scratch, but LLaDA 2.0 found conversion more practical. Dream-7B (converted from Qwen2.5) outperforms LLaDA-8B on many code tasks despite being smaller, suggesting conversion may have advantages [^9^]. However, some argue that training from scratch avoids "AR bias" that limits diffusion's full potential [^34^].

3. **Fixed-length generation limitation**: A fundamental criticism of diffusion LMs is the requirement to pre-specify output length. DreamOn (Feb 2026) addressed this with dynamic expansion/contraction, but some argue this adds complexity that AR models don't have [^76^] [^119^].

4. **Inference speed claims**: While Mercury claims 1000+ tok/s and Seed Diffusion claims 2146 tok/s, direct comparison is difficult due to different hardware (H100 vs. H20), evaluation conditions, and the impact of system prompts on benchmark results [^11^] [^71^].

5. **Open vs. closed source performance gap**: Open-source diffusion LLMs significantly lag closed-source counterparts. Mercury-Coder-Small achieves 86.0% on HumanEval vs. the best open-source diffusion model (Dream-Coder at 76.2%) [^9^]. This gap is attributed to higher-quality training data in commercial settings.

### Recommended Deep-Dive Areas

1. **Coupled-GRPO and RL for diffusion LMs**: The coupled-GRPO technique from DiffuCoder represents a fundamentally new way to do RL with diffusion models. Understanding how complementary mask sampling reduces variance could unlock significant performance gains. The broader space of diffusion-native RL (VRPO, UniGRPO, SPG) warrants systematic comparison.

2. **Block diffusion decoding strategies**: Block diffusion (used in LLaDA 2.0, Stable-DiffCoder, and DSB) is emerging as the serving architecture of choice, but optimal block size scheduling, dynamic adjustment, and interaction with KV-cache mechanisms need deeper study.

3. **AR-to-diffusion conversion recipes**: The gap between converted and from-scratch models is narrowing. Understanding optimal conversion strategies (attention mask annealing schedules, WSD block size progression, knowledge preservation) could democratize diffusion LM development.

4. **Inference acceleration systems**: Speed is the primary commercial advantage. LEAP, SPRINT, DPad, Sparse-dLLM, and other acceleration techniques represent a rich systems research area orthogonal to model design.

5. **Diffusion LMs for code: any-order generation**: The emergent property of diffusion code models to adaptively choose generation order (sketch-first, left-to-right, interleaved) is unique and may be exploitable for novel programming tools beyond what AR models can offer.

6. **Safety and alignment of diffusion LMs**: DiffuGuard found that diffusion LMs have "intrinsic safety" properties but also unique failure modes when attacked with context-based jailbreaks. The safety profile of diffusion LMs differs from AR and needs specialized study.

### Reference URLs

- LLaDA GitHub: https://github.com/ML-GSAI/LLaDA (3.8k stars, MIT license)
- LLaDA 2.X GitHub: https://github.com/inclusionAI/LLaDA2.X (419 stars, Apache 2.0)
- DiffuCoder GitHub: https://github.com/apple/ml-diffucoder (821 stars)
- Dream-Coder GitHub: https://github.com/DreamLM/Dream-Coder (98 stars, Apache 2.0)
- DreamOn GitHub: https://github.com/DreamLM/DreamOn (114 stars, Apache 2.0)
- Stable-DiffCoder GitHub: https://github.com/ByteDance-Seed/Stable-DiffCoder
- DiffuLLaMA GitHub: https://github.com/HKUNLP/DiffuLLaMA
- SEDD: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
- Mercury/Inception Labs: https://www.inceptionlabs.ai/blog/introducing-mercury
- Seed Diffusion: https://seed.bytedance.com/seed_diffusion
- MMaDA: https://github.com/Gen-Verse/MMaDA
- TESS 2: https://github.com/hamishivi/tess-2
- LLaDA-8B-Instruct on HuggingFace: https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct
- Dream-Coder on HuggingFace: https://huggingface.co/collections/Dream-org/dream-coder-7b
