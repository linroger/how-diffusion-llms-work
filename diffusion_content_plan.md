# Content Plan: Diffusion Models for Text and Code Generation
## Research Report — Google DeepMind & Ant Group Focus

---

## Chapter 1: Introduction — Why Diffusion Models for Language

**Technical depth level:** Accessible to technical readers with ML familiarity; no diffusion background assumed.

### Content Points
- Contrast autoregressive (left-to-right, causal attention) vs. diffusion (iterative denoising, bidirectional attention) paradigms at a conceptual level.
- Explain the core value proposition: parallel token generation enables higher throughput; bidirectional context enables global planning; iterative refinement enables error correction.
- Frame the report's scope: text and code generation, with deep focus on Ant Group (LLaDA family) and Google DeepMind (Gemini Diffusion).
- Preview the key question: Can diffusion models displace autoregressive models as the dominant paradigm for language generation?

### Required Tables
**Table 1.1 — Paradigm Comparison: AR vs. Diffusion for Language**
| Dimension | Autoregressive (AR) | Diffusion (Masked) |
|-----------|---------------------|---------------------|
| Attention mechanism | Causal (left-to-right) | Bidirectional (full context) |
| Generation process | Sequential token-by-token | Iterative denoising from masked state |
| Training objective | Next-token prediction | Masked token prediction (ELBO) |
| Latency characteristic | Time to first token fast; total time O(n) | Time to first token slower; parallelizable |
| Error behavior | Irreversible (error accumulation) | Correctable (iterative refinement) |
| Log-likelihood computation | Exact (chain rule) | Approximate (ELBO estimation) |

### Required Charts
- **Figure 1.1 — Schematic diagram:** Side-by-side illustration of AR generation (tokens flowing left to right) vs. diffusion generation (all tokens masked initially, progressively unmasked). No data visualization needed; conceptual diagram.
- **Figure 1.2 — Speed-quality trade-off scatter:** Placeholder for report-wide comparison showing diffusion models (Gemini Diffusion, Seed Diffusion, LLaDA2.1, Mercury) vs. AR models on throughput (x-axis) vs. benchmark score (y-axis).

### Key Quotes
> "We contend that the intelligence of LLMs — manifested in scalability, instruction-following, in-context learning, conversational ability, and compression — stems not from the autoregressive mechanism per se, but rather from the core principle of generative modeling: approximating the true language distribution through maximum likelihood estimation." — LLaDA authors (Nie et al., 2025)

> "What is now proved was once only imagined." — William Blake (epigraph from LLaDA technical report)

### Case Studies / Examples
- **Case: GPT-4 class model making an irreversible error** — Show how AR models cannot recover from an early incorrect token in a code generation task; contrast with diffusion model's ability to revise in later iterations.
- **Example: Gemini Diffusion's iterative refinement** — Concrete walkthrough of a prompt where the model initially generates an approximate response then refines specific tokens across steps.

---

## Chapter 2: Technical Foundations — How Diffusion Language Models Work

**Technical depth level:** Deep technical; intended for ML engineers and researchers. Covers mathematical foundations and architectural specifics.

### Content Points
- **Masked Diffusion formulation:** Forward process (masking tokens), reverse process (predicting masked tokens), ELBO objective.
- **Training pipeline:** Pre-training with random masking → SFT (mask only response tokens) → optional RL alignment.
- **Inference algorithms:** Block parallel decoding, confidence-based thresholding, remasking strategies (low-confidence vs. random).
- **Key architectural detail:** Bidirectional Transformer encoder (not decoder); sinusoidal timestep conditioning; no causal masking.
- **Controlled comparison:** Table from Austin et al. / MDLM showing identical settings where only paradigm differs (attention mask, objective, noise schedule).

### Required Tables
**Table 2.1 — Controlled Experimental Setup: AR vs. MDLM (from Austin et al.)**
| Variable | AR | MDLM |
|----------|-----|------|
| Data | TinyStories, 50M tokens | TinyStories, 50M tokens |
| Steps | 20,000 | 20,000 |
| Batch size | 32 | 32 |
| Sequence length | 512 | 512 |
| d_model | 768 | 768 |
| Layers | 12 | 12 |
| Attention mask | Causal | Bidirectional |
| Training objective | Next-token prediction | Masked token prediction |
| Noise schedule | N/A | Cosine |
| Timestep conditioning | N/A | Sinusoidal embedding |
| Parameters | 123.6M | 162.7M |

**Table 2.2 — Diffusion LM Inference Hyperparameters**
| Hyperparameter | Description | Typical Range | Impact |
|----------------|-------------|---------------|--------|
| Denoising threshold | Confidence threshold for unmasking | 0.85–0.95 | Higher = better quality, slower |
| Block size | Number of tokens processed per block | 16–64 | Trade-off between parallelism and quality |
| Remasking strategy | Which tokens to re-mask if confidence low | low-confidence / random | Affects convergence and quality |
| Number of steps | Total denoising iterations | 10–128 | More steps = higher quality |

### Required Charts
- **Figure 2.1 — Forward and reverse process diagram:** Visual showing sequence going from fully masked (t=1) to fully unmasked (t=0) across timesteps.
- **Figure 2.2 — ELBO estimation variance:** Plot showing variance of ELBO estimator vs. Monte Carlo sample budget (from LLaDA 1.5 paper), demonstrating why VRPO is necessary.

### Key Quotes
> "The key architectural difference from AR models is that diffusion models use bidirectional attention (each position can attend to all unmasked positions) and condition on a noise level or timestep. This bidirectional context is both a strength (richer conditioning) and a difference that complicates direct comparison."

### Case Studies / Examples
- **Worked example:** Step-by-step walkthrough of LLaDA inference on a short prompt ("Explain what artificial intelligence is."), showing the masking/unmasking process across 4-5 timesteps.
- **Code example:** Pseudocode for block parallel decoding algorithm.

---

## Chapter 3: The LLaDA Family — Ant Group's Diffusion Ecosystem

**Technical depth level:** Deep technical with architectural specifics; tracks evolution across model versions.

### Content Points
- **LLaDA (8B):** First large-scale masked diffusion LM trained from scratch. Bidirectional Transformer, 8B parameters. Competitive with LLaMA3 8B on downstream tasks. Joint work by Renmin University and Ant Group.
- **LLaDA 1.5 (8B):** Applied VRPO for RL alignment. Improvements: GSM8K +4.7, HumanEval +3.0, MBPP +1.8, IFEval +4.0, Arena-Hard +4.3. Key innovation: addressing high-variance ELBO estimates in preference optimization.
- **LLaDA-MoE (7B total / 1.4B active):** First diffusion LM with native MoE architecture. 7B total parameters, 1.4B active per step. Pre-trained on 20T tokens across two stages with annealing. Inference via dInfer framework.
- **LLaDA 2.0 (16B mini, 100B flash):** Industry-first 100B-parameter diffusion LM. WSD pre-training strategy enabling knowledge inheritance from AR models. CAP (Confidence-Aware Parallel) training. Block parallel decoding with KV cache reuse. 535 tok/s base speed (2.1x AR models).
- **LLaDA 2.1 (16B mini, 100B flash):** Introduces token editing — models can draft AND correct. Dual-threshold decoding (M2T + T2T). Two modes: Speedy (S) and Quality (Q). Peak 1,587 TPS (mini) / 892 TPS (flash). Multi-Block Editing (MBE) for cross-block refinement. First large-scale RL framework for diffusion LMs using ELBO-based Block-level Policy Optimization.

### Required Tables
**Table 3.1 — LLaDA Family Evolution**
| Model | Release | Parameters | Architecture | Key Innovation | Organization |
|-------|---------|------------|--------------|----------------|--------------|
| LLaDA (8B) | Early 2025 | 8B dense | Bidirectional Transformer | First large-scale diffusion LM from scratch | Renmin U + Ant Group |
| LLaDA 1.5 | May 2025 | 8B dense | + VRPO alignment | Variance-reduced preference optimization | Renmin U + Ant Group |
| LLaDA-MoE | Sep 2025 | 7B / 1.4B active | MoE diffusion | First native MoE diffusion LM; dInfer framework | Ant Group |
| LLaDA 2.0 mini | Dec 2025 | 16B (1.44B active) | MoE + WSD pretraining | 100B scale; CAP training; block parallel decoding | Ant Group (InclusionAI) |
| LLaDA 2.0 flash | Dec 2025 | 100B MoE | MoE + WSD + CAP | Commercial-grade code generation | Ant Group (InclusionAI) |
| LLaDA 2.1 | Feb 2026 | 16B/100B MoE | Token editing (M2T+T2T) | Editable decoding; 892-1587 TPS | Ant Group (InclusionAI) |

**Table 3.2 — LLaDA 2.0/2.1 Code Benchmark Results**
| Benchmark | LLaDA2.0-mini | LLaDA2.0-flash | LLaDA2.1-flash (Q) | LLaDA2.1-mini (S) |
|-----------|---------------|----------------|---------------------|-------------------|
| HumanEval | 86.59 | 94.51 | — | — |
| HumanEval+ | 79.88 | 87.80 | 89.63 | — |
| MBPP | 81.50 | 88.29 | — | — |
| MBPP+ | 74.07 | 79.63 | 77.25 | — |
| BigCodeBench-Full | 32.89 | 41.58 | 39.21 | — |
| LiveCodeBench | 31.50 | 42.29 | 45.37 | — |
| CRUXEval-O | 71.62 | 85.12 | 87.50 | — |
| MultiPL-E | 67.46 | 74.87 | 73.34 | — |

**Table 3.3 — LLaDA 2.1 Speed vs. Quality Trade-off (S Mode vs. Q Mode)**
| Model | Mode | HumanEval+ TPS | TPF | Score trend |
|-------|------|----------------|-----|-------------|
| LLaDA2.1-flash | S (speed) | 891.74 | ~5.9 | Slight drop |
| LLaDA2.1-flash | Q (quality) | Lower | ~3.1 | Matches/exceeds 2.0 |
| LLaDA2.1-mini | S (speed) | 1,586.93 | Higher | Slight drop |
| LLaDA2.1-mini | Q (quality) | Lower | Lower | Exceeds 2.0 |

### Required Charts
- **Figure 3.1 — LLaDA family timeline:** Horizontal timeline from early 2025 to early 2026 showing model releases, key technical innovations, and capability milestones.
- **Figure 3.2 — Benchmark comparison bar chart:** LLaDA 2.0-flash vs. Qwen3-30B-A3B vs. Ling-flash-2.0 across HumanEval, MBPP, LiveCodeBench, BigCodeBench.
- **Figure 3.3 — Speed comparison (TPS):** LLaDA2.1 variants vs. competitors on HumanEval+ showing the 3.5x speed advantage over AR models.
- **Figure 3.4 — Score vs. TPF trade-off:** Scatter plot showing S-mode vs. Q-mode positions for LLaDA2.1-flash and mini.

### Key Quotes
> "LLaDA 2.0 realizes the seamless inheritance of autoregressive model knowledge through a brand-new Warmup-Stable-Decay (WSD) pre-training strategy, avoiding the high cost of training from scratch." — Ant Group technical report

> "We believe that dInfer provides both a practical toolkit and a standardised platform to accelerate research and development in the rapidly growing field of dLLMs." — Ant researchers

> "In Q Mode, LLaDA2.1 surpasses the results of LLaDA2.0 on both mini and flash model." — LLaDA2.1 paper

### Case Studies / Examples
- **Case: dInfer vs. NVIDIA Fast-dLLM vs. vLLM** — Comparative benchmark showing dInfer achieving 1,011 tok/s on HumanEval vs. 91 tok/s (Fast-dLLM) and 294 tok/s (vLLM + Qwen-2.5-3B). Demonstrates 10x speedup over NVIDIA's framework.
- **Case: LLaDA-MoE efficiency** — 7B total / 1.4B active parameters delivering performance of a 3B dense model with significantly fewer FLOPs per step.
- **Worked example:** LLaDA 2.1 token editing mechanism showing how a code draft with errors gets corrected across iterations.

---

## Chapter 4: Gemini Diffusion — Google DeepMind's Approach

**Technical depth level:** Moderate-to-deep technical; limited by Google's partial disclosure of architectural details.

### Content Points
- Positioning as "state-of-the-art experimental text diffusion model" from Google DeepMind.
- Core claims: (1) significantly faster than fastest Gemini model so far; (2) more coherent due to block-wise generation; (3) iterative error correction.
- Benchmark results vs. Gemini 2.0 Flash-Lite: competitive on code benchmarks, trailing on reasoning/science/multilingual.
- Sampling speed: 1,479 tok/s (excluding 0.84s overhead).
- Focus on text generation (not specifically code-optimized), but includes code benchmarks.
- Available as experimental demo for research and development.

### Required Tables
**Table 4.1 — Gemini Diffusion vs. Gemini 2.0 Flash-Lite**
| Benchmark | Category | Gemini Diffusion | Gemini 2.0 Flash-Lite |
|-----------|----------|-----------------|----------------------|
| HumanEval | Code | 89.6% | 90.2% |
| MBPP | Code | 76.0% | 75.8% |
| LiveCodeBench (v6) | Code | 30.9% | 28.5% |
| BigCodeBench | Code | 45.4% | 45.8% |
| SWE-Bench Verified | Code | 22.9% | 28.5% |
| GPQA Diamond | Science | 40.4% | 56.5% |
| AIME 2025 | Math | 23.3% | 20.0% |
| BIG-Bench Extra Hard | Reasoning | 15.0% | 21.0% |
| Global MMLU (Lite) | Multilingual | 69.1% | 79.0% |
| **Sampling speed** | **Speed** | **1,479 tok/s** | Slower |

**Table 4.2 — Code Benchmark Comparison Across Diffusion Models**
| Model | HumanEval | MBPP | LiveCodeBench | BigCodeBench |
|-------|-----------|------|---------------|--------------|
| Gemini Diffusion | 89.6% | 76.0% | 30.9% | 45.4% |
| LLaDA2.0-flash | 94.51% | 88.29% | 42.29% | 41.58% |
| Seed Diffusion Preview | 82.8%* | — | — | — |
| Mercury Coder | 86.0%* | — | — | — |
| DiffuCoder-7B-cpGRPO | 69.5% | 64.2% | 8.4% | — |

### Required Charts
- **Figure 4.1 — Gemini Diffusion benchmark radar chart:** Radar/spider chart comparing Gemini Diffusion vs. Gemini 2.0 Flash-Lite across all reported benchmarks, visually highlighting the code-reasoning trade-off.
- **Figure 4.2 — Speed comparison bar chart:** Gemini Diffusion (1,479 tok/s) vs. Seed Diffusion (2,146 tok/s) vs. LLaDA2.1 (1,587 TPS) vs. Mercury (1,109 tok/s).

### Key Quotes
> "Large-language models are the foundation of generative AI today. We're using a technique called diffusion to explore a new kind of language model that gives users greater control, creativity, and speed in text generation." — Google DeepMind, Gemini Diffusion page

> "Generates entire blocks of tokens at once, meaning it responds more coherently to a user's prompt than autoregressive models." — Google DeepMind

> "Corrects errors during generation for more consistent outputs." — Google DeepMind

### Case Studies / Examples
- **Case: Code generation on LiveCodeBench** — Gemini Diffusion achieves 30.9% vs. Flash-Lite's 28.5%, showing diffusion advantage on this harder benchmark despite trailing on simpler HumanEval.
- **Case: SWE-Bench Verified gap** — Analysis of why Gemini Diffusion trails Flash-Lite (22.9% vs. 28.5%) on real-world software engineering tasks, exploring the hypothesis that iterative refinement may struggle with long-horizon planning tasks.

---

## Chapter 5: Commercial Ecosystem — Mercury, Seed Diffusion, and Competitive Landscape

**Technical depth level:** Moderate technical; emphasizes commercial positioning, pricing, and deployment characteristics.

### Content Points
- **Mercury Coder (Inception, March 2025):** Commercial diffusion code model. 128K context window, 32K max output. Pricing: $0.25/M input, $0.75/M output. Achieves up to 10x throughput vs. speed-optimized AR models. Speed claim: 1,109 tok/s.
- **Seed Diffusion Preview (ByteDance, July 2025):** Discrete-state diffusion for code. Two-stage training: mask-based corruption + edit-based augmentation. 2,146 tok/s on H20 GPUs — fastest reported among diffusion code models. Strong on CanItEdit (54.3%) indicating good code editing capability.
- **Stable-DiffCoder (ByteDance, January 2026):** Open-source code diffusion model (8B). Block diffusion continual pretraining with warmup and block-wise clipped noise schedule. Surpasses AR counterpart on broad code benchmarks.
- **DiffuCoder (Apple + HKU, 2025):** 7B open-source model with coupled-GRPO for code generation. Key insight: generation order becomes more flexible at higher temperatures. Coupled-sampling scheme for log-likelihood estimation.
- Position each player on the speed-quality-cost landscape.

### Required Tables
**Table 5.1 — Commercial Diffusion Models Comparison**
| Attribute | Mercury Coder | Seed Diffusion Preview | Gemini Diffusion | LLaDA2.1 |
|-----------|---------------|----------------------|------------------|----------|
| Organization | Inception | ByteDance Seed | Google DeepMind | Ant Group |
| Release date | Mar 2025 | Jul 2025 | 2025 | Feb 2026 |
| Model size | Unknown | ~8B equivalent | Several B (est.) | 16B/100B MoE |
| Context window | 128K | — | — | 8K–32K |
| Max output | 32K | — | — | — |
| Input price ($/M) | $0.25 | Not disclosed | Free demo | Open source |
| Output price ($/M) | $0.75 | Not disclosed | Free demo | Open source |
| Speed (tok/s) | 1,109 | 2,146 | 1,479 | 892–1,587 |
| Hardware | — | H20 GPU | — | H20 (SGLang) |
| Open source | No | No | No | Yes |

**Table 5.2 — Speed Benchmark Comparison (Code Generation)**
| Model | Throughput (tok/s) | Hardware | Notes |
|-------|-------------------|----------|-------|
| Seed Diffusion | 2,146 | H20 GPU | Fastest reported |
| Gemini Diffusion | 1,479 | — | Excluding 0.84s overhead |
| LLaDA2.1-mini | 1,587 | H20 + quantization | S-mode |
| LLaDA2.1-flash | 892 | H20 + quantization | S-mode |
| Mercury Coder | 1,109 | — | 10x vs. AR claim |
| dInfer (LLaDA-MoE) | 1,011 | — | Framework-level benchmark |
| LLaDA2.0-flash-CAP | 500 | H20 (SGLang TP8) | With CAP training |
| Qwen3-30B-A3B (AR) | 240 | H20 (SGLang) | AR baseline |
| Ling-flash-2.0 (AR) | 257 | H20 (SGLang) | AR baseline |

**Table 5.3 — CanItEdit and Aider Benchmark Results**
| Model | Size | Aider (tries=2) | CanItEdit (pass@1) |
|-------|------|-----------------|-------------------|
| Seed-Diffusion-Preview | — | 44.4 | 54.3 |
| Seed-Coder-8B-Instruct | 8B | 57.1 | 50.5 |
| Qwen2.5-Coder-14B-Instruct | 14B | 69.2 | 52.9 |
| Yi-Coder-9B-Chat | 9B | 54.1 | 50.5 |
| Llama-3.1-8B-Instruct | 8B | 33.1 | 39.5 |

### Required Charts
- **Figure 5.1 — Price-performance positioning:** Scatter plot with input price ($/M tokens) on x-axis and HumanEval score on y-axis, showing commercial positioning of Mercury vs. AR alternatives.
- **Figure 5.2 — Speed comparison horizontal bar chart:** All diffusion models ranked by tok/s, with AR baselines (Qwen3, Ling) shown for comparison.
- **Figure 5.3 — Timeline of commercial releases:** Gantt-style timeline showing release dates from March 2025 (Mercury) through February 2026 (LLaDA2.1).

### Key Quotes
> "Mercury Coder achieves state-of-the-art throughput, outperforming speed-optimized autoregressive models by up to 10 times while maintaining comparable quality on major code benchmarks."

> "Seed Diffusion Preview achieves an inference rate of 2,146 tokens per second on H20 GPUs, surpassing current diffusion benchmarks while either matching or exceeding their accuracy on established code evaluation metrics."

### Case Studies / Examples
- **Case: Mercury API pricing analysis** — Cost comparison for a typical code generation workload (e.g., 1M input + 200K output tokens) between Mercury, GPT-4o, and Claude 3.5 Sonnet.
- **Case: ByteDance Seed Diffusion two-stage training** — Detailed explanation of mask-based corruption followed by edit-based augmentation, and why this avoids carry-over unmasking shortcuts.

---

## Chapter 6: Reinforcement Learning for Diffusion Models — VRPO, Coupled-GRPO, and Beyond

**Technical depth level:** Deep technical; focuses on RL algorithm design for non-autoregressive models.

### Content Points
- **Core challenge:** Diffusion models cannot compute exact log-likelihoods; rely on ELBO approximations which are high-variance. This makes standard RLHF/DPO unstable.
- **VRPO (Variance-Reduced Preference Optimization):** Three techniques: (1) increased sampling budget for ELBOs; (2) optimal allocation (many timesteps, one mask per timestep); (3) antithetic sampling (share random noise between policy and reference). Applied to LLaDA 1.5. Results: GSM8K +4.7, HumanEval +3.0, MBPP +1.8, IFEval +4.0, Arena-Hard +4.3.
- **Coupled-GRPO (DiffuCoder):** Coupled-sampling scheme with complementary mask noise. Two masks generated such that every position is masked in exactly one mask. Log-probability estimate derived by averaging losses from complementary forward passes. Provides full token coverage and more stable gradients. Trained on 21K code examples.
- **AEGPO (Adaptive Entropy-Guided Policy Optimization):** Uses attention entropy as dual-signal proxy. Global: relative entropy change allocates rollout budgets. Local: entropy peaks identify critical exploration timesteps. More efficient than uniform GRPO sampling.
- **Comparison of approaches:** VRPO extends DPO; coupled-GRPO adapts GRPO for diffusion; AEGPO improves sampling efficiency.

### Required Tables
**Table 6.1 — RL Algorithms for Diffusion Language Models Comparison**
| Algorithm | Base Method | Log-Likelihood Estimation | Key Innovation | Applied To | Target Task |
|-----------|-------------|--------------------------|----------------|------------|-------------|
| VRPO | DPO | ELBO with random masking | Optimal MC budget allocation + antithetic sampling | LLaDA 1.5 | General alignment |
| Coupled-GRPO | GRPO | One-step estimation with complementary masks | Coupled-sampling with complementary mask noise | DiffuCoder | Code generation |
| UniGRPO | GRPO | ELBO with random masking | Structured noising strategy | Various | General |
| AEGPO | GRPO | Standard | Attention entropy-guided adaptive sampling | Visual diffusion | Image generation |
| D1 | GRPO | One-step estimation | Prompt masking as regularizer | LLaDA | Reasoning |

**Table 6.2 — VRPO Ablation Results (from LLaDA 1.5 paper)**
| Technique | GSM8K | HumanEval | MBPP | IFEval | Arena-Hard |
|-----------|-------|-----------|------|--------|------------|
| SFT baseline (LLaDA 8B) | 78.6 | 49.4 | 41.0 | 51.39 | 47.47 |
| Full VRPO | 83.3 (+4.7) | 52.4 (+3.0) | 42.8 (+1.8) | 58.23 (+6.84) | 55.43 (+7.96) |

**Table 6.3 — Coupled-GRPO Results (DiffuCoder)**
| Model | HumanEval | MBPP | LiveCodeBench |
|-------|-----------|------|---------------|
| DiffuCoder-7B (no RL) | — | — | — |
| DiffuCoder-7B-cpGRPO | 69.5% | 64.2% | 8.4% |

### Required Charts
- **Figure 6.1 — RL algorithm taxonomy:** Flowchart showing how each diffusion RL algorithm relates to base AR methods (DPO → VRPO; GRPO → coupled-GRPO, UniGRPO, D1, AEGPO).
- **Figure 6.2 — VRPO variance reduction:** Plot showing gradient variance during training with vs. without VRPO techniques (from LLaDA 1.5 paper).
- **Figure 6.3 — Performance improvement bars:** Before/after comparison for VRPO on key benchmarks.

### Key Quotes
> "The core difficulty is that reinforcement learning or direct preference optimization for a diffusion model requires estimating the model's log-likelihood for given outputs — but diffusion models cannot compute an exact log-probability and instead rely on ELBO approximations. These ELBO-based likelihood estimates are noisy (high-variance), which in turn makes preference-based gradient updates extremely unstable."

> "Coupled-GRPO is designed to be diffusion-native by leveraging the unique properties of the DLM generation process... This ensures that every token is evaluated in a partial-masking context during training, providing full token coverage and a more stable gradient signal."

### Case Studies / Examples
- **Case: VRPO optimal budget allocation** — Concrete example showing why sampling many timesteps with one mask each outperforms few timesteps with many masks (variance decomposition analysis).
- **Case: Coupled-GRPO on 21K code examples** — Shows that coupled-GRPO achieves substantial gains even with limited training data, outperforming standard GRPO with larger datasets.
- **Worked example:** Pseudocode for antithetic sampling in VRPO — sharing the same random seed between y_w and y_l ELBO estimates.

---

## Chapter 7: Benchmark Deep-Dive — Code Generation Performance

**Technical depth level:** Deep analytical; focuses on interpreting benchmark results across model families.

### Content Points
- **HumanEval / HumanEval+:** Classic code generation benchmark. LLaDA2.0-flash achieves 94.51% (HumanEval) and 87.80% (HumanEval+). Gemini Diffusion: 89.6%. Mercury: 86.0%. Diffusion models competitive but not uniformly dominant.
- **MBPP / MBPP+:** More complex problems than HumanEval. LLaDA2.0-flash leads with 88.29% (MBPP) and 79.63% (MBPP+). Shows diffusion strength on mid-complexity tasks.
- **LiveCodeBench:** Harder, competition-level problems. LLaDA2.0-flash: 42.29%; Gemini Diffusion: 30.9%. AR models (Qwen3-30B: 41.63%) competitive. LLaDA2.1 improves to 45.37% (Q-mode).
- **BigCodeBench:** Complex real-world code scenarios. LLaDA2.0-flash: 41.58%; Gemini Diffusion: 45.4%. Gemini Diffusion notably strong here.
- **CanItEdit:** Code editing capability. Seed Diffusion Preview: 54.3% — notably strong, suggesting diffusion models excel at editing tasks due to their iterative refinement nature.
- **Aider:** Real-world coding assistant benchmark. Seed Diffusion: 44.4%.
- **MultiPL-E:** Multi-language code generation. LLaDA2.0-flash: 74.87%.
- **CRUXEval:** Code reasoning and execution understanding. LLaDA2.0-flash: 85.12%.

### Required Tables
**Table 7.1 — Master Benchmark Comparison (Code Generation)**
| Benchmark | Gemini Diffusion | LLaDA2.0-flash | LLaDA2.1-flash (Q) | Seed-Diffusion | Mercury | DiffuCoder-7B |
|-----------|-----------------|----------------|-------------------|----------------|---------|---------------|
| HumanEval | 89.6% | 94.51% | — | 82.8%* | 86.0%* | 69.5% |
| MBPP | 76.0% | 88.29% | — | — | — | 64.2% |
| LiveCodeBench (v6) | 30.9% | 42.29% | 45.37% | — | — | 8.4% |
| BigCodeBench | 45.4% | 41.58% | 39.21% | — | — | — |
| CanItEdit | — | — | — | 54.3% | — | — |
| Aider | — | 66.92% | — | 44.4% | — | — |
| HumanEval+ | — | 87.80% | 89.63% | — | — | — |
| MBPP+ | — | 79.63% | 77.25% | — | — | — |
| MultiPL-E | — | 74.87% | 73.34% | — | — | — |
| CRUXEval-O | — | 85.12% | 87.50% | — | — | — |

**Table 7.2 — Diffusion vs. Autoregressive on Code Benchmarks (Average)**
| Category | Diffusion Average | AR Average | Delta |
|----------|-------------------|------------|-------|
| HumanEval family | ~86% | ~82% | +4% diffusion |
| MBPP family | ~77% | ~78% | -1% (comparable) |
| LiveCodeBench | ~29% | ~35% | -6% (AR leads) |
| BigCodeBench | ~42% | ~41% | +1% (comparable) |

### Required Charts
- **Figure 7.1 — Benchmark heatmap:** Heatmap with models on y-axis and benchmarks on x-axis, color-coded by score percentage. Visually reveals diffusion strengths (HumanEval, CanItEdit) vs. weaknesses (LiveCodeBench for some models).
- **Figure 7.2 — LiveCodeBench trajectory over time:** Line chart showing LLaDA family progression on LiveCodeBench from 8B (6.9%) → 2.0-flash (42.29%) → 2.1 (45.37%), demonstrating rapid improvement.
- **Figure 7.3 — Diffusion vs. AR grouped bar chart:** Side-by-side comparison on each benchmark showing best diffusion vs. best AR model.

### Key Quotes
> "Diffusion models' global planning and iterative refinement capabilities are particularly well-suited for the non-sequential nature of code generation." — A Survey on Diffusion Language Models (2025)

> "On HumanEval, LLaDA2.0-flash begins to exhibit clear advantages in complex generative tasks, a sign that the diffusion architecture may hold inherent strengths." — LLaDA2.0 technical report

### Case Studies / Examples
- **Case: CanItEdit analysis** — Why diffusion models excel at code editing (iterative refinement naturally maps to edit operations). Seed Diffusion's 54.3% vs. Llama-3.1's 39.5% demonstrates this structural advantage.
- **Case: BigCodeBench gap analysis** — Why Gemini Diffusion (45.4%) outperforms LLaDA2.0-flash (41.58%) despite trailing on simpler benchmarks — hypothesis: different training data composition or architectural choices.

---

## Chapter 8: Training Compute, Efficiency, and Scaling Laws

**Technical depth level:** Deep technical; quantifies training costs and scaling behavior.

### Content Points
- **Diffusion scaling behavior:** Similar scaling trends to AR when trained on equal data. LLaDA2.0 demonstrates "highly competitive" scaling curves comparable to AR baselines.
- **Training compute gap:** Masked diffusion models historically require up to 16x more compute than AR to match validation NLL. This has been a central obstacle to adoption.
- **Data-constrained advantage:** Recent work shows diffusion models outperform AR in data-constrained settings at sufficient compute. Critical compute frontier follows power law: C_crit(U) ∝ U^2.174.
- **LLaDA 2.0 innovations reducing training cost:** WSD pretraining enables knowledge transfer from AR models (avoiding training from scratch). CAP training improves decoding efficiency without quality loss.
- **MoE efficiency:** LLaDA-MoE uses 1.4B active parameters out of 7B total; LLaDA2.0 uses 1.44B active out of 16B. Achieves 3B dense model performance at ~half the active compute.
- **Inference efficiency:** Block parallel decoding with KV cache reuse. TPF (tokens per forward) as key metric: LLaDA2.0 ~3.1, LLaDA2.1 (S-mode) ~5.9, LLaDA2.1 (Q-mode) ~3.1.

### Required Tables
**Table 8.1 — Training Compute Comparison**
| Model | Parameters | Training Tokens | Compute | Relative to AR |
|-------|------------|----------------|---------|----------------|
| LLaDA (8B) | 8B dense | From scratch | High | ~16x for same NLL |
| LLaDA-MoE | 7B / 1.4B active | 20T+ tokens | Reduced via MoE | ~5x active params vs. dense |
| LLaDA 2.0 mini | 16B / 1.44B active | WSD transfer | Inherited AR knowledge | Avoids from-scratch cost |
| LLaDA 2.0 flash | 100B MoE | WSD + CAP | Large but efficient | Comparable to AR at scale |
| Gemini Diffusion | Several B (est.) | Google-scale | Very large | N/A |

**Table 8.2 — Scaling: When Diffusion Beats AR (Data-Constrained)**
| Unique Tokens | Critical Compute (FLOPs) | Regime |
|---------------|-------------------------|--------|
| 100M | ~10^17 | Very high compute needed |
| 500M | ~10^19 | Achievable (used in paper) |
| 1B | ~10^20 | Standard large-model training |
| 10B | ~10^22 | Frontier-scale |

**Table 8.3 — Inference Efficiency Metrics**
| Model | TPF (Tokens/Forward) | Steps | Equivalent AR passes | Speedup |
|-------|---------------------|-------|---------------------|---------|
| LLaDA 2.0 | 3.1 | ~32 | 1x | Baseline |
| LLaDA 2.1 (S-mode) | 5.9 | ~16 | 0.5x | ~2x faster |
| LLaDA 2.1 (Q-mode) | 3.1 | ~32 | 1x | Quality mode |
| AR (Qwen3) | 1.0 | n (sequence length) | n | Baseline |

### Required Charts
- **Figure 8.1 — Scaling law curves:** Log-log plot of validation loss vs. training compute for diffusion and AR models, showing crossover point where diffusion overtakes AR (from "Diffusion Beats Autoregressive in Data-Constrained Settings").
- **Figure 8.2 — Critical compute frontier:** Power law curve C_crit(U) showing compute required for diffusion to match AR as function of unique tokens.
- **Figure 8.3 — TPF comparison across models:** Bar chart comparing tokens-per-forward across LLaDA 2.0, LLaDA 2.1 (S and Q modes), and AR baseline.
- **Figure 8.4 — Active vs. total parameters:** Stacked bar showing total parameters (full height) vs. active parameters (filled portion) for LLaDA-MoE and LLaDA2.0, illustrating MoE efficiency.

### Key Quotes
> "Diffusion-based training can improve code modeling quality beyond AR training alone." — Stable-DiffCoder authors

> "The LLaDA2.0 series successfully demonstrates that diffusion-based language models are a powerful and scalable alternative to the dominant auto-regressive paradigm. While rapidly narrowing the gap on general benchmarks, they are already showcasing the potential to surpass traditional architectures in complex, structured domains like code generation and tool use."

### Case Studies / Examples
- **Case: Data-constrained scaling experiment** — Training 2.3B parameter diffusion model on 500M unique tokens for 130 epochs; diffusion consistently outperforms AR counterpart on downstream benchmarks at the critical compute threshold.
- **Case: MoE efficiency in practice** — LLaDA-MoE 7B with 1.4B active achieving 3B dense performance; concrete FLOP savings calculation for a standard inference workload.

---

## Chapter 9: Architectural Innovations and Inference Optimization

**Technical depth level:** Deep technical; focuses on specific mechanisms and their empirical impact.

### Content Points
- **Block parallel decoding:** Group tokens into blocks; unmask high-confidence tokens block by block. Enables KV cache reuse across blocks. Core technique in LLaDA 2.0/2.1.
- **Confidence-Aware Parallel (CAP) training:** Auxiliary confidence loss minimizes entropy on correctly predicted tokens. Improves TPF from ~2.5 to ~3.1 without quality loss. LLaDA2.0-flash-CAP achieves 500 TPS vs. 383 TPS baseline.
- **Token editing (LLaDA 2.1):** Dual objective — M2T (mask-to-text: standard unmasking) + T2T (text-to-text: correcting noised tokens). Enables "write then correct" behavior.
- **Dual-threshold decoding:** Configurable thresholds τ_M2T and τ_T2T. S-mode: low M2T threshold (aggressive drafting), moderate T2T (selective edits). Q-mode: high thresholds for both (conservative, high-quality).
- **Multi-Block Editing (MBE):** Iterative refinement across multiple blocks rather than single-pass. Consistent performance gains on reasoning and coding at modest throughput cost.
- **dInfer framework:** Ant Group's inference engine. Up to 3x faster than vLLM, 10x faster than NVIDIA Fast-dLLM. 1,011 tok/s on HumanEval for LLaDA-MoE.
- **FP8 quantization:** Applied to LLaDA 2.1; boosts throughput with minimal score degradation.

### Required Tables
**Table 9.1 — Key Architectural Innovations and Their Impact**
| Innovation | Model | What It Does | Speed Impact | Quality Impact |
|------------|-------|--------------|--------------|----------------|
| Block parallel decoding | LLaDA 2.0 | Parallel unmasking by blocks | +2.1x vs. AR | Minimal loss |
| CAP training | LLaDA 2.0 | Confidence loss sharpening | +30% TPF | Maintained |
| KV cache reuse | LLaDA 2.0 | Cache reuse across blocks | +40% TPS | None |
| Token editing (M2T+T2T) | LLaDA 2.1 | Draft + correct capability | Enables S/Q modes | Improves in Q-mode |
| Dual-threshold decoding | LLaDA 2.1 | Configurable speed/quality | 2x TPF in S-mode | Matches 2.0 in Q-mode |
| Multi-Block Editing | LLaDA 2.1 | Cross-block refinement | -10% TPS | +2-5% score |
| dInfer framework | LLaDA-MoE | Optimized inference engine | 10x vs. Fast-dLLM | None |
| FP8 quantization | LLaDA 2.1 | Low-precision inference | +15-20% TPS | -0.5% score |

**Table 9.2 — CAP Training Ablation (from SGLang blog)**
| Configuration | TPS | Relative Speed | Avg Score |
|---------------|-----|----------------|-----------|
| LLaDA2.0-flash (no CAP) | 383 | 1.0x | Baseline |
| LLaDA2.0-flash-CAP | 500 | 1.3x | Slight improvement |
| AR baseline (Qwen3) | 258 | 0.67x | — |
| AR baseline (Ling) | 237 | 0.62x | — |

### Required Charts
- **Figure 9.1 — Threshold vs. TPF trade-off:** Plot showing score and TPF vs. denoising threshold (0.85 to 0.95) and block size (16 to 64), from LLaDA2.0 paper.
- **Figure 9.2 — S-mode vs. Q-mode comparison:** Grouped bars showing score and TPF for both modes across multiple benchmarks.
- **Figure 9.3 — Inference stack speedup breakdown:** Waterfall chart showing cumulative speedup from block decoding → CAP → KV cache → dInfer → quantization.

### Key Quotes
> "LLaDA2.1's token editing mechanism represents a fundamental shift from pure generation to generate-then-revise, mirroring how human programmers actually write code."

> "dInfer was up to three times faster than vLLM and 10 times faster than Nvidia's own framework Fast-dLLM." — Ant Group benchmarks

### Case Studies / Examples
- **Case: CAP training mechanism** — Detailed walkthrough of how the confidence loss ℒ_conf selectively minimizes entropy on correct predictions, compelling the model to become more decisive.
- **Case: Token editing in action** — Concrete example of LLaDA 2.1 generating a Python function: first draft has a type error, T2T mechanism detects and corrects it in subsequent iteration.
- **Case: dInfer architecture** — How dInfer achieves 10x over Fast-dLLM through optimized kernel scheduling, parallel block execution, and reduced synchronization overhead.

---

## Chapter 10: Market Implications and Future Outlook

**Technical depth level:** Strategic/analytical; less technical depth, more forward-looking analysis.

### Content Points
- **Current state summary:** Diffusion LMs have achieved parity with AR models on code generation benchmarks while delivering 2-10x inference speedups. Key players: Ant Group (open-source leader), Google DeepMind (research frontier), ByteDance (speed leader), Inception (commercial deployment).
- **Competitive dynamics:** Open-source diffusion ecosystem (LLaDA, DiffuCoder, Stable-DiffCoder) vs. closed commercial APIs (Mercury, Gemini Diffusion). Ant Group's open-source strategy via InclusionAI.
- **Use case fit:** Diffusion models excel at (a) high-throughput code generation, (b) code editing/revision tasks, (c) structured generation requiring global planning. Less proven on (a) long-horizon reasoning, (b) multilingual tasks, (c) real-time interactive chat.
- **Adoption barriers:** (1) Ecosystem maturity (tooling, frameworks); (2) Training compute requirements; (3) Limited reasoning capability on hardest benchmarks; (4) Community familiarity with AR paradigm.
- **Future directions:** (1) Scaling to larger parameters (200B+); (2) Deeper RL integration (thinking paradigms); (3) Unified multimodal diffusion; (4) Hardware co-design for diffusion inference.
- **Timeline projection:** 2025 = proof of concept; 2026 = commercial viability; 2027+ = potential paradigm shift.

### Required Tables
**Table 10.1 — Diffusion LM Market Map**
| Category | Organization | Model | Open Source | Focus | Stage |
|----------|-------------|-------|-------------|-------|-------|
| Open-source leader | Ant Group / InclusionAI | LLaDA 2.1 | Yes | General + code | Production |
| Research frontier | Google DeepMind | Gemini Diffusion | Demo only | General text | Experimental |
| Speed leader | ByteDance Seed | Seed Diffusion | No | Code | Preview |
| Open-source code | ByteDance Seed | Stable-DiffCoder | Yes | Code | Production |
| Commercial API | Inception | Mercury Coder | No | Code | Commercial |
| Academic | Apple + HKU | DiffuCoder | Yes | Code | Research |

**Table 10.2 — Task Suitability Analysis**
| Task Type | Diffusion Fit | Key Advantage | Key Limitation | Best Model |
|-----------|--------------|---------------|----------------|------------|
| Code generation (HumanEval) | Strong | Global planning, speed | May miss sequential deps | LLaDA2.0-flash |
| Code editing (CanItEdit) | Very strong | Iterative refinement native | — | Seed Diffusion |
| Complex reasoning (AIME) | Moderate | Bidirectional context | Hard to verify step-by-step | LLaDA2.1 (Q) |
| Real-time chat | Moderate | Speed | Latency to first token | Gemini Diffusion |
| Multilingual | Weak | — | Less training data | Gemini Diffusion |
| Long-horizon planning | Moderate | Revision capability | Step ordering | Early stage |

**Table 10.3 — Pricing Comparison for Code Generation Workloads**
| Provider | Model | Input $/M | Output $/M | Cost for 1M-in + 200K-out | Relative Cost |
|----------|-------|-----------|------------|---------------------------|---------------|
| Inception | Mercury Coder | $0.25 | $0.75 | $400 | 1.0x (baseline) |
| OpenAI | GPT-4o | $2.50 | $10.00 | $4,500 | 11.3x |
| Anthropic | Claude 3.5 Sonnet | $3.00 | $15.00 | $6,000 | 15.0x |
| Google | Gemini 2.0 Flash-Lite | ~$0.15 | ~$0.60 | ~$270 | 0.7x |
| Ant Group | LLaDA 2.1 (self-hosted) | Hardware cost only | — | ~$50-100* | ~0.2x |

*Estimated cloud GPU cost for self-hosted open-source model

### Required Charts
- **Figure 10.1 — Capability frontier map:** 2D scatter with speed (x-axis) and code benchmark average (y-axis), showing positioning of all major models with diffusion vs. AR color coding.
- **Figure 10.2 — Adoption timeline projection:** Timeline showing expected milestones from 2025-2028 for diffusion LMs.
- **Figure 10.3 — Cost-per-request comparison:** Horizontal bar chart comparing cost for a standard coding workload across Mercury, GPT-4o, Claude, and open-source self-hosted options.

### Key Quotes
> "This positions diffusion models as a highly promising direction for the future of language generation." — LLaDA2.0 technical report

> "Ant Group's push into alternative model paradigms underscores how China's Big Tech firms are stepping up efforts in algorithm and software optimisation to offset the country's disadvantages in advanced AI chips." — South China Morning Post

### Case Studies / Examples
- **Case: Mercury's 10x speed claim** — Analysis of what "10x" means in practice: comparing Mercury's throughput to speed-optimized AR models on identical hardware under batch inference conditions.
- **Case: InclusionAI's open-source strategy** — Ant Group's approach of releasing models through InclusionAI, and how this accelerates ecosystem development around diffusion LMs.
- **Scenario analysis:** Three scenarios for 2027 — (1) AR remains dominant, diffusion is niche for high-throughput code; (2) Diffusion achieves parity on most tasks, captures 30% of inference market; (3) Diffusion surpasses AR on key dimensions, triggers paradigm shift.

---

## Appendix A: Glossary of Terms

**Required entries:**
- **Autoregressive (AR) model:** Generates text sequentially, one token at a time, conditioned on all previous tokens.
- **Bidirectional attention:** Attention mechanism where each position can attend to all other positions (not restricted to previous positions).
- **Block parallel decoding:** Inference technique that processes tokens in blocks rather than individually, enabling parallel computation.
- **CAP (Confidence-Aware Parallel) training:** Auxiliary training objective that minimizes entropy on correctly predicted tokens to improve decoding efficiency.
- **Coupled-GRPO:** Reinforcement learning algorithm for diffusion models using complementary mask sampling for stable gradient estimation.
- **dInfer:** Ant Group's open-source inference framework optimized for diffusion language models.
- **ELBO (Evidence Lower Bound):** Lower bound on log-likelihood used to train diffusion models when exact likelihood is intractable.
- **Masking ratio:** Fraction of tokens hidden (masked) during a diffusion step.
- **MBE (Multi-Block Editing):** Cross-block refinement mechanism in LLaDA 2.1 for improving global coherence.
- **MDLM (Masked Diffusion Language Model):** Diffusion model that uses masking as the corruption process.
- **MoE (Mixture of Experts):** Architecture that routes each input to a subset of parameter "experts," reducing active compute.
- **Remasking:** Re-masking low-confidence predictions during inference for iterative refinement.
- **TPS (Tokens Per Second):** Inference throughput metric.
- **TPF (Tokens Per Forward):** Number of tokens generated per forward pass; measures parallelism.
- **VRPO (Variance-Reduced Preference Optimization):** RL alignment algorithm for diffusion models that reduces ELBO estimation variance.
- **WSD (Warmup-Stable-Decay):** Learning rate schedule that enables knowledge transfer from AR to diffusion models.

---

## Appendix B: Model Release Timeline

**Required content:** Chronological list of all significant diffusion language model releases from 2024 to early 2026.

| Date | Model | Organization | Significance |
|------|-------|-------------|--------------|
| 2024 | SEDD | Various | ICML 2024 Best Paper; score-based discrete diffusion |
| 2024 | MDLM | Various | Simplified masking framework with cosine schedule |
| Early 2025 | LLaDA (8B) | Renmin U + Ant Group | First large-scale diffusion LM from scratch |
| Mar 2025 | Mercury Coder | Inception | First commercial diffusion code model |
| May 2025 | LLaDA 1.5 | Renmin U + Ant Group | VRPO for diffusion RL alignment |
| Jul 2025 | Seed Diffusion Preview | ByteDance Seed | 2,146 tok/s; fastest diffusion code model |
| Sep 2025 | LLaDA-MoE | Ant Group | First native MoE diffusion LM; dInfer framework |
| 2025 | Gemini Diffusion | Google DeepMind | DeepMind's experimental text diffusion model |
| 2025 | DiffuCoder (7B) | Apple + HKU | Coupled-GRPO for code generation |
| Dec 2025 | LLaDA 2.0 | Ant Group (InclusionAI) | First 100B diffusion LM; 535 tok/s base |
| Jan 2026 | Stable-DiffCoder | ByteDance Seed | Open-source 8B code diffusion model |
| Feb 2026 | LLaDA 2.1 | Ant Group (InclusionAI) | Token editing; 892-1587 TPS; dual-mode decoding |

---

## Appendix C: Detailed Benchmark Methodology Notes

**Required content:** Brief notes on each benchmark to contextualize results.

- **HumanEval:** 164 hand-written Python programming problems. pass@1 metric. Easy difficulty.
- **HumanEval+:** Extended HumanEval with more tests. Harder to pass.
- **MBPP:** 974 Python problems crowdsourced. Mostly introductory-level.
- **MBPP+:** Extended MBPP with additional tests.
- **LiveCodeBench:** Competition-level coding problems updated continuously. v6 = current version. Significantly harder than HumanEval/MBPP.
- **BigCodeBench:** Complex real-world coding scenarios requiring multiple function calls and library usage.
- **CanItEdit:** Code editing benchmark measuring ability to make targeted changes to existing code.
- **Aider:** Real-world coding assistant benchmark measuring ability to use tools and edit code in repository context.
- **MultiPL-E:** Multi-language extension of HumanEval (Python, Java, C++, etc.).
- **CRUXEval:** Code reasoning and execution understanding benchmark.
- **SWE-Bench Verified:** Real-world software engineering tasks on GitHub issues.

---

*Content plan compiled from technical papers, benchmark reports, and official documentation. All benchmark figures sourced from published results as of early 2026.*
