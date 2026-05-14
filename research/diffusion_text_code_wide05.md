## Facet: Technical Architectures and Training Approaches for Diffusion Language Models

---

### Key Findings

#### 1. Discrete vs. Continuous Embedding Diffusion -- Two Competing Paradigms

- **Discrete diffusion** operates directly on token space using categorical distributions, typically via masked diffusion (MDM), score entropy (SEDD), or uniform transitions (D3PM). This approach has shown the strongest empirical results at scale, with LLaDA-8B and LLaDA2.0-100B being the flagship examples [^204^][^184^].

- **Continuous embedding diffusion** encodes text into continuous space, applies Gaussian diffusion, then rounds back to discrete tokens. Early examples include Diffusion-LM and DiffuSeq [^230^][^233^]. This paradigm struggled historically due to "the extreme sparsity of the simplex space, making score estimation particularly challenging" [^173^].

- **LangFlow (2026)** is the first continuous diffusion language model to rival discrete diffusion, achieving PPL of 30.0 on LM1B and 24.6 on OpenWebText. It connects embedding-space DLMs to Flow Matching via Bregman divergence and introduces: (1) an ODE-based NLL bound, (2) an information-uniform noise schedule with learnable Gumbel distribution, and (3) effective self-conditioning. It "provides the first clear evidence that continuous diffusion is a promising paradigm for language modeling" [^173^].

- **ELF (Embedded Language Flows, 2026)** takes a different approach: it uses continuous-time Flow Matching in a frozen contextual embedding space, keeping the sampling trajectory entirely continuous and discretizing only at the final decoding step. Unlike prior latent diffusion LMs, ELF does not require a separately trained decoder [^227^].

- **The verdict**: Discrete diffusion (particularly masked diffusion) currently dominates at scale, but continuous approaches are rapidly closing the gap. LangFlow explicitly matches state-of-the-art masked diffusion at the same model/data scale. The tradeoff is that discrete diffusion sacrifices "expressive latent spaces, which limits controllable and few-step generation," while continuous diffusion preserves "editability and denser data spaces" [^173^].

#### 2. Masked Diffusion Language Models (MDLM) -- Technical Details

- **Core formulation**: MDLM corrupts text by progressively replacing tokens with a [MASK] token and trains a model to reverse this process. The training objective is a simplified form of the Negative Evidence Lower Bound (NELBO): L_NELBO(x;θ) = E_{t,x_t~q(·|x)}[-w(t)·log p_θ(x|x_t)] [^180^].

- **Key insight from MDLM (Sahoo et al., 2024)**: Multiple parameterizations of masked/discrete diffusion models are mathematically equivalent -- "MDLM, SEDD, and RADD formulations are equivalent under certain conditions," which clarifies relationships among prior formulations and unifies the field [^251^].

- **Architecture**: MDLM uses a diffusion transformer (DiT) architecture with rotary positional embeddings (RoPE). For training, sentence packing creates uniform-length blocks. The model does NOT use causal masking -- it uses full bidirectional attention [^124^][^204^].

- **LLaDA-8B** scales MDLM to 8B parameters, trained from scratch on 2.3T tokens using 0.13M H800 GPU hours. It uses vanilla multi-head attention (not grouped query attention) since LLaDA is incompatible with KV caching. The FFN dimension is reduced to maintain comparable model size [^204^].

- **LLaDA demonstrates strong scalability**: LLaDA "scales effectively up to a computational budget of 10^23 FLOPs, achieving comparable results to self-constructed ARM baselines" and surpasses LLaMA2 7B on nearly all 15 zero/few-shot learning tasks while performing on par with LLaMA3 8B [^204^].

#### 3. Block Diffusion -- LLaDA's Approach and LLaDA2.0's WSD Scheme

- **Block Diffusion Language Models (BDLM)** divide the sequence into equal-length blocks. Blocks are decoded sequentially (autoregressively), but tokens within each block are denoised in parallel via diffusion. This hybrid approach enables variable-length generation while preserving KV-cache-like efficiency [^174^][^184^].

- **BD3-LM (Arriola et al., 2025)** formalized this approach: "It generates text autoregressively over blocks of tokens, but within each block, it uses a diffusion process identical to that of MDLM" [^180^].

- **LLaDA2.0 (2025)** introduces a novel three-stage Warmup-Stable-Decay (WSD) continual pre-training paradigm for converting AR models to diffusion models [^184^]:
  - **Phase 1 (Warmup)**: Progressively increase block size from 1 to 4096, gradually expanding the receptive field. Starting from block size L_B=1 (equivalent to AR), incrementally scale to 4, 32, 64, then 4096.
  - **Phase 2 (Stable)**: Train as full-sequence MDLM with block size 4096, deepening understanding of diffusion dynamics through extensive training on large-scale corpora.
  - **Phase 3 (Decay)**: Gradually reduce block size from 4096 back to a small block size (e.g., 32), converting the model back into an efficient BDLM while preserving global contextual knowledge.

- **LLaDA2.0 also introduces** a document-level attention mask that "restricts self-attention within individual documents, ensuring coherent context modeling" and prevents "spurious dependencies across document boundaries" during packed training sequences [^184^].

- **LLaDA2.0 uses cuDNN** as the attention backend, achieving "more than 1.3x end-to-end speedup and over 90% memory savings in the attention layer compared to the unfused attention implementation" [^184^].

#### 4. Remasking Strategies -- Approaches and Impact

- **Low-confidence remasking**: Remasks tokens with the lowest predicted probabilities, leaving the most confident B/N tokens decoded at each step. Used by LLaDA, MaskGIT, and many others [^174^][^175^].

- **Dynamic low-confidence remasking**: First attempts to remask tokens with confidence below threshold τ; if insufficient tokens reach the target B/N, falls back to standard low-confidence remasking. May result in fewer than N total steps [^174^].

- **High-entropy remasking**: Evaluates the entropy of the predicted token distribution at each position and remasks positions with highest entropy values [^174^].

- **Random remasking**: Uniformly randomly selects tokens to remask [^174^].

- **CoRe (Context-Robust Remasking, 2026)**: A major innovation that addresses the key limitation of confidence-based heuristics -- they "quickly become stale" because "a token may receive high probability under an ambiguous early context, yet later become incompatible once additional tokens stabilize the surrounding structure." CoRe measures sensitivity to context change: "a generated token is considered stable if it remains strongly predicted even when parts of the surrounding context are masked." This shifts selection from "Was the token uncertain when it was chosen?" to "Does the token remain plausible under dynamic context change?" [^182^][^183^]

- **Token-to-Mask Refinement (2026)**: Proposes the "LogitDiff rule" that captures trajectory-level signals at no extra forward-pass cost, providing a unified perspective comparing random remasking, perturbation-based sensitivity, and learned detectors [^181^].

- **STDD (Spatio-Temporal Dynamics-Driven Token Refinement, 2025)**: Proposes detecting "Temporal Variance and Spatial Deviance" of each token -- reflecting convergence status and inter-token correlations -- to adaptively adjust the confidence threshold for every token at every step. Achieves "speedups of up to 8.9 times while faithfully preserving generation quality" [^185^].

- **RemeDi (Self-Reflective Remasking, 2025)**: A training-based approach that jointly predicts token distributions and per-token confidence scores. Uses a two-stage pipeline: (1) Remask SFT where the model learns to identify and remask incorrect tokens, and (2) Remask RL with outcome-based reinforcement learning. Achieves 89.1% on GSM8K and 73.2% on HumanEval [^186^].

- **Impact**: Remasking strategies have emerged as one of the highest-leverage research areas for improving diffusion LLM quality. As noted by the P2 authors, "the choice of denoising path and token order can substantially affect DLM sampling and training" [^177^].

#### 5. Self-Conditioning and Partial Noising

- **Self-conditioning**: Refers to "conditioning the denoiser on its own earlier predictions along the denoising trajectory." For discrete data, Analog Bits applied self-conditioning within a continuous-state bit-diffusion formulation. Strudel et al. studied it in continuous embedding-space diffusion [^188^].

- **SCMDM (Simple Self-Conditioning Adaptation for Masked Diffusion Models, 2026)**: An effective post-training adaptation for masked diffusion language models. "Full self-conditioning is consistently more effective than partial self-conditioning in the post-training regime." Under GPT2 evaluator at 1000 steps, generative perplexity improves from 42.89 (vanilla MDLM) to 37.04 (rate=0.5) to 23.72 (full self-conditioning) [^188^].

- **Degradation of self-conditioning**: A known training challenge where "the self-condition denoising step could easily achieve a low loss by simply copying z_0 as its output," causing the model to "marginalize or even ignore z_t." FastDiSS addresses this via Self-conditioning Perturbation (SCP) and Model-aware Noise Scaling (MANS) [^191^][^192^].

- **Partial noising**: Introduced by DiffuSeq for conditional text generation, where "unlike conventional diffusion models that corrupt the whole z_t, we only impose noising on y_t" (the target portion). An anchoring function replaces the corrupted source with the original source tokens. This is crucial for conditional generation tasks [^230^][^233^].

#### 6. AR-to-Diffusion Conversion Techniques

- **Motivation**: Converting pretrained AR models into diffusion models is "a more data-efficient alternative" to training from scratch. AR models contain rich linguistic knowledge that can be preserved during conversion [^70^].

- **LLaDA2.0 WSD**: As described in Section 3, uses Warmup-Stable-Decay for systematic conversion from AR to diffusion, preserving linguistic knowledge while introducing diffusion capabilities [^184^].

- **I-DLM (Introspective Diffusion Language Models, 2026)**: A breakthrough approach that identifies "introspective consistency" as the key missing principle in prior DLMs. Uses "introspective-consistency training" -- a recipe for converting pretrained AR models into DLMs using just ~5B tokens. Combines causal attention, logit shift, and an all-masked objective. I-DLM is "the first DLM to match the quality of its same-scale AR counterpart" -- achieving 69.6 on AIME-24 and 45.7 on LiveCodeBench-v6, exceeding LLaDA-2.1-mini (16B) by more than 26 and 15 points respectively. It also delivers "about 3x higher throughput than prior state-of-the-art DLMs" [^70^][^228^].

- **Dream (Ye et al., 2025)**: Uses AR-based LLM initialization and context-adaptive noise scheduling to scale diffusion language models. Introduces the "logit shift technique, aligning the diffusion objective with the AR model's logits [i] → token [i+1] mapping" [^70^][^259^].

- **SDAR (Cheng et al., 2025)**: Converts AR models via full-model training on ~50B tokens for block-parallel generation [^70^].

- **NBDiff (Tian et al., 2025)**: Extends SDAR with causal prefix constraints [^70^].

- **DiffuLLaMA (Gong et al., 2025)**: Showed that fine-tuning with a masked diffusion objective "reduces training cost significantly" -- continual pretraining based on LLaMA parameters [^258^][^259^].

#### 7. Training Objectives and Loss Functions

- **Score Entropy Loss (SEDD)**: "A novel score entropy loss that generalizes score matching for discrete spaces." SEDD is parameterized by the concrete score and "beats previous language diffusion models and rivals autoregressive models for both perplexity and quality" [^124^][^214^].

- **Simplified MDLM objective**: Sahoo et al. (2024) and Shi et al. (2024) rewrote the training objective "as a mixture of classical masked language modeling losses or as a continuous-time weighted integral of cross-entropy terms, yielding more scalable training" [^201^].

- **PAPL (Planner-Aware Path Learning, 2025-2026)**: Addresses a critical training-inference mismatch. Standard discrete diffusion training assumes uniformly random denoising paths, but inference uses planners that select non-uniform paths. PAPL derives a new "planned evidence lower bound (P-ELBO)" and implements planner-aware training as "a one-line code change" that uses self-planning to compute weighted loss on more likely generation paths. Results: 40% relative improvement in protein foldability, 4x MAUVE gain in text, and 23% improvement on HumanEval [^225^][^252^][^255^].

- **SNCE (Geometry-Aware Supervision)**: Replaces the standard cross-entropy term with a modified objective in both autoregressive and discrete diffusion training objectives [^200^].

- **CD4LM**: Points out that "standard discrete diffusion models are trained to denoise gold tokens corrupted by random noise, whereas at inference time they denoise self-generated tokens." Introduces a two-step training scheme with step-aware losses and a curriculum [^201^].

#### 8. Timestep Scheduling and Curriculum Strategies

- **Noise schedules**: Common choices include linear scheduling (α_t = 1-t), geometric schedules, log-linear schedules, and cosine schedules. SEDD found that "a log-linear noise schedule helps SEDD Absorb for perplexities" [^124^].

- **LangFlow's information-uniform principle**: Motivates a learnable noise scheduler based on a Gumbel distribution, specifically designed for language modeling. This is a key innovation that adapts the noise schedule to the data modality [^173^].

- **Context-adaptive noise scheduling (Dream)**: "Leverages AR-based LLM initialization and context-adaptive noise scheduling to scale diffusion language models" -- the noise level adapts based on the input context [^259^].

- **WSD curriculum (LLaDA2.0)**: The Warmup-Stable-Decay strategy is essentially a curriculum for AR-to-diffusion conversion, progressively expanding block sizes to smoothly introduce diffusion-style context [^184^].

- **InfoDiffusion**: Introduces "an information entropy-aware noise schedule to guide the model toward a more human-like 'key-info-first' process that prioritizes generating core content" [^14^].

#### 9. Inference Optimization

- **Fast-dLLM (2025)**: Introduces a block-wise approximate KV Cache mechanism for bidirectional diffusion models. Combines KV cache (2-3.6x speedup) with confidence-aware parallel decoding (4-6x speedup) for combined improvements of up to 11-27.6x throughput. "The effectiveness stems from the observation that KV activations exhibit high similarity across adjacent inference steps" [^189^][^190^].

- **Self Speculative Decoding (SSD, 2025)**: A lossless inference acceleration method that "leverages the dLLM itself as both speculative decoding drafter and verifier without auxiliary modules." Achieves up to 3.46x speedup while keeping output identical to stepwise decoding [^229^].

- **Speculative Diffusion Decoding (SpecDiff)**: Uses discrete diffusion models to generate draft sequences for speculative decoding, allowing parallelization of both drafting and verification. Provides "up to 7.2x speedups over standard generation processes and up to 1.75x speedups over existing speculative decoding approaches" [^231^][^238^].

- **I-DLM's Introspective Strided Decoding (ISD)**: Simultaneously generates new tokens and revises prior ones within the same forward pass. At [MASK] positions, the model proposes new tokens; at introspection positions, it revisits previous tokens. Uses adaptive stride: "easy tokens are accepted in parallel while difficult tokens fall back toward AR-quality generation." Achieves TPF≈2.3-2.4x at typical acceptance rates [^70^].

- **Mercury Coder (Inception Labs)**: First commercial-scale diffusion LLM. Mercury Coder Mini achieves 1,109 tokens/sec on NVIDIA H100 GPUs -- "outperform speed-optimized frontier models by up to 10x on average while maintaining comparable quality" [^256^][^85^]. Mercury 2 extends this to reasoning with "5x faster performance" than leading speed-optimized LLMs [^260^].

- **Gemini Diffusion (Google DeepMind)**: "Reports performance comparable to much larger models while offering faster inference" -- reportedly generating over 1,400 tokens per second [^189^][^9^].

- **DART (Diffusion-Inspired Speculative Decoding)**: Adopts a diffusion-inspired masked prediction mechanism tailored to speculative decoding, eliminating autoregressive rollout in the drafting stage [^235^].

#### 10. Scaling Laws for Diffusion Language Models

- **MDM-Prime-v2 (2026)**: A landmark scaling analysis showing that "MDM-Prime-v2 is 21.8x more compute-efficient than autoregressive models (ARM)." Under compute-optimal comparisons, MDM-Prime-v2 achieves PPL of 7.77 on OpenWebText vs. ARM's 12.99, MDM's 18.94, and MDM-Prime's 13.41 [^223^].

- **Scaling behavior differences**: "MDM-based methods scale more effectively when trained on an abundance of tokens" (larger b^). The compute-optimal setup reveals that ARM is ~17.6x more compute-efficient than MDM, while MDM-Prime-v2 achieves a further 21.8x improvement over ARM [^223^].

- **LLaDA scaling**: LLaDA "scales effectively up to a computational budget of 10^23 FLOPs, achieving comparable results to self-constructed ARM baselines trained on the same data across six tasks" [^204^].

- **Predictions**: For MDM-Prime-v2, compute-optimal predictions suggest for target model sizes of 7B, 14B, and 32B, the corresponding optimal training token counts are 2.9T, 7.3T, and 21.7T respectively [^223^].

- **LLaDA-MoE**: Shows that sparse MoE architectures work for diffusion models too, achieving competitive performance with activated parameters of only 1B vs. 8B dense [^211^].

#### 11. Theoretical Understanding -- Why Diffusion Works for Discrete Text

- **Any-order generation**: A core theoretical property. "Autoregressive language models are usually trained with a fixed left-to-right factorization, but this factorization is only one possible ordering of the joint distribution." Diffusion models can "generate tokens in parallel, with arbitrary prompt locations" [^177^][^257^].

- **P-ELBO (Planned Evidence Lower Bound)**: PAPL proves that "the standard discrete diffusion training ELBO does not accurately describe a denoiser that uses a non-uniform planner" and derives a new bound that "incorporates planner-based reverse dynamics directly into the training objective" [^225^][^252^].

- **P2 (Path Planning for Masked Diffusion Sampling, 2025)**: Extracts the full power of MDMs by decomposing each generation step into planning and denoising sub-stages. "P2 establishes a (new) expanded evidence lower bound (ELBO) on the log marginal likelihood of data." Improves performance by up to 68% in story generation (ROUGE) and 33% in code generation (pass@1) [^224^].

- **Introspective consistency**: I-DLM shows that "AR models agree with their own generations, while DLMs often do not." AR training has a structural advantage because "causal masking and logit shifting implicitly enforce introspective consistency." This insight explains why AR-to-diffusion conversion is effective [^70^].

- **Reversal curse**: LLaDA effectively breaks the "reversal curse" with consistent performance across forward and reversal tasks, "notably outperforming GPT-4o in a reversal poem completion task" [^204^].

---

### Major Players & Sources

| Entity | Role/Relevance |
|--------|---------------|
| **LLaDA / LLaDA2.0 Team (ML-GSAI, Renmin University)** | Open-source flagship diffusion LLM; 8B model trained from scratch; LLaDA2.0 scales to 100B/16B via MoE with WSD conversion paradigm [^204^][^184^] |
| **SEDD Team (Stanford - Lou, Meng, Ermon)** | Score Entropy Discrete Diffusion; principled score matching for discrete spaces; strong perplexity results rivaling GPT-2 [^124^][^214^] |
| **MDLM Team (Sahoo et al.)** | Simple and effective masked diffusion formulation; showed mathematical equivalence of multiple discrete diffusion parameterizations [^180^] |
| **LangFlow Team (UIUC - Chen, You et al.)** | First continuous diffusion to rival discrete; Flow Matching via Bregman divergence [^173^] |
| **I-DLM Team (Together AI, Stanford, UIUC)** | First DLM to match same-scale AR quality; introspective consistency principle; efficient AR conversion (~5B tokens) [^70^][^228^] |
| **Inception Labs (Mercury)** | First commercial-scale diffusion LLM; Mercury Coder achieves 1,109 tokens/sec on H100; Mercury 2 for reasoning [^256^][^260^] |
| **Google DeepMind (Gemini Diffusion)** | Commercial diffusion LLM reportedly generating 1,400+ tokens/sec [^189^] |
| **Dream Team (Ye et al.)** | 7B diffusion LLM with AR initialization and logit shift; strong general task performance [^259^] |
| **P2/PAPL Team (Peng et al., Duke)** | Path Planning for masked diffusion sampling; Planner-Aware Path Learning addressing training-inference mismatch [^224^][^225^] |
| **Fast-dLLM Team (NVIDIA, HKU)** | Block-wise approximate KV cache enabling 27.6x throughput improvement [^189^] |
| **Block Diffusion / BD3-LM (Arriola et al.)** | Formalized block diffusion with hybrid AR-diffusion approach [^180^][^226^] |
| **MDM-Prime-v2 Team** | 21.8x compute efficiency over ARMs via binary encoding and index shuffling [^223^] |

---

### Trends & Signals

- **Commercial viability confirmed**: Multiple commercial diffusion LLMs have launched (Mercury, Gemini Diffusion, Seed Diffusion), with Mercury reporting 1,000+ tokens/sec throughput [^75^][^256^][^253^].

- **AR-to-diffusion conversion becoming standard**: Rather than training from scratch, converting pretrained AR models is emerging as the dominant paradigm. LLaDA2.0's WSD, I-DLM's introspective consistency training, and Dream's AR initialization all follow this path [^184^][^70^][^259^].

- **Inference optimization is the critical bottleneck**: "Due to the lack of components analogous to KV cache and the requirement to compute results for all positions in each step, the deployment of dLLMs has consistently been constrained by inference efficiency" [^258^]. Fast-dLLM, SSD, and I-DLM all target this gap [^189^][^229^][^70^].

- **Remasking as a key differentiator**: The choice of remasking strategy has outsized impact on quality. CoRe, STDD, RemeDi, and Token-to-Mask Refinement represent a wave of increasingly sophisticated approaches moving beyond simple confidence thresholds [^182^][^185^][^186^][^181^].

- **Training-inference mismatch recognition**: PAPL's core insight -- that standard training with uniform random masking mismatches planner-based inference -- is gaining recognition as a fundamental issue. PAPL's simple fix (weighted cross-entropy) delivers consistent gains [^225^][^252^].

- **Scaling laws favor diffusion under certain regimes**: MDM-Prime-v2's scaling analysis shows diffusion can be 21.8x more compute-efficient than ARMs, particularly when training on abundant tokens [^223^].

- **Continuous diffusion renaissance**: LangFlow and ELF show that continuous diffusion in embedding space is becoming competitive again, after being overshadowed by discrete approaches [^173^][^227^].

- **Self-conditioning maturation**: From Analog Bits to SCMDM to FastDiSS, self-conditioning is being recognized as crucial for both continuous and discrete diffusion, though training challenges (self-conditioning degradation) remain [^188^][^191^][^192^].

---

### Controversies & Conflicting Claims

- **Discrete vs. continuous -- which is better?**: Discrete diffusion (MDLM/LLaDA) has dominated empirically, but LangFlow challenges this by showing continuous diffusion can match discrete performance while preserving editability and few-step generation advantages. The field remains split [^173^][^204^].

- **Quality vs. speed tradeoff**: "Decoding larger token blocks per denoising step tends to hurt accuracy, whereas smaller blocks increase latency" [^251^]. Fast-dLLM's KV cache approximation may sacrifice some quality; I-DLM claims to avoid this through introspective consistency. The optimal tradeoff point remains debated.

- **Is conversion or from-scratch training better?**: LLaDA (trained from scratch) and I-DLM (converted from AR) both achieve strong results. LLaDA2.0's WSD uses AR initialization, while the original LLaDA-8B was trained from scratch. Dream uses AR initialization. No clear consensus exists on which approach is superior at scale [^204^][^70^][^259^].

- **Any-order generation: blessing or curse?**: The ability to generate tokens in any order provides flexibility but may break sequential dependencies that AR models naturally enforce. Confidence-based parallel decoding can "disrupt critical token dependencies" under conditional independence assumptions [^189^][^257^].

- **Scaling law discrepancies**: MDM-Prime-v2 claims 21.8x compute efficiency over ARMs, but this is under specific conditions (binary encoding, index shuffling) and may not generalize to all diffusion architectures. The standard MDM baseline actually performed worse than ARMs in their analysis (PPL 18.94 vs 12.99) [^223^].

- **Perplexity comparisons**: SEDD and other diffusion models can only compute "upper bounds" on perplexity (not exact NLL), making fair comparisons with ARMs difficult. Different papers use different evaluation protocols, complicating cross-study comparisons [^214^].

---

### Recommended Deep-Dive Areas

1. **I-DLM's introspective consistency principle**: This appears to be the most significant architectural insight since MDLM itself. Understanding why causal masking + logit shift enables DLM quality matching could unify AR and diffusion paradigms. warrants depth because it closes the quality gap for the first time [^70^].

2. **PAPL and training-inference mismatch**: The theoretical proof that standard ELBO doesn't apply to planner-based inference, and the simple fix that delivers 4x MAUVE gains, suggests many reported diffusion LLM results could be substantially improved. warrants depth because it suggests current baselines are underperforming [^225^][^252^].

3. **Compute-optimal scaling of MDM-Prime-v2**: The claim of 21.8x compute efficiency is extraordinary and requires careful validation. The binary encoding + index shuffling techniques may generalize to other architectures. warrants depth because it could change resource allocation decisions [^223^].

4. **Inference optimization stack (Fast-dLLM + SSD + I-DLM serving)**: The gap between theoretical parallel generation and practical throughput is the main barrier to diffusion LLM adoption. The combination of KV-cache approximations, speculative decoding, and AR-compatible serving stacks needs systematic study. warrants depth because deployment depends on it [^189^][^229^][^70^].

5. **Context-robust remasking (CoRe, STDD)**: The insight that confidence-based heuristics become "stale" as context evolves represents a qualitative shift in remasking design. These approaches may be essential for structure-sensitive tasks like code generation. warrants depth because it directly affects generation quality [^182^][^185^].

6. **AR-to-diffusion conversion landscape**: With WSD (LLaDA2.0), introspective consistency (I-DLM), AR initialization (Dream), and SDAR all offering different conversion strategies, a systematic comparison on the same base model would be highly valuable. warrants depth because it affects how the field trains future diffusion LLMs [^184^][^70^][^259^].

---

### Key Source URLs and Dates

- LLaDA (2025-02-14): https://arxiv.org/abs/2502.09992
- LLaDA2.0 (2025-12-10): https://arxiv.org/html/2512.15745v1
- LangFlow (2026-04-14): https://arxiv.org/html/2604.11748v2
- I-DLM (2026-04-13): https://arxiv.org/html/2604.11035v1
- SEDD (2023): https://arxiv.org/pdf/2310.16834v3
- MDLM / Sahoo et al. (2024): https://arxiv.org/html/2603.22216v1 (referenced in Gumbel Distillation)
- Fast-dLLM (2025-05-28): https://arxiv.org/html/2505.22618v1
- P2 Path Planning (2026-03-05): https://arxiv.org/html/2502.03540v5
- PAPL (2026-03-05): https://arxiv.org/html/2509.23405v3
- CoRe (2026-02-04): https://arxiv.org/html/2602.04096v1
- SCMDM (2026-04-28): https://arxiv.org/html/2604.26985v1
- MDM-Prime-v2 (2026-03-17): https://arxiv.org/html/2603.16077v1
- Dream 7B (2025-06-16): https://arxiv.org/html/2508.15487
- Mercury (2025-06-17): https://arxiv.org/abs/2506.17298
- Speculative Diffusion Decoding (2024-08-10): https://arxiv.org/abs/2408.05636
- Self Speculative Decoding (2025-10-05): https://arxiv.org/abs/2510.04147
- DiffuSeq (2022): https://arxiv.org/pdf/2210.08933v3
- Block Diffusion / BD3-LM (2025): https://arxiv.org/html/2605.10020v1
- Promises/Outlooks/Challenges of SEDD (2024-06-17): https://arxiv.org/html/2406.11473v1
- LLaDA-MoE (2025): https://arxiv.org/html/2509.24389v1
- STDD (2025-12-07): https://arxiv.org/abs/2601.04205
- RemeDi (2025-09-28): https://arxiv.org/html/2509.23653v1
- Differences in Text: Diffusion vs AR (2026-04-04): https://arxiv.org/html/2605.12522v1
- A3: Any-Order Generation (2026-01-19): https://arxiv.org/html/2601.13228v1
- FastDiSS (2026): https://arxiv.org/pdf/2604.05551
