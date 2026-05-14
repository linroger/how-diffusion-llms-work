## Dimension 07: Technical Foundations — Discrete vs Continuous Diffusion, Remasking, Architectures (Deep Dive)

**Research Date:** 2026-05-15
**Searches Conducted:** 22 independent search queries across arXiv, OpenReview, official project pages, and technical blog posts
**Total Sources Consulted:** 70+ papers, blogs, and official documentation

---

### 1. LangFlow — First Continuous Diffusion to Rival Discrete

#### Key Findings

- **LangFlow** is the first continuous diffusion language model (DLM) to match or exceed state-of-the-art discrete DLMs on standard language modeling benchmarks [^529^](https://arxiv.org/pdf/2604.11748). It achieves **PPL 30.0 on LM1B** and **PPL 24.6 on OpenWebText** (OWT), surpassing all uniform-state discrete diffusion models and matching masked diffusion baselines.

- **Bregman Divergence Flow Matching:** LangFlow's core theoretical contribution is connecting embedding-space diffusion to Flow Matching via Bregman divergence. For any convex function $f$, the training objective becomes: $\mathcal{L}_f(\theta) = \mathbb{E}_{\gamma \sim \pi, z_\gamma} \left[ \frac{1}{L} \sum_{i=1}^{L} \mathcal{D}_f\left(\mathbf{1}_{x^{(i)}}, \hat{\mathbf{x}}_{\theta}^{(i)}(z_\gamma, \gamma)\right) \right]$ [^528^](https://arxiv.org/html/2604.11748v1). This provides a theoretically grounded cross-entropy loss that streamlines training.

- **ODE-based NLL Bound:** LangFlow derives a novel ODE-based upper bound of negative log-likelihood for evaluation, superseding previous SDE-based ELBOs and providing more accurate likelihood estimation for embedding-space DLMs [^173^](https://arxiv.org/html/2604.11748v2).

- **Information-Uniform Noise Schedule:** Through extensive profiling, LangFlow reveals that the optimal noise schedule for language follows a **Gumbel distribution** of the logarithmic noise-to-signal ratio $\gamma$ — a finding that "greatly differs from conclusions in the image generation domain" [^529^](https://arxiv.org/pdf/2604.11748). This motivates a learnable noise scheduler based on the Gumbel distribution.

- **Self-Conditioning Revelation:** LangFlow reveals an "underexplored discrepancy of self-conditioning between its effect on discrete diffusion and that on continuous diffusion," rectifying the training recipe for continuous diffusion language modeling [^173^](https://arxiv.org/html/2604.11748v2).

- **Zero-Shot Transfer:** LangFlow beats autoregressive baselines on 4 out of 7 zero-shot benchmarks (PTB, Lambada, PubMed, Arxiv) and beats MDLM on 3 out of 7 benchmarks (PTB, Wikitext, Lambada) [^539^](https://caradryanl.github.io/blog/2026/langflow/).

- **Gen. PPL Leadership:** On LM1B, LangFlow achieves Gen. PPL of 81.5, outperforming the best discrete baseline (DUO at 97.6) by more than 15 points [^539^](https://caradryanl.github.io/blog/2026/langflow/).

- **Variational Interpretation:** LangFlow also admits a variational interpretation closely related to Variational Flow Matching (VFM), making explicit the connection between Bregman divergence minimization in token space and variational inference [^528^](https://arxiv.org/html/2604.11748v1).

#### Architecture & Design
- Uses **learnable embeddings** (jointly trained token embedding matrix) as the continuous state space
- Applies **continuous-time Flow Matching** with Bregman divergence objective
- Employs **information-uniform noise scheduling** with Gumbel-distributed $\gamma$
- Incorporates **self-conditioning** with effects "substantially different from discrete diffusion"
- All diffusion happens in embedding space; no intermediate discretization steps

---

### 2. ELF (Embedded Language Flows) — He Kaiming's Team, Frozen Embedding Space

#### Key Findings

- **ELF** (Embedded Language Flows) is a continuous diffusion language model from **Kaiming He's team at MIT** that demonstrates continuous DLMs can be made effective "with minimal adaptation to the discrete domain" [^530^](https://arxiv.org/pdf/2605.10938). The core insight: stay entirely in continuous embedding space until the final timestep.

- **Frozen Embedding Space:** Unlike LangFlow (which uses jointly trained embeddings), ELF uses **frozen pretrained embeddings** from a T5 encoder. The denoising process operates entirely on these frozen representations — "predominantly stays within the continuous embedding space until the final time step, where it maps to discrete tokens using a shared-weight network" [^531^](https://arxiv.org/html/2605.10938v1).

- **10x Training Data Efficiency:** ELF achieves superior results while using **10x fewer training tokens** than comparable models (45B vs. 500B+ tokens). ELF-B (105M parameters) achieves Gen. PPL ~24.1 on OWT with just 32 sampling steps, surpassing 170M-parameter baselines trained on 10x more data [^532^](https://eu.36kr.com/en/p/3807465382190852) [^530^](https://arxiv.org/pdf/2605.10938).

- **Classifier-Free Guidance Compatibility:** Because ELF stays in continuous space throughout, it can directly adapt established image-domain diffusion techniques like **classifier-free guidance (CFG)** without modification — something discrete DLMs struggle with [^531^](https://arxiv.org/html/2605.10938v1).

- **Flow Matching Foundation:** ELF uses continuous-time Flow Matching with a linear (rectified-flow) interpolant [^531^](https://arxiv.org/html/2605.10938v1). This brings "flow-based training and sampling into language diffusion, allowing ELF to benefit from recent advances in Flow Matching."

- **Key Design Choices:** Per ELF's comprehensive survey of continuous DLMs in their Appendix A, ELF is the only method to simultaneously use: (1) continuous-time Flow Matching, (2) frozen pretrained encoder representations, (3) no per-step discretization during training or inference, and (4) no separate decoder [^531^](https://arxiv.org/html/2605.10938v1).

- **Authors:** Keya Hu*, Linlu Qiu*, Yiyang Lu, Hanhong Zhao, Tianhong Li, Yoon Kim, Jacob Andreas, Kaiming He (*equal contribution, MIT CSAIL) [^535^](https://github.com/libo-huang/kaiming-he-arxiv-papers).

- **Impact:** ELF was described as a "spiritual successor to SED (Self-conditioned Embedding Diffusion, 2022), bringing the approach into the modern era with significant improvements — much like LangFlow did for CDCD" [^536^](https://di.gg/ai/4gzqmawk?rank=16).

---

### 3. CoRe Remasking — Context-Robust Approach

#### Key Findings

- **CoRe** (Context-Robust Remasking) is a **training-free** framework that selects revision targets based on robustness to perturbations of the conditioning context, rather than static confidence scores [^182^](https://arxiv.org/html/2602.04096v1).

- **Core Mechanism:** Instead of ranking tokens by confidence, CoRe measures whether each token is "still strongly predicted when parts of its surrounding context are masked." A reliable token should remain strongly predicted under masked-context perturbations. CoRe performs a "lightweight stress test" by evaluating tokens under restricted masked-context perturbations and prioritizes those with the largest drop in support for revision [^182^](https://arxiv.org/html/2602.04096v1).

- **Efficiency:** CoRe achieves context-aware remasking with **~6% more forward passes** than standard decoding — a remarkably low overhead for the quality gains achieved [^182^](https://arxiv.org/html/2602.04096v1).

- **Code Generation Gains:** CoRe achieves the largest gains on structure-sensitive tasks: **+9.2% on MBPP** accuracy, "reducing syntax and logic inconsistencies where baselines fail" [^182^](https://arxiv.org/html/2602.04096v1).

- **Critical Finding on Confidence-Based Remasking:** CoRe reveals that "standard confidence-based remasking strategies (e.g., ReMDM) can **degrade code performance** in our experiments." This is a significant critique of confidence-based approaches [^182^](https://arxiv.org/html/2602.04096v1).

- **Framing as Robust Optimization:** CoRe frames revision as a "distributionally robust optimization problem" — identifying "context-brittle tokens" whose likelihood is not stable under context perturbations [^182^](https://arxiv.org/html/2602.04096v1).

- **Compute-Matched Validation:** The gains are validated by compute-matched controls, "where random or margin-based revision yields negligible gains," confirming that the context-robust selection criterion is doing meaningful work [^182^](https://arxiv.org/html/2602.04096v1).

---

### 4. STDD — Spatio-Temporal Dynamics, 8.9x Speedups

#### Key Findings

- **STDD** (Spatio-Temporal Dynamics-Driven Token Refinement) is a remasking approach that dynamically detects each token's **Temporal Variance** (convergence status) and **Spatial Deviance** (inter-token correlations) to adaptively adjust the confidence threshold for every token at every step [^527^](https://arxiv.org/html/2601.04205v1).

- **Speedups:** When integrated with LLaDA-Instruct-8B, STDD achieves:
  - **3.07x on GSM8K**
  - **8.9x on MBPP** (massive code generation acceleration)
  - **3.74x on MATH**
  
  When applied to Dream-7B: 3.41x (GSM8K), 2.91x (MBPP), 3.65x (MATH) [^527^](https://arxiv.org/html/2601.04205v1).

- **Quality Preservation:** Beyond efficiency, STDD demonstrates "either comparable or superior accuracy." On GSM8K, STDD achieves **83.1 accuracy** vs. baseline Confidence method (79.2), Fast-dLLM (79.2), and DUS (72.1). On MATH, STDD establishes a new state-of-the-art at **35.1** vs. baseline's 33.4 [^527^](https://arxiv.org/html/2601.04205v1).

- **Key Insight:** Mainstream remasking strategies rely on a "single global confidence threshold, overlooking the temporal–spatial dynamics of individual tokens." STDD addresses redundant iterations and constrained parallelism introduced by fixed-threshold remasking [^527^](https://arxiv.org/html/2601.04205v1).

- **MBPP Exceptional Performance:** The 8.9x speedup on MBPP "far surpasses Fast-dLLM's 4.15x and DUS's 2.70x, indicating a superior ability to accelerate code generation tasks" [^527^](https://arxiv.org/html/2601.04205v1).

---

### 5. RemeDi — Self-Reflective Remasking with RL, 89.1% GSM8K

#### Key Findings

- **RemeDi** (Remasking-enabled Diffusion Language Model) introduces a **self-reflective remasking approach** trained via a two-stage pipeline: Remask SFT followed by Remask RL. It achieves **89.1% on GSM8K**, 52.9% on MATH, 73.2% on HumanEval, and 59.4% on MBPP — state-of-the-art among open-source DLMs at the time [^186^](https://arxiv.org/html/2509.23653v1).

- **Two-Stage Training:**
  1. **Remask SFT:** The model learns to identify and remask incorrect tokens while predicting masked tokens. Input sequences are constructed by randomly masking tokens or replacing them with random alternatives to simulate noise. The noise schedule follows a criterion that noise level monotonically decreases over steps.
  2. **Remask RL:** Further fine-tuning with outcome-based reinforcement learning, optimizing entire generation trajectories toward higher rewards by learning how to remask and predict tokens in each step [^186^](https://arxiv.org/html/2509.23653v1).

- **Architecture:** RemeDi jointly predicts token distributions and per-token confidence scores. At each diffusion step, high-confidence tokens are unmasked while low-confidence ones are (re-)masked, regardless of whether they have been previously unmasked [^186^](https://arxiv.org/html/2509.23653v1).

- **Variable-Length Block-Wise Generation:** RemeDi adapts LLaDA into a DLM capable of variable-length generation. Generation proceeds block by block (L=32 tokens each), with each block undergoing a full reverse diffusion process. Block-wise causal masking enforces causality [^186^](https://arxiv.org/html/2509.23653v1).

- **General Task Performance:** 24.5% on AlpacaEval, 85.4% on IFEval, 87.7% on ARC-C [^186^](https://arxiv.org/html/2509.23653v1).

- **Unmasking Policy Stream (UPS):** RemeDi attaches a dedicated UPS to the base LLaDA architecture, trained with $\lambda_{UPS} = 0.3$. The UPS parameters use a higher learning rate ($2.0 \times 10^{-5}$) than original parameters ($2.0 \times 10^{-6}$) [^186^](https://arxiv.org/html/2509.23653v1).

- **Training Data:** Remask SFT uses constructed noisy sequences; Remask RL uses outcome-based rewards on the final generation quality.

---

### 6. PAPL — Planned ELBO, Training-Inference Mismatch, One-Line Fix

#### Key Findings

- **PAPL** (Planner-Aware Path Learning) identifies a fundamental mismatch: popular planner-guided inference paths (like greedy ancestral sampling) use non-uniform masking at inference, but standard denoiser training uses uniformly random masking. This creates a training-inference disconnect [^225^](https://arxiv.org/html/2509.23405v3).

- **P-ELBO (Planner-Aware Evidence Lower Bound):** PAPL derives a novel generalized lower bound that takes planning into account in the reverse dynamics of a DLM. The standard DLM ELBO is a special case of P-ELBO. Importantly, P-ELBO recovers all current planning strategies (MaskGIT, P2, etc.) as principled instances [^225^](https://arxiv.org/html/2509.23405v3).

- **One-Line Code Change:** PAPL's practical implementation amounts to a **single line code change** with **no additional computational overhead** from standard DLM training. It leverages the denoiser's own confidence to compute a weighted loss on more likely generation paths [^225^](https://arxiv.org/html/2509.23405v3).

- **Results:**
  - **Protein sequences:** 40% relative increase in foldability (59.40% vs. 42.43%), surpassing larger diffusion (DPLM-650M) and autoregressive (ProGen2-2.7B, ESM3) baselines while preserving diversity [^225^](https://arxiv.org/html/2509.23405v3) [^634^](https://arxiv.org/html/2509.23405v1).
  - **Code generation:** HumanEval pass@1 from 18.5 to 20.8 (+12.4%), pass@10 from 31.1 to 38.4, HumanEval-Infill pass@1 from 30.0 to 32.5 [^225^](https://arxiv.org/html/2509.23405v3).
  - **Text generation:** Up to **4x improvement in MAUVE** and reduces generative perplexity by over 40% [^225^](https://arxiv.org/html/2509.23405v3).

- **Protein Quality Metrics:** pLDDT: 81.48 (PAPL) vs. 81.32 (baseline); pTM: 0.72 vs. 0.65; pAE: 8.97 vs. 12.00 — all structural metrics improved while diversity (entropy 3.12, uniqueness 91.73%) remained on par [^225^](https://arxiv.org/html/2509.23405v3).

- **Generality:** PAPL is a unifying framework that can accommodate various planning functions (top probability margin, RDM, block-denoising, confidence thresholding) through properly adapted versions [^225^](https://arxiv.org/html/2509.23405v3).

- **Limitation:** The method requires some post-training to test whether a given decoding strategy's performance can be improved — "this training overhead makes our methodology expensive to implement with large planning models compared to just testing performance at inference time" [^225^](https://arxiv.org/html/2509.23405v3).

---

### 7. SCMDM Self-Conditioning — Full vs Partial, ~50% Perplexity Reduction

#### Key Findings

- **SCMDM** (Self-Conditioned Masked Diffusion Models) is a lightweight **post-training adaptation** for pretrained MDMs that conditions each denoising step on the model's own previous clean-state predictions. It requires only minimal architectural change and **no extra denoiser evaluations during sampling** [^575^](https://arxiv.org/html/2604.26985v1).

- **Dramatic Perplexity Reduction:** SCMDM achieves nearly a **50% reduction in generative perplexity** on OWT-trained models: from **42.89 to 23.72** (GPT2 evaluator at 1000 steps), using only 3.25B additional post-training tokens on a model pretrained for 262B tokens [^575^](https://arxiv.org/html/2604.26985v1) [^580^](https://arxiv.org/pdf/2604.26985).

- **Full vs. Partial Self-Conditioning:** SCMDM's key finding: in the post-training regime, **full self-conditioning consistently outperforms partial self-conditioning**. While partial (rate=0.5) improves over vanilla MDLM (Gen. PPL 42.89 -> 37.04), full self-conditioning achieves 23.72 — a massive additional improvement [^580^](https://arxiv.org/pdf/2604.26985).
  - "Once the base denoiser's self-generated clean-state estimates become informative, the specialization to refinement is preferable to mixing conditional and unconditional objectives."
  - This "reveals a refinement-specialization effect that has not previously been characterized" [^575^](https://arxiv.org/html/2604.26985v1).

- **Key Technical Insight:** In standard masked diffusion, if a token remains masked after a reverse update, the model discards its clean-state prediction for that position. Still-masked positions must be repeatedly inferred from the mask token alone. SCMDM carries this clean-state distribution forward, enabling still-masked positions to be refined across steps [^575^](https://arxiv.org/html/2604.26985v1).

- **Zero-Shot Improvements:** SCMDM also improves zero-shot perplexity across standard downstream corpora: PTB (101.71 -> 99.13), Wikitext (37.82 -> 36.48), Lambada (50.04 -> 49.09), AG News (70.60 -> 68.13), Pubmed (44.45 -> 41.72), Arxiv (39.14 -> 37.75) [^580^](https://arxiv.org/pdf/2604.26985).

- **LLM-as-Judge Improvement:** Mean Gemma-31B judge score shifts from 33.1 (MDLM) to 36.5 (SCMDM), "indicating improved generation quality" [^580^](https://arxiv.org/pdf/2604.26985).

- **Multi-Domain:** SCMDM works across natural language, discretized image synthesis (CIFAR-10 FID: 86.48 -> 78.59), small molecule generation, and genomic sequence modeling [^575^](https://arxiv.org/html/2604.26985v1).

- **No Additional Sampling Cost:** At inference, the clean-state prediction from step t+1 is reused as the self-conditioning input at step t — no extra forward passes needed [^575^](https://arxiv.org/html/2604.26985v1).

---

### 8. Fast-dLLM KV Cache — Block-Wise Approximate, 27.6x Throughput

#### Key Findings

- **Fast-dLLM** introduces a **block-wise approximate KV Cache** mechanism tailored for bidirectional diffusion models, combined with **confidence-aware parallel decoding**, achieving up to **27.6x throughput improvement** with minimal accuracy loss [^291^](https://openreview.net/forum?id=3Z3Is6hnOT) [^608^](https://openreview.net/pdf?id=3Z3Is6hnOT).

- **Core Problem:** Open-source Diffusion LLMs lack KV caching (standard in AR models) and suffer quality degradation when decoding multiple tokens simultaneously. LLaDA performs best when generating tokens one at a time and "soon degrades when decoding multiple tokens simultaneously" [^581^](https://arxiv.org/pdf/2505.22618).

- **Block-Wise Approximate KV Cache:**
  - Before generating a block, compute and store KV Cache for other blocks to reuse
  - Within each block, the same cache is reused for multiple decoding steps
  - After completing a block, update the cache for all tokens
  - "Visualizations confirm the high similarity with adjacent inference steps within the block" [^595^](https://arxiv.org/pdf/2505.22618).

- **Confidence-Aware Parallel Decoding:** Selectively decodes tokens whose confidence exceeds a dynamic threshold, rather than selecting top-K. This "mitigates dependency violations and maintains generation quality" while achieving up to **13.3x inference speed-up** [^608^](https://openreview.net/pdf?id=3Z3Is6hnOT).

- **Detailed Speedup Breakdown:**
  - GSM8K (8-shot, gen length 1024): 27.6x end-to-end — from 266 seconds to 12 seconds; 19.3 tok/s
  - Accuracy preserved: 76.0 vs. 77.3 (vanilla LLaDA)
  - HumanEval: 44.5% (actually +1.2% vs baseline) with 3.7x throughput [^605^](https://www.themoonlight.io/en/review/fast-dllm-training-free-acceleration-of-diffusion-llm-by-enabling-kv-cache-and-parallel-decoding) [^613^](https://view.inews.qq.com/a/20250530A05D2U00).

- **DualCache:** A bidirectional version that caches not only prefix tokens but also suffix tokens (consisting of masked tokens), providing further acceleration [^595^](https://arxiv.org/pdf/2505.22618).

- **Institutional Collaboration:** Joint work by NVIDIA, HKU, and MIT. Fast-dLLM v2 extends this to block diffusion with only ~1B tokens of fine-tuning [^606^](http://mp.weixin.qq.com/s?__biz=MzkyODMyNTAwMA==&mid=2247495209&idx=2&sn=98070dd450fef3fc156ec964aac636c3).

---

### 9. I-DLM Introspective Consistency — Causal Masking + Logit Shift, ~5B Token Conversion

#### Key Findings

- **I-DLM** (Introspective Diffusion Language Model) identifies that existing DLMs lack **introspective consistency** — the property that AR models have where generation and introspection (self-verification) agree. I-DLM is the **first DLM to match same-scale AR model quality** while substantially outperforming prior DLMs [^70^](https://arxiv.org/html/2604.11035v1).

- **Introspective Acceptance Rate:** I-DLM formalizes this as $\alpha = \frac{1}{N} \sum_k \min(1, p_k(x_k)/q_k(x_k))$, where $p$ is the causal anchor distribution and $q$ is the generation distribution. For AR models, $p = q$ by construction ($\alpha = 1$). SDAR achieves only 0.699, LLaDA 2.0-flash only 0.568 — indicating substantial divergence between what models generate and what they endorse [^70^](https://arxiv.org/html/2604.11035v1).

- **Efficient Conversion:** I-DLM converts Qwen3-8B to I-DLM-8B using only **4.5B tokens** on 8 H100 GPUs. This is **12x more token-efficient than SDAR**, which needs ~54B tokens for conversion and still yields much worse quality (10.0 vs. 69.6 on AIME-24) [^636^](https://arxiv.org/pdf/2604.11035) [^70^](https://arxiv.org/html/2604.11035v1).

- **Three Key Techniques:**
  1. **Causal attention with logit shift:** Clean positions produce the causal anchor $p$ (verify distribution), while masked positions produce tokens $q$ (decode distribution). The hidden state at position $i$ predicts token $i+1$ rather than token $i$, preserving the AR model's logits[i] -> token[i+1] mapping.
  2. **Introspective Strided Decoding (ISD):** Simultaneously generates new tokens and revises prior ones in the same forward pass. After step 1, every step produces both accepted tokens and fresh tokens in one forward pass — no additional verification cost.
  3. **AR-compatible serving stack:** Direct integration into AR serving systems (e.g., SGLang) with gated residual LoRA adapters [^70^](https://arxiv.org/html/2604.11035v1).

- **Performance (I-DLM-8B):**
  - AIME-24: 69.6 (vs. SDAR 10.0, LLaDA-2.1-mini 43.3)
  - MATH-500: 96.8 (vs. Qwen3-8B 95.8)
  - HumanEval: 93.3 (vs. Qwen3-8B 95.1 — within 2 points)
  - MBPP: 92.2 (vs. Qwen3-8B 93.4)
  - LiveCodeBench-v6: 45.7 (vs. Qwen3-8B 50.3) [^638^](https://github.com/Introspective-Diffusion/I-DLM).

- **I-DLM-32B surpasses LLaDA-2.1-flash (100B):** +16.7 on AIME-25, +11.7 on LiveCodeBench-v6 [^70^](https://arxiv.org/html/2604.11035v1).

- **Throughput:** I-DLM-8B achieves **3.1x higher throughput** than LLaDA-2.1-mini (16B) on MATH-500, and **4.0x higher throughput** than SDAR (8B) [^70^](https://arxiv.org/html/2604.11035v1).

- **Lossless Mode:** I-DLM supports a near-lossless mode via LoRA adapters (rank 128) where introspection relies on base model weights, making the output "bit-for-bit lossless with respect to the base AR model" [^70^](https://arxiv.org/html/2604.11035v1).

- **Training Insight:** I-DLM trains for 2 epochs with a stride curriculum: N=2 for epoch 1, N=3 for epoch 2. Cross-entropy loss on clean tokens uses reduced weight (0.2) during initial stride expansions to prioritize masked token learning [^70^](https://arxiv.org/html/2604.11035v1).

---

### 10. Scaling Laws — MDM-Prime-v2 21.8x Compute Efficiency Claims

#### Key Findings

- **MDM-Prime-v2** claimed to be **21.8x more compute-efficient than autoregressive models (ARM)**, achieving 7.77 perplexity on OWT vs. ARM's 12.99, MDM's 18.94, and MDM-Prime's 13.41. At 1.1B parameters, it demonstrated superior zero-shot accuracy on commonsense reasoning tasks [^223^](https://arxiv.org/html/2603.16077v1).

- **However, MDM-Prime-v2 was withdrawn** from arXiv on March 30, 2026 (v2 marked as withdrawn) [^576^](https://arxiv.org/abs/2603.16077) [^633^](https://arxiv.org/abs/2603.16077). The withdrawal reason is not publicly stated, raising questions about the validity of the 21.8x claim.

- **Binary Encoding + Index Shuffling:** MDM-Prime-v2's claimed innovation was incorporating binary encoding and index shuffling to address two limitations of MDM-Prime: (1) lack of tools to guide token granularity hyperparameter choice, and (2) function form of the subtokenizer degrading likelihood estimation with BPE tokenizers [^223^](https://arxiv.org/html/2603.16077v1).

- **MDM-Prime (v1) was valid and published:** The original MDM-Prime paper ("Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking") was published at NeurIPS 2025, achieving 15.36 PPL on OWT — outperforming ARM (17.54), MDM (21.52), and hybrid variants (17.58). It was the first MDM-based approach to surpass ARM without autoregressive formulation [^637^](https://arxiv.org/abs/2505.18495) [^641^](https://chen-hao-chao.github.io/mdm-prime/) [^642^](https://neurips.cc/virtual/2025/poster/116103).

- **Standard MDM Scaling (Pre-Prime):** The foundational scaling law work (Nie et al., 2025) showed that standard MDMs require **~16x more compute** than ARMs to match validation loss, with compute-optimal MDMs being ~2x smaller than AR counterparts [^622^](https://arxiv.org/html/2602.15014v1) [^623^](https://arxiv.org/pdf/2410.18514?).

- **Low-Variance Training Loss Improvement:** Using a low-variance training loss (while evaluating with the correct likelihood) improves MDM scaling to ~14x (instead of ~16x) more compute than AR — a 12% improvement [^622^](https://arxiv.org/html/2602.15014v1).

- **Duo (Uniform-State Diffusion) Scaling:** Duo requires ~23x more compute than AR to match perplexity, though it can offer faster inference via few-step generation enabled by self-correction [^622^](https://arxiv.org/html/2602.15014v1).

- **Data-Constrained Regime Finding:** A separate line of work (Swerdlow et al., 2025) found that when models train for multiple epochs on repeated data (data-constrained), **diffusion models consistently surpass AR models** in validation loss, suggesting the previously observed inefficiency is "largely a consequence of evaluating them solely in the single-epoch regime" [^589^](https://arxiv.org/html/2507.15857v7).

---

### Major Players & Sources

| Entity | Role/Relevance |
|--------|---------------|
| **LangFlow** (Chen et al., 2026) | First continuous DLM to rival discrete; Bregman divergence FM framework. arXiv:2604.11748 |
| **ELF / Kaiming He (MIT)** | Frozen embedding space continuous DLM; 10x training efficiency. arXiv:2605.10938 |
| **CoRe** (Context-Robust Remasking) | Training-free context-robust remasking; +9.2% MBPP. arXiv:2602.04096 |
| **STDD** | Spatio-temporal dynamics remasking; 8.9x speedup on MBPP. arXiv:2601.04205 |
| **RemeDi** | Self-reflective remasking with RL; 89.1% GSM8K. arXiv:2509.23653 |
| **PAPL** | Planner-aware ELBO; one-line code fix; 40% protein foldability gain. arXiv:2509.23405 |
| **SCMDM** (UVA) | Post-training self-conditioning; 50% perplexity reduction. arXiv:2604.26985 |
| **Fast-dLLM** (NVIDIA/HKU/MIT) | Block-wise KV cache; 27.6x throughput. arXiv:2505.22618 |
| **I-DLM** | Introspective consistency; first DLM to match AR quality; 4.5B token conversion. arXiv:2604.11035 |
| **MDM-Prime-v2** (withdrawn) | Claimed 21.8x compute efficiency; withdrawn March 2026. arXiv:2603.16077 |
| **LLaDA** (Nie et al.) | Foundational 8B masked diffusion LM; 2.3T training tokens. arXiv:2502.09992 |
| **DREAM 7B** (HKU/Huawei) | AR-initialized diffusion with CART noise rescheduling. arXiv:2508.15487 |
| **ReMDM** (Cornell/Kuleshov) | Principled remasking sampler; inference-time scaling. arXiv:2503.00307 |
| **Mercury Coder** (Inception Labs) | Commercial dLLM; 1000+ tokens/sec on H100. mercury.inceptionlabs.ai |
| **Gemini Diffusion** (Google DeepMind) | Experimental; 1479 tokens/sec. deepmind.google/models/gemini-diffusion/ |

---

### Trends & Signals

1. **Continuous diffusion is rapidly closing the gap with discrete.** LangFlow and ELF both demonstrate that continuous-space diffusion can now match or exceed discrete diffusion — a major reversal from just a year ago when discrete dominated. Both use different approaches (LangFlow: jointly trained embeddings + Bregman FM; ELF: frozen embeddings + standard FM) suggesting the paradigm shift is robust [^529^](https://arxiv.org/pdf/2604.11748) [^530^](https://arxiv.org/pdf/2605.10938).

2. **Remasking is the critical battleground for diffusion quality.** The diversity of remasking approaches (CoRe, STDD, RemeDi, ReMDM, confidence-based, random) and their significant quality impacts (+9.2% on MBPP, 8.9x speedups) indicate that inference-time strategy may matter as much as training for diffusion LMs [^182^](https://arxiv.org/html/2602.04096v1) [^527^](https://arxiv.org/html/2601.04205v1) [^186^](https://arxiv.org/html/2509.23653v1).

3. **Training-inference alignment is becoming a central theme.** PAPL's planner-aware ELBO, SCMDM's self-conditioning, I-DLM's introspective consistency, and FastDiSS's training-inference matching all address the same fundamental problem: diffusion models are trained under conditions that don't match how they're used at inference. The "one-line code change" nature of PAPL suggests these fixes can be remarkably simple yet impactful [^225^](https://arxiv.org/html/2509.23405v3) [^575^](https://arxiv.org/html/2604.26985v1) [^70^](https://arxiv.org/html/2604.11035v1) [^614^](https://arxiv.org/html/2604.05551v1).

4. **Post-training adaptation is extremely effective.** SCMDM achieves 50% perplexity reduction with just 3.25B post-training tokens. I-DLM matches AR quality with 4.5B tokens. Fast-dLLM v2 adapts AR models with only ~1B tokens. This suggests the diffusion community is converging on efficient conversion rather than from-scratch training [^575^](https://arxiv.org/html/2604.26985v1) [^70^](https://arxiv.org/html/2604.11035v1) [^669^](https://arxiv.org/abs/2509.26328).

5. **Commercial diffusion LMs are arriving.** Mercury Coder (1000+ tok/s), Gemini Diffusion (1479 tok/s), and the acquisition of Inception Labs signal that the industry sees diffusion as a viable alternative to autoregressive models for production deployment [^71^](https://www.inceptionlabs.ai/blog/introducing-mercury) [^272^](https://deepmind.google/models/gemini-diffusion/) [^85^](https://arxiv.org/html/2506.17298v1).

6. **Self-conditioning behaves differently across regimes.** SCMDM's finding that full self-conditioning beats partial in the post-training regime (but not necessarily from-scratch) is a significant design insight that challenges the conventional 50% dropout wisdom from continuous diffusion [^575^](https://arxiv.org/html/2604.26985v1).

7. **Block diffusion as the practical middle ground.** Fast-dLLM v2, BD3-LM, SDAR, and I-DLM all converge on block-wise diffusion (autoregressive across blocks, diffusion within blocks) as the most deployable architecture — it preserves KV caching, enables parallel generation, and requires minimal adaptation from AR models [^669^](https://arxiv.org/abs/2509.26328) [^70^](https://arxiv.org/html/2604.11035v1).

---

### Controversies & Conflicting Claims

1. **MDM-Prime-v2's 21.8x claim vs. withdrawal.** The paper claimed MDM-Prime-v2 was 21.8x more compute-efficient than ARMs, which would have been revolutionary. However, the paper was **withdrawn from arXiv** on March 30, 2026 without explanation [^576^](https://arxiv.org/abs/2603.16077). This casts serious doubt on the validity of the claim. The original MDM-Prime (v1, NeurIPS 2025) remains valid with more modest but still impressive results (PPL 15.36 on OWT, surpassing ARM's 17.54) [^637^](https://arxiv.org/abs/2505.18495).

2. **Diffusion vs. AR compute efficiency: 16x worse or competitive?** Nie et al. (2025) established that standard MDMs require ~16x more compute than ARMs to match validation loss [^622^](https://arxiv.org/html/2602.15014v1). However, MDM-Prime (v1) surpassed ARM PPL without AR formulation [^637^](https://arxiv.org/abs/2505.18495), and data-constrained regime studies show diffusion surpassing AR on repeated data [^589^](https://arxiv.org/html/2507.15857v7). The true efficiency relationship appears highly dependent on evaluation regime and model design.

3. **Confidence-based remasking can degrade performance.** CoRe explicitly finds that "standard confidence-based remasking strategies (e.g., ReMDM) can degrade code performance" [^182^](https://arxiv.org/html/2602.04096v1). This contradicts the widespread adoption of confidence-based approaches in LLaDA, Dream, and other leading DLMs, suggesting that different domains may need fundamentally different remasking strategies.

4. **Continuous vs. discrete: which is fundamentally better?** LangFlow claims to be "the first continuous DLM that exceeds discrete DLMs on multiple tasks" [^173^](https://arxiv.org/html/2604.11748v2), while ELF "substantially outperforms leading discrete and continuous DLMs" [^531^](https://arxiv.org/html/2605.10938v1). These are concurrent claims from competing approaches — LangFlow (Bregman FM, learned embeddings) vs. ELF (standard FM, frozen embeddings). The field has not yet converged on which design is superior.

5. **I-DLM's causal approach vs. full bidirectional attention.** I-DLM argues that moving closer to AR (causal attention, logit shift) is the right trajectory, achieving the first DLM-AR quality match [^70^](https://arxiv.org/html/2604.11035v1). This challenges the conventional wisdom that bidirectional attention is diffusion's key advantage. Other works (LLaDA, DREAM) maintain full bidirectionality. The community has not resolved this tension.

6. **Full vs. partial self-conditioning.** SCMDM's finding that full self-conditioning dominates partial in the post-training regime directly contradicts the "commonly used 50% dropout strategy" [^575^](https://arxiv.org/html/2604.26985v1). This challenges established practice from both continuous diffusion (where partial is standard) and from-scratch DLM training.

---

### Recommended Deep-Dive Areas

1. **Why was MDM-Prime-v2 withdrawn?** The 21.8x compute efficiency claim was extraordinary and the withdrawal without explanation is suspicious. Understanding what went wrong could reveal important limitations in subtoken diffusion approaches. Deep investigation into the original v1 paper's methodology and any errors discovered would be valuable.

2. **LangFlow vs. ELF design comparison:** These represent the two leading continuous diffusion paradigms (learned embeddings + Bregman FM vs. frozen embeddings + standard FM). A systematic comparison of their design choices, especially the embedding strategies and flow objectives, could establish best practices for continuous DLMs.

3. **Remasking strategy design space:** With at least 6 distinct approaches (confidence-based, CoRe, STDD, RemeDi, ReMDM, random), each with different tradeoffs across domains, the remasking design space is ripe for systematic characterization. CoRe's finding that confidence-based approaches can degrade code performance is particularly noteworthy.

4. **I-DLM's introspective consistency principle:** The finding that DLMs lack introspective consistency and that fixing this enables AR-level quality with minimal training data (4.5B tokens) is potentially transformative. Understanding the theoretical foundations and whether this principle extends to non-causal DLMs warrants deep investigation.

5. **PAPL's planner-aware training across domains:** PAPL achieves remarkable gains (40% protein foldability, 4x MAUVE, 23% code improvement) with a one-line code change. Understanding why planner-awareness helps so much in some domains (protein folding) more than others, and whether the principle extends to more complex planning functions, is a high-value direction.

6. **Practical deployment of diffusion LMs:** With Mercury (1000+ tok/s), Gemini Diffusion (1479 tok/s), and Fast-dLLM (27.6x acceleration), diffusion LMs are approaching production viability. Understanding the real-world tradeoffs between quality, latency, throughput, and cost across these different acceleration approaches is critical for adoption.

7. **Data-constrained regime advantages:** The finding that diffusion models "consistently surpass AR models in validation loss" in data-constrained settings with repeated exposures [^589^](https://arxiv.org/html/2507.15857v7) challenges the conventional 16x inefficiency wisdom. Understanding when and why diffusion benefits from repeated data could reshape training strategies.

8. **Self-conditioning in the post-training regime:** SCMDM's finding about full vs. partial self-conditioning represents a significant shift from established practice. Understanding why full self-conditioning works better post-training, and whether this applies to other model classes, could broadly improve diffusion model quality.

---

### Key Quantitative Summary

| Method | Key Metric | Value | Context |
|--------|-----------|-------|---------|
| LangFlow | PPL (LM1B) | 30.0 | Best among continuous DLMs |
| LangFlow | PPL (OWT) | 24.6 | Matches MDLM (23.2 gap) |
| ELF | Gen. PPL (OWT, 32 steps) | ~24.1 | 10x fewer training tokens |
| CoRe | MBPP improvement | +9.2% | Over confidence-based remasking |
| STDD | MBPP speedup | 8.9x | Over baseline LLaDA |
| RemeDi | GSM8K | 89.1% | SOTA open-source DLM |
| PAPL | Protein foldability | 59.40% | +40% vs. baseline (42.43%) |
| PAPL | HumanEval pass@1 | 20.8 | From 18.5 (+12.4%) |
| SCMDM | Gen. PPL reduction | 42.89→23.72 | ~50% reduction on OWT |
| Fast-dLLM | Throughput speedup | 27.6x | LLaDA, 1024 gen length |
| I-DLM | AIME-24 | 69.6% | vs. SDAR 10.0%, LLaDA-2.1 43.3% |
| I-DLM | Training tokens | 4.5B | vs. SDAR 54B (12x more) |
| MDM-Prime | PPL (OWT) | 15.36 | First MDM > ARM (17.54) |
| MDM-Prime-v2 | Claimed efficiency | 21.8x | **Paper withdrawn** |
| Mercury Coder | Inference speed | 1000+ tok/s | H100 hardware |
| Gemini Diffusion | Inference speed | 1479 tok/s | Production demo |

---

### Source Index

- [^528^](https://arxiv.org/html/2604.11748v1) LangFlow: Continuous Diffusion Rivals Discrete in Language Modeling (HTML)
- [^529^](https://arxiv.org/pdf/2604.11748) LangFlow PDF
- [^530^](https://arxiv.org/pdf/2605.10938) ELF PDF
- [^531^](https://arxiv.org/html/2605.10938v1) ELF HTML
- [^532^](https://eu.36kr.com/en/p/3807465382190852) 36kr: ELF article
- [^535^](https://github.com/libo-huang/kaiming-he-arxiv-papers) Kaiming He papers tracker
- [^536^](https://di.gg/ai/4gzqmawk?rank=16) MIT researchers introduce ELF
- [^539^](https://caradryanl.github.io/blog/2026/langflow/) LangFlow blog summary
- [^182^](https://arxiv.org/html/2602.04096v1) CoRe: Context-Robust Remasking
- [^527^](https://arxiv.org/html/2601.04205v1) STDD: Spatio-Temporal Dynamics-Driven Token Refinement
- [^186^](https://arxiv.org/html/2509.23653v1) RemeDi: Self-Reflective Remasking
- [^225^](https://arxiv.org/html/2509.23405v3) PAPL: Planner-Aware Path Learning
- [^575^](https://arxiv.org/html/2604.26985v1) SCMDM: Simple Self-Conditioning Adaptation
- [^580^](https://arxiv.org/pdf/2604.26985) SCMDM PDF
- [^291^](https://openreview.net/forum?id=3Z3Is6hnOT) Fast-dLLM OpenReview
- [^581^](https://arxiv.org/pdf/2505.22618) Fast-dLLM PDF
- [^595^](https://arxiv.org/pdf/2505.22618) Fast-dLLM PDF (duplicate)
- [^605^](https://www.themoonlight.io/en/review/fast-dllm-training-free-acceleration-of-diffusion-llm-by-enabling-kv-cache-and-parallel-decoding) Fast-dLLM review
- [^608^](https://openreview.net/pdf?id=3Z3Is6hnOT) Fast-dLLM OpenReview PDF
- [^613^](https://view.inews.qq.com/a/20250530A05D2U00) Tencent: Fast-dLLM article
- [^70^](https://arxiv.org/html/2604.11035v1) I-DLM: Introspective Diffusion Language Models
- [^636^](https://arxiv.org/pdf/2604.11035) I-DLM PDF
- [^638^](https://github.com/Introspective-Diffusion/I-DLM) I-DLM GitHub
- [^223^](https://arxiv.org/html/2603.16077v1) MDM-Prime-v2 (withdrawn)
- [^576^](https://arxiv.org/abs/2603.16077) MDM-Prime-v2 arXiv (withdrawn)
- [^637^](https://arxiv.org/abs/2505.18495) MDM-Prime (v1)
- [^641^](https://chen-hao-chao.github.io/mdm-prime/) MDM-Prime project page
- [^642^](https://neurips.cc/virtual/2025/poster/116103) NeurIPS 2025: MDM-Prime
- [^622^](https://arxiv.org/html/2602.15014v1) Scaling Beyond Masked Diffusion
- [^589^](https://arxiv.org/html/2507.15857v7) Diffusion Beats Autoregressive in Data-Constrained Settings
- [^666^](https://arxiv.org/html/2503.00307v2) ReMDM: Remasking Discrete Diffusion
- [^71^](https://www.inceptionlabs.ai/blog/introducing-mercury) Mercury announcement
- [^85^](https://arxiv.org/html/2506.17298v1) Mercury paper
- [^272^](https://deepmind.google/models/gemini-diffusion/) Gemini Diffusion
- [^614^](https://arxiv.org/html/2604.05551v1) FastDiSS: Few-step Match Many-step
- [^669^](https://arxiv.org/abs/2509.26328) Fast-dLLM v2
- [^670^](https://arxiv.org/html/2509.26328v1) Fast-dLLM v2 HTML
