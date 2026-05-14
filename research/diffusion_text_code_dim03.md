## Facet: LLaDA2.1 — Token Editing and EBPO RL (Deep Dive)

---

### 1. T2T Editing Mechanism — Technical Details

**Key Findings:**

- LLaDA2.1 introduces **Token-to-Token (T2T) editing** as a core innovation, enabling the model to retrospectively correct already-committed tokens during the generation process. This is a departure from standard absorbing-state diffusion which only allows `[MASK]` → token transitions [^330^].

- The T2T mechanism operates through **dual probability thresholds** at each timestep t:
  - **Unmasking Set Γ_t**: Positions where `x_t^i = [MASK]` and `p_θ(v_t^i | x_t) > τ_mask`
  - **Editing Set Δ_t**: Positions where `x_t^i ≠ v_t^i` (current token differs from top candidate) and `p_θ(v_t^i | x_t) > τ_edit` [^164^]

- The state evolution is formalized as:
  ```
  x_{t-1}^i = v_t^i   if i ∈ Γ_t ∪ Δ_t
  x_{t-1}^i = x_t^i   otherwise
  ```
  This means the model can simultaneously unmask new positions AND edit already-visible tokens in the same forward pass [^164^].

- The T2T editing head was trained to recover from **uniformly random token perturbations** in the T2T training stream — different from the semantically plausible errors the model actually makes at inference [^307^].

- Three **structural failure modes** of T2T have been identified by follow-up work:
  1. **Correction inertia**: When the posterior is multimodal, no single alternative crosses the confidence threshold, so obvious errors survive editing [^181^]
  2. **Premature replacement**: T2T swaps a correct token for an incorrect one under incomplete/polluted context [^181^]
  3. **Positional lock-in**: T2T can replace visible tokens but cannot reopen positions for a longer span [^307^]

- Follow-up work (Token-to-Mask, T2M) proposes resetting suspicious tokens to `[M]` rather than overwriting them, improving accuracy by +13.33 points on AIME 2025 and +8.56 on CMATH [^307^].

**Verbatim excerpt:**
> "LLaDA2.1 repairs such errors with Token-to-Token (T2T) editing, which re-examines previously unmasked tokens and overwrites them when an alternative becomes sufficiently confident." [^307^]

---

### 2. M2T vs T2T — Differences, Usage, and Configuration

**Key Findings:**

- **Mask-to-Token (M2T)** is the standard diffusion operation: predicting tokens for masked positions. This is the "drafting" capability [^331^].

- **Token-to-Token (T2T)** is the novel editing operation: correcting already-revealed tokens by replacing them with better alternatives. This is the "editing" capability [^331^].

- **When each is used:**
  - M2T runs first to fill masked positions at each denoising step
  - T2T then re-examines all non-mask, non-prompt positions and overwrites those where the argmax differs from the current token with sufficient confidence [^308^]
  - The inner loop iterates until all masks are filled AND no further T2T edits are triggered, at which point generation advances to the next block [^308^]

- **Training alignment**: Both CPT and SFT use a **unified Mixture of M2T and T2T objective**:
  - **Drafting Stream (M2T)**: Model learns to predict correct tokens at masked positions
  - **Editing Stream (T2T)**: Model learns to recover original tokens from random noise perturbations [^421^]

- **Multi-turn Forward (MTF)** data augmentation exposes the model to diverse iterative editing scenarios during training, simulating the multi-round refinement that occurs at inference [^331^]

- **Q Mode defaults** (from T2M paper): `τ_m2t = 0.7`, `τ_t2t = 0.5`, block length B = 32, greedy decoding (temperature 0) [^308^]

**Verbatim excerpt:**
> "By consistently applying this dual-stream supervision from CPT through SFT, we ensure that LLaDA2.1 is fundamentally conditioned to function as both a fast drafter and a precise editor within a single parameter space." [^164^]

---

### 3. EBPO — Mathematical Formulation and Gradient Estimation

**Key Findings:**

- **EBPO (ELBO-based Block-level Policy Optimization)** is the first large-scale RL framework specifically tailored for diffusion LLMs [^330^]. It was developed by the Ant Group team and applied to a 100B-parameter diffusion model for the first time [^346^].

- **Core challenge**: Standard policy gradient methods require sequence-level log-likelihoods, which are intractable for diffusion models due to their non-autoregressive, parallel decoding nature [^346^].

- **EBPO solution**: Uses the **Evidence Lower Bound (ELBO)** as a tractable proxy for the true likelihood, estimating gradients through **block-level conditional probabilities** computed in parallel via **Vectorized Likelihood Estimation** [^331^].

- The EBPO objective maximizes a **clipped surrogate function** weighted by a probability ratio ρ (similar to PPO-style clipping), ensuring stable policy updates [^331^].

- Block-conditional log probabilities are **aggregated across discretized timesteps and blocks**, allowing efficient computation within a single forward pass [^331^].

- **Key difference from standard RL**: Instead of computing sequence-level log-likelihoods (impossible for dLLMs), EBPO operates at the **block level**, using the block-causal attention structure to compute tractable conditional probabilities [^334^].

- The RL training extends the **AReaL framework** with specialized likelihood estimation and advantage estimation protocols that leverage diffusion sampling, explicitly supporting both T2T and M2T modes [^164^].

**Verbatim excerpt:**
> "To address this problem, the Ant team proposed and adopted an ELBO-based Block-level Policy Optimization (EBPO) method, which is specifically designed and adapted for the editable decoding structure. More importantly, the team applied reinforcement learning to a diffusion model with hundreds of billions of parameters for the first time." [^346^]

---

### 4. Speed Mode vs Quality Mode — Thresholds and Benchmarks

**Key Findings:**

- LLaDA2.1 introduces **two configurable operational modes** governed by the dual thresholds (τ_M2T, τ_T2T) [^336^]:

- **Speedy Mode (S Mode)**:
  - Employs a **low mask threshold** τ_M2T to aggressively draft by filling many positions per step
  - Uses a **moderate editing threshold** τ_T2T to restrict edits to high-confidence swaps
  - Yields maximal **Tokens Per Forward (TPF)** and throughput
  - Example config: `threshold = 0.5`, `editing_threshold = 0.0` [^430^]
  - TPF: **5.93** for Flash model (vs. 3.08 for LLaDA2.0) [^424^]

- **Quality Mode (Q Mode)**:
  - Both thresholds are raised so only high-confidence actions are taken
  - `threshold = 0.7`, `editing_threshold = 0.5` [^430^]
  - TPF: **3.64** for Flash model (vs. 5.93 in S Mode)
  - Surpasses LLaDA2.0 benchmark scores on both mini and flash models [^164^]
  - `max_post_steps ≥ 5` (recommended: 16) [^430^]

- **Benchmark comparison** (Flash 100B model):

| Metric | S Mode | Q Mode | LLaDA2.0 |
|--------|--------|--------|----------|
| Avg Score | 72.34 | 73.54 | 72.43 |
| TPF | 5.93 | 3.64 | 3.08 |
| HumanEval+ TPS (quantized) | 892 | — | ~535 |

  [^428^]

- Moving from Q-Mode to S-Mode approximately **doubles TPF** while causing only a negligible (~0.1–0.2 absolute) average score drop [^336^]

- Domain-specific speed variation: **highest in code domain** (structured output), **lowest in instruction following** (open-ended generation) [^164^]

**Verbatim excerpt:**
> "LLaDA2.1 also made a bolder design: one model supports two modes: quality mode and extreme-speed mode... Users can switch between quality and extreme-speed modes with just one configuration according to their actual needs." [^346^]

---

### 5. Multi-Block Editing (MBE) — Cross-Block Revision

**Key Findings:**

- **Multi-Block Editing (MBE)** allows the model to **revisit and revise previously generated blocks** based on the content of newly decoded blocks, improving global consistency [^164^].

- **Basic operation**: Without MBE, decoding and editing are performed within a single block only — tokens are generated under threshold-based constraints, and local edits revise intermediate outputs before the block is finalized [^164^].

- **MBE extension**: After generating a new block, the model can look back at earlier blocks and apply edits based on the additional context from the newly decoded content [^334^].

- **Performance impact of MBE** (from Table 4 of paper):
  - MBE yields **consistent performance improvements** across benchmarks for both Flash and Mini variants
  - Cost: **modest reduction in throughput**
  - Gains are **particularly evident on reasoning and coding tasks**
  - Example: AIME 2025 Flash improves from 63.33 to 70.0 with MBE; LiveCodeBench Flash improves from 44.05 to 46.48 [^164^]

- **Average improvement**: With MBE, Flash avg score goes from 70.69 to 72.67; Mini avg from 57.63 to 58.24 [^164^]

- **Technical implementation**: MBE operates on the block-causal attention structure where previously generated blocks are frozen as context. MBE relaxes this by allowing edits to cross block boundaries [^334^].

**Verbatim excerpt:**
> "Multi-Block Editing (MBE) adds another dimension. By allowing the model to revisit previously decoded blocks based on newly generated context, MBE consistently improves scores on reasoning and coding benchmarks with only modest throughput reduction." [^334^]

---

### 6. Quantization Approach — 1,587 TPS Achievement

**Key Findings:**

- LLaDA2.1 adopts **per-block FP8 quantization** to balance inference speed and model accuracy [^164^].

- FP8 (8-bit floating point) reduces memory bandwidth requirements and increases compute throughput compared to FP16/BF16 precisions [^427^].

- **Speed results with quantization**:
  - **LLaDA2.1-mini**: 1,586.93 TPS peak on HumanEval+ (quantized) vs. 1,496.67 TPS (unquantized) [^164^]
  - **LLaDA2.1-flash**: 891.74 TPS peak on HumanEval+ (quantized) vs. 746.66 TPS (unquantized) [^164^]

- **Score impact of quantization is minimal**: On HumanEval+, the score change is only -0.61 points for mini with quantization [^336^]

- The per-block (rather than per-tensor) granularity of FP8 scaling is well-suited to MoE architectures, as different experts may have different dynamic ranges [^427^]

- **Comparison with competitors** (quantized):
  - LLaDA2.1-Flash: **892 TPS** on HumanEval+
  - Qwen3-30B-A3B: 240 TPS
  - Ling-flash-2.0: 257 TPS
  - LLaDA2.1 is **~3.5× faster** than comparable AR models [^424^]

**Verbatim excerpt:**
> "After quantization, LLaDA2.1-flash achieves a peak TPS of 891.74 on HumanEval+, while LLaDA2.1-mini reaches 1586.93 in peak TPS, demonstrating significant speed advantages." [^164^]

---

### 7. Inference Infrastructure — Alpha-MoE, FP8, SGLang

**Key Findings:**

- **Customized SGLang**: LLaDA2.1 uses a customized version of SGLang (developed jointly by Ant Group Team and SGLang Team) as the inference engine. Day-0 support for LLaDA 2.0 diffusion models was added to SGLang in December 2025 [^399^].

- **Alpha-MoE megakernel**: Integrates Alpha-MoE (from Aleph-Alpha), which **combines two FusedMoE computations into a single kernel**, reducing kernel launch overhead and improving memory locality [^164^].

- **Per-block FP8 quantization**: As detailed above, balances speed and accuracy [^164^].

- **Block-wise causal masked attention**: The key attention innovation for dLLMs:
  - Within a block: **bidirectional attention** (all positions attend to each other)
  - Across blocks: **strictly causal attention** (block j attends only to blocks 0...j)
  - This allows the KV cache for the entire long context to be computed in a **single forward pass** [^164^]
  - The attention mask is `M_atm = tril(1_Nk×Nk)` at the block level, expanded to token-level resolution [^308^]

- **Radix caching and batching support**: Enabled for block diffusion LLMs in SGLang, optimizing memory usage and throughput for concurrent requests [^164^].

- **Training infrastructure**:
  - CPT/SFT: Uses **dFactory** framework with dedicated optimized implementation for MTF stage [^164^]
  - RL: Extends **AReaL framework** with specialized protocols; uses **ASystem** for distributed orchestration; customized SGLang as rollout engine [^164^]

**Verbatim excerpt:**
> "We use a customized version of SGLang for inference. To further accelerate the inference speed, we integrate Alpha-MoE, a MoE megakernel that combines the two FusedMoE computations into one kernel, and adopt per-block FP8 quantization to balance the inference speed and model accuracy." [^164^]

---

### 8. Stuttering Artifacts — Definition, Causes, Mitigation

**Key Findings:**

- **Stuttering artifacts** are "n-gram repetitions where phrases or words loop on themselves" — a direct consequence of independent parallel sampling in diffusion models [^334^].

- These artifacts occur when the masking threshold is set **too aggressively low** (Speed Mode), causing the model to commit tokens with insufficient confidence [^430^].

- The phenomenon is analogous to **repetition loops** seen in early non-autoregressive text generation: when tokens are sampled independently without sufficient conditioning on each other, the model can enter repetitive patterns [^432^].

- **When stuttering occurs**:
  - Primarily in **general chat/open-ended generation** scenarios with aggressive S Mode settings [^164^]
  - Less common in **structured domains** (code, math) where T2T editing is particularly effective
  - More likely when `threshold` is very low (e.g., < 0.5) [^430^]

- **Mitigation strategies**:
  1. **Use Quality Mode** for open-ended generation: higher thresholds (0.7–0.95) reduce error rate [^430^]
  2. **T2T editing**: The self-correction mechanism catches and fixes many repetitive patterns [^334^]
  3. **MBE**: Cross-block refinement corrects inconsistencies that can lead to stuttering [^164^]
  4. **Temperature = 0.0**: Recommended by the LLaDA team for reliability; higher temperatures increase noise [^430^]
  5. **Post-processing**: `max_post_steps ≥ 5` allows additional editing passes for self-correction [^430^]

- **Trade-off acknowledgment**: The paper explicitly notes that "aggressively lowering the masking threshold τ_mask can quickly generate 'rough drafts'... the model's self-correction can partially alleviate the 'stuttering' artifacts" [^164^]

**Verbatim excerpt:**
> "Under very aggressive masking settings, the model can still produce what the paper calls 'stuttering' artifacts, essentially n-gram repetitions where phrases or words loop on themselves, a direct consequence of independent parallel sampling." [^334^]

---

### 9. Comparison with LLaDA2.0 — Specific Improvements

**Key Findings:**

- **Architecture continuity**: LLaDA2.1 keeps the same model sizes (16B Mini, 100B Flash), same MoE architecture, and minimal change of training data compared to LLaDA2.0 [^330^]. The focus is on **decoding versatility** over parameter scaling.

- **Key addition: T2T editing**: LLaDA2.0 only supported Mask-to-Token transitions. LLaDA2.1 adds Token-to-Token editing, enabling self-correction [^332^].

- **Dual-mode decoding**: LLaDA2.1 introduces configurable S Mode and Q Mode; LLaDA2.0 required separate model variants (e.g., CAP versions) for speed optimization, which incurred serious accuracy loss [^346^].

- **RL alignment**: LLaDA2.1 adds the EBPO-based RL stage, which LLaDA2.0 did not have [^330^].

- **MBE**: LLaDA2.1 introduces Multi-Block Editing for cross-block refinement [^164^].

- **Performance comparison** (Flash 100B):

| Metric | LLaDA2.0 | LLaDA2.1 S Mode | LLaDA2.1 Q Mode |
|--------|----------|-----------------|-----------------|
| Avg Score | 72.43 | 72.34 | 73.54 |
| TPF | 3.08 | 5.93 | 3.64 |
| HumanEval+ Score | 87.80 | 89.63 | 89.63 |

  [^428^]

- **Speed improvement**: LLaDA2.1-flash S Mode achieves **892 TPS** vs. LLaDA2.0-flash-CAP's **535 TPS** — a **1.67× speedup** on the same size model [^424^].

- **Q Mode surpasses LLaDA2.0**: Despite keeping model size constant, LLaDA2.1 Q Mode outperforms LLaDA2.0 on both mini and flash variants, showing that editing improves not just speed but also quality [^164^].

- **Inference infrastructure upgrade**: LLaDA2.1 adds Alpha-MoE megakernels, per-block FP8 quantization, and enhanced SGLang support — none of which were in LLaDA2.0 [^164^].

**Verbatim excerpt:**
> "Notice that LLaDA2.1 extends its previous version (LLaDA2.0) by prioritizing decoding versatility over mere parameter scaling or benchmark peaking. By keeping the model size constant and minimal change of training data, we prove that our novel editing scheme enables lightning-fast execution with minimal overhead." [^330^]

---

### 10. Self-Correction Capabilities — How T2T Enables Error Correction

**Key Findings:**

- **Exposure bias in dLLMs**: Once decoding errors occur, dLLMs tend to become increasingly conservative in subsequent steps, significantly slowing generation. Autoregressive models exhibit lower exposure bias and can self-correct through extended chain-of-thought reasoning [^400^].

- **T2T as self-correction primitive**: After every M2T step, each committed token is re-examined. If a different token's predicted probability exceeds `τ_t2t`, the committed token is overwritten [^181^].

- **Three mechanisms enabling self-correction**:
  1. **Dual-stream training**: The T2T training stream teaches the model to recover original tokens from random noise perturbations, building error-rectification capability [^421^]
  2. **MTF augmentation**: Multi-turn Forward data augmentation simulates iterative editing scenarios, so the model practices multiple rounds of refinement [^334^]
  3. **Configurable thresholds**: Users can tune `editing_threshold` and `max_post_steps` to control correction aggressiveness [^430^]

- **T2T correction example**: In the Heraclitus quote example from the paper, when the model initially generates "walks" but later sees "river" as context, T2T replaces "walks" with "steps" — correcting the famous quote [^400^].

- **Effectiveness**: T2T editing helps LLaDA2.1 match the accuracy of autoregressive models at similar scale [^181^]. However, follow-up work (T2M) shows that remasking suspect tokens can be more reliable than overwriting — repairing 41.3% of "last-mile corruption" errors on CMATH [^181^].

- **Limitation**: The paper acknowledges that "the RL stage and T2T editing mechanism currently operate separately. Future work aims to merge them, using RL to directly optimize self-correction behavior" [^334^].

**Verbatim excerpt:**
> "LLaDA2.1 addresses this by alternating two steps. Mask-to-Token (M2T) is the standard mask-filling step: at each iteration, the model commits the most confidently predicted tokens among currently masked positions. Token-to-Token (T2T) editing then re-examines visible tokens and overwrites those for which a different candidate exceeds a confidence threshold." [^307^]

---

### Major Players & Sources

| Entity | Role/Relevance |
|--------|---------------|
| **Ant Group / InclusionAI** | Primary developer and funder of LLaDA2.1; released models under Apache 2.0 license |
| **Zhejiang University** | Academic collaborator on the LLaDA2.1 research paper |
| **Westlake University** | Academic collaborator |
| **Southern University of Science and Technology** | Academic collaborator |
| **Aleph-Alpha** | Developer of Alpha-MoE megakernel used in LLaDA2.1 inference |
| **SGLang Team (LMSYS/UC Berkeley)** | Developed the customized inference engine for LLaDA2.1 |
| **Lin Yao (SJTU / Zhongguancun Academy)** | Author of T2M follow-up paper identifying T2T failure modes and proposing remasking |
| **AReaL Framework team** | Developed the RL training infrastructure extended for EBPO |
| **dFactory (InclusionAI)** | Training framework for CPT/SFT stages |

---

### Trends & Signals

1. **Diffusion LLMs moving from proof-of-concept to production**: LLaDA2.1 represents a transition from "diffusion LLMs are viable" (LLaDA 8B) to "diffusion LLMs are practical" (100B + 892 TPS) [^74^]

2. **Self-correcting generation as a paradigm**: The draft-and-edit approach (aggressive drafting + retroactive correction) is emerging as a viable alternative to autoregressive generation, especially for structured outputs [^334^]

3. **Speed-quality trade-off becoming configurable**: Rather than binary fast/slow choices, LLaDA2.1 turns the speed-quality spectrum into a user-configurable continuum via threshold tuning [^336^]

4. **RL for diffusion models becoming tractable**: EBPO demonstrates that policy optimization for dLLMs at 100B+ scale is possible, opening the door for RLHF-style alignment of diffusion models [^346^]

5. **Domain-specific diffusion advantages**: Coding tasks benefit most from parallel decoding + editing (892 TPS on HumanEval+), while open-ended generation remains more challenging [^164^]

6. **Follow-up research improving T2T**: The T2M work (May 2026) shows the community actively identifying and fixing T2T limitations, with +13.33 AIME improvement [^307^]

---

### Controversies & Conflicting Claims

1. **T2T replacement vs. T2M remasking**: The original LLaDA2.1 paper advocates for token overwriting (T2T), but follow-up work by Lin Yao et al. argues that remasking to `[M]` is fundamentally more reliable — avoiding context pollution and correction inertia. T2M achieves +13.33 on AIME vs. T2T baseline [^307^]. The conflict centers on whether detection and correction should be coupled (T2T) or decoupled (T2M).

2. **Speed numbers vs. practical utility**: While 892 TPS is impressive, some analyses note that "the inference infrastructure isn't something you spin up over a weekend" and that for many applications, "if your bottleneck isn't token-level throughput on structured tasks, LLaDA2.1 definitely isn't your next deployment" [^334^].

3. **Higher base error rates**: Despite editing, "diffusion language models still exhibit higher base error rates than autoregressive models. The editing mechanism compensates for this, but it's compensation, not elimination" [^334^].

4. **AR-shaped data bias**: Research on parallel decoding (NAP paper, Feb 2026) suggests that standard training data has strong sequential dependencies, which may limit how much genuine parallelism dLLMs can exploit regardless of decoding strategy [^429^].

---

### Recommended Deep-Dive Areas

1. **EBPO mathematical formulation**: The exact equations (Eq. 4 and 5 in the paper) for the clipped surrogate objective and block-conditional log probability aggregation warrant detailed study for replication and extension. The paper is at https://arxiv.org/abs/2602.08676 [^330^].

2. **T2M vs. T2T empirical comparison**: The Token-to-Mask work shows dramatic gains on math benchmarks. Understanding when remasking dominates replacement — and whether this can be integrated into LLaDA2.1's training — is a high-value direction.

3. **MBE mechanism details**: The exact algorithm for how MBE revisits previous blocks, what triggers cross-block edits, and the computational overhead model is not fully detailed in available sources.

4. **Alpha-MoE megakernel performance**: Understanding the exact speedup from fusing two FusedMoE computations, and how this scales with expert count and batch size.

5. **Per-block FP8 quantization recipe**: The specific scaling strategy (per-block vs. per-tensor, delayed vs. current) used in LLaDA2.1 and its accuracy/speed trade-off profile.

6. **Integration of RL and editing**: The paper explicitly identifies as future work the merger of EBPO RL with T2T editing optimization. This could be transformative for self-improving diffusion models.

7. **Domain-specific threshold tuning**: The observation that structured domains (code, math) tolerate aggressive S Mode while open-ended chat requires Q Mode suggests an adaptive thresholding mechanism could yield further gains.

---

### Source URLs

- **Primary paper**: https://arxiv.org/abs/2602.08676 (LLaDA2.1: Speeding Up Text Diffusion via Token Editing, Feb 2026) [^330^]
- **GitHub repository**: https://github.com/inclusionAI/LLaDA2.X [^407^]
- **Hugging Face collection**: https://huggingface.co/collections/inclusionAI/llada21 [^422^]
- **T2M follow-up paper**: https://arxiv.org/abs/2604.18738 (Remask, Don't Replace: Token-to-Mask Refinement, May 2026) [^307^]
- **LLaDA2.0 paper**: https://arxiv.org/abs/2512.15745 (Scaling Up Diffusion Language Models to 100B, Dec 2025) [^24^]
- **SGLang diffusion support**: https://lmsys.org/blog/2025-12-19-diffusion-llm/ [^399^]
- **Alpha-MoE**: https://aleph-alpha.com/alpha-moe-a-megakernel-for-faster-tensor-parallel-inference/ [^164^]
- **Tech report**: https://github.com/inclusionAI/LLaDA2.X/blob/main/llada2_1_tech_report.pdf [^422^]
