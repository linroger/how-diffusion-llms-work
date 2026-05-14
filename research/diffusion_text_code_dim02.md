## Facet: LLaDA2.0 — Scaling Diffusion Language Models to 100B Parameters (Deep Dive)

---

### Key Findings

1. **First 100B diffusion language model successfully trained and open-sourced.** LLaDA2.0 (arXiv:2512.15745v2, Dec 2025) represents the first time discrete diffusion LLMs have been scaled to 100B total parameters, using a systematic conversion from auto-regressive base models rather than training from scratch [^24^][^162^].

2. **WSD 3-phase training achieves smooth AR-to-diffusion conversion.** The Warmup-Stable-Decay strategy progressively increases block size (1→4→32→64→4096), runs large-scale full-sequence diffusion training, then decays back to block size 32 for efficient inference. This avoids catastrophic forgetting while enabling stable optimization [^24^][^311^].

3. **Inference speed of 535 TPS — 2.1x faster than comparable AR models.** LLaDA2.0-flash-CAP achieves 535 tokens/second on benchmark tasks, compared to 256 TPS for Ling-flash-2.0 and 237 TPS for Qwen3-30B-A3B-Instruct-2507 under identical serving conditions (SGLang with TP8 on H20) [^389^][^403^].

4. **Strong benchmark parity with AR models, with advantages in coding and agent tasks.** Across 47 benchmarks, LLaDA2.0-flash averages 73.18 vs. Qwen3-30B-A3B-Instruct-2507's 73.60. It leads in HumanEval (94.51 vs 93.29), MBPP (88.29 vs 86.65), BFCL v3 (75.43 vs 73.19), and LiveCodeBench (42.29 vs 41.63) [^162^][^24^].

5. **Document-level attention mask prevents cross-document semantic contamination.** A specialized block-wise attention mask (combining block-diagonal, offset block-causal, and block-causal masks) ensures attention operates strictly within document boundaries, which is critical for stable bidirectional diffusion training on packed sequences [^24^][^184^].

6. **Full open-source release under Apache 2.0.** Model weights (16B and 100B), training code (dFactory), inference engine (dInfer), and technical reports are all publicly available on HuggingFace and GitHub [^390^][^374^].

7. **Top-k checkpoint merge improves generalization.** A post-training strategy that selects the best k checkpoints based on validation perplexity and averages their parameters, smoothing the parameter landscape and yielding more robust models than EMA alone [^24^][^310^].

8. **CAP training enables high-confidence parallel decoding.** The auxiliary confidence loss selectively minimizes entropy on correctly predicted tokens, sharpening the model's predictions and enabling threshold-based parallel decoding to accept more tokens per forward pass [^24^][^404^].

---

### Major Players & Sources

| Entity | Role/Relevance |
|--------|---------------|
| **Ant Group (InclusionAI Team)** | Primary developer and funder. Hosts the project at `github.com/inclusionAI`. Provides infrastructure, engineering (dInfer, dFactory), and most authors. [^106^] |
| **Renmin University of China** | Academic collaboration. Key figures: Ji-Rong Wen (renowned IR/NLP researcher), Chongxuan Li. Brings expertise in diffusion models and information retrieval. [^414^] |
| **Zhejiang University** | Academic collaboration. Key figures: Jiaqi Hu, Junbo Zhao (tech-leader). Contributes expertise in computer vision and multimodal ML. [^414^] |
| **Westlake University** | Academic collaboration. Key figures: Zhenzhong Lan (tech-leader), Zhanchao Zhou. Contributes expertise in NLP and representation learning. [^414^] |
| **HKUST** | Academic collaboration. Key figure: Xiaocheng Lu. Contributes expertise in systems and efficient ML. [^414^] |
| **SGLang/LMSYS** | Inference ecosystem partner. SGLang provided day-0 support for LLaDA2.0 block diffusion inference. Joint engineering for production deployment. [^389^][^399^] |
| **Ling Team (Ant Group)** | Developed the AR base models (Ling-mini-2.0, Ling-flash-2.0) that LLaDA2.0 converts from. Ling-V2 models open-sourced separately. [^397^] |

---

### Trends & Signals

- **Diffusion LLMs closing the gap with AR models at scale:** LLaDA2.0 demonstrates that diffusion models can not only match but exceed AR baselines in structured tasks (code, agent) while offering superior inference parallelism. The average score gap between LLaDA2.0-flash (73.18) and Qwen3-30B-A3B-Instruct-2507 (73.60) is only 0.42 points [^162^].

- **AR-to-diffusion conversion as a paradigm:** Rather than training diffusion models from scratch (which is data-inefficient and unstable), LLaDA2.0's conversion approach preserves pretrained AR knowledge while introducing diffusion capabilities. This is ~7x more compute-efficient than equivalent dense models [^433^].

- **Block diffusion enabling AR-style inference optimizations:** By reverting to block size 32 during inference, LLaDA2.0 can leverage KV-cache reuse, tensor parallelism, and other optimizations originally designed for AR models, making diffusion models production-deployable for the first time at 100B scale [^389^][^24^].

- **MoE architecture with extreme sparsity (1/32 activation):** Both LLaDA2.0 variants use MoE with 256 routed + 1 shared expert, activating only 8 experts per token. The flash model has 100B total but only 6.1B active parameters — competitive with ~40B dense models [^369^][^392^].

- **Ecosystem maturation (dFactory + dInfer + SGLang):** Ant Group has built a complete toolchain — dFactory for training, dInfer for inference, and SGLang integration for serving — making diffusion LLMs practically accessible [^374^][^427^].

- **Follow-on work (LLaDA2.1, LLaDA2.0-Uni):** Released Feb 2026, LLaDA2.1 adds token editing to the diffusion process, achieving 891 TPS on flash and 1587 TPS on mini. LLaDA2.0-Uni extends to multimodal understanding and generation [^398^][^436^].

---

### Controversies & Conflicting Claims

1. **Exact TPS figures vary across sources.** The technical report claims 535 TPS for LLaDA2.0-flash-CAP [^24^][^403^], while the LMSYS blog reports ~500 TPS [^389^], and some third-party sources cite slightly different numbers. These discrepancies likely reflect different serving configurations (batch size, hardware tuning) rather than fundamental disagreement. All sources agree on the 2.1x speedup over AR baselines.

2. **Preview vs. final model performance gap.** The preview models (trained without full post-training) show significant performance deficits: LLaDA2.0-flash-preview scores only 23.33 on AIME 2025 vs. 60.00 for the final model, and 29.07 vs. 42.29 on LiveCodeBench [^162^]. This highlights that post-training (SFT + CAP + DPO) is responsible for a large portion of the final model's capabilities, not just the base diffusion conversion.

3. **Comparison baseline choices.** The paper compares against Qwen3-30B-A3B-Instruct-2507 and Ling-flash-2.0 but does not compare against the strongest AR models (e.g., GPT-4o, Claude). Some critics may argue this is an unfair comparison. However, the paper's stated goal is to demonstrate parity with strong open-source AR models of comparable scale, not to surpass closed frontier models.

4. **Long-context degradation at 64k.** While native 32k performance is strong (LLaDA2.0-flash maintains >93 RULER score), extending to 64k via YaRN scaling shows "predictable performance cost" [^162^]. This suggests the model is not natively trained on 64k contexts and relies on extrapolation.

5. **CAP model's slight quality-speed tradeoff.** CAP training improves TPS but the paper notes a small performance cost on some tasks. The LLaDA2.0-mini-CAP scores 70.90 on BFCL v3 vs. 74.11 for the non-CAP preview [^162^], suggesting the confidence sharpening may introduce some rigidity in predictions.

---

### Recommended Deep-Dive Areas

- **WSD training dynamics**: The exact data volume and training steps for each block-size transition are not fully specified. Understanding the compute budget allocation across warmup/stable/decay phases would help reproduce and optimize the method.

- **Top-k checkpoint merge mechanism**: While the paper cites Tian et al. (2025) WSM scheduler, the exact selection criteria (which k, based on what validation metrics) and the performance gain attributable to merging vs. single checkpoint are not quantified in detail.

- **Training compute and cost**: The paper does not specify exact GPU hours, cluster configuration, wall-clock training time, or dollar cost. This information would enable meaningful comparison with training from scratch and with other AR models.

- **DPO for diffusion models**: The use of ELBO as a substitute for log-likelihood in DPO is novel but the paper does not provide extensive ablation studies comparing it with alternatives like VRPO (from LLaDA 1.5).

- **Complementary masking effectiveness**: The paper notes complementary masking only works on corpus <100B tokens [^184^], which limits its applicability. Understanding why it fails at scale would be valuable.

- **Community adoption metrics**: Download counts, community forks, and production deployments are not publicly tracked. Quantifying real-world adoption would strengthen claims about practical viability.

- **Comparison with other dLLMs**: The paper focuses on AR baselines. A detailed comparison with other diffusion LLMs (Dream, LLaDA 1.5, Seed-Diffusion) at similar scales would better position LLaDA2.0 within the dLLM landscape.

---

## Detailed Technical Analysis

### 1. WSD Training Dynamics

The Warmup-Stable-Decay (WSD) strategy is the core training innovation of LLaDA2.0. It decomposes the AR-to-diffusion conversion into three coordinated phases:

**Phase 1: Warmup (Progressive Block Size Expansion)**
- Starting from the AR base model (block size = 1, equivalent to standard autoregressive generation)
- Block sizes: **1 → 4 → 32 → 64 → 4096** [^24^][^311^]
- At each transition, the model is trained on "moderate-scale data" to ensure smooth adaptation
- "Each block-size transition is trained on moderate-scale data to ensure smooth adaptation. This progressive enlargement allows the model to smoothly adapt its internal representations to handle larger contextual spans and more complex masking patterns." [^24^]
- When block size reaches 4096, the BDLM becomes equivalent to a standard MDLM (full-sequence bidirectional denoising)

**Phase 2: Stable (Large-Scale MDLM Training)**
- Block size fixed at 4096
- "Large-scale corpora" training to deepen understanding of diffusion dynamics
- "The 'clean' part of the attention computation no longer needs to be maintained. This significantly reduces the computational cost of attention" [^24^]
- This is where the bulk of the training compute is invested

**Phase 3: Decay (Block Size Reduction for Inference)**
- Gradually reduce block size: **4096 → 2048 → ... → 32** [^24^]
- "This decay process distills the global contextual knowledge learned during MDLM into a compact blockwise structure"
- Final block size of 32 chosen as optimal tradeoff between quality and speed (ablation shows block size 16 gives highest score 70.26 but slowest 2.44 TPF; block size 32 gives 70.15 score with 2.55 TPF; block size 64 degrades both) [^162^]

**Training Stability Mechanism:**
- During AR-to-diffusion transition, gradient explosion can occur at high mask ratios because masked token embeddings decay to zero during AR training (masked tokens are never observed)
- Solution: Add independent Gaussian noise to the embedding layer output for masked tokens during initial iterations
- "This ensures that the L2 norm of the masked token's embedding remains significant to avoid gradient explosion, thereby stabilizing the training process" [^24^]
- This is done instead of randomly reinitializing masked token embeddings, which would cause catastrophic forgetting

### 2. Document-Level Attention Mask

The document-level attention mask is critical for handling packed heterogeneous documents during training. Without it, standard attention would incorrectly attend across document boundaries, causing semantic confusion.

**Mathematical Formulation:**

For a concatenated sequence x_full of length 2L (comprising x_t followed by x_0), with block size L_B and block index b(k) = ⌊k/L_B⌋, the attention mask M ∈ {0,1}^{2L×2L} is:

```
M_ij = {
  1_{b(i)=b(j)}        if i ∈ x_t and j ∈ x_t    (block-diagonal within noisy)
  1_{b(i)>b(j-L)}      if i ∈ x_t and j ∈ x_0    (offset block-causal cross-attention)
  1_{b(i-L)≥b(j-L)}    if i ∈ x_0 and j ∈ x_0    (causal block attention within clean)
  0                    otherwise                   (prevent x_0 → x_t attention)
}
```

The three mask components are:
- **M_BD (Block-Diagonal):** Within the noisy sequence x_t, tokens only attend to others in the same block
- **M_OBC (Offset Block-Causal):** From noisy x_t to clean x_0, attention is allowed only to earlier blocks
- **M_BC (Block-Causal):** Within the clean sequence x_0, blocks can attend to themselves and all preceding blocks [^24^][^184^]

**For MDLM (block size = 4096, full sequence), the mask simplifies to:**
```
M_ij = {
  1, if i,j belong to the same document
  0, otherwise
}
```

**Impact:**
- The paper compared document-level attention mask with random-length and CART techniques, and found it "more fundamental in CPT training compared to these techniques, and it consistently achieves superior performance" [^93^]
- Crucial for preventing cross-document contamination and ensuring coherent bidirectional modeling

### 3. Top-k Checkpoint Merge

Based on the WSM (Warmup-Stable-Merge) scheduler by Tian et al. (2025) [^310^]:

**Mechanism:**
1. After BDLM pre-training completes, identify the top k best-performing checkpoints based on validation metrics (typically perplexity)
2. Arithmetically average the parameters (weights and biases) of these k checkpoints
3. The merged model serves as the initialization for post-training

**Key Properties:**
- **Optimizer-agnostic:** Unlike EMA which requires integration into the training loop, checkpoint merge is a purely offline procedure
- **Explicit ensemble:** "It explicitly selects and averages distinct, high-performing model states, consolidating their strengths rather than merely smoothing the final training step" [^24^]
- **Different from EMA:** "While EMA is an in-training technique that continuously smooths parameters, merging is an offline procedure" [^24^]
- Smoothes the parameter landscape, mitigates overfitting, and yields a more stable and generalizable model

**Limitations:**
- The exact value of k is not specified in the paper
- Performance gain from merging vs. single checkpoint is not quantified in the ablation studies

### 4. CAP (Confidence-Aware Parallel Training)

CAP training is the key technique enabling efficient parallel decoding.

**Core Idea:**
Standard SFT loss (L_SFT) ensures correctness but provides diminishing incentive to sharpen the predictive distribution for tokens already correctly predicted. The confidence loss (L_conf) addresses this by:
- Selectively minimizing the entropy of the model's output distribution p_θ(x_0 | x_t, c)
- Applied only to tokens that are correctly predicted in a given step

**Objective Function:**
```
L(θ) = L_SFT(θ) + λ * L_conf(θ)
```

Where λ is a hyperparameter balancing the two objectives [^24^][^404^].

**Impact on Inference Speed:**
- LLaDA2.0-flash (standard): 383 TPS
- LLaDA2.0-flash-CAP: 535 TPS (+40% throughput)
- Compared to AR baselines under identical serving (SGLang TP8 on H20):
  - Ling-flash-2.0: 256 TPS
  - Qwen3-30B-A3B-Instruct-2507: 237 TPS
  - **LLaDA2.0-flash-CAP achieves up to 2.1x speedup** [^389^][^403^]

**LMSYS Blog Measurements:**
- All numbers collected under consistent serving environment (SGLang with TP8 on H20)
- LLaDA2.0-flash-CAP: 500 TPS
- Standard LLaDA2.0-flash: 383 TPS
- AR baselines: 258 TPS and 237 TPS
- 1.9x speedup over AR baselines with small batch sizes [^389^]

### 5. Comparison with AR Baselines

#### LLaDA2.0-mini (16B) vs. Ling-mini-2.0 (16B)

| Benchmark | Qwen3-8B | Ling-mini-2.0 | LLaDA2.0-mini-preview | LLaDA2.0-mini |
|-----------|----------|---------------|----------------------|---------------|
| **Average** | 63.42 | 65.77 | 54.67 | **64.34** |
| MMLU | 80.94 | 82.15 | 72.49 | 80.53 |
| MMLU-Pro | 65.48 | 63.72 | 49.22 | 63.22 |
| GPQA | 46.59 | 56.80 | 23.74 | 47.98 |
| BBH | 79.48 | 83.70 | 70.64 | 78.21 |
| HumanEval | 84.76 | 85.98 | 80.49 | **86.59** |
| MBPP | 78.92 | 84.07 | 77.75 | 81.50 |
| GSM8K | 93.63 | 94.62 | 89.01 | 94.24 |
| MATH | 86.28 | 94.66 | 73.50 | 93.22 |
| AIME 2025 | 22.08 | 47.66 | 10.00 | 36.67 |
| IFEval | 86.90 | 76.16 | 62.50 | 80.78 |

Key: LLaDA2.0-mini (64.34) closely approaches Ling-mini-2.0 (65.77), demonstrating fundamental viability of the diffusion approach. LLaDA2.0-mini leads in coding (HumanEval 86.59 vs 85.98) and instruction following (IFEval 80.78 vs 76.16) [^162^].

#### LLaDA2.0-flash (100B) vs. Ling-flash-2.0 (100B) and Qwen3-30B-A3B

| Benchmark | Qwen3-30B | Ling-flash-2.0 | LLaDA2.0-flash-preview | LLaDA2.0-flash |
|-----------|-----------|---------------|----------------------|---------------|
| **Average** | 73.60 | 72.15 | 65.97 | **73.18** |
| MMLU | 87.13 | 87.98 | 83.15 | 87.69 |
| HumanEval | 93.29 | 85.98 | 88.41 | **94.51** |
| MBPP | 86.65 | 85.01 | 86.65 | **88.29** |
| LiveCodeBench | 41.63 | 44.11 | 29.07 | **42.29** |
| GSM8K | 96.36 | 95.45 | 95.75 | 96.06 |
| MATH | 96.70 | 96.10 | 83.52 | 95.44 |
| AIME 2025 | 61.88 | 55.89 | 23.33 | **60.00** |
| BFCL v3 | 73.19 | 67.57 | 74.86 | **75.43** |

Key: LLaDA2.0-flash achieves clear parity with Qwen3-30B-A3B (73.18 vs 73.60) and leads in coding, agent, and math tasks. The preview model's massive gap on AIME (23.33 vs 60.00) highlights the importance of full post-training [^162^].

### 6. Full 47 Benchmark Results

**Evaluation Dimensions:** [^371^]
- **Knowledge (10):** MMLU, MMLU-Pro, CMMLU, C-Eval, GAOKAO-Bench, ARC-c, GPQA, SciBench, PHYBench, TriviaQA
- **Reasoning (12):** BBH, BBH Extra Hard, BBH-CN, MuSR, ZebraLogic, PrOntoQA, PIQA, OCNLI, HellaSwag, KOR-Bench, DROP, SQuAD 2.0
- **Coding (13):** CRUXEval, MBPP, MBPP+, MultiPL-E, HumanEval, HumanEval+, HumanEvalFix, HumanEval-CN, BigCodeBench, LiveCodeBench, Aider, Spider, BIRD-SQL
- **Math (9):** GSM8K, MATH, OlympiadBench, AIME 2025, Omni-MATH, HARDMath2, GSM-Plus, CMATH
- **Agent & Alignment (4):** IFEval, BFCL v3, CodeIF-Bench, Nexus FC

**LLaDA2.0-flash standout results:**
- **Coding dominance:** HumanEval 94.51 (best among all compared models), MBPP 88.29, HumanEval-CN 89.02
- **Math excellence:** GSM8K 96.06, MATH 95.44, CMATH 96.90
- **Agent leadership:** BFCL v3 75.43 (surpasses all AR baselines)
- **Areas of weakness:** HARDMath2 (4.27), SciBench (4.13) — these are extremely difficult benchmarks where all models struggle

### 7. Training Compute and Infrastructure

**Pre-training:**
- **Backend:** Megatron-LM with 5D parallelism: DP (Data Parallelism), PP (Pipeline Parallelism), TP (Tensor Parallelism), CP (Context Parallelism), EP (Expert Parallelism) [^24^][^415^]
- **Attention optimization:** cuDNN backend achieves >1.3x end-to-end speedup and >90% memory savings in attention layer vs. unfused TransformerEngine attention [^24^]
- **Load balancing:** Zig-zag partitioning strategy for block diffusion attention mask across CP group [^24^]
- **Masked token consistency:** Generated on a single model-parallel rank and broadcast to all other ranks within MP ranks [^24^]

**Post-training:**
- **Framework:** dFactory built on VeOmni distributed training framework [^24^]
- **Parallelism:** DP + EP
- **Data packing:** Multiple short sequences concatenated into longer sequences for throughput [^24^]

**Notable gaps in public information:**
- Exact GPU hours not disclosed
- Exact cluster size not disclosed
- Total training data volume not specified
- Wall-clock training time not reported
- Dollar cost not reported

However, the Ling base models (which LLaDA2.0 converts from) were trained on 20T+ tokens [^392^], suggesting the conversion process requires significantly less data than full pre-training from scratch.

### 8. MoE Architecture Details

**Architecture Parameters (confirmed via vLLM plugin):** [^369^]
```
num_experts: 256 (routed experts)
num_experts_per_tok: 8 (top-k selected)
num_shared_experts: 1 (always active)
moe_intermediate_size: 512
n_group: 8 (expert groups)
topk_group: 4 (groups to select)
routed_scaling_factor: 2.5
```

**Ling 2.0 MoE Design:** [^392^][^433^]
- **Activation ratio:** 1/32 (256 experts, 8 activated + 1 shared)
- **Ling-mini-2.0:** 16B total, 1.4B active (789M non-embedding)
- **Ling-flash-2.0:** 100B total, 6.1B active (4.8B non-embedding)
- **Routing:** Sigmoid-based, aux-loss-free
- **Additional features:** MTP layers, QK-Norm, Partial-RoPE
- **Efficiency claim:** 7x equivalent dense performance — Ling-mini with 1.4B active matches 7-8B dense; Ling-flash with 6.1B active matches ~40B dense [^433^][^392^]

**Conversion:** Training scripts require model weights in "merged-expert" format. Conversion scripts are provided in dFactory to merge separate expert weights into unified checkpoints [^374^].

### 9. Open-Source Availability

**HuggingFace Collection:** https://hf.co/collections/inclusionAI/llada-20 [^390^][^438^]

**Model Variants:** [^390^]
| Model | Description |
|-------|-------------|
| `inclusionAI/LLaDA2.0-mini` | 16B MoE instruction-tuned |
| `inclusionAI/LLaDA2.0-flash` | 100B MoE instruction-tuned |
| `inclusionAI/LLaDA2.0-mini-CAP` | CAP-enhanced for speed |
| `inclusionAI/LLaDA2.0-flash-CAP` | CAP-enhanced for speed |
| `inclusionAI/LLaDA2.1-mini` | Next-gen with token editing |
| `inclusionAI/LLaDA2.1-flash` | Next-gen with token editing |

**Code Repositories:**
- **Main repo:** https://github.com/inclusionAI/LLaDA2.X (Apache 2.0) [^106^]
- **Training:** https://github.com/inclusionAI/dFactory [^374^]
- **Inference:** https://github.com/inclusionAI/dInfer [^427^]
- **SGLang integration:** https://github.com/sgl-project/sglang/issues/12766 [^411^]
- **Multimodal extension:** https://github.com/inclusionAI/LLaDA2.0-Uni [^436^]

**Also available on:** ModelScope (for China-based users) [^406^]

**License:** Apache 2.0 [^390^][^408^]

**vLLM Plugin:** https://github.com/vllm-project/dllm-plugin supports production inference with TP, block diffusion, and custom decoding algorithms [^369^]

### 10. Collaboration Model

**Full Author List (31 authors, alphabetical order by last name):** [^414^]

**Ant Group (Affiliation 1):** Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Zenan Huang, Chengxi Li, Jianguo Li (tech-leader), Zehuan Li, Huabin Liu, Lin Liu, Guoshan Lu, Yuxin Ma, Jianfeng Tan, Lanning Wei, Yipeng Xing, Xiaolu Zhang, Jun Zhou, Junlin Zhou, Liwang Zhu, Yihong Zhuang

**Renmin University of China (Affiliation 2):** Chongxuan Li, Ji-Rong Wen

**Zhejiang University (Affiliation 3):** Jiaqi Hu, Junbo Zhao (tech-leader)

**Westlake University (Affiliation 4):** Zhenzhong Lan (tech-leader), Zhanchao Zhou

**HKUST (Affiliation 5):** Xiaocheng Lu

**Tech Leaders (†):** Zhenzhong Lan, Jianguo Li, Junbo Zhao, Da Zheng

**Key Contributors to SGLang Integration:** Tiwei Bie (tiwei.btw@antgroup.com), Zehuan Li, Mingliang Gong, Zenan Huang, Kun Chen, Ling Liu [^411^]

**Engineering Ecosystem:**
- dFactory: Training framework built on VeOmni [^374^]
- dInfer: Custom inference engine adapted for block diffusion [^427^]
- SGLang: Open-source inference framework with day-0 LLaDA2.0 support [^389^]

---

## Appendix: Inference Configuration

**Recommended settings for LLaDA2.0 models:** [^162^]
- Temperature: 0.0
- Block size: 32 (optimal quality-speed tradeoff)
- Decoding threshold: 0.95 (maximizes quality)

**Optimal hyperparameter ablation results:**
- Threshold 0.95: Highest score (70.15), lowest TPF (2.55)
- Threshold 0.85: Peak speed (3.31 TPF) but score drops to 67.90
- Block size 16: Highest score (70.26) but slowest (2.44 TPF)
- **Block size 32: Best balance — 70.15 score, 2.55 TPF** [^162^]

**Serving configuration (from LMSYS):** [^389^]
- SGLang with TP8 on 8x H20 GPUs
- Block diffusion LLM framework
- Full KV cache support
- Tensor parallelism
- CUDA graph optimization
- Streaming I/O

---

## Appendix: Key Links

| Resource | URL |
|----------|-----|
| arXiv Paper | https://arxiv.org/abs/2512.15745 |
| HuggingFace Collection | https://hf.co/collections/inclusionAI/llada-20 |
| GitHub (LLaDA2.X) | https://github.com/inclusionAI/LLaDA2.X |
| GitHub (dFactory) | https://github.com/inclusionAI/dFactory |
| GitHub (dInfer) | https://github.com/inclusionAI/dInfer |
| SGLang RFC | https://github.com/sgl-project/sglang/issues/12766 |
| LMSYS Blog | https://lmsys.org/blog/2025-12-19-diffusion-llm/ |
| Technical Report PDF | https://github.com/inclusionAI/LLaDA2.0/blob/main/tech_report.pdf |
| LLaDA2.1 (Successor) | https://arxiv.org/abs/2602.08676 |
