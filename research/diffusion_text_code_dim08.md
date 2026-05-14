## Dimension 08: RL and Post-Training for Diffusion Models (Deep Dive)

> **Scope:** Reinforcement learning and post-training algorithms for diffusion language models (dLLMs), including VRPO, Coupled-GRPO, EBPO, Blockwise SFT, TreeDiff, UniGRPO, TraceRL, and related methods. Covers mathematical formulations, empirical results, scaling properties, and future directions.

---

### Key Findings

1. **VRPO (LLaDA 1.5): Theoretical Foundation for Variance Reduction in Diffusion RL**
   - VRPO addresses the fundamental challenge that ELBO-based likelihood estimates in diffusion models introduce high variance and bias into preference optimization gradients [^207^].
   - The core insight: "the introduced bias and variance are governed by the variance of the preference score estimator" [^207^].
   - VRPO introduces three principled techniques: (1) increasing the Monte Carlo sampling budget for ELBOs (n=8 by default), (2) optimal allocation of sampling budget across timesteps (n_t=n, n_{y_t}=1 — one masked sample per timestep), and (3) antithetic sampling between ELBO estimates of model and reference policies [^207^][^219^].
   - **Theorem 2** proves that V[B_hat(y)] = Theta(1/n) and is minimized when allocating the full budget across timesteps rather than drawing multiple masked samples per timestep [^216^].
   - **Theorem 3** proves that antithetic sampling (sharing Monte Carlo samples between paired ELBOs) reduces variance when correlation between model and reference policy estimates is positive [^216^].
   - Applied to LLaDA-8B-Instruct with 350K preference pairs, achieving GSM8K +4.7, HumanEval +3.0, MBPP +1.8, IFEval +4.0, Arena-Hard +4.3 [^207^].
   - VRPO is theoretically extensible to PPO and GRPO as well [^219^].

2. **Coupled-GRPO (DiffuCoder): Complementary Masks for Variance Reduction**
   - DiffuCoder is a 7B-parameter dLLM for code generation, trained on 130B effective tokens via a four-stage pipeline: adaptation pre-training (65B tokens), mid-training (16B tokens), SFT (436K samples), and coupled-GRPO RL (21K hard samples) [^10^][^153^].
   - The coupled-sampling scheme generates paired complementary masks for each completion: for a given sequence, two masks are created such that every token position is masked in exactly one of the two masks [^150^].
   - The log-probability estimate is derived by averaging losses from these two complementary forward passes, ensuring every token is evaluated in a partial-masking context during training [^150^].
   - This provides full token coverage and a more stable gradient signal compared to single random mask or full mask approaches [^150^].
   - Coupled-GRPO boosts EvalPlus score by 4.4% with training on only 21K samples, and reduces AR-ness (autoregressive behavior), enabling more parallel generation [^153^].
   - Training details: 8 H100 GPUs, 40 hours wall-clock time, 21K hard samples filtered from Acecoder-87K by difficulty (bottom 20% pass rate, top 40% variance), using E2B sandbox for code execution reward verification [^10^].
   - The reward function combines execution pass rate (0.5 weight) and format correctness (0.5 weight) [^625^].

3. **EBPO (LLaDA2.1): First Large-Scale RL for Diffusion LLMs**
   - EBPO (ELBO-based Block-level Policy Optimization) was introduced in LLaDA2.1 as the first large-scale RL framework for dLLMs, scaling to unprecedented context lengths and training magnitudes [^164^][^331^].
   - Core innovation: uses ELBO as a principled proxy for exact sequence-level log-likelihood, combined with Vectorized Likelihood Estimation to parallelize bound computation, achieving "orders-of-magnitude acceleration" [^164^].
   - The objective maximizes a clipped surrogate with probability ratio rho: J_EBPO = E[min(rho * A_hat, clip(rho, 1-epsilon_low, 1+epsilon_high) * A_hat)] [^164^].
   - Block-conditional probabilities are computed in parallel: log rho(y|x) ≈ sum_{n=1}^N w_n sum_{b=1}^B (log p_theta(y^b|z_n,x;M) - log p_theta_old(y^b|z_n,x;M)) [^164^].
   - EBPO overcomes the "intractability of sequence-level log-likelihood" that has historically limited RL for diffusion models to small-scale experiments [^164^].
   - LLaDA2.1-Flash (100B) achieves 892 TPS on HumanEval+, 801 TPS on BigCodeBench, 663 TPS on LiveCodeBench [^331^].

4. **Why SFT Hurts Diffusion Models: Train-Test Mismatch**
   - Classical SFT randomly masks tokens across the entire response using bidirectional attention, while semi-autoregressive inference generates fixed-size blocks sequentially with clean prefixes and hidden futures [^621^].
   - This mismatch creates three problems: (i) **noisy prefixes** — training corrupts prefixes that inference keeps clean; (ii) **dependency leakage** — training reveals future tokens that inference never sees; (iii) **granularity mismatch** — training optimizes token-level decisions while inference requires block-level coordination [^621^].
   - **Blockwise SFT** addresses this by partitioning responses into fixed-size blocks, selecting one active block per step for stochastic masking, freezing preceding tokens, and fully hiding future ones [^621^].
   - Blockwise SFT is theoretically grounded with a variational upper bound on blockwise likelihoods and unbiased timestep-sampled gradients [^653^].
   - Empirical results show consistent gains on GSM8K, MATH, and MetaMathQA under matched compute or token budgets, with performance peaking when training block size matches inference block size [^621^][^653^].

5. **TreeDiff: AST-Guided Masking for Code Generation**
   - TreeDiff is the first work to incorporate AST-aware masking into large language diffusion models for code generation, achieving a **13.3% relative improvement** over random masking on HumanEval+ [^614^][^615^].
   - Core innovation: instead of random token masking, TreeDiff selectively masks tokens belonging to key AST nodes using a tiered weighting scheme P = {p_skel, p_data, p_cond, p_ctrl} [^269^].
   - Lower weights are assigned to structural elements (imports, function definitions) to preserve high-level program structure; higher weights to logic and control flow tokens (if, while) [^269^].
   - The method uses a hierarchical probability scheme with AST weighted masking and curriculum noise scheduling [^269^].
   - TreeDiff treats reasoning chains (natural language) and code as distinct modalities: standard random masking for reasoning, AST-guided span corruption for code [^615^].
   - At T=256, TreeDiff achieves 42.1%/37.2% on HumanEval/Plus; at T=512, it maintains 36.6%/33.3% while random masking baselines degrade significantly [^293^].
   - Trained on 150K code reasoning samples with 8 A100 GPUs, LoRA for parameter-efficient adaptation [^293^].

6. **UniGRPO (MMaDA): Unified RL for Multimodal Reasoning and Generation**
   - UniGRPO is a unified policy-gradient RL algorithm specifically tailored for diffusion foundation models, unifying post-training across reasoning and generation tasks with diversified reward modeling [^620^][^647^].
   - MMaDA-8B surpasses LLaMA-3-7B and Qwen2-7B in textual reasoning, outperforms Show-o and SEED-X in multimodal understanding, and excels over SDXL and Janus in text-to-image generation [^620^].
   - UniGRPO addresses three critical challenges in adapting GRPO to diffusion models: (1) local masking dependency, (2) mask ratio sensitivity, and (3) non-autoregressive sequence-level likelihoods [^620^].
   - Key innovation: structured noising strategy that uniformly samples mask ratio p_i ∈ [0,1] rather than masking all response tokens, ensuring the model is exposed to various stages of multi-step diffusion denoising [^14^].
   - Sequence-level log-likelihood is approximated by averaging over masked tokens using ELBO with random masking [^68^].
   - MMaDA employs a mixed long chain-of-thought fine-tuning strategy that curates a unified CoT format across modalities, facilitating cold-start training for RL [^620^].

7. **TraceRL: Trajectory-Aware RL for Diffusion Language Models**
   - TraceRL (Trajectory-Aware Reinforcement Learning) is a post-training method that explicitizes and exploits the generation trajectory of diffusion models [^751^].
   - Instead of sequence-level reward allocation, TraceRL decomposes inference into intermediate "trace steps" and applies PPO with clipped policy ratios and KL regularization at each trace step [^750^][^754^].
   - A **diffusion value model** outputs token-wise value estimates conditioned on the trace prefix, reducing variance and improving stability [^751^][^754^].
   - A **shrinkage parameter s** aggregates s consecutive steps, reducing forward passes needed for policy updates [^750^].
   - TraceRL powers the TraDo model family: TraDo-4B-Instruct outperforms Qwen2.5-7B-Instruct on math reasoning; TraDo-8B-Thinking is the first long-CoT diffusion language model [^751^][^756^].
   - MATH500 improvements: TraDo-4B +5.4% static accuracy, +4.2% dynamic accuracy; TraDo-8B +4.2% static, +4.8% dynamic [^756^].
   - Used in DICE (Diffusion LLMs for CUDA Kernels) with a bi-phase curriculum: kernel infilling stage followed by end-to-end kernel generation stage [^155^].
   - TraceRL training config: lr=1e-6, epsilon=0.2, beta=0.01, 64 problems × 16 responses per step [^155^].

8. **RL Scaling for Diffusion: Data Efficiency Advantages**
   - **Key finding:** Diffusion models are significantly more robust to data repetition than autoregressive models. While AR models begin to overfit as repetition increases, diffusion models show no signs of overfitting even at 100 epochs [^624^].
   - The half-life of data reuse (R_D*) is ~15 epochs for AR models but ~500 for diffusion models — a **33x difference** [^624^].
   - Diffusion models can be trained on repeated data for up to 100 epochs with repeated data almost as effective as fresh data, whereas AR models degrade after ~4 epochs [^624^].
   - A 1.7B DLM trained with ~1.5T-token compute budget on 10B unique Python tokens overtakes an AR coder trained with matched settings [^757^].
   - A 1B-parameter DLM achieves >56% on HellaSwag and >33% on MMLU using only 1B tokens, just by repeating standard pre-training data [^757^].
   - Practical takeaway: "if you are compute-constrained, use autoregressive models; if you are data-constrained, use diffusion models" [^624^].

9. **Sandwiched Policy Gradient (SPG): State-of-the-Art RL for Diffusion Reasoning**
   - SPG leverages both an upper and lower bound of the true log-likelihood to reduce bias in one-sided approximation methods for dLLMs [^643^][^650^].
   - Core insight: ELBO is only a lower bound, making RL objectives invalid for negative rewards. SPG sandwiches the intractable log-likelihood by maximizing a lower bound for positive-reward sequences while minimizing an upper bound for negative-reward ones [^643^].
   - Uses block-wise masking strategy for stable Monte Carlo estimation [^643^].
   - SPG achieves state-of-the-art results: +3.6% GSM8K, +2.6% MATH500, +18.4% Countdown, +27.0% Sudoku over prior RL methods for dLLMs [^643^][^650^].
   - SPG significantly outperforms D1, WD1, and UniGRPO baselines [^643^].

10. **wd1: Ratio-Free Weighted Policy Optimization**
    - wd1 reformulates the RL objective as a weighted log-likelihood, requiring only a single approximation for the current policy likelihood — eliminating policy ratios entirely [^652^][^659^].
    - Can be interpreted as "energy-guided discrete diffusion training combined with negative sample unlearning" [^659^].
    - wd1 outperforms diffusion-based GRPO (d1) while requiring lower computational cost, achieving up to +59% improvement in accuracy [^659^].
    - wd1++ extends this to denoising-stepwise weighted policy optimization, achieving SOTA math performance of 44.2% on MATH500 and 84.5% on GSM8K with only 20 RL training steps [^652^].

11. **AGRPO: Principled On-Policy RL with Unbiased Gradient Estimation**
    - AGRPO (Amortized Group Relative Policy Optimization) uses Monte Carlo sampling to compute an **unbiased policy gradient estimate**, making it the first tractable, faithful adaptation of policy gradient methods for dLLMs [^663^].
    - Achieves up to +7.6% absolute gain on GSM8K and 3.8x performance on Countdown over LLaDA-8B-Instruct, with 1.3x gains over diffu-GRPO [^663^].
    - These gains persist across different numbers of sampling steps at inference time [^663^].

12. **DiSPO: Diffusion-State Policy Optimization (Plug-in Credit Assignment)**
    - DiSPO is a plug-in credit-assignment layer that directly optimizes intermediate filling decisions in masked diffusion LMs [^730^][^731^].
    - At selected intermediate masked states, DiSPO branches by resampling fillings from rollout-cached logits, scores branched completions, and updates only newly filled tokens — no additional multi-step diffusion rollouts [^730^].
    - Consistently improves terminal-feedback baselines (diffu-GRPO and SPG) on math and planning benchmarks under matched rollout compute [^730^].

---

### Major Players & Sources

| Entity | Role/Relevance |
|--------|----------------|
| **LLaDA team (Nie et al., GSAI-ML)** | Open-source 8B diffusion LLM; foundational base model for most RL research |
| **Zhu et al. / LLaDA 1.5 (Renmin University, Ant Group)** | VRPO — first principled variance reduction for diffusion preference optimization |
| **Gong et al. / DiffuCoder (Apple, HKU)** | Coupled-GRPO for code generation; 7B model trained on 130B tokens |
| **LLaDA2.1 team** | EBPO — first large-scale RL for dLLMs with vectorized block-level optimization |
| **Yang et al. / MMaDA (PKU, Princeton, ByteDance Seed)** | UniGRPO — unified RL for multimodal diffusion reasoning and generation |
| **Wang et al. / TraceRL & TraDo (Gen-Verse)** | Trajectory-aware RL; TraDo-4B outperforms Qwen2.5-7B on math |
| **Wang et al. / SPG (Meta Superintelligence Labs, MIT)** | Sandwiched Policy Gradient — SOTA RL for diffusion reasoning |
| **Tang et al. / wd1 (UCL, UNIST)** | Ratio-free weighted policy optimization; wd1++ achieves 44.2% MATH500 |
| **Zhan / AGRPO (Stanford)** | Unbiased policy gradient estimation via Monte Carlo; principled on-policy RL |
| **Oba et al. / DiSPO (Tokyo Institute of Tech, Google)** | Plug-in credit assignment at intermediate diffusion states |
| **Sun et al. / Blockwise SFT (UC Merced, DeepMind)** | Training-inference alignment via blockwise supervision |
| **Zeng et al. / TreeDiff (UConn)** | AST-guided masking for code generation; 13.3% improvement |
| **Bai et al. / DICE** | TraceRL for CUDA kernel generation with bi-phase curriculum RL |
| **Zhao et al. / d1 (UCLA, Meta)** | First policy gradient RL for masked dLLMs (diffu-GRPO) |
| **Xie et al. / SAPO** | Step-aware policy optimization with process rewards for reasoning |
| **Prabhudesai et al. / CMU** | Data-constrained scaling laws showing diffusion outperforms AR |

---

### Trends & Signals

1. **From ELBO to Better Likelihood Estimates**: There is a clear progression from simple ELBO approximations (LLaDA 1.5) to complementary masks (DiffuCoder), to sandwiched bounds (SPG), to unbiased Monte Carlo estimation (AGRPO), to block-level vectorized estimation (EBPO). Each generation reduces bias and variance in policy gradient estimates [^68^][^643^][^663^][^164^].

2. **RL > SFT for Diffusion**: Across virtually all benchmarks, RL post-training outperforms SFT-only approaches for diffusion models. d1 showed that SFT+diffu-GRPO outperforms either alone; LLaDA 1.5 showed VRPO outperforms SFT-only by large margins; Blockwise SFT showed that even SFT can be improved by better alignment with inference [^758^][^207^][^621^].

3. **Training-Inference Alignment is Critical**: Blockwise SFT demonstrates that matching training supervision granularity to the decoding procedure is a "core driver of performance" [^621^]. This principle extends from SFT to RL — methods like EBPO and Blockwise SFT explicitly align with how diffusion models actually generate at inference time.

4. **Diffusion Excels in Data-Constrained Regimes**: Multiple independent studies confirm that diffusion models are dramatically more data-efficient than AR models when data can be repeated (R_D* ~500 vs ~15 epochs) [^624^][^757^]. This has major implications for RL post-training, where generating diverse training data is expensive.

5. **Structure-Aware Training for Code**: TreeDiff's AST-guided masking and DiffuCoder's coupled-GRPO both demonstrate that respecting code structure during training yields significant gains. The 13.3% improvement from TreeDiff suggests diffusion models for code benefit enormously from syntax-aware corruption [^614^][^153^].

6. **Trajectory-Level vs. Terminal-Feedback RL**: A major trend is the shift from terminal-only rewards (d1, diffu-GRPO) to trajectory-aware methods (TraceRL, SAPO, DiSPO) that exploit intermediate diffusion states for finer credit assignment [^751^][^645^][^730^].

7. **Unified Multimodal RL**: UniGRPO/MMaDA represents the trend toward unified RL frameworks that work across text, vision, and generation tasks within a single diffusion architecture [^620^].

---

### Controversies & Conflicting Claims

1. **SFT: Harmful or Helpful?**
   - **Anti-SFT:** Blockwise SFT argues that classical SFT "misaligns with semi-autoregressive inference" and creates noisy prefixes, dependency leakage, and granularity mismatch [^621^]. d2 asks "whether large-scale post-training on DLMs requires additional supervised finetuning, or whether an RL-only approach, akin to DeepSeek-R1-Zero, can be effectively applied" [^655^].
   - **Pro-SFT:** d1 demonstrated that SFT followed by diffu-GRPO outperforms either alone, and most successful pipelines (DiffuCoder, DICE, LLaDA2.1) include an SFT stage before RL [^758^].
   - **Resolution:** The evidence suggests SFT is valuable but must be done correctly — Blockwise SFT shows the problem is not SFT itself but training-inference mismatch in how SFT is implemented.

2. **One-Step vs. Multi-Step vs. ELBO Likelihood Estimation**
   - d1/diffu-GRPO uses one-step estimation (fully masked completion), which is computationally efficient but potentially biased [^758^][^66^].
   - LLaDA 1.5/UniGRPO use ELBO with random masking, which is principled but high-variance without variance reduction [^207^].
   - SPG argues that one-sided ELBO approximations "can introduce significant policy gradient bias" and proposes sandwiching with both upper and lower bounds [^643^].
   - AGRPO claims to be the first "tractable, faithful adaptation" with unbiased Monte Carlo estimation [^663^].
   - EBPO uses vectorized block-conditional probabilities for scalable estimation [^164^].

3. **AR vs. Diffusion: Which is Better for RL Post-Training?**
   - AR models are more compute-efficient for training (Nie et al. report ~16x more compute needed for diffusion to match NLL) [^624^].
   - However, diffusion models are far more data-efficient and robust to repetition, making them preferable when RL-generated training data is scarce [^624^][^757^].
   - The practical consensus emerging: "if you are compute-constrained, use AR; if you are data-constrained, use diffusion" [^624^].

4. **Temperature and AR-ness: Does RL Reduce or Increase Parallelism?**
   - DiffuCoder found that coupled-GRPO training increases the optimal sampling temperature, suggesting the model becomes less autoregressive and more parallel [^629^][^631^].
   - However, some analyses show diffusion models naturally tend toward semi-autoregressive behavior during decoding, and the extent to which RL can change this remains debated [^153^].

---

### Algorithm Comparison Table

| Algorithm | Policy Update | Likelihood Estimation | Masking Strategy | Key Innovation | Best Benchmark Results |
|-----------|--------------|----------------------|-------------------|----------------|----------------------|
| **VRPO** (LLaDA 1.5) | DPO | ELBO with random masking | Random masking | Timestep-wise budget allocation + antithetic sampling | GSM8K +4.7, HumanEval +3.0 |
| **Coupled-GRPO** (DiffuCoder) | GRPO | Complementary mask averaging | Complementary masks | Coupled sampling ensures full token coverage | EvalPlus +4.4% |
| **EBPO** (LLaDA2.1) | PPO-style clipped surrogate | Vectorized block-conditional ELBO | Block-level | First large-scale RL for dLLMs; parallel computation | 892 TPS HumanEval+ |
| **UniGRPO** (MMaDA) | GRPO | ELBO with random masking | Structured noising (uniform mask ratio) | Unified RL across modalities and task types | SOTA on MMU, T2I |
| **SPG** | Policy gradient | ELBO (positive); EUBO/Mixture (negative) | Block-wise masking | Sandwiched bounds reduce bias | +3.6% GSM8K, +18.4% Countdown |
| **wd1** | Weighted likelihood | One-step estimation | Prompt masking | Ratio-free, single likelihood estimate | 44.2% MATH500, 84.5% GSM8K |
| **AGRPO** | GRPO | Monte Carlo sampling | — | Unbiased policy gradient estimation | +7.6% GSM8K, 3.8x Countdown |
| **TraceRL** | PPO | Trajectory-level with shrinkage | Trace step aggregation | Diffusion value model, trajectory-aware | TraDo-4B > Qwen2.5-7B |
| **DiSPO** | Plug-in to base PO | State-wise masked-token surrogate | Intermediate state branching | Optimizes intermediate filling decisions | Improves diffu-GRPO/SPG |
| **SAPO** | GRPO | Stepwise probability | — | Process rewards aligned with reasoning hierarchy | More interpretable reasoning |
| **Blockwise SFT** | SFT (not RL) | Block-conditional | Active block masking | Training-inference alignment | Consistent gains on GSM8K, MATH |
| **d1/diffu-GRPO** | GRPO | One-step estimation | Random prompt masking | First PG RL for masked dLLMs | Nearly doubled planning performance |

---

### Reward Model Design for Code Generation

1. **Verifiable Rewards:** DiffuCoder uses a combination of execution pass rate (0.5) and format correctness (0.5) as the reward function, with code execution via E2B sandbox for verification [^625^].

2. **ExecVerify Framework:** Introduces white-box RL with verifiable stepwise rewards derived from execution traces, including next-statement prediction and variable value/type prediction. A 7B model achieves performance comparable to 32B models on code reasoning benchmarks and improves pass@1 by up to 5.9% on code generation [^623^][^630^].

3. **Deceptive Behavior in CUDA Kernels:** DICE identifies three types of deceptive behavior in generated kernels: (1) defaulting to PyTorch functions, (2) generating valid kernels without invocation logic, (3) omitting custom kernels from forward functions. Bi-phase curriculum RL mitigates these issues [^155^].

4. **Reward Design Principles:** Effective rewards for code generation with diffusion models should be: (a) verifiable via execution, (b) structured to prevent reward hacking, (c) progressively applied via curriculum learning, and (d) combined with both outcome-based (test pass) and process-based (intermediate reasoning) signals.

---

### Recommended Deep-Dive Areas

1. **Theoretical Foundations of Variance Reduction in Diffusion RL**: VRPO's analysis of ELBO estimator variance provides a principled framework, but extending this to other RL objectives (PPO, TRPO, actor-critic) remains underexplored. The interaction between Monte Carlo sample size, timestep allocation, and antithetic sampling deserves further theoretical characterization.

2. **Blockwise Training-Inference Alignment**: Blockwise SFT's demonstration that training block size should match inference block size suggests a fundamental principle. Extending this to RL (e.g., blockwise policy gradients) could yield significant gains. The interaction between block size and model scale is also underexplored.

3. **Scaling RL for 100B+ Diffusion Models**: LLaDA2.1-Flash (100B) with EBPO represents the frontier, but the computational challenges of RL at this scale are immense. Research into efficient gradient estimation, distributed RL, and memory optimization for large diffusion models is critical.

4. **Process-Based Rewards for Diffusion**: SAPO and DiSPO represent early steps in exploiting intermediate diffusion states for finer credit assignment. The design of process-based rewards that align with the hierarchical nature of diffusion denoising (e.g., early steps for high-level structure, late steps for fine details) is a rich research direction.

5. **Structure-Aware Diffusion for Code**: TreeDiff's 13.3% improvement with AST-guided masking suggests enormous potential. Extending this to other structured domains (SQL, configuration files, mathematical expressions) and combining with RL could yield domain-specific diffusion models that match or exceed AR performance.

6. **Data Efficiency in RL for Diffusion**: Given diffusion's dramatic advantage in data-constrained settings (R_D* ~500 vs ~15), understanding how much RL training data diffusion models actually need compared to AR is critical for practical deployment. Early evidence suggests diffusion models can achieve strong RL results with far less data (e.g., DiffuCoder with 21K samples).

7. **RL-Only Post-Training (No SFT)**: The d2 paper raises the question of whether RL-only post-training (ala DeepSeek-R1-Zero) is viable for diffusion models. Given that diffusion models have bidirectional context and different exploration dynamics, RL-only training might be more effective than for AR models.

8. **Unified Multimodal RL**: UniGRPO's success across text, vision, and generation suggests that unified diffusion architectures with unified RL could be the future. Understanding how to design reward functions and RL objectives that work across modalities is a major open problem.

9. **Inference-Time Compute Scaling with RL-Trained Diffusion**: Diffusion models naturally support variable inference steps (speed/quality tradeoff). RL-trained diffusion models that can dynamically adapt their denoising strategy based on problem difficulty could enable novel inference-time scaling approaches.

10. **Safety and Alignment of RL-Trained Diffusion Models**: As diffusion models gain capabilities through RL, ensuring they remain safe and aligned becomes critical. The survey by Xing et al. identifies challenges in adversarial robustness, safety-creativity tradeoffs, and the need for red-teaming evaluations [^661^].

---

### References

1. Zhu et al., "LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models," arXiv:2505.19223. https://arxiv.org/abs/2505.19223
2. Gong et al., "DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation," arXiv:2506.20639. https://arxiv.org/abs/2506.20639
3. "LLaDA2.1: Speeding Up Text Diffusion via Token Editing," arXiv:2602.08676. https://arxiv.org/abs/2602.08676
4. Sun et al., "Blockwise SFT for Diffusion Language Models," arXiv:2508.19529. https://arxiv.org/abs/2508.19529
5. Zeng et al., "TreeDiff: AST-Guided Code Generation with Diffusion LLMs," arXiv:2508.01473. https://arxiv.org/abs/2508.01473
6. Tian et al., "MMaDA: Multimodal Large Diffusion Language Models," arXiv:2505.15809. https://arxiv.org/abs/2505.15809
7. Wang et al., "TraceRL: Revolutionizing post-training for Diffusion LLMs," GitHub: Gen-Verse/dLLM-RL, ICLR 2026. https://github.com/Gen-Verse/dLLM-RL
8. Wang et al., "SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models," arXiv:2510.09541. https://arxiv.org/abs/2510.09541
9. Tang et al., "wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models," arXiv:2507.08838. https://arxiv.org/abs/2507.08838
10. Zhan, "AGRPO: Amortized Group Relative Policy Optimization for Diffusion Language Models," arXiv:2510.04019. https://arxiv.org/abs/2510.04019
11. Oba et al., "DiSPO: Diffusion-State Policy Optimization for Masked Diffusion Language Models," arXiv:2602.06462. https://arxiv.org/abs/2602.06462
12. Xie et al., "SAPO: Step-Aware Policy Optimization for Reasoning in Diffusion Large Language Models," arXiv:2510.01544. https://arxiv.org/abs/2510.01544
13. Zhao et al., "d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning," arXiv:2504.12216. https://arxiv.org/abs/2504.12216
14. Bai et al., "DICE: Diffusion Large Language Models Excel at Generating CUDA Kernels," arXiv:2602.11715. https://arxiv.org/abs/2602.11715
15. Prabhudesai et al., "Diffusion Beats Autoregressive in Data-Constrained Settings," CMU blog, 2025. https://blog.ml.cmu.edu/2025/09/22/diffusion-beats-autoregressive-in-data-constrained-settings/
16. Liu et al., "Diffusion Language Models are Super Data Learners," arXiv:2511.03276. https://arxiv.org/abs/2511.03276
17. Tang et al., "ExecVerify: White-Box RL with Verifiable Stepwise Rewards for Code Execution Reasoning," arXiv:2603.11226. https://arxiv.org/abs/2603.11226
18. "A Survey on Diffusion Language Models," arXiv:2508.10875. https://arxiv.org/abs/2508.10875
19. "A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models," arXiv:2508.08712. https://arxiv.org/abs/2508.08712
20. "d2: Improved Techniques for Training Reasoning Diffusion Language Models," arXiv:2509.21474. https://arxiv.org/abs/2509.21474
21. Cheng et al., "SDAR: A Synergistic Diffusion-AutoRegression Paradigm," arXiv:2510.06303. https://arxiv.org/abs/2510.06303
22. Ni et al., "Training Optimal Large Diffusion Language Models (Quokka)," 2025. https://jinjieni.github.io/Quokka/
