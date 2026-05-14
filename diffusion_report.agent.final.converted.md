# Executive Summary

Diffusion language models (dLLMs) have transitioned from theoretical curiosities to production-viable alternatives to autoregressive (AR) architectures for text and code generation. This report examines the complete landscape of diffusion-based code models through 330+ research queries executed across 18 specialized agents covering 12 analytical dimensions — from foundational architectures and training methodologies to commercial deployment and future outlook. The central finding is that diffusion models have achieved parity with AR models on standard code generation benchmarks while delivering 2-5x inference speed advantages, and have established a decisive edge in code editing tasks where their any-order generation capability creates structural advantages. ^1^ ^2^ ^3^## Key Findings

**Quality parity on standard benchmarks with accelerating inference advantages.** The leading diffusion models now match or exceed comparable AR baselines on HumanEval and MBPP. LLaDA2.0-flash (Ant Group) achieves 94.51% on HumanEval and 88.29% on MBPP, surpassing the AR baseline Qwen3-30B-A3B-Instruct at 93.29% and 86.65% respectively. ^4^ ^5^Gemini Diffusion (Google DeepMind) scores 89.6% on HumanEval, exceeding Seed-Coder-8B-Instruct's 84.8%. ^2^Simultaneously, these models deliver substantially higher throughput: Seed Diffusion achieves 2,146 tokens/second on H20 GPUs, Gemini Diffusion reaches 1,479 tok/s, and Mercury Coder Mini sustains 1,109 tok/s on H100 hardware. ^6^ ^3^ ^7^Under controlled serving conditions (SGLang with TP8 on H20), LLaDA2.0-flash-CAP achieves 535 TPS versus 256 TPS for Ling-flash-2.0, a 2.1x speedup. ^8^ ^9^**Code editing emerges as diffusion's uncontested advantage.** While benchmark parity on completion tasks is significant, the most consequential finding is diffusion's superiority on code editing benchmarks. Stable-DiffCoder achieves 60.0% on CanItEdit versus 50.5% for its AR counterpart Seed-Coder — a 9.5 percentage point absolute gap representing an 18.8% relative improvement. ^10^ ^11^This advantage is not incidental: code editing is inherently non-sequential (changing a function signature requires updating all callers simultaneously), which aligns precisely with diffusion's any-order generation capability. ^1^Complementing this result, TreeDiff's AST-guided masking achieves a 13.3% relative improvement over random masking on HumanEval+ by structuring the diffusion process around code syntax. ^12^ ^13^The combination of superior editing performance and stronger length extrapolation on RepoQA (diffusion models maintain >30% accuracy where AR models drop below 10% at extended contexts) positions diffusion models to capture the IDE assistant market for editing workflows. ^14^**Reinforcement learning, not architecture alone, drives the largest quality gains.** The narrative around diffusion models emphasizes inference speed, yet the performance data reveals that post-training reinforcement learning (RL) is responsible for the most significant quality improvements. LLaDA2.0-flash-preview scores only 29.07 on LiveCodeBench versus 42.29 for the fully post-trained model — a 45% improvement attributable to supervised fine-tuning (SFT), confidence-aware parallel decoding (CAP), and direct preference optimization (DPO), not the base diffusion conversion. ^4^Similarly, DiffuCoder's coupled-GRPO achieves a +4.4% EvalPlus improvement with only 21K training examples. ^15^ ^16^EBPO (ELBO-based Block-level Policy Optimization), introduced with LLaDA2.1, represents the first large-scale RL framework for dLLMs, scaling to 100B parameters. ^17^ ^18^VRPO from the LLaDA 1.5 work establishes the theoretical foundation, proving that variance in diffusion preference optimization can be systematically reduced through antithetic sampling and optimal Monte Carlo budget allocation. ^19^The implication is that organizations mastering RL for diffusion — Ant Group with EBPO, Apple with coupled-GRPO — will outperform competitors focused solely on architectural innovations. ^20^**Two leaders, divergent strategies: Google DeepMind's commercial closed-source approach versus Ant Group's open-source ecosystem play.** Google DeepMind drives commercial innovation through Gemini Diffusion, an experimental block-diffusion model announced at Google I/O 2025 that demonstrates ~5x speedup over Gemini 2.0 Flash Lite. ^3^ ^21^Built on the MD4 theoretical framework published at NeurIPS 2024 and leveraging the AR2Diff conversion methodology, Gemini Diffusion remains waitlist-only with no production API. ^22^ ^23^ ^24^In contrast, Ant Group has constructed the most comprehensive open-source diffusion LLM ecosystem: LLaDA2.0 (100B parameters, Apache 2.0), LLaDA2.1 (with token editing via T2T), complete training infrastructure (dFactory), inference engine (dInfer), and SGLang integration. ^5^ ^25^ ^26^ ^27^Ant Group's CodeFuse NES system demonstrates real-world deployment at scale, serving over 20,000 developers through a Tab-key interaction paradigm achieving 51.55% acceptance for location predictions and 43.44% for edit suggestions. ^28^**China leads the open-source diffusion LLM race.** All major open-source diffusion language models originate from Chinese institutions: Ant Group (LLaDA family), ByteDance (Seed Diffusion, Stable-DiffCoder), Renmin University (GSAI-ML), and Tsinghua University (SIA-Lab). ^5^ ^6^ ^29^US contributions — Google DeepMind's Gemini Diffusion, Inception Labs' Mercury, Apple's DiffuCoder — are predominantly closed-source or limited release. This geographic distribution inverts the AR landscape where US-based organizations (OpenAI, Anthropic, Meta, Google) lead open-weight releases. The open-source diffusion ecosystem may develop under Chinese institutional leadership, with implications for global access to state-of-the-art dLLM weights and tooling.

## The Competitive Landscape: Models, Speed, and Benchmarks

The following table summarizes the leading diffusion models for code generation, their architectural characteristics, speed claims, and availability status:

| Model | Organization | Parameters | Speed (tok/s) | Hardware | HumanEval | Availability |
|-------|-------------|------------|---------------|----------|-----------|--------------|
| LLaDA2.0-Flash | Ant Group | 100B (6.1B active) | 535 ^8^| H20, SGLang TP8 | 94.51% ^4^| Open-source (Apache 2.0) |
| Gemini Diffusion | Google DeepMind | Undisclosed | 1,479 ^3^| Unknown | 89.6% ^2^| Waitlist/experimental |
| Mercury Coder Mini | Inception Labs | Undisclosed | 1,109 ^7^| H100 | 88.0% ^7^| Production API |
| Mercury Coder Small | Inception Labs | Undisclosed | 737 ^7^| H100 | 90.0% ^7^| Production API |
| Stable-DiffCoder-8B | ByteDance/Tsinghua | 8B | Not reported | — | 86.6% ^10^| Open-source |
| Seed Diffusion | ByteDance | Undisclosed | 2,146 ^6^| H20 | 79.4% ^2^| Preview/demo only |
| LLaDA2.1-Flash | Ant Group | 100B | 892 ^17^| — | Not reported | Open-source |
| LLaDA2.1-Mini | Ant Group | 16B (1.4B active) | 1,587 ^17^| Quantized | Not reported | Open-source |
| DiffuCoder-7B | Apple | 7B | Not reported | — | 69.5% ^2^| Research release |

Three observations emerge from this landscape. First, only Inception Labs' Mercury Coder offers a production API with established pricing ($0.25 per million input tokens, $0.75-1.00 per million output tokens), despite Gemini Diffusion and Seed Diffusion having been announced earlier. ^30^ ^31^This integration gap — the absence of native IDE plugins for any diffusion model — is the primary barrier to enterprise adoption, not model quality. ^32^Second, the open-source releases from Ant Group and ByteDance provide the entire toolchain (training, inference, serving), lowering barriers to entry for organizations seeking to deploy diffusion models internally. Third, speed comparisons across models must be treated cautiously: reported figures use different hardware (H20 versus H100 versus TPU), serving stacks, and measurement methodologies. ^33^Only the controlled SGLang comparison provides an apples-to-apples speedup metric.

The speed-quality relationship across these models is visualized in Figure 1, which positions each model by its inference throughput and HumanEval score relative to AR baselines.

![Figure 1: Diffusion vs. Autoregressive Models — Speed-Quality Tradeoff](/mnt/agents/output/fig_exec_summary_speed_quality.png)

**Figure 1.** Diffusion models (circles for closed commercial, squares for open-source) cluster in the upper-right quadrant relative to AR baselines (triangles), demonstrating the concurrent achievement of higher speed and comparable or superior code generation quality. LLaDA2.0-Flash achieves the highest HumanEval score at 94.51% while Seed Diffusion pushes throughput to 2,146 tok/s. The 2-5x speed advantage zone is indicated for reference. Data sources: ^6^ ^5^ ^7^ ^4^ ^17^ ^3^ ^8^## Benchmark Performance by Task Type

Whether diffusion models "win" or "lose" depends entirely on which benchmarks are emphasized. The following matrix organizes benchmark results by task category, revealing a clear pattern:

| Task Category | Benchmark | Diffusion Best | AR Best | Diffusion Avg | AR Avg | Diffusion Advantage? |
|-------------|-----------|--------------|---------|--------------|--------|---------------------|
| Standard completion | HumanEval | 94.51% (LLaDA2.0) ^4^| 93.29% (Qwen3) | 66.7% ^2^| 71.3% | Parity (best models) |
| Standard completion | MBPP | 88.29% (LLaDA2.0) ^4^| 86.65% (Qwen3) | 61.2% ^2^| 60.8% | Parity |
| Standard completion | BigCodeBench | 45.4% (Gemini) ^3^| 45.8% (Flash-Lite) | — | — | Parity |
| Code editing | CanItEdit | 60.0% (Stable-DiffCoder) ^10^| 50.5% (Seed-Coder) | 57.2% | 50.5% | **Yes (+18.8% relative)** |
| Competitive programming | LiveCodeBench v6 | 30.9% (Gemini) ^2^| 26.0% (Qwen3) | 14.9% ^2^| 18.9% | **No (-21.2% relative)** |
| Long-context understanding | RepoQA 4K+ | >30% (DiffuCoder) ^14^| <10% (Llama-2) | — | — | **Yes (3x+ retention)** |
| Real-world SE | SWE-Bench | 22.9% (Gemini) ^3^| 28.5% (Flash-Lite) | — | — | **No (-19.6% relative)** |

This matrix reveals a structural pattern explained by the Non-Autoregressive Paradox (NAP): training data's sequential structure forces diffusion models into AR-like decoding patterns. ^34^On benchmarks rewarding sequential reasoning — competitive programming and multi-step software engineering tasks — diffusion models underperform because their parallel generation disrupts logical chains. On benchmarks rewarding parallel processing — code editing, infilling, and long-context retrieval — they overperform because bidirectional attention enables global error correction and any-order generation. The NAP paper demonstrates that even with fully parallel decoding, Dream-7B exhibits high "ARness" (~0.92), meaning its most confident tokens almost always follow the sequential order. ^34^This has critical implications for application design: diffusion models should be deployed for editing-centric workflows rather than competitive programming or complex agentic tasks requiring sequential tool use.

## Commercial Landscape and Market Outlook

The diffusion language model market remains in an embryonic commercial phase. Inception Labs has raised $50 million in seed funding (led by Menlo Ventures, with participation from Microsoft M12, Snowflake Ventures, Databricks Ventures, and Nvidia NVentures) at an approximate $500 million valuation. ^35^ ^31^Angel investors Andrew Ng and Andrej Karpathy provide academic credibility alongside corporate venture participants signaling potential distribution channels. ^36^The broader diffusion models market — spanning image, video, text, and audio — is projected to grow from $2.23 billion in 2025 to $7.42 billion by 2030 at a 27.2% compound annual growth rate (CAGR). ^37^Despite this investment activity, enterprise adoption faces a critical blocker: IDE integration. GitHub Copilot's dominance derives not from model quality alone but from deep embedding in VS Code and JetBrains IDEs with real-time suggestion display, ghost text, and seamless UX. ^32^Diffusion models remain API-only. Ant Group's NES system demonstrates that with proper UX integration (a Tab-key workflow serving 20,000 developers), diffusion models can achieve meaningful real-world usage. ^28^The next 12 months will be decisive: if Mercury or an open-source alternative achieves Copilot-level integration, the inflection point for diffusion adoption could arrive suddenly.

## Ten Cross-Dimension Insights

The 12-dimensional research analysis produced ten insights with direct strategic relevance. In addition to the five findings highlighted above, five further conclusions emerge:

**The AR-to-diffusion conversion paradigm creates a structural moat.** Rather than training diffusion models from scratch, all leading implementations convert pretrained AR base models: LLaDA2.0 converts Ant Group's Ling models, Stable-DiffCoder converts Seed-Coder, and Dream converts Qwen2.5. ^5^ ^11^This means the expensive investment in AR pretraining does not disappear — it transfers. Organizations without strong AR base models are at a disadvantage, reinforcing the positions of Ant Group, ByteDance, and Google.

**Block diffusion is the pragmatic production architecture.** The original vision of fully parallel generation (all tokens simultaneously) has given way to block diffusion — semi-autoregressive blocks of ~32 tokens with parallel intra-block generation. ^38^All production dLLMs (Gemini, LLaDA2.0, Stable-DiffCoder, Mercury) converge on block size ~32, suggesting this is the permanent production architecture rather than a transitional compromise. ^5^ ^39^**Inference acceleration research outpaces model research.** Fast-dLLM (ICLR 2026) achieves 27.6x throughput improvement via block-wise approximate KV caching. ^40^Elastic-Cache achieves 45.1x on longer sequences. ^41^FreeCache achieves 34x speedup on Dream-7B with negligible accuracy loss. ^42^These order-of-magnitude improvements arrive faster than new model architectures, suggesting diffusion speed advantages will compound.

**The "true diffusion" debate is a distraction.** Philosophical disputes about whether masked diffusion models constitute "true" diffusion versus "BERT with extra steps" are rendered moot by architectural convergence: A3 (Any-order Autoregressive) demonstrates that AR models can achieve any-order generation too. ^34^What matters is not categorical purity but practical benefits — parallel generation, iterative refinement, and any-order capability — which both paradigms are adopting from each other.

**Data efficiency advantages favor diffusion in resource-constrained settings.** Diffusion models are significantly more robust to data repetition than AR models: the half-life of data reuse is approximately 500 epochs for diffusion versus ~15 epochs for AR — a 33x difference. ^43^A 1.7B-parameter DLM trained on 10B unique tokens with repetition can overtake an AR coder trained with matched fresh-data compute. ^44^This property is particularly relevant for enterprise code generation, where proprietary training data is inherently limited and expensive to curate. The practical takeaway is that in data-constrained environments, diffusion training delivers superior model quality per token of unique training data consumed.

## Research Scope and Methodology

This report synthesizes findings from 330+ research queries across 18 specialized agents examining 12 analytical dimensions: model architectures (Gemini Diffusion, LLaDA2.0/2.1), training methodologies (WSD, AR-to-diffusion conversion), inference acceleration (Fast-dLLM, Elastic-Cache, FreeCache), reinforcement learning (VRPO, coupled-GRPO, EBPO), code-specific techniques (TreeDiff, Stable-DiffCoder), the open-source ecosystem, ByteDance's contributions, commercial landscape and enterprise adoption, benchmarking and evaluation, and future outlook and projections. Sources span arXiv preprints, official technical reports, conference proceedings (NeurIPS 2024, ICLR 2026), vendor publications, and independent benchmarking studies. Cross-verification across agents classified findings into high-confidence (confirmed by ≥2 independent sources), medium-confidence (single authoritative source), and low-confidence tiers, with explicit conflict zones identified for contested claims. Key conflict zones include speed comparison fairness (different hardware and serving stacks make cross-model speed claims difficult to verify), diffusion versus AR performance on LiveCodeBench (top diffusion models can compete but the average lags), and the disputed MDM-Prime-v2 21.8x efficiency claim (withdrawn without explanation).

The report covers models from six organizations: Google DeepMind (Gemini Diffusion, MD4, AR2Diff), Ant Group/InclusionAI (LLaDA2.0, LLaDA2.1, LLaDA-MoE, CodeFuse NES, dFactory, dInfer), ByteDance/Seed (Seed Diffusion, Stable-DiffCoder, Seed-Coder), Inception Labs (Mercury Coder), Apple (DiffuCoder), and academic collaborators (Dream, SEDD, GSAI-ML). Collectively, these models represent the full spectrum of current diffusion language model development — from 7B-parameter research releases to 100B-parameter production systems, from closed commercial APIs to fully open-source toolchains.

The evidence presented across subsequent chapters supports a measured but optimistic assessment: diffusion models for code generation have crossed the threshold from experimental to viable, with a clear beachhead in code editing, a maturing open-source ecosystem led by Chinese institutions, and inference speed advantages that compound with each optimization breakthrough. The central strategic question is no longer whether diffusion models can generate code effectively, but which organization will deliver the IDE integration that translates technical capability into developer adoption at scale. For executives and researchers evaluating this rapidly evolving landscape, the chapters that follow provide the technical depth, competitive intelligence, and quantitative evidence needed to inform investment, partnership, and architectural decisions.
-e 


## 2. Technical Foundations of Diffusion Language Models

The adaptation of diffusion models from continuous image domains to discrete language domains represents one of the most significant architectural shifts in generative modeling over the past three years. Where diffusion originally operated on the principle of progressively denoising Gaussian-corrupted pixels, language generation requires reasoning over discrete token vocabularies. This fundamental mismatch — continuous versus discrete state spaces — has driven the development of multiple competing paradigms, each with distinct mathematical formulations and empirical trade-offs. This chapter examines the three architectural families that have emerged: discrete diffusion over token spaces, embedding-space diffusion with rounding, and masked diffusion models (MDMs) that have rapidly become the dominant implementation choice for production systems.

### 2.1 From Images to Text: Adapting Diffusion for Discrete Data

#### 2.1.1 The Core Challenge: From Pixels to Tokens

Diffusion models, as originally formulated for image generation, operate in continuous Euclidean space where the forward process gradually injects Gaussian noise into pixel-valued tensors and the reverse process learns to denoise them. Text, by contrast, is fundamentally discrete: a vocabulary of tokens $\mathcal{V}$ (typically 32,000 to 100,000 entries in modern subword tokenizers) forms a finite, unordered state space. Applying Gaussian noise to a discrete token ID produces a meaningless real-valued vector, and no natural notion exists for "partially noising" a token in the same way that a pixel can be incrementally corrupted toward $\mathcal{N}(0, I)$.

Two principal strategies have emerged to resolve this incompatibility. The first, **discrete diffusion**, defines the forward process as a Markov chain over the finite vocabulary using a transition matrix $\mathbf{Q}_t$ that stochastically maps each token to other tokens or a special [MASK] state at each timestep. The second, **embedding-space diffusion**, maps tokens to continuous vector representations, applies standard Gaussian diffusion in that continuous space, and rounds the final continuous vectors back to discrete tokens at the end of the reverse process. Both approaches have demonstrated strong empirical results, though masked diffusion — a specific discrete formulation — has come to dominate the practical landscape.

#### 2.1.2 Discrete Diffusion: Markov Chains over Vocabulary Space

The discrete diffusion formulation, introduced by D3PM (Discrete Denoising Diffusion Probabilistic Models) and subsequently refined by SEDD (Score Entropy Based Discrete Diffusion), defines the forward process through a series of transition matrices. At each timestep $t$, the distribution over tokens is computed as $\mathbf{x}_t \sim \mathbf{Q}_t \mathbf{x}_{t-1}$, where $\mathbf{Q}_t \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}$ governs how each token transitions to other vocabulary entries or to an absorbing [MASK] state. The training objective simplifies under certain conditions to a weighted cross-entropy loss integrated over masking rates:

$$\mathcal{L}_{\text{CE}}(\theta) = \mathbb{E}_{t \sim U(0,1), \mathbf{x}_t} \left[ -\sum_{i} w(t) \log p_\theta(x_0^{(i)} \mid \mathbf{x}_t, t) \right]$$

where $w(t)$ is a timestep-dependent weighting function and $\mathbf{x}_t$ represents the partially masked or corrupted sequence. This formulation has the practical advantage that training reduces to a form of masked language modeling, which can leverage the same infrastructure developed for BERT-style pretraining. The denoising model $p_\theta$ learns to predict the original token at each position given the corrupted context, and the objective closely resembles the cross-entropy loss used in autoregressive (AR) language modeling — albeit with bidirectional rather than unidirectional context.

Discrete diffusion models of this family have demonstrated strong scaling properties and remain the dominant paradigm at scale. Standard masked diffusion models (MDMs) empirically require approximately 16 times more compute than autoregressive models to match validation loss under standard single-epoch training regimes ^45^. However, subsequent innovations have narrowed this gap substantially: MDM-Prime (v1), published at NeurIPS 2025, achieved a perplexity of 15.36 on OpenWebText (OWT), surpassing autoregressive baselines at 17.54 — the first MDM-based approach to do so without autoregressive formulation ^46^.

#### 2.1.3 Embedding-Space Diffusion: Continuous Flows with Rounding

The alternative approach maps discrete tokens to continuous embeddings, applies standard Gaussian diffusion or flow matching in the embedding space, and rounds back to discrete tokens at the final timestep. Two landmark models — LangFlow and ELF — have demonstrated that this approach can now match or exceed discrete diffusion on standard benchmarks, marking a reversal from earlier results where continuous-space models consistently lagged their discrete counterparts.

**LangFlow** is the first continuous diffusion language model to surpass state-of-the-art discrete diffusion models on multiple standard language modeling benchmarks ^47^. Its core theoretical contribution is connecting embedding-space diffusion to Flow Matching via Bregman divergence. For any convex function $f$, the training objective becomes:

$$\mathcal{L}_f(\theta) = \mathbb{E}_{\gamma \sim \pi, z_\gamma} \left[ \frac{1}{L} \sum_{i=1}^{L} \mathcal{D}_f\left(\mathbf{1}_{x^{(i)}}, \hat{\mathbf{x}}_{\theta}^{(i)}(z_\gamma, \gamma)\right) \right]$$

where $\mathcal{D}_f$ denotes the Bregman divergence and $\gamma$ is the log-noise-to-signal ratio sampled from an information-uniform noise schedule ^48^. LangFlow achieves a perplexity of 30.0 on LM1B and 24.6 on OWT, surpassing all uniform-state discrete diffusion baselines and matching masked diffusion performance ^47^. Notably, LangFlow's profiling reveals that the optimal noise schedule for language follows a Gumbel distribution over $\gamma$ — a finding that "greatly differs from conclusions in the image generation domain" ^47^— motivating a learnable scheduler grounded in this distributional form.

**ELF** (Embedded Language Flows), from Kaiming He's team at MIT, takes a different design path. Rather than jointly training embeddings with the diffusion model as LangFlow does, ELF operates on **frozen pretrained embeddings** from a T5 encoder. The denoising process stays entirely within the continuous embedding space until the final timestep, where a shared-weight network maps representations back to discrete tokens ^49^. This design choice yields remarkable training efficiency: ELF-B (105M parameters) achieves a generative perplexity of approximately 24.1 on OWT with only 32 sampling steps, while using **10 times fewer training tokens** than comparable models (45B versus 500B+ tokens) ^50^ ^51^. ELF is also the only method to simultaneously employ continuous-time Flow Matching, frozen pretrained encoder representations, no per-step discretization during training or inference, and no separate decoder ^49^.

**Table 2.1: Discrete versus Continuous Diffusion Approaches for Language Modeling**

| Aspect | Discrete Diffusion (D3PM/SEDD/MDLM) | Continuous Diffusion (LangFlow/ELF) |
|:---|:---|:---|
| State space | Finite vocabulary with [MASK] absorbing state | Continuous embedding vectors |
| Forward process | Markov chain via transition matrix $\mathbf{Q}_t$ | Gaussian noise / flow interpolation in embedding space |
| Training objective | Weighted cross-entropy over token predictions | Bregman divergence flow matching (LangFlow); standard FM (ELF) |
| Embedding strategy | Token embeddings learned jointly | Jointly trained (LangFlow); frozen pretrained T5 (ELF) |
| Noise schedule | Cosine / learned over discrete timesteps | Gumbel-distributed $\gamma$ (LangFlow); linear rectified flow (ELF) ^47^ ^49^|
| Perplexity (OWT) | 15.36 (MDM-Prime v1) ^46^| 24.6 (LangFlow), ~24.1 (ELF-B) ^47^ ^51^|
| Training data efficiency | Standard (requires ~16× compute vs. AR) ^45^| ELF uses 10× fewer tokens than comparables ^50^|
| Classifier-free guidance | Not directly applicable | Native compatibility via continuous space ^49^|
| Key advantage | Mature ecosystem, strong at scale | Training efficiency, image-domain technique transfer |

The choice between discrete and continuous paradigms involves fundamental trade-offs. Discrete diffusion models benefit from a more mature training infrastructure and have demonstrated stronger results at the largest scales (LLaDA 8B, MDLM), while continuous models offer superior training data efficiency and native compatibility with established image-domain diffusion techniques such as classifier-free guidance. The field has not yet converged on which design is fundamentally superior ^52^ ^53^.

### 2.2 Masked Diffusion: The Dominant Paradigm

While both discrete and continuous formulations remain active research areas, a specific variant of discrete diffusion — **masked diffusion language models (MDLMs)** — has emerged as the dominant paradigm for production-scale diffusion language models. MDLMs simplify the general discrete diffusion framework by restricting the forward process to a single operation: masking tokens. Rather than allowing arbitrary token-to-token transitions through a full transition matrix, the forward process merely replaces each token with a special [MASK] token according to a learned or scheduled masking rate.

#### 2.2.1 MDLM: Simplifying Diffusion to Masked Language Modeling

MDLM (Masked Diffusion Language Models) demonstrated that a simplified formulation, where training reduces to masked language modeling with a learned masking schedule, achieves state-of-the-art results among diffusion models at the GPT-2 scale. The training objective becomes a straightforward cross-entropy loss: the model learns to predict the original tokens at masked positions given the unmasked context. At inference time, the model starts from a fully masked sequence and iteratively unmasks tokens according to a learned or heuristic schedule.

The practical appeal of MDLM is substantial. Because the training objective is structurally identical to BERT's masked language modeling objective, existing pretraining infrastructure — including data pipelines, distributed training frameworks, and optimizer configurations — requires minimal modification to implement masked diffusion. Furthermore, the masking schedule — determining what fraction of tokens are masked at each timestep — can be learned end-to-end or specified analytically, giving practitioners fine-grained control over the speed-quality trade-off during inference. This simplicity has made MDLM and its descendants the default choice for organizations building production diffusion language models.

#### 2.2.2 MD4: Forward-Backward Consistency and Learned Masking

MD4 (Masked Diffusion 4), presented at NeurIPS 2024 by Google DeepMind, advanced the MDLM framework with two key contributions. First, **mean parameterization** ensures mathematical consistency between the forward masking process and the reverse denoising process. In standard formulations, the model estimates the posterior distribution over clean tokens given a masked sequence, but this estimate may not be consistent with the actual forward process that generated the masked sequence. Mean parameterization constrains the reverse process to match the forward process in expectation, eliminating the training-inference mismatch that plagues simpler formulations where the model is trained to predict tokens under random masking but must generate under a structured unmasking schedule at inference time.

Second, **GenMD4** extends MD4 by optimizing state-dependent masking schedules via REINFORCE, allowing the model to learn which tokens to unmask at each step based on the current state of the partially decoded sequence. Rather than applying a fixed schedule that unmasks tokens at a predetermined rate, GenMD4 treats the masking decision as a policy optimization problem, using the REINFORCE gradient estimator to maximize expected generation quality. This learned schedule outperforms fixed schedules across standard benchmarks, particularly for longer sequences where early errors compound. Both MD4 and GenMD4 were open-sourced by Google DeepMind, establishing them as foundational reference implementations for the field.

#### 2.2.3 Block Diffusion: The Pragmatic Production Compromise

In practice, fully parallel generation of all tokens simultaneously — while theoretically appealing — creates substantial engineering challenges. Chief among these is the incompatibility with KV caching, the inference acceleration technique that stores key-value activations from previously generated tokens to avoid redundant computation. Pure diffusion models generate tokens in an any-order fashion, making standard KV caching impossible because the model cannot assume a fixed generation order.

**Block diffusion** addresses this by dividing sequences into fixed-size blocks (typically 32 tokens) and applying bidirectional attention within each block while maintaining causal attention between blocks. This semi-autoregressive design enables KV cache compatibility — the prefix blocks can be cached as in standard AR inference — while preserving parallel generation within each block. Fast-dLLM v2, BD3-LM, SDAR, and I-DLM all converge on block-wise diffusion as the most deployable architecture ^54^ ^55^. LLaDA2.0, Stable-DiffCoder, and Gemini Diffusion all employ block sizes of approximately 32 tokens at inference, suggesting this configuration represents a stable operational optimum rather than a temporary engineering workaround.

**Table 2.2: Evolution of Masked Diffusion Language Models**

| Model / Framework | Key Innovation | Contribution Type | Open Source | Institution |
|:---|:---|:---|:---|:---|
| MDLM (2024) | Simplified to masked LM with learned schedule | Foundation | Yes | Cornell / Kuleshov group |
| MD4 (NeurIPS 2024) | Mean parameterization; forward-backward consistency | Architecture | Yes | Google DeepMind |
| GenMD4 | State-dependent masking via REINFORCE | Learned inference | Yes | Google DeepMind |
| MDM-Prime v1 (NeurIPS 2025) | Partial masking; first MDM to surpass AR on OWT ^46^| Architecture | Yes | Academic |
| Block Diffusion (2025) | Bidirectional within blocks, causal between blocks | Engineering | Yes | Multiple |
| LLaDA (2025) | 8B-parameter open-source DLM; 2.3T training tokens | Scale | Yes | GSAI-ML |
| LLaDA2.0 (2025) | Block size 32; WSD conversion from AR models | Production | Yes | Ant Group |

The progression from MDLM to MD4 to production deployments like LLaDA2.0 and Gemini Diffusion illustrates a clear trend: the field is converging on block-wise masked diffusion as the canonical architecture for production language models. This convergence reflects a pragmatic compromise between the theoretical ideal of fully parallel generation and the engineering constraints of production serving infrastructure.

### 2.3 Remasking and Decoding Strategies

The defining characteristic of diffusion language models during inference is their iterative nature: tokens are not committed in a single forward pass but are generated through a sequence of denoising steps, each of which may revise previously unmasked tokens. This iterative refinement is both a strength — enabling error correction during generation — and a critical bottleneck, as the choice of which tokens to revise (remask) at each step profoundly impacts both output quality and generation speed.

#### 2.3.1 Remasking as the Critical Quality-Determining Step

The standard decoding strategy for masked diffusion models proceeds as follows: at each step, the model predicts logits for all currently masked positions, samples tokens from these distributions, and adds them to the partially decoded sequence. A **remasking** strategy then selects which tokens — including previously unmasked ones — should be re-masked before the next denoising iteration. This remasking decision is the single most important inference-time choice for diffusion language model quality.

The simplest and most widely used approach is **low-confidence remasking**: tokens whose predicted probability falls below a fixed or dynamic threshold are selected for revision. While intuitive, this approach has a significant limitation: a token may have high predicted probability yet still be incorrect if the surrounding context is itself unreliable. CoRe (Context-Robust Remasking) demonstrated that standard confidence-based strategies can actually **degrade** performance on code generation tasks ^56^.

CoRe addresses this by framing remasking as a distributionally robust optimization problem. Rather than ranking tokens by their raw confidence scores, CoRe performs a lightweight stress test: it evaluates whether each token remains strongly predicted when parts of its surrounding context are masked. Tokens whose likelihood drops substantially under context perturbation are flagged as "context-brittle" and prioritized for revision ^56^. This training-free framework achieves a **+9.2% improvement on MBPP** accuracy while adding only approximately 6% more forward passes than standard decoding ^56^. The gains are validated by compute-matched controls where random or margin-based revision yields negligible improvement, confirming that the context-robust selection criterion itself drives the quality improvement ^56^.

**STDD** (Spatio-Temporal Dynamics-Driven Token Refinement) takes a different approach by detecting each token's temporal variance (convergence status) and spatial deviance (inter-token correlations) to adaptively adjust the confidence threshold for every token at every step ^57^. When integrated with LLaDA-Instruct-8B, STDD achieves **8.9× speedup on MBPP** — far surpassing Fast-dLLM's 4.15× and DUS's 2.70× — while simultaneously improving accuracy from 79.2 to 83.1 on GSM8K ^57^. The key insight underlying STDD is that mainstream remasking strategies rely on a single global confidence threshold, overlooking the fact that individual tokens converge at different rates and have different dependency structures ^57^.

![Figure 2.1: Remasking Strategy Performance Comparison](/mnt/agents/output/fig_2_1_remashing_comparison.png)

The chart above illustrates the fundamental trade-off landscape in remasking strategy design. RemeDi achieves the highest raw accuracy through reinforcement learning-driven remasking but offers no speedup. STDD achieves the best efficiency-quality frontier, delivering both the highest speedup and strong accuracy. CoRe occupies a middle ground, trading modest compute overhead (~6%) for substantial quality gains on structure-sensitive tasks. The baseline confidence strategy is strictly dominated: it offers the lowest accuracy and no speed advantage.

#### 2.3.2 Key Finding: Confidence-Based Remasking Can Degrade Code Performance

CoRe's finding that "standard confidence-based remasking strategies (e.g., ReMDM) can degrade code performance in our experiments" represents a significant challenge to the field's default practice ^56^. Code generation is particularly sensitive to this issue because programming languages have rigid syntactic constraints — a single misplaced bracket or incorrect variable reference can render an entire function uncompilable. Confidence scores at the token level do not capture these structural dependencies. A token may be individually high-confidence (the model strongly predicts a closing brace) yet contextually wrong (the brace closes the wrong scope).

This limitation has driven interest in structure-aware alternatives. TreeDiff, for instance, incorporates Abstract Syntax Tree (AST)-aware masking for code generation, achieving a **13.3% relative improvement** over random masking on HumanEval+ by selectively masking tokens belonging to key AST nodes using a tiered weighting scheme ^58^ ^13^. Lower weights are assigned to structural elements (imports, function definitions) while higher weights target logic and control flow tokens, preserving program skeletons during the diffusion process ^58^.

#### 2.3.3 RemeDi: RL-Superior Remasking via Self-Reflection

**RemeDi** (Remasking-enabled Diffusion Language Model) represents the most sophisticated remasking approach developed to date, achieving **89.1% on GSM8K**, 52.9% on MATH, 73.2% on HumanEval, and 59.4% on MBPP — state-of-the-art among open-source diffusion language models at the time of publication ^59^. RemeDi's architecture jointly predicts token distributions and per-token confidence scores, with an Unmasking Policy Stream (UPS) attached to a base LLaDA model. At each diffusion step, high-confidence tokens are unmasked while low-confidence ones are re-masked, regardless of whether they have been previously unmasked ^59^.

The critical innovation is a **two-stage training pipeline**: Remask SFT followed by Remask RL. In the first stage, the model learns to identify and remask incorrect tokens while simultaneously predicting masked tokens, using sequences constructed by randomly masking tokens or replacing them with random alternatives. The second stage applies outcome-based reinforcement learning, optimizing entire generation trajectories toward higher rewards by learning how to remask and predict tokens at each step ^59^. This reinforcement learning stage is what elevates RemeDi above all other remasking approaches: the RL optimization discovers remasking policies that are not accessible through hand-designed heuristics. The UPS parameters use a higher learning rate ($2.0 \times 10^{-5}$) than original model parameters ($2.0 \times 10^{-6}$), enabling rapid adaptation of the remasking policy ^59^.

**Table 2.3: Remasking Strategy Comparison**

| Strategy | Type | Overhead | MBPP Gain | GSM8K Gain | Key Mechanism | Training-Free |
|:---|:---|:---|:---|:---|:---|:---|
| Confidence (baseline) | Low-confidence threshold | None | Baseline | Baseline | Fixed threshold on predicted probability | Yes |
| CoRe ^56^| Context-robust selection | ~6% forward passes | +9.2% | +3.3% | Perturbation-based brittleness detection | Yes |
| STDD ^57^| Spatio-temporal dynamics | None (speedup) | +5.6% | +3.9% | Per-token adaptive threshold from convergence/deviance | Yes |
| ReMDM | Principled resampling | ~1-2× baseline | +1.9% | +1.3% | Inference-time scaling via resampling | Yes |
| RemeDi ^59^| RL-trained policy | RL training stage | +9.2% | +9.9% | Two-stage SFT+RL with dedicated UPS stream | No |

The diversity of approaches and their significant quality impacts indicate that remasking strategy may matter as much as training for diffusion language model performance. Inference-time strategy design has become as active a research frontier as training methodology. Several broader trends emerge from this landscape: training-inference alignment is becoming a central theme (PAPL's planner-aware ELBO achieves up to 4× MAUVE improvement with a one-line code change) ^60^; post-training adaptation is proving remarkably effective (SCMDM achieves 50% perplexity reduction with only 3.25B post-training tokens) ^61^; and structure-aware approaches show particular promise for code generation where syntax constraints create unique challenges.

### 2.4 Any-Order Generation and Self-Correction

#### 2.4.1 Core Advantage: Iterative Refinement and Error Correction

The fundamental architectural advantage of diffusion language models over autoregressive models is the ability to generate tokens in any order and to revise previously generated tokens during the generation process. Autoregressive models commit to each token permanently once it is generated; errors propagate irreversibly down the sequence. Diffusion models, by contrast, can revisit and correct early errors as the sequence converges, a property that has been described as "self-correcting generation."

This property is particularly valuable for tasks where global consistency matters more than local fluency. Code generation exemplifies this: a function must have consistent variable names, matching brackets, and type-correct assignments across the entire body. The ability to generate a rough draft of the full function and then iteratively refine details is a natural fit for diffusion's any-order capability. Stable-DiffCoder's 60.0% accuracy on CanItEdit versus Seed-Coder's 50.5% — an 18.8% relative advantage — is attributed in part to this structural editing capability ^58^ ^13^.

#### 2.4.2 Self-Conditioning: Conditioning on Previous Predictions

**SCMDM** (Self-Conditioned Masked Diffusion Models) demonstrates that a lightweight post-training adaptation — conditioning each denoising step on the model's own previous clean-state predictions — can dramatically improve generation quality with minimal architectural modification. SCMDM achieves nearly a **50% reduction in generative perplexity** on OpenWebText-trained models: from 42.89 to 23.72 at 1000 sampling steps, using only 3.25B additional post-training tokens on a model pretrained for 262B tokens ^61^ ^62^.

The key technical insight is that in standard masked diffusion, if a token remains masked after a reverse update, the model discards its clean-state prediction for that position. Still-masked positions must therefore be repeatedly inferred from the mask token alone at each step. SCMDM carries the clean-state distribution forward from previous steps, enabling still-masked positions to be refined incrementally ^61^. Critically, SCMDM's experiments reveal that **full self-conditioning consistently outperforms partial self-conditioning** in the post-training regime. While partial conditioning (rate=0.5) improves over vanilla MDLM (generative perplexity 42.89 to 37.04), full self-conditioning achieves 23.72 — a massive additional improvement ^62^. This finding contradicts the "commonly used 50% dropout strategy" from continuous diffusion and reveals a refinement-specialization effect that had not been previously characterized ^61^.

At inference time, SCMDM requires no additional forward passes — the clean-state prediction from step $t+1$ is reused as the self-conditioning input at step $t$ — making it a zero-overhead post-training optimization ^61^. The method generalizes across domains, improving CIFAR-10 FID from 86.48 to 78.59 in discrete image synthesis and showing gains in small molecule generation and genomic sequence modeling ^61^.

#### 2.4.3 Theoretical Limitations: Linear Steps for Reasoning Tasks

Despite the practical advantages of any-order generation, theoretical analysis has identified important limitations. Feng et al. proved that masked diffusion models require a **linear number of steps** (in sequence length) to achieve a low sequence-level error rate on reasoning tasks, effectively eliminating the speed advantage of parallel generation for that task class. This result arises because reasoning tasks require each token to be conditionally dependent on the correct resolution of preceding logical steps; generating steps out of order creates dependency violations that can only be resolved through sufficient iterative refinement.

This theoretical finding aligns with empirical observations on competitive programming benchmarks. The Beyond Autoregression study found that diffusion models average 14.9% on LiveCodeBench versus 18.9% for autoregressive models ^12^— a gap attributed in part to the sequential reasoning demands of competitive programming problems, where each step in a proof or algorithm construction depends critically on the correct resolution of all prior steps. Conversely, diffusion models excel on tasks that benefit from global restructuring, such as code editing (CanItEdit) and long-context infilling, where the any-order generation capability provides genuine advantage over left-to-right AR generation by enabling the model to draft a complete solution and then revise structurally inconsistent portions.

The practical implication is that diffusion language models may find their strongest niche in applications requiring iterative refinement and global restructuring — code editing, document revision, creative writing — rather than in tasks demanding strictly sequential reasoning. Block diffusion architectures partially mitigate this limitation by imposing causal structure across blocks while maintaining parallel generation within each block, but the fundamental trade-off between parallelism and sequential dependency resolution remains an active area of investigation.

The tension between diffusion's bidirectional flexibility and the sequential structure of certain tasks has also motivated hybrid approaches. I-DLM (Introspective Diffusion Language Model) moves closer to autoregressive behavior by employing causal attention with logit shift, achieving the first diffusion language model to match autoregressive quality with only 4.5B training tokens ^55^. This convergence suggests that the architectural boundary between autoregressive and diffusion models is blurring, with future production systems likely to incorporate elements of both paradigms optimized for specific task requirements.
-e 


## 3. Google DeepMind: From MD4 to Gemini Diffusion

Google DeepMind's diffusion language model program represents one of the most systematic corporate research efforts to challenge autoregressive (AR) dominance in large language model (LLM) text generation. Beginning with foundational theoretical work in mid-2024, advancing through conversion methodology research, and culminating in the experimental release of Gemini Diffusion at Google I/O 2025, DeepMind has pursued a multi-pronged strategy that combines theoretical simplification, empirical transfer learning, and production-scale architecture engineering. This chapter examines each of these research streams in detail, tracing the intellectual lineage from the MD4 theoretical framework through the AR2Diff conversion paradigm to the production block diffusion architecture of Gemini Diffusion.

### 3.1 Research Lineage and Key Contributors

#### 3.1.1 A Compressed Timeline of Theoretical and Engineering Progress

DeepMind's diffusion text generation program developed across an exceptionally compressed eighteen-month window, from January 2024 to May 2025. Four milestones define this trajectory, each building upon or complementing the prior contributions.

The first milestone, AR2Diff (January 2024), established the feasibility of converting pretrained AR models into diffusion models through lightweight adaptation ^23^. This was followed by MD4 (June 2024, NeurIPS), which provided the foundational theoretical framework for masked diffusion on discrete data, including a dramatically simplified training objective and mean parameterization ^22^. Between MD4 and Gemini Diffusion, the CANDI framework (approximately December 2024) addressed the theoretical problem of coupling discrete and continuous diffusion processes, identifying and resolving what its authors termed "temporal dissonance" between discrete corruption and continuous denoising ^63^. Finally, Gemini Diffusion (May 2025, Google I/O) represented the first production-scale diffusion LLM from DeepMind, demonstrating 1,479 tok/s average throughput with block diffusion architecture ^3^.

| Milestone | Date | Venue | Core Contribution | Key Authors |
|---|---|---|---|---|
| AR2Diff | Jan 2024 | arXiv | AR-to-diffusion conversion via SUNDAE loss; prefix LM + decoder-only as optimal architecture | Kehang Han, Kathleen Kenealy, Aditya Barua, Noah Fiedel, Noah Constant ^23^|
| MD4 | Jun 2024 | NeurIPS 2024 | Simplified continuous-time ELBO to weighted cross-entropy integral; mean parameterization; GenMD4 state-dependent masking | Jiaxin Shi*, Kehang Han*, Zhe Wang, Arnaud Doucet, Michalis K. Titsias ^22^|
| CANDI | ~Dec 2024 | arXiv (Oct 2025) | Hybrid discrete-continuous diffusion; resolved temporal dissonance; outperforms masked diffusion at low NFE | Patrick Pynadath, Jiaxin Shi, Fuheng Zhang ^63^|
| Gemini Diffusion | May 2025 | Google I/O 2025 | Block diffusion with intra-block bidirectional + inter-block causal attention; 1,479 tok/s; ~5x faster than Flash-Lite | DeepMind Gemini team (Brendan O'Donoghue, Oriol Vinyals, Jack Rae) ^3^|

Table 3.1: DeepMind diffusion language model research timeline, January 2024–May 2025. Each milestone built directly upon prior work: AR2Diff established the conversion paradigm; MD4 provided the theoretical foundation; CANDI resolved discrete-continuous coupling; Gemini Diffusion integrated all prior insights into a production system.

This timeline reveals a deliberate research strategy: theoretical simplification (MD4) preceded empirical validation (AR2Diff's conversion experiments), which in turn preceded architectural integration (Gemini Diffusion). The CANDI contribution, though published later, filled a theoretical gap between MD4's discrete formulation and Gemini Diffusion's hybrid implementation. Specifically, CANDI addressed "temporal dissonance"—a phenomenon where continuous diffusion underperforms on discrete data because, at noise levels where discrete corruption preserves enough structure for conditional learning, continuous denoising is trivial, and vice versa ^63^. By decoupling discrete and continuous corruption, CANDI enabled simultaneous learning of conditional structure and continuous geometry, outperforming masked diffusion at low Number of Function Evaluations (NFE) ^63^. This property directly supports Gemini Diffusion's ability to achieve high-quality output with relatively few denoising steps.

Notably, Jiaxin Shi and Kehang Han appear as co-first authors on MD4, while Han also led AR2Diff, indicating DeepMind's investment in a cohesive research group rather than disconnected projects. The overlapping authorship across MD4, AR2Diff, and Gemini Diffusion suggests that theoretical insights flowed directly into engineering decisions, rather than being developed in isolation.

#### 3.1.2 Key Researchers and Institutional Vision

The intellectual leadership behind DeepMind's diffusion program spans theoretical research, engineering implementation, and executive sponsorship. Five individuals stand out as particularly influential.

**Jiaxin Shi**, research scientist at Google DeepMind and first author of MD4, is the program's primary theoretical architect. A Tsinghua PhD with postdoctoral experience at Stanford and Microsoft Research, Shi described masked diffusion in a 2024 LoG New York Meetup talk as "a simple and general framework that unlocks the full potential of diffusion models for discrete data" ^64^. His work on MD4 established the simplified variational objective and mean parameterization that likely underpin Gemini Diffusion's training.

**Kehang Han**, Shi's co-first author on MD4 and lead author of AR2Diff, directed the empirical validation of diffusion conversion at scale. Han's AR2Diff paper demonstrated that lightweight adaptation from AR checkpoints could produce competitive diffusion models at 280M–1.7B parameter scales, establishing the conversion paradigm later adopted by LLaDA2.0 and other systems ^23^.

**Brendan O'Donoghue**, research scientist at DeepMind and a lead on the Gemini Diffusion project, served as the program's primary public-facing technical authority. In a June 2025 interview, O'Donoghue articulated four major technical advantages of diffusion over AR generation: lower latencies through parallel generation, adaptive computation that scales with task difficulty, non-causal reasoning via bidirectional attention, and iterative self-correction during the denoising process ^21^. He also acknowledged two disadvantages: higher serving costs due to multiple forward passes per denoising step, and elevated Time-to-First-Token (TTFT) since "the first token can only appear when the entire sequence of tokens is ready" ^21^.

**Oriol Vinyals**, VP of Research and Deep Learning Lead at DeepMind and Co-Head of the Gemini project, provided executive-level endorsement. Vinyals stated, "It's been a dream of mine to remove the need for 'left to right' text generation," positioning diffusion as a long-term strategic direction rather than an isolated experiment ^65^. The model's demo at Google I/O ran so fast that the presentation team had to slow the video down to make it watchable ^65^.

**Jack Rae**, Principal Scientist at DeepMind, characterized Gemini Diffusion as a "landmark moment," noting that "until now, autoregressive models had consistently outperformed diffusion models in text quality, and it wasn't clear whether that gap could ever be closed" ^65^. This assessment frames Gemini Diffusion as achieving near-parity with production AR models for the first time at a major AI lab.

### 3.2 MD4: The Foundational Framework

MD4 (Masked Diffusion 4, subtitled "Simplified and Generalized Masked Diffusion for Discrete Data") was published at NeurIPS 2024 and provides the theoretical foundation for DeepMind's diffusion language model work ^22^. Its three technical contributions—a simplified training objective, mean parameterization, and state-dependent masking—each address critical bottlenecks in discrete diffusion training.

#### 3.2.1 Simplified Continuous-Time ELBO to Weighted Cross-Entropy

The central theoretical result of MD4 is that the continuous-time Evidence Lower Bound (ELBO) for masked diffusion models simplifies to a weighted integral of cross-entropy losses. Formally, the training objective becomes:

$$L = \int_0^1 w(t) \cdot \text{CE\_loss}(t) \, dt$$

where $w(t)$ is a time-dependent weighting factor related to the signal-to-noise ratio (SNR) at diffusion timestep $t$, and $\text{CE\_loss}(t)$ is the standard cross-entropy loss evaluated at the masking rate determined by $t$ ^22^.

This simplification is operationally significant. Prior masked diffusion formulations required specialized loss functions with complex per-timestep weighting schemes. MD4 showed that the theoretically correct objective is, in essence, cross-entropy averaged over masking schedules—an objective that requires no new infrastructure for teams already training AR language models. As the authors noted, the continuous-time variational objective reduces to "a simple weighted integral of cross-entropy losses" ^22^. This result lowered the barrier to entry for diffusion language model training and enabled the AR2Diff conversion methodology discussed in Section 3.3.

The weighting function $w(t)$ encodes the relative importance of different noise levels during training. At low noise (few masked tokens), the model learns fine-grained token prediction; at high noise (many masked tokens), the model learns coarse structure. The integral formulation ensures balanced learning across all noise regimes.

#### 3.2.2 Mean Parameterization Replacing Score Parameterization

MD4's second contribution introduces mean parameterization to replace the conventional score parameterization used in continuous diffusion models. Score parameterization, standard in diffusion models for continuous data (images, audio), estimates the gradient of the log-density with respect to the input. For discrete data—where gradients are undefined—this requires approximations that introduce training instability.

Mean parameterization directly parameterizes the expected value of the clean data given the noised state, ensuring consistency between the forward corruption process and the backward denoising process ^66^. This forward-backward consistency eliminates a class of training instabilities that plagued earlier discrete diffusion formulations, particularly at extreme noise levels where score estimates become unreliable. By predicting the mean rather than the score, MD4's parameterization naturally respects the discrete support of text tokens, avoiding the "score truncation" artifacts that occur when continuous score estimates are discretized.

#### 3.2.3 GenMD4: State-Dependent Masking via REINFORCE

The generalized MD4 framework (GenMD4) extends the basic formulation by allowing each token's unmasking probability to depend not only on the diffusion timestep but also on the token's identity. As the authors explain, "the probability of unmasking a token depends not only on time, but also on the token's value" ^67^. This means common tokens (e.g., punctuation, frequent words) and rare tokens (e.g., technical terms, proper nouns) can follow different corruption schedules optimized for their statistical properties.

The forward transition in GenMD4 is defined as $q(x_t | x_s) = \text{Cat}(x_t; Q(s,t)^T x_s)$, where $Q(s,t)$ incorporates state-dependent rates through an $\alpha_t$ vector function ^67^. The masking schedule parameters themselves are learned, not hand-designed, using a REINFORCE leave-one-out estimator to compute low-variance unbiased gradients ^67^. This is a notable departure from prior work where masking schedules were either uniform or followed fixed heuristics (e.g., cosine, linear).

Empirically, MD4 achieved state-of-the-art results among diffusion models at GPT-2 scale on 4 out of 5 zero-shot language modeling tasks on OpenWebText ^22^. On the character-level text8 benchmark, it attained the best Bits Per Character (BPC) result among diffusion models, while on CIFAR-10 it achieved 2.75 Bits Per Dimension (BPD)—better than autoregressive models of similar sizes. On ImageNet 64x64, it reached 3.40 BPD, comparable to larger Transformer AR models ^22^. These cross-domain results (text, character-level language, images at two resolutions) demonstrate that MD4's framework generalizes beyond any single modality.

The open-source JAX implementation (github.com/google-deepmind/md4) has enabled follow-on research across the broader diffusion LLM community. The repository includes full training and sampling algorithms for both text (OpenWebText) and image (CIFAR-10, ImageNet) datasets, with state-dependent masking schedule implementation using REINFORCE optimization ^22^. This release follows DeepMind's broader pattern of publishing research openly, though notably, the specific block diffusion architecture used in Gemini Diffusion, the AR2Diff implementation, and the CANDI codebase remain closed-source.

The practical significance of MD4's theoretical simplification extends beyond DeepMind. By showing that the correct training objective is fundamentally a weighted cross-entropy integral, MD4 enabled practitioners to adapt existing AR training infrastructure—data pipelines, optimization schedules, distributed training frameworks—with minimal modification. The mean parameterization eliminated the need for score estimation tricks that complicated earlier discrete diffusion implementations, while GenMD4's learned masking schedules removed the manual tuning burden of designing corruption schedules. These simplifications collectively lowered the activation energy required for the broader research community to experiment with diffusion language models.

### 3.3 AR2Diff: Transfer Learning from Autoregressive Models

AR2Diff ("Transfer Learning for Text Diffusion Models"), published in January 2024, addressed a practical question that MD4's theoretical framework enabled: can the vast computational investment in pretrained AR models be transferred to diffusion models? ^23^#### 3.3.1 Three-Stage Conversion via SUNDAE Loss

The AR2Diff methodology prescribes a three-stage conversion pipeline. First, an AR decoder is pretrained with causal attention on a large text corpus. Second, this checkpoint is continued as a diffusion model with bidirectional attention enabled—this is the critical architectural modification, as it allows tokens to attend to future positions within the same training context. Third, the model is fine-tuned as a diffusion model on downstream tasks ^23^. The paper denotes models by the number of additional pretraining steps in stage two (AR2Diff_N, where N ranges from 0 to 100K).

The diffusion training uses a simplified variant of the SUNDAE (Structured Unified Noise Denoising Autoencoder) text diffusion loss ^23^. SUNDAE itself builds upon the MD4 theoretical framework by providing a concrete non-AR training objective that preserves the pretrained knowledge embedded in AR checkpoints. The key architectural change—enabling bidirectional attention during diffusion training—transforms a causal decoder into a full bidirectional encoder-decoder during the diffusion phase.

Models were tested at three scales: Base (280M parameters), Large (approximately 700M parameters), and XL (1.7B parameters), using a pretraining mixture of 80% multilingual web pages and 20% Python code ^23^.

#### 3.3.2 Optimal Architecture: Decoder-Only with Prefix LM Objective

Through extensive architectural ablations, AR2Diff identified a consistent winner: "training a decoder-only model with a prefix LM objective is best or near-best across several tasks" ^68^. This finding directly informed subsequent DeepMind diffusion architectures, including Gemini Diffusion's design.

| Method | Size | WMT14 En-Fr (BLEU) | SQuAD (F1) | MBPP (Pass@80%) |
|---|---|---|---|---|
| Autoregressive | Base (280M) | 33.27 | 68.11 | 5.5 |
| Diffusion (from scratch) | Base | 29.83 | 77.41 | 12.2 |
| AR2Diff_0 | Base | 29.62 | 64.77 | 1.1 |
| AR2Diff_10K | Base | 29.41 | 68.12 | 4.4 |
| AR2Diff_100K | Base | 29.92 | 71.87 | 7.7 |
| Autoregressive | Large (~700M) | 34.92 | 78.43 | 15.5 |
| Diffusion (from scratch) | Large | 29.36 | 80.56 | 12.2 |
| AR2Diff_100K | Large | 32.20 | 80.71 | 10.0 |
| Autoregressive | XL (1.7B) | 35.48 | 84.08 | 15.5 |
| Diffusion (from scratch) | XL | 29.30 | 82.78 | 18.8 |
| AR2Diff_100K | XL | 32.55 | 83.54 | 15.5 |

Table 3.2: AR2Diff performance comparison across three model scales and training methodologies ^23^. Diffusion models (trained from scratch) outperform AR on SQuAD and MBPP at all scales. AR2Diff_N improves monotonically with conversion steps N, approaching AR quality on generation tasks while retaining diffusion advantages on discriminative tasks.

The table reveals several important patterns. On SQuAD (reading comprehension), diffusion models consistently outperform AR: 77.41 vs. 68.11 at Base, 80.56 vs. 78.43 at Large, and 82.78 vs. 84.08 (near parity) at XL. On MBPP (code synthesis), diffusion achieves 18.8% at XL compared to AR's 15.5%—a 21% relative improvement ^23^. However, on WMT14 En-Fr (machine translation), AR maintains a consistent advantage across all scales, suggesting that generation tasks requiring strict left-to-right coherence benefit less from diffusion's bidirectional structure.

The AR2Diff_N models show monotonic improvement with additional conversion steps (N), with AR2Diff_100K approaching or exceeding the AR baseline on discriminative tasks (SQuAD) while remaining below it on pure generation tasks (WMT). This suggests that the conversion process preserves AR knowledge for generation while unlocking diffusion-specific capabilities for bidirectional understanding.

#### 3.3.3 Significance: The Conversion Paradigm

AR2Diff's most lasting contribution is establishing that AR pretraining investment transfers to diffusion. This finding created the "conversion paradigm" adopted by LLaDA2.0 (converting Ling models), ByteDance's Stable-DiffCoder (converting Seed-Coder), and Apple's DiffuLLaMA (converting LLaMA-2) ^23^. Rather than training diffusion models from scratch—a computationally expensive proposition—practitioners can leverage existing AR checkpoints and continue training with diffusion objectives.

The inference speed analysis further supports diffusion's practical potential: "as the decoding sequence length increases from 500 tokens (e.g., MBPP task) to 4,000 tokens, the speedup gained by diffusion (using 10 steps) increases from 10x to 30x" ^23^. However, the paper also notes a caveat: a single AR step (14ms/token) was still faster than a single diffusion step (179ms/step) in their implementation, due to the lack of KV caching for diffusion—a limitation that subsequent work on block diffusion and Fast-dLLM has addressed.

AR2Diff's conversion paradigm carries broader strategic implications for the competitive landscape of foundation models. The finding that AR pretraining investment transfers to diffusion creates a structural advantage for organizations that have already invested in large-scale AR training—Google (Gemini), Ant Group (Ling/LLaDA), and ByteDance (Seed-Coder)—while making it harder for new entrants without strong AR base models to compete. The "moat" of expensive AR pretraining does not disappear in a diffusion future; it transfers. This dynamic explains why the most prominent open-source diffusion LLMs (LLaDA2.0, Stable-DiffCoder, DiffuLLaMA) all originate from organizations with substantial prior AR training infrastructure rather than from pure-play diffusion startups.

### 3.4 Gemini Diffusion: Production Deployment

Gemini Diffusion, announced at Google I/O on May 20, 2025, represents DeepMind's attempt to translate eighteen months of theoretical and empirical research into a production-scale diffusion language model ^3^. It is the first diffusion LLM from a major AI lab to achieve near-parity with production AR models on real tasks. Its name positions it within the Gemini family, but its underlying architecture—block diffusion with iterative denoising—differs fundamentally from the autoregressive Transformers that power Gemini 2.5 Pro and Flash.

#### 3.4.1 Block Diffusion Architecture

Gemini Diffusion's defining architectural innovation is **block diffusion**: a hybrid attention pattern that combines intra-block bidirectional attention with inter-block causal attention ^38^. Within each block (typically 32 tokens), every position can attend to every other unmasked position, enabling non-causal reasoning and global error correction. Between blocks, standard causal masking preserves the autoregressive structure needed for KV cache compatibility.

This design represents a pragmatic compromise between the ideal of fully parallel generation (all tokens simultaneously) and the operational reality of existing inference infrastructure. As O'Donoghue explained, the bidirectional attention "allows non-causal reasoning to take place and allows the model to make global edits within a block to produce more coherent text" ^21^. The architecture also incorporates a U-Net-like encoder-decoder structure with skip connections to preserve low-level information across layers ^69^.

The block diffusion architecture enables two capabilities that fully parallel diffusion lacks: KV caching between blocks (reducing memory pressure during inference) and streaming generation (outputting completed blocks while subsequent blocks are still being processed). These properties make block diffusion compatible with existing LLM serving stacks, a crucial consideration for production deployment.

The convergence on block diffusion across all major production diffusion LLMs—Gemini Diffusion, LLaDA2.0 (block size 32), Stable-DiffCoder, and Inception Labs' Mercury—suggests this is not a temporary architectural compromise but the permanent production paradigm ^38^. Block sizes of approximately 32 tokens have emerged as a de facto standard, large enough to enable meaningful parallel computation within each block while small enough to limit the coordination problem that plagues fully parallel generation. The U-Net encoder-decoder structure with skip connections, adopted from image diffusion architectures, enables information to flow across layers without degradation—a property particularly important for maintaining coherence in long-form text generation ^69^.

#### 3.4.2 Performance Specifications

Gemini Diffusion achieves 1,479 tokens/second average throughput across evaluated tasks, with overhead (TTFT) of 0.84 seconds from prompt input to generation start ^3^. On programming tasks, it reaches up to 2,000 tokens/second even accounting for tokenization, prefill, and safety checks ^65^. DeepMind reports this as approximately 5x faster than Gemini 2.0 Flash-Lite ^21^.

On code generation benchmarks, Gemini Diffusion demonstrates near-parity with Flash-Lite: 89.6% on HumanEval (vs. Flash-Lite's 90.2%), 76.0% on MBPP (vs. 75.8%), and 45.4% on BigCodeBench (vs. 45.8%) ^3^. The 30.9% score on LiveCodeBench (v6) actually exceeds Flash-Lite's 28.5% ^3^. These results support O'Donoghue's claim that "the gap between the two techniques is essentially closed in terms of benchmark performance, at least at the relatively small sizes we have scaled up to" ^21^.

A second defining operational feature is **adaptive computation**: the number of denoising steps automatically adjusts to task complexity. As O'Donoghue explained, "diffusion models will converge to a sequence of tokens at different rates depending on the task's difficulty. This allows the model to consume fewer resources (and have lower latencies) on easy tasks and more on harder ones" ^21^. Unlike autoregressive models, which expend identical compute per token regardless of whether the task is trivial or complex, diffusion models can terminate early when the sequence has converged to a stable solution. This property is theoretically unique to diffusion and represents a potential efficiency advantage that has not yet been fully exploited in production systems.

A third defining feature is **iterative self-correction**: tokens sampled during the denoising process can be revised in subsequent steps. As O'Donoghue described, "the denoising process involves sampling, which can introduce errors just like in autoregressive models. However, unlike autoregressive models, the tokens are passed back into the denoiser, which then has an opportunity to correct the error" ^21^. This property makes diffusion particularly suited for text editing applications, where O'Donoghue noted "diffusion models are uniquely applicable for scenarios where text needs to be modified in-place, such as grammar correction, adapting content for different personas, or integrating SEO keywords directly into existing drafts" ^70^. Gemini Diffusion's "Instant Edit" mode enables precisely this workflow: users paste existing text and edit it in real-time with minimal prompting ^21^.

#### 3.4.3 Performance Gaps: The Coordination Problem

Despite strong code generation results, Gemini Diffusion exhibits significant deficits on benchmarks requiring deep reasoning, scientific knowledge, or multilingual capability.

| Benchmark | Category | Gemini Diffusion | Gemini 2.0 Flash-Lite | Gap (pp) |
|---|---|---|---|---|
| LiveCodeBench (v6) | Code | 30.9% | 28.5% | +2.4 |
| BigCodeBench | Code | 45.4% | 45.8% | -0.4 |
| LBPP (v2) | Code | 56.8% | 56.0% | +0.8 |
| HumanEval | Code | 89.6% | 90.2% | -0.6 |
| MBPP | Code | 76.0% | 75.8% | +0.2 |
| GPQA Diamond | Science | 40.4% | 56.5% | -16.1 |
| AIME 2025 | Mathematics | 23.3% | 20.0% | +3.3 |
| BIG-Bench Extra Hard | Reasoning | 15.0% | 21.0% | -6.0 |
| Global MMLU (Lite) | Multilingual | 69.1% | 79.0% | -9.9 |

Table 3.3: Gemini Diffusion vs. Gemini 2.0 Flash-Lite benchmark comparison ^3^. Code generation shows near-parity (average gap: +0.5pp), while science reasoning (GPQA Diamond), complex reasoning (BIG-Bench Extra Hard), and multilingual tasks (Global MMLU) show substantial deficits. AIME 2025 is an exception where diffusion exceeds AR.

The 16.1 percentage point gap on GPQA Diamond (40.4% vs. 56.5%) is the largest deficit and the most significant barrier to diffusion adoption for scientific applications. Research on the "coordination problem" in parallel generation provides a theoretical explanation: "Think First, Diffuse Fast" demonstrated that diffusion models suffer from a coordination problem on multi-step reasoning—AR models build coherence token-by-token, while diffusion must coordinate all positions simultaneously ^71^. The same study showed that plan conditioning (using an AR model to generate a plan that the diffusion model follows) improves diffusion LLM reasoning by +11.6 percentage points on GSM8K ^71^, suggesting that diffusion models need external sequential guidance for complex reasoning.

![Figure 3.1: Gemini Diffusion vs. Gemini 2.0 Flash-Lite Benchmark Performance Gaps](/mnt/agents/output/fig3_1_gemini_diffusion_benchmarks.png)

The visualization reveals a clear pattern: Gemini Diffusion excels at tasks where parallel processing and iterative refinement provide advantage (code generation, mathematics) while underperforming on tasks requiring sequential multi-step reasoning (science, complex reasoning) and fine-grained token-level control across diverse languages. The near-zero average gap on code benchmarks (+0.5pp) contrasts sharply with the double-digit deficits on science (-16.1pp), reasoning (-6.0pp), and multilingual (-9.9pp) tasks. This pattern suggests that diffusion's parallel generation paradigm is well-suited for structured outputs like code (where syntax enforces global consistency) but less suited for open-ended reasoning (where each step depends on the prior).

The AIME 2025 mathematics result (+3.3pp over Flash-Lite) is a notable exception. Mathematics problems, while requiring reasoning, have well-defined structure and verifiable answers—properties that may benefit from diffusion's iterative refinement. The model can sample multiple solution paths and correct errors during denoising, a capability less applicable to open-ended science questions where the reasoning chain itself is the answer.

The 9.9 percentage point gap on Global MMLU Lite (69.1% vs. 79.0%) raises questions about diffusion's suitability for multilingual tasks. Multilingual evaluation requires fine-grained token-level control across languages with diverse morphological structures, and bidirectional attention within blocks may not equally benefit all language families. Languages with agglutinative morphology (e.g., Turkish, Japanese) or extensive compounding (e.g., German) may require more sequential processing than block-parallel generation can provide. Additionally, if Gemini Diffusion was trained with a different multilingual data mixture than Flash-Lite, the gap may reflect data distribution differences as much as architectural limitations.

Understanding why diffusion excels at code but struggles with science and multilingual tasks is critical for guiding future architecture development. Code has deterministic syntax and verifiable semantics—properties that align with diffusion's iterative refinement and global consistency checking. Science questions, by contrast, require open-ended multi-hop reasoning across unstructured knowledge, a task where sequential chain-of-thought generation provides clear advantages. This task-dependent performance profile suggests that the future of text generation may not be a single architecture but rather a hybrid ecosystem where diffusion and AR models serve different use cases.

#### 3.4.4 Current Status: Research Preview, Not Product

Gemini Diffusion remains available only through an experimental waitlist since its May 2025 announcement, with no production API ^3^. DeepMind's official positioning is explicit: "Gemini Diffusion is currently available as an experimental demo to help develop and refine future models" ^3^. This framing distinguishes Gemini Diffusion from production Gemini variants (2.5 Pro, 2.5 Flash) and places it in the research pipeline rather than the product catalog.

The experimental status reflects both the performance gaps documented above and practical deployment challenges. O'Donoghue acknowledged "higher cost of serving" as a fundamental disadvantage, since each denoising step requires a full forward pass through the model ^21^. The 0.84-second TTFT overhead makes diffusion uncompetitive for short generations where AR models can produce the first token immediately. These infrastructure challenges, combined with the -16.1pp science reasoning gap, likely contribute to Google's cautious rollout.

Several contextual factors frame DeepMind's strategic positioning. First, Google has successfully deployed diffusion for other modalities—images (Imagen, Nano Banana) and video (Veo 3)—establishing internal expertise that transfers to text. Second, leadership statements from Vinyals and Rae indicate long-term commitment beyond the current experimental release. Third, the broader industry context includes Inception Labs raising $50M for its Mercury diffusion LLM and the emergence of open-source alternatives (LLaDA 2.0 at 100B parameters, Dream 7B), suggesting that diffusion LLMs are transitioning from research curiosity to competitive product category ^72^ ^73^.

The gap between Gemini Diffusion and Gemini 2.5 Pro remains substantial: on GPQA Diamond, Pro achieves 83.0% versus Diffusion's 40.4%; on AIME 2025, Pro scores 83.0% versus Diffusion's 23.3% ^74^ ^75^. These comparisons suggest that diffusion's current value proposition is speed and editing capability, not frontier model quality. For applications where 1,479 tok/s throughput and inline editing outweigh the need for deep reasoning, Gemini Diffusion offers a compelling alternative. For science, complex reasoning, and multilingual tasks, autoregressive models retain a decisive advantage that diffusion architectures have yet to close.

Looking forward, DeepMind's diffusion program appears positioned along two potential trajectories. The first is continued refinement of block diffusion as a specialized system for code generation and text editing, where it already achieves competitive quality at superior speed. The second—more ambitious—path involves closing the reasoning gap through hybrid approaches: plan conditioning (using AR models to generate reasoning plans that diffusion executes), anchored diffusion (constraining certain positions to guide generation), or alternating AR and diffusion steps within a single generation. The presence of MD4 authors on the Gemini Diffusion team suggests that theoretical innovations from the research pipeline will continue to inform production architecture decisions, maintaining the tight coupling between foundational research and engineering implementation that has characterized DeepMind's approach to date.
-e 


## 4. Ant Group: The LLaDA Ecosystem

No single organization has committed more engineering resources or released more open-source artifacts for diffusion language models than Ant Group. Through its InclusionAI initiative, the Alibaba-affiliated fintech giant has produced the LLaDA model family — the only open-source diffusion LLMs scaled beyond 100B parameters — alongside a complete toolchain spanning training (dFactory), inference (dInfer), and serving integration (SGLang). This chapter examines the full scope of Ant Group's diffusion LLM program: the organizational philosophy driving open-source publication, the technical innovations that enabled LLaDA2.0 to reach parity with autoregressive (AR) baselines at 100B scale, the token-editing and reinforcement learning advances in LLaDA2.1, the developer-facing CodeFuse ecosystem, and the academic collaboration network that underpins the research.

### 4.1 Inclusion AI and the Open-Source Strategy

#### 4.1.1 AGI-as-Public-Good Philosophy

InclusionAI, Ant Group's open-source AI research division, operates under a philosophy articulated by CTO He Zhengyu: "AGI should be a public good — a shared milestone for humanity's intelligent future." ^22^This framing is deliberately inclusive, positioning Ant Group's releases as contributions to global science rather than competitive moats. All models are released under the MIT license, with model weights distributed through Hugging Face and ModelScope, full deployment support via vLLM and SGLang, and OpenAI-compatible API endpoints provided through third-party hosting services ^76^.

The strategic rationale for this openness is pragmatic as well as ideological. As noted by AI researcher Nathan Lambert, InclusionAI recognizes that "Western companies likely won't pay for their services, so having open models is their only open door to meaningful adoption and influence." ^77^In a landscape dominated by well-capitalized Western closed-source providers, open-source publication becomes a distribution strategy — a way to ensure Ant Group's architectural choices and engineering standards propagate through the global developer community.

InclusionAI organizes its research into three model families, each targeting a distinct capability tier ^22^:

- **Ling** (灵): Efficiency-focused sparse Mixture-of-Experts (MoE) language models, designed for high-throughput inference with minimal active parameters per token.
- **Ring**: Advanced reasoning models featuring explicit chain-of-thought pathways, competitive with frontier thinking models from OpenAI and DeepSeek.
- **Ming** (明): Native omnimodal systems processing text, image, audio, and video within unified architectures, including Diffusion Transformer (DiT)-based image generation ^78^.

This tripartite structure ensures that diffusion research (primarily within the Ling line, from which LLaDA models are derived) sits alongside complementary efforts in reasoning and multimodal understanding, enabling cross-pollination of techniques.

#### 4.1.2 Rapid Iteration: Six Ling Versions in Twelve Months

The pace of Ant Group's model releases rivals that of any frontier lab globally. Between April 2025 and April 2026, the Ling family underwent six major iterations ^77^: Ling-Plus (293B sparse MoE) marked the organization's entry into the open foundation model race; Ling 1.5 delivered a substantial capability upgrade in July 2025; Ling 2.0 / Ring 2.0 (September–October 2025) introduced three model sizes under a unified MoE architecture guided by empirical scaling laws; Ling-2.5-1T and Ring-2.5-1T (February 2026) pushed context windows to one million tokens and achieved International Mathematical Olympiad (IMO) 2025 Gold Medal standard ^79^; finally, Ling-2.6-flash (April 2026) arrived after anonymous testing as "Elephant Alpha" on OpenRouter, trending at #1 with over 100 billion daily token calls ^72^.

Ring-2.5-1T's mathematical reasoning capabilities are particularly noteworthy: the model scored 35/42 on IMO 2025 (Gold Medal threshold) and 105/126 on the Chinese Mathematical Olympiad (CMO) 2025, surpassing China's national team cutoff ^79^. These results establish Ant Group as a genuine competitor in frontier reasoning, not merely an efficiency-focused engineering shop.

#### 4.1.3 The Complete Toolchain: dFactory, dInfer, and SGLang

Releasing model weights alone does not make a model practically usable. Ant Group has invested heavily in the surrounding infrastructure to bridge the gap between research artifact and deployable system. The toolchain comprises three components:

**dFactory** is the distributed training framework, built on the VeOmni distributed training backend, providing optimized implementations for all stages of diffusion LLM training — from continual pre-training (CPT) through supervised fine-tuning (SFT) and reinforcement learning (RL) ^26^. dFactory supports data packing (concatenating multiple short sequences for throughput), specialized block-diffusion attention implementations, and the Multi-Turn Forward (MTF) augmentation pipeline introduced in LLaDA2.1 ^17^.

**dInfer** is a custom inference engine specifically adapted for block diffusion decoding, providing low-latency serving with KV-cache reuse, tensor parallelism, and CUDA graph optimization ^80^. dInfer handles the unique requirements of diffusion generation — block-wise causal masked attention, parallel token acceptance within blocks, and threshold-based decoding — that standard inference engines designed for AR models do not natively support.

**SGLang integration** represents the deployment layer. The LMSYS team provided day-zero support for LLaDA2.0 block diffusion inference in December 2025 ^9^ ^8^, and a customized SGLang version jointly developed by Ant Group and the SGLang team serves as LLaDA2.1's production inference engine ^17^. This collaboration extends SGLang's Radix caching and batching support to block diffusion LLMs, optimizing memory usage and throughput for concurrent requests ^17^.

The existence of this complete toolchain — training, inference, and serving — distinguishes Ant Group's diffusion program from every other open-source or commercial diffusion LLM effort. While competitors publish model weights, none provide the same depth of engineering infrastructure for production deployment.

### 4.2 LLaDA2.0: Scaling Diffusion to 100B Parameters

#### 4.2.1 Architecture: The First 100B Diffusion LLM

LLaDA2.0, published in December 2025 (arXiv:2512.15745v2), represents the first successful scaling of discrete diffusion language models to 100B total parameters ^5^ ^4^. The model is not trained from scratch; instead, it is produced by systematically converting pretrained autoregressive base models from the Ling 2.0 family into diffusion models through a three-phase Warmup-Stable-Decay (WSD) training strategy. This conversion approach is approximately seven times more compute-efficient than training an equivalent dense model from scratch ^81^.

Two model variants are released:

| Parameter | LLaDA2.0-mini | LLaDA2.0-flash |
|:---|:---:|:---:|
| Total parameters | 16B | 100B |
| Active parameters (MoE) | 1.4B | 6.1B |
| Non-embedding active | 789M | 4.8B |
| Routed experts | 256 | 256 |
| Shared experts | 1 | 1 |
| Experts activated/token | 8 | 8 |
| Activation ratio | 1/32 | 1/32 |
| MoE intermediate size | 512 | 512 |
| Routed scaling factor | 2.5 | 2.5 |
| License | Apache 2.0 ^25^| Apache 2.0 ^25^|

**Table 4.1: LLaDA2.0 Architecture Specifications.** Both variants use sigmoid-based, auxiliary-loss-free routing with MTP (Multi-Token Prediction) layers, QK-Norm, and Partial-RoPE. The 1/32 activation ratio — meaning only one of every 32 parameters participates in computing any given token — was identified as optimal through Ling Scaling Laws, small-scale experiments fitted to power-law predictions before committing GPUs to full-scale training ^82^. At 6.1B active parameters, LLaDA2.0-flash is computationally comparable to a ~40B dense model while offering the memory and throughput advantages of extreme sparsity. Both models are available on HuggingFace and ModelScope under the Apache 2.0 license, alongside training code (dFactory), inference engine (dInfer), and comprehensive technical reports ^25^ ^26^.

#### 4.2.2 WSD Three-Phase Training

The Warmup-Stable-Decay (WSD) strategy is the core technical innovation enabling smooth AR-to-diffusion conversion. It decomposes the transition into three coordinated phases that progressively expand and then contract the model's exposure to bidirectional context ^5^ ^83^.

**Phase 1: Warmup (Progressive Block Size Expansion).** Starting from the AR base model where block size equals 1 (standard autoregressive generation), training proceeds through a sequence of block size transitions: 1 → 4 → 32 → 64 → 4096. At each transition, the model is trained on "moderate-scale data" to ensure smooth adaptation ^5^. When block size reaches 4096, the Block Diffusion Language Model (BDLM) becomes equivalent to a standard Masked Diffusion Language Model (MDLM) with full-sequence bidirectional denoising. This progressive enlargement allows internal representations to adapt to larger contextual spans and more complex masking patterns without catastrophic forgetting.

**Phase 2: Stable (Large-Scale MDLM Training).** With block size fixed at 4096, the model trains on large-scale corpora to deepen its understanding of diffusion dynamics. A critical optimization at this stage: because the full sequence is processed as a single block, the "clean" part of the attention computation no longer requires the complex block-wise masks used in Phase 1, significantly reducing computational cost ^5^. The Ling base models were trained on over 20 trillion tokens ^84^, suggesting that the diffusion conversion process requires substantially less data than full pretraining from scratch — a key efficiency advantage of the AR-to-diffusion paradigm. The pretraining backend uses Megatron-LM with five-dimensional parallelism: data parallelism (DP), pipeline parallelism (PP), tensor parallelism (TP), context parallelism (CP), and expert parallelism (EP) ^5^ ^85^. A cuDNN attention backend achieves greater than 1.3× end-to-end speedup and over 90% memory savings in the attention layer compared to unfused TransformerEngine attention ^5^, while a zig-zag partitioning strategy balances the block-diffusion attention mask workload across the context parallelism group ^5^.

**Phase 3: Decay (Block Size Reduction for Inference).** The model gradually reduces block size from 4096 through intermediate values down to 32. This decay process "distills the global contextual knowledge learned during MDLM into a compact blockwise structure" ^5^. The final block size of 32 was chosen as the optimal quality-speed tradeoff: ablation studies show that block size 16 yields the highest score (70.26) but slowest throughput (2.44 tokens per forward pass, TPF), while block size 64 degrades both quality and speed ^4^.

**Training stability mechanism.** During the AR-to-diffusion transition, gradient explosion can occur at high mask ratios because masked token embeddings decay toward zero during AR pretraining — masked tokens are never observed by the AR model. LLaDA2.0 addresses this by adding independent Gaussian noise to the embedding layer output for masked tokens during initial iterations, ensuring that the L2 norm of masked token embeddings remains significant and stabilizing the training process ^5^. This approach avoids the alternative of randomly reinitializing masked token embeddings, which would cause catastrophic forgetting of pretrained knowledge.

**Document-level attention mask.** A specialized block-wise attention mask prevents cross-document semantic contamination when packing heterogeneous documents during training. For a concatenated sequence comprising noisy tokens $x_t$ followed by clean tokens $x_0$, the attention mask $M \in \{0,1\}^{2L \times 2L}$ encodes three constraints: block-diagonal attention within the noisy sequence (tokens attend only to others in the same block), offset block-causal cross-attention from noisy to clean tokens, and block-causal attention within the clean sequence ^5^ ^86^. For full MDLM training (block size = 4096), this simplifies to a document-level mask where attention operates strictly within document boundaries. The paper finds this mechanism "more fundamental" than complementary techniques such as random-length masking or CART for achieving stable bidirectional diffusion training ^87^.

#### 4.2.3 Benchmark Results: Parity with Strong AR Models

Across 47 benchmarks organized into five evaluation dimensions — knowledge (10 benchmarks), reasoning (12), coding (13), mathematics (9), and agent/alignment (4) — LLaDA2.0-flash achieves an average score of 73.18, compared to 73.60 for Qwen3-30B-A3B-Instruct-2507 and 72.15 for Ling-flash-2.0 ^4^ ^5^. The 0.42-point gap versus Qwen3-30B demonstrates fundamental parity: diffusion models at 100B scale can match comparably sized AR models across a broad capability spectrum.

![Figure 4.1: LLaDA2.0-flash Benchmark Performance vs. Autoregressive Baselines](/mnt/agents/output/fig4_1_llada2_benchmarks.png)

The pattern within these aggregates reveals task-specific advantages. LLaDA2.0-flash leads decisively on coding benchmarks: HumanEval at 94.51 (versus 93.29 for Qwen3-30B and 85.98 for Ling-flash-2.0), MBPP at 88.29 (versus 86.65), and LiveCodeBench at 42.29 (versus 41.63) ^4^. It also leads on agent tasks, with BFCL v3 at 75.43 — surpassing all AR baselines including Qwen3-30B at 73.19 ^4^. Mathematics performance is strong: GSM8K at 96.06, MATH at 95.44, and AIME 2025 at 60.00, though Qwen3-30B maintains a slight edge on the most challenging competition problems (61.88 on AIME 2025) ^4^.

| Benchmark | Qwen3-30B-A3B | Ling-flash-2.0 | LLaDA2.0-flash | Leader |
|:---|:---:|:---:|:---:|:---:|
| **Average (47 tasks)** | **73.60** | 72.15 | 73.18 | Qwen3-30B |
| MMLU | 87.13 | 87.98 | 87.69 | Ling-flash-2.0 |
| HumanEval | 93.29 | 85.98 | **94.51** | LLaDA2.0-flash |
| MBPP | 86.65 | 85.01 | **88.29** | LLaDA2.0-flash |
| LiveCodeBench | 41.63 | 44.11 | 42.29 | Ling-flash-2.0 |
| GSM8K | 96.36 | 95.45 | 96.06 | Qwen3-30B |
| MATH | 96.70 | 96.10 | 95.44 | Qwen3-30B |
| AIME 2025 | **61.88** | 55.89 | 60.00 | Qwen3-30B |
| BFCL v3 | 73.19 | 67.57 | **75.43** | LLaDA2.0-flash |
| IFEval | **86.90** | 76.16 | 80.78 | Qwen3-30B |

**Table 4.2: LLaDA2.0-flash Benchmark Comparison.** Bold values indicate the highest score in each row. LLaDA2.0-flash leads on four of the nine highlighted benchmarks, with particularly strong advantages in coding (HumanEval, MBPP) and agent tasks (BFCL v3). The preview model (trained without full post-training) scores only 23.33 on AIME 2025 and 29.07 on LiveCodeBench ^4^, demonstrating that post-training — supervised fine-tuning, CAP training, and DPO — contributes a substantial portion of the final model's capabilities, not merely the base diffusion conversion. Areas of weakness include SciBench (4.13) and HARDMath2 (4.27), extremely difficult benchmarks where all compared models struggle ^4^.

The pattern of strengths and weaknesses carries strategic implications. LLaDA2.0-flash's coding dominance (94.51 on HumanEval, best among all compared models) and agent-task leadership (75.43 on BFCL v3) suggest that diffusion models possess structural advantages in domains where parallel information access and iterative refinement matter more than strict sequential dependency. Code generation, in particular, benefits from the ability to consider multiple function signatures, variable names, and control-flow structures simultaneously rather than committing to each token in lockstep order. Conversely, the model's slight underperformance on instruction following (IFEval at 80.78 versus Qwen3-30B's 86.90) and hardest mathematical reasoning (AIME 2025 at 60.00 versus 61.88) points to residual challenges in tasks requiring extended chain-of-thought reasoning where sequential accumulation of context provides decisive advantages. This task-specific performance profile suggests that diffusion and autoregressive models may occupy complementary niches rather than competing as zero-sum replacements.

#### 4.2.4 Inference Speed: 535 TPS and the 2.1× Speedup

Inference speed represents the most commercially significant advantage of diffusion LLMs, and LLaDA2.0 delivers measurable gains under controlled conditions. LLaDA2.0-flash-CAP achieves 535 tokens per second (TPS) on benchmark tasks, compared to 256 TPS for Ling-flash-2.0 and 237 TPS for Qwen3-30B-A3B-Instruct-2507 under identical serving configurations (SGLang with tensor parallelism of 8 on H20 GPUs) ^9^ ^88^. This constitutes a 2.1× speedup over comparable AR models, verified independently by the LMSYS team ^9^.

The standard (non-CAP) LLaDA2.0-flash achieves 383 TPS, meaning CAP training alone contributes a 40% throughput improvement ^9^. The LMSYS blog reports slightly different absolute figures — approximately 500 TPS for LLaDA2.0-flash-CAP and 1.9× speedup over AR baselines at small batch sizes ^9^— but the directional finding of roughly 2× speedup is consistent across all measurement sources. Discrepancies in absolute TPS figures reflect differences in batch size, hardware tuning, and measurement methodology rather than fundamental disagreement.

Two technical innovations enable these speeds. **Top-k checkpoint merge**, based on the Warmup-Stable-Merge (WSM) scheduler by Tian et al. (2025) ^89^, selects the best $k$ checkpoints based on validation perplexity and averages their parameters. This optimizer-agnostic, offline procedure explicitly ensembles distinct high-performing model states, smoothing the parameter landscape and yielding more robust generalization than exponential moving average (EMA) alone ^5^. **CAP (Confidence-Aware Parallel) training** adds an auxiliary confidence loss to standard SFT, selectively minimizing entropy on correctly predicted tokens. By sharpening the model's predictive distribution, CAP enables threshold-based parallel decoding to accept more tokens per forward pass: standard LLaDA2.0-flash at 383 TPS jumps to 535 TPS with CAP applied ^5^ ^90^.

The tradeoff is modest: the LLaDA2.0-mini-CAP scores 70.90 on BFCL v3 versus 74.11 for the non-CAP preview ^4^, suggesting that confidence sharpening introduces some rigidity in predictions that slightly reduces performance on certain reasoning tasks. For deployment scenarios where throughput is the primary constraint, this exchange is favorable.

### 4.3 LLaDA2.1: Token Editing and EBPO Reinforcement Learning

Released in February 2026 (arXiv:2602.08676), LLaDA2.1 introduces three major innovations that advance diffusion LLMs from "viable" to "practical": Token-to-Token (T2T) editing for in-generation self-correction, dual-mode configurable decoding (Speed Mode versus Quality Mode), and EBPO — the first large-scale reinforcement learning framework specifically designed for diffusion models ^91^ ^20^. Rather than scaling parameters further, LLaDA2.1 prioritizes "decoding versatility over mere parameter scaling or benchmark peaking" ^91^, keeping the same 16B and 100B model sizes while dramatically expanding what the decoding process can achieve.

#### 4.3.1 Dual-Mode Generation: M2T Drafting and T2T Editing

LLaDA2.1 operates through two complementary mechanisms. **Mask-to-Token (M2T)** is the standard diffusion operation: at each denoising step, the model predicts tokens for currently masked positions, filling in the sequence progressively. This is the "drafting" capability ^18^. **Token-to-Token (T2T)** is the novel editing operation: after each M2T step, the model re-examines all already-revealed (non-mask, non-prompt) positions and overwrites tokens where an alternative candidate exceeds a confidence threshold ^18^ ^92^.

The T2T mechanism is formalized through dual probability thresholds at each timestep $t$: an **unmasking set** $\Gamma_t$ containing positions where the current token is `[MASK]` and the predicted probability exceeds $\tau_{\text{mask}}$, and an **editing set** $\Delta_t$ containing positions where the current token differs from the top candidate and the candidate's probability exceeds $\tau_{\text{edit}}$ ^17^. The state evolution simultaneously applies both operations:

$$x_{t-1}^i = v_t^i \quad \text{if } i \in \Gamma_t \cup \Delta_t; \qquad x_{t-1}^i = x_t^i \quad \text{otherwise}$$

This means a single forward pass can both unmask new positions and edit already-visible tokens ^17^. The inner loop iterates until all masks are filled and no further T2T edits are triggered, at which point generation advances to the next block ^92^.

Training aligns both capabilities through a unified mixture of M2T and T2T objectives: a drafting stream teaches the model to predict correct tokens at masked positions, while an editing stream teaches recovery from random noise perturbations ^93^. Multi-Turn Forward (MTF) data augmentation further exposes the model to diverse iterative editing scenarios during training, simulating the multi-round refinement that occurs at inference ^18^.

Three structural failure modes of T2T have been identified by follow-up research: correction inertia (when the posterior is multimodal, no single alternative crosses the confidence threshold), premature replacement (swapping a correct token for an incorrect one under incomplete context), and positional lock-in (T2T can replace visible tokens but cannot reopen positions for longer-span corrections) ^94^ ^95^. The Token-to-Mask (T2M) follow-up proposes resetting suspicious tokens to `[MASK]` rather than overwriting them, improving accuracy by +13.33 points on AIME 2025 and +8.56 on CMATH ^95^. These findings suggest that the T2T mechanism, while powerful, is an evolving design space rather than a finalized solution.

#### 4.3.2 Speed Mode vs. Quality Mode: Configurable Decoding

LLaDA2.1 introduces two operational modes governed by the dual thresholds $(\tau_{\text{M2T}}, \tau_{\text{T2T}})$, allowing users to configure the speed-quality tradeoff at inference time ^96^:

**Speed Mode (S Mode)** employs a low mask threshold ($\tau_{\text{M2T}} \approx 0.5$) to aggressively draft by filling many positions per step, combined with a moderate editing threshold to restrict edits to high-confidence swaps. Example configuration: `threshold = 0.5`, `editing_threshold = 0.0` ^97^. This yields a TPF (tokens per forward pass) of 5.93 for the Flash model — nearly double the 3.08 TPF of LLaDA2.0 ^98^.

**Quality Mode (Q Mode)** raises both thresholds so only high-confidence actions are taken: `threshold = 0.7`, `editing_threshold = 0.5` ^97^. TPF drops to 3.64, but benchmark scores surpass those of LLaDA2.0 on both mini and flash variants, demonstrating that T2T editing improves not just speed but also quality through self-correction ^17^.

| Configuration | LLaDA2.1 S Mode | LLaDA2.1 Q Mode | LLaDA2.0 (baseline) |
|:---|:---:|:---:|:---:|
| Avg Score (Flash 100B) | 72.34 | **73.54** | 72.43 |
| TPF (Flash) | **5.93** | 3.64 | 3.08 |
| HumanEval+ Score (Flash) | **89.63** | **89.63** | 87.80 |
| HumanEval+ TPS (Flash, quantized) | **892** | — | ~535 |
| HumanEval+ TPS (Mini, quantized) | — | — | 1,587 ^17^|
| $\tau_{\text{M2T}}$ | 0.5 | 0.7 | N/A |
| $\tau_{\text{T2T}}$ | 0.0 | 0.5 | N/A |
| `max_post_steps` | N/A | $\geq$ 5 (rec. 16) | N/A |

**Table 4.3: LLaDA2.1 Speed Mode vs. Quality Mode Comparison.** S Mode approximately doubles TPF relative to LLaDA2.0 while causing only a ~0.1–0.2 absolute average score drop compared to Q Mode ^96^. Q Mode surpasses LLaDA2.0's scores despite identical model size and minimal training data changes, proving that the editing mechanism itself confers quality advantages. Domain-specific speed variation is notable: highest throughput occurs in code generation (structured output tolerates aggressive drafting), while lowest throughput occurs in instruction following (open-ended generation requires conservative thresholds) ^17^.

#### 4.3.3 Multi-Block Editing: Cross-Block Revision

Multi-Block Editing (MBE) extends T2T's local correction capability across block boundaries. Without MBE, decoding and editing are confined within a single block — tokens are generated under threshold constraints and local edits revise intermediate outputs before the block is finalized ^17^. MBE relaxes this constraint: after generating a new block, the model can revisit earlier blocks and apply edits based on the additional context provided by newly decoded content ^99^.

The performance impact is substantial. With MBE enabled, LLaDA2.1-flash's average score improves from 70.69 to 72.67; the mini variant improves from 57.63 to 58.24 ^17^. Specific benchmarks show dramatic gains: AIME 2025 Flash improves from 63.33 to 70.0 with MBE, and LiveCodeBench Flash improves from 44.05 to 46.48 ^17^. These gains are "particularly evident on reasoning and coding tasks" where global consistency across long outputs matters most, with only a "modest reduction in throughput" as the cost ^99^.

#### 4.3.4 EBPO: Reinforcement Learning for Diffusion Models

EBPO (ELBO-based Block-level Policy Optimization) is the first large-scale RL framework tailored specifically for diffusion LLMs ^91^. It addresses a fundamental challenge: standard policy gradient methods require sequence-level log-likelihoods, which are intractable for diffusion models due to their non-autoregressive, parallel decoding nature ^20^.

EBPO's solution uses the Evidence Lower Bound (ELBO) as a tractable proxy for the true likelihood, estimating gradients through block-level conditional probabilities computed in parallel via vectorized likelihood estimation ^18^. The objective maximizes a clipped surrogate function weighted by a probability ratio $\rho$ (analogous to PPO-style clipping), ensuring stable policy updates ^18^. Block-conditional log probabilities are aggregated across discretized timesteps and blocks, enabling efficient computation within a single forward pass ^18^. The RL training extends the AReaL framework with specialized likelihood estimation and advantage estimation protocols that explicitly support both T2T and M2T modes ^17^.

The significance of EBPO extends beyond LLaDA2.1 itself. As noted by independent analysts, "the team applied reinforcement learning to a diffusion model with hundreds of billions of parameters for the first time" ^20^. This opens the door for RLHF-style alignment of diffusion models at a scale previously thought intractable. The use of ELBO as a substitute for log-likelihood in preference optimization is conceptually related to the DPO adaptations explored in earlier diffusion RL work such as VRPO (from LLaDA 1.5), but EBPO's block-conditional formulation and vectorized estimation make it the first approach to operate practically at 100B scale. However, the paper acknowledges that "the RL stage and T2T editing mechanism currently operate separately. Future work aims to merge them, using RL to directly optimize self-correction behavior" ^99^— a merger that could yield transformative self-improving diffusion models capable of learning to edit their own outputs through reward signals rather than fixed confidence thresholds.

#### 4.3.5 Inference Infrastructure: Alpha-MoE, FP8 Quantization, and Custom SGLang

LLaDA2.1's inference infrastructure represents a substantial upgrade over LLaDA2.0. Three components are critical:

**Alpha-MoE megakernel**, adapted from Aleph-Alpha, fuses two FusedMoE computations into a single kernel call, reducing kernel launch overhead and improving memory locality ^17^. This is particularly impactful for MoE architectures where expert routing introduces substantial kernel dispatch costs.

**Per-block FP8 quantization** reduces memory bandwidth requirements and increases compute throughput. On the HumanEval+ benchmark, quantization achieves 891.74 TPS for LLaDA2.1-flash (versus 746.66 TPS unquantized) and 1,586.93 TPS for LLaDA2.1-mini (versus 1,496.67 TPS unquantized) ^17^. The score impact is minimal — only -0.61 points on HumanEval+ for the mini variant ^96^. The per-block (rather than per-tensor) granularity of FP8 scaling is well-suited to MoE architectures, as different experts may have different dynamic ranges ^80^.

**Block-wise causal masked attention** enables the KV cache for the entire long context to be computed in a single forward pass. Within a block, attention is fully bidirectional (all positions attend to each other); across blocks, attention is strictly causal (block $j$ attends only to blocks $0$ through $j$) ^17^. The attention mask at the block level is $M_{\text{atm}} = \text{tril}(\mathbf{1}_{N_k \times N_k})$, expanded to token-level resolution ^92^. This structure, combined with customized SGLang providing Radix caching and batching support for block diffusion, makes LLaDA2.1 the most production-optimized open-source diffusion LLM available ^17^ ^8^.

![Figure 4.2: Inference Speed Evolution from LLaDA2.0 to LLaDA2.1](/mnt/agents/output/fig4_2_llada_speed_evolution.png)

The cumulative effect of these infrastructure innovations is striking: LLaDA2.1-flash in S Mode achieves 892 TPS, approximately 3.5× faster than comparable AR models under the same conditions (Qwen3-30B at 240 TPS, Ling-flash-2.0 at 257 TPS) ^98^. The mini variant with FP8 quantization reaches 1,587 TPS ^17^— speeds that place diffusion LLMs firmly in the realm of interactive, real-time applications.

A recognized limitation is **stuttering artifacts** — n-gram repetitions where phrases loop on themselves, a direct consequence of independent parallel sampling in diffusion models when the masking threshold is set too aggressively ^99^ ^97^. These artifacts primarily occur in open-ended generation scenarios with S Mode; structured domains (code, math) are less affected because T2T editing is particularly effective at catching repetitive patterns ^17^. Mitigation strategies include using Q Mode for chat applications, applying T2T editing and MBE for cross-block correction, and setting temperature to 0.0 for reliability ^97^.

### 4.4 CodeFuse: From Research to Developer Tools

#### 4.4.1 NES: Next Edit Suggestion for 20,000+ Developers

While LLaDA demonstrates that diffusion LLMs can achieve competitive benchmark scores, the CodeFuse NES (Next Edit Suggestion) system demonstrates that they can serve real developers at scale. NES is an instruction-free, low-latency code editing framework deployed at Ant Group serving over 20,000 developers through a seamless Tab-key interaction pattern ^28^. Rather than requiring developers to describe desired changes in natural language, NES learns from historical editing trajectories to implicitly capture coding goals and habits, eliminating context-switching between code and prose ^100^.

The system employs a **dual-model architecture** ^101^ ^38^:

- The **NES-Location Model** predicts the next most probable edit location using the developer's historical editing patterns, achieving 75.6% accuracy in placement prediction. It uses binary rewards (+1.0 for exact match, -1.0 otherwise) during RL training ^102^.
- The **NES-Edit Model** generates precise code modifications for the predicted location, delivering a 27.7% Exact Match Rate and 91.36% Edit Similarity. It uses hierarchical rewards (+1.0 for exact match, +0.5 × Edit Similarity for ES > 0.5, -1.0 otherwise) ^102^.

In production, NES achieves effective acceptance rates of 51.55% for location predictions and 43.44% for edit suggestions ^28^— meaning developers accept roughly half of all suggested edit locations and nearly half of the generated edits. Inference latency remains under 250 milliseconds, achieved through Prefix Caching (PC) and Speculative Decoding (SD) optimizations running on Nvidia L20 GPUs ^38^.

The training pipeline is two-stage: supervised fine-tuning on large-scale historical editing datasets establishes foundational capabilities, followed by reinforcement learning with DAPO (Dynamic sAmpling Policy Optimization) to refine both models ^102^. Dataset construction involves converting raw editing trajectories into structured tuples containing pre-edit code state, historical trajectory, and ground-truth edit, with an LLM-based relevance filter classifying edits as "modification" (logically connected to history) or "preservation" (uncorrelated) — the latter becoming negative samples that teach the model when *not* to suggest a change ^102^.

The NES paper was accepted at FSE Companion 2026, and both SFT and DAPO datasets are publicly available on HuggingFace ^101^ ^38^. This real-world deployment — thousands of daily code changes handled through simple Tab key sequences — provides perhaps the strongest evidence that diffusion-aligned code models can deliver genuine productivity improvements in production software engineering workflows.

#### 4.4.2 DAPO: Dynamic Sampling Policy Optimization

DAPO, the RL algorithm powering NES's post-training, was originally developed by Yu et al. (2025) as an open-source large-scale RL system for LLM reasoning enhancement, achieving 50 points on AIME 2024 with Qwen2.5-32B using only 50% of the training steps required by DeepSeek-R1-Zero-Qwen-32B ^9^. DAPO is a variant of Group Relative Policy Optimization (GRPO) that addresses three known failure modes in standard GRPO: entropy collapse, reward noise, and training instability.

Four techniques distinguish DAPO ^9^ ^103^:

1. **Clip-Higher** increases the upper clipping limit ($\epsilon_{\text{high}}$ from 0.2 to 0.28) to promote diversity and avoid entropy collapse, allowing the model to explore high-entropy, low-probability tokens essential for reasoning.
2. **Dynamic Sampling** filters out prompts with accuracy equal to 0 or 1, ensuring each batch contains samples with effective gradients. If initial sampling produces all-correct or all-incorrect outputs, additional samples are drawn until diversity is achieved.
3. **Token-Level Policy Gradient Loss** operates at the token level rather than averaging within each response, weighting longer sequences more heavily — described as "super key for Long-CoT" scenarios ^92^.
4. **Overlong Reward Shaping** uses soft punishment for longer responses with an expected maximum length of 16,384 tokens, reducing reward noise and stabilizing training ^9^.

For NES specifically, DAPO is adapted with hierarchical reward functions tailored to code editing. The DAPO-trained model demonstrates improved similarity scores for modification tasks and "better aligns with the high-frequency practices observed in real-world development" ^38^. The DAPO system is fully open-sourced, including training code and datasets ^29^.

#### 4.4.3 CodeFuse Survey: Mapping the Code LLM Landscape

Ant Group's contributions to code intelligence extend beyond specific models to systematic knowledge synthesis. The CodeFuse survey paper, "Unifying the Perspectives of NLP and Software Engineering: A Survey on Language Models for Code" (November 2023), covers 70+ models, 40+ evaluation tasks, 180+ datasets, and 900+ related works ^104^. Unlike previous surveys, it integrates software engineering (SE) with natural language processing (NLP) perspectives — SE applying language models for development automation, NLP adopting SE tasks for language model evaluation — providing a bidirectional lens on the field. The survey is maintained as a living document on GitHub ^104^.

The CodeFuse ecosystem also includes CodeFuse-13B (an early multilingual code LLM supporting 40+ programming languages, deployed to 5,000+ engineers via IDE plugins) ^105^ ^106^, CodeFuse-MFTCoder (multi-task fine-tuning framework), DevOps-ChatBot (AI assistant for the software development lifecycle), CodeFuse-Query (static code analysis platform processing 10B+ lines daily), and CodeFuse IDE (an AI-integrated development environment based on OpenSumi) ^107^.

### 4.5 Academic Collaborations

#### 4.5.1 Partnership Model and Author Network

The LLaDA2.0 paper lists 31 authors from five institutions, reflecting a deeply collaborative research model ^108^. Ant Group provides the majority of authors (23, including four technical leaders) and all engineering infrastructure. Academic partners contribute specialized expertise: Renmin University of China (Ji-Rong Wen, a renowned information retrieval and NLP researcher, and Chongxuan Li, an expert in diffusion models) ^108^; Zhejiang University (Jiaqi Hu and Junbo Zhao, contributing computer vision and multimodal ML expertise) ^108^; Westlake University (Zhenzhong Lan, a technical leader, and Zhanchao Zhou, contributing NLP and representation learning expertise) ^108^; and HKUST (Xiaocheng Lu, contributing systems and efficient ML knowledge) ^108^.

This five-institution network represents one of the largest collaborative efforts in open-source diffusion LLM research. The academic partners bring theoretical depth in diffusion modeling, information retrieval, and systems optimization, while Ant Group contributes scale engineering, compute infrastructure, and product-market feedback loops from internal deployment. The LLaDA2.1 paper continues this collaboration with Zhejiang University, Westlake University, and Southern University of Science and Technology ^91^.

#### 4.5.2 Chinese Institutional Leadership in Open-Source Diffusion LLMs

A striking pattern emerges from the global diffusion LLM landscape: all major open-source diffusion models originate from Chinese institutions. Ant Group's LLaDA family (8B to 100B), ByteDance's Seed Diffusion and Stable-DiffCoder, Renmin University's GSAI-ML, and Tsinghua's SIA-Lab collectively produce the entire open-source ecosystem ^5^ ^77^. US contributions — Google DeepMind's Gemini Diffusion, Inception Labs' Mercury, and Apple's DiffuCoder — are either closed-source or limited release. This is the inverse of the autoregressive LLM landscape, where US-based companies (OpenAI, Anthropic, Meta, Google) lead open-weight releases.

The implications are significant for the trajectory of diffusion LLM development. Just as DeepSeek's open-weight releases disrupted the AR landscape by providing high-performance alternatives to closed Western models, Ant Group's LLaDA releases — complete with training frameworks, inference engines, and serving integrations — provide a full-stack open-source diffusion alternative. Western organizations relying on closed-source diffusion APIs may face competitive pressure from open Chinese alternatives that offer not only model weights but also the engineering infrastructure required for production deployment. This dynamic also reinforces a structural advantage identified across the diffusion LLM field: the AR-to-diffusion conversion paradigm means organizations with strong pretrained AR models (Ant Group with Ling, ByteDance with Seed-Coder) gain a head start in the diffusion race, since their expensive AR pretraining investments transfer directly. New entrants without strong AR base models face a higher barrier to producing competitive diffusion LLMs, potentially consolidating leadership among the small set of organizations that have already achieved trillion-token AR pretraining at scale.

Ant Group's strategic bet on diffusion models, executed through the InclusionAI initiative and validated by the LLaDA2.0 and LLaDA2.1 results, positions the organization as the single most important contributor to open-source diffusion LLM research globally. The combination of 100B-parameter models, complete toolchains, real-world developer deployments at 20,000+ scale, and open academic collaboration creates an ecosystem that no other organization — commercial or academic — currently matches in breadth or depth.
-e 


## 5. ByteDance Seed: Training Curriculum and Open Source

ByteDance Seed has emerged as the third major institutional force in diffusion-based code generation, following Google DeepMind and Ant Group, with a research program characterized by two distinctive features: a deeply engineered training curriculum designed to overcome the pathologies of masked diffusion, and a commitment to controlled experimental methodology that isolates the effect of the diffusion paradigm itself. Where Ant Group's LLaDA ecosystem prioritizes scale and toolchain completeness, and Google DeepMind's Gemini Diffusion leverages massive proprietary compute, ByteDance Seed's contribution lies in its rigorous training design and its willingness to subject diffusion models to the scientific discipline of identical-architecture, identical-data comparisons against autoregressive (AR) baselines.

The Seed program operates across three model tiers: Seed Diffusion Preview, a preview-level diffusion language model optimized for inference speed; Seed-Coder, an 8-billion-parameter AR model trained on 6 trillion tokens that serves as the foundation for both the Seed Diffusion conversion and the open-source Stable-DiffCoder release; and Stable-DiffCoder itself, the result of applying block diffusion continual pretraining (CPT) to Seed-Coder while holding every other variable constant. This tiered structure -- AR foundation, diffusion conversion, speed-optimized preview -- allows ByteDance to draw causal conclusions about the diffusion contribution that are unavailable from studies that vary architecture, data, and training paradigm simultaneously ^11^.

### 5.1 Seed Diffusion Preview: Four-Pillar Architecture

Released on July 31, 2025, Seed Diffusion Preview is built upon four interconnected technical pillars that together achieve 2,146 tokens per second (tok/s) on H20 GPUs -- a figure reported to be 5.4x faster than comparable autoregressive models of similar scale -- while maintaining competitive code quality across standard benchmarks ^109^ ^110^. The architecture does not treat diffusion as a single monolithic training objective but as a carefully choreographed sequence of design choices, each addressing a specific failure mode observed in prior discrete diffusion language models (DLLMs).

#### 5.1.1 Four-Pillar System

The first pillar, the **Two-Stage Curriculum (TSC)**, addresses the training dynamics of the diffusion forward process. For the first 80% of training steps, Seed Diffusion uses standard mask-based corruption where tokens are progressively replaced with [MASK] according to a noise schedule $\gamma_t$. For the final 20%, an edit-based corruption process is introduced, applying token-level insertions, deletions, and substitutions controlled by Levenshtein distance ^6^. The combined loss function is:

$$\mathcal{L}_{\text{diff}}(\theta) = -\mathbb{E}_{q_{\text{edit}},t} \log p_\theta(x_0|x_t) - \mathbb{E}_{q_{\text{mask}},t}\left[\frac{\gamma'_t}{\gamma_t} \sum \mathbf{1}[x_t[i]=m] \log p_\theta(x_0[i]|x_t[i])\right]$$

This two-stage structure was motivated by the observation that purely mask-based diffusion creates a harmful inductive bias: the model learns that unmasked tokens are always correct, which prevents self-correction during inference ^111^. The edit-based phase forces the model to re-evaluate all positions, including those that appear uncorrupted.

The second pillar, **Constrained-Order Training**, tackles the problem of redundant generation orders. Standard diffusion training exposes the model to all possible generation trajectories, many of which are "redundant, detrimental, or misaligned with the natural structure of language" ^112^. Seed Diffusion generates a large pool of candidate trajectories using the pre-trained model, filters them by maximizing the Evidence Lower Bound (ELBO), and fine-tunes on these high-quality distilled trajectories ^112^ ^39^. The constrained-order loss takes the form $\mathcal{L}_c(\theta) = \mathbb{E}_{\tau \sim U(T), (x_i, x_0) \in \tau} [-\lambda(x_i) \log p_\theta(x_0 | f(x_i))]$, where $\tau$ denotes a trajectory and $f(x_i)$ represents the state transformation function ^112^.

The third pillar, **On-Policy Diffusion Learning**, optimizes for minimal sampling steps at inference. The team established that trajectory length $\|\tau\|$ is proportional to the inverse of average Levenshtein distance between trajectory states: $\|\tau\| \propto \mathbb{E}_{i,j} \mathbf{1}/d_{\text{Lev}}(\tau[i], \tau[j])$ ^6^. Directly minimizing trajectory length caused unstable training, so the surrogate objective optimizes the inverse of average Levenshtein distance, which implicitly prunes low-quality paths. A model-based verifier $V(\cdot)$ ensures that correctness is preserved throughout this aggressive step reduction ^6^ ^39^.

The fourth pillar, **Block-wise Parallel Sampling with KV-Cache**, provides the inference infrastructure that makes the speed claims achievable. The system uses block-wise semi-autoregressive decoding where each block is denoised in parallel while blocks maintain causal order ^39^. KV-cache reuse eliminates redundant computation across blocks. Ablation studies on H20 GPUs found a block size of 32 tokens to be the optimal operating point, with per-block latency of 1.40 milliseconds -- compared to 1.20 ms for 16-token blocks and 1.80 ms for 64-token blocks ^39^. This block-size sensitivity reflects a fundamental trade-off: smaller blocks offer lower per-step latency but require more sequential steps, while larger blocks amortize overhead but increase per-step cost.

#### 5.1.2 Two-Stage Training Curriculum: Breaking the "Unmasked = Correct" Shortcut

A critical design decision in Seed Diffusion is the explicit **rejection of carry-over unmasking** -- the common practice of copying unmasked input tokens directly to the model output ^111^. While this practice improves perplexity metrics during training, it introduces what the Seed team terms "a detrimental inductive bias": the model learns that any token that was not masked must be correct, leading to overconfidence and an inability to revise its own outputs during iterative decoding ^111^. This pathology is particularly damaging for code generation, where models must frequently revise function signatures, update variable names, or refactor logic across multiple locations.

The edit-based augmentation used in the final 20% of training (the "C" in TSC) forces the model to **re-evaluate all tokens, including unmasked ones** ^111^. The forward process samples corrupted sequences using a predefined edit operation set comprising deletions, insertions, and substitutions, with the total edit-operation number $k_t$ approximately controlling Levenshtein distance ^6^. The signal-to-noise ratio scheduler $\alpha_t$ is constrained to $[0, 0.1]$ to maintain density estimation integrity ^113^. This specific design choice was validated by its elimination of "unexpected behavior such as repetitions in the sampling process" ^6^-- a known failure mode in diffusion language models where the model enters loops of repeated tokens.

The 80/20 split between mask-based and edit-based training was not derived from first principles but from empirical exploration. The ratio reflects a practical need to first establish broad density estimation capabilities through masking before introducing the more complex edit operations that require the model to already possess a coherent understanding of the target distribution. This phased approach mirrors curriculum learning traditions in machine learning but applies them to the forward process of a diffusion model for the first time at this scale.

#### 5.1.3 Performance: Speed, Quality, and Training Budget

Seed Diffusion Preview reports 2,146 tok/s on H20 GPUs, which is 5.4x faster than comparable AR models of similar scale ^109^ ^110^. On the CanItEdit benchmark -- 105 hand-crafted problems covering detailed and underspecified instructions -- Seed Diffusion Preview scores 54.3%, placing it ahead of the Qwen2.5-Coder-14B-Instruct baseline at 52.9% despite Seed Diffusion's presumably smaller parameter count ^10^. The model was trained on a budget of 1.3 trillion tokens across 160,000 steps with a batch size of 512 ^114^.

Several important caveats attach to these headline figures. The 2,146 tok/s claim has not been independently verified, as Seed Diffusion Preview is not open-sourced (unlike Stable-DiffCoder), and the specific comparison AR models and exact benchmarking conditions are not fully detailed in announcement materials ^110^. The hardware specification (H20 GPUs) differs from competitors reporting on H100s (Mercury Coder at 1,109 tok/s) or unspecified hardware (Gemini Diffusion at 1,479 tok/s), making direct comparison difficult without normalization for compute capacity ^109^. Additionally, Seed Diffusion Preview reportedly outperforms Mercury Coder and Gemini Diffusion on raw speed while remaining competitive with AR models on HumanEval, MBPP, and other code generation benchmarks ^109^.

### 5.2 Stable-DiffCoder: Controlled Open-Source Release

Where Seed Diffusion Preview operates as a closed preview demonstrating speed optimization, Stable-DiffCoder represents ByteDance Seed's primary scientific contribution to the diffusion-for-code literature: the first large-scale controlled study demonstrating that diffusion training can outperform AR training on code when architecture, data, and training pipeline are held constant ^11^. This experimental design shifts the burden of proof in the diffusion-versus-AR debate -- previously, DLLMs had to demonstrate they could match AR models; Stable-DiffCoder's results suggest that AR models must now justify their paradigm against a demonstrably superior alternative on at least some tasks.

#### 5.2.1 Block Diffusion Continual Pretraining from Seed-Coder

Stable-DiffCoder is built upon Seed-Coder, an 8-billion-parameter AR model trained on 6 trillion tokens ^11^. The conversion from AR to diffusion proceeds through **Block Diffusion Continual Pretraining (CPT)**, a technical innovation designed to address the instability that has historically plagued attempts to transition AR models to diffusion objectives.

The stability challenge is significant. The ByteDance team observed "significant instability in gradient norms during the CPT of DLLMs" ^115^. Ablation studies revealed two critical failure modes: omitting block clipping leads to a high fraction of zero-mask steps, effectively wasting compute on batches with no corruption signal; and skipping the warmup phase results in gradient norm spikes exceeding 10x the baseline magnitude ^114^. These findings explain why prior AR-to-diffusion conversions have often required extensive hyperparameter tuning or have failed entirely at large scale.

Stable-DiffCoder's CPT employs two stabilizing mechanisms. First, a **warmup strategy** gradually increases mask pattern difficulty and removes cross-entropy weighting over "a few thousand steps, sufficient to ramp up maximum corruption smoothly" ^115^ ^114^. Second, a **block-wise clipped noise schedule** sets linear schedule boundaries tailored for block diffusion with a fixed block size of $B=4$ tokens for code ^114^ ^116^. The schedule is clipped per block to ensure at least one mask per block, eliminating wasted compute. A mixing weight $\lambda$ starts at approximately 0.5 and is annealed to zero as the block size increases during training ^114^. The total training budget for the CPT stage is 1.3 trillion tokens (160,000 steps, batch size 512) ^114^.

Seed-Coder's data pipeline itself represents a notable contribution. Departing from "human-centric" approaches that rely on hand-crafted filtering rules, the team used DeepSeek-V2-Chat as an oracle to score 222,066 code files across four dimensions -- readability, modularity, clarity, and reusability -- on a 0--10 scale ^117^. A 1.3-billion-parameter Llama 2 model with a regression head was then fine-tuned for one epoch to serve as an efficient quality scorer at scale, filtering the bottom ~10% of files and yielding approximately 1 trillion unique tokens covering 89 programming languages ^117^. The full pretraining corpus comprises 5 trillion tokens for regular pretraining plus 1 trillion for continued pretraining, totaling 6 trillion tokens ^36^.

#### 5.2.2 The CanItEdit Result: 60.0% versus 50.5%

The headline result from Stable-DiffCoder is a **9.5 percentage point gap** on CanItEdit, with Stable-DiffCoder-8B-Instruct scoring 60.0% against Seed-Coder-8B-Instruct's 50.5% -- an 18.8% relative improvement ^10^ ^11^. This makes Stable-DiffCoder the top performer on CanItEdit across all compared models at the time of reporting, including Qwen2.5-Coder-14B-Instruct (52.9%) and Yi-Coder-9B-Chat (50.5%), both of which use the autoregressive paradigm and comparable or larger parameter counts ^10^.

The authors hypothesize that this gain "benefits from the denoising nature of DLLMs: random masking and reconstruction inherently train the model on edit- and infill-like patterns, enabling it to better exploit editing supervision and extract more editing-related knowledge from the same data" ^10^. This mechanism -- where the diffusion training objective functions as a form of implicit data augmentation for editing tasks -- represents a principled explanation for why diffusion models excel at code editing specifically, even when their performance on standard completion benchmarks (HumanEval, MBPP) shows smaller or negligible differences.

However, the advantage is not uniform. On Aider, a multi-turn editing benchmark requiring long context windows, Stable-DiffCoder scores 54.9%, slightly below Seed-Coder's 57.1% ^10^. The authors attribute this to the 8,192-token training window being insufficient for Aider's multi-turn requirements, which exceed this length ^10^. This creates an important tension in the diffusion-for-code landscape: diffusion's any-order modeling helps single-turn editing but block-size constraints may hurt long-context editing tasks.

#### 5.2.3 Controlled Comparison Methodology

Stable-DiffCoder's core scientific contribution extends beyond the raw performance numbers to the **experimental design** that produced them ^118^ ^115^. The methodology adheres to four strict controls:

1. **Identical architecture**: Seed-Coder's 8-billion-parameter architecture is reused without modification ^11^.
2. **Identical data**: The same 6-trillion-token pretraining corpus from Seed-Coder is used for the diffusion conversion ^11^.
3. **Identical training pipeline**: Preprocessing, supervised fine-tuning (SFT) data, and evaluation protocols remain unchanged ^118^.
4. **Single variable changed**: Only the training objective switches -- from pure AR to AR pretraining -> block diffusion CPT -> SFT ^10^.

The authors explicitly state their motivation: "existing code DLLMs still lag behind strong AR baselines under comparable budgets. We revisit this setting in a controlled study" ^11^. This design allows the conclusion that "diffusion-based training can improve code modeling quality beyond AR training alone, even under tightly controlled data and architecture constraints" ^118^.

At smaller scale (2.5 billion parameters), the team further identified two preconditions for successful DLLM training: **Clean Evidence** -- pre-annealing AR checkpoints must retain clean, malleable knowledge suitable for diffusion conversion; and **Alignment** -- consistency between training and inference processes must be maintained to avoid distribution shift ^115^. These findings provide practical guidance for organizations attempting similar conversions.

#### 5.2.4 Low-Resource Language Benefits and Cross-Model Benchmarks

Stable-DiffCoder demonstrates **particularly large gains in low-resource programming languages** on the MultiPL-E multilingual code generation benchmark ^10^ ^11^. Languages such as C# and PHP show gains exceeding 10 percentage points ^116^. The authors' explanation is that "diffusion-style stochastic sampling can effectively amplify learning signals from low-resource code by exposing the model to multiple corrupted-and-denoised views of the same underlying example, thereby improving generalization in data-scarce languages" ^10^. This mechanism -- where diffusion corruption acts as **principled data augmentation** for scarce samples -- is one of the key theoretical contributions of the work ^11^.

A notable caveat is that these gains partially attenuate after supervised fine-tuning. Because the SFT stage extensively supplements scarce data for languages like C# and PHP, "the advantage in multilingual coding capabilities has been reduced" for the final instruct model relative to the base model ^11^. This suggests that the diffusion augmentation benefit is most pronounced when the model must generalize from limited exposure, and is partially diluted when explicit data supplementation is applied.

The following table consolidates benchmark results across Seed Diffusion Preview, Stable-DiffCoder, their AR baseline Seed-Coder, and selected competitors:

| Model | Paradigm | Params | CanItEdit | HumanEval | Aider | MultiPL-E (C#/PHP) |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| Seed-Coder-8B-Instruct | AR | 8B | 50.5% ^10^| 84.8% | 57.1% ^10^| Baseline |
| Stable-DiffCoder-8B-Instruct | Diffusion (block CPT) | 8B | **60.0%** ^10^| 86.6% ^11^| 54.9% ^10^| +10%+ ^116^|
| Seed-Diffusion-Preview | Diffusion (full) | -- | 54.3% ^10^| Competitive ^110^| -- | -- |
| Qwen2.5-Coder-14B-Instruct | AR | 14B | 52.9% ^10^| 86.2% | -- | -- |
| Yi-Coder-9B-Chat | AR | 9B | 50.5% ^10^| 85.0% | -- | -- |

The CanItEdit column most clearly illustrates the diffusion advantage: Stable-DiffCoder's 60.0% represents a 9.5 percentage point improvement over its AR twin trained on identical data, and simultaneously outperforms a 14-billion-parameter competitor (Qwen2.5-Coder) by 7.1 percentage points. On HumanEval, the advantage narrows to approximately 1.8 percentage points, consistent with the pattern that diffusion models show their largest benefits on editing tasks while achieving near-parity on generation tasks. The Aider regression (54.9% versus 57.1%) serves as a reminder that diffusion's advantages are task-dependent and that long-context multi-turn editing remains a challenge for models trained with limited context windows.

![Figure 5.1](/mnt/agents/output/fig_5_1_bytedance_benchmark_comparison.png)

*Figure 5.1 -- Code generation and editing performance comparison across ByteDance models and competitors. CanItEdit scores show the largest diffusion advantage (18.8% relative improvement for Stable-DiffCoder over Seed-Coder), while HumanEval scores cluster tightly, indicating that the diffusion paradigm's primary differentiator is editing rather than generation capability. Data sources: arXiv papers, ByteDance blog announcements.*

### 5.3 SIA-Lab and Research Collaboration

The research output of ByteDance Seed does not exist in isolation. It is produced through SIA-Lab, a formal joint laboratory between Tsinghua University's Institute for AI Industry Research (AIR) and ByteDance Seed ^6^ ^29^. This institutional arrangement shapes the research agenda, shares talent across organizational boundaries, and connects ByteDance's commercial objectives with academic publication norms.

#### 5.3.1 Joint Lab Structure and Governance

SIA-Lab operates as a formal research entity with dual affiliation for its personnel. Core contributors to the Seed Diffusion paper -- including Yuxuan Song and Zheng Zhang -- hold dual affiliation with ByteDance Seed and Tsinghua AIR/SIA-Lab ^6^. Supervision spans both institutions: Jingjing Liu, Wei-Ying Ma, and Ya-Qin Zhang represent Tsinghua AIR, while Yonghui Wu, Hao Zhou, and Mingxuan Wang represent ByteDance Seed ^6^. The DAPO (Dynamic Adaptive Policy Optimization) paper, which introduced reinforcement learning innovations for large language models, lists the identical shared affiliation: "SIA-Lab of Tsinghua AIR and ByteDance Seed" as the fourth institutional author ^29^.

This represents a **deep institutional partnership** with shared talent and co-authored publications, not a conventional funding relationship or advisory arrangement. The collaboration produces papers across multiple domains: diffusion language models (Seed Diffusion), code generation (Seed-Coder, Stable-DiffCoder), and reinforcement learning (DAPO). The breadth of this output suggests that SIA-Lab functions as a semi-autonomous research unit with its own agenda, staffed by researchers who move across project boundaries within the lab's scope.

#### 5.3.2 Author Overlap with DAPO and Shared RL Methodology DNA

The overlap between the diffusion model research team and the DAPO team is substantial and systematic. The following table documents the shared personnel across papers, revealing a research community whose members contribute to both diffusion and reinforcement learning projects:

| Person | Seed Diffusion | Stable-DiffCoder | DAPO | Affiliation |
|:---|:---:|:---:|:---:|:---|
| Yuxuan Song | Project Lead | Contributor | Dataset | ByteDance Seed + Tsinghua AIR |
| Zheng Zhang | Project Lead | -- | Algorithm | ByteDance Seed + Tsinghua AIR |
| Jing Su | Contributor | Contributor | -- | ByteDance Seed |
| Hongli Yu | -- | Contributor | Infra/Dataset | ByteDance Seed |
| Hao Zhou | Supervision | Supervision | Supervision | ByteDance Seed |
| Jingjing Liu | Supervision | -- | Supervision | Tsinghua AIR |
| Wei-Ying Ma | Supervision | -- | Supervision | Tsinghua AIR |
| Ya-Qin Zhang | Supervision | -- | Supervision | Tsinghua AIR |
| Lin Yan | -- | -- | Supervision | ByteDance Seed + Tsinghua AIR |
| Mingxuan Wang | Supervision | -- | Supervision | ByteDance Seed |

The concentration of shared authors at the supervision level is particularly significant. Hao Zhou, Jingjing Liu, Wei-Ying Ma, and Ya-Qin Zhang all serve as supervisors across both the diffusion and DAPO papers, suggesting that strategic research direction is coordinated at this level rather than emerging independently from individual research groups ^6^ ^29^ ^11^. The project-lead level shows more specialization -- Yuxuan Song leads Seed Diffusion while Qiying Yu leads DAPO -- but cross-contribution remains common.

The technical connection between these research streams is methodological rather than direct algorithmic. The on-policy diffusion learning component of Seed Diffusion uses reinforcement-learning-style optimization (minimizing expected sampling steps with verifier guidance), which conceptually connects to DAPO's policy optimization innovations ^6^. However, DAPO itself applies to autoregressive models (specifically Qwen2.5-32B), not diffusion models ^29^. The shared infrastructure includes the veRL/verl training frameworks and a common methodological vocabulary around policy optimization, reward shaping, and verifier-guided training. This positions SIA-Lab as one of the few research groups worldwide with deep expertise in both diffusion language models and reinforcement learning for language models -- a combination that could prove decisive as the field moves toward RL-enhanced diffusion training, a direction no published paper has yet fully explored.

The institutional model represented by SIA-Lab -- a formal joint lab between a top-tier Chinese university and a major technology company, with dual-affiliation researchers, shared supervision, and cross-domain publications -- may serve as a template for industry-academia partnerships in the diffusion model era. The arrangement gives ByteDance access to academic credibility and fundamental research talent, while Tsinghua gains commercial relevance and computational resources for its researchers. Whether this model can scale beyond the specific individuals involved, or whether it depends on the relationships of a small group of senior researchers, will become clearer as SIA-Lab's publications continue across subsequent project cycles.
-e 


## 6. Open-Source Ecosystem and Community Models

The open-source diffusion language model (dLLM) ecosystem has matured rapidly since LLaDA's initial release in early 2025, with five distinct model families now defining the landscape: LLaDA from Ant Group and Renmin University, Dream from HKU and Huawei, DiffuCoder from Apple, SEDD from Stanford, and the commercial Mercury system from Inception Labs. Each represents a fundamentally different approach to training, licensing, and community engagement. The cumulative result is a diverse ecosystem that, while still smaller than its autoregressive counterpart, offers researchers and practitioners a range of architectural choices with permissive licensing and transparent training methodologies.

The trajectory of open-source dLLMs reveals a field transitioning from proof-of-concept to production-ready tooling. LLaDA-8B established that diffusion models could match autoregressive baselines on general language tasks; Dream demonstrated that autoregressive-to-diffusion conversion was a viable and potentially superior training paradigm; DiffuCoder introduced reinforcement learning (RL) techniques specifically designed for diffusion's non-causal structure; SEDD provided the mathematical foundation upon which much of this work rests; and Mercury proved that the approach could attract substantial commercial investment. Together, these models form a coherent innovation pipeline from theoretical foundations to deployed products.

**Table 6.1** summarizes the core characteristics of the major open-source diffusion LLM families, encompassing their architectural origins, training scale, licensing terms, and community adoption metrics as of mid-2025.

| Model Family | Organization | Parameters | Training Data | License | GitHub Stars | Key Innovation |
|:---|:---|:---|:---|:---|:---|:---|
| LLaDA-8B | Renmin U. / Ant Group | 8B | 2.3T tokens, 130K H800 hrs ^119^| MIT | 3,800 ^120^| First open-source dLLM, from-scratch training |
| LLaDA 1.5 | Renmin U. / Ant Group | 8B | 350K preference pairs ^121^| MIT | — | VRPO variance-reduced preference optimization ^122^|
| Dream 7B | HKU / Huawei | 7B | AR-initialized (Qwen2.5) ^123^| Apache 2.0 | ~1,231 ^124^| Context-adaptive noise rescheduling, Shift Operation ^123^|
| Dream-Coder 7B | HKU / Huawei | 7B | OpenCoder, Stack-Edu, Dolmino ^125^| Apache 2.0 | ~95 ^126^| Full transparency release (recipes + checkpoints) ^127^|
| DiffuCoder 7B | Apple | 7B | 130B code tokens ^16^| Apple OSS | 821 ^128^| Coupled-GRPO RL for diffusion ^16^|
| SEDD | Stanford | GPT-2 scale | OpenWebText ^129^| Open (GitHub) | — | Score entropy discrete diffusion ^129^|

The table reveals a clear geographic and institutional split in the open-source ecosystem. Chinese institutions dominate the permissive open-source releases: Ant Group and Renmin University drive the LLaDA family under MIT licensing, while HKU and Huawei's Noah's Ark Lab release Dream under Apache 2.0. US contributions come primarily from corporate research laboratories (Apple's DiffuCoder) and academic groups (Stanford's SEDD), with Inception Labs choosing a closed-source commercial path for Mercury. This concentration of open-source leadership among Chinese institutions represents a notable inversion of the autoregressive LLM landscape, where US-based organizations have historically led open-weight releases.

### 6.1 LLaDA: The Pioneer

#### 6.1.1 LLaDA-8B Architecture and Training

LLaDA-8B, presented as an oral paper at NeurIPS 2025, was the first large-scale open-source diffusion language model, establishing the foundational training pipeline that subsequent dLLMs would follow, adapt, or reject ^73^. Pre-trained from scratch on 2.3 trillion tokens using approximately 130,000 H800 GPU hours, LLaDA-8B closely follows the LLaMA3-8B architecture but with three critical modifications: vanilla multi-head attention replaces Grouped Query Attention (GQA), bidirectional attention removes the causal mask that constrains autoregressive models, and the feed-forward network dimension is adjusted to 12,288 to maintain comparable parameter count at approximately 8.02 billion total parameters ^119^.

The training pipeline proceeds through standard stages: data preparation, pre-training with uniform masking ratios sampled from [0, 1], supervised fine-tuning (SFT) on 4.5 million instruction pairs, and evaluation ^119^. During SFT, only response tokens are masked while prompt tokens remain fully visible — the diffusion equivalent of autoregressive models computing loss exclusively on generated tokens during instruction tuning ^73^. The optimizer configuration uses AdamW with a Warmup-Stable-Decay (WSD) learning rate schedule: 2,000-iteration warmup to 4e-4, a stable phase at 4e-4, a mid-training drop to 1e-4 after processing 1.2 trillion tokens, and final decay to 1e-5 over the last 0.3 trillion tokens ^130^.

The evaluation results established a critical baseline for the field. On MMLU, LLaDA-8B scores 65.9 versus LLaMA3-8B's 65.4; on HumanEval, LLaDA reaches 33.5% versus LLaMA3's 34.2% ^73^. Perhaps more significantly, LLaDA surpasses GPT-4o on reversal poem completion — a task that directly tests bidirectional reasoning capability — providing concrete evidence that diffusion's non-sequential generation confers genuine advantages on certain task types ^73^. The model's release under the MIT license — one of the most permissive open-source licenses available — enabled unrestricted commercial use, modification, and redistribution with minimal attribution requirements, catalyzing rapid community adoption that accumulated 3,800 GitHub stars within months of release ^120^.

#### 6.1.2 LLaDA 1.5 and VRPO

LLaDA 1.5 addressed the central challenge of aligning diffusion LLMs with human preferences: the high variance in Evidence Lower Bound (ELBO)-based likelihood estimates required for preference optimization ^122^. Diffusion models cannot compute exact log-probability for generated sequences and instead rely on ELBO approximations. These estimates are inherently noisy, making preference-based gradient updates — which depend on precise likelihood differentials between preferred and dispreferred outputs — numerically unstable ^131^.

Variance-Reduced Preference Optimization (VRPO) introduces three principled variance-reduction techniques ^122^. First, increased sampling budget for ELBO estimates uses more random draws of diffusion timestep and mask pattern configurations to reduce Monte Carlo noise. Second, optimal allocation recognizes that for a fixed total sample budget, sampling many different diffusion timesteps with only one mask per timestep minimizes estimation variance more effectively than alternative allocation strategies. Third, antithetic sampling shares identical random noise configurations between winning and losing outputs in preference comparisons, so random errors in their respective log-likelihood estimates tend to cancel when computing the preference differential ^122^ ^132^.

The training cost is remarkably efficient: approximately 405 H100 GPU hours for 8 Monte Carlo samples, representing less than 0.5% of the pre-training compute expenditure ^121^. Training on 350,000 preference pairs covering creative writing (35%), knowledge QA (18%), NLP tasks (16%), mathematics (14%), recommendations (7%), code (5%), reasoning (3%), and safety tasks, VRPO delivers substantial improvements over the SFT-only LLaDA Instruct baseline: GSM8K +4.7 points, HumanEval +3.0, MBPP +1.8, IFEval +4.0, and Arena-Hard +4.3 ^122^. The cost-effectiveness of these gains — high-quality alignment for under half a percent of pre-training compute — established a template for subsequent diffusion RL work.

#### 6.1.3 Scaling Behavior and LLaDA 2.0

LLaDA's from-scratch training demonstrated that diffusion LLMs could achieve competitive results when given data and compute budgets comparable to autoregressive counterparts, but the approach proved data-inefficient relative to alternative pathways. The development of LLaDA 2.0, which scales to 100 billion parameters via autoregressive-to-diffusion conversion rather than from-scratch training, represented a pivotal strategic shift ^5^. LLaDA 2.0 employs a novel three-phase Warmup-Stable-Decay (WSD) block-level training paradigm that progressively converts pretrained autoregressive checkpoints into diffusion models, preserving the linguistic knowledge encoded in AR pretraining while gaining diffusion's parallel generation capability ^5^. This conversion approach proved substantially more efficient than training from scratch, and the resulting 100-billion-parameter model — released as LLaDA 2.0-flash with 6.1 billion active parameters via Mixture-of-Experts (MoE) architecture — became the largest open-source diffusion LLM available ^133^.

### 6.2 Dream: AR-Initialized Adaptive Decoding

#### 6.2.1 Dream-7B Architecture

Dream-7B represents a fundamentally different philosophy from LLaDA's from-scratch approach. Rather than training a diffusion model de novo, Dream initializes from pretrained autoregressive weights — specifically Qwen2.5-Coder — and adapts them for diffusion-based generation through a "Shift Operation" that preserves the positional relationships learned during AR pretraining ^123^. Under this strategy, the model continues to use hidden state $h_i$ to generate predictions for position $i+1$, contrasting with conventional diffusion that predicts masked tokens at their original positions. This preserves the sequential reasoning patterns that AR models learn while enabling non-sequential generation during inference ^123^.

The core technical innovation is context-adaptive token-level noise rescheduling. Dream re-determines the noise level for each masked token by measuring its contextual "informationness" using a mixture of geometric distributions. The weighting function quantifies each clean token's contribution to predicting masked tokens:

$$w(t, x_t, n) = \frac{1}{2} \sum_i \left[ \mathbf{1}[x_t^i \neq \text{MASK}] \cdot \text{Geo}(p, |n-i|-1) \right]$$

where parameter $p$ controls sharpness: smaller $p$ yields uniform contribution from all clean tokens, while larger $p$ emphasizes nearby clean tokens ^123^. This adaptive rescheduling addresses a key limitation of uniform masking — not all tokens are equally informative, and treating them as such wastes model capacity on low-information positions.

Dream supports multiple remasking strategies including random, `maskgit_plus` (top-1 confidence selection), `topk_margin` (top1-top2 margin heuristic), and entropy-based remasking (using token distribution entropy as a confidence proxy). The entropy strategy with `alg_temp=0` serves as the default and corresponds to low-confidence remasking, which empirical evaluation has shown to be critical for diffusion LLM performance ^124^.

#### 6.2.2 Dream-Coder: Full Transparency Release

Dream-Coder 7B, released on July 15, 2025 by HKU and Huawei Noah's Ark Lab, represents the most transparent open-source release in the diffusion LLM ecosystem ^127^. Positioned as a "fully open" 7-billion-parameter dLLM for code, it was trained exclusively on open-source and publicly available data across all stages: adaptation, SFT, and RL ^134^. The release includes not only model checkpoints (Dream-Coder-7B and Dream-Coder-7B-Instruct) but complete training recipes, preprocessing pipelines, and inference code — a level of transparency that enables full reproducibility and community extension ^134^.

Training data sources include OpenCoder, Stack-Edu, Dolmino, and DCLM-Baseline ^125^. The post-training recipe consists of two stages: SFT with random truncation and padding penalty to mitigate padding pathologies inherent in fixed-length diffusion training, followed by RL with verifiable rewards over curated high-quality prompts using a tailored RL recipe for diffusion language models ^134^. Benchmark results are competitive with commercial diffusion models: 21.4% pass@1 on LiveCodeBench (2410-2505) — on par with Mercury Coder Small's 22.9% — alongside 82.9% on HumanEval, 79.6% on MBPP, and 73.1% on EvalPlus ^135^ ^136^.

A notable emergent property of Dream-Coder is its any-order generation capability, which manifests in three distinct patterns depending on task complexity ^134^. For complex algorithms, the model exhibits sketch-first behavior: generating structural elements (function signatures, control flow) before filling implementation details. For straightforward completions, it defaults to left-to-right generation. For logic-intensive tasks, it produces interleaved reasoning — generating key conditional checks first, then supporting code. This adaptivity suggests that diffusion models can dynamically select generation strategies matched to task structure, a capability that rigidly sequential AR models cannot replicate.

#### 6.2.3 DreamOn: Variable-Length Generation

DreamOn, released in February 2026, addresses a fundamental limitation of masked diffusion models: fixed-length generation. Standard dLLMs require pre-specifying output length, which is either wasteful (padding short responses to maximum length) or constraining (truncating long responses) ^134^. DreamOn introduces `[expand]` and `[delete]` special tokens that enable variable-length generation by allowing the model to dynamically resize the output sequence during the denoising process. The reported 26.4% improvement on key benchmarks demonstrates that the fixed-length constraint was a genuine performance bottleneck rather than merely an inconvenience ^134^. This innovation is particularly relevant for code generation, where output length varies dramatically — from one-line completions to multi-file implementations.

### 6.3 DiffuCoder: Apple's RL-Enhanced Approach

#### 6.3.1 Coupled-GRPO

DiffuCoder 7B, Apple's open-source diffusion model for code, introduces coupled-GRPO (Group Relative Policy Optimization), a reinforcement learning algorithm specifically designed for the non-causal structure of diffusion models ^16^. The model is trained on 130 billion tokens of code data across four stages: adaptation pre-training on a 400-billion-token code corpus from RefineCode and Stackv2 with early stopping at 65 billion tokens; mid-training on 16 billion tokens of annealed code data; instruction tuning on 436,000 SFT samples using classifier-free guidance; and RL post-training via coupled-GRPO on 21,000 hard samples from Acecoder-87K ^16^ ^137^.

The coupled-GRPO algorithm's central innovation is complementary mask sampling. For each training completion, two masks are constructed such that every token position is masked in exactly one of the two masks — together they cover all completion tokens exhaustively ^16^. This construction guarantees three properties: each token's log-probability is computed at least once (ensuring non-zero learning signal everywhere), log-probability estimations are more accurate because they are evaluated under realistic partial-masking contexts rather than full masking, and the approach generates $2\lambda$ additional samples compared to baseline GRPO without increasing computational cost ^16^.

The empirical results validate the approach. Coupled-GRPO achieves a +4.4% improvement on EvalPlus over the SFT-only baseline, with final scores of 72.0% on HumanEval, 65.2% on MBPP, and 75.1% on EvalPlus ^16^. An important secondary finding is that RL training shifts the optimal sampling temperature from 0.2 toward higher values, reducing the model's reliance on strict autoregressive causal decoding patterns. This suggests that RL enables diffusion models to discover and exploit their non-sequential generation capability more effectively than SFT alone.

#### 6.3.2 RL Versus SFT for Diffusion Models

DiffuCoder's training experiments yielded a striking finding that has implications for the entire dLLM field: reinforcement learning consistently outperforms supervised fine-tuning for diffusion model post-training, while standard SFT provides only marginal gains ^16^. The underlying cause is train-test mismatch — SFT trains the model on fully masked outputs but inference requires generating under partially masked conditions, and this distributional shift undermines SFT effectiveness. Diffusion models are inherently trained to denoise from any masking configuration, but SFT optimizes only the fully masked endpoint, wasting the model's intermediate-denoising capability. RL, by contrast, evaluates the model under realistic inference conditions (variable masking) and rewards successful completion, directly optimizing for the metric of interest without distributional assumptions ^16^.

#### 6.3.3 Open-Source Release

DiffuCoder was released under Apple's standard open-source license (similar in permissiveness to Apache 2.0), with full code, training recipes, and model weights published on HuggingFace and GitHub ^128^. The repository has accumulated 821 GitHub stars, reflecting solid community interest, though substantially below LLaDA's 3,800 ^128^. Apple also actively supports community porting efforts, including an MLX implementation for Apple Silicon inference. However, DiffuCoder's strong code performance comes with limited general capability — the model is specialized for code generation and does not compete with general-purpose dLLMs like LLaDA or Dream on non-code benchmarks ^16^.

**Table 6.2** presents a detailed comparison of post-training RL techniques across the major open-source dLLM families, illustrating the rapid maturation of diffusion-specific alignment methods.

| Technique | Model | RL Algorithm | Training Data | Compute Cost | Key Improvement | Mechanism |
|:---|:---|:---|:---|:---|:---|:---|
| VRPO | LLaDA 1.5 | Variance-reduced DPO | 350K preference pairs ^121^| 405 H100 hrs ^121^| GSM8K +4.7, HumanEval +3.0 ^122^| Optimal allocation + antithetic sampling |
| Coupled-GRPO | DiffuCoder 7B | Complementary mask GRPO | 21K hard code samples ^16^| — | EvalPlus +4.4% ^16^| Full token coverage via paired masks |
| Dream-Coder RL | Dream-Coder 7B | Verifiable reward RL | Curated high-quality prompts ^134^| — | LiveCodeBench 18.3% ^135^| Tailored RL recipe for diffusion LMs |
| EBPO | LLaDA 2.1 | Entropy-based preference opt. | Large-scale preference data | — | LiveCodeBench 42.29 ^133^| Leverages inpainting for exploration |
| d1-LLaDA | LLaDA family | GRPO for MDMs | Task-specific | — | Task-variable ^138^| One-step log-prob with random masking |

The diversity of RL approaches reflects the field's rapid innovation rate. VRPO addresses variance reduction for general alignment; coupled-GRPO solves the token-coverage problem specific to code generation; Dream-Coder's verifiable reward RL leverages the deterministic nature of code correctness; and EBPO exploits diffusion's inpainting capability to guide exploration. This specialization suggests that, unlike autoregressive models where a single RL algorithm (PPO, then DPO) achieved broad dominance, diffusion models may require task-specific RL formulations to realize their full potential.

![Code Generation Benchmarks: Open-Source Diffusion LLMs](/mnt/agents/output/fig_diffusion_benchmark_comparison.png)

**Figure 6.1** compares code generation performance across open-source diffusion LLMs and closed-source baselines on three standard benchmarks. Dream-Coder achieves the highest scores among fully open-source models, approaching Mercury Coder Small (closed API) on HumanEval while trailing on LiveCodeBench, which tests more complex competitive-programming-style problems. The substantial gap between open-source diffusion models and the Qwen3-8B autoregressive baseline on LiveCodeBench (26.0% versus a range of 8.4%–18.3% for open-source dLLMs) underscores that competitive programming remains a challenge for the diffusion paradigm, though Dream-Coder's 21.4% on the v4 benchmark ^134^shows the gap is narrowing.

### 6.4 SEDD: ICML 2024 Best Paper

#### 6.4.1 Score Entropy for Discrete Diffusion

SEDD (Score Entropy Discrete Diffusion), authored by Aaron Lou, Chenlin Meng, and Stefano Ermon from Stanford University, received the ICML 2024 Best Paper award for establishing the mathematical foundation that enables diffusion models to operate effectively in discrete spaces such as text token vocabularies ^139^ ^140^. The central technical challenge SEDD addresses is that standard diffusion models rely on score matching — estimating $\nabla_x \log p(x)$, the gradient of the data log-density — which generalizes naturally to continuous spaces (where gradients exist) but fails for discrete data like text token indices (where gradients are undefined) ^139^.

SEDD's solution replaces gradient-based score matching with probability ratio estimation. Rather than modeling how the log-density changes with infinitesimal input perturbations, SEDD parameterizes the reverse discrete diffusion process using ratios of the data distribution $p_t(y) / p_t(x)$. These probability ratios are learned through a novel score entropy loss that naturally extends score matching to discrete spaces ^129^ ^141^:

$$D_{SE}(p_\theta(\cdot|x_t) \,||\, p_t(\cdot|x_0)) = \sum_y \frac{p_t(y|x_0)}{p_t(x)} \left[ \log\frac{p_t(y|x_0)}{p_t(x)} - \log s_\theta(x_t)_y \right]$$

This formulation has the elegant property that it reduces to standard score matching in the continuous limit while remaining well-defined for discrete vocabularies of arbitrary size.

Empirically, SEDD reduces perplexity by 25–75% compared to existing language diffusion paradigms and achieves 6–8× better generative perplexity than un-annealed GPT-2 at the same model scale ^129^. Critically, SEDD can match GPT-2 generation quality with 32× fewer network evaluations, and it enables controllable infilling — matching nucleus sampling quality while supporting non-left-to-right prompting patterns ^142^. The infilling capability is particularly significant because it demonstrates that diffusion models can naturally support the bidirectional context access that autoregressive models implement only through cumbersome workarounds.

#### 6.4.2 Theoretical Significance and Influence

SEDD's theoretical contribution extends beyond its immediate empirical results. By proving that probability ratios (not gradients) are the correct generalization of score matching to discrete spaces, SEDD provided the mathematical legitimacy that the entire discrete diffusion field required ^139^. The paper directly influenced the development of LLaDA (which uses a simplified variant of SEDD's probability-ratio formulation), MDLM (Masked Diffusion Language Model, which extends the framework to larger scales), and subsequent theoretical work on discrete diffusion convergence properties. The intellectual lineage from SEDD to commercial systems is explicit: SEDD co-author Stefano Ermon co-founded Inception Labs, which commercialized the diffusion-language approach as the Mercury model family ^139^ ^143^. This represents a rare instance of academic theoretical research translating directly into a commercial product within a single generation of technology development.

### 6.5 Mercury: First Commercial Diffusion LLM for Code

#### 6.5.1 Inception Labs and Funding

Inception Labs, founded in 2024 in Palo Alto by a trio of Stanford, UCLA, and Cornell professors — Stefano Ermon (Stanford, co-inventor of diffusion methods underlying Midjourney and Sora), Aditya Grover (UCLA), and Volodymyr Kuleshov (Cornell) — represents the first venture-backed company dedicated exclusively to commercializing diffusion language models ^36^ ^144^. The founding team has collaborated for over a decade on generative modeling research, and their combined expertise spans the theoretical (Ermon's work on score-based models), algorithmic (Grover's work on structured prediction), and systems (Kuleshov's work on efficient inference) dimensions of the technology stack.

In November 2025, Inception Labs announced a $50 million seed round led by Menlo Ventures, with participation from Mayfield, Innovation Endeavors, NVentures (NVIDIA), M12 (Microsoft), Snowflake Ventures, Databricks Investment, and angel investors including Andrew Ng and Andrej Karpathy ^145^ ^146^. The funding round, unusually large for a seed-stage company, reflects investor conviction that diffusion LLMs represent a genuine architectural alternative to autoregressive models rather than merely an academic curiosity. The reported post-money valuation of approximately $500 million signals strong market expectations for the technology's commercial viability ^36^.

#### 6.5.2 Mercury Coder Performance

Mercury Coder, launched in February 2025, is the world's first commercially available diffusion LLM, delivered through a managed API rather than open weights ^7^ ^145^. Mercury Coder Mini achieves 1,109 tokens per second on NVIDIA H100 GPUs, while Mercury Coder Small reaches 737 tokens per second — approximately 5–10× faster than speed-optimized frontier autoregressive models while maintaining comparable output quality ^147^ ^7^. On quality benchmarks, Mercury Coder Small achieves 90.0% on HumanEval, 76.6% on MBPP, 80.4% on EvalPlus, 25.0% on LiveCodeBench, and 45.5% on BigCodeBench ^7^.

The API pricing model — $0.25 per million input tokens and $1.00 per million output tokens — positions Mercury substantially below frontier autoregressive model pricing, with the cost advantage derived from diffusion's parallel generation reducing per-token inference compute ^36^ ^148^. Mercury supports a 32,000-token context window and is available via platform.inceptionlabs.ai as well as through third-party integrations including Amazon Bedrock, OpenRouter, and Poe. Developer tool integrations include Continue.dev for VS Code, ProxyAI, Buildglare, and Kilo Code, though native IDE plugin support remains less mature than GitHub Copilot's deep VS Code and JetBrains integration ^36^.

#### 6.5.3 Licensing and Availability Landscape

**Table 6.3** provides a comprehensive comparison of licensing terms, weight availability, and commercial accessibility across all major diffusion LLMs, both open-source and proprietary.

| Model | License | Weights Available | Code Available | Training Data Known | Commercial Use | Access Method |
|:---|:---|:---|:---|:---|:---|:---|
| LLaDA 8B / 1.5 | MIT ^120^| Yes | Yes | Yes | Yes | HuggingFace |
| LLaDA 2.0 (100B) | Open ^133^| Yes | Yes | Partial | Yes | HuggingFace |
| Dream 7B | Apache 2.0 ^149^| Yes | Yes | Yes | Yes | HuggingFace |
| Dream-Coder 7B | Apache 2.0 ^127^| Yes | Yes | Yes | Yes | HuggingFace |
| DiffuCoder 7B | Apple OSS ^128^| Yes | Yes | Yes | Yes | HuggingFace + GitHub |
| SEDD | Open (GitHub) | Yes | Yes | Yes | Yes | GitHub |
| Mercury | Proprietary ^36^| No | No | No | Via API | API only |
| Gemini-Diffusion | Proprietary ^150^| No | No | No | Via API | API only |
| Seed-Diffusion | Proprietary | No | No | No | Via API | Internal / API |

The licensing landscape reveals a sharp bifurcation. Open-source releases (LLaDA, Dream, DiffuCoder, SEDD) provide full transparency across weights, code, and training data, with permissive licenses enabling commercial adoption. Proprietary models (Mercury, Gemini-Diffusion, Seed-Diffusion) offer only API access, with no visibility into model architecture, training data, or inference infrastructure. This bifurcation has practical consequences: open-source models enable researchers to study and improve diffusion architectures, customize models for domain-specific applications, and deploy on private infrastructure, while proprietary models offer managed reliability, optimized inference, and continuous improvement without operational burden.

The open-source ecosystem's trajectory suggests continued expansion. LLaDA's family has grown from 8B to 100B parameters within a single year; Dream has spawned specialized variants for code (Dream-Coder) and variable-length generation (DreamOn); community tools like dInfer (efficient inference by InclusionAI), the Information-Gain Sampler (ICML 2026, a unified decoding framework for masked diffusion models), and A-CFG (adaptive classifier-free guidance) are building out the infrastructure layer that mature ecosystems require ^151^ ^152^ ^153^. The open-source diffusion LLM ecosystem remains smaller than its autoregressive counterpart — measured by total GitHub stars, community contributors, or third-party integrations — but its growth rate and the concentration of innovation within it suggest that the gap will narrow substantially over the coming year.

The critical question facing the ecosystem is whether open-source diffusion models can close the quality gap with closed commercial systems. Current evidence is mixed: on HumanEval, Dream-Coder's 76.2% trails Mercury Coder Small's 86.0% by nearly 10 percentage points, but Dream-Coder matches Mercury on LiveCodeBench v4 at 21.4% versus 22.9% ^134^. A comprehensive empirical study attributes the average performance gap to "the availability of higher-quality training data in commercial settings" ^2^, suggesting that the gap is not architectural but data-related. If this diagnosis is correct, careful data curation and continued RL innovation — both of which the open-source community is actively pursuing — should enable convergence. The open-source diffusion LLM ecosystem has established architectural viability, training methodology, and licensing frameworks. The next phase will determine whether it can match the scale and quality of commercial deployments.
-e 


## 7. Reinforcement Learning and Post-Training for Diffusion

The gap between a diffusion language model's capabilities after pre-training and its performance after targeted post-training has proven remarkably wide. On LiveCodeBench, the LLaDA2.0-flash preview scored only 29.07 before post-training interventions; the final model reached 42.29 — a 45% improvement attributable almost entirely to supervised fine-tuning (SFT), confidence-aware prediction (CAP), and direct preference optimization (DPO), rather than to the base diffusion conversion itself. This pattern repeats across the literature: reinforcement learning (RL) post-training is not merely an additive refinement for diffusion models but a transformative step that addresses structural limitations inherent in how these models are trained.

This section examines the algorithms that have made such gains possible. The discussion begins with the fundamental train-test mismatch that limits standard SFT, proceeds through the principal variance-reduction and policy-optimization methods — VRPO, Coupled-GRPO, and EBPO — and concludes with the broader landscape of on-policy and multimodal RL approaches that are reshaping the diffusion post-training paradigm.

### 7.1 Why RL Outperforms SFT for Diffusion

#### 7.1.1 Three Problems with Standard SFT

Classical supervised fine-tuning for diffusion language models applies bidirectional attention across the entire response, randomly masking tokens without regard for the inference-time decoding procedure. When the same model generates text at inference, however, it typically operates in a semi-autoregressive blockwise mode: fixed-size blocks are decoded sequentially, with previously generated tokens forming clean prefixes and future tokens remaining fully hidden. This structural discrepancy creates three distinct pathologies ^154^.

First, **noisy prefixes** arise because SFT training randomly corrupts tokens throughout the response, including positions that serve as conditioning context during inference. At generation time, these prefix tokens are already committed and remain uncorrupted; the model has never seen clean prefixes as conditioning during training. Second, **dependency leakage** occurs when randomly remasked positions in the training objective reveal information about future tokens that the model will not have access to during blockwise decoding. The training signal effectively "cheats" by allowing the model to attend to tokens that inference will keep masked. Third, **granularity mismatch** reflects the fundamental difference between optimizing individual token-level predictions during training and making coordinated block-level decisions during inference. Token-level cross-entropy loss does not capture the block-conditional dependencies that govern actual generation ^154^.

Empirically, these three problems limit the effectiveness of SFT for diffusion models. Standard SFT provides marginal gains over the pre-trained base, and in some cases can degrade performance by reinforcing mismatched training dynamics.

| Problem | Description | Training Behavior | Inference Behavior | Impact on Performance |
|---------|-------------|-------------------|-------------------|----------------------|
| Noisy prefixes | Random masking corrupts prefix tokens used as conditioning | Prefix tokens may be masked or corrupted | Prefix tokens are clean and committed | Model never learns to condition on clean prefixes |
| Dependency leakage | Future tokens are visible through random remasking | Bidirectional attention sees all positions | Blockwise decoding hides future blocks | Model overestimates available context |
| Granularity mismatch | Loss computed at token level | Token-level cross-entropy optimization | Block-level commitment decisions | Misaligned optimization objective ^154^|

Blockwise SFT addresses each of these problems by restructuring the training objective to mirror the inference procedure. The response is partitioned into fixed-size blocks, and at each training step exactly one block is selected as "active" for stochastic masking. All preceding blocks are frozen (clean prefixes), and all subsequent blocks are fully hidden. This architecture yields a variational upper bound on blockwise likelihoods with unbiased timestep-sampled gradients ^155^. Experiments under matched compute or token budgets show consistent gains on GSM8K, MATH, and MetaMathQA, with performance peaking when the training block size matches the inference block size — confirming that training-inference alignment is a core driver of diffusion model performance ^154^ ^155^.

#### 7.1.2 Structure-Aware Training

The principle of aligning training with inference structure extends beyond blockwise partitioning to the semantic structure of the training data itself. TreeDiff, the first work to incorporate Abstract Syntax Tree (AST)-aware masking into diffusion language models for code generation, demonstrates that random token masking is suboptimal for structured programming languages. TreeDiff assigns tiered masking probabilities to different AST node types: structural elements (imports, function definitions) receive lower weights to preserve high-level program architecture, while logic and control flow tokens (if statements, while loops) receive higher weights to focus learning on algorithmic reasoning ^58^. The method employs a hierarchical probability scheme combining AST-weighted masking with curriculum noise scheduling, and treats reasoning chains (natural language) and code as distinct modalities with modality-specific corruption strategies ^13^.

The empirical gains from this approach are substantial. TreeDiff achieves a 13.3% relative improvement over random masking on HumanEval+ ^12^ ^13^. At generation length $T = 256$, TreeDiff scores 42.1% on HumanEval and 37.2% on HumanEval+; at $T = 512$, it maintains 36.6% and 33.3% respectively while random masking baselines degrade significantly under the longer generation horizon ^156^. These results confirm that structure-aware training is not merely an incremental improvement but a qualitatively different approach that respects the hierarchical nature of code.

#### 7.1.3 The Post-Training Multiplier: Evidence from LLaDA2.0

The magnitude of improvement from post-training relative to base model quality deserves quantitative emphasis. LLaDA2.0-flash-preview, the model immediately after the WSD (Warmup-Stabilize-Decay) diffusion conversion without any post-training, scored 29.07 on LiveCodeBench. The final model, after SFT, CAP, and DPO, reached 42.29 — a 45% relative improvement ^5^. This gain is not from scaling parameters, increasing model size, or modifying the architecture; it is purely from post-training interventions. The implication is that diffusion models are currently in an "RL-shaped" regime where the quality bottleneck lies in the alignment of training with inference, not in the representational capacity of the base model.

This pattern aligns with findings from d1, the first policy gradient RL work for masked diffusion models, which showed that SFT followed by diffu-GRPO outperforms either stage in isolation ^157^. Across virtually all benchmarks, RL-augmented training pipelines consistently surpass SFT-only approaches, suggesting that the train-test mismatch documented by Blockwise SFT is not a minor optimization concern but a fundamental limitation that RL methods are uniquely positioned to address.

### 7.2 VRPO: Variance-Reduced Preference Optimization

#### 7.2.1 Variance Reduction for Diffusion Preference Optimization

Preference optimization for diffusion models faces a fundamental statistical challenge: the Evidence Lower Bound (ELBO), which serves as the standard proxy for intractable sequence-level log-likelihoods, introduces both bias and variance into policy gradient estimates. The magnitude of this corruption is governed directly by the variance of the preference score estimator, and without intervention, gradient estimates can be too noisy to support stable learning ^19^.

Variance-Reduced Preference Optimization (VRPO), introduced in LLaDA 1.5 by Zhu et al., addresses this through three principled techniques that together reduce the variance of preference optimization gradients to levels that enable effective post-training ^19^ ^158^.

The first technique is **increased Monte Carlo sampling budget**: VRPO uses $n = 8$ samples for ELBO estimation by default, increasing the sample count beyond what standard implementations typically employ. Theorem 2 in the VRPO paper establishes that the variance of the preference score estimator $V[\hat{B}(y)] = \Theta(1/n)$, confirming that larger sampling budgets reduce variance inversely with sample size ^121^.

The second technique is **optimal allocation of the sampling budget across timesteps**. Rather than drawing multiple masked samples per timestep, VRPO allocates the full budget of $n$ samples across different timesteps while drawing only one masked sample per timestep ($n_t = n$, $n_{y_t} = 1$). Theorem 2 proves that this allocation minimizes variance compared to alternatives ^121^.

The third technique is **antithetic sampling between paired ELBO estimates**. By sharing Monte Carlo samples between the ELBO estimates of the model policy and the reference policy, VRPO induces positive correlation between the paired estimates. Theorem 3 proves that this reduces variance whenever the correlation between model and reference policy estimates is positive, which holds in practice for fine-tuning scenarios where the model policy remains close to the reference ^121^.

#### 7.2.2 Cost-Efficient Results on LLaDA-8B

VRPO was applied to LLaDA-8B-Instruct with 350K preference pairs. The training cost is remarkably low: less than 0.5% of the pre-training compute budget ^19^. The results demonstrate substantial gains across multiple benchmarks: GSM8K +4.7 absolute points, HumanEval +3.0, MBPP +1.8, IFEval +4.0, and Arena-Hard +4.3. VRPO is theoretically extensible to PPO and GRPO variants, though the initial implementation used a DPO-style policy update ^19^ ^158^.

The significance of these results lies not only in the magnitude of improvement but in the efficiency: VRPO achieves these gains with a fraction of the computational investment that comparable AR model post-training requires. For compute-constrained practitioners, VRPO provides a principled, theoretically grounded approach to diffusion model alignment that does not demand massive additional resources.

### 7.3 Coupled-GRPO: Complementary Mask Sampling

#### 7.3.1 Core Innovation: Full Token Coverage Through Complementary Masks

Coupled-GRPO, developed for the DiffuCoder 7B-parameter code generation model, addresses a coverage problem inherent in standard masking approaches for diffusion RL. When a single random mask is applied to a sequence for policy gradient estimation, some tokens may be unmasked in both the chosen and rejected completions, leaving gaps in the training signal. Over many training steps, these coverage gaps accumulate into biased gradient estimates that favor certain token positions over others ^159^.

The coupled-sampling scheme generates **paired complementary masks** for each completion: for a given sequence, two masks are created such that every token position is masked in exactly one of the two masks. The log-probability estimate is derived by averaging losses from these two complementary forward passes, ensuring every token is evaluated in a partial-masking context during training ^159^. This design provides full token coverage and a more stable gradient signal compared to single random mask or full-mask approaches.

#### 7.3.2 Results and Training Pipeline

DiffuCoder is trained on 130 billion effective tokens through a four-stage pipeline: adaptation pre-training (65B tokens), mid-training (16B tokens), SFT on 436K samples, and coupled-GRPO RL on 21K hard samples ^15^ ^16^. The hard samples are filtered from the Acecoder-87K dataset by selecting problems in the bottom 20% by pass rate and top 40% by solution variance, ensuring that RL training focuses on the most challenging and educationally valuable examples ^15^.

The reward function combines execution pass rate (0.5 weight) and format correctness (0.5 weight), with code execution verified via the E2B sandbox ^160^. Training completes in 40 hours on 8 H100 GPUs — a modest hardware investment for the gains achieved. Coupled-GRPO boosts the EvalPlus score by 4.4% over the SFT-only model ^16^. Beyond raw score improvement, coupled-GRPO training reduces autoregressive behavior ("AR-ness"), enabling more parallel generation and higher effective throughput at inference time ^161^ ^162^.

| Stage | Data Volume | Tokens | Hardware | Duration |
|-------|-------------|--------|----------|----------|
| Adaptation pre-training | — | 65B | — | — |
| Mid-training | — | 16B | — | — |
| SFT | 436K samples | — | — | — |
| Coupled-GRPO RL | 21K hard samples | — | 8 × H100 | 40 hours |

The table above summarizes the DiffuCoder training pipeline. The progression from broad pre-training through increasingly targeted stages — narrowing from 65B tokens to 21K carefully filtered hard examples — exemplifies the data-concentration strategy that diffusion models enable. The 21K hard samples were selected by filtering Acecoder-87K for problems with bottom-20% pass rates and top-40% solution variance, yielding the most challenging examples for RL optimization ^15^.

### 7.4 EBPO: ELBO-Based Block-Level Policy Optimization

#### 7.4.1 Scaling RL to 100B Parameters

ELBO-based Block-level Policy Optimization (EBPO), introduced in LLaDA2.1 by the Ant Group team, represents the first successful application of large-scale RL to a diffusion language model at the 100-billion-parameter scale ^17^ ^18^. Prior to EBPO, RL for diffusion models had been limited to small-scale experiments, primarily because standard policy gradient methods require sequence-level log-likelihoods that are computationally intractable for non-autoregressive, parallel-decoding diffusion architectures ^20^.

EBPO overcomes this intractability through two core innovations. First, it uses the ELBO as a principled proxy for exact sequence-level log-likelihood, accepting the inherent bias of the lower bound in exchange for computational tractability. Second, it introduces **Vectorized Likelihood Estimation** to parallelize bound computation across blocks and timesteps, achieving what the authors describe as "orders-of-magnitude acceleration" relative to naive estimation ^17^.

The EBPO objective maximizes a clipped surrogate function weighted by a probability ratio $\rho$, following the PPO-style clipping approach that has become standard in AR model alignment:

$$J_{\text{EBPO}} = \mathbb{E}\left[\min\left(\rho \cdot \hat{A}, \text{clip}(\rho, 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}}) \cdot \hat{A}\right)\right]$$

Block-conditional log probabilities are aggregated in parallel across discretized timesteps and blocks:

$$\log \rho(y|x) \approx \sum_{n=1}^{N} w_n \sum_{b=1}^{B} \left(\log p_\theta(y^b | z_n, x; M) - \log p_{\theta_{\text{old}}}(y^b | z_n, x; M)\right)$$

where $z_n$ denotes masked intermediate states, $w_n$ are aggregation weights, and the sum over $b$ indexes blocks within the response ^17^. This block-level formulation is the key departure from standard RL: instead of computing sequence-level log-likelihoods (impossible for diffusion models due to their fully connected attention structure), EBPO operates at the block level, leveraging block-causal attention to compute tractable conditional probabilities within each block.

#### 7.4.2 Results and Deployment

LLaDA2.1-Flash (100B), trained with EBPO, achieves 892 tokens per second (TPS) on HumanEval+, 801 TPS on BigCodeBench, and 663 TPS on LiveCodeBench when quantized to FP8 ^18^. The RL training extends the AReaL framework with specialized likelihood estimation and advantage estimation protocols that explicitly support both the T2T (Token-to-Token editing) and M2T (Mask-to-Token drafting) modes of LLaDA2.1's decoding architecture ^17^. EBPO's success at the 100B scale demonstrates that the algorithmic barriers to RL for large diffusion models are surmountable, opening the door for RLHF-style alignment pipelines comparable to those already standard for autoregressive models.

### 7.5 On-Policy and Other RL Approaches

The landscape of RL methods for diffusion models extends well beyond VRPO, Coupled-GRPO, and EBPO. A growing body of research has produced a diverse toolkit of algorithms, each addressing different aspects of the diffusion RL problem: variance in gradient estimates, credit assignment across denoising steps, multimodal generalization, and computational scalability. This section surveys the major approaches and their comparative performance.

| Algorithm | Policy Update | Likelihood Estimation | Masking Strategy | Key Innovation | Best Benchmark Results |
|-----------|--------------|----------------------|-------------------|----------------|----------------------|
| **VRPO** (LLaDA 1.5) | DPO-style | ELBO with random masking, $n=8$ | Random; timestep-wise allocation | Theorems 2-3: optimal budget allocation + antithetic sampling | GSM8K +4.7, HumanEval +3.0, IFEval +4.0 ^19^|
| **Coupled-GRPO** (DiffuCoder) | GRPO | Complementary mask averaging | Complementary mask pairs | Full token coverage via paired complementary masks | EvalPlus +4.4%, reduced AR-ness ^16^|
| **EBPO** (LLaDA2.1) | PPO-style clipped surrogate | Vectorized block-conditional ELBO | Block-level | First large-scale RL for dLLMs at 100B; parallel block computation | 892 TPS HumanEval+ ^18^|
| **UniGRPO** (MMaDA) | GRPO | ELBO with random masking | Structured noising ($p_i \in [0,1]$ uniform) | Unified RL across text, vision, T2I generation | SOTA on MMU, T2I ^163^|
| **SPG** | Policy gradient | ELBO (positive); EUBO/Mixture (negative) | Block-wise masking | Sandwiched upper and lower bounds reduce bias | +3.6% GSM8K, +18.4% Countdown, +27.0% Sudoku ^164^|
| **wd1** | Weighted likelihood | One-step estimation | Prompt masking | Ratio-free, requires only single likelihood estimate | 44.2% MATH500, 84.5% GSM8K ^165^|
| **AGRPO** | GRPO | Monte Carlo sampling | — | First unbiased policy gradient for dLLMs | +7.6% GSM8K, 3.8× Countdown ^166^|
| **TraceRL** | PPO with trace steps | Trajectory-level with shrinkage | Trace step aggregation | Diffusion value model; trajectory-aware credit assignment | TraDo-4B > Qwen2.5-7B on math ^167^|
| **DiSPO** | Plug-in to base PO | State-wise masked-token surrogate | Intermediate state branching | Optimizes intermediate filling decisions | Improves diffu-GRPO/SPG baselines ^168^|

The comparison table above reveals a clear progression in the field. Early methods (VRPO, Coupled-GRPO) focused on reducing variance in ELBO-based estimates through sampling strategies. Intermediate methods (SPG, AGRPO) attacked the bias problem by sandwiching the true log-likelihood between bounds or by computing unbiased Monte Carlo estimates. The most recent approaches (EBPO, TraceRL, DiSPO) exploit the sequential structure of diffusion denoising for finer-grained credit assignment. wd1 occupies a unique position as a ratio-free method that achieves remarkable efficiency — wd1++ reaches 44.2% on MATH500 and 84.5% on GSM8K with only 20 RL training steps ^165^.

![RL Post-Training Techniques for Diffusion LLMs: Key Performance Results](/mnt/agents/output/fig_rl_techniques_comparison.png)

**Figure 7.1 — RL Post-Training Techniques for Diffusion LLMs: Key Performance Results.** The bar chart displays the headline performance metric for six leading RL methods applied to diffusion language models. VRPO shows absolute gains on GSM8K; Coupled-GRPO shows EvalPlus improvement; SPG shows GSM8K gains from sandwiched bounds; wd1++ shows the total MATH500 score achieved with only 20 RL steps; AGRPO shows GSM8K gains from unbiased gradients; TraceRL shows MATH500 gains via trajectory-aware credit assignment. Annotations beneath each bar indicate the key technical innovation or training resource requirement. Note that wd1++ reports a total score (44.2% MATH500) rather than an absolute gain, as it represents state-of-the-art performance for diffusion models on mathematical reasoning.

#### 7.5.1 Seed Diffusion and TraceRL: On-Policy Optimization

Two notable approaches explore on-policy RL for diffusion models from different angles. Seed Diffusion implements end-to-end on-policy optimization with step minimization, directly optimizing the generation process to reduce the number of denoising steps while maintaining output quality. TraceRL takes a trajectory-aware approach, decomposing inference into intermediate "trace steps" and applying PPO with clipped policy ratios and KL regularization at each step rather than allocating rewards only at sequence completion ^169^ ^170^.

TraceRL introduces a **diffusion value model** that outputs token-wise value estimates conditioned on the trace prefix, reducing variance and improving stability over terminal-reward-only methods ^167^ ^170^. A **shrinkage parameter** $s$ aggregates $s$ consecutive trace steps, reducing the number of forward passes required for each policy update ^169^. TraceRL powers the TraDo model family: TraDo-4B-Instruct outperforms Qwen2.5-7B-Instruct on math reasoning despite having only 4B parameters, and TraDo-8B-Thinking is the first long-chain-of-thought diffusion language model ^167^ ^171^. On MATH500, TraDo-4B achieves +5.4% static accuracy and +4.2% dynamic accuracy over the base model; TraDo-8B achieves +4.2% static and +4.8% dynamic ^171^. TraceRL has also been applied to CUDA kernel generation in the DICE framework, where a bi-phase curriculum — kernel infilling followed by end-to-end generation — mitigates deceptive behaviors including defaulting to PyTorch functions and generating valid kernels without invocation logic ^172^.

#### 7.5.2 UniGRPO/MMaDA: Unified Multimodal RL

UniGRPO extends the GRPO framework to unified multimodal reasoning and generation through the MMaDA 8B-parameter model. It addresses three critical challenges in adapting GRPO to diffusion models: local masking dependency (where masking at one position affects gradients at others), mask ratio sensitivity (where different masking ratios produce widely varying gradient magnitudes), and non-autoregressive sequence-level likelihoods ^163^.

UniGRPO's key innovation is a **structured noising strategy** that uniformly samples mask ratio $p_i \in [0, 1]$ rather than masking all response tokens, ensuring the model is exposed to various stages of multi-step diffusion denoising during training ^173^. Sequence-level log-likelihood is approximated by averaging over masked tokens using ELBO with random masking ^174^. MMaDA employs a mixed long chain-of-thought fine-tuning strategy that curates a unified CoT format across modalities, facilitating cold-start training for RL ^163^. The results demonstrate the breadth of UniGRPO's applicability: MMaDA-8B surpasses LLaMA-3-7B and Qwen2-7B in textual reasoning, outperforms Show-o and SEED-X in multimodal understanding, and excels over SDXL and Janus in text-to-image generation ^163^.

#### 7.5.3 RL Scaling: The 33× Data Robustness Advantage

A finding with broad implications for diffusion model training comes from scaling-law research on data repetition. When training with repeated data, autoregressive models begin to overfit after approximately 4 epochs, showing clear signs of degradation. Diffusion models, by contrast, exhibit no signs of overfitting even after 100 epochs of repetition ^43^. The quantitative measure of this difference is the **half-life of data reuse**, denoted $R_D^*$: the number of epochs after which repeated data becomes half as effective as fresh data. For AR models, $R_D^* \approx 15$ epochs; for diffusion models, $R_D^* \approx 500$ epochs — a **33× difference** ^43^.

| Property | Autoregressive Models | Diffusion Models | Ratio |
|----------|----------------------|-------------------|-------|
| Half-life of data reuse ($R_D^*$) | ~15 epochs | ~500 epochs | 33× |
| Overfit onset | ~4 epochs | >100 epochs (no clear onset) | >25× |
| 1.7B model trained on 10B unique Python tokens | Baseline | Overtakes AR with matched 1.5T-token compute ^44^| — |
| 1B model trained on 1B repeated tokens | — | >56% HellaSwag, >33% MMLU ^44^| — |
| Practical guidance | Prefer when compute-constrained | Prefer when data-constrained ^43^| — |

The practical implication for RL post-training is significant. RL-generated training data is expensive to produce: each preference pair requires multiple model rollouts and often execution-based verification. The diffusion model's tolerance for data repetition means that the same RL training corpus can be reused far more extensively than would be possible for AR models, reducing the data generation burden by an order of magnitude. This advantage compounds with the already favorable compute characteristics of methods like VRPO (<0.5% pre-training cost) and coupled-GRPO (21K hard samples).

The broader methodological progression across the field follows a clear arc. Early work used one-step likelihood estimation (d1/diffu-GRPO), which is computationally efficient but potentially biased ^157^ ^132^. The next generation employed ELBO with random masking (VRPO, UniGRPO), which is principled but high-variance without variance reduction ^19^. SPG addressed the one-sided bias of ELBO by sandwiching the true log-likelihood between upper and lower bounds ^164^. AGRPO introduced unbiased Monte Carlo gradient estimation ^166^. EBPO scaled these insights to 100B parameters through vectorized block-conditional computation ^17^. The frontier now lies in trajectory-aware methods (TraceRL, DiSPO, SAPO) that exploit the sequential structure of diffusion denoising for finer credit assignment at intermediate states ^167^ ^168^ ^175^. Each generation has reduced both the bias and variance of policy gradient estimates, bringing diffusion RL closer to the maturity already achieved for autoregressive models.

Sandwiched Policy Gradient (SPG) warrants additional discussion as the current state-of-the-art for diffusion reasoning tasks. SPG leverages both an upper bound (Evidence Upper Bound, EUBO) and a lower bound (ELBO) of the true log-likelihood. For sequences with positive rewards, SPG maximizes the lower bound; for sequences with negative rewards, it minimizes the upper bound. This two-sided approach eliminates the bias inherent in one-sided approximations, which can be particularly severe when negative rewards are involved ^164^ ^176^. SPG achieves gains of +3.6% on GSM8K, +2.6% on MATH500, +18.4% on Countdown, and +27.0% on Sudoku over prior RL methods for diffusion models, significantly outperforming D1, WD1, and UniGRPO baselines ^164^. wd1 takes a different approach, reformulating the RL objective as a weighted log-likelihood that requires only a single approximation for the current policy likelihood — eliminating policy ratios entirely. wd1++ extends this to denoising-stepwise weighted policy optimization, achieving the strongest reported math performance among diffusion models ^165^ ^177^.

The convergence of these diverse approaches on a shared conclusion is noteworthy: RL is not an optional refinement for diffusion language models but an essential component of the training pipeline. The magnitude of gains — from VRPO's +4.7 on GSM8K to AGRPO's +7.6 to SPG's +18.4 on Countdown — far exceeds what would be achievable through SFT alone. The 45% improvement from post-training observed in LLaDA2.0, and the 13.3% gain from structure-aware masking in TreeDiff, both point to the same underlying principle: diffusion models benefit disproportionately from training procedures that respect their unique inference-time structure. RL algorithms that are designed with this structure in mind — whether through complementary masks, block-level optimization, sandwiched bounds, or trajectory-aware credit assignment — consistently deliver the largest improvements.
-e 


## 8. Inference Speed Optimization and Deployment

### 8.1 The Speed Advantage: Claims vs Reality

#### 8.1.1 Headline Claims and the Hardware Discrepancy Problem

Inference speed is the most frequently cited commercial advantage of diffusion language models. The headline numbers are striking: Seed Diffusion reports 2,146 tokens per second (tok/s) on H20 GPUs ^6^; LLaDA2.1 Mini reaches 1,587 transactions per second (TPS) with quantization ^17^; Gemini Diffusion clocks 1,479 tok/s on undisclosed hardware ^3^; and Mercury Coder Mini achieves 1,109 tok/s on H100 GPUs ^7^. These figures, if taken at face value, suggest that diffusion models have already surpassed autoregressive (AR) models by an order of magnitude in inference throughput.

The reality is considerably more nuanced. A systematic analysis of the measurement conditions behind each claim reveals that no two models were evaluated under comparable circumstances. Table 1 enumerates the hardware and software configurations for the major reported speed claims.

**Table 1: Reported Speed Claims and Their Measurement Conditions**

| Model | Reported Speed | GPU | Memory Bandwidth | Serving Stack | Notes |
|-------|---------------|-----|------------------|---------------|-------|
| Seed Diffusion | 2,146 tok/s ^6^| H20 (96 GB HBM3) | 4.0 TB/s | Custom | Inference-optimized GPU, not directly comparable to H100 results |
| LLaDA2.1 Mini (quantized) | 1,587 TPS ^17^| Unspecified | Unknown | Custom | Quantized; hardware undisclosed; Speed Mode (S-mode) |
| Gemini Diffusion | 1,479 tok/s ^3^| Unknown | Unknown | Unknown | Experimental model; no hardware or serving stack disclosed |
| Mercury Coder Mini | 1,109 tok/s ^7^| H100 (80 GB HBM2e) | 3.35 TB/s | Proprietary | Lower memory bandwidth than H20; proprietary serving stack |
| LLaDA2.0-flash-CAP | 535 TPS ^8^| H20, SGLang TP8 | 4.0 TB/s | SGLang | Controlled fair-comparison environment |

The hardware discrepancy alone renders cross-model comparisons unreliable. The NVIDIA H20 is specifically designed for inference-heavy workloads, offering 96 GB of HBM3 memory with 4.0 TB/s bandwidth — higher than the H100's 3.35 TB/s ^178^. Seed Diffusion's 2,146 tok/s on H20 versus Mercury's 1,109 tok/s on H100 cannot therefore be interpreted as evidence that Seed Diffusion's architecture is twice as fast; a meaningful portion of the gap may be attributable to the hardware advantage. ByteDance itself acknowledges this limitation, noting that "direct comparison with baselines is challenging due to differing test conditions: Mercury Coder was evaluated on a proprietary dataset with H100s, while Gemini Diffusion's speed was averaged over a mixed-task benchmark using unknown hardware" ^6^.

Beyond hardware, the serving stack introduces additional variability. vLLM's PagedAttention and SGLang's RadixAttention — the dominant optimized inference engines for AR models — were designed for unidirectional autoregressive generation and offer no native support for bidirectional diffusion attention ^33^. Models evaluated on custom or proprietary serving stacks may benefit from optimizations unavailable to others, further biasing comparisons.

![Figure 8.1: Reported headline speed claims versus controlled comparison under identical hardware and serving conditions](/mnt/agents/output/fig8_1_speed_comparison.png)

#### 8.1.2 Controlled Comparison: The SGLang Benchmark

The most rigorous controlled comparison to date comes from LMSYS's SGLang integration, which evaluated LLaDA2.0-flash-CAP alongside AR baselines on identical hardware (H20 GPU) with identical serving infrastructure (SGLang with tensor parallelism at TP8) ^8^. Under these conditions, LLaDA2.0-flash-CAP achieved 535 TPS, while the AR baselines averaged approximately 247 TPS (258 TPS and 237 TPS for the two AR variants). This yields a **2.1x speedup** for diffusion over AR — a meaningful advantage, but far below the 5-10x claims that pervade vendor marketing materials. Inception Labs, for instance, claims Mercury Coder is "up to 10x faster" than speed-optimized AR models ^7^; Peng et al.'s controlled study found no evidence supporting multiples of that magnitude under fair conditions ^33^.

#### 8.1.3 The Measurement Conditions Problem

Peng et al. (Renmin University) identify three systematic biases in existing diffusion language model (dLLM) efficiency evaluations ^33^. First, inconsistent serving environments: many papers measure speed under HuggingFace Transformers, a reference implementation not optimized for production throughput, while others use highly tuned engines. Second, unfair generation length controls: diffusion models can directly control output length, whereas AR models naturally stop at the end-of-sequence token; forcing AR models to generate to a fixed length inflates their apparent latency. Third, batch size sensitivity: acceleration strategies yield significant gains at batch size 1 — the regime most relevant for interactive applications — but their advantage diminishes as batch size grows, eventually falling behind AR models with mature serving stacks ^33^. The interaction of these three effects explains why headline claims diverge so dramatically from controlled results.

### 8.2 Training-Free Acceleration Techniques

The most rapid progress in diffusion inference optimization has come from **training-free** methods that operate off-the-shelf on existing pretrained models. This paradigm is attractive because it requires no additional training data or compute — a crucial consideration for production deployment where retraining billion-parameter models is prohibitively expensive. Figure 8.2 summarizes the speedups achieved by the leading training-free techniques.

![Figure 8.2: Training-free acceleration techniques ranked by reported speedup over baseline diffusion inference](/mnt/agents/output/fig8_2_acceleration_methods.png)

#### 8.2.1 Fast-dLLM: Block-Wise Approximate KV Cache

Fast-dLLM, developed by NVIDIA, the University of Hong Kong, and MIT (Song Han's group) and published at ICLR 2026, is the most widely cited acceleration framework for diffusion LLMs ^179^. Its core contribution is a **block-wise approximate KV cache** that exploits the structure of diffusion generation. The approach partitions generation into blocks; KV states of the fixed context (the prompt and any completed blocks) are cached and reused across denoising steps, refreshed only at block boundaries. A "DualCache" variant extends this by caching both prefix and suffix blocks ^179^.

The second pillar of Fast-dLLM is **confidence-aware parallel decoding**. Unlike prior approaches that select a fixed number of tokens per denoising step — which disrupts token dependencies under the conditional independence assumption — Fast-dLLM dynamically selects only tokens whose confidence exceeds a global threshold ^180^. This preserves dependency structure while maximizing parallelism at each step.

Empirically, Fast-dLLM achieves up to **27.6x throughput improvement** on LLaDA and Dream models across GSM8K, MATH, HumanEval, and MBPP benchmarks, with the highest acceleration observed at longer generation lengths (1,024 tokens) ^40^. Confidence-aware parallel decoding alone contributes 13.3x speedup ^181^. Fast-dLLM v2 extends the framework to convert pretrained AR models into block diffusion models with only approximately 1B tokens of fine-tuning, achieving up to 2.5x speedup over standard AR decoding through block-level KV cache reuse and intra-block parallel decoding ^182^ ^183^.

#### 8.2.2 Elastic-Cache: Attention/Depth-Aware Adaptive KV Caching

Elastic-Cache, from MBZUAI's VILA Lab (Zhiqiang Shen) and also published at ICLR 2026, achieves the highest reported speedups among training-free methods at **45.1x** on long sequences ^41^. The method rests on three empirical observations about diffusion model behavior during inference: (1) distant MASK tokens primarily act as length bias and can be cached block-wise beyond the active prediction window; (2) KV dynamics increase with network depth, suggesting that selective refresh from deeper layers is sufficient; and (3) the most-attended token exhibits the smallest KV drift, providing a conservative lower bound on cache change ^41^.

Elastic-Cache jointly decides when to refresh (via an attention-aware drift test on the most-attended token) and where to refresh (via a depth-aware schedule that recomputes from a chosen layer onward while reusing shallow-layer caches and off-window MASK caches). This yields 8.7x speedup on GSM8K at 256-token generation lengths, 45.1x on longer sequences, and 4.8x on HumanEval, while consistently maintaining higher accuracy than the no-cache baseline ^41^. Notably, Elastic-Cache achieves 6.8x higher throughput than existing confidence-based approaches on GSM8K, demonstrating the compounding benefit of attention-aware and depth-aware caching combined.

#### 8.2.3 FreeCache and Guided Diffusion

FreeCache leverages the observation that "the impact of future tokens on earlier positions rapidly diminishes over denoising steps" to directly cache KV states of already-decoded "clean" tokens ^42^. Because tokens that have converged to their final values no longer need to participate fully in subsequent denoising iterations, their KV states can be frozen and reused — a delayed caching strategy analogous to the delayed commitment pattern in diffusion sampling. FreeCache achieves up to **34x speedup** (averaged on PiQA) on Dream-7B with negligible accuracy drop, and for the first time enables diffusion models to achieve generation speed comparable to same-sized AR models ^42^. It also enables long-context diffusion (exceeding 1,024 tokens) without performance degradation. The Guided Diffusion component augments this with a lightweight autoregressive "guider" model that directs unmasking toward optimal token positions, combining the parallelism of diffusion with the sequential guidance of AR models ^42^.

#### 8.2.4 SSD: Lossless Self-Speculative Decoding

SSD (Self Speculative Decoding), from Shanghai Jiao Tong University, Shanghai AI Lab, and Huawei, occupies a unique position as the only **lossless** acceleration method among the major training-free techniques ^184^. SSD uses the dLLM itself as both the speculative decoding drafter and the verifier, eliminating the need for auxiliary draft models. It generates predictions for multiple positions simultaneously, then verifies them through hierarchical verification trees in a single forward pass — exploiting the dLLM's inherent parallel prediction capability ^185^.

On Dream-7B-Instruct, SSD achieves **3.46x speedup** (from 6.37 to 22.07 TPS) with a 77.4% reduction in decoding steps, while producing output **identical** to stepwise decoding ^185^. On LLaDA-8B-Instruct, it achieves 2.11x speedup. The lossless property makes SSD particularly attractive for production deployments where output quality guarantees are paramount — unlike methods that trade a small accuracy loss for large speedups, SSD preserves exact model behavior.

**Table 2: Training-Free Acceleration Techniques — Comparative Overview**

| Method | Institution | Speedup | Lossless? | Key Mechanism | Sequence Length Sensitivity |
|--------|------------|---------|-----------|---------------|---------------------------|
| Elastic-Cache | MBZUAI (VILA Lab) | 45.1x (long) ^41^| No | Attention/depth-aware adaptive KV cache | Higher on long sequences |
| FreeCache | — | 34x ^42^| No | Delayed KV cache of "clean" tokens + AR guider | Enables >1024 tokens |
| Fast-dLLM v1 | NVIDIA + HKU + MIT | 27.6x ^40^| No | Block-wise KV cache + confidence-aware parallel decode | 27.6x at 1024 tokens |
| COVER | — | 11.6x ^186^| No | KV cache override for revocable decoding | Context-preserving |
| Sparse-dLLM | — | Up to 10x ^187^| No | Attention-aware bidirectional cache eviction | Sparse attention dependent |
| dLLM-Cache | — | 9.1x ^188^| No | Long-interval prompt cache + adaptive response cache | V-verify mechanism |
| WINO | SJTU | 6-10x ^189^| No | Revocable draft-and-verify decoding | Draft correction cycles |
| dKV-Cache | NUS | 2-10x ^190^| Near-lossless | Delayed KV caching one step post-decoding | Decode variant: near-lossless |
| SSD | SJTU + Huawei | 3.46x ^185^| **Yes** | Self-speculative hierarchical verification | Constant per verify tree |
| Di4C | — | ~2x ^191^| No | Inter-token correlation distillation | Step-count dependent |

The training-free paradigm dominates the acceleration landscape for good reason: these methods require no model retraining, no calibration data, and no hyperparameter search per deployment target. They can be applied to any pretrained diffusion model and combined with each other (e.g., Fast-dLLM's block cache with SSD's verification trees) for compounding gains. The maturity of this ecosystem — six independent methods published within a twelve-month window, all at major venues — signals that the community has converged on inference optimization as the highest-leverage near-term improvement vector.

### 8.3 Architectural Speed Optimizations

Beyond training-free caching and speculative decoding, several architectural and system-level optimizations target the core efficiency of diffusion model inference.

#### 8.3.1 Alpha-MoE Megakernel

Ant Group's Alpha-MoE architecture introduces a **megakernel** that fuses two consecutive FusedMoE (Mixture of Experts) operations into a single kernel launch, eliminating the inter-kernel launch overhead that becomes a bottleneck at small batch sizes ^5^. This is particularly impactful for diffusion models because their parallel decoding generates multiple tokens simultaneously, creating bursty small-batch compute patterns that stress the GPU's kernel dispatch pipeline. The megakernel approach, combined with Ant's dInfer inference engine, contributes to the 535 TPS achieved by LLaDA2.0-flash-CAP on H20 hardware — a 2.1x speedup over AR baselines in the same stack ^5^ ^8^.

#### 8.3.2 Per-Block FP8 Quantization

LLaDA2.1 introduces per-block FP8 quantization as part of its Speed Mode (S-mode), achieving **1,587 TPS on the Mini (16B) variant** with a quality degradation of only -0.61 score points on coding benchmarks ^17^. Unlike global quantization schemes that apply uniform precision reduction across all layers, per-block quantization adapts the numeric format to each block's sensitivity, preserving critical precision in layers where small perturbations affect output quality while aggressively compressing less sensitive blocks. This block-adaptive approach is particularly well-suited to diffusion models because their block-wise generation structure naturally aligns with the quantization granularity.

#### 8.3.3 CAP: Confidence-Aware Parallel Decoding

Confidence-Aware Parallel (CAP) decoding, introduced alongside LLaDA2.0-flash, adds an auxiliary confidence loss during training that sharpens the model's prediction confidence distribution ^8^. The sharpened confidence scores enable more aggressive threshold-based parallel token acceptance at inference time: tokens whose confidence exceeds the threshold are committed in parallel, while uncertain tokens continue through additional denoising steps. CAP alone achieves a 2.1x speedup over AR baselines when combined with the dInfer/SGLang serving stack ^8^. The training-time confidence sharpening is a complementary approach to Fast-dLLM's inference-time thresholding — together, they suggest a direction in which diffusion models are trained explicitly for efficient parallel decoding rather than having parallelism extracted post hoc.

### 8.4 Deployment Infrastructure

#### 8.4.1 The Emerging Serving Stack

The deployment infrastructure for diffusion LLMs remains immature relative to the AR ecosystem. As Peng et al. note, "major machine learning ecosystems provide only limited optimization and deployment support for DLMs, making efficient serving of DLMs difficult" ^33^. Neither vLLM nor TensorRT-LLM — the two dominant production inference engines for AR models — offers native diffusion support. vLLM's PagedAttention architecture is designed for unidirectional autoregressive attention patterns and cannot efficiently handle the bidirectional attention masks required by diffusion models. TensorRT-LLM similarly targets AR generation workflows, though NVIDIA's Model Optimizer does support diffusion vision models (Stable Diffusion, SDXL) with FP8/INT8 quantization ^192^, suggesting a pathway for future text diffusion support.

Into this gap, two projects have emerged: **dInfer** (Ant Group) and **SGLang** (LMSYS). dInfer is an open-source inference engine purpose-built for diffusion LLMs, supporting block diffusion, optimized batch inference, and FP8 quantization ^193^. SGLang provides the most mature integration, offering day-0 support for LLaDA2.0 via a Request for Comments (RFC) implementation that includes block diffusion logic, full KV cache support, streaming I/O, tensor parallelism up to TP8, and CUDA graph optimization ^8^ ^194^. Ant Group has pursued a dual strategy: developing dInfer v0.2.0 for model-specific optimizations while contributing mature features upstream to SGLang, with the stated intent that "more mature features in dInfer are undergoing to transport to SGLang" ^5^.

**Table 3: Deployment Infrastructure for Diffusion Language Models**

| Infrastructure | Diffusion Support | Models | Key Features | Maturity |
|---------------|-------------------|--------|-------------|----------|
| dInfer (Ant Group) | Native | LLaDA2.x, block diffusion | CUDA graph capture, FP8 quant, SGLang backend ^193^| v0.2.0, open source |
| SGLang (LMSYS) | Day-0 via RFC | LLaDA2.0 | RadixAttention, TP8, streaming I/O, KV cache ^8^| Production-ready |
| vLLM | None (AR only) | AR models | PagedAttention — incompatible with bidirectional attention ^33^| Mature for AR |
| TensorRT-LLM | None (vision only) | AR models; SD/SDXL for vision | Hardware-specific optimizations unavailable for text diffusion ^192^| Mature for AR |

The infrastructure gap is the primary deployment barrier for diffusion models at scale. Production serving requires not just model execution but also request scheduling, batching, tensor parallelism, memory management, and streaming — all of which must be reimplemented or adapted for diffusion's non-sequential generation pattern. Until vLLM or TensorRT-LLM adds native diffusion support, organizations deploying diffusion models at scale face a significant engineering investment.

#### 8.4.2 Batching Dynamics: The Crossover Point

A critical finding from Peng et al.'s systematic study is that diffusion models and AR models exhibit fundamentally different batching dynamics ^33^. At **batch size 1** — the regime governing interactive applications such as chatbots and IDE code completion — diffusion models with parallel decoding can outperform AR models. Block diffusion with parallel decoding is consistently fastest across all sequence lengths at this batch size, achieving up to 3.1x speedup over the block diffusion baseline ^33^. This is Mercury Coder's sweet spot: single-user IDE interactions where latency to first token completion matters more than throughput per GPU.

As **batch size increases**, the picture inverts. AR models benefit more from batching because their sequential token generation maps efficiently to batched matrix multiplications, while diffusion models' parallel decoding is already compute-saturated at batch size 1. The **turning point occurs at batch size 2–4**, where AR models overtake block diffusion in throughput ^33^. Beyond this point, AR models with optimized serving stacks (vLLM, SGLang) maintain increasing throughput, while diffusion model throughput stays nearly constant (compute-bound) until out-of-memory errors terminate scaling. LLaDA with dual cache scales better than pure parallel decoding but still hits OOM at very large batch sizes ^33^.

For **low-latency interactive applications** — chat, code completion, real-time editing — diffusion models at batch size ~1 with parallel decoding offer a genuine advantage. For **high-throughput serving** — batch APIs, document generation, embedding pipelines — AR models currently win. This bifurcation has strategic implications: diffusion models are best positioned to displace AR models in latency-sensitive, single-user contexts rather than high-throughput server-side workloads.

#### 8.4.3 The TTFT Challenge and Streaming Solutions

Diffusion models face a fundamental latency profile challenge that AR models do not: **time to first token (TTFT)**. In autoregressive models, TTFT is dominated by prompt prefill (typically 300–1,500 ms), after which tokens stream at a consistent inter-token latency of 10–30 ms ^195^. The user perceives a short initial wait followed by continuous output. In diffusion models, the entire sequence is generated iteratively through multiple denoising steps, and **no token can be considered final until denoising completes** ^33^. This effectively makes the TTFT equal to the total generation time — a qualitatively worse user experience for streaming applications.

Block diffusion provides a partial solution. By generating sequences in autoregressive blocks (e.g., 32 tokens at a time) with parallel intra-block diffusion, completed blocks can be streamed to the user while subsequent blocks are being generated ^182^. LLaDA2.x, Seed Diffusion, and Mercury Coder all employ variants of block-wise generation to enable progressive output ^6^ ^17^ ^7^. Confidence-aware parallel decoding further reduces the number of steps required before tokens stabilize, shortening the effective block generation time ^8^. Semi-autoregressive scheduling complements this by enabling progressive output even within blocks, reducing perceived latency at the cost of some generation quality.

As Redis's analysis of LLM application latency notes, "streaming is the single biggest perceived-latency optimization for LLM apps" ^195^. Diffusion models must solve streaming through block-wise or progressive generation to match AR's user experience. The block diffusion architecture, which sacrifices some parallelism for streaming compatibility, appears to be the pragmatic production standard — all commercially deployed diffusion models use it. However, the residual latency gap between block-streamed diffusion and true AR streaming remains a competitive disadvantage that must be offset by either higher generation quality or lower cost.

The cost dimension partially compensates for this latency disadvantage. Mercury Coder's pricing of $0.25 per million input tokens and $0.75–$1.00 per million output tokens is approximately **12x cheaper** than Claude Sonnet 4.5 ($3.00/$15.00 per million) and up to 30–40x cheaper than GPT-4 Turbo ($10.00/$30.00 per million) ^30^. Buildglare, a low-code web development tool, uses Mercury Coder as a "cheap embedder" — larger models plan changes, Mercury executes them at roughly an order of magnitude lower cost ^196^. The parallel generation capability enables higher GPU utilization, and the lower per-token cost may offset the streaming latency penalty for batch and non-interactive workloads.
-e 


## 9. Benchmarks and Performance Evaluation

The question of whether diffusion language models (DLMs) can match or exceed autoregressive (AR) models on code-related tasks does not yield a single answer. Instead, the conclusion depends almost entirely on which benchmarks are consulted. This chapter synthesizes performance data across all major evaluation suites and model families, revealing a systematic pattern: diffusion models achieve parity or superiority on benchmarks that reward global context understanding and parallelizable tasks, while they exhibit consistent gaps on benchmarks that demand sequential step-by-step reasoning. This *benchmark selection effect* means that both advocates and critics of the diffusion paradigm can cite credible, peer-reviewed evidence to support their positions—a Rorschach test that complicates any simple verdict on diffusion's readiness for production code workflows.

### 9.1 The Benchmark Selection Effect

The most comprehensive controlled comparison of diffusion and autoregressive code models to date, Zhang et al.'s "Beyond Autoregression" (2025), evaluated nine diffusion LLMs from six families against four AR baselines across HumanEval, MBPP, LiveCodeBench, and RepoQA ^1^. The study's aggregate findings immediately reveal the benchmark dependency: on HumanEval, diffusion models averaged 66.7% versus 71.3% for AR models—a narrow 4.6 percentage point (pp) gap—with the best diffusion model (Gemini Diffusion at 89.6%) actually surpassing the best AR baseline (Seed-Coder-8B-Instruct at 84.8%) ^2^. On MBPP, the averages essentially inverted: diffusion averaged 61.2% versus 60.8% for AR, with Seed-Diffusion-Preview reaching 79.4% against Seed-Coder's 70.8% ^2^. Yet on LiveCodeBench v6, diffusion models averaged 14.9% versus 18.9% for AR—a 27% relative deficit—and on the broader v1-v6 corpus, the gap widened further to 19.1% versus 25.8% ^2^.

This divergence creates three distinct benchmark tiers. In the *parity tier*—HumanEval, MBPP, and BigCodeBench—diffusion models match or exceed their AR counterparts. Gemini Diffusion scores 45.4% on BigCodeBench, statistically tying Flash-Lite at 45.8% ^21^, while Mercury-Coder-Small achieves 45.5% ^7^and Seed-Diffusion-Preview reaches 45.4% ^197^. In the *AR-advantage tier*—LiveCodeBench and SWE-Bench—diffusion models trail consistently. The SWE-Bench Verified gap is particularly stark: Gemini Diffusion scores 22.9% against Flash-Lite's 28.5% ^3^, a 5.6pp deficit that reflects the benchmark's demands for multi-step agentic reasoning. In the *diffusion-advantage tier*—CanItEdit and RepoQA—diffusion models demonstrate decisive superiority. Stable-DiffCoder-8B-Instruct achieves 60.0% on CanItEdit versus Seed-Coder-8B-Instruct's 50.5% ^10^, an 18.8% relative improvement, while on RepoQA's long-context retrieval task, Mercury-Coder-Small exhibits approximately 15% performance degradation when extrapolating from 8K to 64K tokens, compared to Qwen3-8B's nearly 30% drop ^14^.

The benchmark selection effect operates through two mechanisms. First, most existing code benchmarks were designed for left-to-right generation. HumanEval provides a function signature and docstring, then asks the model to complete the body in sequential order—a task structure that aligns with AR generation patterns ^2^. Second, the nature of the task itself determines which paradigm excels. Code editing (CanItEdit) is fundamentally non-sequential: modifying a function signature requires updating all call sites, a change pattern that benefits from any-order token generation ^10^. Competitive programming (LiveCodeBench) requires sequential algorithmic reasoning—designing a solution step by step, which stresses the very capability where diffusion models face structural challenges.

The result is a landscape where the choice of evaluation suite predetermines the conclusion. A researcher emphasizing HumanEval and BigCodeBench can credibly claim that "the gap... is essentially closed in terms of benchmark performance" ^21^. A researcher emphasizing LiveCodeBench and SWE-Bench can equally credibly conclude that "diffusion LLMs are not yet able to replace AR LLMs at the current stage" ^2^. Both statements are factually correct within their respective benchmark frames. Figure 9.1 visualizes this divergence across seven major benchmarks, quantifying the performance gap in percentage points for each evaluation suite.

![Figure 9.1: The Benchmark Selection Effect—diffusion vs. autoregressive performance gaps across seven code evaluation benchmarks. Positive values indicate diffusion advantage; negative values indicate AR advantage. Abbreviations: GD = Gemini Diffusion, SC = Seed-Coder, SD = Seed-Diffusion, SDC = Stable-DiffCoder, FL = Flash-Lite, Q3 = Qwen3-8B, MC = Mercury-Coder.](/mnt/agents/output/fig_benchmark_selection_effect.png)

The figure reveals a striking pattern: diffusion models lead on four of seven benchmarks (HumanEval, MBPP, CanItEdit, RepoQA) and trail on two (BigCodeBench effectively ties, SWE-Bench). The magnitude of advantages on CanItEdit (+9.5pp) and RepoQA (+15.0pp less degradation) exceeds the magnitude of disadvantages on SWE-Bench (-5.6pp), suggesting that when diffusion models excel, they excel by larger margins than when they lag. The LiveCodeBench result (+4.9pp for Gemini Diffusion specifically, though negative for diffusion averages) illustrates the model-dependency within benchmarks: while diffusion averages trail, the best individual diffusion model (Gemini Diffusion at 30.9%) can still outperform some AR competitors (Qwen3-8B at 26.0%) ^2^.

### 9.2 Code Generation Benchmarks Deep Dive

#### 9.2.1 HumanEval: The Parity Baseline

HumanEval remains the most widely cited code generation benchmark, comprising 164 hand-written Python programming problems with test-based evaluation. On this benchmark, diffusion models have achieved full competitive parity with AR models, with several individual models establishing new performance records.

Table 9.1 presents a comprehensive comparison across model families, architectures, and parameter scales.

**Table 9.1 — Comprehensive Code Generation Benchmark Comparison Across Model Families**

| Model | Family | Size | Architecture | HumanEval | MBPP | LiveCodeBench v6 | BigCodeBench | CanItEdit |
|-------|--------|------|-------------|-----------|------|------------------|-------------|-----------|
| Gemini Diffusion | Google DeepMind | — | Block diffusion | 89.6% ^3^| 62.9% ^3^| 30.9% ^3^| 45.4% ^3^| — |
| Seed-Diffusion-Preview | ByteDance | — | Diffusion | — | 79.4% ^2^| 33.7% (v1-v6) ^2^| 45.4% ^197^| 54.3% ^198^|
| Stable-DiffCoder-8B-Instruct | ByteDance | 8B | Diffusion (CPT) | 86.6% ^10^| 77.6% ^10^| 23.5% ^10^| — | 60.0% ^10^|
| Mercury-Coder-Small | Inception Labs | — | Diffusion | 86.0% ^2^| — | 22.9% ^2^| 45.5% ^7^| — |
| Mercury-Coder-Mini | Inception Labs | — | Diffusion | 88.0% ^199^| — | — | — | — |
| LLaDA2.0-flash | Ant Group | 6B/100B | Masked diffusion | 94.51% ^10^| 88.29% ^10^| 42.29% ^10^| — | — |
| Dream-Coder-v0-Instruct | Huawei/HKU | 7B | Adaptive diffusion | 76.2% ^2^| — | 21.4% ^200^| 21.4% ^200^| — |
| DiffuCoder-7B-cpGRPO | Apple/HKU | 7B | Masked diffusion | 69.5% ^2^| — | — | 40.4% ^15^| — |
| Seed-Coder-8B-Instruct | ByteDance | 8B | Autoregressive | 84.8% ^2^| 70.8% ^2^| 24.7% ^10^| — | 50.5% ^10^|
| Qwen3-8B | Alibaba | 8B | Autoregressive | — | — | 26.0% (v6) / 42.3% (v1-v6) ^2^| — | 45.7% ^2^|
| Flash-Lite | Google | — | Autoregressive | — | — | — | 45.8% ^21^| — |
| DeepSeek-Coder-6.7B-Instruct | DeepSeek | 6.7B | Autoregressive | 77.4% ^2^| — | — | — | — |

The table reveals several important patterns. At the top end, LLaDA2.0-flash achieves 94.51% on HumanEval—the highest score recorded by any diffusion model—using only 6B active parameters within a 100B total parameter mixture-of-experts (MoE) architecture ^10^. This surpasses not only all diffusion competitors but also Qwen3-30B at 93.29% ^10^, demonstrating that parameter count alone does not determine performance. Gemini Diffusion's 89.6% represents the best score among dense (non-MoE) diffusion architectures, exceeding Seed-Coder-8B-Instruct by 4.8pp ^2^. Among open-source diffusion models, Stable-DiffCoder-8B-Instruct at 86.6% surpasses its AR counterpart Seed-Coder-8B-Instruct (84.8%) in a controlled comparison using identical training data and model architecture ^10^, providing perhaps the cleanest evidence that diffusion training itself can improve code generation quality.

The distribution of scores also reveals a quality hierarchy within diffusion models. Closed-source or commercially deployed models (Gemini Diffusion, Mercury-Coder, Seed-Diffusion, LLaDA2.0-flash) cluster in the 86–95% range, while open-source models (Dream-Coder at 76.2%, DiffuCoder at 69.5%) trail by 10–20pp ^2^. This gap likely reflects training data quality and scale rather than architectural limitations, suggesting that diffusion code models are more data-hungry than their AR equivalents—a pattern consistent with the broader finding that diffusion training requires more tokens to reach equivalent loss values ^34^.

#### 9.2.2 MBPP: Diffusion Competitiveness Confirmed

The Mostly Basic Python Programming (MBPP) benchmark, with 974 crowd-sourced Python problems, confirms the HumanEval pattern. Diffusion models averaged 61.2% versus 60.8% for AR models in the "Beyond Autoregression" study—a statistically negligible difference ^2^. The standout result is Seed-Diffusion-Preview at 79.4%, surpassing Seed-Coder-8B-Instruct's 70.8% by 8.6pp ^2^. Stable-DiffCoder-8B-Instruct achieves 77.6% ^10^, while Gemini Diffusion scores 62.9% ^3^—notably lower than other top diffusion models, suggesting that Google's training prioritization may differ from ByteDance's code-focused approach. LLaDA2.0-flash reaches 88.29% ^10^, again demonstrating the quality of Ant Group's post-training pipeline.

The MBPP results are particularly significant because MBPP problems are more diverse in difficulty and domain than HumanEval's curated set. The fact that diffusion models match AR performance on this broader corpus undermines any claim that diffusion success is limited to narrow, memorization-prone evaluation suites.

#### 9.2.3 LiveCodeBench: The Persistent Competitive Programming Gap

LiveCodeBench represents the most consistent and concerning weakness for diffusion code models. This contamination-free benchmark (containing problems released after model training cutoffs) measures true generalization on competitive programming tasks ^2^. On LiveCodeBench v6, diffusion models averaged 14.9% against 18.9% for AR models—a 4.0pp absolute gap representing a 27% relative disadvantage ^2^. The gap is even wider on the v1-v6 aggregate: 19.1% versus 25.8% ^2^.

Several factors explain this gap. First, competitive programming requires chain-of-thought (CoT) reasoning—designing algorithms step by step, considering edge cases, and iteratively refining solutions. The NAP paper (Li et al., 2026) demonstrates that diffusion language models trained on standard sequential CoT data converge to autoregressive-like decoding patterns, with Global ARness@1 scores around 0.92 for Dream models ^34^. When these models are forced to use true parallel decoding, reasoning accuracy collapses: on Dream-7B evaluated on GSM8K, accuracy drops from 78.0% at 1,024 diffusion steps to 46.5% at 256 steps ^34^. This *AR-collapse* phenomenon means diffusion models underperform precisely when their parallel advantage is most needed.

Second, LiveCodeBench's task structure rewards sequential reasoning. Each problem requires reading a complex specification, identifying the algorithmic approach, implementing it, and verifying against hidden test cases. The logical dependencies between these steps create a sequential reasoning chain that diffusion's any-order generation does not naturally optimize for ^34^. Third, open-source diffusion models have had lower exposure to competitive programming content, as the "Beyond Autoregression" authors note that "open-source diffusion LLMs lag behind closed-source counterparts, possibly due to training data composition" ^2^.

However, the LiveCodeBench gap is narrowing. Gemini Diffusion at 30.9% outperforms Qwen3-8B's 26.0% on v6 ^2^, demonstrating that well-resourced diffusion models can compete. Stable-DiffCoder at 23.5% essentially matches Seed-Coder-8B-Instruct's 24.7%—a gap of only 1.2pp ^10^. And the trajectory is positive: Dream-v0 scored 13.3% on v1-v6 in April 2025, while Dream-Coder reached 24.8% by July 2025—nearly doubling in three months ^2^.

#### 9.2.4 BigCodeBench: Real-World Parity

BigCodeBench evaluates "challenging, real-world coding problems with rich context and tool-like function calls" ^10^—tasks that require integrating multiple libraries, understanding complex APIs, and generating production-quality code. On this benchmark, diffusion models demonstrate near-perfect parity with AR models. Gemini Diffusion scores 45.4% versus Flash-Lite's 45.8%—a 0.4pp difference that falls within evaluation noise ^21^. Mercury-Coder-Small achieves 45.5% ^7^, and Seed-Diffusion-Preview reaches 45.4% ^197^. DiffuCoder with coupled-GRPO reinforcement learning achieves 40.4%, a 4.7pp improvement over its pre-RL baseline of 35.7% ^15^.

The BigCodeBench results are significant because this benchmark most closely approximates real-world developer workflows—integrating external libraries, handling edge cases, and producing complete functional programs rather than isolated algorithmic solutions. The parity here suggests that for production code generation tasks, diffusion models are ready for practical deployment. As the VentureBeat analysis concludes, "the gap between the two techniques is essentially closed" on real-world code generation ^21^.

### 9.3 Code Editing and Specialized Benchmarks

#### 9.3.1 CanItEdit: Diffusion's Decisive Advantage

CanItEdit, a benchmark for code editing capability, represents diffusion models' most decisive victory over autoregressive counterparts. Code editing is fundamentally different from code completion: rather than generating a program sequentially from a prompt, editing requires understanding an existing codebase, identifying what needs to change, and making targeted modifications that preserve functionality while altering behavior. This non-sequential task structure aligns precisely with diffusion's any-order generation capability.

**Table 9.2 — CanItEdit and Code Editing Benchmark Comparison**

| Model | Architecture | Size | CanItEdit pass@1 | Aider (tries=2) | Key Strength |
|-------|-------------|------|-----------------|-----------------|--------------|
| Stable-DiffCoder-8B-Instruct | Diffusion (CPT) | 8B | **60.0%** ^10^| 54.9% ^10^| Random masking trains edit patterns |
| Seed-Diffusion-Preview | Diffusion | — | 54.3% ^198^| — | Two-stage curriculum (mask→edit) |
| Qwen2.5-Coder-14B-Instruct | Autoregressive | 14B | 52.9% ^10^| — | Larger parameter count |
| Seed-Coder-8B-Instruct | Autoregressive | 8B | 50.5% ^10^| 57.1% ^10^| AR counterpart to Stable-DiffCoder |
| Yi-Coder-9B-Chat | Autoregressive | 9B | 50.5% ^10^| — | General coding model |
| DeepSeek-Coder-33B-Instruct | Autoregressive | 33B | 46.2% ^10^| — | 4x larger than Stable-DiffCoder |
| Qwen3-8B | Autoregressive | 8B | 45.7% ^10^| 55.6% ^10^| Strong general model |

Stable-DiffCoder-8B-Instruct's 60.0% CanItEdit score surpasses all competitors, including models four times larger (DeepSeek-Coder-33B at 46.2%) ^10^. The 18.8% relative improvement over Seed-Coder-8B-Instruct (60.0% versus 50.5%) is remarkable given that the two models share identical architecture and training data—the difference is purely the diffusion training paradigm ^10^. The authors hypothesize that "random masking and reconstruction inherently train the model on edit- and infill-like patterns, enabling it to better exploit editing supervision" ^10^. During diffusion training, the model learns to reconstruct randomly masked token spans within existing code, a process structurally identical to code editing: given partial context, determine what belongs in the missing region.

Seed-Diffusion-Preview's 54.3% ^198^demonstrates that curriculum learning—progressing from mask-based training to edit-based training—can boost editing capability by 4.8pp over AR baselines ^198^. On the Aider multi-turn editing benchmark, Stable-DiffCoder achieves 54.9% (tries=2), slightly trailing Seed-Coder's 57.1% but comparable to Qwen3-8B's 55.6% ^10^—showing that while diffusion dominates single-turn editing, multi-turn iterative editing remains competitive.

Mercury Coder extends this editing advantage to fill-in-the-middle (FIM) tasks, achieving 84.8% on FIM benchmarks versus Flash-Lite's 60.1%—a 24.7pp advantage that represents one of the largest diffusion wins across any code evaluation ^7^. FIM tasks, which require completing code in the middle of existing programs, are structurally identical to the masked reconstruction objective used in diffusion training.

#### 9.3.2 RepoQA: Superior Length Extrapolation

RepoQA's "Searching Needle Function" task evaluates long-context code understanding across 500 problems drawn from 50 repositories ^14^. This benchmark reveals a structural advantage for diffusion models that has implications for repository-level development tools.

**Table 9.3 — Long-Context and Specialized Benchmark Performance**

| Model | Architecture | RepoQA 4K | RepoQA 64K | Degradation (8K→64K) | SWE-Bench Verified |
|-------|-------------|-----------|------------|---------------------|-------------------|
| DiffuCoder-7B-cpGRPO | Diffusion | >30% ^14^| — | Minimal | — |
| Mercury-Coder-Small | Diffusion | — | — | ~15% ^14^| — |
| Qwen3-8B | Autoregressive | — | — | ~30% ^14^| — |
| Llama-2-7B-Chat | Autoregressive | <10% ^14^| — | Severe | — |
| Gemini Diffusion | Diffusion | — | — | — | 22.9% ^3^|
| Flash-Lite | Autoregressive | — | — | — | 28.5% ^21^|
| Claude Opus 4.6 | Autoregressive | — | — | — | 80.6% ^201^|

RepoQA results show that diffusion models maintain significantly higher retrieval accuracy as context length increases. At 4K tokens input, AR model (Llama-2-7B-Chat-HF) retrieval accuracy drops below 10%, while DiffuCoder-7B-cpGRPO maintains above 30% ^14^. When extrapolating beyond the training window (8K to 64K), Mercury-Coder-Small shows only approximately 15% performance decrease, while Qwen3-8B drops by nearly 30%—twice the degradation rate ^14^. The "Beyond Autoregression" authors conclude that "diffusion LLMs remain relatively robust as context length increases, whereas the performance of AR LLMs declines rapidly" ^14^.

This advantage is hypothesized to stem from diffusion models' bidirectional attention during training. While AR models attend only to preceding tokens, diffusion models attend to all positions simultaneously during the denoising process, learning representations that are less position-dependent and more robust to context expansion. For repository-level coding tasks—where relevant functions may be defined thousands of tokens away from the insertion point—this structural advantage could prove decisive.

#### 9.3.3 SWE-Bench: The Agentic Evaluation Gap

SWE-Bench Verified represents the most demanding code evaluation, requiring models to resolve real GitHub issues across diverse Python repositories. The benchmark demands multi-step reasoning: understanding issue descriptions, exploring codebase structure, locating relevant files, diagnosing root causes, and generating correct patches. Gemini Diffusion scores 22.9% versus Flash-Lite's 28.5% ^3^—a 5.6pp gap that appears concerning.

However, this comparison requires careful interpretation. The Gemini Diffusion evaluation uses "non-agentic evaluation (single turn edit only), max prompt length of 32K" ^3^. SWE-Bench leaderboards show that agentic models with iterative feedback loops vastly outperform single-turn approaches—Claude Opus 4.6 achieves 80.6% and Gemini 3.1 Pro reaches 80.8% in agentic mode ^201^, compared to mid-20s percentages for single-turn models. No diffusion model has yet been evaluated on SWE-Bench in an agentic setting. Given diffusion models' strengths in global planning and single-pass solution generation, it is plausible that iterative agentic workflows could significantly improve their SWE-Bench performance. The current gap may reflect evaluation methodology rather than fundamental capability limitations.

### 9.4 Root Cause Analysis

#### 9.4.1 The AR-Collapse Phenomenon

The NAP paper (Li et al., February 2026) provides the most compelling theoretical explanation for the benchmark-dependent performance pattern ^34^. The paper demonstrates that diffusion language models trained on standard sequential data—code corpora organized in left-to-right reading order—converge to autoregressive-like decoding patterns, a phenomenon the authors term *AR-collapse*. Even when given the freedom to generate tokens in any order, these models exhibit high ARness (Global ARness@1 ~ 0.92 for Dream models), meaning "their most confident tokens are almost always the next tokens in the sequence" ^34^.

This behavior has a critical implication: diffusion models sacrifice their parallel generation advantage in order to maintain accuracy. When forced to use true parallel decoding (low ARness), reasoning accuracy collapses. On Dream-7B evaluated on GSM8K, accuracy drops from 78.0% at 1,024 steps to 46.5% at 256 steps—a 31.5pp collapse ^34^. The authors attribute this to a dependency on "sequential stability": standard supervision creates reasoning chains where each step depends on the previous one, and when the model is forced to commit to multiple positions simultaneously, these chains break.

The NAP paper further demonstrates that this is a training data problem, not an architectural limitation. By restructuring supervision as "multiple independent reasoning trajectories"—training on data where reasoning steps are less sequentially dependent—the authors achieve a +14.4% improvement on GSM8K under parallel decoding ^34^. This data-centric solution suggests that the LiveCodeBench gap could narrow significantly with training data specifically designed to support parallel reasoning.

#### 9.4.2 Benchmark Design Bias

The AR-collapse phenomenon interacts with benchmark design to produce the selection effect observed in Section 9.1. Most code benchmarks were created before diffusion language models existed and implicitly assume left-to-right generation. HumanEval and MBPP present function signatures and ask models to complete bodies sequentially—a task structure that naturally favors AR generation ^2^. Research on PythonSaga notes that "more than 80% of the problems [in HumanEval/MBPP] are perceived as easy" and "existing benchmarks lack a comprehensive evaluation of their diversity in terms of programming concepts and difficulty level" ^37^. Easy problems do not stress non-sequential reasoning capabilities.

LiveCodeBench requires step-by-step algorithmic reasoning precisely because competitive programming problems demand sequential logical chains. The gap between diffusion and AR is largest on this benchmark because it most strongly rewards the sequential reasoning that current diffusion training does not optimize for ^2^. Conversely, CanItEdit rewards non-sequential thinking—modifying code requires understanding global context and making targeted changes, which aligns with diffusion's bidirectional attention. The one benchmark type where diffusion consistently wins is the one type that is fundamentally non-sequential.

The ARness-accuracy tradeoff creates an incentive problem. Diffusion models can achieve competitive accuracy on sequential benchmarks by mimicking AR behavior (high ARness), but this defeats the purpose of parallel generation. Conversely, forcing low-ARness parallel decoding collapses reasoning accuracy ^34^. Current benchmarks reward high-ARness behavior, creating pressure for diffusion models to become "AR models in diffusion clothing" rather than developing genuinely parallel reasoning capabilities.

#### 9.4.3 Toward Diffusion-Native Benchmarks

The benchmark selection effect implies that resolving the diffusion-vs-AR debate requires new evaluation suites designed specifically for non-sequential code tasks. Several directions are emerging. Fill-in-the-middle benchmarks, where Mercury Coder already demonstrates a 24.7pp advantage over Flash-Lite ^7^, directly measure the capability diffusion training optimizes for. Repository-level editing benchmarks that require cross-file modifications would stress-test the global context understanding where diffusion shows superior length extrapolation ^14^. Multi-turn agentic coding tasks would evaluate whether diffusion models' global planning capabilities translate to iterative software engineering workflows ^201^.

The NAP paper's data-centric approach—restructuring training data to support parallel reasoning—suggests a complementary path: benchmarks that explicitly measure performance under varying degrees of parallelism ^34^. Current benchmarks implicitly reward AR-like behavior; benchmarks that reward low-ARness parallel decoding would incentivize the development of truly parallel reasoning capabilities. CRUXEval, which tests code execution reasoning through input and output prediction tasks, already shows promise in this direction: Stable-DiffCoder outperforms Seed-Coder on Output-CoT (60.0% versus 54.8%) because "the inputs and outputs are inherently structured rather than strictly following left-to-right causal logic" ^10^. EndoCoT-style reasoning benchmarks that require explicit step-by-step reasoning chains during generation could push diffusion models to develop better internal reasoning mechanisms without sacrificing parallel decoding ^202^.

The evidence suggests a task-dependent conclusion rather than a paradigm-level verdict. Diffusion models are not universally superior or inferior to AR models; they are structurally better suited to tasks requiring global context understanding, parallel pattern completion, and non-sequential modification—precisely the tasks that dominate real-world software engineering. The competitive programming gap, while real, may narrow as training data improves and as diffusion-native reasoning techniques (EndoCoT, NAP-style data restructuring) mature. For code editing, repository-level context retrieval, and fill-in-the-middle completion, diffusion models already demonstrate decisive advantages that are unlikely to be reversed by incremental AR improvements. The question is not whether diffusion will replace AR for code, but for which code tasks—and the benchmark data provides an increasingly clear map of the dividing line.
-e 


## 10. Future Outlook and Strategic Implications

The preceding nine chapters have traced the technical foundations, institutional strategies, and competitive dynamics of diffusion language models (DLMs) for text and code generation. This final chapter synthesizes those threads into actionable conclusions. Rather than repeating findings, it extracts ten cross-cutting insights that span multiple dimensions of the analysis, identifies the structural forces that will shape the field through 2027, and offers a framework for monitoring progress.

### 10.1 Key Cross-Cutting Insights

#### 10.1.1 Code Editing as the Strategic Beachhead

The most consistent empirical finding across this report is that diffusion models do not need to surpass autoregressive (AR) models on every benchmark to achieve commercial relevance. Stable-DiffCoder achieves 60.0% on CanItEdit — an 18.8 percentage point advantage over its AR counterpart Seed-Coder (50.5%) — while maintaining competitive but not dominant scores on HumanEval (86.6% vs. 84.8%) and MBPP ^6^. This pattern is not coincidental. Code editing is an inherently non-sequential task: changing a function signature requires updating all callers simultaneously, a workflow that maps directly to diffusion's any-order generation capability ^34^. By contrast, code completion rewards left-to-right sequential reasoning, the paradigm for which AR models are optimized.

The commercial implication is specific and actionable. The diffusion opportunity lies in displacing GitHub Copilot's completion-centric paradigm with an editing-centric workflow, not in competing head-to-head on completion benchmarks. Ant Group's internal CodeFuse NES system already demonstrates this thesis at scale: its Tab-key workflow serves 20,000+ developers by rethinking the interaction model around diffusion's strengths ^196^. Organizations evaluating diffusion adoption should prioritize editing-heavy codebases — large monorepos with frequent refactoring, legacy code modernization, and multi-file update tasks — rather than measuring success on competitive programming leaderboards where AR models retain structural advantages.

#### 10.1.2 The AR-to-Diffusion Conversion Moat

A second structural insight concerns the pathway by which production diffusion models are created. Chapter 2 established that block diffusion (block size ~32) has become the pragmatic production consensus. Chapters 3 through 5 revealed that every major production model — LLaDA2.0, Stable-DiffCoder, Gemini Diffusion — was created not by training from scratch but by converting a pretrained AR model through a multi-phase conversion protocol ^203^ ^6^ ^3^. Ant Group converts its Ling-family AR models; ByteDance converts Seed-Coder; Google's Gemini Diffusion converts Gemini 2.0 Flash-Lite.

This convergence creates an underappreciated competitive dynamic summarized in Table 10.1. Organizations that have invested heavily in pretraining large AR models possess a structural advantage: their "sunk cost" in AR pretraining does not become stranded when entering the diffusion race. Instead, it transfers. The conversion process — typically involving a warmup-stable-decay (WSD) schedule over 5–50 billion tokens — preserves the semantic knowledge embedded in the AR base while reconfiguring the generation dynamics ^203^.

**Table 10.1 — The AR-to-Diffusion Conversion Moat: Organizational Advantage Analysis**

| Organization | AR Base Model | Diffusion Derivative | Conversion Cost | Structural Advantage |
|:---:|:---:|:---:|:---:|:---|
| Ant Group | Ling (various sizes) | LLaDA2.0 100B MoE | ~50B tokens WSD | Deep AR investment; toolchain moat (dFactory, dInfer, SGLang) |
| ByteDance | Seed-Coder | Stable-DiffCoder | Block diffusion CPT | End-to-end control; CanItEdit 60.0% ^6^|
| Google DeepMind | Gemini 2.0 Flash-Lite | Gemini Diffusion | Internal (undisclosed) | Largest AR model ecosystem; TPU serving infrastructure |
| Inception Labs | External (undisclosed) | Mercury family | Proprietary | First-mover commercial API; $50M funding ^204^|
| Renmin Univ. / GSAI-ML | Qwen family | LLaDA ecosystem | Open-source conversion | Open-source community; academic research pipeline |

The table reveals a two-tier structure. Incumbent organizations with deep AR investments (Ant, ByteDance, Google) gain diffusion capabilities at marginal incremental cost. Pure-play diffusion entrants (Inception Labs) must either license or independently develop base model quality, facing higher effective barriers. This dynamic reinforces concentration among existing large-model providers and makes it difficult for startups to enter the diffusion model race without partnership strategies.

#### 10.1.3 RL Is the Secret Weapon, Not Speed

The dominant public narrative around diffusion models centers on inference speed — 2,146 tok/s from ByteDance's Seed Diffusion, 1,479 tok/s from Gemini Diffusion, 1,109 tok/s from Mercury Coder ^6^ ^3^ ^205^. While these figures are real, they are not the primary driver of quality improvement. The data in Figure 10.1 tell a different story.

![Figure 10.1 — Quality Gain Decomposition: LLaDA2.0-flash Development Pipeline](/mnt/agents/output/fig_quality_gain_decomposition.png)

*Figure 10.1.* Decomposition of LiveCodeBench score improvement for LLaDA2.0-flash across development stages. The base AR model (Qwen3-30B) scores 28.5%. Post-conversion to diffusion yields only a marginal gain to 29.07%. The bulk of quality improvement — approximately 45% of the total gain from conversion to final model — comes from the final RL post-training stage (EBPO and VRPO). Data sourced from Ant Group technical report ^203^with post-training breakdown from preview versus final model comparison.

LLaDA2.0-flash-preview scored only 29.07 on LiveCodeBench immediately after conversion from the AR base. The final model reaches 42.29 — but this 45% improvement is attributable almost entirely to post-training reinforcement learning (SFT, CAP, DPO, and EBPO), not to the diffusion architecture itself ^203^. Apple's coupled-GRPO achieves a +4.4% EvalPlus gain with only 21,000 examples, demonstrating RL's extraordinary data efficiency for diffusion models ^206^. The VRPO algorithm introduced for LLaDA 1.5 outperforms DPO, IPO, and SLiC baselines, establishing that diffusion models are currently "RL-shaped" — they benefit disproportionately from RL post-training due to the inherent train-test mismatch that SFT methods cannot resolve ^207^.

The strategic implication is clear: organizations mastering RL recipes for diffusion (EBPO, coupled-GRPO, VRPO) will outperform competitors focused exclusively on architectural novelty or inference acceleration. The RL recipe may be more important than the diffusion recipe.

### 10.2 The China Open-Source Inversion

#### 10.2.1 The Pattern: Chinese Institutional Leadership

A striking geographic pattern emerges from the analysis of open-source diffusion LLMs. Table 10.2 documents the full landscape of significant diffusion model releases as of mid-2025.

**Table 10.2 — Geographic Distribution of Diffusion LLM Development: The Open-Source Inversion**

| Model / Family | Institution | Country | Open-Source? | Parameter Scale | Primary Contribution |
|:---:|:---:|:---:|:---:|:---:|:---|
| LLaDA2.0 / 2.1 | Ant Group (GSAI-ML) | China | Yes | 100B MoE | Largest open-source DLM; EBPO RL; token editing |
| Stable-DiffCoder | ByteDance Seed | China | Partial | 32B | CanItEdit 60.0%; two-stage curriculum ^6^|
| Seed Diffusion | ByteDance | China | Preview only | 7B–32B | Fastest code DLM at 2,146 tok/s ^6^|
| Dream / DreamOn | Renmin Univ. / GSAI-ML | China | Yes | 7B | Fixed-length solution; open-source ecosystem |
| DiffuLLaMA | Tsinghua SIA-Lab | China | Yes | 7B | Early AR-to-diffusion conversion baseline |
| MDLM | Various (incl. Stanford) | Mixed | Yes | Various | Foundational discrete diffusion framework |
| Gemini Diffusion | Google DeepMind | USA | No | Unknown | 1,479 tok/s production API ^3^|
| Mercury / Mercury 2 | Inception Labs | USA | No | Unknown | First commercial DLM API; 1,109 tok/s ^205^|
| DiffuCoder | Apple | USA | No | 7B | Coupled-GRPO RL innovation |

The pattern is unambiguous. Every major open-source diffusion LLM originates from a Chinese institution: Ant Group (LLaDA family), ByteDance (Seed Diffusion, Stable-DiffCoder), Renmin University / GSAI-ML (Dream, DreamOn), and Tsinghua University (DiffuLLaMA). American contributions — Google DeepMind's Gemini Diffusion, Inception Labs' Mercury, and Apple's DiffuCoder — are exclusively closed-source or limited release ^208^ ^209^. This represents the precise inverse of the AR landscape, where US-based organizations (OpenAI, Anthropic, Meta, Google) dominate open-weight releases and Chinese institutions primarily produce closed models.

The reasons for this inversion merit consideration. Chinese institutional researchers appear to have concentrated effort around the LLaDA framework, creating a cumulative open-source ecosystem (dFactory for training, dInfer for inference, SGLang integration for serving) that lowers barriers to entry and attracts further contributions ^203^. The US approach, by contrast, has channeled diffusion development through commercial entities (Inception Labs, Google product teams) with corresponding IP protection.

#### 10.2.2 Implications for Western Organizations

Western organizations that wish to build products atop diffusion models face a strategic choice. They can adopt closed-source APIs (Mercury, Gemini Diffusion) and accept the cost, latency, and dependency risks that accompany any closed API strategy. Alternatively, they can build on open Chinese models (LLaDA2.0, Dream) and accept the geopolitical, compliance, and supply-chain risks associated with Chinese-origin model weights. There is currently no major open-source diffusion LLM from a Western institution — a gap that creates competitive pressure but also opportunity for whichever US or European organization first releases a competitive open-weight diffusion model.

#### 10.2.3 The DeepSeek Parallel

The comparison to DeepSeek's disruption of the AR landscape is apt. DeepSeek demonstrated that a Chinese research organization could release an open-weight model competitive with leading Western closed models, forcing price reductions across the industry and challenging assumptions about the relationship between capital investment and model quality. The diffusion ecosystem may follow a similar trajectory: if LLaDA2.0 or a successor achieves parity with closed-source diffusion APIs while remaining freely available, it could exert comparable price and access pressure ^210^. The key uncertainty is whether the diffusion user base will grow sufficiently to create analogous market impact.

### 10.3 Diffusion vs. Autoregressive: The Convergence Hypothesis

#### 10.3.1 A3 and the Any-Order Challenge from the AR Side

The boundary between autoregressive and diffusion paradigms is blurring from both directions. On the AR side, A3 (Any-order Any-subset Autoregressive Modeling) reformulates group prediction into a generalized autoregressive framework that preserves dependency depth while enabling any-order, any-subset generation ^211^. A3-8B outperforms state-of-the-art diffusion models (Dream 7B, DiffuLlama 7B) on question answering, commonsense reasoning, and infilling tasks despite using only 2 billion training tokens compared to 65 billion for DiffuLlama ^211^.

This result is significant because it challenges a core claimed advantage of diffusion. If AR models can achieve flexible generation order without abandoning autoregression, then "any-order capability" ceases to be a differentiator exclusive to diffusion. A3 demonstrates that the advantage may lie in training data structure and objective function design rather than in the fundamental generation mechanism.

#### 10.3.2 The "Pseudo Diffusion" Debate: Distraction or Genuine Concern?

From the diffusion side, a philosophical critique has gained traction: are masked diffusion language models merely "BERT with extra steps"? The observation has empirical grounding — modern DLMs train a model to recover texts with varying masking ratios (30%, 50%, 90%, 100%), which structurally resembles BERT's masked language modeling objective extended to higher masking rates ^212^. The SEDD framework (score entropy discrete diffusion) and the iterative refinement process distinguish DLMs from BERT in principle, but the architectural similarity invites scrutiny ^206^.

This debate, however, is largely a distraction from practical evaluation. The relevant question is not whether a model is "truly" diffusion but whether it achieves the practical benefits sought from diffusion: parallel generation, iterative refinement, any-order capability, and competitive quality. On these metrics, current production DLMs deliver demonstrable value regardless of taxonomic classification.

More concerning is the empirical finding that practical fast DLMs "frequently converge to left-to-right, autoregressive-like decoding dynamics" because training data — including chain-of-thought rationales — encodes strong sequential dependencies ^34^. Li et al.'s NAP (Non-Autoregressive Parallel DLMs) approach demonstrates that restructuring training data to contain multiple independent reasoning trajectories can mitigate this AR-collapse, achieving 60.9% accuracy versus 46.5% for standard Long-CoT at 256 steps (4x parallelism) on GSM8K ^34^. This suggests the AR-collapse is a data problem rather than a fundamental architectural limitation — but it also means that diffusion's parallelism advantage is contingent on solving a difficult data engineering challenge.

#### 10.3.3 The Hybrid Future: Obsoleting the Debate

The convergence trend points toward a future in which the AR-versus-diffusion debate becomes obsolete. Multiple hybrid approaches now combine elements of both paradigms. CALM (Confident Adaptive Language Modeling) dynamically selects between AR and non-AR generation. Projected Autoregression applies diffusion-style iterative refinement within an AR backbone. TiDAR (Time-Dependent Autoregressive Refinement) interleaves autoregressive steps with parallel denoising rounds.

The theoretical analysis by Feng et al. provides a rigorous foundation for understanding why hybridization may be optimal: masked diffusion models achieve near-optimal perplexity in constant steps (efficient for fluency), but for low sequence error rate — critical for reasoning chains — required sampling steps scale linearly with sequence length, eliminating the efficiency advantage ^213^. This metric-dependent efficiency result suggests that different tasks genuinely favor different generation paradigms. A controlled experiment by Vicentino (2026) trained AR and MDLM Transformers on identical data and compute, finding that AR models produce fluent but structurally repetitive outputs (99.8% begin with the same word) while MDLM generates more diverse narratives (93.4% unique five-word openings) at the cost of occasional grammatical inconsistencies ^150^.

The convergence hypothesis, then, is not that one paradigm will defeat the other but that the boundary between them will dissolve. Future models may autoregressively plan structure and diffusively fill content, or vice versa, with the combination chosen dynamically per task. Stefano Ermon's prediction that "within a few years, all frontier models will be diffusion models" ^208^may prove accurate in spirit — not because diffusion replaces autoregression entirely, but because the distinction ceases to be meaningful.

### 10.4 Predictions for 2026–2027

#### 10.4.1 The Make-or-Break Year

Multiple signals converge on 2026 as the decisive period for diffusion LLMs. Three commercial providers now operate (Mercury, Gemini Diffusion, Seed Diffusion). Open-source models have reached 100 billion parameters (LLaDA2.0). Speed optimization research has achieved order-of-magnitude improvements (Fast-dLLM 27.6x throughput, Elastic-Cache 45.1x on long sequences) ^207^ ^214^. RL techniques are maturing rapidly (EBPO, coupled-GRPO, VRPO). The infrastructure gap — once a critical barrier — is beginning to close with SGLang integration and dedicated serving stacks.

Yet the current reality on quality leaderboards is sobering. No diffusion model ranks in the top 10 on LMSYS Chatbot Arena ^215^. Google's Gemini Diffusion matches "Gemini 2.0 Flash-Lite" — a budget-tier model, not a frontier model — and lags on GPQA Diamond (40.4% vs. 56.5%) and BIG-Bench Extra Hard (15.0% vs. 21.0%) ^3^. Mercury 2's AIME 2025 score of 91.1 and GPQA of 73.6 represent meaningful progress ^216^, but diffusion models have not yet demonstrated consistent frontier-level quality across the full benchmark suite.

The inflection point, if it occurs, will likely be sudden rather than gradual. Diffusion models benefit from compounding advantages: speed enables more inference-time compute within the same latency budget; RL post-training shows 33x data robustness advantages over SFT; and inference acceleration research is advancing faster than model research ^207^. The combined effect could produce a qualitative leap when these threads converge.

#### 10.4.2 Critical Monitoring: IDE Integration Over Benchmarks

The most important leading indicator for diffusion commercialization is not benchmark improvement but IDE (Integrated Development Environment) integration. GitHub Copilot's dominance derives not from superior model quality but from seamless embedding in VS Code and JetBrains — real-time suggestion display, ghost text, and frictionless acceptance. Diffusion models currently lack equivalent integration. The exception — Ant Group's NES system with its Tab-key workflow serving 20,000 developers — demonstrates that proper UX integration can drive real-world adoption even without benchmark supremacy ^196^.

The critical monitoring framework should prioritize three metrics over raw benchmark scores: (1) the number of developers actively using diffusion-based coding tools, (2) the depth of IDE integration (native plugin vs. API wrapper), and (3) enterprise case studies demonstrating measurable productivity gains. Continue.dev's integration of Mercury for Next-Edit ^217^and Buildglare's use of Mercury Coder for real-time editing ^196^are early signals, but the scale remains orders of magnitude below Copilot's reported millions of users.

#### 10.4.3 Predictions and Monitoring Indicators

Table 10.3 synthesizes the key predictions and monitoring indicators for 2026–2027, organized by confidence level and timeframe.

**Table 10.3 — Key Predictions and Monitoring Indicators for 2026–2027**

| Prediction | Confidence | Timeframe | Critical Indicator | If True / If False |
|:---|:---:|:---:|:---|:---|
| Diffusion achieves >5% developer tool market share | Medium | H2 2026 | IDE plugin download counts; active developer surveys | Accelerates ecosystem investment; risk of marginalization |
| 100B+ open-source DLM matches mid-tier AR API (GPT-4o-mini class) | High | H1 2026 | LMSYS Elo ranking; MMLU-Pro score >75 | Validates open-source Chinese model pathway; reinforces closed-source moat |
| Block diffusion (size ~32) remains dominant production architecture | High | Through 2027 | Adoption of dynamic block scheduling (DSB) | Pragmatic compromise persists; fully parallel methods breakthrough |
| RL-only training (no pre-training) becomes viable for specialized domains | Medium | 2027 | Domain-specific DLM trained purely via RL on <1B tokens | Dramatically lowers entry barrier; confirms pre-training moat |
| Multimodal unified diffusion (text + vision in one backbone) ships commercially | Medium | H2 2026 | LLaDA2.0-Uni or MMaDA successor in production API | Eliminates hybrid AR+diffusion pipelines; text-only remains standard |
| Diffusion matches AR on competitive programming (Codeforces div.2) | Low | 2027 | LiveCodeBench >50% at scale comparable to AR frontier | Paradigm tipping point; confirms reasoning limitation |
| Variable-length generation solved for general text (not just code) | Medium | H1 2027 | DreamOn-style [expand]/[delete] generalized | Removes major UX barrier; fixed-length remains code-specific |
| Process reward models (PRMs) for code diffusion achieve >20% gain over outcome-only RL | Medium | 2027 | PRM-trained DLM on SWE-Bench verified | Enables systematic debugging capability; RL improvement plateaus |
| Western institution releases competitive open-weight DLM | Low | 2027 | Open-source release >30B parameters, HumanEval >85% | Rebalances geographic distribution; Chinese open-source dominance continues |
| Inference cost for diffusion falls below AR equivalent at equal quality | High | H1 2026 | Price per 1M tokens on OpenRouter/Vercel below GPT-4o-mini | Cost-driven adoption accelerates; serving complexity remains barrier |

The predictions cluster into three thematic groups. High-confidence predictions concern architectural and economic fundamentals: block diffusion will persist because it is the pragmatic compromise between parallelism and infrastructure compatibility; inference costs will fall because acceleration research is outpacing model research; and open-source models at the 100B scale will match mid-tier AR APIs because the conversion pipeline has been validated. Medium-confidence predictions concern adoption dynamics and research advances: developer market share growth, multimodal unification, and variable-length generation generalization are plausible but depend on engineering execution rather than fundamental breakthroughs. Low-confidence predictions concern paradigm-level shifts: matching AR on competitive programming, which requires solving the sequential reasoning limitation that Feng et al. identified as theoretically costly ^213^, and Western open-source releases, which depend on strategic decisions by organizations that have so far chosen closed approaches.

The final prediction in Table 10.3 — that inference cost for diffusion falls below AR equivalent at equal quality by H1 2026 — is perhaps the most consequential for near-term adoption. Mercury 2 is already priced at $0.25 per million input tokens, dramatically below frontier AR models ^216^. If this price advantage holds at quality parity, cost-driven adoption will accelerate regardless of whether diffusion achieves top-tier benchmark scores. The monitoring framework should therefore track API pricing on aggregator platforms (OpenRouter, Vercel) as closely as benchmark leaderboards.

The diffusion LLM field stands at an inflection point characteristic of emerging paradigms: the foundational research has demonstrated feasibility, the first commercial deployments are live, and the open-source ecosystem is growing rapidly. Whether 2026 marks the transition from experimental to mainstream depends less on any single breakthrough than on the compounding effect of simultaneous progress across RL optimization, inference acceleration, and — most critically — developer experience integration. Organizations and investors should weight the latter more heavily than benchmark improvements when evaluating diffusion's trajectory.
