## Dimension 04: Ant Group Ecosystem — CodeFuse, NES, Ling, and Inclusion AI (Deep Dive)

---

## 1. CodeFuse NES Technical Details — Dual-Model Architecture for Next Edit Suggestion

### Key Findings

- **NES (Next Edit Suggestion)** is an instruction-free, low-latency code editing framework that leverages learned historical editing trajectories to implicitly capture developers' goals and coding habits, eliminating the need for explicit natural language instructions. [^110^]

- **Dual-Model Architecture**: NES features two specialized models working in tandem:
  - **NES-Location Model**: Predicts the next most probable edit location using the developer's historical editing patterns, achieving **75.6% accuracy** in predicting edit placement.
  - **NES-Edit Model**: Generates personalized and precise code modifications for the current edit location, delivering a **27.7% Exact Match Rate (EMR)** and **91.36% Edit Similarity (ES)**. [^102^] [^297^]

- **Training Pipeline**: Two-stage methodology:
  1. **Supervised Fine-Tuning (SFT)** on large-scale historical editing datasets establishes foundational capabilities.
  2. **Reinforcement Learning with DAPO** (Dynamic sAmpling Policy Optimization) refines both models using task-specific reward functions. The Location Model uses binary rewards (+1.0 for exact match, -1.0 otherwise), while the Edit Model uses hierarchical rewards (+1.0 exact match, +0.5xEdit Similarity for ES > 0.5, -1.0 otherwise). [^303^]

- **Inference Optimizations**: NES leverages Prefix Caching (PC) and Speculative Decoding (SD) to achieve substantial inference speedups. The final model (Qwen3-4B+SFT+DAPO) is optimized for **Nvidia L20 GPUs**, delivering **under 250ms** inference time. [^297^]

- **Incremental Difference Detection**: The framework uses a real-time code change monitor based on incremental computation. It operates in two steps: (1) incremental difference calculation that narrows computational scope from entire files to actively modified code segments, and (2) instant difference merging that consolidates fragmented editing histories into cohesive recommendations. The custom NES diff format enriches standard diff by prefixing every line with absolute line numbers for clearer context. [^102^]

- **Dataset Construction**: Training instances are created by converting raw editing trajectories into structured tuples containing pre-edit code state, historical editing trajectory, and ground-truth edit. An LLM-based relevance filter classifies edits as "modification" (logically connected to history) or "preservation" (uncorrelated), transforming irrelevant sequences into negative samples that teach the model when NOT to suggest a change. [^303^]

- **Open-sourced Datasets**: SFT and DAPO datasets are publicly available at https://huggingface.co/datasets/codefuse-ai/CodeFuse_codeedit, with remaining data to be released incrementally. These datasets "have been instrumental in achieving accuracy and similarity scores that are dozens of times higher on code editing tasks across a variety of models." [^297^]

- **Conference Publication**: The NES paper was accepted at the **34th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (FSE Companion '26)**, Montreal, July 2026. [^102^]

---

## 2. NES Deployment at Scale — 20,000+ Developers

### Key Findings

- **Deployment Scale**: NES is deployed at Ant Group serving **over 20,000 developers** through a seamless Tab-key interaction (Tab→Tab→Tab), achieving effective acceptance rates of **51.55% for location predictions** and **43.44% for edits**. [^110^]

- **Real-World Impact**: The system handles thousands of daily code changes. Developers can perform complex tasks like refactoring or API call chains through simple Tab key sequences, "eliminating manual instructions, reducing cognitive load, and minimizing context-switching." [^117^]

- **Benchmark Evaluation**: NES was evaluated on a benchmark of 4 programming languages (Java, Python, TypeScript, TypeScriptReact) from 1000 open-source and industrial projects. The test dataset comprises 1,000 samples per language divided into Modification (500 samples where subsequent edits occur in different locations) and Preservation (500 samples where edits remain in same location or terminate) scenarios. [^297^]

- **Inference Performance**: Under 4-card H20 conditions, the system achieves inference speeds that enable real-time responsiveness. The SFT+DAPO training approach significantly improves performance on both Qwen2.5-Coder-7B and Seed-Coder-8B-Instruct baseline models. [^297^]

---

## 3. DAPO (Dynamic sAmpling Policy Optimization) — RL Method for NES

### Key Findings

- **Origin**: DAPO was originally developed by the DAPO team (Yu et al., 2025) as an open-source large-scale RL system for LLM reasoning enhancement, achieving **50 points on AIME 2024** with Qwen2.5-32B using only 50% of the training steps required by DeepSeek-R1-Zero-Qwen-32B. [^389^]

- **Core Algorithm**: DAPO is a variant of GRPO (Group Relative Policy Optimization) that introduces four key techniques to address problems in standard GRPO: entropy collapse, reward noise, and training instability. [^389^]

- **Four Key Techniques**:
  1. **Clip-Higher**: Increases the upper clipping limit (epsilon_high from 0.2 to 0.28) to promote diversity and avoid entropy collapse. This allows the model to better explore high-entropy, low-probability tokens that may be essential for reasoning. [^389^]
  2. **Dynamic Sampling**: Filters out prompts with accuracy equal to 0 or 1, ensuring each batch contains samples with effective gradients. If initial sampling produces all-correct or all-incorrect outputs, more samples are drawn until diversity is achieved. [^387^]
  3. **Token-Level Policy Gradient Loss**: Unlike GRPO which computes loss at the sample level (averaging within each response), DAPO operates at the token level, weighting longer sequences more heavily. This is "super key for Long-CoT" scenarios. [^308^]
  4. **Overlong Reward Shaping**: Uses soft punishment for longer responses with an expected maximum length of 16,384 tokens and additional 4,096 tokens as soft punish cache. Reduces reward noise and stabilizes training. [^389^]

- **DAPO in NES**: For the NES code editing application, DAPO is used with hierarchical reward functions. The DAPO-trained model demonstrates improved similarity scores for modification tasks and "better aligns with the high-frequency practices observed in real-world development." [^297^]

- **Open Source**: The DAPO system is fully open-sourced at https://dapo-sia.github.io/, including training code and datasets. [^136^]

---

## 4. Ling Model Family Evolution

### Key Findings

- **Brand Name**: "Ling" is also known as "BaiLing" (百灵) in Chinese — Ant Group's AI model family. [^348^]

- **Ling-Plus (April 2025)**: First public release — a 293B sparse MoE model marking Ant Group's entry into the open foundation model race. [^306^]

- **Ling 1.5 (July 2025)**: Significant update with improved capabilities. [^306^]

- **Ling 2.0 / Ring 2.0 (September/October 2025)**: Major release with three model sizes under unified MoE architecture, all guided by Ling Scaling Laws:
  - **Ling-mini-2.0**: 16B total / 1.4B active parameters
  - **Ling-flash-2.0**: ~100B total / 6.1B active parameters
  - **Ling-1T**: 1 trillion total / ~50B active parameters (1/32 MoE activation ratio) [^304^] [^452^]

- **Ling 2.0 Technical Innovations**:
  - **Ling Scaling Law**: Purpose-built empirical scaling framework using "Ling Wind Tunnel" — small MoE runs fitted to power laws to predict loss and optimal architecture before committing GPUs to 1T scale. Discovered 1/32 activation ratio as optimal. [^452^]
  - **Architecture**: Each MoE layer has 256 routed experts + 1 shared expert. Router picks 8 routed experts per token (~3.5% activation). Aux-loss-free routing with sigmoid scoring. MTP (Multi-Token Prediction) layers for compositional reasoning. QK Normalization for stable convergence. [^451^] [^452^]
  - **Pre-training**: 20+ trillion tokens, with reasoning-heavy sources gradually increasing to ~50% of corpus. 40%+ dedicated to chain-of-thought data in final phase. [^446^]
  - **Post-training**: Evo-CoT (Evolutionary Chain-of-Thought) + LPO (Linguistic-Unit Policy Optimization) for sentence-level alignment. [^302^]
  - **FP8 Training**: Largest known FP8-trained foundation model with <0.1% loss deviation, achieving 40%+ end-to-end training acceleration. [^451^]

- **Ling-2.5-1T / Ring-2.5-1T (February 2026)**:
  - Ling-2.5-1T: 1T total / 63B active parameters, supports 1 million token context via YaRN extension from native 256K. Hybrid linear attention architecture (MLA + Lightning Linear). Agentic RL training. On AIME 2026, matches frontier thinking models using ~5,890 tokens vs. typical 15k-23k. [^341^] [^372^]
  - Ring-2.5-1T: World's first hybrid linear-architecture thinking model. 1:7 ratio fusion of MLA and Lightning Linear attention. Achieves IMO 2025 Gold Medal (35/42) and CMO 2025 (105/126, surpassing China's national team cutoff). 10x memory access reduction and 3x throughput improvement in 32K+ long text generation. [^341^]

- **Ling-2.6-flash (April 2026)**:
  - Released April 22, 2026 after anonymous testing as "Elephant Alpha" on OpenRouter (trending #1, 100B+ daily token calls).
  - 104B total / 7.4B active parameters. 262K context window, 32K max output.
  - Hybrid linear MoE architecture. 340 tokens/sec inference on 4x H20.
  - SOTA on BFCL-V4, TAU2-bench, SWE-bench Verified, Claw-Eval, PinchBench for its size class.
  - Pricing: $0.10/1M input tokens, $0.30/1M output tokens.
  - 86% reduction in inference cost vs. comparable models (15M output tokens vs. 110M+ for Nemotron-3-Super on same tasks). [^437^] [^440^] [^442^]

- **Timeline Summary**:

| Date | Milestone |
|------|-----------|
| Apr 2025 | Ling-Plus (293B sparse MoE) — first public release |
| Jul 2025 | Ling 1.5 update |
| Sep-Oct 2025 | Ling/Ring 2.0 series (mini, flash, 1T) + Ring-1T-preview |
| Oct 2025 | Ling-1T open-sourced under MIT license |
| Feb 2026 | Ling-2.5-1T, Ring-2.5-1T, Ming-Flash-Omni-2.0 |
| Apr 2026 | Ling-2.6-flash (104B/7.4B) released |

---

## 5. Ring Reasoning Models — IMO Gold Medal Achievements

### Key Findings

- **Ring-1T (October 2025)**: The **world's first open-source trillion-parameter reasoning model**. Built on Ling 2.0 architecture, trained from Ling-1T-base. Three key innovations for trillion-scale RL:
  - **IcePop**: Token-level discrepancy masking and clipping to eliminate train-inference misalignment
  - **C3PO++**: Budget-controlled rollout scheduling for efficient long rollout processing
  - **ASystem**: High-performance RL framework with SingleController+SPMD for fully asynchronous operations
  - Achieved: AIME-2025: 93.4, HMMT-2025: 86.72, CodeForces: 2088, ARC-AGI-1: 55.94
  - **IMO-2025: Silver Medal level** (4 problems solved + partial proof of Problem 2 in single submission, without code generation or external solvers). [^446^] [^455^]

- **Ring-2.5-1T (February 2026)**: World's first hybrid linear-architecture thinking model.
  - **1:7 ratio fusion of MLA and Lightning Linear attention** — in long text generation above 32K, memory access is reduced by 10x and throughput improved by 3x vs. previous generation.
  - **IMO 2025: 35/42 (Gold Medal standard)**
  - **CMO 2025: 105/126 (surpassing China's national team cutoff)**
  - Surpassed all comparison models including DeepSeek-v3.2-Thinking, Kimi-K2.5-Thinking, GPT-5.2-thinking-high in mathematical competition reasoning benchmarks (IMOAnswerBench, HMMT-25) and LiveCodeBench-v6. [^341^] [^370^] [^460^]

- **Ring-1T's Significance**: As noted by VentureBeat, "Ring-1T aims to compete with other reasoning models like GPT-5 and the o-series from OpenAI, as well as Google's Gemini 2.5." The paper on Ring-1T represents "a significant milestone in democratizing large-scale reasoning intelligence." [^450^]

---

## 6. Inclusion AI Organization — Structure, Mission, Open-Source Strategy

### Key Findings

- **InclusionAI** is Ant Group's comprehensive open-source AI organization/technological ecosystem. Its name reflects the company's philosophy that "AGI should be a public good — a shared milestone for humanity's intelligent future." [^299^] [^348^]

- **He Zhengyu (CTO of Ant Group)**: "At Ant Group, we believe Artificial General Intelligence (AGI) should be a public good — a shared milestone for humanity's intelligent future. We are dedicated to building practical, inclusive AGI services that benefit everyone, which requires constantly pushing technology forward." [^348^]

- **Three Model Families**:
  - **Ling** (灵): Efficiency-focused sparse MoE language models
  - **Ring**: Advanced "thinking" models with explicit chain-of-thought pathways
  - **Ming** (明): Native omnimodal systems for text, image, audio, and video [^348^]

- **Philosophy and Strategy**: As noted by Nathan Lambert in his Interconnects interview, InclusionAI's messaging is "surprisingly rare in the intense geopolitical era of AI — saying AI is shared for humanity." The organization recognizes that "Western companies likely won't pay for their services, so having open models is their only open door to meaningful adoption and influence." [^306^]

- **Open Source Commitment**:
  - All models released under **MIT license**
  - Model weights available on Hugging Face and ModelScope
  - Full deployment support via vLLM and SGLang
  - APIs available through third-party providers with OpenAI-compatible format
  - Training data, code, and intermediate checkpoints released for embedding models [^299^]

- **Key Technical Leads** (from NES paper, Ling 2.0 paper, F2LLM-v2 paper): Peng Di, Changxin Tian, Ziyin Zhang, Hang Yu, Siyang Xiao, Xianying Zhu, Junhong Xie, Dajun Chen, Wei Jiang, Yong Li — all affiliated with Ant Group (several with dual affiliations to UNSW Sydney, Shanghai Jiao Tong University). [^102^] [^330^]

- **Coverage**: InclusionAI spans foundational models, multimodal intelligence, reasoning, novel architectures, and embodied AI (through subsidiary Robbyant). [^377^]

---

## 7. Ming-Omni Multimodal Architecture — Diffusion for Image Generation

### Key Findings

- **Ming-Omni**: A unified multimodal model capable of processing images, audio, video, and text, while also generating speech and images. It is "the first open-source model we are aware of to match GPT-4o in modality support." [^91^]

- **Architecture Overview**: Ming-Omni extracts visual and audio tokens with dedicated encoders, combines them with text tokens, processes through **Ling (MoE architecture with modality-specific routers)**, generates speech through an audio decoder, and enables image generation via a **Diffusion Transformer (DiT) model**. [^91^]

- **Modality-Specific Routers**: Each input token is tagged by its source modality; the router computes a distribution over experts specific to the modality. This enables specialization (experts learn distinct functions for different modalities) and efficient fusion within a unified framework. [^443^]

- **Image Generation — Ming-Lite-Uni Integration**:
  - Uses a **Diffusion Transformer (DiT)** architecture
  - **Multi-scale Learnable Tokens**: The LLM generates aligned latent tokens at multiple spatial resolutions capturing features at global, mid-level, and fine granularity
  - **Alignment Loss**: MSE between intermediate DiT hidden states and LLM's final semantic representations ensures semantic consistency
  - **Multi-scale Representation Alignment**: Bridges the perceptual LLM and generation components through feature-level alignment
  - Tasks: Text-to-image, style transfer, high-fidelity editing
  - FID reaches SOTA on image generation benchmarks [^99^] [^443^]

- **Speech Generation**:
  - Autoregressive audio decoder (similar to CosyVoice) generates discrete audio tokens
  - Byte Pair Encoding (BPE) applied to discrete audio tokens, compressing sequence length by ~36% (from 50Hz to ~32Hz)
  - Two-stage training: first for understanding, second for generation with frozen MLLM [^99^]

- **Training Pipeline**: Two distinct phases:
  1. **Perception Training**: Pre-training, instruction tuning, and alignment tuning following M2-omni pipeline. Three sub-stages progressively incorporate additional tasks.
  2. **Generation Training**: Trains audio decoder and DiT module in parallel while the perceptual MLLM is frozen. For image generation, only the connector, multiscale learnable queries, and DiT blocks are trained. [^96^]

- **Ming-Flash-Omni-2.0 (February 2026)**: Next-generation omnimodal model and "the industry's first model to unify speech, audio, and music within a single architecture." [^341^]
  - **Unified Acoustic Synthesis**: Integrates Speech, Audio, and Music in a single channel using Continuous Autoregression + DiT head. Enables zero-shot voice cloning and nuanced attribute control (emotion, timbre, ambient atmosphere).
  - **12.5Hz Ultra-Low Frame Rate Continuous Speech Tokenizer**: Self-developed, achieves high-fidelity reconstruction of audio/music signals in unified latent space.
  - **Patch-by-Patch Compression**: 4-frame compression strategy reduces generation sequence length, alleviates exposure bias in long audio generation.
  - **3.1Hz Inference Frame Rate**: Industry-low, enabling real-time generation speed.
  - **Image Generation**: Native multi-task architecture unifying segmentation, generation, and editing with spatiotemporal semantic decoupling. [^445^] [^441^]

---

## 8. CodeFuse Survey Paper — 900+ Works Surveyed

### Key Findings

- **"Unifying the Perspectives of NLP and Software Engineering: A Survey on Language Models for Code"** (arXiv:2311.07989, November 2023) — a comprehensive survey from Ant Group researchers. [^451^]

- **Scope**: Covers **70+ models, 40+ evaluation tasks, 180+ datasets, and 900+ related works**. Unlike previous surveys, it integrates software engineering (SE) with natural language processing (NLP) by discussing both perspectives: SE applies language models for development automation, while NLP adopts SE tasks for language model evaluation. [^451^]

- **Key Contributions**:
  - Breaks down code processing models into general language models (GPT family) and specialized models pretrained on code with tailored objectives
  - Discusses the historical transition from statistical models and RNNs to pretrained Transformers and LLMs
  - Goes beyond programming to review LLMs' application in requirement engineering, testing, deployment, and operations
  - Identifies key challenges and potential future directions
  - Kept open and updated on GitHub at https://github.com/codefuse-ai/Awesome-Code-LLM [^451^]

- **CodeFuse-13B**: An earlier (2023) open-sourced 13B parameter multilingual Code LLM from Ant Group specifically designed for English and Chinese prompts, supporting 40+ programming languages. Achieved HumanEval pass@1 score of 37.10%. Deployed to 5,000+ engineers at Ant Group via IDE plugins (VSCode, JetBrains, Ant CloudIDE) and web-based chat. Published at ICSE-SEIP '24. [^461^] [^473^]

- **CodeFuse Ecosystem Products** (from codefuse.ai):
  - **CodeFuse-MFTCoder**: Multi-task fine-tuning framework
  - **DevOps-ChatBot**: AI assistant for software development lifecycle
  - **DevOps-Eval**: Evaluation suite for DevOps foundation models
  - **DevOps-Model**: Chinese DevOps large language models
  - **CodeFuse-Query**: Static code analysis platform (10B+ lines daily)
  - **ModelCache**: Semantic cache for LLMs
  - **TestAgent**: Open-source testing domain LLM
  - **CodeFuse IDE**: AI-integrated development environment based on OpenSumi [^467^]

---

## 9. F2LLM-v2 Embeddings — 200+ Languages, MTEB Benchmarks

### Key Findings

- **F2LLM-v2**: A new family of general-purpose, multilingual embedding models in 8 sizes ranging from 80M to 14B parameters. Released March 2026 by Ant Group and Shanghai Jiao Tong University. [^330^]

- **Training**: Trained on a curated composite of **60 million publicly available high-quality data samples**. Supports **more than 200 languages**, with emphasis on previously underserved mid- and low-resource languages. [^330^]

- **Architecture**: Based on Qwen3 Transformer decoder architecture using EOS token as sequence representation. [^337^]

- **Key Techniques**:
  - **Two-stage LLM-based embedding training pipeline**
  - **Matryoshka Representation Learning (MRL)**: Enables useful versions at different scales
  - **Model Pruning**: Removes less important network parts
  - **Knowledge Distillation**: Transfers capabilities from larger to smaller models
  - Results in models "far more efficient than previous LLM-based embedding models while retaining competitive performances" [^330^]

- **Benchmark Results**: **F2LLM-v2-14B ranks first on 11 MTEB benchmarks**. Smaller models also set new SOTA for resource-constrained applications. [^330^]

- **Open Release**: All models, data, code, and intermediate checkpoints fully released — promoting "global AI equity" and addressing both "Linguistic Inclusivity" (supporting 200+ languages) and "Computational Inclusivity" (making models run on phones without sacrificing accuracy). [^350^]

- **Comparison with Google's Gemini Embedding 2**: F2LLM-v2 focuses on efficient text representation across 200+ languages; Gemini Embedding 2 delivers unified full-modal space (text, image, video, audio, PDF). F2LLM-v2's key differentiator is its comprehensive open release and emphasis on low-resource languages. [^337^]

---

## 10. Robbyant Embodied AI — LingBot Model Stack

### Key Findings

- **Robbyant** is an embodied intelligence company within Ant Group, dedicated to advancing embodied intelligence through cutting-edge software and hardware technologies. CEO: Zhu Xing. [^377^]

- **Mission**: Creating "robotic companions and caregivers that truly understand and enhance people's everyday lives" across elderly care, medical assistance, and household tasks. [^377^]

- **Complete LingBot Stack** (5 models as of April 2026):

| Model | Function | Release |
|-------|----------|---------|
| **LingBot-Depth** | High-precision spatial perception / depth sensing | Jan 2026 |
| **LingBot-VLA** | Vision-Language-Action "universal brain" for robots | Jan 2026 |
| **LingBot-World** | World model for millisecond-level real-time interaction | Jan 2026 |
| **LingBot-VA** | Auto-regressive video-action model for robot control | Early 2026 |
| **LingBot-Map** | Streaming 3D reconstruction for real-time spatial understanding | Apr 2026 |

- **LingBot-VLA**: Vision-Language-Action model serving as a "universal brain" for real-world robotics.
  - Pre-trained on **20,000+ hours** of large-scale real-world interaction data
  - Covers **9 mainstream dual-arm robot configurations** (AgileX, Galaxea R1Pro, RILite, AgiBot G1)
  - Cross-morphology transfer: single model deployable across single-arm, dual-arm, and humanoid platforms
  - Successfully adapted to robots from Galaxea Dynamics and AgileX Robotics
  - **1.5x to 2.8x faster training** vs. frameworks like StarVLA and OpenPI
  - Sets new record on GM-100 benchmark (100 real-world tasks from Shanghai Jiao Tong University)
  - Includes complete production-ready codebase with data processing, fine-tuning, and automated evaluation tools
  - Open-sourced with model weights, code, and tech report [^377^] [^367^]

- **LingBot-Depth**: High-precision spatial perception model using Masked Depth Modeling (MDM).
  - Analyzes RGB images to reconstruct missing depth information
  - **70%+ relative error reduction** in indoor scenes vs. competing models
  - **47% RMSE reduction** on sparse Structure-from-Motion tasks
  - Trained on 2 million curated RGB-depth pairs from Orbbec Gemini 330 cameras
  - Strategic partnership with Orbbec for integration into next-generation depth cameras [^447^] [^448^]

- **LingBot-World**: World model for environmental simulation.
  - Hybrid data acquisition: large-scale curated web videos + Unreal Engine synthetic data
  - Millisecond-level real-time interaction
  - Zero-shot generalization: single real-world image → interactive video stream
  - Enables AI agents to "imagine" the physical world for trial-and-error learning [^382^] [^383^]

- **LingBot-Map**: Streaming 3D reconstruction model (April 2026).
  - Pure auto-regressive modeling built on Geometric Context Transformer
  - **Geometric Context Attention (GCA)** mechanism for efficient cross-frame geometric information
  - Oxford Spires: 6.42m Absolute Trajectory Error (~2.8x improvement over previous best)
  - ETH3D: 98.98 F1 score (21+ percentage points ahead of runner-up)
  - ~20 FPS inference, stable on sequences exceeding 10,000 frames [^443^] [^444^]

- **Strategic Significance**: Robbyant's "Evolution of Embodied AI Week" (January 2026) saw the release of three models in one week, demonstrating rapid iteration. The full stack represents "an important extension of Ant Group's AGI strategy from the digital realm to physical perception." [^382^]

---

## Major Players & Sources

| Entity | Role/Relevance |
|--------|---------------|
| **Ant Group** | Parent company; Alibaba affiliate; owner of Alipay. Funds and operates all AI initiatives. |
| **InclusionAI** | Ant Group's open-source AI organization. Hosts Ling, Ring, Ming model families. |
| **CodeFuse Team** | Ant Group's code intelligence initiative. Developed NES, CodeFuse-13B, MFTCoder, CodeFuse IDE, DevOps tools. |
| **Robbyant** | Embodied AI subsidiary within Ant Group. CEO: Zhu Xing. Developed LingBot stack (5 models). |
| **Peng Di** | Lead researcher (Ant Group / UNSW Sydney). Key author on NES and F2LLM-v2 papers. |
| **He Zhengyu** | CTO of Ant Group. Articulates AGI-as-public-good vision. |
| **Ziqi Liu / Chen Liang** | Technical leads on Ling scaling laws and FP8 training (per Interconnects interview). |
| **DAPO Team (Yu et al.)** | Developed DAPO RL algorithm used for NES training. Independent but collaborative. |
| **Orbbec** | Strategic partner for LingBot-Depth depth camera integration. |

---

## Trends & Signals

- **Aggressive Open-Source Strategy**: Ant Group releases all frontier models under MIT license, including complete trillion-parameter model families — a clear strategic choice to gain global influence and adoption. [^299^] [^348^]

- **Rapid Iteration Cadence**: From Ling-Plus (April 2025) to Ling-2.6-flash (April 2026) — 6 major iterations in 12 months, comparable to DeepSeek's pace. [^306^]

- **MoE-First Architecture**: All models use Mixture-of-Experts with carefully tuned sparsity ratios (1/32), guided by empirical scaling laws. Efficiency gains of 7x+ over dense baselines. [^452^]

- **FP8 Training Leadership**: Ling-1T is the largest known FP8-trained model, achieving 40%+ training acceleration. Ant Group optimized quantization/dequantization bottlenecks specifically for MoE layers. [^451^] [^306^]

- **From Digital to Physical AGI**: Robbyant's embodied AI stack represents Ant Group's deliberate expansion from digital AI (code, language) to physical-world intelligence (robotics, 3D perception). [^382^]

- **Token Efficiency as Competitive Advantage**: Ling-2.6-flash's positioning emphasizes completing tasks with dramatically fewer tokens than competitors (15M vs. 110M+), directly translating to cost savings. [^440^]

- **Multimodal Unification**: Ming-Flash-Omni-2.0 is the industry's first model to unify speech, audio, and music in a single architecture — pushing toward true "omni" capability. [^341^]

---

## Controversies & Conflicting Claims

- **FP8 Training Speed**: Chen Liang (Ant Group) noted in an interview that DeepSeek's FP8 recipe initially had lower MFU than expected: "We find that actually the MFU is not very high. And sometimes it's even slower than BF16 training." Ant Group had to optimize quantization/dequantization specifically for MoE layers to achieve claimed speedups. [^306^]

- **Model Naming Confusion**: "Ling" refers to both the overall model family AND the specific series of non-thinking models, while also being known as "BaiLing" (百灵) in Chinese. The Ring and Ming names also create a three-way naming system that can be confusing for external observers. [^348^]

- **IMO Self-Testing Claims**: Ring-2.5-1T's IMO 2025 gold medal (35/42) and CMO 2025 (105/126) results are described as "self-testing" — meaning they were evaluated by Ant Group rather than through official competition channels. This is standard practice in the industry but worth noting for benchmarking rigor. [^341^]

- **Elephant Alpha Stealth Launch**: Ling-2.6-flash was tested anonymously as "Elephant Alpha" on OpenRouter before official announcement, with the tagline "you can't spell Elephant without ant." While creative, this stealth marketing approach was criticized by some as potentially misleading to developers evaluating models. [^447^]

---

## Recommended Deep-Dive Areas

1. **NES Dataset Construction Methodology**: The combination of incremental difference detection, LLM-based relevance filtering, and the custom NES diff format represents a novel approach to code editing data. The open-sourced datasets and their "orders of magnitude" improvement claims warrant detailed study for replication.

2. **DAPO for Code Editing (vs. Math Reasoning)**: DAPO was originally designed for math reasoning (AIME). Its adaptation for code editing tasks in NES — with hierarchical reward functions and edit similarity metrics — represents an interesting transfer that could inform RL-for-code research more broadly.

3. **Ling Scaling Laws**: The "Ling Wind Tunnel" methodology for predicting optimal MoE architecture at trillion scale before committing GPU resources is a practical contribution to efficient large-model development. The finding that 1/32 activation ratio is optimal across scales from 16B to 1T is significant.

4. **Ming-Omni's Diffusion-LLM Bridge**: The multi-scale learnable tokens and representation alignment mechanism that connects a frozen LLM to a DiT image generator is architecturally interesting. How this preserves semantic fidelity while allowing generation flexibility could inform future multimodal systems.

5. **Robbyant's Full Embodied AI Stack**: The combination of depth perception (LingBot-Depth), world models (LingBot-World), VLA (LingBot-VLA), video-action (LingBot-VA), and 3D mapping (LingBot-Map) represents one of the most comprehensive open-source embodied AI ecosystems. The cross-morphology generalization claims warrant independent verification.

6. **FP8 Training at Trillion Scale**: Ant Group's work on optimizing FP8 specifically for MoE architectures — fusing switch gated functions with quantization, minimizing dequantization overhead — represents practical engineering knowledge that is scarce in the literature.

7. **F2LLM-v2's Low-Resource Language Support**: With 200+ languages and #1 rankings on 11 MTEB benchmarks, the model's performance on genuinely low-resource languages (beyond just standard multilingual benchmarks) would be valuable to verify independently.

---

## Sources

- [^110^] NES arXiv paper v1: https://arxiv.org/abs/2508.02473 (2025-08-04)
- [^102^] NES arXiv paper v2: https://arxiv.org/html/2508.02473v2 (2026-03-31)
- [^297^] NES arXiv paper v3: https://arxiv.org/html/2508.02473v3 (2026-04-01)
- [^117^] NES arXiv v1 (earlier title): https://arxiv.org/html/2508.02473v1 (2025-08-04)
- [^303^] ChatPaper NES analysis: https://chatpaper.com/paper/173335 (2026-01-14)
- [^308^] DAPO advances NLP presentation: https://self-supervised.cs.jhu.edu/fa2025/files/presentations/Why-RL-Sep16-AdvancesNLP.pdf
- [^389^] DAPO paper: https://arxiv.org/pdf/2503.14476 (2025-03-17)
- [^136^] DAPO arXiv v1: https://arxiv.org/html/2503.14476v1 (2025-03-17)
- [^387^] Comparative Analysis PPO/GRPO/DAPO: https://arxiv.org/html/2512.07611v1
- [^91^] Ming-Omni paper: https://arxiv.org/html/2506.09344v1 (2025-06-11)
- [^96^] Ming-Omni training: https://arxiv.org/html/2506.09344
- [^99^] Ming-Omni literature review: https://www.themoonlight.io/en/review/ming-omni-a-unified-multimodal-model-for-perception-and-generation (2025-06-13)
- [^443^] Ming-Omni Emergent Mind: https://www.emergentmind.com/topics/ming-omni (2025-06-30)
- [^439^] Ming-Omni Image Generation Design Spec (GitHub): https://github.com/sgl-project/sglang-omni/issues/304 (2026-04-16)
- [^441^] Ming GitHub: https://github.com/inclusionAI/ming (2026-02-11)
- [^445^] Ming-flash-omni-2.0 technical deep dive: https://www.cnblogs.com/rtedev/p/19614715 (2026-02-14)
- [^304^] Ling 2.0 paper: https://arxiv.org/abs/2510.22115 (2025-10-25)
- [^302^] DeepLearning.ai on Ling-1T: https://www.deeplearning.ai/the-batch/all-about-ant-groups-ling-1t-an-open-non-reasoning-model-that-outperforms-closed-competitors/ (2025-10-23)
- [^451^] SiliconFlow Ling-1T blog: https://www.siliconflow.com/blog/ling-1t-now-live-on-siliconflow-a-trillion-scale-leap-in-efficient-reasoning (2025-10-16)
- [^452^] MarkTechPost Ling 2.0: https://www.marktechpost.com/2025/10/30/ant-group-releases-ling-2-0-a-reasoning-first-moe-language-model-series-built-on-the-principle-that-each-activation-enhances-reasoning-capability/ (2025-10-30)
- [^298^] Automated Survey (Ling-1T): https://arxiv.org/html/2306.02781v4 (2026-04-09)
- [^299^] HowAIWorks Ling-1T announcement: https://howaiworks.ai/blog/ant-group-ling-1t-announcement (2025-10-13)
- [^306^] Interconnects interview with InclusionAI: https://www.interconnects.ai/p/inside-a-chinese-frontier-lab-inclusion (2025-11-12)
- [^335^] Ling AI Model Family overview: https://howaiworks.ai/models/ling (2026-05-01)
- [^341^] Ant Group Ling-2.5-1T and Ring-2.5-1T release: https://www.morningstar.com/news/business-wire/20260215551663/ (2026-02-16)
- [^342^] Yahoo Finance Ling-2.5: https://finance.yahoo.com/news/ant-group-releases-ling-2-100100729.html (2026-02-16)
- [^348^] BusinessWire Ling-1T: https://www.businesswire.com/news/home/20251009240721/en/ (2025-10-09)
- [^370^] AI Base Ring-2.5-1T: https://news.aibase.com/news/25520 (2026-02-13)
- [^371^] Open Source For You: https://www.opensourceforu.com/2026/02/ant-makes-frontier-ai-public-with-ling-2-5-1t-and-ring-2-5-1t-release/ (2026-02-17)
- [^372^] Clore.ai Ling-2.5 guide: https://docs.clore.ai/guides/language-models/ling25 (2026-02-20)
- [^446^] Ring-1T paper: https://arxiv.org/pdf/2510.18855
- [^455^] Ring-1T arXiv: https://arxiv.org/abs/2510.18855 (2025-10-21)
- [^450^] VentureBeat Ring-1T: https://venturebeat.com/ai/inside-ring-1t-ant-engineers-solve-reinforcement-learning-bottlenecks-at (2025-10-23)
- [^460^] QuantumZeitgeist Ring-1T-2.5: https://quantumzeitgeist.com/ant-group-thinking-models-ai-benchmarks/ (2026-02-13)
- [^437^] IT Home Ling-2.6-flash: https://www.ithome.com/0/941/911.htm (2026-04-22)
- [^440^] Yahoo Finance Ling-2.6: https://finance.yahoo.com/sectors/technology/articles/ant-group-unveils-ling-2-083000237.html (2026-04-22)
- [^442^] BusinessWire Ling-2.6: https://www.businesswire.com/news/home/20260422256825/en/ (2026-04-22)
- [^436^] CNBlogs Elephant Alpha: https://www.cnblogs.com/zktww/p/19907918 (2026-04-23)
- [^447^] Kilo.ai blog: https://blog.kilo.ai/p/the-elephant-is-out-of-the-bag-meet (2026-04-21)
- [^451^] ADS CodeFuse Survey: https://ui.adsabs.harvard.edu/abs/2023arXiv231107989Z/abstract
- [^461^] CodeFuse-13B paper: https://arxiv.org/html/2310.06266v2 (2024-01-10)
- [^473^] ACM CodeFuse-13B: https://dl.acm.org/doi/10.1145/3639477.3639719 (2024-05-31)
- [^467^] CodeFuse website: https://codefuse.ai/
- [^463^] AI Base CodeFuse IDE: https://www.aibase.com/news/14276 (2024-12-26)
- [^330^] F2LLM-v2 paper: https://arxiv.org/html/2603.19223v1 (2026-03-19)
- [^332^] F2LLM-v2 arXiv: https://arxiv.org/abs/2603.19223 (2026-03-19)
- [^337^] AI Rosetta comparison: https://www.airosetta.com/en/news/f2llm-v2-vs-gemini-embedding-2 (2026-03-26)
- [^350^] WisPaper F2LLM-v2: https://www.wispaper.ai/en/blog/f2llm-v2-inclusive-performant-efficient-embeddings-multilingual-world-20260322/eng
- [^377^] BusinessWire LingBot-VLA: https://www.businesswire.com/news/home/20260127455032/en/ (2026-01-28)
- [^367^] Robotics News LingBot-VLA: https://roboticsandautomationnews.com/2026/03/13/robbyant-open-sources-lingbot-vla-model-as-a-universal-brain-for-robots/99640/ (2026-03-13)
- [^382^] AFP LingBot-World: https://www.afp.com/pt/node/3812518 (2026-01-31)
- [^383^] AI Journ LingBot-World: https://aijourn.com/robbyant-open-sources-lingbot-world-a-world-model-for-millisecond-level-real-time-interaction/ (2026-01-29)
- [^444^] BusinessWire LingBot-Map: https://www.businesswire.com/news/home/20260416714351/en/ (2026-04-16)
- [^443^] AInvest LingBot-Map: https://www.ainvest.com/news/robbyant-lingbot-map-real-time-3d-perception-breakthrough-ignites-embodied-ai-curve-2604/ (2026-04-16)
- [^447^] Amsterdam Startup Map LingBot-Depth: https://startupmap.iamsterdam.com/news/feed/ant-group-s-robbyant-open-sources-lingbot-depth-ai-model-to-enhance-robot-depth-sensing-1 (2026-01-27)
- [^448^] AI Journ LingBot-Depth: https://aijourn.com/ant-group-subsidiary-robbyant-unveils-spatial-perception-ai-model-lingbot-depth/ (2026-01-27)
- [^384^] AFP LingBot-Map: https://www.afp.com/en/infos/ant-groups-robbyant-unveils-lingbot-map-streaming-3d-reconstruction-model-real-time-spatial (2026-04-16)
