## Facet: Ant Group's Diffusion Model and Code Generation Research

---

### Key Findings

#### 1. LLaDA2.0: The First 100B-Parameter Diffusion Language Model (Dec 2025)

- **LLaDA2.0** represents the first time a discrete diffusion large language model (dLLM) has been scaled to 100 billion parameters, establishing a new paradigm for frontier-scale deployment [^1^]. The model was developed by a joint team from Ant Group, Renmin University, Zhejiang University, Westlake University, and HKUST [^2^].

- **Core Innovation — Systematic AR-to-Diffusion Conversion**: Rather than training from scratch, LLaDA2.0 converts pretrained autoregressive (AR) models to diffusion models through a novel **3-phase block-level WSD (Warmup-Stable-Decay) continual pre-training paradigm** [^3^]:
  - **Phase 1 (Warmup)**: Progressively increases block size in block diffusion language models (BDLM) from 1 → 4 → 32 → 64 → 4096, gradually transforming the AR model into a full-sequence masked diffusion language model (MDLM). This smooths the transition by initially generating just a few tokens in parallel per block [^3^].
  - **Phase 2 (Stable)**: Large-scale full-sequence diffusion training at block size 4096 (equivalent to classical MDLM), processing entire sequences with bidirectional attention. This is the bulk of compute [^3^].
  - **Phase 3 (Decay)**: Reverts to compact block sizes (e.g., 32 tokens per block) to optimize for deployment efficiency with KV-cache reuse and fast variable-length generation [^3^].

- **Document-Level Attention Mask**: To prevent spurious dependencies across document boundaries in packed training sequences, LLaDA2.0 introduces a document-level attention mask that restricts self-attention within individual documents, ensuring coherent bidirectional modeling within semantic boundaries [^3^].

- **Top-k Checkpoint Merge**: A strategy to enhance performance and generalization by averaging the parameters of the best-performing checkpoints, smoothing the parameter landscape [^3^].

- **CAP (Confidence-Aware Parallel) Training**: An auxiliary confidence loss that selectively minimizes the entropy of the model's output distribution for correctly predicted tokens, compelling the model to increase certainty on its correct predictions. This improves decoding efficiency while maintaining competitive performance — LLaDA2.0-flash-CAP achieves **535 tokens/s**, 2.1x faster than comparable AR models [^4^][^5^].

- **Post-training**: After continual pre-training, the model undergoes standard post-training alignment including Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) [^6^].

#### 2. LLaDA2.0 Model Variants and Architecture

- **LLaDA2.0-mini (16B)**: 16B total parameters with ~1.44B-1.4B active parameters per token (MoE). 20 layers, 16 attention heads, context length 32,768 tokens, RoPE position embeddings, vocabulary ~157k tokens [^7^][^8^].

- **LLaDA2.0-flash (100B)**: 100B total parameters with ~6.1B active parameters per token (MoE). 32 layers, 32 heads, context length 32,768 tokens [^7^][^9^].

- Both use the **Ling 2.0** architecture as their AR base model, which is a Mixture-of-Experts (MoE) architecture with 256 routed experts plus 1 shared expert and a 1/32 activation ratio [^10^].

#### 3. LLaDA2.0 Benchmark Performance

- Evaluated on **47 benchmarks** across five dimensions: Knowledge, Reasoning, Coding, Math, and Agent & Alignment [^2^].

- **LLaDA2.0-mini** achieves 64.34 average, closely approaching Ling-mini-2.0 (65.77) and Qwen3-8B (63.42). Key coding results: HumanEval 86.59, MBPP 81.50, HumanEval+ 79.88 [^2^].

- **LLaDA2.0-flash** achieves 73.18 average, on par with Qwen3-30B-A3B-Instruct-2507 (73.60). Key coding results: HumanEval 94.51, MBPP 88.29, HumanEval+ 87.80, LiveCodeBench 42.29. Surpasses AR peers on BFCL v3 (75.43) and AIME 2025 (60.00) [^2^][^11^].

- The diffusion architecture shows particular strength in **complex structured generation tasks** like code generation and agent function calling [^11^].

#### 4. LLaDA2.1: Token Editing Innovation (Feb 2026)

- Released February 2026, LLaDA2.1 introduces **Token-to-Token (T2T) editing** alongside the conventional Mask-to-Token (M2T) scheme, enabling the model to go back and edit its own mistakes [^12^].

- **Dual-Mode Decoding**: Creates two configurable "personas":
  - **Speed Mode (S-Mode)**: Lower denoising threshold, drafts fast and fixes later via T2T editing. Achieves **892 tokens/s on HumanEval+**, **801 TPS on BigCodeBench**, **663 TPS on LiveCodeBench** for the 100B Flash model [^12^][^13^].
  - **Quality Mode (Q-Mode)**: Conservative thresholds, higher benchmark scores [^12^].

- **RL for Diffusion**: Implements the first large-scale RL framework for dLLMs using **ELBO-based Block-level Policy Optimization (EBPO)**, using the Evidence Lower Bound as a tractable proxy for policy gradients, since sequence-level log-likelihoods are intractable for diffusion models [^14^].

- **Multi-Block Editing (MBE)**: Allows the model to revisit previously decoded blocks based on newly generated context, consistently improving reasoning and coding benchmarks with modest throughput reduction [^15^].

- After quantization, LLaDA2.1-mini achieves **1,587 TPS** [^15^].

#### 5. LLaDA-MoE: Native MoE Diffusion from Scratch (Sep 2025)

- Developed jointly by Ant Group and Renmin University, **LLaDA-MoE** was the **industry's first native MoE architecture diffusion language model**, launched at the 2025 Bund Conference [^16^].

- **Key distinction**: Unlike LLaDA2.0 (which converts AR to diffusion), LLaDA-MoE was trained **from scratch** as a diffusion model with MoE architecture, trained on ~20T data [^16^].

- Architecture: 7B-A1B MoE (total 7B parameters, 1.4B activated) [^16^].

- Performance: Outperforms LLaDA1.0/1.5 and Dream-7B on code, math, and Agent tasks, approaching or surpassing Qwen2.5-3B-Instruct by activating only 1.4B parameters [^16^].

- The team used Ant's self-developed distributed framework **ATorch** for EP (Expert Parallelism) parallel and other acceleration technologies [^16^].

#### 6. CodeFuse: Code Intelligence Initiative

- **NES (Next Edit Suggestion)**: A dual-model, instruction-free, low-latency code editing framework deployed at Ant Group serving **20,000+ developers** [^17^][^18^]:
  - **NES-Location Model**: Predicts the next edit location using historical editing patterns (75.6% accuracy) [^17^].
  - **NES-Edit Model**: Generates precise code changes (27.7% Exact Match Rate) [^17^].
  - Inference time under **250ms** via Prefix Caching and Speculative Decoding [^17^].
  - Effective acceptance rates: 51.55% for location predictions, 43.44% for code edits [^17^].
  - Uses Tab-key interaction (Tab → Tab → Tab) for seamless workflow [^17^].
  - Open-sourced SFT and DAPO (Dynamic sAmpling Policy Optimization) datasets [^17^].
  - Published at FSE Companion '26 (Montreal) [^17^].

- **"Unifying the Perspectives of NLP and Software Engineering" Survey** (2023): A comprehensive survey on language models for code by the CodeFuse team, covering 70+ models, 40+ evaluation tasks, 180+ datasets, and 900+ related works. Authors include Ziyin Zhang, Chaoyu Chen, Bingchang Liu, Cong Liao, Zi Gong, Hang Yu, Jianguo Li, and Rui Wang from Ant Group and Shanghai Jiao Tong University [^19^].

- **F2LLM-v2**: A family of multilingual embedding models (8 sizes from 80M to 14B) supporting 200+ languages, ranking first on 11 MTEB benchmarks for the 14B variant [^20^].

#### 7. Inclusion AI: Ant Group's Research Division

- **inclusionAI** is Ant Group's open-source AI organization, operating under the motto "AI Built By Everyone, For Everyone" [^21^].

- Maintains **three model families** [^22^]:
  - **Ling (灵)**: General-purpose language models, the foundational "base" series
  - **Ming (明)**: Multimodal models (e.g., Ming-lite-omni, Ming-Omni) covering image, text, audio, video
  - **Ring (环)**: Reasoning/thinking models based on Ling base with RL/long-CoT training

- Three core organizational principles: (1) Ant incubated but independent, (2) MoE-first architecture, (3) Scenario anchors in financial/payment/risk control domains [^22^].

- HuggingFace profile: 28 models, 11 datasets as of early 2026 [^23^].

#### 8. Ling Language Model Family

- **Ling 2.0** (Oct 2025): Built on 1/32 activation MoE with 256 routed experts + 1 shared expert. Released in three sizes: mini (16B total/1.4B active), flash (100B total/6.1B active), and 1T (1T total/50B active). Validated with FP8 end-to-end training at 1T scale [^10^][^24^].

- **Ling-2.5-1T** (Feb 2026): 1T total parameters, 63B active, hybrid linear attention architecture (MLA + Lightning Linear Attention), 1M token context window via YaRN, trained on 29T tokens. Companion model Ring-2.5-1T achieves gold medal level at IMO 2025 and CMO 2025 [^25^].

- **Ling-2.6-flash** (Apr 2026): 104B total parameters, 7.4B active, hybrid linear MoE. Achieves 340 TPS prefill on 4x H20, ranked top tier for size class. SOTA on BFCL-V4, TAU2-bench, SWE-bench Verified, Claw-Eval, PinchBench. 86% reduction in inference token consumption vs. comparable models [^26^][^27^].

- Ling models are the **AR base models** that LLaDA2.0 converts to diffusion [^3^].

#### 9. Ming-Omni: Unified Multimodal with Diffusion Image Generation

- **Ming-Omni** is a unified multimodal model capable of processing images, text, audio, and video, while also supporting speech generation and image generation [^28^].

- **First open-source model to match GPT-4o** in modality support [^28^].

- Architecture: Uses dedicated encoders for each modality, processes through **Ling** (MoE with modality-specific routers), generates speech through an audio decoder, and enables image generation via a **diffusion model (DiT module)** [^28^][^29^].

- For image generation, uses a **lightweight bridging approach** called Ming-Lite-Uni with multi-scale learnable tokens and multi-scale representation alignment. The perceptual MLLM is kept frozen during generation training, with a dedicated diffusion model (DiT) for high-quality generation [^29^].

- Training: Two-phase approach — perception training (understanding) followed by generation training (audio decoder + image diffusion) [^29^].

- **Ming-omni-tts**: A related unified audio generation model using autoregressive architecture + Diffusion Transformer for joint speech, music, and sound effect generation [^30^].

#### 10. Robbyant: Embodied AI Division

- **Robbyant** is an embodied intelligence company within Ant Group, building foundational models for embodied AI [^31^].

- Open-sourced a suite of embodied AI models:
  - **LingBot-VLA**: Vision-language-action "universal brain" model, pre-trained on 20,000+ hours of real-world data across 9 robot configurations, adapted to robots from Galaxea Dynamics and AgileX Robotics [^31^][^32^].
  - **LingBot-Depth**: High-precision spatial perception model with 70% REL reduction on indoor scenes [^33^].
  - **LingBot-World**: World model for simulation and training [^31^].
  - **LingBot-VA**: Video prediction model for robotic reasoning [^31^].
  - **LingBot-Map**: Streaming 3D reconstruction at 20 FPS from RGB [^34^].

- Partnered with Leju Robot and Orbbec for real-world deployment [^31^].

---

### Major Players & Sources

| Entity | Role/Relevance |
|--------|---------------|
| **Ant Group / inclusionAI** | Primary research organization. Hosts LLaDA, Ling, Ming, Ring families. Open-sources all models [^21^] |
| **Zhenzhong Lan** | Director of General Artificial Intelligence Research Center at Ant Group; Adjunct Researcher at Westlake University; Founder of Westlake Xinchen; key figure in LLaDA-MoE launch [^16^] |
| **Chongxuan Li** | Assistant Professor at Renmin University Guangqi Institute of AI; co-author on LLaDA papers [^2^] |
| **Jianguo Li** | Tech leader at Ant Group, co-author on LLaDA2.0 paper [^2^] |
| **Hang Yu** | Ant Group researcher, lead on F2LLM-v2 embeddings, CodeFuse survey co-author [^19^][^20^] |
| **Peng Di** | Ant Group & UNSW Sydney, lead author of NES paper [^17^] |
| **Renmin University of China** | Key collaborator on LLaDA-MoE and LLaDA2.0; Chongxuan Li's group [^2^][^16^] |
| **Zhejiang University** | Collaborator on LLaDA2.0 (Jiaqi Hu, Junbo Zhao) [^2^] |
| **Westlake University** | Collaborator on LLaDA2.0 (Zhenzhong Lan, Zhanchao Zhou) [^2^] |
| **HKUST** | Collaborator on LLaDA2.0 (Xiaocheng Lu) [^2^] |
| **Shanghai Jiao Tong University** | Collaborator on F2LLM-v2 and CodeFuse survey [^19^][^20^] |
| **Zhu Xing** | CEO of Robbyant, Ant Group's embodied AI unit [^31^] |

---

### Trends & Signals

1. **Diffusion LLMs at Scale**: Ant Group has demonstrated that dLLMs can scale to 100B+ parameters while remaining competitive with AR models, challenging the "language models must be autoregressive" dogma [^11^][^16^].

2. **AR-to-Diffusion Conversion as Paradigm**: LLaDA2.0's systematic conversion approach (rather than training from scratch) represents a practical and cost-effective scaling strategy, preserving AR pretraining knowledge while changing the generation mechanism [^3^][^6^].

3. **MoE + Diffusion Synergy**: The "MoE amplifier" law applies to dLLMs just as it does to AR models, enabling massive parameter counts with manageable active compute per token [^16^][^10^].

4. **Token Editing as Speed-Quality Solution**: LLaDA2.1's T2T editing mechanism represents a paradigm shift that transcends the traditional speed-quality tradeoff, achieving both fast decoding (892 TPS) and high quality [^12^][^13^].

5. **Open-Source Strategy**: Ant Group has fully open-sourced LLaDA2.0, LLaDA2.1, Ling, Ming, Ring, and Robbyant models with Apache 2.0/MIT licenses, positioning as a major open-source contributor alongside Alibaba [^21^][^28^].

6. **Unified Multimodal with Diffusion**: Ming-Omni's approach of using dedicated diffusion models for image generation while keeping the MLLM frozen represents a practical architectural choice for unified perception-generation models [^29^].

7. **Industry Deployment at Scale**: NES serves 20,000+ developers at Ant Group with under 250ms latency, demonstrating real-world viability of specialized code AI models [^17^].

8. **FP8 Training Leadership**: Ling-1T was the largest open-source model trained entirely with FP8 precision, validating reduced numerical precision for trillion-parameter models [^24^].

---

### Controversies & Conflicting Claims

1. **Diffusion vs. AR Performance Gap**: While LLaDA2.0-flash achieves parity with Qwen3-30B-A3B on average (73.18 vs. 73.60), AR models still lead on some benchmarks. The LLaDA2.1 paper acknowledges that "despite impressive speed numbers, diffusion language models still exhibit higher base error rates than autoregressive models. The editing mechanism compensates for this, but it's compensation, not elimination" [^15^][^2^].

2. **Stuttering Artifacts**: Under aggressive masking settings, diffusion models can produce "stuttering" artifacts — n-gram repetitions where phrases loop, a direct consequence of independent parallel sampling [^15^].

3. **RL and Editing Operate Separately**: The LLaDA2.1 paper acknowledges a key gap: "the RL stage and T2T editing mechanism currently operate separately. Future work aims to merge them, using RL to directly optimize self-correction behavior" [^15^].

4. **Inference Infrastructure Complexity**: The described inference infrastructure ("Alpha-MoE megakernels, per-block FP8 quantization, customized SGLang, radix caching") is not trivial to deploy [^15^].

5. **Code Generation Specificity**: While LLaDA2.0 excels at coding benchmarks, the broader claim that diffusion models will replace AR models remains debated — the advantage appears most pronounced in structured generation tasks (code, tool use) rather than open-ended generation [^11^][^2^].

---

### Recommended Deep-Dive Areas

1. **WSD Training Dynamics**: The Warmup-Stable-Decay conversion strategy is a key technical innovation. Deep investigation into how block-size scheduling affects optimization stability, knowledge preservation, and final model quality would be valuable. The document-level attention mask mechanism also warrants closer examination.

2. **CAP (Confidence-Aware Parallel)**: This technique for improving parallel decoding efficiency by sharpening predictive distributions represents a critical inference optimization. Understanding how confidence thresholds interact with block sizes could unlock further speed improvements.

3. **T2T Editing and Self-Correction**: LLaDA2.1's token editing mechanism represents a fundamental departure from traditional diffusion generation. The interplay between M2T drafting and T2T editing, the confidence threshold dynamics, and the potential for RL-optimized self-correction are all fertile research directions.

4. **AR-to-Diffusion Conversion Tradeoffs**: Comparing LLaDA2.0 (converted from AR) vs. LLaDA-MoE (trained from scratch) could illuminate when conversion is preferable to native training, and what knowledge is preserved vs. lost during conversion.

5. **EBPO (ELBO-based Block-level Policy Optimization)**: The first large-scale RL framework for dLLMs uses a novel approach to policy gradients. Deep investigation into this method could enable broader application of RL to diffusion language models.

6. **Ming-Omni's Diffusion Bridge**: The multi-scale learnable token scheme for connecting frozen MLLM representations to diffusion image generation represents a novel architectural pattern. Understanding when this bridging approach works vs. unified token spaces could inform future multimodal model design.

7. **Code-Specific Diffusion Advantages**: LLaDA2.0 shows outsized advantages on coding benchmarks. Understanding why diffusion models excel at structured code generation (bidirectional context for syntax checking? iterative refinement for compilation errors?) could inform domain-specific diffusion architectures.

8. **MoE Scaling Laws for Diffusion**: The observation that MoE architectures scale similarly for dLLMs and AR models suggests transferable scaling laws. Developing diffusion-specific MoE scaling laws could guide efficient resource allocation for future dLLM training.

---

### Source Index

[^1^]: GitHub — inclusionAI/LLaDA2.X. "LLaDA2.0 is the diffusion language model series developed by InclusionAI team, Ant Group." https://github.com/inclusionAI/LLaDA2.X (Dec 2025)

[^2^]: LLaDA2.0 paper (arXiv:2512.15745v2). "LLaDA2.0: Scaling Up Diffusion Language Models to 100B." Authors: Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, et al. (Ant Group, Renmin University, Zhejiang University, Westlake University, HKUST). https://arxiv.org/html/2512.15745v2 (Dec 2025)

[^3^]: LLaDA2.0 paper, Section 4. "Continual Pre-training via Warmup-Stable-Decay (WSD)." https://arxiv.org/html/2512.15745v2

[^4^]: LLaDA2.0 paper, Section 5.2. "Confidence-Aware Parallel Training." https://arxiv.org/html/2512.15745v2

[^5^]: AI Base News. "Ant Group Open Sources LLaDA2.0, the Industry's First 100B-Parameter Diffusion Language Model." https://news.aibase.com/news/23651 (Dec 2025)

[^6^]: Daily Dose of DS. "Diffusion LLMs from the Ground Up: Training, Inference, and Practical Engineering." https://www.dailydoseofds.com/diffusion-models-part-2/ (Apr 2026)

[^7^]: Codersera. "Run LLaDA2.1-mini: Diffusion Language Model Guide 2026." https://codersera.com/blog/run-llada21-mini-guide/ (Apr 2026)

[^8^]: LLaDA2.0 paper, evaluation tables. LLaDA2.0-mini specs.

[^9^]: 36Kr. "Milestone: First 100B Diffusion Language Model Unveiled." https://eu.36kr.com/en/p/3592063556468736 (Dec 2025)

[^10^]: MarkTechPost. "Ant Group Releases Ling 2.0: A Reasoning-First MoE Language Model Series." https://www.marktechpost.com/2025/10/30/ant-group-releases-ling-2-0/ (Oct 2025)

[^11^]: 36Kr. LLaDA2.0-flash "scored an average of 73.18, on par with the powerful AR model Qwen3-30B-A3B-Instruct-2507 (73.60)."

[^12^]: The Menon Lab Blog. "LLaDA2.1: The Diffusion LLM That Hits 892 Tokens Per Second." https://blog.themenonlab.com/blog/llada21-diffusion-llm-892-tokens-second (Mar 2026)

[^13^]: Maxim AI Blog. "Beyond Autoregression: LLaDA2.1 and the Case for Self-Editing Language Models." https://www.getmaxim.ai/blog/beyond-autoregression-llada2-1/ (Feb 2026)

[^14^]: LLaDA2.1 paper (arXiv:2602.08676v3). "LLaDA2.1: Speeding Up Text Diffusion via Token Editing." https://arxiv.org/pdf/2602.08676 (Feb 2026)

[^15^]: Maxim AI Blog. Analysis of LLaDA2.1 limitations. https://www.getmaxim.ai/blog/beyond-autoregression-llada2-1/

[^16^]: AI Base. "Challenging Conventional Wisdom! Ant Group and Renmin University to Launch the First Native MoE Diffusion Language Model in the Industry at the 2025 Bund Conference." https://www.aibase.com/news/21246 (Sep 2025)

[^17^]: NES paper (arXiv:2508.02473v3). "NES: An Instruction-Free, Low-Latency Next Edit Suggestion Framework." FSE Companion '26. Authors: Peng Di, Siyang Xiao, Xianying Zhu, Junhong Xie, et al. (Ant Group & UNSW Sydney). https://arxiv.org/html/2508.02473v3 (Apr 2026)

[^18^]: arXiv abstract for NES. https://arxiv.org/abs/2508.02473 (Aug 2025)

[^19^]: CodeFuse Survey (arXiv:2311.07989v7). "Unifying the Perspectives of NLP and Software Engineering: A Survey on Language Models for Code." Authors: Ziyin Zhang, Chaoyu Chen, Bingchang Liu, Cong Liao, Zi Gong, Hang Yu, Jianguo Li, Rui Wang (Ant Group & SJTU). https://arxiv.org/html/2311.07989v7 (Jun 2024)

[^20^]: F2LLM-v2 paper (arXiv:2603.19223). "Inclusive, Performant, and Efficient Embeddings for a Multilingual World." Authors: Ziyin Zhang, Zihan Liao, Hang Yu, Peng Di, Rui Wang (Ant Group & SJTU). https://arxiv.org/html/2603.19223 (Mar 2026)

[^21^]: inclusionAI GitHub. https://github.com/inclusionAI (May 2026)

[^22^]: CNBlog analysis. "OpenRouter's 'Elephant' Revealed: elephant-alpha is Ant's Ling-2.6-flash." https://www.cnblogs.com/zktww/p/19907918 (Apr 2026)

[^23^]: arXiv:2509.25397v1. "A Cartography of Open Collaboration in Open Source AI." Appendix B.11 Ant Group.

[^24^]: arXiv survey. "An Automated Survey of Generative Artificial Intelligence." Ling-1T section. https://arxiv.org/html/2306.02781v4 (Apr 2026)

[^25^]: Ling-V2.5 GitHub. "Ling-2.5-1T, Inclusive Intelligence, Instant Impact." https://github.com/inclusionAI/Ling-V2.5 (Feb 2026)

[^26^]: BusinessWire. "Ant Group Unveils Ling-2.6-Flash: A Major Leap in AI Efficiency." https://www.businesswire.com/news/home/20260422256825/en/ (Apr 2026)

[^27^]: AFP. "Ant Group Unveils Ling-2.6-Flash: A Major Leap in AI Efficiency." https://www.afp.com/en/infos/ant-group-unveils-ling-26-flash-major-leap-ai-efficiency (Apr 2026)

[^28^]: Ming-Omni project page. "Ming-Omni: A Unified Multimodal Model for Perception and Generation." https://lucaria-academy.github.io/Ming-Omni/

[^29^]: Ming-Omni paper (arXiv:2506.09344). "Ming-Omni: A Unified Multimodal Model for Perception and Generation." https://arxiv.org/html/2506.09344v1 (Jun 2025)

[^30^]: AI All Info. "Ming-omni-tts — Ant Group Open Source Unified Audio Generation Model." https://www.ai-all.info/en/ai-models/ming-omni-tts (Feb 2026)

[^31^]: BusinessWire. "Ant Group's Robbyant Teams Up with Leju to Bridge Embodied Intelligence and Real-World Applications." https://www.businesswire.com/news/home/20260315008842/en/ (Mar 2026)

[^32^]: MENAfn. "Robbyant Open-Sources Lingbot-VLA 'Universal Brain' Model For Embodied AI Robots." https://menafn.com/1110858601/ (Mar 2026)

[^33^]: BusinessWire. "Ant Group Subsidiary Robbyant Unveils Spatial Perception AI Model LingBot-Depth." https://finance.yahoo.com/news/ant-group-subsidiary-robbyant-unveils-060200706.html (Jan 2026)

[^34^]: BetaBriefing. "Ant Group's Robbyant Open-Sources LingBot-Map." https://betabriefing.ai/channels/the-robot-beat/briefings/2026-04-17/ (Apr 2026)
