## Facet: ByteDance Seed's Diffusion Code Models

---

### Key Findings

#### 1. Seed Diffusion Preview (August 2025)

**Core Architecture & Speed Claims:**
- Seed Diffusion Preview is a large-scale language model based on **discrete-state diffusion** for code generation, achieving an inference speed of **2,146 tokens/s on H20 GPUs** — establishing what ByteDance claims as "new state of the art on the speed-quality Pareto frontier for code models" [^50^].
- The model is **5.4x faster** than similarly sized autoregressive (AR) models (~400 tokens/s for AR vs. 2,146 tokens/s for Seed Diffusion) [^54^].
- ByteDance emphasizes that direct comparison with competitors is challenging: "Mercury Coder was evaluated on a proprietary dataset with H100s, while Gemini Diffusion's speed was averaged over a mixed-task benchmark using unknown hardware" [^50^].
- The model is estimated to be in the **8-15B parameter range** based on benchmark table placements, though the exact parameter count was not published [^54^].

**Key Technical Innovations (Four Pillars):** [^56^]
1. **Two-stage curriculum learning**: Mask-based diffusion training (Stage 1, 80% of steps) → Edit-based diffusion training (Stage 2, 20% of steps)
2. **Constrained-order diffusion**: Introduces structured priors of code dependencies after the two-stage curriculum
3. **On-policy (same-policy) learning**: Optimizes the number of generation steps via surrogate loss based on pairwise edit distance
4. **Block-level parallel diffusion sampling**: Block-wise semi-autoregressive decoding with KV-Cache reuse

**Benchmark Performance:** [^11^]

| Benchmark | Task Type | Seed Diffusion Preview |
|-----------|-----------|----------------------|
| HumanEval | Function completion | 79.4% |
| HumanEval+ | Enhanced function completion | 73.8% |
| LiveCodeBench v1-v6 | Competitive programming | 72.6% (on 1055 problems) |
| CanItEdit | Code editing/repair | **54.3%** (vs. Seed-Coder's 50.5%) |
| BigCodeBench | Real-world multi-library | 72.6% |
| MBXP (multilingual) | Multi-language generation | 72.6% average across 10+ languages |

**Critical Finding on Editing Tasks:** Seed Diffusion Preview demonstrates a **notable boost on editing tasks** — CanItEdit pass@1 improved from 50.5% to 54.3% after adding Stage 2 edit-based training, proving the model learns to "repair, not merely complete, code" [^54^]. This is attributed to diffusion's inherent advantage in "global planning" tasks [^55^].

**Engineering Details:** [^54^]
- **Block-wise semi-autoregressive decoding**: Each block denoised in parallel; blocks maintain causal order (block n depends on blocks 0…n-1)
- **KV-Cache reuse**: Keys/values from earlier blocks cached and fed to later blocks
- **Optimal block size**: 32 tokens (sweet spot between single-pass cost and parallel throughput)
- Block size 16: 1.20ms latency/block; Block size 32: 1.40ms; Block size 64: 1.80ms

**Publication:** arXiv:2508.02193 [cs.CL], August 4, 2025. Demo at https://studio.seed.ai/exp/seed_diffusion/ [^50^]

---

#### 2. Stable-DiffCoder (January 2026)

**Core Contribution:**
- Stable-DiffCoder is an **open-source** code diffusion LLM family (Base + Instruct) that conducts a **controlled study** to fairly compare diffusion vs. autoregressive training [^48^][^20^].
- Built directly on the **Seed-Coder architecture, data, and training pipeline** — enabling strict isolation of the diffusion training effect [^49^].
- Under identical architecture and data settings, Stable-DiffCoder **"overall outperforms its AR counterpart on a broad suite of code benchmarks"** [^48^].

**Key Technical Innovation — Block Diffusion CPT:** [^149^]
- **Block diffusion continual pretraining (CPT)**: A tailored warmup strategy + block-wise clipped noise schedule for stable training
- The CPT stage enables "efficient knowledge learning and stable training" by reusing the pretrained AR model's weights
- This demonstrates that "diffusion-based training can improve code modeling quality beyond AR training alone" [^49^]

**Exact Benchmark Numbers:** [^83^]

**Base Model Comparison:**
| Model | Type | HumanEval | HumanEval+ | MBPP | MBPP+ |
|-------|------|-----------|------------|------|-------|
| Seed-Coder-8B-Base | AR | 77.4% | 68.3% | 82.0% | 69.0% |
| Stable-DiffCoder-8B-Base | DLLM | **79.3%** | **73.8%** | **83.6%** | 67.7% |
| Dream-Coder-7B-Base | DLLM | 66.5% | 60.4% | 75.9% | 61.6% |
| DiffuCoder-7B-Base | DLLM | 67.1% | 60.4% | 74.2% | 60.9% |

**Instruct Model Comparison:**
| Benchmark | Seed-Coder-8B-Instruct | Stable-DiffCoder-8B-Instruct |
|-----------|----------------------|---------------------------|
| HumanEval | ~77% | **Significantly improved** |
| HumanEval+ | ~70% | **Significantly improved** |
| MBPP | ~80% | **Outperforms all ~8B instruct models** |
| MHPP | Good | **Best among all compared models** |
| BigCodeBench | Good | **Substantial improvements**; only behind DeepSeek-Coder-V2-Instruct (21B/236B) |
| LiveCodeBench v5 | 24.7% | 23.5% (slightly behind AR, but matches Qwen3-8B) |
| CanItEdit | 50.5% | **60.0%** (huge 9.5-point gain) |
| Aider | 57.1 | 54.9 |

**Critical Finding — CanItEdit:** Stable-DiffCoder achieves **60.0% on CanItEdit** vs. Seed-Coder's 50.5%, representing a massive **18.8% relative improvement** in code editing capability [^160^]. This confirms that diffusion-based any-order modeling provides inherent advantages for structured code editing and global planning tasks.

**Low-Resource Languages:** Diffusion-based corruption through data augmentation benefits low-resource programming languages [^48^].

**Publication:** arXiv:2601.15892 [cs.CL], January 22, 2026. GitHub: https://github.com/ByteDance-Seed/Stable-DiffCoder. HuggingFace: https://huggingface.co/collections/ByteDance-Seed/stable-diffcoder [^20^][^48^]

---

#### 3. Seed-Coder — The AR Baseline (May 2025)

**Architecture & Design Philosophy:**
- Seed-Coder is a family of **8B-parameter open-source code LLMs** with three variants: Base, Instruct, and Reasoning [^59^][^171^]
- Built on **Llama 3 architecture** with 8.2 billion parameters and 32K token context window [^90^]
- Core innovation: **Model-centric data pipeline** that "lets the code model curate data for itself" — using LLMs (not hand-crafted rules) for scoring and filtering code data [^171^]

**Training Data Pipeline:** [^171^]
- **6 trillion tokens** of code pretraining corpus
- Data from four categories: file-level codes, repository-level codes, GitHub commits, code-related web data
- Two-phase pretraining: (1) regular pretraining with file-level codes + web data, (2) continued pretraining with all four categories
- Training data curated via LLM-based quality filters that "effectively capture nuanced standards of code quality that are difficult to quantify explicitly"
- Fill-in-the-Middle (FIM) format applied to both phases

**Training Methodology:** [^90^]
- Instruct model: SFT on millions of synthetic samples (style augmentation + LLM-generated instructions) followed by **Direct Preference Optimization (DPO)**
- Reasoning model: Enhanced with **reinforcement learning (RL)** for multi-step reasoning

**Key Philosophy:** The authors draw from Sutton's "The Bitter Lesson": "human-centric approaches would ultimately prove restrictive and tend to impede advancements of code LLMs in the long run" [^171^].

**Publication:** arXiv:2506.03524 [cs.CL], May 2025. GitHub: https://github.com/ByteDance-Seed/seed-oss [^171^]

---

#### 4. Technical Innovations Deep Dive

**4.1 Two-Stage Training Curriculum** [^54^]
- **Stage 1 — Mask-Filling (first 80% of training steps)**: Random tokens replaced with [MASK]; model learns local pattern completion
- **Stage 2 — Edit-Based Noise (final 20%)**: Instead of masks, model sees insertions, deletions, and substitutions guided by Levenshtein distance; model must re-evaluate every token including untouched ones
- **Impact**: CanItEdit pass@1 rises from 50.5% → 54.3% after adding Stage 2
- **Key Insight**: Pure mask-based diffusion learns a harmful shortcut ("if a token is not masked, it must be correct"); edit-based perturbation breaks this

**4.2 Constrained-Order Learning** [^54^]
- After two-stage curriculum, the team synthesizes millions of generation trajectories with the pretrained model
- Trajectories filtered by Evidence Lower Bound (ELBO) to keep only high-probability, dependency-correct paths
- Model fine-tuned on these distilled trajectories, learning which tokens depend on which
- Result: model keeps parallel generation superpower while obeying code grammar rules

**4.3 On-Policy Learning for Step Reduction** [^54^]
- Objective: Minimize trajectory length |τ| while a verifier V guarantees correctness
- Uses surrogate loss based on pairwise edit distance between intermediate states (avoids hard step-count loss)
- During training, model implicitly prunes low-quality paths, achieving faster convergence without collapse

**4.4 Block Diffusion Continual Pretraining (Stable-DiffCoder)** [^149^]
- Reuses pretrained AR weights and applies diffusion CPT on top
- Tailored warmup strategy + block-wise clipped noise schedule for stability
- Systematic analysis of training dynamics provides "practical and efficient guidelines for diffusion-based model training"
- The sampling process functions as "a principled and effective form of data augmentation"

---

#### 5. Speed Comparison with Competitors

| Model | Developer | Speed | Hardware | Notes |
|-------|-----------|-------|----------|-------|
| **Seed Diffusion Preview** | **ByteDance Seed** | **2,146 tok/s** | **H20 GPUs** | **Fully open benchmarks** |
| Mercury Coder Mini | Inception Labs | 1,109 tok/s | H100 GPUs | Proprietary dataset [^85^] |
| Mercury Coder Small | Inception Labs | 737 tok/s | H100 GPUs | Commercial model [^85^] |
| Gemini Diffusion | Google DeepMind | ~1000 tok/s (reported) | Unknown | Mixed-task benchmark [^87^] |
| AR models (comparable) | Various | ~400 tok/s | Various | Baseline for 5.4x claim |

**Important Caveat:** As noted by ByteDance, "Direct comparison with baselines is challenging due to differing test conditions: Mercury Coder was evaluated on a proprietary dataset with H100s, while Gemini Diffusion's speed was averaged over a mixed-task benchmark using unknown hardware" [^50^].

**Mercury Coder Details (Competitor):** [^85^]
- First commercial-scale diffusion LLM (Inception Labs, March 2025)
- Claims 5-10x faster than speed-optimized frontier models while maintaining comparable quality
- Ranks 2nd on quality and fastest overall on Copilot Arena
- Mercury's founders pioneered the first diffusion models for images and co-invented DPO, Flash Attention, and Decision Transformers

**Gemini Diffusion Details (Competitor):** [^87^]
- Developed by Google DeepMind
- Reports "performance comparable to much larger models while offering faster inference"
- ~10x speedups in decoding with about 1000 tokens per second

---

#### 6. ByteDance Seed Research Team & Collaborations

**SIA-Lab (Scalable Large Model Intelligent Technology Joint Research Center):** [^141^]
- Joint laboratory between **Tsinghua AIR (Institute for AI Industry Research)** and **ByteDance Seed**
- Founded in 2023 with the goal of enhancing pre-trained LLM capabilities
- Directed by prominent figures including Dr. Ya-Qin Zhang (Member, Chinese Academy of Engineering; Dean of AIR)
- Key researchers: Wei-Ying Ma (Chief Scientist at AIR), Hao Zhou (Research Associate Professor at AIR, formerly ByteDance AI Lab)

**Key Personnel:**
- **Yuxuan Song**: Project Lead on Seed Diffusion; also contributed to Stable-DiffCoder and DAPO [^11^][^83^]
- **Zheng Zhang**: Project Lead on Seed Diffusion; Core Contributor to DAPO (open-source RL system) [^11^][^136^]
- **Chenghao Fan**: Project Lead on Stable-DiffCoder; affiliated with Huazhong University of Science and Technology (HUST) [^83^]
- **Mingxuan Wang**: Supervision on Seed Diffusion; affiliated with both ByteDance Seed and SIA-Lab [^11^]
- **Yonghui Wu**: Supervision; leads at ByteDance Seed [^11^]
- **Hao Zhou**: Co-leads SIA-Lab; Research Associate Professor at Tsinghua AIR; leads GenSI research group [^147^]
- **Hongli Yu**: Contributor to Seed Diffusion; also on DAPO dataset team [^11^][^136^]

**Collaboration Pattern:** Multiple ByteDance Seed papers list authors with triple affiliations: (1) ByteDance Seed, (2) Tsinghua AIR, (3) SIA-Lab of Tsinghua AIR and ByteDance Seed [^11^][^136^]. This indicates deep integration between industry and academia.

---

#### 7. Relationship Between Seed Diffusion and Stable-DiffCoder

**Lineage:**
1. **Seed-Coder** (May 2025): AR baseline — 8B code LLM family with model-centric data pipeline
2. **Seed Diffusion Preview** (August 2025): Experimental diffusion code model — closed preview, focused on speed+quality Pareto frontier, not open-sourced
3. **Stable-DiffCoder** (January 2026): Open-source diffusion code model — builds directly on Seed-Coder architecture/data, adds block diffusion CPT

**Key Differences:**
| Aspect | Seed Diffusion Preview | Stable-DiffCoder |
|--------|----------------------|------------------|
| Release | August 2025 | January 2026 |
| Availability | Closed preview (demo only) | **Fully open-source** (MIT License) |
| Base Architecture | Seed-Coder (estimated) | Explicitly Seed-Coder |
| Key Innovation | 4-pillar system (curriculum, constrained-order, on-policy, block sampling) | Block diffusion CPT with warmup + clipped noise |
| Speed Focus | **Primary focus** (2,146 tok/s) | Secondary |
| Quality Focus | Maintained at high speed | **Surpasses AR counterpart** in controlled comparison |
| CanItEdit | 54.3% | **60.0%** |
| Contributors | Song, Zhang (ByteDance+Tsinghua) | Fan (HUST+ByteDance), Heng, Li |

**Acknowledgment Chain:** Seed Diffusion "thank[s] the Seed-Coder team for their help with the data pipelines" [^50^]; Stable-DiffCoder "reuses the Seed-Coder architecture, data, and training pipeline" [^48^]. This confirms a clear lineage.

---

### Major Players & Sources

- **ByteDance Seed Team** (Yonghui Wu, Mingxuan Wang): Core developer of all three models. Founded 2023, dedicated to crafting "industry's most advanced AI foundation models" [^20^]
- **Tsinghua AIR + SIA-Lab** (Ya-Qin Zhang, Hao Zhou, Wei-Ying Ma): Academic research arm; provides theoretical foundation and talent pipeline [^141^][^147^]
- **Inception Labs** (Volodymyr Kuleshov, Stefano Ermon, Aditya Grover): Mercury Coder — first commercial diffusion LLM, primary speed competitor [^85^]
- **Google DeepMind**: Gemini Diffusion — major tech company's diffusion LLM effort [^87^]
- **Huazhong University of Science and Technology** (Wei Wei, Chenghao Fan): Collaborated on Stable-DiffCoder [^83^]
- **Open-source diffusion code model ecosystem**: LLaDA (Renmin Univ.), Dream (HKU + Huawei), DiffuCoder (Apple + HKU), Dream-Coder [^161^]

---

### Trends & Signals

1. **Diffusion for code is becoming mainstream**: By early 2026, there are at least 6 major diffusion code LLM families (Seed Diffusion, Stable-DiffCoder, Mercury, Gemini Diffusion, Dream-Coder, DiffuCoder) [^161^]. A 2025 survey on diffusion language models explicitly lists code generation as a major application area alongside conventional NLP [^150^].

2. **"Controlled comparison" is the gold standard**: Stable-DiffCoder's explicit design to isolate the diffusion training effect (keeping architecture, data, and pipeline identical) represents a maturation of the field — moving from "diffusion works" to "diffusion works *better than* AR under fair conditions" [^48^].

3. **Editing tasks favor diffusion**: Both Seed Diffusion and Stable-DiffCoder show their largest advantages over AR baselines on CanItEdit (54.3% vs 50.5% and 60.0% vs 50.5% respectively). This aligns with theoretical expectations: diffusion's any-order modeling provides "richer data reuse" and better "global planning" for edit operations [^54^][^83^].

4. **Inference speed is the immediate differentiator, but not the only one**: As Seed Diffusion's authors note, "faster inference is merely the most immediate benefit of discrete diffusion. Exploring alternatives to the conventional left-to-right modeling order represents a valuable research direction" [^50^].

5. **Block-wise semi-autoregressive decoding is the practical sweet spot**: Full parallel decoding remains challenging; block-level approaches (32-token blocks for Seed Diffusion, block diffusion CPT for Stable-DiffCoder) provide the best speed-quality tradeoff [^54^][^149^].

6. **Academic-industry collaboration is critical**: The SIA-Lab model (Tsinghua AIR + ByteDance Seed) produces multiple high-impact papers (Seed Diffusion, DAPO, ThinkDial, Enigmata) with shared authorship across both institutions [^136^][^146^].

---

### Controversies & Conflicting Claims

1. **Speed comparison apples-to-apples problem**: ByteDance acknowledges that "direct comparison with baselines is challenging due to differing test conditions" [^50^]. Mercury uses H100s + proprietary dataset; Gemini uses unknown hardware + mixed-task benchmark. Seed Diffusion's 2,146 tok/s on H20s may not be directly comparable. The 5.4x speedup claim compares against AR models on different hardware configurations.

2. **LiveCodeBench performance — diffusion slightly trails AR**: Stable-DiffCoder-8B-Instruct scores 23.5% vs. Seed-Coder-8B-Instruct's 24.7% on LiveCodeBench v5 [^83^]. This is one of the few benchmarks where the diffusion variant does *not* surpass its AR counterpart, suggesting diffusion may have limitations on competitive programming tasks requiring deep sequential reasoning.

3. **"Why mask diffusion does not work" counter-argument**: A paper by Sun et al. titled "Why mask diffusion does not work" (arxiv 2510.03289) is cited in Stable-DiffCoder's references [^160^], suggesting there is active debate in the community about the fundamental limitations of masked diffusion approaches.

4. **Seed Diffusion Preview not open-sourced**: Despite being announced as an "experimental" model with "future open-source plans," the weights were not released as of the research cutoff [^54^]. Stable-DiffCoder (January 2026) eventually delivered on the open-source promise, but with different authors and a different technical approach.

5. **Model size ambiguity**: Neither Seed Diffusion Preview nor the papers explicitly state the parameter count, leading to estimates of 8-15B based on benchmark table placement [^54^]. This lack of transparency makes strict comparisons difficult.

---

### Recommended Deep-Dive Areas

1. **Block Diffusion CPT Mechanics**: The tailored warmup strategy and block-wise clipped noise schedule in Stable-DiffCoder represent a significant training innovation. Understanding the exact mathematical formulation and hyperparameter choices would be valuable for reproducing results.

2. **Edit-Based Corruption Formalization**: The Levenshtein-distance-guided edit perturbation in Stage 2 of Seed Diffusion's curriculum is a novel approach to breaking the "unmasked = correct" shortcut. A deeper analysis of the corruption distribution and its theoretical properties would strengthen the field.

3. **Constrained-Order Learning as Distillation**: The ELBO-based trajectory filtering and constrained-order fine-tuning can be viewed as a form of policy distillation. Understanding how this relates to broader RL/distillation techniques could lead to further improvements.

4. **Scaling Laws for Code Diffusion**: The Seed Diffusion authors explicitly call for "exploring its scaling properties" as a key future direction [^50^]. Whether diffusion code models exhibit different scaling behavior than AR models (especially in the data-limited regime) is an open question.

5. **Diffusion for Non-Code Domains**: Both Seed Diffusion and Stable-DiffCoder focus exclusively on code. The extent to which these techniques transfer to mathematical reasoning, general language, or multimodal tasks remains unexplored. Stable-DiffCoder's authors note: "Whether text diffusion sampling can provide even greater benefits in broader domains remains an open question" [^83^].

6. **SIA-Lab Collaboration Model**: The deep integration between Tsinghua AIR and ByteDance Seed (shared lab, shared authorship, shared projects) represents a distinctive research model. Understanding how this structure enables rapid iteration from research to product could inform future academic-industry partnerships.

7. **Inference-Time Trade-offs**: The relationship between denoising steps, block size, and quality in block-wise diffusion sampling has not been fully characterized. Optimal scheduling for different code task types (generation vs. editing vs. completion) could unlock further speedups.

---

### Key Source URLs

- Seed Diffusion Preview paper: https://arxiv.org/abs/2508.02193
- Seed Diffusion project page: https://seed.bytedance.com/seed_diffusion
- Stable-DiffCoder paper: https://arxiv.org/abs/2601.15892
- Stable-DiffCoder GitHub: https://github.com/ByteDance-Seed/Stable-DiffCoder
- Stable-DiffCoder HuggingFace: https://huggingface.co/collections/ByteDance-Seed/stable-diffcoder
- Seed-Coder paper: https://arxiv.org/abs/2506.03524
- Seed-Coder GitHub: https://github.com/ByteDance-Seed/seed-oss
- Mercury Coder paper: https://arxiv.org/abs/2506.17298
- SIA-Lab announcement: https://beta.hyper.ai/en/headlines/ca2470889142a64c0940f8b598821c6b
- Hao Zhou (Tsinghua AIR): https://zhouh.github.io/

---

*Research compiled from 10+ independent web searches across arXiv papers, official blog posts, GitHub repositories, and academic sources. All findings include inline citations to authoritative sources.*
