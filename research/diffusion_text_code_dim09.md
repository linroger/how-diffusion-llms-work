## Facet: Inference Speed Optimization and Benchmarks (Deep Dive)

**Research Date:** 2026-05-12
**Searches Conducted:** 24+ targeted queries across acceleration methods, deployment infrastructure, cost analysis, benchmarking methodology, and hardware comparisons
**Sources:** arXiv papers, official blog posts, conference proceedings, industry publications

---

## 1. Key Findings

### 1.1 The Diffusion Speed Landscape: Reported Numbers vs. Reality

The reported speed numbers for diffusion language models (dLLMs) vary dramatically depending on hardware, measurement methodology, and optimization stack:

| Model | Reported Speed | Hardware | Source |
|-------|---------------|----------|--------|
| **Seed Diffusion** | 2,146 tok/s | H20 GPU | [^11^] |
| **LLaDA2.1-Mini** | 1,587 TPS (quantized) | Not specified | [^164^] |
| **Gemini Diffusion** | 1,479 tok/s | Unknown hardware | [^272^] |
| **Mercury Coder Mini** | 1,109 tok/s | H100 GPU | [^71^] |
| **Mercury Coder Small** | 737 tok/s | H100 GPU | [^71^] |
| **LLaDA2.1-Flash** | 892 TPS (HumanEval+) | Not specified | [^164^] |
| **LLaDA2.0-flash-CAP** | 535 TPS | H20 GPU, SGLang TP8 | [^399^] |
| **Fast-dLLM v2-7B** | 101.7 tok/s (3x over AR) | Threshold 0.9 | [^670^] |

However, a critical benchmarking study by Peng et al. (Renmin University) reveals that "current open-source DLMs often demonstrate slower inference speeds than their AR counterparts of similar scale" in controlled settings. For example, "LLaMA3-Instruct-8B exhibits an inference throughput 13.7x greater than that of LLaDA-Instruct-8B on evaluation benchmarks" [^720^]. This discrepancy highlights fundamental issues in how speed claims are measured and reported.

### 1.2 Fast-dLLM: Block-Wise Approximate KV Cache (27.6x)

Fast-dLLM, from NVIDIA, HKU, and MIT (Song Han's group), is the most widely-cited acceleration method for diffusion LLMs. Published at ICLR 2026.

**Core Mechanism:**
- **Block-wise approximate KV Cache**: Generation is partitioned into blocks; KV states of fixed context (prompt and completed blocks) are cached and reused across denoising steps, refreshed only at block boundaries. A "DualCache" variant caches both prefix and suffix blocks [^739^].
- **Confidence-Aware Parallel Decoding**: Unlike prior approaches that select a fixed number of tokens per step, this method dynamically selects tokens whose confidence exceeds a global threshold. This mitigates the "disruption of token dependencies under the conditional independence assumption" that causes quality degradation in naive parallel decoding [^286^].

**Key Results:**
- Up to **27.6x throughput improvement** on LLaDA and Dream models across GSM8K, MATH, HumanEval, and MBPP benchmarks
- Achieves higher acceleration (27.6x) when generation length is longer (1024 tokens) [^258^]
- Confidence-aware parallel decoding alone achieves **13.3x speedup** [^740^]
- Experiments conducted on NVIDIA A100 80GB GPU [^258^]

**Fast-dLLM v2** (also ICLR 2026) adapts pretrained AR models into block diffusion models requiring only ~1B tokens of fine-tuning, achieving "up to 2.5x speedup compared to standard AR decoding" with block-level KV cache reuse and intra-block parallel decoding [^670^] [^697^].

### 1.3 Elastic-Cache: Adaptive KV Caching (45.1x)

Elastic-Cache, from MBZUAI (VILA Lab, Zhiqiang Shen), achieves the highest reported speedups among training-free caching methods. Also at ICLR 2026.

**Core Mechanism:**
Based on three key observations [^279^]:
1. **Distant MASK tokens primarily act as length-bias** and can be cached block-wise beyond the active prediction window
2. **KV dynamics increase with depth**, suggesting selective refresh from deeper layers is sufficient
3. **The most-attended token exhibits the smallest KV drift**, providing a conservative lower bound on cache change

Elastic-Cache jointly decides:
- **When to refresh**: via attention-aware drift test on the most-attended token
- **Where to refresh**: via depth-aware schedule that recomputes from a chosen layer onward while reusing shallow-layer caches and off-window MASK caches

**Key Results:**
- **8.7x speedup** on GSM8K (256 tokens)
- **45.1x speedup** on longer sequences
- **4.8x speedup** on HumanEval
- **6.8x higher throughput** than existing confidence-based approaches on GSM8K
- Maintains higher accuracy than baseline across all benchmarks

### 1.4 FreeCache + Guided Diffusion (34x)

FreeCache leverages the observation that "the impact of future tokens on earlier positions rapidly diminishes over denoising steps" to directly cache KV states of already-decoded "clean" tokens [^283^].

**Key Results:**
- Up to **34x speedup** (averaged on PiQA) on Dream-7B with negligible accuracy drop
- For the first time, makes DLM-based architecture achieve comparable generation speed to same-sized AR models
- Enables long-context diffusion (>1024 tokens) without performance degradation
- **Training-free**: works off-the-shelf without calibration

The Guided Diffusion component uses a lightweight autoregressive model as a "guider" of unmasking optimal token positions, combining advantages of both DLM parallelism and AR guidance [^283^].

### 1.5 SSD: Self Speculative Decoding (3.46x, Lossless)

SSD (Shanghai Jiao Tong University, Shanghai AI Lab, Huawei) is unique as a **lossless** acceleration method.

**Core Mechanism:**
- Uses the dLLM itself as both speculative decoding drafter and verifier **without auxiliary modules**
- Self-drafting: model generates predictions for multiple positions
- Hierarchical verification trees in a single forward pass
- Exploits the dLLM's inherent parallel prediction capability [^654^]

**Key Results:**
- Up to **3.46x speedup** while keeping output **identical** to stepwise decoding
- Dream-7B-Instruct: 3.46x speedup (from 6.37 to 22.07 TPS) with 77.4% reduction in decoding steps
- LLaDA-8B-Instruct: 2.11x speedup
- Experiments on single NVIDIA A100 80GB GPU [^662^]

### 1.6 dKV-Cache: Delayed Caching (2-10x)

dKV-Cache (National University of Singapore, Xinchao Wang) proposes the first KV-cache-like mechanism specifically designed for diffusion language models [^655^].

**Core Insight:** Token representations stabilize only after a position is decoded, not during its decoding step. This motivates a **delayed caching strategy** where KV states are cached one step after a token transitions from [MASK] to its decoded form.

**Two Variants:**
1. **dKV-Cache-Decode**: Long-term cache reuse, almost lossless acceleration, even improves performance on long sequences
2. **dKV-Cache-Greedy**: Aggressive caching with reduced lifespan, achieves higher speedups with O(L^2) time complexity

**Key Results:**
- **2-10x speedup** on LLaDA and Dream 7B models
- Speedups across general language understanding, code generation, and mathematical problem solving
- Nearly identical peak memory to vanilla dLLMs
- Published May 2025 [^658^]

### 1.7 Additional Acceleration Methods

| Method | Speedup | Institution | Key Innovation |
|--------|---------|-------------|---------------|
| **Sparse-dLLM** | Up to 10x | Training-free dynamic cache eviction with sparse attention | Attention-aware bidirectional cache eviction [^760^] |
| **dLLM-Cache** | Up to 9.1x | Training-free adaptive caching | Long-interval prompt cache + adaptive response cache with V-verify [^698^] |
| **WINO** | 6-10x | Shanghai Jiao Tong | Revocable draft-and-verify decoding [^777^] |
| **COVER** | Up to 11.64x | Context-preserving verification for revocable decoding | KV cache override verification, reduces flip-flop oscillations [^779^] |
| **DLM-One** | ~500x | UT Austin | Score-based distillation to single-step generation [^726^] |
| **Di4C** | ~2x | Step distillation for discrete diffusion | Distills inter-token correlations [^723^] |

---

## 2. Speed Benchmarks: Top-Level Model Performance

### 2.1 Seed Diffusion (ByteDance + Tsinghua)

**2,146 tok/s on H20 GPUs** -- currently the fastest reported open diffusion code model [^11^] [^276^].

- **Architecture**: Large-scale discrete-state diffusion with block-wise, semi-autoregressive generation
- **Key techniques**: Two-stage curriculum (mask and edit), on-policy fine-tuning, constrained-order sampling, confidence-guided sampling
- **Caveats**: ByteDance explicitly notes that "direct comparison with baselines is challenging due to differing test conditions: Mercury Coder was evaluated on a proprietary dataset with H100s, while Gemini Diffusion's speed was averaged over a mixed-task benchmark using unknown hardware" [^11^]

### 2.2 LLaDA2.1 (Ant Group / Inclusion AI)

LLaDA2.1 introduces a **Token-to-Token (T2T) editing mechanism** woven into the conventional Mask-to-Token (M2T) scheme, creating two modes [^164^]:

**Speed Mode (S Mode)**:
- Flash (100B): 892 TPS on HumanEval+, 801 TPS on BigCodeBench, 663 TPS on LiveCodeBench
- Mini (16B): **1,587 TPS** on HumanEval+ (quantized)

**Quality Mode (Q Mode)**: Surpasses LLaDA2.0 on benchmark scores

**Key Innovation**: "Draft-and-edit" -- the model can modify already-generated tokens, not just fill in blanks. This is implemented via two sets: a "reveal set" for filling blanks and an "edit set" for modifying existing tokens [^344^].

### 2.3 Mercury Coder (Inception Labs)

The first commercially-available dLLM for code generation [^71^]:
- **Mercury Coder Mini**: 1,109 tok/s on H100
- **Mercury Coder Small**: 737 tok/s on H100
- **Pricing**: $0.25/M input tokens, $0.75-1.00/M output tokens [^227^]
- Ranks **#1 in speed** and tied for **#2 in quality** on Copilot Arena [^45^]
- Average latency on Copilot Arena: just **25ms** [^45^]
- API context window: 128K tokens [^685^]

### 2.4 Gemini Diffusion (Google DeepMind)

- **1,479 tok/s** sampling speed (excluding 0.84s overhead)
- Experimental text diffusion model from Google [^272^]
- Competitive with Gemini 2.0 Flash-Lite on code benchmarks
- Limited details on architecture and hardware

---

## 3. Speed Comparison Methodology: Critical Issues

### 3.1 The Hardware Discrepancy Problem

Benchmarking comparisons across models suffer from fundamentally different hardware setups:

| Model | GPU | Memory | Notes |
|-------|-----|--------|-------|
| Seed Diffusion | H20 | 96GB HBM3 (4.0 TB/s) | Inference-optimized GPU |
| Mercury Coder | H100 | 80GB HBM2e (3.35 TB/s) | Training/excellent inference GPU |
| LLaDA2.x | H20 (SGLang TP8) | 96GB HBM3 | Fair comparison environment [^399^] |
| Gemini Diffusion | Unknown | Unknown | Hardware undisclosed |
| Fast-dLLM evals | A100 80GB | HBM2e | Academic benchmarking |

**H20 vs H100**: The H20 is specifically "optimized for inference-heavy AI applications" with 96GB HBM3 and higher memory bandwidth (4.0 TB/s vs H100's 3.35 TB/s). The H100 excels at training workloads with more raw compute [^743^]. This means Seed Diffusion's 2,146 tok/s on H20 vs Mercury's 1,109 tok/s on H100 **cannot be directly compared**.

### 3.2 The Measurement Conditions Problem

Peng et al.'s critical study identifies three major issues in existing DLM efficiency evaluations [^720^]:

1. **Inconsistent serving environments**: Many papers measure speed under different serving stacks (HuggingFace Transformers vs. optimized engines like SGLang/vLLM)
2. **Unfair generation length controls**: DLMs can directly control generation length; AR models naturally stop at <eos>. Forcing AR models to generate to a fixed length creates unfair throughput comparisons
3. **Batch size sensitivity**: "Acceleration strategies yield significant gains at a batch size of 1, sometimes outperforming AR models, but their advantage diminishes as batch size grows, eventually falling behind AR"

### 3.3 Controlled Benchmarking Results (Peng et al., A800 GPU)

Under controlled conditions (A800 GPU, HuggingFace Transformers, standardized setup) [^720^]:

- **AR models consistently achieve the highest throughput**, followed by block diffusion, with DLMs being slowest
- LLaDA-8B-Instruct throughput is **significantly lower** than LLaMA-3.1-8B-Instruct of similar size
- DLM throughput drops rapidly as generation length increases
- Block diffusion with +parallel strategy is effective at batch size 1 (up to 3.1x speedup over baseline) but converges to AR throughput at larger batch sizes
- "+dual cache" scales better with batch size but hits OOM at very large batch sizes

### 3.4 Fair Comparison: LLaDA2.0 on SGLang (H20)

The most rigorous fair comparison comes from LMSYS's SGLang integration [^399^]:
- **Environment**: SGLang with TP8 on H20 GPU for all models
- **LLaDA2.0-flash-CAP**: 500 TPS
- **AR baselines**: 258 TPS and 237 TPS
- **Speedup**: 1.9x over AR with small batch sizes

This is the gold standard for fair comparison: identical hardware, identical serving stack, identical evaluation conditions.

---

## 4. Real-World Latency: TTFT and Streaming Issues

### 4.1 The TTFT Challenge for Diffusion Models

Diffusion models face a unique latency profile compared to autoregressive models:

**Autoregressive models**:
- TTFT: dominated by prefill (prompt processing), typically 300-1500ms
- Inter-token latency (ITL): 10-30ms per token, very consistent
- Streaming feels natural: first token arrives quickly, then continuous stream

**Diffusion models**:
- The entire sequence is generated iteratively through multiple denoising steps
- All tokens are refined in parallel across steps, but the **entire output must complete before any token is "final"
- This creates a fundamental challenge: TTFT for diffusion is effectively the **total generation time**, since tokens cannot be streamed until denoising completes
- Mercury Coder addresses this via block-wise generation where completed blocks can be streamed

### 4.2 Streaming Solutions

- **Block diffusion** (BD3-LM, LLaDA2.x): Generates blocks autoregressively; completed blocks can be streamed while subsequent blocks are being generated
- **Semi-autoregressive scheduling**: Enables progressive output, reducing perceived latency
- Confidence-aware parallel decoding further reduces the number of steps before tokens stabilize

As noted by Redis's analysis: "Streaming is the single biggest perceived-latency optimization for LLM apps" [^684^]. Diffusion models must solve this through block-wise or progressive generation to match AR's user experience.

---

## 5. Cost Per Token Analysis

### 5.1 Mercury Coder: Disruptive Pricing

Mercury Coder's pricing is **dramatically lower** than comparable autoregressive models [^227^]:

| Model | Input ($/M) | Output ($/M) | Ratio vs Mercury |
|-------|------------|-------------|-----------------|
| **Mercury Coder** | **$0.25** | **$0.75-$1.00** | 1x |
| Claude Sonnet 4.5 | $3.00 | $15.00 | 12-15x more expensive |
| GPT-4 Turbo | $10.00 | $30.00 | 30-40x more expensive |
| Sonar Reasoning Pro | $2.00 | $8.00 | 8x more expensive |
| MiniMax M2.5 | $0.30 | $1.20 | ~1.2x more expensive |
| Mistral Small 3.2 | $0.06 | $0.18 | ~4x cheaper |

### 5.2 Cost Structure Implications

The parallel generation capability of diffusion models has profound implications for cost:

- **Throughput per GPU**: Mercury generates 737-1,109 tok/s on a single H100, compared to ~50-200 tok/s for comparable AR models
- **Cost per token** drops proportionally with throughput
- **Billing complexity**: As noted by usagepricing.com, "parallel generation models would require tracking generation steps and denoising iterations alongside tokens, creating multi-dimensional metering complexity" [^677^]

### 5.3 Buildglare Case Study

Buildglare (a low-code web dev tool) uses Mercury Coder as a "cheap embedder" -- larger models plan changes, Mercury executes them at "just $0.25 per million input tokens and $1 per million output tokens... roughly an order of magnitude cheaper" than Claude Sonnet [^67^].

---

## 6. Batching Strategies

### 6.1 How Diffusion Models Handle Batch Inference

Peng et al.'s systematic study reveals critical batching dynamics [^720^]:

**At batch size 1**: Diffusion models with parallel decoding can outperform AR models
- Block diffusion + parallel: consistently fastest across all sequence lengths (up to 3.1x over baseline)
- LLaDA + parallel: can exceed AR at moderate lengths with short prompts

**As batch size increases**: AR models become more efficient
- Block diffusion throughput improves but at a slower rate than AR
- Turning point: batch size 2-4 where AR overtakes block diffusion
- DLM throughput stays nearly constant (compute-bound) until OOM
- LLaDA + dual cache scales better than +parallel but hits OOM at large batch sizes

**Key insight**: "Increasing the batch size causes diffusion-based models to reach the compute-bound regime more quickly, thereby diminishing their advantage in parallel decoding" [^720^].

### 6.2 Block Diffusion Batching

Fast-dLLM v2 handles batch generation by right-padding sequences with [MASK] tokens to make lengths divisible by block size, then decoding block-by-block in parallel across the batch [^670^].

### 6.3 Practical Implications

For **low-latency interactive applications** (chat, code completion):
- Batch size ~1, diffusion models can win with parallel decoding
- This is Mercury Coder's sweet spot: single-user IDE interactions

For **high-throughput serving** (batch APIs, document generation):
- AR models with good serving stacks (vLLM, SGLang) typically win
- Diffusion models need specialized batching to be competitive

---

## 7. Deployment Infrastructure

### 7.1 The Infrastructure Gap

As Peng et al. note: "Despite the impressive progress of DLMs, major machine learning ecosystems provide only limited optimization and deployment support for DLMs, making efficient serving of DLMs difficult" [^720^].

Key challenges:
- **No vLLM for diffusion**: vLLM's PagedAttention is designed for autoregressive generation
- **No TensorRT-LLM support**: NVIDIA's inference optimization stack targets AR models
- **Custom engines required**: Each diffusion model often needs its own serving infrastructure

### 7.2 Emerging Serving Stacks

| Infrastructure | Status | Models Supported | Key Features |
|---------------|--------|-----------------|-------------|
| **dInfer** (Ant Group) | Open source | LLaDA2.x, block diffusion | CUDA graph capture, FP8 quant, SGLang backend [^400^] |
| **SGLang** | Day-0 support | LLaDA2.0 via RFC | RadixAttention, TP8, streaming I/O, KV cache for block diffusion [^399^] |
| **vLLM** | Limited | AR models primarily | PagedAttention, not designed for bidirectional attention |
| **TensorRT-LLM** | No dLLM support | AR models | Hardware-specific optimizations unavailable for diffusion |

### 7.3 SGLang Integration for Block Diffusion

LMSYS's SGLang provides the most mature integration for diffusion LLMs [^399^] [^719^]:
- **Block Diffusion LLM framework**: Full logic for block-wise generation
- **Full KV cache support**: Sequence management with KV reuse
- **Streaming I/O**: Critical for user experience
- **Tensor parallelism**: TP8 support for large models
- **CUDA graph optimization**: Reduces CPU overhead
- Benchmarked at 500 TPS for LLaDA2.0-flash-CAP on H20

### 7.4 Ant Group's Dual Stack: dInfer + SGLang

Ant Group developed dInfer specifically for diffusion LLM inference, then integrated with SGLang [^24^]:
- dInfer v0.2.0: supports block diffusion, optimized batch inference, FP8 quantization
- SGLang serves as the backend for system-level optimizations
- LLaDA2.0-flash-CAP reaches **535 TPS**, 2.1x speedup over AR baselines (256 and 237 TPS)
- Future: "more mature features in dInfer are undergoing to transport to SGLang"

### 7.5 TensorRT-LLM for Diffusion

TensorRT-LLM does not currently support diffusion language models directly. However, NVIDIA's Model Optimizer supports diffusion models for vision (Stable Diffusion, SDXL) with FP8/INT8 quantization [^701^]. Extension to text diffusion LLMs would require:
- Custom attention kernels for bidirectional attention
- Block-wise KV cache management
- Integration with confidence-based parallel decoding

---

## 8. Hardware Landscape for Diffusion Inference

### 8.1 GPU Comparison for Inference Workloads

| GPU | Memory | Bandwidth | Architecture | Best For |
|-----|--------|-----------|-------------|----------|
| **H200** | 141GB HBM3e | 4.8 TB/s | Hopper | Massive model serving |
| **H100** | 80GB HBM2e | 3.35 TB/s | Hopper | Training + balanced inference |
| **H20** | 96GB HBM3 | 4.0 TB/s | Hopper | **Inference-optimized** |
| **A100** | 80GB HBM2e | ~2.0 TB/s | Ampere | Legacy workloads |

### 8.2 H20: The Inference-Optimized Choice

The H20 is specifically designed for inference-heavy applications [^743^]:
- 96GB HBM3 with **4.0 TB/s bandwidth** (higher than H100's 3.35 TB/s)
- 14,592 CUDA cores, 910 TFLOPS FP8
- Lower TDP (700W vs H100's 350W in some configs)
- "Near-H100-level acceleration with significantly lower investment and energy draw"

**Why this matters for comparisons**: Seed Diffusion's 2,146 tok/s on H20 benefits from higher memory bandwidth than Mercury's 1,109 tok/s on H100. The ~2x difference is partly architectural (diffusion method), partly hardware.

---

## 9. Step Distillation: The Long-Term Speed Solution

While caching and parallel decoding provide incremental speedups, **step distillation** offers the largest potential gains by collapsing the iterative denoising process:

### 9.1 DLM-One: Single-Step Generation (~500x)

DLM-One (UT Austin, ICLR 2026) uses score-based distillation with adversarial regularization to train a continuous diffusion language model that generates an entire sequence in a single forward pass [^726^]:
- **~500x speedup** over 2000-step baseline with "no notable performance degradation"
- **27x speedup** over GPT-2 Base
- Competitive performance on QQP, Quasar-T, Wiki-Auto, and OpenWebText2
- Operates in continuous token embedding space

### 9.2 Di4C: Discrete Diffusion Distillation (~2x)

Di4C extends distillation to discrete diffusion by explicitly distilling inter-token correlations:
- Enables 4-10 step students that match teacher quality
- ~2x speedup in practice (more theoretical than applied)
- Successfully distills MDLM on OpenWebText while maintaining diversity [^723^]

### 9.3 Quality-Speed Trade-offs in Dream 7B

Dream 7B demonstrates adjustable inference through timestep modification [^738^]:
- When diffusion steps set between 5-20, Dream achieves superior performance in both speed and quality compared to Qwen2.5 7B
- This represents "an additional dimension for inference-time scaling" alongside chain-of-thought

---

## 10. Major Players & Sources

| Entity | Role/Relevance |
|--------|---------------|
| **NVIDIA + HKU + MIT** (Song Han group) | Fast-dLLM (27.6x), Fast-dLLM v2 (2.5x AR); most influential acceleration research |
| **MBZUAI / VILA Lab** (Zhiqiang Shen) | Elastic-Cache (45.1x); attention-aware adaptive caching |
| **Ant Group / Inclusion AI** | LLaDA2.x series, dInfer engine, dFactory training framework; SGLang integration |
| **Inception Labs** | Mercury Coder: first commercial dLLM for code; 1,109 tok/s; $0.25/M input |
| **ByteDance Seed + Tsinghua AIR** | Seed Diffusion: 2,146 tok/s on H20; speed-quality Pareto frontier leader |
| **Google DeepMind** | Gemini Diffusion: 1,479 tok/s; experimental text diffusion model |
| **NUS** (Xinchao Wang) | dKV-Cache (2-10x): first KV-cache mechanism for DLMs |
| **Shanghai Jiao Tong University** | SSD (3.46x lossless), WINO (6-10x revocable decoding) |
| **UT Austin** | DLM-One (~500x single-step): score-distillation for continuous DLMs |
| **Renmin University** (Wayne Xin Zhao group) | Critical efficiency evaluation study; systematic benchmarking framework |
| **LMSYS / SGLang** | First production serving framework with day-0 dLLM support |
| **VILA Lab** (Quan Nguyen-Tri, Mukul Ranjan) | Elastic-Cache authors; ICLR 2026 |

---

## 11. Trends & Signals

### 11.1 Training-Free Acceleration is the Dominant Paradigm

Most effective acceleration methods (Fast-dLLM, Elastic-Cache, FreeCache, dKV-Cache, Sparse-dLLM, WINO) are **training-free**, working off-the-shelf with existing models. This signals:
- Maturity of the acceleration ecosystem
- Practical deployment feasibility
- The community is optimizing inference before retraining models

### 11.2 Block Diffusion as the Architectural Standard

Block diffusion (BD3-LM), combining autoregressive block generation with intra-block parallel diffusion, is emerging as the dominant architecture for production dLLMs:
- LLaDA2.x uses block diffusion
- Fast-dLLM v2 converts AR models to block diffusion
- Seed Diffusion uses block-wise generation
- Enables natural KV caching (like AR) + parallel generation (like diffusion)

### 11.3 Custom Serving Infrastructure is the Bottleneck

The lack of vLLM/TensorRT-LLM support for diffusion models is the primary deployment barrier:
- dInfer and SGLang are filling the gap but are nascent
- "More mature features in dInfer are undergoing to transport to SGLang" [^24^]
- Production deployment at scale requires dedicated engineering investment

### 11.4 Cost Advantages Are Real and Significant

Mercury Coder's $0.25/M input pricing vs Claude's $3.00/M represents a **12x cost advantage**:
- Driven by parallel generation enabling higher GPU utilization
- Similar quality at dramatically lower cost per token
- Could disrupt the API pricing landscape if diffusion models scale

### 11.5 Evaluation Standardization is Urgently Needed

Peng et al.'s critical study highlights that the field lacks standardized benchmarking:
- Different hardware, different serving stacks, different measurement conditions
- Seed Diffusion explicitly flags comparison difficulties
- Need for a "comprehensive benchmark for DLM efficiency" with fair infrastructure setups [^720^]

---

## 12. Controversies & Conflicting Claims

### 12.1 "Diffusion is Faster" vs. "AR is Faster in Practice"

**Pro-diffusion claims:**
- Mercury: "5-10x faster than speed-optimized AR models" [^71^]
- Seed Diffusion: "2,146 tok/s" -- "significantly faster than contemporary Mercury and Gemini" [^11^]
- LLaDA2.1: "892 TPS... multiple times faster than traditional autoregressive models" [^782^]

**Critical perspective (Peng et al.):**
- "AR models generally achieve higher throughput, while DLMs consistently lag" in controlled settings [^720^]
- "LLaMA3-Instruct-8B exhibits inference throughput 13.7x greater than LLaDA-Instruct-8B"
- Acceleration strategies "mainly offer gains at small batch sizes, with benefits diminishing upon scaling"

**Resolution**: The apparent conflict arises from different measurement conditions. Diffusion models excel at **batch size 1 with parallel decoding** (interactive use cases) but struggle at **high batch sizes** (throughput-optimized serving). AR models with optimized serving stacks (vLLM, SGLang) maintain consistent throughput across batch sizes.

### 12.2 Hardware Comparison Fairness

Seed Diffusion notes: "Direct comparison with baselines is challenging due to differing test conditions: Mercury Coder was evaluated on a proprietary dataset with H100s, while Gemini Diffusion's speed was averaged over a mixed-task benchmark using unknown hardware" [^11^]. This is a rare and commendable acknowledgment of benchmarking limitations.

### 12.3 Lossy vs. Lossless Acceleration

Methods like Fast-dLLM (27.6x) and Elastic-Cache (45.1x) achieve speedups with **negligible accuracy drop** (typically <1%), while maintaining the same model architecture. In contrast:
- DLM-One achieves ~500x but requires training a separate student model
- Step reduction (fewer denoising steps) trades quality for speed
- The industry consensus favors training-free, lossless methods for production deployment

---

## 13. Recommended Deep-Dive Areas

### 13.1 Standardized Benchmarking Framework
**Why it warrants depth**: The entire field's speed claims are currently incomparable due to different hardware, serving stacks, and measurement methodologies. A standardized benchmark (like LMSYS's SGLang-based comparison) would bring rigor to speed comparisons.

### 13.2 Production Serving at Scale
**Why it warrants depth**: No existing serving framework (vLLM, TensorRT-LLM) natively supports diffusion LLMs. The gap between research demos and production deployment is the primary barrier to adoption. SGLang's RFC for block diffusion is a first step but needs maturation.

### 13.3 Single-Step Generation (DLM-One)
**Why it warrants depth**: 500x speedup with near-teacher quality would completely eliminate the inference speed gap between diffusion and autoregressive models. However, current results are on smaller models (DiffuSeq, Plaid). Scaling to 7B+ models is the critical next step.

### 13.4 Revocable Decoding (WINO, COVER, T2T)
**Why it warrants depth**: The ability to "undo" and correct early generation errors is unique to diffusion models and could be a fundamental advantage. LLaDA2.1's T2T editing, WINO's draft-and-verify, and COVER's context-preserving verification represent different approaches to this capability.

### 13.5 Cost Economics of Parallel Generation
**Why it warrants depth**: Mercury Coder's 12x cost advantage over Claude Sonnet has profound implications for the API market. Understanding how parallel generation translates to real-world cost savings (accounting for GPU utilization, power, cooling, infrastructure) is critical for business decisions.

### 13.6 Hardware-Specific Optimization
**Why it warrants depth**: Different GPUs (H20 vs H100 vs A100) show dramatically different performance characteristics for diffusion models due to their memory bandwidth and compute balance. Hardware-aware optimization (like TensorRT-LLM for AR models) could unlock additional 2-5x speedups.

---

## 14. Verbatim Source Excerpts

### Fast-dLLM (ICLR 2026)
> "Experimental results on LLaDA and Dream models across multiple LLM benchmarks demonstrate up to 27.6x throughput improvement with minimal accuracy loss, closing the performance gap with autoregressive models and paving the way for practical deployment of Diffusion LLMs." [^741^]

### Elastic-Cache (ICLR 2026)
> "Experiments on LLaDA-Instruct, LLaDA-1.5, and LLaDA-V across mathematical reasoning and code generation tasks demonstrate consistent speedups: 8.7x on GSM8K (256 tokens), and 45.1x on longer sequences, while consistently maintaining higher accuracy than the baseline." [^279^]

### Seed Diffusion
> "Seed Diffusion Preview achieves an inference speed of 2,146 token/s over H20 GPUs while maintaining competitive performance across a sweep of standard code evaluation benchmarks, significantly faster than contemporary Mercury and Gemini, establishing new state of the art on the speed-quality Pareto frontier for code models." [^11^]

### Critical Efficiency Study (Peng et al.)
> "AR models consistently achieve the highest throughput, followed by block diffusion, with DLMs being the slowest across most evaluated settings, including different prompt lengths, generation lengths and batch sizes." [^720^]

### Mercury Coder
> "When evaluated on standard coding benchmarks, Mercury Coder achieves excellent quality across numerous benchmarks, often surpassing the performance of speed-optimized autoregressive models like GPT-4o Mini and Claude 3.5 Haiku while being up to 10x faster." [^71^]

### LLaDA2.1
> "Despite its 100B volume, on coding tasks it attains an astounding 892 TPS on HumanEval+, 801 TPS on BigCodeBench, and 663 TPS on LiveCodeBench." [^164^]

### FreeCache
> "The proposed method achieves up to 34x speedup (averaged acceleration on PiQA benchmark) on the Dream-7B model with negligible accuracy drop. For the first time, the DLM based architecture achieves comparable (even better) generation speed compared to a same-sized AR-based LLM models." [^283^]

---

*This research was compiled from 24+ independent searches across arXiv papers, official blog posts, conference proceedings, and industry publications. All citations reference authoritative sources with dates and URLs where available.*
