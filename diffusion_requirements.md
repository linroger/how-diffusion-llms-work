# Requirements: Diffusion Models for Text and Code Generation — Research Report

---

## 1. Audience Requirements

**Primary Audience: Technical researchers and practitioners in AI/ML**
- The report must serve readers with prior knowledge of LLM architectures (autoregressive vs diffusion) but who need depth on the specific diffusion-for-text/code subdomain
- Content must be accessible enough for engineering leaders making technology adoption decisions, yet detailed enough for researchers seeking technical depth
- Assumed familiarity with core concepts: Transformers, attention mechanisms, masked language modeling, reinforcement learning from human feedback (RLHF), code generation benchmarks (HumanEval, MBPP)

**Secondary Audience: Technology strategists and investors**
- Sections on commercial landscape, competitive positioning, and future outlook must provide actionable strategic insight
- Quantitative comparisons (speed, cost, benchmark scores) must be presented clearly with methodology notes

**Implicit audience need: Cross-organizational comparative analysis**
- The user named two specific organizations (Google DeepMind, Ant Group) but the broader context of who else is competing matters for positioning
- The report must enable readers to understand relative strengths, weaknesses, and strategic postures across all major players

---

## 2. Scope Requirements

### 2.1 Organizational Scope

**Tier 1 — Mandatory deep coverage:**
- **Google DeepMind**: Gemini Diffusion (experimental production model), MD4 (foundational NeurIPS 2024 framework), AR2Diff (conversion methodology), CANDI (continuous-discrete hybrid), key researchers (Brendan O'Donoghue, Jiaxin Shi, Kehang Han, Jack Rae, Oriol Vinyals)
- **Ant Group / Inclusion AI**: LLaDA2.0 (100B open-source diffusion LLM), LLaDA2.1 (token editing, EBPO RL), CodeFuse NES (20,000-developer deployment), Inclusion AI open-source organization (Ling, Ring, Ming model families), DAPO RL algorithm

**Tier 2 — Significant coverage required:**
- **ByteDance Seed**: Seed Diffusion (2,146 tok/s, fastest), Stable-DiffCoder (open-source), CanItEdit advantage
- **Inception Labs**: Mercury Coder (first commercial diffusion LLM for code), $50M funding, Mercury 2, enterprise adoption stories, Stefano Ermon leadership
- **Open-source ecosystem**: LLaDA (foundational 8B), Dream/DreamOn (AR-to-diffusion conversion), DiffuCoder (Apple, coupled-GRPO), SEDD (ICML 2024 Best Paper), MDLM

**Tier 3 — Contextual coverage:**
- **Academic contributors**: Renmin University/GSAI-ML, HKU, Tsinghua, MIT (Kaiming He's ELF), Stanford, Cornell, Purdue
- **Inference acceleration research**: Fast-dLLM (NVIDIA/HKU/MIT, 27.6x), I-DLM (introspective consistency), STDD, CoRe remasking, RemeDi
- **Challenger approaches**: A3 (Any-Order AR from Yisen Wang), AR models as competitive baselines

### 2.2 Technical Scope

**Core technical domains the report must cover:**
1. **Architecture paradigms**: Block diffusion (intra-block bidirectional + inter-block causal), fully parallel generation, AR-to-diffusion conversion, hybrid approaches
2. **Training methodologies**: WSD 3-phase training (LLaDA2.0), from-scratch vs conversion, CAP (Confidence-Aware Parallel decoding), RL post-training (EBPO, coupled-GRPO, VRPO, DAPO)
3. **Inference optimization**: KV caching for diffusion, confidence-aware parallel decoding, speculative decoding, quantization, block-wise scheduling
4. **Discrete vs continuous diffusion**: Masked diffusion (MD4, LLaDA), continuous embedding-space diffusion (LangFlow, ELF), the "temporal dissonance" problem (CANDI)
5. **Remasking strategies**: Confidence-based, context-robust (CoRe), spatio-temporal (STDD), self-reflective (RemeDi), ReMDM
6. **Scaling characteristics**: Scaling laws for diffusion vs AR, data-constrained regime advantages, compute efficiency claims and controversies

### 2.3 Benchmark and Evaluation Scope

**The report must present and analyze performance across:**
- Code completion: HumanEval, MBPP, BigCodeBench
- Competitive programming: LiveCodeBench (all versions), SWE-Bench Verified
- Code editing/infilling: CanItEdit, FIM (Fill-in-the-Middle)
- Long-context code: RepoQA
- General reasoning: GPQA Diamond, AIME, BIG-Bench Extra Hard
- General language: MMLU, GSM8K
- Speed benchmarks: Tokens/second (with hardware normalization notes), throughput, latency (TTFT)

### 2.4 Commercial and Ecosystem Scope

**The report must cover:**
- Production deployment status of each model (experimental, preview, production API)
- Pricing comparisons (per-million-token costs)
- Enterprise adoption metrics (developer counts, named customers)
- IDE integration landscape (native vs third-party, Tab-key workflows)
- Open-source availability (model weights, training code, inference engines, licenses)
- Funding and valuation data for commercial entities

---

## 3. Depth Requirements

### 3.1 Technical Depth — Per Dimension

**For Google DeepMind contributions:**
- Gemini Diffusion: Full architectural description (block diffusion, bidirectional attention, U-Net structure), complete benchmark table with gap analysis, training methodology inference, Brendan O'Donoghue's four advantages / two disadvantages framework, experimental status and roadmap signals
- MD4: Mathematical formulation of simplified ELBO, mean parameterization, GenMD4 state-dependent masking schedules, performance results across domains
- AR2Diff: Three-stage conversion process, prefix LM + decoder-only finding, scaling results across model sizes
- CANDI: "Temporal dissonance" concept, decoupling of discrete/continuous corruption
- Research timeline: Chronological progression from AR2Diff (Jan 2024) to MD4 (Jun 2024) to CANDI (Dec 2024) to Gemini Diffusion (May 2025)

**For Ant Group contributions:**
- LLaDA2.0: WSD 3-phase training dynamics (block size progression 1→4→32→64→4096→32), document-level attention masks, top-k checkpoint merge, CAP training, full open-source toolchain (dFactory, dInfer, SGLang), MoE architecture (256 routed + 1 shared expert, 1/32 activation)
- LLaDA2.1: Token-to-token editing (T2T), EBPO RL for diffusion at scale, 1,587 TPS on mini
- CodeFuse NES: Dual-model architecture (Location + Edit), DAPO RL with hierarchical rewards, incremental difference detection, under-250ms inference, 20,000+ developer deployment, Tab-key workflow
- Inclusion AI: Three model families (Ling/Ring/Ming), open-source strategy (MIT license), ecosystem breadth (code, language, reasoning, multimodal, embodied AI via Robbyant)

### 3.2 Analytical Depth

**The report must provide analysis beyond raw facts:**
- Cross-model benchmark comparison tables with normalization notes
- Confidence classification for each finding (High/Medium/Low per cross-verification)
- Conflict zone identification where sources disagree (speed comparison fairness, LiveCodeBench gap, withdrawn papers)
- Root-cause analysis for performance gaps (why diffusion excels at code editing but lags on competitive programming)
- Strategic implications of the AR-to-diffusion conversion paradigm (moat analysis)
- Geopolitical dimension of open-source leadership (China vs US in diffusion LLMs)

### 3.3 Synthesis Depth

**The report must synthesize across dimensions to produce:**
- Strategic insights that are not visible from any single dimension alone
- "Killer app" identification (code editing as the strongest commercial use case)
- Convergence trends (block diffusion as production paradigm, RL as secret weapon, AR-to-diffusion conversion as standard)
- Prediction framework for the next 12 months (make-or-break period assessment)
- Comparison with the broader AR LLM landscape (positioning, not isolation)

---

## 4. Special Focus Areas

### 4.1 Google DeepMind — Explicitly Requested

The user's query names Google DeepMind as a primary focus. The report must deliver:
- Complete research lineage tracing from foundational theory (MD4) through methodology (AR2Diff) to production model (Gemini Diffusion)
- Identification and profiling of all key DeepMind researchers involved
- Analysis of DeepMind's strategic position (experimental status vs long-term investment signals)
- Comparison of Gemini Diffusion with both its AR counterparts (Flash-Lite) and other diffusion models
- Assessment of DeepMind's open-source contribution (MD4 code) vs closed-source product (Gemini Diffusion)

### 4.2 Ant Group — Explicitly Requested

The user's query names Ant Group as a second primary focus. The report must deliver:
- Complete coverage of the LLaDA model family evolution (1.0 → 2.0 → 2.1 → MoE variants)
- Deep technical analysis of the WSD training methodology
- Assessment of Ant Group's open-source strategy and its geopolitical implications
- Coverage of production deployment at scale (CodeFuse NES, 20,000+ developers)
- Positioning within the broader Inclusion AI ecosystem (Ling, Ring, Ming, Robbyant)

### 4.3 Diffusion-for-Code Specifically

The user's query specifies "text and coding" with emphasis on both. The report must:
- Provide more depth on code-specific findings than general text diffusion
- Analyze the code editing advantage (CanItEdit +18.8% relative over AR)
- Explain why code benefits from non-causal generation while natural language reasoning does not
- Cover code-specific models: Seed Diffusion, Stable-DiffCoder, DiffuCoder, Mercury Coder, CodeFuse NES
- Include code-specific benchmarks as primary (HumanEval, MBPP, LiveCodeBench, CanItEdit, BigCodeBench)

### 4.4 Research Paper Collection

The user's query explicitly requests "save the research papers too." The report must:
- Include a comprehensive bibliography with full citations (arXiv IDs, conference venues, URLs)
- Identify which papers are open-access vs behind paywalls
- Note which papers have open-sourced code/model weights
- Flag withdrawn or disputed papers (e.g., MDM-Prime-v2)
- Provide direct links to papers, code repositories, and model weights where available

---

## 5. Required Report Elements

### 5.1 Structural Elements

The report must include:

1. **Executive Summary** — 1-2 page overview of the entire landscape, key findings, and strategic implications
2. **Introduction / Background** — Primer on diffusion models for discrete data, contrast with autoregressive paradigm, why the field matters now
3. **Google DeepMind Deep Dive** — Self-contained section covering all DeepMind contributions with technical depth
4. **Ant Group / Inclusion AI Deep Dive** — Self-contained section covering all Ant Group contributions with technical depth
5. **Other Major Players** — ByteDance Seed, Inception Labs/Mercury, Apple/DiffuCoder, academic groups
6. **Open-Source Ecosystem** — Comprehensive survey of available open-source diffusion LLMs
7. **Technical Foundations** — Discrete vs continuous diffusion, remasking, architectures, inference acceleration
8. **Benchmark Analysis** — Detailed comparison tables across all benchmarks with normalization methodology
9. **Commercial Landscape** — Pricing, deployment status, enterprise adoption, IDE integration
10. **Cross-Dimension Insights** — 10 synthesized insights derived from combining multiple research dimensions
11. **Cross-Verification Report** — Confidence-classified findings (High/Medium/Low) with conflict zones identified
12. **Future Outlook** — Diffusion vs AR debate, predictions, convergence trends, make-or-break timeline
13. **Appendix: Bibliography** — Complete paper listing with citations and links
14. **Appendix: Glossary** — Technical terms and acronyms

### 5.2 Quantitative Elements

The report must present:
- Benchmark comparison tables with multiple models across multiple tasks
- Speed comparison tables with hardware normalization notes
- Timeline tables for model releases and key milestones
- Performance gap analysis tables (diffusion vs AR)
- Commercial pricing comparison tables
- Confidence-tier classification for all major findings

### 5.3 Qualitative Elements

The report must include:
- Direct quotes from key researchers (O'Donoghue, Vinyals, Rae, Ermon, Shi)
- Analytical commentary on strategic implications
- Controversy and conflict identification with resolution status
- Recommended deep-dive areas for future research
- Assessment of withdrawn or disputed claims

---

## 6. Implicit Requirements

### 6.1 Research Quality and Rigor

- **Cross-verification**: The report must classify findings by confidence tier based on source independence. Findings confirmed by 2+ independent sources are High Confidence. Single authoritative sources are Medium. Weak or unverified claims are Low.
- **Conflict resolution**: Where sources disagree (speed comparison fairness, LiveCodeBench gap, SFT effectiveness), the report must present all sides, explain the methodological issues, and assign a resolution status.
- **Distinguish claims from verified facts**: Distinguish between vendor self-reports (e.g., speed claims), independent evaluations, and peer-reviewed results.
- **Flag withdrawn papers**: MDM-Prime-v2's 21.8x efficiency claim must be clearly marked as disputed due to paper withdrawal.

### 6.2 Balanced Perspective

- **Avoid advocacy**: The report must present both the case for diffusion (speed, editing, open-source leadership) and the case against (reasoning gaps, competitive programming lag, higher serving costs, TTFT overhead).
- **Avoid benchmark cherry-picking**: The report must explicitly note that "whether diffusion models win or lose depends on which benchmarks are emphasized" and present the full range.
- **Contextualize hype**: Speed claims must include hardware normalization notes. The "5-10x faster" headline must be contrasted with the controlled 2.1x result.

### 6.3 Temporal Awareness

- **Date sensitivity**: All claims must be dated. Research in this field moves fast — a finding from January 2025 may be superseded by March 2025.
- **Timeline construction**: The report must present a coherent timeline showing how the field evolved and is continuing to evolve.
- **Predictive framing**: The "next 12 months will be decisive" insight must be presented with supporting evidence and confidence assessment.

### 6.4 Practical Utility

- **Actionable implications**: The report must answer "so what?" — what should practitioners, researchers, and strategists do with this information?
- **Adoption guidance**: For organizations considering diffusion models, the report should provide decision criteria (use cases that favor diffusion, use cases that don't, integration requirements).
- **Research guidance**: The recommended deep-dive areas from each dimension should be synthesized into a research agenda.

### 6.5 Preservation of Source Material

- **Research paper archive**: The user's explicit request to "save the research papers" implies the research artifacts must be preserved and organized. The bibliography must link to all papers, and the research collection must be maintained as a reference corpus.
- **Traceability**: Every factual claim must be traceable to its source via citation markers.
- **Source diversity**: The report must draw from official papers, technical interviews, benchmark aggregators, blog posts, and conference proceedings — with source authority noted for each.

### 6.6 Output Format Requirements

- **Single comprehensive document**: The report must be a unified, navigable document — not a collection of disconnected sections.
- **Markdown format**: Output as structured Markdown for portability and readability.
- **Cross-references**: Internal cross-references between sections (e.g., from benchmark analysis to technical foundations).
- **Tables for data**: Quantitative comparisons must use Markdown tables, not prose.
- **Citation markers**: Superscript citation references linking to the bibliography.
- **No decorative elements**: No summary tables, overview charts, or decorative formatting. Content-focused structure only.
