# Research Artifact Synthesis: Diffusion Models for Text and Code Generation

---

## Top 20 Most Important Facts and Data Points

**1. LLaDA2.0 is the first 100B-parameter open-source diffusion language model**, trained via systematic AR-to-diffusion conversion using a three-phase Warmup-Stable-Decay (WSD) strategy that progressively expands block size from 1 to 4096 and then decays back to 32 for efficient inference [^24^][^162^].

**2. Seed Diffusion Preview achieves 2,146 tokens/second**, making it the fastest reported diffusion LLM — 5.4x faster than comparable autoregressive models on the same H20 GPU hardware [^168^][^508^].

**3. Gemini Diffusion achieves near-parity with Gemini 2.0 Flash-Lite on code benchmarks** — HumanEval 89.6% vs. 90.2%, MBPP 76.0% vs. 75.8%, BigCodeBench 45.4% vs. 45.8% — while delivering 1,479-2,000 tok/s, roughly 5-10x faster than autoregressive models [^272^][^37^].

**4. LLaDA2.0-flash-CAP achieves 535 TPS under controlled serving conditions** (SGLang TP8 on H20), which is 2.1x faster than Ling-flash-2.0 (256 TPS) and Qwen3-30B-A3B-Instruct-2507 (237 TPS) under identical conditions [^389^][^403^].

**5. Stable-DiffCoder achieves 60.0% on CanItEdit vs. its AR counterpart Seed-Coder's 50.5%** — a 9.5 percentage point advantage on code editing, the largest demonstrated diffusion advantage on any code benchmark in a controlled comparison [^83^][^160^].

**6. The "Beyond Autoregression" study finds diffusion models lag on competitive programming** — averaging 14.9% vs. AR's 18.9% on LiveCodeBench v6 — while achieving parity on HumanEval (66.7% avg vs. 71.3%) and MBPP (61.2% avg vs. 60.8%) [^9^][^5^].

**7. Post-training RL produces the largest quality gains, not base diffusion conversion.** LLaDA2.0-flash-preview scores only 29.07 on LiveCodeBench vs. 42.29 for the final model — a 45% improvement from SFT + CAP + DPO post-training [^162^]. DiffuCoder's coupled-GRPO achieves +4.4% on EvalPlus with only 21K examples [^10^].

**8. EBPO (ELBO-based Block-level Policy Optimization) is the first large-scale RL framework for diffusion LLMs**, applied to a 100B-parameter model for the first time by the Ant Group team. It uses the ELBO as a tractable proxy for sequence-level log-likelihoods, which are intractable for non-autoregressive models [^330^][^346^].

**9. Block diffusion with block size ~32 has emerged as the consensus production architecture** across all major diffusion LLMs: Gemini Diffusion [^297^], LLaDA2.0 [^24^], Stable-DiffCoder [^479^], and Mercury Coder. This represents a pragmatic compromise between full parallelism and infrastructure compatibility.

**10. AR-to-diffusion conversion preserves pretrained knowledge** and is approximately 7x more compute-efficient than training equivalent dense models from scratch. I-DLM achieves quality matching with only ~5B tokens [^7^], and DiffuLLaMA converts LLaMA2-7B with <200B tokens [^6^].

**11. Diffusion models show dramatically superior length extrapolation on RepoQA.** When context exceeds training window (8K to 64K), Mercury-Coder-Small shows only ~15% performance decrease while Qwen3-8B drops by nearly 30%. At 4K tokens, DiffuCoder-7B maintains >30% accuracy where Llama-2 drops below 10% [^526^].

**12. LLaDA2.1 introduces Token-to-Token (T2T) editing**, enabling the model to retrospectively correct already-committed tokens during generation. Follow-up work (Token-to-Mask, T2M) shows remasking suspicious tokens outperforms overwriting, improving AIME 2025 scores by +13.33 points [^307^][^181^].

**13. Stable-DiffCoder's controlled experimental design** isolates the effect of diffusion training by holding architecture, data, and training pipeline constant — changing only the training objective. This design provides the strongest evidence that diffusion training itself (not data or architecture advantages) drives performance differences [^20^][^149^].

**14. The NAP paper identifies "ARness" as a critical metric** — diffusion models trained on sequential data converge to AR-like decoding patterns. On Dream-7B/GSM8K, forcing true parallel decoding drops accuracy from 78.0% (1024 steps) to 46.5% (256 steps), suggesting current benchmarks implicitly reward AR-like behavior [^429^].

**15. LLaDA2.1 achieves 892 TPS on HumanEval+ (quantized)** for its 100B Flash model, and 1,587 TPS on its 16B Mini model, using per-block FP8 quantization combined with Alpha-MoE megakernels [^164^][^424^].

**16. Gemini Diffusion shows significant gaps on reasoning and science benchmarks** compared to frontier AR models: GPQA Diamond 40.4% vs. Flash-Lite's 56.5% (-16.1pp), BIG-Bench Extra Hard 15.0% vs. 21.0% (-6.0pp), Global MMLU Lite 69.1% vs. 79.0% (-9.9pp) [^272^].

**17. Seed-Coder's data pipeline uses model-centric quality scoring** — a 1.3B Llama 2 regression model fine-tuned on DeepSeek-V2-Chat oracle scores across four dimensions (readability, modularity, clarity, reusability) — filtering the bottom 10% to produce 6T training tokens [^520^][^514^].

**18. All major open-source diffusion LLMs come from Chinese institutions**: Ant Group (LLaDA family), ByteDance/Tsinghua (Seed Diffusion, Stable-DiffCoder), and Renmin University (GSAI-ML). US contributions (Google DeepMind, Inception Labs, Apple) are closed-source or limited release [^4^][^24^][^11^].

**19. Diffusion models achieve 84.8% on fill-in-the-middle tasks** (Mercury Coder) vs. Flash-Lite's 60.1% — a 24.7pp advantage on this inherently non-sequential code task [^71^].

**20. Plan conditioning improves diffusion LLM reasoning by +11.6pp on GSM8K**, suggesting diffusion models fundamentally need external sequential guidance for complex multi-step reasoning tasks [^89^].

---

## Key Themes Across Dimensions

### Theme 1: Code Editing as the Uncontested Diffusion Advantage
Across every dimension where code editing is measured, diffusion models demonstrate decisive superiority. Stable-DiffCoder's 60.0% on CanItEdit vs. 50.5% for its AR counterpart [^83^] represents the single largest controlled-effect size favoring diffusion over autoregression in the entire literature. Mercury Coder's 24.7pp advantage on fill-in-the-middle tasks [^71^] confirms this is not limited to one model family. The mechanism is well-understood: code editing is inherently non-sequential (changing a function signature requires updating all callers simultaneously), which aligns precisely with diffusion's any-order generation capability [^83^][^160^]. The Seed Diffusion team's two-stage curriculum (mask-based followed by edit-based corruption) was specifically designed to break the "unmasked = correct" inductive bias that prevents autoregressive models from self-correction [^276^].

### Theme 2: The Block Diffusion Consensus
What began as a research curiosity — generating all tokens in parallel — has converged to a pragmatic compromise. Every production diffusion LLM now uses block diffusion (semi-autoregressive blocks with parallel intra-block generation): Gemini Diffusion, LLaDA2.0 (block size 32), Stable-DiffCoder (block size 4 for code), and Mercury Coder [^297^][^24^][^479^]. Block size 32 appears across multiple independent implementations as the optimal tradeoff between quality and throughput. This convergence is not temporary but structural — block diffusion is the only approach that enables KV cache compatibility, tensor parallelism, and streaming output while preserving some parallel generation benefit [^389^]. The research implication is clear: optimizing block diffusion (dynamic block sizes, better block scheduling, cross-block attention) matters more than pursuing fully parallel generation.

### Theme 3: RL as the Dominant Post-Training Paradigm
A consistent finding across Ant Group (EBPO), Apple (coupled-GRPO), and LLaDA 1.5 (VRPO) is that standard supervised fine-tuning provides only marginal gains for diffusion models due to train-test mismatch — diffusion models are trained on masked corruption but evaluated on sequential generation [^10^][^330^]. RL methods specifically designed for diffusion's non-autoregressive nature produce dramatically larger improvements: LLaDA2.0's preview-to-final gap of 45% on LiveCodeBench comes almost entirely from post-training [^162^]; DiffuCoder's coupled-GRPO achieves +4.4% EvalPlus with only 21K examples [^10^]; EBPO enables the first RL at 100B scale [^346^]. The insight is that diffusion models are "RL-shaped" — they benefit disproportionately from RL because their training objective does not align with their inference-time generation pattern.

### Theme 4: Chinese Open-Source Leadership vs. Western Closed-Source
Every major open-source diffusion LLM originates from Chinese institutions: Ant Group's LLaDA2.0/2.1 (100B, fully open-sourced under Apache 2.0 with complete toolchain) [^24^][^390^]; ByteDance/Tsinghua's Stable-DiffCoder and Seed Diffusion [^11^][^83^]; Renmin University's GSAI-ML group. US contributions — Google DeepMind's Gemini Diffusion (closed API, experimental), Inception Labs' Mercury Coder (closed-source, $50M funded) [^437^], Apple's DiffuCoder (open weights but limited) — are predominantly closed. This pattern mirrors DeepSeek's disruption of the AR landscape and suggests the open-source diffusion ecosystem will develop with Chinese institutional leadership.

### Theme 5: The Benchmark Selection Problem
Whether diffusion models "win" or "lose" depends entirely on which benchmarks are emphasized [^9^][^37^]. HumanEval and BigCodeBench show parity or slight diffusion advantage. LiveCodeBench and SWE-Bench show diffusion lagging. CanItEdit and RepoQA show diffusion winning decisively. The NAP paper provides the theoretical explanation: training data's sequential structure forces diffusion models into AR-like behavior, meaning they underperform on benchmarks that reward sequential reasoning (competitive programming) and overperform on benchmarks that reward parallel processing (editing, long context) [^429^]. This is not merely a matter of opinion — it is empirically demonstrable that different tasks genuinely favor different paradigms.

### Theme 6: Inference Acceleration as a Catch-Up Arms Race
The pace of inference optimization research for diffusion models is extraordinary. Fast-dLLM achieves 27.6x throughput improvement with training-free block-wise KV caching [^296^]; Elastic-Cache achieves 45.1x on long sequences; LLaDA2.1 reaches 1,587 TPS through per-block FP8 quantization and Alpha-MoE megakernels [^164^]; CAP training provides 2.1x speedup [^389^]. These are order-of-magnitude improvements arriving faster than new model architectures. The acceleration curve for diffusion is steeper than for AR due to catch-up effects and the richness of the unexplored optimization space — diffusion inference has many more degrees of freedom (block size, threshold scheduling, editing aggressiveness, remasking strategy) than autoregressive decoding.

---

## Contradictions and Tensions

### Tension 1: "Gap is Closed" vs. "Diffusion Lags"
Brendan O'Donoghue of Google DeepMind states "the gap between the two techniques is essentially closed in terms of benchmark performance" [^37^]. The "Beyond Autoregression" paper states "diffusion LLMs are not yet able to replace AR LLMs at the current stage" [^9^]. Both statements are factually defensible depending on benchmark weighting. HumanEval/BigCodeBench support O'Donoghue; LiveCodeBench/SWE-Bench support the Zhang et al. position. This is not a factual dispute but a values dispute about which capabilities matter most.

### Tension 2: Speed Claims vs. Controlled Reality
Vendor headline speed claims (Seed Diffusion: 2,146 tok/s, Gemini Diffusion: 1,479 tok/s, Mercury: 1,109 tok/s) are not directly comparable because they run on different hardware (H20 vs. H100 vs. TPU) under different conditions [^168^][^272^][^31^]. The only controlled comparison — SGLang on H20 GPUs — shows LLaDA2.0-flash-CAP at 535 TPS vs. AR baselines at 237-256 TPS, a 2.1x speedup that is impressive but far from the 5-10x headline claims [^389^]. The "Beyond Autoregression" paper further notes that "current open-source diffusion LLMs often remain substantially slower than comparable ARMs in practice" due to lack of optimized inference infrastructure [^784^].

### Tension 3: T2T Editing vs. T2M Remasking
The LLaDA2.1 paper advocates token overwriting (T2T) as its core editing mechanism [^330^]. Follow-up work by Lin Yao et al. argues that remasking to [MASK] (T2M) is fundamentally more reliable — avoiding context pollution and correction inertia — and achieves +13.33 points on AIME 2025 [^307^]. The conflict centers on whether detection and correction should be coupled (T2T) or decoupled (T2M). This is an active research dispute with significant practical implications for diffusion model design.

### Tension 4: Discrete vs. Continuous Diffusion
Discrete diffusion (masked diffusion, as in LLaDA and MD4) dominates at scale empirically — LLaDA reaches 100B parameters, MDLM achieves strong results [^348^][^24^]. Continuous diffusion (LangFlow, ELF) offers theoretical advantages in editability and classifier-free guidance compatibility but has not yet matched discrete performance at scale [^7^]. This tension is unresolved and represents a fundamental architectural choice with no clear consensus.

### Tension 5: High ARness as Feature vs. Bug
The NAP paper demonstrates that diffusion models trained on sequential data develop high ARness (Global ARness@1 ~0.92 for Dream), meaning their "most confident tokens are almost always the next tokens in the sequence" [^429^]. This enables competitive accuracy on standard benchmarks but largely defeats the purpose of parallel generation. High ARness is simultaneously what makes diffusion models practical (they can compete on existing benchmarks) and what limits their theoretical advantage (they are not truly exploiting parallel decoding). Resolving this tension requires training data restructuring — using non-sequential supervision — rather than architectural changes.

### Tension 6: Preview vs. Final Model Attribution
LLaDA2.0-flash-preview scores 23.33 on AIME 2025 vs. 60.00 for the final model — a massive gap created entirely by post-training [^162^]. This creates ambiguity about what to attribute to "diffusion" vs. "post-training recipe." The preview represents the raw diffusion capability; the final represents the complete system. Both are legitimate objects of study, but conflating them leads to over- or under-attribution of diffusion's inherent capabilities.

### Tension 7: Enterprise Adoption Lag Despite Benchmark Competitiveness
Diffusion code models achieve benchmark parity (or advantage on editing tasks) but have minimal enterprise adoption compared to GitHub Copilot. The barrier is not model quality but integration — GitHub Copilot is deeply embedded in VS Code and JetBrains with real-time ghost text and seamless UX, while diffusion models are predominantly API-only [^11^][^4^]. Ant Group's NES system demonstrates that with proper UX integration (Tab-key workflow), diffusion models can serve 20,000+ developers [^4^], but this is internal-only. The 2026 timeframe is predicted as decisive: either diffusion achieves IDE integration parity or the "diffusion will replace AR" narrative requires revision [^12^].

---

## Most Compelling Narrative Threads for the Report

### Thread 1: "The Editing Paradigm Shift" — Code Editing as Diffusion's Beachhead
This narrative positions code editing (not general text generation) as the entry point where diffusion models will first achieve mainstream adoption. The evidence is overwhelming: CanItEdit (+18.8% relative improvement) [^83^], fill-in-the-middle (+24.7pp) [^71^], and Stable-DiffCoder's controlled comparison isolating diffusion training as the causal factor [^20^]. Code editing is inherently non-sequential — changing a function signature requires updating all callers simultaneously — making it the perfect match for diffusion's any-order generation. The commercial opportunity is not competing with Copilot on completion but replacing its editing workflow entirely. This thread connects Insight 1 (code editing as killer app), HC-6 (diffusion excels at editing), and the ByteDance Seed findings on edit-based corruption [^276^].

### Thread 2: "The Conversion Moat" — How AR Pretraining Investment Transfers
Organizations with strong pretrained AR models gain structural advantages in the diffusion race because AR-to-diffusion conversion (LLaDA2.0's WSD, ByteDance's Stable-DiffCoder, DeepMind's AR2Diff) is more efficient than training from scratch [^24^][^62^][^160^]. This means the billions already invested in AR pretraining don't become stranded assets — they transfer. The implication is that the diffusion landscape will be dominated by the same organizations that dominate AR (Ant Group, ByteDance, Google), making it difficult for pure-play diffusion startups to compete. This thread connects Insight 2, HC-3, and the finding that I-DLM achieves quality matching with only ~5B tokens [^7^].

### Thread 3: "The RL Revolution" — Post-Training as the Real Differentiator
The narrative that diffusion models are primarily about speed is wrong. The data reveals that post-training RL produces the largest quality gains: LLaDA2.0's 45% preview-to-final improvement comes from SFT + CAP + DPO, not from diffusion conversion itself [^162^]. Organizations mastering RL for diffusion (Ant Group with EBPO, Apple with coupled-GRPO) will outperform those focused only on architecture or speed. EBPO's innovation — using the ELBO as a tractable proxy for sequence-level log-likelihoods — solved the fundamental obstacle to scaling RL for diffusion models [^330^][^346^]. This thread connects Insight 3, HC-4, and the finding that DiffuCoder's coupled-GRPO achieves +4.4% on EvalPlus with only 21K examples [^10^].

### Thread 4: "The China Open-Source Inversion" — How the Diffusion Landscape Mirrors DeepSeek
All major open-source diffusion LLMs come from Chinese institutions while US contributions remain closed-source. Ant Group's LLaDA2.0/2.1 (100B, Apache 2.0, complete toolchain including dFactory, dInfer, and SGLang integration) [^24^][^390^] represents the most comprehensive open-source release in the diffusion space. ByteDance/Tsinghua's Stable-DiffCoder provides the strongest controlled experimental evidence for diffusion superiority [^83^]. This is the inverse of the AR landscape where OpenAI, Anthropic, Meta, and Google lead. The pattern suggests the open-source diffusion ecosystem will develop with Chinese institutional leadership, potentially disrupting the closed-source Western providers as DeepSeek disrupted the AR landscape. This thread connects Insight 4, HC-8, and the SIA-Lab institutional collaboration structure [^11^][^136^].

### Thread 5: "The Paradigm Boundary Blur" — AR and Diffusion Converging
The architectural boundary between AR and diffusion is becoming indistinct. Block diffusion is already a hybrid (sequential inter-block, parallel intra-block) [^297^]. The NAP paper shows that diffusion models trained on sequential data become AR-like [^429^]. The A3 paper shows AR models can achieve any-order generation [^12^]. Both camps adopt each other's techniques. The future may be hybrid architectures that combine the best of both paradigms — making the AR-vs-diffusion debate obsolete. This thread connects Insight 8, the block diffusion consensus (HC-5), and the ARness research trajectory [^429^].

### Thread 6: "The Benchmark Rorschach Test" — Why the Diffusion Debate Cannot Be Resolved by Numbers Alone
The same evidence produces opposite conclusions depending on which benchmarks are weighted. HumanEval shows parity. LiveCodeBench shows gaps. CanItEdit shows decisive diffusion advantage. RepoQA shows diffusion superiority on long context. The NAP paper explains why: different tasks genuinely favor different paradigms — sequential reasoning tasks favor AR, parallel structure tasks favor diffusion [^429^]. This means the diffusion-vs-AR question cannot be answered with a single score. The practical implication is that the "winner" will be determined by which use cases dominate the market, not by aggregate benchmark rankings. This thread connects Insight 6, the benchmark fairness analysis in dim10, and the ARness-Accuracy tradeoff [^429^].

### Thread 7: "The Integration Gap" — Why IDE UX Matters More Than Model Quality
Despite benchmark parity, diffusion code models have minimal enterprise adoption because they lack native IDE integration. GitHub Copilot's competitive advantage is not model quality but its seamless embedding in VS Code and JetBrains — ghost text, real-time suggestions, one-tab acceptance. Ant Group's NES system demonstrates that with proper UX (Tab-key workflow), diffusion models can achieve real-world usage at scale (20,000+ developers) [^4^]. The next breakthrough may come from a startup building the "Copilot wrapper" for diffusion rather than from model improvements. The 2026 timeframe is decisive: three commercial providers now exist (Mercury, Gemini Diffusion, Seed Diffusion), open-source models have reached 100B parameters, and speed optimization research is hitting order-of-magnitude improvements [^437^][^31^]. If diffusion cannot achieve meaningful market share in 2026 despite these advances, the narrative may need revision. This thread connects Insight 9, Insight 10, and the enterprise adoption analysis from dim11.

---

*Synthesis compiled from seven research artifacts covering Google DeepMind (dim01), Ant Group LLaDA2.0 (dim02), Ant Group LLaDA2.1 (dim03), ByteDance Seed (dim05), Code benchmarks (dim10), cross-dimension insights, and cross-verification confidence tiers. All citation indices preserved from source documents.*
