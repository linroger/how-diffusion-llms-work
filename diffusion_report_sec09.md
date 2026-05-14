## 9. Benchmarks and Performance Evaluation

The question of whether diffusion language models (DLMs) can match or exceed autoregressive (AR) models on code-related tasks does not yield a single answer. Instead, the conclusion depends almost entirely on which benchmarks are consulted. This chapter synthesizes performance data across all major evaluation suites and model families, revealing a systematic pattern: diffusion models achieve parity or superiority on benchmarks that reward global context understanding and parallelizable tasks, while they exhibit consistent gaps on benchmarks that demand sequential step-by-step reasoning. This *benchmark selection effect* means that both advocates and critics of the diffusion paradigm can cite credible, peer-reviewed evidence to support their positions—a Rorschach test that complicates any simple verdict on diffusion's readiness for production code workflows.

### 9.1 The Benchmark Selection Effect

The most comprehensive controlled comparison of diffusion and autoregressive code models to date, Zhang et al.'s "Beyond Autoregression" (2025), evaluated nine diffusion LLMs from six families against four AR baselines across HumanEval, MBPP, LiveCodeBench, and RepoQA [^5^]. The study's aggregate findings immediately reveal the benchmark dependency: on HumanEval, diffusion models averaged 66.7% versus 71.3% for AR models—a narrow 4.6 percentage point (pp) gap—with the best diffusion model (Gemini Diffusion at 89.6%) actually surpassing the best AR baseline (Seed-Coder-8B-Instruct at 84.8%) [^9^]. On MBPP, the averages essentially inverted: diffusion averaged 61.2% versus 60.8% for AR, with Seed-Diffusion-Preview reaching 79.4% against Seed-Coder's 70.8% [^9^]. Yet on LiveCodeBench v6, diffusion models averaged 14.9% versus 18.9% for AR—a 27% relative deficit—and on the broader v1-v6 corpus, the gap widened further to 19.1% versus 25.8% [^9^].

This divergence creates three distinct benchmark tiers. In the *parity tier*—HumanEval, MBPP, and BigCodeBench—diffusion models match or exceed their AR counterparts. Gemini Diffusion scores 45.4% on BigCodeBench, statistically tying Flash-Lite at 45.8% [^37^], while Mercury-Coder-Small achieves 45.5% [^71^] and Seed-Diffusion-Preview reaches 45.4% [^482^]. In the *AR-advantage tier*—LiveCodeBench and SWE-Bench—diffusion models trail consistently. The SWE-Bench Verified gap is particularly stark: Gemini Diffusion scores 22.9% against Flash-Lite's 28.5% [^272^], a 5.6pp deficit that reflects the benchmark's demands for multi-step agentic reasoning. In the *diffusion-advantage tier*—CanItEdit and RepoQA—diffusion models demonstrate decisive superiority. Stable-DiffCoder-8B-Instruct achieves 60.0% on CanItEdit versus Seed-Coder-8B-Instruct's 50.5% [^83^], an 18.8% relative improvement, while on RepoQA's long-context retrieval task, Mercury-Coder-Small exhibits approximately 15% performance degradation when extrapolating from 8K to 64K tokens, compared to Qwen3-8B's nearly 30% drop [^526^].

The benchmark selection effect operates through two mechanisms. First, most existing code benchmarks were designed for left-to-right generation. HumanEval provides a function signature and docstring, then asks the model to complete the body in sequential order—a task structure that aligns with AR generation patterns [^9^]. Second, the nature of the task itself determines which paradigm excels. Code editing (CanItEdit) is fundamentally non-sequential: modifying a function signature requires updating all call sites, a change pattern that benefits from any-order token generation [^83^]. Competitive programming (LiveCodeBench) requires sequential algorithmic reasoning—designing a solution step by step, which stresses the very capability where diffusion models face structural challenges.

The result is a landscape where the choice of evaluation suite predetermines the conclusion. A researcher emphasizing HumanEval and BigCodeBench can credibly claim that "the gap... is essentially closed in terms of benchmark performance" [^37^]. A researcher emphasizing LiveCodeBench and SWE-Bench can equally credibly conclude that "diffusion LLMs are not yet able to replace AR LLMs at the current stage" [^9^]. Both statements are factually correct within their respective benchmark frames. Figure 9.1 visualizes this divergence across seven major benchmarks, quantifying the performance gap in percentage points for each evaluation suite.

![Figure 9.1: The Benchmark Selection Effect—diffusion vs. autoregressive performance gaps across seven code evaluation benchmarks. Positive values indicate diffusion advantage; negative values indicate AR advantage. Abbreviations: GD = Gemini Diffusion, SC = Seed-Coder, SD = Seed-Diffusion, SDC = Stable-DiffCoder, FL = Flash-Lite, Q3 = Qwen3-8B, MC = Mercury-Coder.](fig_benchmark_selection_effect.png)

The figure reveals a striking pattern: diffusion models lead on four of seven benchmarks (HumanEval, MBPP, CanItEdit, RepoQA) and trail on two (BigCodeBench effectively ties, SWE-Bench). The magnitude of advantages on CanItEdit (+9.5pp) and RepoQA (+15.0pp less degradation) exceeds the magnitude of disadvantages on SWE-Bench (-5.6pp), suggesting that when diffusion models excel, they excel by larger margins than when they lag. The LiveCodeBench result (+4.9pp for Gemini Diffusion specifically, though negative for diffusion averages) illustrates the model-dependency within benchmarks: while diffusion averages trail, the best individual diffusion model (Gemini Diffusion at 30.9%) can still outperform some AR competitors (Qwen3-8B at 26.0%) [^9^].

### 9.2 Code Generation Benchmarks Deep Dive

#### 9.2.1 HumanEval: The Parity Baseline

HumanEval remains the most widely cited code generation benchmark, comprising 164 hand-written Python programming problems with test-based evaluation. On this benchmark, diffusion models have achieved full competitive parity with AR models, with several individual models establishing new performance records.

Table 9.1 presents a comprehensive comparison across model families, architectures, and parameter scales.

**Table 9.1 — Comprehensive Code Generation Benchmark Comparison Across Model Families**

| Model | Family | Size | Architecture | HumanEval | MBPP | LiveCodeBench v6 | BigCodeBench | CanItEdit |
|-------|--------|------|-------------|-----------|------|------------------|-------------|-----------|
| Gemini Diffusion | Google DeepMind | — | Block diffusion | 89.6% [^272^] | 62.9% [^272^] | 30.9% [^272^] | 45.4% [^272^] | — |
| Seed-Diffusion-Preview | ByteDance | — | Diffusion | — | 79.4% [^9^] | 33.7% (v1-v6) [^9^] | 45.4% [^482^] | 54.3% [^793^] |
| Stable-DiffCoder-8B-Instruct | ByteDance | 8B | Diffusion (CPT) | 86.6% [^83^] | 77.6% [^83^] | 23.5% [^83^] | — | 60.0% [^83^] |
| Mercury-Coder-Small | Inception Labs | — | Diffusion | 86.0% [^9^] | — | 22.9% [^9^] | 45.5% [^71^] | — |
| Mercury-Coder-Mini | Inception Labs | — | Diffusion | 88.0% [^85^] | — | — | — | — |
| LLaDA2.0-flash | Ant Group | 6B/100B | Masked diffusion | 94.51% [^83^] | 88.29% [^83^] | 42.29% [^83^] | — | — |
| Dream-Coder-v0-Instruct | Huawei/HKU | 7B | Adaptive diffusion | 76.2% [^9^] | — | 21.4% [^3^] | 21.4% [^3^] | — |
| DiffuCoder-7B-cpGRPO | Apple/HKU | 7B | Masked diffusion | 69.5% [^9^] | — | — | 40.4% [^10^] | — |
| Seed-Coder-8B-Instruct | ByteDance | 8B | Autoregressive | 84.8% [^9^] | 70.8% [^9^] | 24.7% [^83^] | — | 50.5% [^83^] |
| Qwen3-8B | Alibaba | 8B | Autoregressive | — | — | 26.0% (v6) / 42.3% (v1-v6) [^9^] | — | 45.7% [^9^] |
| Flash-Lite | Google | — | Autoregressive | — | — | — | 45.8% [^37^] | — |
| DeepSeek-Coder-6.7B-Instruct | DeepSeek | 6.7B | Autoregressive | 77.4% [^9^] | — | — | — | — |

The table reveals several important patterns. At the top end, LLaDA2.0-flash achieves 94.51% on HumanEval—the highest score recorded by any diffusion model—using only 6B active parameters within a 100B total parameter mixture-of-experts (MoE) architecture [^83^]. This surpasses not only all diffusion competitors but also Qwen3-30B at 93.29% [^83^], demonstrating that parameter count alone does not determine performance. Gemini Diffusion's 89.6% represents the best score among dense (non-MoE) diffusion architectures, exceeding Seed-Coder-8B-Instruct by 4.8pp [^9^]. Among open-source diffusion models, Stable-DiffCoder-8B-Instruct at 86.6% surpasses its AR counterpart Seed-Coder-8B-Instruct (84.8%) in a controlled comparison using identical training data and model architecture [^83^], providing perhaps the cleanest evidence that diffusion training itself can improve code generation quality.

The distribution of scores also reveals a quality hierarchy within diffusion models. Closed-source or commercially deployed models (Gemini Diffusion, Mercury-Coder, Seed-Diffusion, LLaDA2.0-flash) cluster in the 86–95% range, while open-source models (Dream-Coder at 76.2%, DiffuCoder at 69.5%) trail by 10–20pp [^9^]. This gap likely reflects training data quality and scale rather than architectural limitations, suggesting that diffusion code models are more data-hungry than their AR equivalents—a pattern consistent with the broader finding that diffusion training requires more tokens to reach equivalent loss values [^429^].

#### 9.2.2 MBPP: Diffusion Competitiveness Confirmed

The Mostly Basic Python Programming (MBPP) benchmark, with 974 crowd-sourced Python problems, confirms the HumanEval pattern. Diffusion models averaged 61.2% versus 60.8% for AR models in the "Beyond Autoregression" study—a statistically negligible difference [^9^]. The standout result is Seed-Diffusion-Preview at 79.4%, surpassing Seed-Coder-8B-Instruct's 70.8% by 8.6pp [^9^]. Stable-DiffCoder-8B-Instruct achieves 77.6% [^83^], while Gemini Diffusion scores 62.9% [^272^]—notably lower than other top diffusion models, suggesting that Google's training prioritization may differ from ByteDance's code-focused approach. LLaDA2.0-flash reaches 88.29% [^83^], again demonstrating the quality of Ant Group's post-training pipeline.

The MBPP results are particularly significant because MBPP problems are more diverse in difficulty and domain than HumanEval's curated set. The fact that diffusion models match AR performance on this broader corpus undermines any claim that diffusion success is limited to narrow, memorization-prone evaluation suites.

#### 9.2.3 LiveCodeBench: The Persistent Competitive Programming Gap

LiveCodeBench represents the most consistent and concerning weakness for diffusion code models. This contamination-free benchmark (containing problems released after model training cutoffs) measures true generalization on competitive programming tasks [^9^]. On LiveCodeBench v6, diffusion models averaged 14.9% against 18.9% for AR models—a 4.0pp absolute gap representing a 27% relative disadvantage [^9^]. The gap is even wider on the v1-v6 aggregate: 19.1% versus 25.8% [^9^].

Several factors explain this gap. First, competitive programming requires chain-of-thought (CoT) reasoning—designing algorithms step by step, considering edge cases, and iteratively refining solutions. The NAP paper (Li et al., 2026) demonstrates that diffusion language models trained on standard sequential CoT data converge to autoregressive-like decoding patterns, with Global ARness@1 scores around 0.92 for Dream models [^429^]. When these models are forced to use true parallel decoding, reasoning accuracy collapses: on Dream-7B evaluated on GSM8K, accuracy drops from 78.0% at 1,024 diffusion steps to 46.5% at 256 steps [^429^]. This *AR-collapse* phenomenon means diffusion models underperform precisely when their parallel advantage is most needed.

Second, LiveCodeBench's task structure rewards sequential reasoning. Each problem requires reading a complex specification, identifying the algorithmic approach, implementing it, and verifying against hidden test cases. The logical dependencies between these steps create a sequential reasoning chain that diffusion's any-order generation does not naturally optimize for [^429^]. Third, open-source diffusion models have had lower exposure to competitive programming content, as the "Beyond Autoregression" authors note that "open-source diffusion LLMs lag behind closed-source counterparts, possibly due to training data composition" [^9^].

However, the LiveCodeBench gap is narrowing. Gemini Diffusion at 30.9% outperforms Qwen3-8B's 26.0% on v6 [^9^], demonstrating that well-resourced diffusion models can compete. Stable-DiffCoder at 23.5% essentially matches Seed-Coder-8B-Instruct's 24.7%—a gap of only 1.2pp [^83^]. And the trajectory is positive: Dream-v0 scored 13.3% on v1-v6 in April 2025, while Dream-Coder reached 24.8% by July 2025—nearly doubling in three months [^9^].

#### 9.2.4 BigCodeBench: Real-World Parity

BigCodeBench evaluates "challenging, real-world coding problems with rich context and tool-like function calls" [^83^]—tasks that require integrating multiple libraries, understanding complex APIs, and generating production-quality code. On this benchmark, diffusion models demonstrate near-perfect parity with AR models. Gemini Diffusion scores 45.4% versus Flash-Lite's 45.8%—a 0.4pp difference that falls within evaluation noise [^37^]. Mercury-Coder-Small achieves 45.5% [^71^], and Seed-Diffusion-Preview reaches 45.4% [^482^]. DiffuCoder with coupled-GRPO reinforcement learning achieves 40.4%, a 4.7pp improvement over its pre-RL baseline of 35.7% [^10^].

The BigCodeBench results are significant because this benchmark most closely approximates real-world developer workflows—integrating external libraries, handling edge cases, and producing complete functional programs rather than isolated algorithmic solutions. The parity here suggests that for production code generation tasks, diffusion models are ready for practical deployment. As the VentureBeat analysis concludes, "the gap between the two techniques is essentially closed" on real-world code generation [^37^].

### 9.3 Code Editing and Specialized Benchmarks

#### 9.3.1 CanItEdit: Diffusion's Decisive Advantage

CanItEdit, a benchmark for code editing capability, represents diffusion models' most decisive victory over autoregressive counterparts. Code editing is fundamentally different from code completion: rather than generating a program sequentially from a prompt, editing requires understanding an existing codebase, identifying what needs to change, and making targeted modifications that preserve functionality while altering behavior. This non-sequential task structure aligns precisely with diffusion's any-order generation capability.

**Table 9.2 — CanItEdit and Code Editing Benchmark Comparison**

| Model | Architecture | Size | CanItEdit pass@1 | Aider (tries=2) | Key Strength |
|-------|-------------|------|-----------------|-----------------|--------------|
| Stable-DiffCoder-8B-Instruct | Diffusion (CPT) | 8B | **60.0%** [^83^] | 54.9% [^83^] | Random masking trains edit patterns |
| Seed-Diffusion-Preview | Diffusion | — | 54.3% [^793^] | — | Two-stage curriculum (mask→edit) |
| Qwen2.5-Coder-14B-Instruct | Autoregressive | 14B | 52.9% [^83^] | — | Larger parameter count |
| Seed-Coder-8B-Instruct | Autoregressive | 8B | 50.5% [^83^] | 57.1% [^83^] | AR counterpart to Stable-DiffCoder |
| Yi-Coder-9B-Chat | Autoregressive | 9B | 50.5% [^83^] | — | General coding model |
| DeepSeek-Coder-33B-Instruct | Autoregressive | 33B | 46.2% [^83^] | — | 4x larger than Stable-DiffCoder |
| Qwen3-8B | Autoregressive | 8B | 45.7% [^83^] | 55.6% [^83^] | Strong general model |

Stable-DiffCoder-8B-Instruct's 60.0% CanItEdit score surpasses all competitors, including models four times larger (DeepSeek-Coder-33B at 46.2%) [^83^]. The 18.8% relative improvement over Seed-Coder-8B-Instruct (60.0% versus 50.5%) is remarkable given that the two models share identical architecture and training data—the difference is purely the diffusion training paradigm [^83^]. The authors hypothesize that "random masking and reconstruction inherently train the model on edit- and infill-like patterns, enabling it to better exploit editing supervision" [^83^]. During diffusion training, the model learns to reconstruct randomly masked token spans within existing code, a process structurally identical to code editing: given partial context, determine what belongs in the missing region.

Seed-Diffusion-Preview's 54.3% [^793^] demonstrates that curriculum learning—progressing from mask-based training to edit-based training—can boost editing capability by 4.8pp over AR baselines [^793^]. On the Aider multi-turn editing benchmark, Stable-DiffCoder achieves 54.9% (tries=2), slightly trailing Seed-Coder's 57.1% but comparable to Qwen3-8B's 55.6% [^83^]—showing that while diffusion dominates single-turn editing, multi-turn iterative editing remains competitive.

Mercury Coder extends this editing advantage to fill-in-the-middle (FIM) tasks, achieving 84.8% on FIM benchmarks versus Flash-Lite's 60.1%—a 24.7pp advantage that represents one of the largest diffusion wins across any code evaluation [^71^]. FIM tasks, which require completing code in the middle of existing programs, are structurally identical to the masked reconstruction objective used in diffusion training.

#### 9.3.2 RepoQA: Superior Length Extrapolation

RepoQA's "Searching Needle Function" task evaluates long-context code understanding across 500 problems drawn from 50 repositories [^526^]. This benchmark reveals a structural advantage for diffusion models that has implications for repository-level development tools.

**Table 9.3 — Long-Context and Specialized Benchmark Performance**

| Model | Architecture | RepoQA 4K | RepoQA 64K | Degradation (8K→64K) | SWE-Bench Verified |
|-------|-------------|-----------|------------|---------------------|-------------------|
| DiffuCoder-7B-cpGRPO | Diffusion | >30% [^526^] | — | Minimal | — |
| Mercury-Coder-Small | Diffusion | — | — | ~15% [^526^] | — |
| Qwen3-8B | Autoregressive | — | — | ~30% [^526^] | — |
| Llama-2-7B-Chat | Autoregressive | <10% [^526^] | — | Severe | — |
| Gemini Diffusion | Diffusion | — | — | — | 22.9% [^272^] |
| Flash-Lite | Autoregressive | — | — | — | 28.5% [^37^] |
| Claude Opus 4.6 | Autoregressive | — | — | — | 80.6% [^803^] |

RepoQA results show that diffusion models maintain significantly higher retrieval accuracy as context length increases. At 4K tokens input, AR model (Llama-2-7B-Chat-HF) retrieval accuracy drops below 10%, while DiffuCoder-7B-cpGRPO maintains above 30% [^526^]. When extrapolating beyond the training window (8K to 64K), Mercury-Coder-Small shows only approximately 15% performance decrease, while Qwen3-8B drops by nearly 30%—twice the degradation rate [^526^]. The "Beyond Autoregression" authors conclude that "diffusion LLMs remain relatively robust as context length increases, whereas the performance of AR LLMs declines rapidly" [^526^].

This advantage is hypothesized to stem from diffusion models' bidirectional attention during training. While AR models attend only to preceding tokens, diffusion models attend to all positions simultaneously during the denoising process, learning representations that are less position-dependent and more robust to context expansion. For repository-level coding tasks—where relevant functions may be defined thousands of tokens away from the insertion point—this structural advantage could prove decisive.

#### 9.3.3 SWE-Bench: The Agentic Evaluation Gap

SWE-Bench Verified represents the most demanding code evaluation, requiring models to resolve real GitHub issues across diverse Python repositories. The benchmark demands multi-step reasoning: understanding issue descriptions, exploring codebase structure, locating relevant files, diagnosing root causes, and generating correct patches. Gemini Diffusion scores 22.9% versus Flash-Lite's 28.5% [^272^]—a 5.6pp gap that appears concerning.

However, this comparison requires careful interpretation. The Gemini Diffusion evaluation uses "non-agentic evaluation (single turn edit only), max prompt length of 32K" [^272^]. SWE-Bench leaderboards show that agentic models with iterative feedback loops vastly outperform single-turn approaches—Claude Opus 4.6 achieves 80.6% and Gemini 3.1 Pro reaches 80.8% in agentic mode [^803^], compared to mid-20s percentages for single-turn models. No diffusion model has yet been evaluated on SWE-Bench in an agentic setting. Given diffusion models' strengths in global planning and single-pass solution generation, it is plausible that iterative agentic workflows could significantly improve their SWE-Bench performance. The current gap may reflect evaluation methodology rather than fundamental capability limitations.

### 9.4 Root Cause Analysis

#### 9.4.1 The AR-Collapse Phenomenon

The NAP paper (Li et al., February 2026) provides the most compelling theoretical explanation for the benchmark-dependent performance pattern [^429^]. The paper demonstrates that diffusion language models trained on standard sequential data—code corpora organized in left-to-right reading order—converge to autoregressive-like decoding patterns, a phenomenon the authors term *AR-collapse*. Even when given the freedom to generate tokens in any order, these models exhibit high ARness (Global ARness@1 ~ 0.92 for Dream models), meaning "their most confident tokens are almost always the next tokens in the sequence" [^429^].

This behavior has a critical implication: diffusion models sacrifice their parallel generation advantage in order to maintain accuracy. When forced to use true parallel decoding (low ARness), reasoning accuracy collapses. On Dream-7B evaluated on GSM8K, accuracy drops from 78.0% at 1,024 steps to 46.5% at 256 steps—a 31.5pp collapse [^429^]. The authors attribute this to a dependency on "sequential stability": standard supervision creates reasoning chains where each step depends on the previous one, and when the model is forced to commit to multiple positions simultaneously, these chains break.

The NAP paper further demonstrates that this is a training data problem, not an architectural limitation. By restructuring supervision as "multiple independent reasoning trajectories"—training on data where reasoning steps are less sequentially dependent—the authors achieve a +14.4% improvement on GSM8K under parallel decoding [^429^]. This data-centric solution suggests that the LiveCodeBench gap could narrow significantly with training data specifically designed to support parallel reasoning.

#### 9.4.2 Benchmark Design Bias

The AR-collapse phenomenon interacts with benchmark design to produce the selection effect observed in Section 9.1. Most code benchmarks were created before diffusion language models existed and implicitly assume left-to-right generation. HumanEval and MBPP present function signatures and ask models to complete bodies sequentially—a task structure that naturally favors AR generation [^9^]. Research on PythonSaga notes that "more than 80% of the problems [in HumanEval/MBPP] are perceived as easy" and "existing benchmarks lack a comprehensive evaluation of their diversity in terms of programming concepts and difficulty level" [^885^]. Easy problems do not stress non-sequential reasoning capabilities.

LiveCodeBench requires step-by-step algorithmic reasoning precisely because competitive programming problems demand sequential logical chains. The gap between diffusion and AR is largest on this benchmark because it most strongly rewards the sequential reasoning that current diffusion training does not optimize for [^9^]. Conversely, CanItEdit rewards non-sequential thinking—modifying code requires understanding global context and making targeted changes, which aligns with diffusion's bidirectional attention. The one benchmark type where diffusion consistently wins is the one type that is fundamentally non-sequential.

The ARness-accuracy tradeoff creates an incentive problem. Diffusion models can achieve competitive accuracy on sequential benchmarks by mimicking AR behavior (high ARness), but this defeats the purpose of parallel generation. Conversely, forcing low-ARness parallel decoding collapses reasoning accuracy [^429^]. Current benchmarks reward high-ARness behavior, creating pressure for diffusion models to become "AR models in diffusion clothing" rather than developing genuinely parallel reasoning capabilities.

#### 9.4.3 Toward Diffusion-Native Benchmarks

The benchmark selection effect implies that resolving the diffusion-vs-AR debate requires new evaluation suites designed specifically for non-sequential code tasks. Several directions are emerging. Fill-in-the-middle benchmarks, where Mercury Coder already demonstrates a 24.7pp advantage over Flash-Lite [^71^], directly measure the capability diffusion training optimizes for. Repository-level editing benchmarks that require cross-file modifications would stress-test the global context understanding where diffusion shows superior length extrapolation [^526^]. Multi-turn agentic coding tasks would evaluate whether diffusion models' global planning capabilities translate to iterative software engineering workflows [^803^].

The NAP paper's data-centric approach—restructuring training data to support parallel reasoning—suggests a complementary path: benchmarks that explicitly measure performance under varying degrees of parallelism [^429^]. Current benchmarks implicitly reward AR-like behavior; benchmarks that reward low-ARness parallel decoding would incentivize the development of truly parallel reasoning capabilities. CRUXEval, which tests code execution reasoning through input and output prediction tasks, already shows promise in this direction: Stable-DiffCoder outperforms Seed-Coder on Output-CoT (60.0% versus 54.8%) because "the inputs and outputs are inherently structured rather than strictly following left-to-right causal logic" [^83^]. EndoCoT-style reasoning benchmarks that require explicit step-by-step reasoning chains during generation could push diffusion models to develop better internal reasoning mechanisms without sacrificing parallel decoding [^857^].

The evidence suggests a task-dependent conclusion rather than a paradigm-level verdict. Diffusion models are not universally superior or inferior to AR models; they are structurally better suited to tasks requiring global context understanding, parallel pattern completion, and non-sequential modification—precisely the tasks that dominate real-world software engineering. The competitive programming gap, while real, may narrow as training data improves and as diffusion-native reasoning techniques (EndoCoT, NAP-style data restructuring) mature. For code editing, repository-level context retrieval, and fill-in-the-middle completion, diffusion models already demonstrate decisive advantages that are unlikely to be reversed by incremental AR improvements. The question is not whether diffusion will replace AR for code, but for which code tasks—and the benchmark data provides an increasingly clear map of the dividing line.
