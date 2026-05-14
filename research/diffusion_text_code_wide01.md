## Facet: Google DeepMind's Diffusion Model Research for Text and Code Generation

---

### Key Findings

#### 1. Gemini Diffusion: The Flagship Experimental Model

- **Released**: May 20, 2025 at Google I/O 2025 [^50^][^51^][^55^]. The model was announced "quietly" without keynote stage time but quickly became known as the "sleeper hit" of the event [^55^][^90^].

- **Architecture**: Uses a diffusion-based approach for text generation instead of traditional autoregressive token-by-token prediction. Starts with random noise and iteratively refines it into coherent text through multiple denoising steps [^37^][^50^]. Employs **bidirectional attention** in the denoiser, allowing tokens to attend to future tokens within the same generation block [^37^][^39^]. Uses **block diffusion** with compact block sizes (typically 32 tokens) for production deployment [^31^].

- **Speed Claims**: 
  - 1,479 tokens/second (excluding overhead) [^91^][^172^]
  - Up to 2,000 tokens/second on programming tasks (including overhead like tokenization, prefill, and safety checks) [^91^][^108^]
  - Initial latency as low as 0.84 seconds [^91^][^172^]
  - ~5x faster than Gemini 2.0 Flash Lite [^53^][^64^]
  - During demos, video had to be slowed down to make generation watchable [^91^]

- **Performance Benchmarks** (vs. Gemini 2.0 Flash-Lite) [^37^][^172^]:

| Benchmark | Type | Gemini Diffusion | Gemini 2.0 Flash-Lite |
|---|---|---|---|
| LiveCodeBench (v6) | Code | 30.9% | 28.5% |
| BigCodeBench | Code | 45.4% | 45.8% |
| LBPP (v2) | Code | 56.8% | 56.0% |
| SWE-Bench Verified* | Code | 22.9% | 28.5% |
| HumanEval | Code | 89.6% | 90.2% |
| MBPP | Code | 76.0% | 75.8% |
| GPQA Diamond | Science | 40.4% | 56.5% |
| AIME 2025 | Mathematics | 23.3% | 20.0% |
| BIG-Bench Extra Hard | Reasoning | 15.0% | 21.0% |
| Global MMLU (Lite) | Multilingual | 69.1% | 79.0% |

*Non-agentic evaluation (single turn edit only), max prompt length of 32K.

- **Key Insight**: Gemini Diffusion excels at coding and math tasks (where non-local consistency matters) but underperforms on reasoning, scientific knowledge, and multilingual benchmarks [^37^][^91^]. As Brendan O'Donoghue stated: "the gap between the two techniques is essentially closed in terms of benchmark performance, at least at the relatively small sizes we have scaled up to. In fact, there may be some performance advantage for diffusion in some domains where non-local consistency is important, for example, coding and reasoning" [^37^].

- **Availability**: Currently available only as an **experimental demo** through a waitlist at deepmind.google/models/gemini-diffusion [^172^][^49^]. No public API, no model weights released, no announced plan to open-source [^173^].

#### 2. AR2Diff: Transfer Learning from Autoregressive to Diffusion

- **Paper**: "Transfer Learning for Text Diffusion Models" (arXiv:2401.17181, January 30, 2024) [^29^][^57^][^62^]

- **Authors**: Kehang Han, Kathleen Kenealy (co-first authors), Aditya Barua, Noah Fiedel, Noah Constant -- all Google DeepMind/Google [^62^]

- **Core Idea**: AR2Diff is a lightweight adaptation procedure that transforms pretrained autoregressive models into text diffusion models, exploring whether text diffusion can replace AR decoding for training and deploying LLMs [^29^][^62^].

- **Key Findings** [^29^][^57^][^62^]:
  - Training a decoder-only model with a **prefix LM objective** is best or near-best across several tasks
  - On **machine translation**, text diffusion underperforms standard AR approach
  - On **code synthesis and extractive QA**, diffusion models trained from scratch outperform AR models in many cases
  - AR2Diff adaptation of AR models to diffusion decoding produces **quality gains**
  - "These results are promising given that text diffusion is relatively underexplored and can be significantly faster than AR decoding for long text generation"

#### 3. MD4: Masked Diffusion for Discrete Data (Foundational Framework)

- **Paper**: "Simplified and Generalized Masked Diffusion for Discrete Data" (NeurIPS 2024) [^84^][^103^][^124^]

- **Authors**: Jiaxin Shi, Kehang Han (co-first authors), Zhe Wang, Arnaud Doucet, Michalis K. Titsias -- Google DeepMind [^84^][^103^]

- **Core Contribution**: Provides a simple and general framework for masked diffusion models. Shows that the continuous-time variational objective is a simple weighted integral of cross-entropy losses. Enables training generalized masked diffusion models with **state-dependent masking schedules** [^84^][^103^].

- **Key Results** [^84^][^124^]:
  - Models trained on OpenWebText surpass prior diffusion language models at GPT-2 scale
  - Superior performance on 4 out of 5 zero-shot language modeling tasks
  - Vastly outperforms previous discrete diffusion models on pixel-level image modeling: 2.75 bits/dim (CIFAR-10) and 3.40 bits/dim (ImageNet 64x64) -- better than autoregressive models of similar sizes
  - Code available at https://github.com/google-deepmind/md4 [^84^][^93^]

- **Technical Detail**: MD4 is a masked diffusion model where at each timestep, non-masked tokens transition to [MASK] with probability beta_t. At final time T, all tokens are masked [^82^][^83^].

#### 4. Other DeepMind Diffusion Research

- **CANDI** (Jiaxin Shi et al., 2025): "Hybrid Discrete-Continuous Diffusion Models" -- introduces a framework that decouples discrete and continuous corruption for discrete data generation. Outperforms masked diffusion at low NFE on text generation [^184^].

- **ELF (Embedded Language Flows)**: He Kaiming's team (MIT/Google DeepMind) introduced continuous diffusion in embedding space that achieves competitive performance with 10x fewer training tokens [^177^].

- **Early Stopping Overparameterized Diffusion Models** (2025): Leverages MD4 codebase for language diffusion experiments [^82^].

- **Compositional Generalization in Diffusion** (2025): Also uses MD4 codebase, showing DeepMind's sustained investment in the framework [^83^].

#### 5. Key Technical Innovations from DeepMind

- **Bidirectional attention in denoiser**: Unlike autoregressive models, diffusion language models allow tokens to attend to all positions (both past and future) within a generation block, enabling non-causal reasoning and global coherence [^37^][^39^]

- **Iterative refinement / self-correction**: The denoising process passes tokens back through the denoiser, allowing error correction that AR models cannot perform once a token is emitted [^37^][^39^]

- **Adaptive computation**: Diffusion models converge at different rates depending on task difficulty -- consuming fewer resources on easy tasks and more on harder ones [^37^][^39^]

- **Block diffusion architecture**: Uses compact block sizes (typically 32 tokens) as the standard for production deployment [^31^]

- **State-dependent masking schedules**: From MD4, enabling flexible prioritization of token masking/unmasking [^84^]

- **Prefix LM objective**: Identified as optimal for training text diffusion models [^62^]

#### 6. Disadvantages and Limitations Acknowledged by DeepMind

- **Higher serving costs**: Due to complexity of the denoising process [^37^][^39^]
- **Higher time-to-first-token (TTFT)**: First token only appears when entire sequence is ready, unlike AR which produces first token immediately [^37^][^39^][^63^]
- **Fixed-length generation**: Diffusion models can only generate text segments of fixed length, struggling with essays or multi-paragraph narratives [^90^][^92^]
- **Weaker narrative flow**: Without left-to-right generation, models can lose natural flow and logical progression [^90^][^92^]
- **Lower performance on reasoning and multilingual tasks**: GPQA Diamond (40.4% vs 56.5%), BIG-Bench Hard (15.0% vs 21.0%), Global MMLU Lite (69.1% vs 79.0%) [^37^][^110^]

---

### Major Players & Sources

| Entity | Role / Relevance |
|---|---|
| **Brendan O'Donoghue** | Research scientist at Google DeepMind; lead on Gemini Diffusion project; vocal advocate for diffusion over autoregression. Identified key advantages (lower latency, adaptive computation, non-causal reasoning, self-correction) and disadvantages (higher serving cost, TTFT) [^37^][^39^][^63^] |
| **Jack Rae** | Principal scientist at Google DeepMind. Called Gemini Diffusion a "landmark moment" -- "It wasn't clear that the gap [between diffusion and autoregressive] would ever be closed...the result is a fascinating and powerful model that is also lightning fast" [^55^][^90^][^91^] |
| **Oriol Vinyals** | VP of Research and Deep Learning Lead at Google DeepMind, Co-Head of Gemini project. Described Gemini Diffusion as a personal milestone: "It's been a dream of mine to remove the need for 'left to right' text generation" [^91^] |
| **Jiaxin Shi** | Google DeepMind researcher; lead author of MD4 paper (NeurIPS 2024); also co-author of CANDI hybrid diffusion [^84^][^103^][^184^] |
| **Kehang Han** | Google DeepMind researcher; co-first author on both AR2Diff and MD4 papers -- central to DeepMind's text diffusion research [^62^][^84^] |
| **Kathleen Kenealy** | Google DeepMind researcher; co-first author on AR2Diff paper [^62^] |
| **Noah Fiedel & Noah Constant** | Google DeepMind researchers; co-authors on AR2Diff; known for language model infrastructure work [^62^] |
| **Michalis Titsias & Arnaud Doucet** | DeepMind senior researchers; co-authors on MD4 paper; experts in probabilistic inference [^84^][^114^] |
| **Stefano Ermon** | Stanford professor; co-founder/CEO of Inception Labs (Mercury). Called Gemini Diffusion "biggest endorsement yet" of text diffusion; predicts "all frontier models will be diffusion models" within a few years [^55^][^90^] |
| **Nathan Lambert** | AI2 researcher; called Gemini Diffusion the "biggest endorsement yet" but noted "we have no details so can't compare well" [^55^][^167^] |
| **Alexander Doria** | Cofounder of Pleias; described Gemini Diffusion as "so much faster, potentially better for some tasks" [^55^][^90^] |

---

### Trends & Signals

- **Industry validation of text diffusion paradigm**: Gemini Diffusion and Mercury Coder (Inception Labs) both launched in 2025, proving dLLMs work at production quality and scale. As of early 2026, dLLMs have moved from research papers to deployed products [^31^][^55^].

- **Dominant pattern is AR-to-diffusion conversion**: "The dominant pattern in early 2026 is to initialize from a pre-trained AR model and continue training with the diffusion objective, rather than training from scratch. This is more compute-efficient and produces competitive results" [^31^].

- **Block diffusion as standard**: "Block diffusion with a compact block size (typically 32 tokens) has emerged as the standard architecture for production deployment" [^31^].

- **Coding as killer app for diffusion LLMs**: Both Gemini Diffusion and Mercury Coder are heavily optimized for code. Diffusion models excel at coding due to non-local consistency requirements and ability to make global edits -- matching or exceeding AR counterparts [^37^][^45^][^71^].

- **Speed-quality frontier shift**: Mercury Coder Mini achieves 1,109 tok/sec (HumanEval 88.0%); Mercury Small at 737 tok/sec (HumanEval 90.0%) -- both 5-10x faster than comparable AR models while matching quality [^71^][^85^].

- **Venture capital flowing to diffusion LLMs**: Inception Labs raised $50 million seed round (led by Menlo Ventures) in November 2025 specifically for diffusion-based AI models [^162^][^164^].

- **Community skepticism about openness**: Significant frustration that Gemini Diffusion has no public weights, no API, and no open-source plan -- limiting reproducible research. Open-source alternatives like LLaDA-8B gaining community traction [^173^].

---

### Controversies & Conflicting Claims

#### 1. Open Access vs. Proprietary Lock-in
- **Conflict**: Google DeepMind released Gemini Diffusion as waitlist-only experimental demo with no weights, no API, and no announced plan to open-source. The research community has expressed frustration.
- **Evidence**: Nathan Lambert (AI2): "biggest endorsement yet of the [text diffusion] model, but we have no details so can't compare well" [^55^]. Hacker News discussion highlights that open-source analogues like LLaDA-8B exist, making Gemini Diffusion's lack of openness a barrier [^173^].

#### 2. Narrative Text Generation Weakness
- **Conflict**: While diffusion models excel at code, they struggle with long-form narrative text due to fixed-length generation constraints.
- **Evidence**: "Some researchers have noted that while diffusion models are fast and flexible, they can only generate text segments of a fixed length, and so may struggle with writing essays or multi-paragraph narratives. Because they don't build sentences one word at a time, diffusion models can lose the kind of natural flow and logical progression that transformer-based models are optimized for" [^90^][^92^].

#### 3. Benchmark Comparison Fairness
- **Conflict**: Gemini Diffusion is compared against Gemini 2.0 Flash-Lite, which is an older budget model, not the current frontier models (2.5 Pro, etc.).
- **Evidence**: "With Gemini Diffusion, a diffusion-based language model achieves performance comparable to current models for the first time, even though Gemini 2.0 Flash-Lite is an older budget model from Google" [^91^]. When compared to newer Gemini 2.5 Pro, Gemini Diffusion is outperformed on all benchmarks (AIME: 23.3% vs 83.0%, GPQA: 40.4% vs 83.0%) [^183^].

#### 4. "Pseudo" Discrete Diffusion Debate
- **Conflict**: Some researchers argue that masked diffusion models (like MD4/Gemini Diffusion) don't satisfy the formal definition of diffusion processes.
- **Evidence**: The Authentic Discrete Diffusion (ADD) paper states: "Such approaches effectively replicate masked language modeling as popularized by BERT, and do not satisfy the formal definition of a diffusion process... For this reason, we refer to such methods as 'pseudo' discrete diffusion (PDD)" [^104^][^181^]. This is a conceptual/philosophical debate about what counts as "true" diffusion.

#### 5. Will All Frontier Models Be Diffusion?
- **Conflict**: Stefano Ermon (Inception Labs) predicts "all frontier models will be diffusion models" within a few years [^55^]. Others are more cautious.
- **Evidence**: Current diffusion models still lag on reasoning, science, and multilingual tasks. The autoregressive paradigm has massive ecosystem lock-in (tooling, optimization, research community). DeepMind itself continues to invest heavily in both approaches.

#### 6. Time-to-First-Token (TTFT) Trade-off
- **Conflict**: Diffusion models' parallel generation requires waiting for the entire sequence before outputting any token, creating higher perceived latency for streaming applications.
- **Evidence**: Brendan O'Donoghue acknowledges: "higher cost of serving and slightly higher time-to-first-token (TTFT), since autoregressive models will produce the first token right away. For diffusion, the first token can only appear when the entire sequence of tokens is ready" [^37^][^39^].

---

### Recommended Deep-Dive Areas

#### 1. MD4 Architecture and Masking Schedules
**Why**: MD4 (arXiv:2406.04329, NeurIPS 2024) is the foundational codebase that underpins much of DeepMind's diffusion language model work. It achieves state-of-the-art results on both text and pixel-level image modeling, and the state-dependent masking schedules represent a key technical innovation. Understanding MD4 is essential for understanding Gemini Diffusion's underlying mechanics.

#### 2. AR2Diff Transfer Learning Methodology
**Why**: AR2Diff provides the technical pathway for converting existing AR models to diffusion. Given that "the dominant pattern in early 2026 is to initialize from a pre-trained AR model," understanding the exact transfer learning procedure, its compute requirements, and the quality trade-offs is critical for the field.

#### 3. Scaling Laws for Diffusion Language Models
**Why**: Very little is publicly known about how Gemini Diffusion scales with parameters and compute. The AR2Diff paper only tested at "relatively small sizes." Understanding scaling laws is essential to determine whether diffusion can match autoregressive performance at the largest scales (GPT-4 class).

#### 4. Inference Infrastructure and Serving Economics
**Why**: Brendan O'Donoghue explicitly identified "higher cost of serving" as a key disadvantage. For production deployment, understanding the actual serving costs, memory requirements, and infrastructure needs compared to optimized AR serving (vLLM, TensorRT-LLM) is critical.

#### 5. Safety and Alignment for Diffusion LLMs
**Why**: Safety techniques like RLHF were designed for autoregressive generation. How do preference optimization, safety filtering, and harm mitigation work when text is generated in parallel blocks? This is a major unanswered question.

#### 6. Comparison with Mercury/Inception Labs
**Why**: Inception Labs (Mercury Coder) is the main commercial competitor in the diffusion LLM space. A detailed technical comparison between Gemini Diffusion and Mercury's approach would illuminate different architectural choices and their trade-offs.

#### 7. The "Pseudo Diffusion" Debate
**Why**: The question of whether masked diffusion is "true" diffusion or just BERT-style masked language modeling with extra steps has implications for how the field categorizes and evaluates these models. The ADD paper's critique deserves careful analysis.

---

### Sources and Citations

[^29^]: arXiv:2401.17181, "Transfer Learning for Text Diffusion Models" (AR2Diff), Kehang Han et al., Google DeepMind, January 30, 2024. https://ar5iv.labs.arxiv.org/html/2401.17181

[^31^]: "Diffusion LLMs from the Ground Up: Training, Inference, and Practical Engineering," Daily Dose of DS, April 19, 2026. https://www.dailydoseofds.com/diffusion-models-part-2/

[^37^]: VentureBeat, "Beyond GPT architecture: Why Google's Diffusion approach could reshape LLM deployment," June 13, 2025. https://venturebeat.com/technology/beyond-gpt-architecture-why-googles-diffusion-approach-could-reshape-llm-deployment

[^39^]: VentureBeat (republished), "Beyond GPT architecture: Why Google's Diffusion approach could reshape LLM deployment," June 13, 2025. http://venturebeat.com/ai/beyond-gpt-architecture-why-googles-diffusion-approach-could-reshape-llm-deployment

[^45^]: arXiv:2506.17298, "Mercury: Ultra-Fast Language Models Based on Diffusion," Inception Labs, June 2025. https://arxiv.org/abs/2506.17298

[^49^]: Topmost Ads, "Gemini Diffusion Text Generation Model: Deep Dive into AI's Future," June 19, 2025. https://topmostads.com/gemini-diffusion-text-generation-deep-dive/

[^50^]: CometAPI, "What is Gemini Diffusion? All You Need to Know," May 25, 2025. https://www.cometapi.com/what-is-gemini-diffusion/

[^51^]: ITmedia, "Google DeepMindから拡散言語モデル「Gemini Diffusion」登場 文字通り爆速で文章・コード生成：Google I/O 2025," May 21, 2025. https://www.itmedia.co.jp/aiplus/articles/2505/21/news115.html

[^55^]: Fortune, "Gemini Diffusion didn't get stage time at Google I/O—but AI insiders are calling it 'ChatGPT on steroids'," May 21, 2025. https://fortune.com/2025/05/21/gemini-diffusion-google-io-sleeper-hit-blazing-speed-ai-model-wars/

[^62^]: arXiv:2401.17181v1, "Transfer Learning for Text Diffusion Models," Han et al., Google DeepMind, January 2024. https://arxiv.org/pdf/2401.17181

[^63^]: ONMINE, "Why Google's Diffusion approach could reshape LLM deployment." https://onmine.io/beyond-gpt-architecture-why-googles-diffusion-approach-could-reshape-llm-deployment/

[^64^]: HyperAI, "Google's Gemini Diffusion: How a New Approach Could Revolutionize Large Language Model Deployment," June 13, 2025. https://hyper.ai/en/headlines/6baf034407326636f909f2d07f462957

[^71^]: Inception Labs, "Introducing Mercury, the World's First Commercial-Scale Diffusion Large Language Model," February 26, 2025. https://www.inceptionlabs.ai/blog/introducing-mercury

[^82^]: arXiv:2505.16959v1, "Early Stopping Overparameterized Diffusion Models," 2025. https://arxiv.org/html/2505.16959v1

[^83^]: arXiv:2502.12089v2, "How compositional generalization and creativity improve as diffusion models are trained," 2025. https://arxiv.org/html/2502.12089v2

[^84^]: arXiv:2406.04329v4, "Simplified and Generalized Masked Diffusion for Discrete Data," Shi et al., Google DeepMind, January 2025. https://arxiv.org/html/2406.04329v4

[^85^]: arXiv:2506.17298, "Mercury: Ultra-Fast Language Models Based on Diffusion," Inception Labs, June 17, 2025. https://arxiv.org/abs/2506.17298

[^90^]: Yahoo Tech, "Gemini Diffusion was the sleeper hit of Google I/O and some say its blazing speed could reshape the AI model wars," May 21, 2025. https://tech.yahoo.com/articles/gemini-diffusion-sleeper-hit-google-215427722.html

[^91^]: The Decoder, "Gemini Diffusion could be Google's most important I/O news that slipped under the radar," May 21, 2025. https://the-decoder.com/gemini-diffusion-could-be-googles-most-important-i-o-news-that-slipped-under-the-radar/

[^92^]: AOL Finance, "Gemini Diffusion was the sleeper hit of Google I/O," May 21, 2025. https://www.aol.com/finance/gemini-diffusion-sleeper-hit-google-215427313.html

[^93^]: DeepWiki, "Installation and Setup | google-deepmind/md4," May 2, 2025. https://deepwiki.com/google-deepmind/md4/1.1-installation-and-setup

[^103^]: arXiv:2406.04329v3, "Simplified and Generalized Masked Diffusion for Discrete Data," Shi et al., Google DeepMind. https://arxiv.org/html/2406.04329v3

[^104^]: arXiv:2510.01047v1, "Authentic Discrete Diffusion Model," October 2025. https://arxiv.org/html/2510.01047v1

[^108^]: OpenTools AI, "Google DeepMind's Gemini Diffusion: A Game-Changer in AI Speed and Consistency," June 16, 2025. https://opentools.ai/news/google-deepminds-gemini-diffusion-a-game-changer-in-ai-speed-and-consistency

[^110^]: CTOL Digital, "Google DeepMind Unveils Gemini Diffusion - A Paradigm Shift in AI Text Generation," May 21, 2025. https://www.ctol.digital/news/google-deepmind-gemini-diffusion-ai-text-generation-paradigm-shift/

[^114^]: Bocconi University seminar, "Masked Diffusion Models for Discrete Data," Michalis Titsias (Google DeepMind), November 7, 2024. https://bidsa.unibocconi.eu/masked-diffusion-models-discrete-data

[^124^]: NeurIPS 2024, "Simplified and Generalized Masked Diffusion for Discrete Data," Shi et al., Google DeepMind. https://proceedings.neurips.cc/paper_files/paper/2024/file/bad233b9849f019aead5e5cc60cef70f-Paper-Conference.pdf

[^162^]: Inception Labs Blog, "Announcing our $50 Million Seed Round," November 6, 2025. https://www.inceptionlabs.ai/blog/categories/company

[^163^]: The Information Bottleneck, "Stefano Ermon on Diffusion LLMs, Mercury & Why the Future of AI Won't Be Autoregressive," March 18, 2026. https://www.the-information-bottleneck.com/stefano-ermon-on-diffusion-llms-mercury-why-the-future-of-ai-wont-be-autoregressive/

[^164^]: Menlo Ventures, "From the Lab to the Frontier: The Story Behind Inception," March 25, 2026. https://menlovc.com/perspective/from-the-lab-to-the-frontier-the-story-behind-inception/

[^167^]: OpenTools AI, "Google DeepMind's Gemini Diffusion: A Game-Changer in AI Speed and Consistency," June 16, 2025. https://opentools.ai/news/google-deepminds-gemini-diffusion-a-game-changer-in-ai-speed-and-consistency

[^172^]: Google DeepMind, "Gemini Diffusion" (official model page). https://deepmind.google/models/gemini-diffusion/

[^173^]: AINews, "OpenAI buys Jony Ive's io for $6.5b, LMArena lands $100m seed from a16z," May 21, 2025. https://news.smol.ai/issues/25-05-21-openai-io

[^177^]: 36Kr, "New Work on 'Diffusion Models' by He Kaiming's Team," May 13, 2026. https://eu.36kr.com/en/p/3807465382190852

[^181^]: arXiv:2510.01047v1, "Authentic Discrete Diffusion Model," October 2025. https://arxiv.org/html/2510.01047v1

[^183^]: LLM-Stats, "Gemini 2.5 Pro vs Gemini Diffusion Comparison," May 2025. https://llm-stats.com/models/compare/gemini-2.5-pro-vs-gemini-diffusion

[^184^]: arXiv:2510.22510v1, "CANDI: Hybrid Discrete-Continuous Diffusion Models," Pynadath, Shi, Zhang, October 2025. https://arxiv.org/html/2510.22510v1
