# How Diffusion LLMs Work

A single-page, bilingual walkthrough of diffusion language models. It starts at "why don't we just bolt image diffusion onto text" and ends somewhere around Gemini Diffusion and Ant Group's LLaDA family — with a couple of dozen interactive demos along the way.

🌐 **Live site:** [linroger.github.io/how-diffusion-llms-work](https://linroger.github.io/how-diffusion-llms-work/)

🇬🇧 English README · [🇨🇳 中文 README](README.zh-CN.md)

---

## What's on the page

The site is split into six parts. You can read it top to bottom or jump straight to whatever section you're stuck on.

- **Foundations** — what changes when you swap pixels for tokens, why discrete diffusion was needed, and the MD4 / MDLM simplification that made the whole thing trainable.
- **Architecture** — what's different about the Transformer when you train it this way. Mostly: the attention mask.
- **Training** — a slow-motion walkthrough of one training step, the different noise schedules, and how Ant Group reuses an AR checkpoint instead of starting from scratch.
- **Inference** — the denoising loop, four remasking strategies, block diffusion, confidence-aware parallel decoding, and how LLaDA 2.1 edits tokens it already wrote.
- **The Stacks** — DeepMind's path (MD4 → AR2Diff → Gemini Diffusion) and Ant Group's open-source line (LLaDA → LLaDA 2.1).
- **Playground** — build your own denoising loop, see the reverse curse happen live.

Every section has at least one figure you can poke at. Everything's rendered in the browser; there's no model running on a server.

## Who it's for

If you've used ChatGPT, remember a little probability, and can read pseudocode without flinching, you'll be fine. Each part has a primer callout for the prerequisite that section needs (Markov chains, attention masks, KV cache, etc.), so you don't have to bring much in. A bit of patience with math notation helps.

## Running it locally

```bash
cd site
python3 -m http.server 8000
# then visit http://localhost:8000
```

No bundler, no `npm install`, no build step. It's static HTML, CSS, and vanilla JS, with D3 pulled in from a CDN.

## What's in the repo

```
site/
├── index.html        the page
├── styles.css        design system, dark + light, mobile
├── i18n.js           every visible string in EN and 简体中文
├── app.js            language toggle, scroll-spy, helpers
└── viz/              one self-contained script per figure (hero.js, sec1-…, sec2-…)
```

If you want to dig past the visualizations, the markdown files at the repo root (`diffusion_llms_mechanistic_deepdive.md`, `diffusion_report.agent.final.md`, `research/`) are the long-form research notes the page was built from.

## Where the claims come from

I tried to keep every concrete number tied back to a paper or the matching code. The full list lives in the **Sources** section at the bottom of the site; the ones the page leans on most:

- [MD4](https://arxiv.org/abs/2406.04329) — Shi et al., NeurIPS 2024 ([code](https://github.com/google-deepmind/md4))
- [LLaDA](https://arxiv.org/abs/2502.09992) — Nie et al., Feb 2025 ([code](https://github.com/ML-GSAI/LLaDA))
- [LLaDA 2.0](https://arxiv.org/abs/2512.15745) — Ant Group, Dec 2025 ([code](https://github.com/inclusionAI/LLaDA2.X))
- [BD3-LM (block diffusion)](https://arxiv.org/abs/2503.09573) — Arriola et al., ICLR 2025
- [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/)
- [LMSYS SGLang Day-0 LLaDA 2.0](https://www.lmsys.org/blog/2025-12-19-diffusion-llm/)

## License

MIT — see [LICENSE](LICENSE).

---

Built as a portfolio / explainer project. Bug reports and corrections are very welcome — open an issue or a PR.
