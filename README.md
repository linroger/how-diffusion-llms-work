# How Diffusion LLMs Work

An interactive, illustrated explainer of **diffusion language models** for text and code — from first principles to Google DeepMind's Gemini Diffusion and Ant Group's LLaDA 2 series.

🌐 **Live site:** [linroger.github.io/how-diffusion-llms-work](https://linroger.github.io/how-diffusion-llms-work/)

🇬🇧 / 🇨🇳 Bilingual (English / 简体中文)

---

## What this is

A single-page educational site that builds up diffusion LLMs **layer by layer, from first principles**:

1. **Foundations** — why image diffusion doesn't translate to text, what discrete diffusion fixes, and the MD4/MDLM simplification that made the whole field viable
2. **Architecture** — what changes in the Transformer (bidirectional attention, no GQA, time-step embedding), with side-by-side mask comparisons
3. **Training** — the forward process in slow motion, noise schedule comparisons, and Ant Group's WSD curriculum for converting an AR checkpoint into a diffusion model
4. **Inference** — the denoising loop, four remasking strategies, block diffusion, Confidence-Aware Parallel decoding, and LLaDA 2.1's Token-to-Token editing
5. **The Stacks** — Google DeepMind's MD4 → AR2Diff → Gemini Diffusion lineage, and Ant Group's open-source LLaDA 1 → LLaDA 2.1 family
6. **Playground** — build your own denoising loop, see the reverse curse demonstrated live

Twenty interactive visualizations in total, each driven by D3 + custom canvas/SVG, all rendered client-side with no build step.

## How to run locally

```bash
cd site
python3 -m http.server 8000
# open http://localhost:8000
```

No `npm install`, no bundler, no build step — pure HTML/CSS/vanilla JS with D3 loaded from CDN.

## Structure

```
site/
├── index.html              20-section page shell
├── styles.css              Design system (dark mode, editorial-technical)
├── i18n.js                 199 strings × 2 locales (EN + 简体中文)
├── app.js                  Language toggle, scroll-spy TOC, smooth scroll
└── viz/                    One self-contained script per visualization
    ├── hero.js
    ├── sec1-ar-vs-diffusion.js
    ├── sec2-image-diffusion.js
    ├── ...
    └── sec20-reverse-curse.js
```

In the project root, the markdown files (`diffusion_llms_mechanistic_deepdive.md`, `diffusion_report.agent.final.md`, `research/*.md`) are the underlying research that the site is built on — useful if you want to dig deeper than the visualizations.

## Sources

Every claim in the site is grounded in primary sources. The full source list is in the `Sources` section at the bottom of the live page, but the load-bearing papers are:

- [MD4 — Shi et al., NeurIPS 2024](https://arxiv.org/abs/2406.04329) ([code](https://github.com/google-deepmind/md4))
- [LLaDA — Nie et al., Feb 2025](https://arxiv.org/abs/2502.09992) ([code](https://github.com/ML-GSAI/LLaDA))
- [LLaDA 2.0 — Ant Group et al., Dec 2025](https://arxiv.org/abs/2512.15745) ([code](https://github.com/inclusionAI/LLaDA2.X))
- [BD3-LM — Arriola et al., ICLR 2025](https://arxiv.org/abs/2503.09573) (block diffusion)
- [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/)
- [LMSYS SGLang Day-0 LLaDA 2.0](https://www.lmsys.org/blog/2025-12-19-diffusion-llm/)

## License

MIT — see [LICENSE](LICENSE).

---

*Built as a portfolio / explainer project. Feedback welcome via issues.*
