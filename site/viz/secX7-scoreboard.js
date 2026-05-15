/* Extra 7 — Win/Loss Scoreboard
   A grouped bar chart that visualizes the empirical pattern from the literature:
   diffusion wins on editing/parallel tasks, ties on standard code, struggles on multi-hop reasoning.
   Hover any bar for the numerical breakdown and the reason. */
(function () {
  'use strict';

  const ROWS_EN = [
    {
      label: 'Code editing (CanItEdit)', diff: 60.0, ar: 50.5,
      desc: "Stable-DiffCoder-8B vs Seed-Coder-8B in the cleanest controlled comparison in the literature. Same architecture, same data, only the training objective differs. The +9.5pp gap is structural — editing is inherently non-sequential."
    },
    {
      label: 'Fill-in-the-middle', diff: 84.8, ar: 60.1,
      desc: "Mercury Coder vs Gemini Flash-Lite. FIM tasks require simultaneously conditioning on left and right context — exactly what bidirectional attention enables."
    },
    {
      label: 'HumanEval (code)', diff: 89.6, ar: 90.2,
      desc: "Standard code generation. Effectively tied. The diffusion edge fades on tasks where AR's serial coherence is sufficient."
    },
    {
      label: 'MBPP (code)', diff: 76.0, ar: 75.8,
      desc: "Standard code generation, slightly larger problems. Same story: parity."
    },
    {
      label: 'LiveCodeBench', diff: 30.9, ar: 28.5,
      desc: "Competition-level coding problems. Diffusion holds its own on the best models, falls behind on weaker ones."
    },
    {
      label: 'AIME 2025 (math)', diff: 23.3, ar: 20.0,
      desc: "Gemini Diffusion vs Gemini 2.0 Flash-Lite. Diffusion slightly ahead on contest-level math — bidirectional context helps with global structure."
    },
    {
      label: 'GPQA Diamond (science)', diff: 40.4, ar: 56.5,
      desc: "Multi-hop scientific reasoning. The coordination problem hits hard: parallel commit struggles when each conclusion depends on the previous one."
    },
    {
      label: 'BBH (reasoning)', diff: 15.0, ar: 21.0,
      desc: "BIG-Bench Hard. Long reasoning chains. -6pp gap. Same coordination issue: serial dependencies don't play well with parallel commit."
    },
    {
      label: 'Global MMLU (knowledge)', diff: 69.1, ar: 79.0,
      desc: "Knowledge retrieval and short-form QA. -10pp gap. The pattern is consistent: diffusion underperforms when context windows are tight and dependencies serial."
    },
  ];

  const ROWS_ZH = [
    { label: '代码编辑 (CanItEdit)', diff: 60.0, ar: 50.5,
      desc: 'Stable-DiffCoder-8B vs Seed-Coder-8B —— 文献中最干净的受控对照实验。同架构、同数据，仅训练目标不同。+9.5pp 的差距是结构性的：编辑天然非顺序。' },
    { label: '中段补全 FIM', diff: 84.8, ar: 60.1,
      desc: 'Mercury Coder vs Gemini Flash-Lite。FIM 任务要求同时在左右上下文下做判断 —— 正是双向注意力的所擅长。' },
    { label: 'HumanEval（代码）', diff: 89.6, ar: 90.2,
      desc: '标准代码生成。基本打平。AR 顺序连贯性足以处理这类任务时，扩散的优势消失。' },
    { label: 'MBPP（代码）', diff: 76.0, ar: 75.8,
      desc: '标准代码生成、稍复杂的题目。同样：基本打平。' },
    { label: 'LiveCodeBench', diff: 30.9, ar: 28.5,
      desc: '竞赛级编码题。最强扩散模型尚能持平，普通的会落后。' },
    { label: 'AIME 2025（数学）', diff: 23.3, ar: 20.0,
      desc: 'Gemini Diffusion vs Gemini 2.0 Flash-Lite。竞赛数学上扩散略胜 —— 双向上下文帮助把握整体结构。' },
    { label: 'GPQA Diamond（科学）', diff: 40.4, ar: 56.5,
      desc: '多跳科学推理。协调问题命中要害：并行提交在每个结论都依赖于上一个结论时表现挣扎。' },
    { label: 'BBH（推理）', diff: 15.0, ar: 21.0,
      desc: 'BIG-Bench Hard 长推理链。-6pp 差距。同样的协调问题：串行依赖与并行提交不合。' },
    { label: 'Global MMLU（知识）', diff: 69.1, ar: 79.0,
      desc: '知识检索与短答 QA。-10pp 差距。模式一致：上下文窄、依赖串行时，扩散落后。' },
  ];

  function classify(diff, ar) {
    const d = diff - ar;
    if (d >= 5) return 'win';
    if (d <= -5) return 'loss';
    return 'tie';
  }

  function render() {
    const container = document.getElementById('vizX7');
    if (!container) return;
    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    const rows = lang === 'zh' ? ROWS_ZH : ROWS_EN;
    container.innerHTML = '';

    const legend = document.createElement('div');
    legend.className = 'x7-legend';
    legend.innerHTML = lang === 'zh' ?
      `<span class="x7-leg x7-leg-win">扩散胜出 (≥+5pp)</span><span class="x7-leg x7-leg-tie">基本打平 (±5pp)</span><span class="x7-leg x7-leg-loss">扩散落后 (≤-5pp)</span>` :
      `<span class="x7-leg x7-leg-win">Diffusion wins (≥+5pp)</span><span class="x7-leg x7-leg-tie">Tie (±5pp)</span><span class="x7-leg x7-leg-loss">Diffusion loses (≤-5pp)</span>`;
    container.appendChild(legend);

    rows.forEach((row) => {
      const cls = classify(row.diff, row.ar);
      const max = Math.max(row.diff, row.ar, 60);
      const r = document.createElement('div');
      r.className = 'x7-row x7-row-' + cls;
      r.innerHTML = `
        <div class="x7-label">${row.label}</div>
        <div class="x7-bars">
          <div class="x7-barrow">
            <div class="x7-bartag">${lang === 'zh' ? '扩散' : 'Diffusion'}</div>
            <div class="x7-bartrack"><div class="x7-bar x7-bar-diff" style="width:${(row.diff/max)*100}%"></div></div>
            <div class="x7-barval">${row.diff.toFixed(1)}</div>
          </div>
          <div class="x7-barrow">
            <div class="x7-bartag">AR</div>
            <div class="x7-bartrack"><div class="x7-bar x7-bar-ar" style="width:${(row.ar/max)*100}%"></div></div>
            <div class="x7-barval">${row.ar.toFixed(1)}</div>
          </div>
          <div class="x7-delta">${row.diff > row.ar ? '+' : ''}${(row.diff - row.ar).toFixed(1)}</div>
        </div>
        <div class="x7-desc">${row.desc}</div>
      `;
      // Click expands the description; hover shows hint
      r.addEventListener('click', () => r.classList.toggle('expanded'));
      container.appendChild(r);
    });
  }

  function init() {
    window.addEventListener('langchange', render);
    window.addEventListener('palettechange', render);
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
