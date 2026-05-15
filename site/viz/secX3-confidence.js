/* Extra 3 — Confidence Threshold Explorer
   Each token in the prediction has a top-1 confidence. Drag τ.
   Tokens with confidence ≥ τ commit (filled). Tokens below τ stay masked.
   Hover any token to see the top-3 candidates and their probabilities.
   This is the most direct "look inside CAP / Fast-dLLM decoding" viz.  */
(function () {
  'use strict';

  // Fake but plausible top-3 distributions for each token (deterministic per sentence)
  function buildPredictions(sentence, seed) {
    const rng = DLM.makeRNG(seed);
    return sentence.map((trueTok, i) => {
      // Top-1 confidence based on token "difficulty"
      const base = 0.55 + 0.42 * Math.exp(-trueTok.length / 7);
      const p1 = Math.max(0.35, Math.min(0.99, base + (rng() - 0.5) * 0.25));
      const rem = 1 - p1;
      const p2 = rem * (0.45 + rng() * 0.3);
      const p3 = rem - p2;
      // Generate plausible alternative tokens
      const alts = ['a', 'the', 'of', 'and', 'in', 'is', 'it', 'an', 'on', 'to', 'for', 'with', 'by'];
      const alt1 = alts[Math.floor(rng() * alts.length)];
      const alt2 = alts[Math.floor(rng() * alts.length)];
      return {
        truth: trueTok,
        top1: { tok: trueTok, p: p1 },
        top2: { tok: alt1, p: p2 },
        top3: { tok: alt2, p: p3 },
      };
    });
  }

  function render() {
    const container = document.getElementById('vizX3');
    if (!container) return;
    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    const tauSlider = document.getElementById('vizX3-tau');
    const tau = tauSlider ? parseInt(tauSlider.value) / 100 : 0.85;

    const tauReadout = document.getElementById('vizX3-tval');
    if (tauReadout) tauReadout.textContent = tau.toFixed(2);

    const sentence = DLM.pickSentence(0);
    const preds = buildPredictions(sentence, 31);

    const strip = container.querySelector('.x3-strip');
    if (!strip) return;
    strip.innerHTML = '';

    let committed = 0;
    preds.forEach((p, i) => {
      const cell = document.createElement('div');
      cell.className = 'x3-cell';
      const commitsThisStep = p.top1.p >= tau;
      if (commitsThisStep) {
        cell.classList.add('committed');
        committed++;
      } else {
        cell.classList.add('masked');
      }

      // Confidence bar inside
      cell.innerHTML = `
        <div class="x3-cell-conf-bar" style="height:${(p.top1.p * 100).toFixed(0)}%"></div>
        <div class="x3-cell-tok">${commitsThisStep ? p.top1.tok : '[MASK]'}</div>
        <div class="x3-cell-prob">${(p.top1.p * 100).toFixed(0)}</div>
      `;

      // tooltip on hover
      cell.addEventListener('mouseenter', () => {
        const tip = container.querySelector('.x3-tooltip');
        if (!tip) return;
        tip.innerHTML = `
          <div class="x3-tip-title">${lang === 'zh' ? '位置' : 'Position'} ${i + 1} ${lang === 'zh' ? '· 模型预测' : '· model prediction'}</div>
          <div class="x3-tip-row"><span class="x3-tip-tok x3-tip-tok-1">${p.top1.tok}</span> <span class="x3-tip-bar"><span style="width:${(p.top1.p*100).toFixed(0)}%"></span></span> <span class="x3-tip-p">${(p.top1.p * 100).toFixed(1)}%</span></div>
          <div class="x3-tip-row"><span class="x3-tip-tok">${p.top2.tok}</span> <span class="x3-tip-bar"><span style="width:${(p.top2.p*100).toFixed(0)}%"></span></span> <span class="x3-tip-p">${(p.top2.p * 100).toFixed(1)}%</span></div>
          <div class="x3-tip-row"><span class="x3-tip-tok">${p.top3.tok}</span> <span class="x3-tip-bar"><span style="width:${(p.top3.p*100).toFixed(0)}%"></span></span> <span class="x3-tip-p">${(p.top3.p * 100).toFixed(1)}%</span></div>
          <div class="x3-tip-foot">${
            p.top1.p >= tau ?
              (lang === 'zh' ? '✓ 超过 τ — 本步提交' : '✓ above τ — commits this step') :
              (lang === 'zh' ? '✗ 低于 τ — 保留遮蔽' : '✗ below τ — stays masked')
          }</div>
        `;
        tip.classList.add('show');
      });
      cell.addEventListener('mouseleave', () => {
        container.querySelector('.x3-tooltip')?.classList.remove('show');
      });

      strip.appendChild(cell);
    });

    const stats = container.querySelector('.x3-stats');
    if (stats) {
      const tpf = committed;
      stats.innerHTML = lang === 'zh' ?
        `<div><span class="x1-stat-val">τ = ${tau.toFixed(2)}</span><span class="x1-stat-lbl">置信度阈值</span></div>
         <div><span class="x1-stat-val">${committed}/${sentence.length}</span><span class="x1-stat-lbl">本步提交</span></div>
         <div><span class="x1-stat-val">${tpf}</span><span class="x1-stat-lbl">本步 TPF</span></div>` :
        `<div><span class="x1-stat-val">τ = ${tau.toFixed(2)}</span><span class="x1-stat-lbl">confidence threshold</span></div>
         <div><span class="x1-stat-val">${committed}/${sentence.length}</span><span class="x1-stat-lbl">commit this step</span></div>
         <div><span class="x1-stat-val">${tpf}</span><span class="x1-stat-lbl">TPF this step</span></div>`;
    }
  }

  function init() {
    const container = document.getElementById('vizX3');
    if (!container) return;
    container.innerHTML = `
      <div class="x3-tooltip"></div>
      <div class="x3-strip"></div>
      <div class="x1-stats"></div>
    `;
    document.getElementById('vizX3-tau')?.addEventListener('input', render);
    window.addEventListener('langchange', render);
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
