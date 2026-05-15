/* Extra 1 — Mask Rate Scrubber
   Drag t from 0 to 1. Watch the same sentence get progressively masked.
   For each t, show how many tokens are masked, what fraction the loss weights them at,
   and intuition: "the model sees X% of context and must guess the rest". */
(function () {
  'use strict';

  function tokenColor(tok, masked) {
    return masked ? 'mask' : 'committed';
  }

  function render() {
    const container = document.getElementById('vizX1');
    if (!container) return;

    const slider = document.getElementById('vizX1-t');
    const t = slider ? parseInt(slider.value) / 100 : 0.5;
    const tReadout = document.getElementById('vizX1-tval');
    if (tReadout) tReadout.textContent = t.toFixed(2);
    const lang = document.documentElement.getAttribute('data-lang') || 'en';

    // Deterministic mask pattern per t — so dragging produces smooth transitions
    const sentence = DLM.pickSentence(0);
    const rng = DLM.makeRNG(7);
    const masks = sentence.map(() => rng()); // each position's "threshold"
    const state = sentence.map((tok, i) => {
      const isMasked = masks[i] < t;
      return { token: isMasked ? DLM.MASK : tok, state: isMasked ? 'mask' : 'committed' };
    });

    const stripEl = container.querySelector('.x1-strip');
    if (stripEl) DLM.renderTokens(stripEl, state);

    const maskedCount = state.filter((s) => s.state === 'mask').length;
    const totalCount = state.length;
    const visibleCount = totalCount - maskedCount;
    const weight = t > 0 ? (1 / t).toFixed(2) : '∞';

    const stats = container.querySelector('.x1-stats');
    if (stats) {
      stats.innerHTML = lang === 'zh' ?
        `<div><span class="x1-stat-val">${(t*100).toFixed(0)}%</span><span class="x1-stat-lbl">遮蔽率 t</span></div>
         <div><span class="x1-stat-val">${maskedCount}/${totalCount}</span><span class="x1-stat-lbl">被遮蔽 token</span></div>
         <div><span class="x1-stat-val">${visibleCount}/${totalCount}</span><span class="x1-stat-lbl">作为上下文</span></div>
         <div><span class="x1-stat-val">${weight}×</span><span class="x1-stat-lbl">每 token 损失权重</span></div>` :
        `<div><span class="x1-stat-val">${(t*100).toFixed(0)}%</span><span class="x1-stat-lbl">mask rate t</span></div>
         <div><span class="x1-stat-val">${maskedCount}/${totalCount}</span><span class="x1-stat-lbl">tokens masked</span></div>
         <div><span class="x1-stat-val">${visibleCount}/${totalCount}</span><span class="x1-stat-lbl">visible context</span></div>
         <div><span class="x1-stat-val">${weight}×</span><span class="x1-stat-lbl">loss weight w(t)</span></div>`;
    }

    // Caption that updates: easy / medium / hard
    const cap = container.querySelector('.x1-caption-dyn');
    if (cap) {
      let regime;
      if (lang === 'zh') {
        if (t < 0.25) regime = '<strong>简单：</strong> 模型看到大部分上下文，只需填几个空。损失高度加权 — 每次错误都很贵。';
        else if (t < 0.6) regime = '<strong>中等：</strong> 一半的句子被遮蔽。这是训练分布的"主力"区域。';
        else if (t < 0.9) regime = '<strong>困难：</strong> 大部分都看不见。模型必须从极少的暗示中重建语义。';
        else regime = '<strong>极限：</strong> 几乎完全噪声。这是推理开始时的状态 — 从纯遮蔽生成。';
      } else {
        if (t < 0.25) regime = '<strong>Easy regime:</strong> the model sees most context, only fills a few blanks. Loss is heavily weighted — each mistake is expensive.';
        else if (t < 0.6) regime = '<strong>Medium regime:</strong> half the sentence is masked. This is the "workhorse" region of training.';
        else if (t < 0.9) regime = '<strong>Hard regime:</strong> most of the context is gone. The model must reconstruct meaning from very few hints.';
        else regime = '<strong>Extreme:</strong> nearly pure noise. This is the state inference starts from — generating from a fully-masked tape.';
      }
      cap.innerHTML = regime;
    }
  }

  function init() {
    const container = document.getElementById('vizX1');
    if (!container) return;

    // build internal structure once
    container.innerHTML = `
      <div class="x1-strip token-strip token-strip-large"></div>
      <div class="x1-stats"></div>
      <div class="x1-caption-dyn"></div>
    `;

    document.getElementById('vizX1-t')?.addEventListener('input', render);
    window.addEventListener('langchange', render);
    window.addEventListener('palettechange', render);
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
