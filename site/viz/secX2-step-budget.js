/* Extra 2 — Step Budget Trade-off
   Drag N (number of denoising steps) from 1 to sequence-length.
   Watch the same sentence resolve through that many discrete steps, side-by-side
   with quality readout (lower N = more parallel commits per step = lower expected quality).
   This is the SINGLE clearest intuition for "speed vs quality" in diffusion LLMs. */
(function () {
  'use strict';

  let timer = null;
  let playing = false;
  let currentStep = 0;
  let lastN = 8;

  function buildSchedule(n, totalTokens) {
    // For step budget N over T tokens: at each step, ceil(T/N) tokens commit.
    // Tokens commit in confidence-order; we simulate via a fixed seed.
    const conf = [];
    const rng = DLM.makeRNG(123);
    for (let i = 0; i < totalTokens; i++) conf.push(rng());
    const indexed = conf.map((c, i) => ({ i, c })).sort((a, b) => b.c - a.c);
    const sched = new Array(totalTokens);
    const perStep = Math.ceil(totalTokens / n);
    for (let k = 0; k < totalTokens; k++) {
      sched[indexed[k].i] = Math.min(n, Math.floor(k / perStep) + 1);
    }
    return sched;
  }

  function render() {
    const container = document.getElementById('vizX2');
    if (!container) return;
    const lang = document.documentElement.getAttribute('data-lang') || 'en';

    const slider = document.getElementById('vizX2-n');
    const n = slider ? parseInt(slider.value) : 4;
    const nReadout = document.getElementById('vizX2-nval');
    if (nReadout) nReadout.textContent = n;
    if (n !== lastN) { lastN = n; currentStep = 0; }

    const sentence = DLM.pickSentence(0);
    const schedule = buildSchedule(n, sentence.length);

    const state = sentence.map((tok, i) => {
      const commitsAt = schedule[i];
      if (currentStep >= commitsAt) {
        return { token: tok, state: commitsAt === currentStep ? 'commit' : 'committed' };
      }
      return { token: DLM.MASK, state: 'mask' };
    });

    const strip = container.querySelector('.x2-strip');
    if (strip) DLM.renderTokens(strip, state);

    // Quality estimate: lower N = lower quality. Realistic curve from LLaDA2.0 ablation
    // approximating score vs TPF.
    const score = Math.min(99, 60 + 30 * (1 - Math.exp(-n / 6)));
    const tpf = (sentence.length / n).toFixed(1);

    const stats = container.querySelector('.x2-stats');
    if (stats) {
      stats.innerHTML = lang === 'zh' ?
        `<div><span class="x1-stat-val">${n}</span><span class="x1-stat-lbl">步数 N</span></div>
         <div><span class="x1-stat-val">${tpf}</span><span class="x1-stat-lbl">每步 token (TPF)</span></div>
         <div><span class="x1-stat-val">${score.toFixed(1)}</span><span class="x1-stat-lbl">预期质量分</span></div>
         <div><span class="x1-stat-val">${currentStep}/${n}</span><span class="x1-stat-lbl">当前步</span></div>` :
        `<div><span class="x1-stat-val">${n}</span><span class="x1-stat-lbl">steps N</span></div>
         <div><span class="x1-stat-val">${tpf}</span><span class="x1-stat-lbl">tokens / step (TPF)</span></div>
         <div><span class="x1-stat-val">${score.toFixed(1)}</span><span class="x1-stat-lbl">expected quality</span></div>
         <div><span class="x1-stat-val">${currentStep}/${n}</span><span class="x1-stat-lbl">at step</span></div>`;
    }

    const cap = container.querySelector('.x2-caption-dyn');
    if (cap) {
      let msg;
      if (lang === 'zh') {
        if (n <= 2) msg = '<strong>极速：</strong> 每步几乎全部 token 一起提交。条件独立假设被严重拉伸 — 质量下降。';
        else if (n <= 6) msg = '<strong>平衡：</strong> 每步几个 token。生产 dLLM 大多在这个区间运行。';
        else if (n <= sentence.length / 2) msg = '<strong>谨慎：</strong> 每步少量 token，质量逼近 AR 上限。';
        else msg = '<strong>逐个：</strong> 每步只一个 token — 与自回归计算量等价，没有并行收益。';
      } else {
        if (n <= 2) msg = '<strong>Aggressive:</strong> nearly every token commits in parallel each step. The conditional-independence assumption is heavily stretched — quality drops.';
        else if (n <= 6) msg = '<strong>Balanced:</strong> a handful of tokens per step. This is the band most production dLLMs operate in.';
        else if (n <= sentence.length / 2) msg = '<strong>Cautious:</strong> few tokens per step, quality approaches the AR ceiling.';
        else msg = '<strong>One-at-a-time:</strong> equivalent to autoregressive compute — no parallel benefit.';
      }
      cap.innerHTML = msg;
    }
  }

  function play() {
    if (playing) return;
    playing = true;
    currentStep = 0;
    const slider = document.getElementById('vizX2-n');
    const n = slider ? parseInt(slider.value) : 4;
    render();
    const tick = () => {
      currentStep++;
      if (currentStep > n) { playing = false; return; }
      render();
      timer = setTimeout(tick, Math.max(180, 1400 / n));
    };
    timer = setTimeout(tick, 400);
  }

  function reset() {
    if (timer) clearTimeout(timer);
    playing = false;
    currentStep = 0;
    render();
  }

  function init() {
    const container = document.getElementById('vizX2');
    if (!container) return;
    container.innerHTML = `
      <div class="x2-strip token-strip token-strip-large"></div>
      <div class="x1-stats"></div>
      <div class="x2-caption-dyn"></div>
    `;
    document.getElementById('vizX2-n')?.addEventListener('input', () => { currentStep = 0; render(); });
    document.getElementById('vizX2-play')?.addEventListener('click', play);
    document.getElementById('vizX2-reset')?.addEventListener('click', reset);
    window.addEventListener('langchange', render);
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
