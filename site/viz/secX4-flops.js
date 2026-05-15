/* Extra 4 — AR vs Diffusion FLOPs / time comparison
   Two animated counters: same sequence length and model, AR vs Diffusion.
   Each "forward pass" advances the AR counter by 1 token, the Diffusion counter by N tokens.
   Watch the bars race to completion. The clearest intuition for "why diffusion is faster". */
(function () {
  'use strict';

  let arProg = 0, diffProg = 0;
  let playing = false;
  let timer = null;
  let totalTokens = 96;
  let diffSteps = 12;

  function render() {
    const container = document.getElementById('vizX4');
    if (!container) return;
    const lang = document.documentElement.getAttribute('data-lang') || 'en';

    const lenSlider = document.getElementById('vizX4-len');
    const stepSlider = document.getElementById('vizX4-steps');
    if (lenSlider) totalTokens = parseInt(lenSlider.value);
    if (stepSlider) diffSteps = parseInt(stepSlider.value);

    if (arProg > totalTokens) arProg = totalTokens;
    if (diffProg > totalTokens) diffProg = totalTokens;

    const arPct = (arProg / totalTokens) * 100;
    const diffPct = (diffProg / totalTokens) * 100;

    const arPasses = arProg; // AR uses 1 pass per token
    const diffPasses = Math.ceil((diffProg / totalTokens) * diffSteps);

    container.querySelector('.x4-ar-bar').style.width = arPct + '%';
    container.querySelector('.x4-diff-bar').style.width = diffPct + '%';
    container.querySelector('.x4-ar-out').textContent = `${arProg} / ${totalTokens}`;
    container.querySelector('.x4-diff-out').textContent = `${diffProg} / ${totalTokens}`;
    container.querySelector('.x4-ar-passes').textContent = arPasses;
    container.querySelector('.x4-diff-passes').textContent = diffPasses;

    const lenReadout = document.getElementById('vizX4-lenval');
    const stepsReadout = document.getElementById('vizX4-stepsval');
    if (lenReadout) lenReadout.textContent = totalTokens;
    if (stepsReadout) stepsReadout.textContent = diffSteps;

    // Speed multiple
    const speedup = totalTokens / diffSteps;
    container.querySelector('.x4-speedup-val').textContent = speedup.toFixed(1) + '×';

    // Update labels based on language
    const labels = container.querySelector('.x4-labels');
    if (labels) {
      const arLabel = lang === 'zh' ? '自回归 — 每个 token 一次前向' : 'Autoregressive — 1 forward / token';
      const diffLabel = lang === 'zh' ? `扩散 — ${diffSteps} 次前向，每次 ${(totalTokens/diffSteps).toFixed(1)} token` : `Diffusion — ${diffSteps} forwards × ${(totalTokens/diffSteps).toFixed(1)} tokens/pass`;
      labels.querySelector('.x4-ar-name').textContent = arLabel;
      labels.querySelector('.x4-diff-name').textContent = diffLabel;
    }
  }

  function play() {
    if (playing) return;
    playing = true;
    arProg = 0;
    diffProg = 0;
    render();
    const arPerTick = 1;
    const diffPerTick = Math.max(1, Math.floor(totalTokens / diffSteps));
    const tick = () => {
      arProg = Math.min(totalTokens, arProg + arPerTick);
      diffProg = Math.min(totalTokens, diffProg + diffPerTick);
      render();
      if (arProg >= totalTokens && diffProg >= totalTokens) { playing = false; return; }
      timer = setTimeout(tick, 60);
    };
    timer = setTimeout(tick, 200);
  }

  function reset() {
    if (timer) clearTimeout(timer);
    playing = false;
    arProg = 0;
    diffProg = 0;
    render();
  }

  function init() {
    const container = document.getElementById('vizX4');
    if (!container) return;
    container.innerHTML = `
      <div class="x4-speedup">
        <span class="x4-speedup-lbl" data-i18n-lazy="speedup">Diffusion is</span>
        <span class="x4-speedup-val">8.0×</span>
        <span class="x4-speedup-lbl">faster per output</span>
      </div>
      <div class="x4-labels">
        <div class="x4-row">
          <div class="x4-row-info">
            <div class="x4-row-name x4-ar-name">Autoregressive — 1 forward / token</div>
            <div class="x4-row-out"><span class="x4-ar-out">0 / 96</span> · <span class="x4-ar-passes">0</span> passes</div>
          </div>
          <div class="x4-track"><div class="x4-bar x4-ar-bar"></div></div>
        </div>
        <div class="x4-row">
          <div class="x4-row-info">
            <div class="x4-row-name x4-diff-name">Diffusion — 12 forwards × 8.0 tokens/pass</div>
            <div class="x4-row-out"><span class="x4-diff-out">0 / 96</span> · <span class="x4-diff-passes">0</span> passes</div>
          </div>
          <div class="x4-track"><div class="x4-bar x4-diff-bar"></div></div>
        </div>
      </div>
    `;
    document.getElementById('vizX4-len')?.addEventListener('input', render);
    document.getElementById('vizX4-steps')?.addEventListener('input', render);
    document.getElementById('vizX4-play')?.addEventListener('click', play);
    document.getElementById('vizX4-reset')?.addEventListener('click', reset);
    window.addEventListener('langchange', render);
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
