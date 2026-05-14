/* §11 — Denoising loop animator with prediction distributions */
(function () {
  'use strict';

  let N = 8;
  let currentStep = 0;
  let target = null;
  let confidences = null;
  let commitOrder = null;
  let playing = false;
  let timer = null;

  function setup() {
    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    target = DLM.pickSentence(0);
    confidences = DLM.makeConfidence(target, 33);
    // Each position commits at step = ceil(N * (1 - rank/length))
    // i.e., highest-confidence positions commit first
    const order = target.map((_, i) => i).sort((a, b) => confidences[b] - confidences[a]);
    commitOrder = new Array(target.length);
    order.forEach((idx, rank) => {
      // commit step = 1 + floor(rank / (target.length / N))
      commitOrder[idx] = Math.min(N, 1 + Math.floor(rank / Math.max(1, target.length / N)));
    });
  }

  function render() {
    const container = document.getElementById('viz11-1');
    if (!container) return;
    if (!target) setup();
    const lang = document.documentElement.getAttribute('data-lang') || 'en';

    container.innerHTML = '';

    // Render up to currentStep+1 step snapshots
    const maxShown = Math.min(N + 1, currentStep + 1);
    for (let s = 0; s < maxShown; s++) {
      const stepEl = document.createElement('div');
      stepEl.className = 'denoise-step';
      const header = document.createElement('div');
      header.className = 'denoise-step-header';
      const stepLabel = lang === 'zh' ? `第 ${s} 步` : `step ${s}`;
      const committed = target.filter((_, i) => commitOrder[i] <= s).length;
      header.innerHTML = `<span>${stepLabel}</span><span style="color:#a9a8a3">${committed} / ${target.length} ${lang === 'zh' ? '已提交' : 'committed'}</span>`;
      stepEl.appendChild(header);

      const strip = document.createElement('div');
      strip.className = 'token-strip';
      strip.style.padding = '0';
      strip.style.minHeight = '40px';
      target.forEach((tok, i) => {
        const el = document.createElement('span');
        el.className = 'token';
        if (commitOrder[i] <= s) {
          el.classList.add(commitOrder[i] === s ? 'commit' : 'committed');
          el.textContent = tok;
        } else {
          el.classList.add('mask');
          el.textContent = DLM.MASK;
          // Show small confidence indicator at the active step
          if (s === currentStep) {
            const c = confidences[i];
            el.title = `top-1 confidence: ${c.toFixed(2)}`;
            el.style.boxShadow = `0 0 0 1px rgba(95, 176, 199, ${c * 0.6})`;
          }
        }
        strip.appendChild(el);
      });
      stepEl.appendChild(strip);
      container.appendChild(stepEl);
    }
  }

  function step() {
    if (currentStep >= N) return;
    currentStep++;
    render();
  }
  function play() {
    if (playing) return;
    playing = true;
    const loop = () => {
      step();
      if (currentStep < N) timer = setTimeout(loop, 700);
      else playing = false;
    };
    timer = setTimeout(loop, 350);
  }
  function reset() {
    if (timer) clearTimeout(timer);
    playing = false;
    currentStep = 0;
    setup();
    render();
  }

  function init() {
    const slider = document.getElementById('viz11-1-n');
    const nval = document.getElementById('viz11-1-nval');
    if (slider) {
      slider.addEventListener('input', () => {
        N = parseInt(slider.value);
        if (nval) nval.textContent = N;
        reset();
      });
    }
    document.getElementById('viz11-1-step')?.addEventListener('click', step);
    document.getElementById('viz11-1-play')?.addEventListener('click', play);
    document.getElementById('viz11-1-reset')?.addEventListener('click', reset);
    window.addEventListener('langchange', () => { setup(); render(); });
    setup();
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
