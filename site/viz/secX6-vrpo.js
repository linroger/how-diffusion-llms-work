/* Extra 6 — VRPO variance demo
   Shows two ELBO trajectories side-by-side: naive Monte Carlo (high variance, jagged)
   vs antithetic sampling (paired noise, lower variance, smoother).
   Each "iteration" advances both — see how the gap between the two converges. */
(function () {
  'use strict';

  let timer = null;
  let playing = false;
  let step = 0;
  const TOTAL = 80;

  // Pre-compute traces deterministically
  function makeTraces(seed) {
    const rngA = DLM.makeRNG(seed);
    const rngB = DLM.makeRNG(seed + 17);
    const traceNaive = []; // each estimate is mean +/- noise
    const traceAnti = [];
    const targetLogP = -1.4; // "true" log-likelihood
    for (let i = 0; i < TOTAL; i++) {
      // Naive: independent draws, variance ~ 0.4
      const noiseA = (rngA() - 0.5) * 0.9;
      traceNaive.push(targetLogP + noiseA);
      // Antithetic: variance ~ 0.07
      const noiseB = (rngB() - 0.5) * 0.18;
      traceAnti.push(targetLogP + noiseB);
    }
    return { naive: traceNaive, anti: traceAnti, target: targetLogP };
  }

  function render() {
    const container = document.getElementById('vizX6');
    if (!container) return;
    const lang = document.documentElement.getAttribute('data-lang') || 'en';

    const { naive, anti, target } = makeTraces(91);

    // Build SVG once if needed
    if (container.children.length === 0) {
      container.innerHTML = `
        <div class="vrpo-grid">
          <div class="vrpo-pane">
            <div class="vrpo-pane-title vrpo-naive-title">${lang === 'zh' ? '朴素采样 — 每次新随机数' : 'Naive sampling — fresh randomness each call'}</div>
            <svg class="vrpo-svg" id="vrpo-svg-naive" viewBox="0 0 320 160"></svg>
            <div class="vrpo-readout"><span class="vrpo-lbl">${lang === 'zh' ? '梯度方差' : 'gradient variance'}</span><span class="vrpo-var vrpo-var-naive">0.41</span></div>
          </div>
          <div class="vrpo-pane">
            <div class="vrpo-pane-title vrpo-anti-title">${lang === 'zh' ? '对偶采样 — 策略与参考共享噪声' : 'Antithetic — policy & reference share noise'}</div>
            <svg class="vrpo-svg" id="vrpo-svg-anti" viewBox="0 0 320 160"></svg>
            <div class="vrpo-readout"><span class="vrpo-lbl">${lang === 'zh' ? '梯度方差' : 'gradient variance'}</span><span class="vrpo-var vrpo-var-anti">0.07</span></div>
          </div>
        </div>
        <div class="vrpo-caption">${lang === 'zh' ?
          '同一个真实 log-likelihood（虚线）下，朴素 ELBO 估计上下颠簸（左图）；对偶采样让策略和参考使用同一组随机 mask，差分时噪声大部分相消（右图）。后者训练曲线更稳，超参更宽容。' :
          'Same true log-likelihood (dashed line). Naive ELBO bounces around (left); antithetic sampling uses the same random masks for policy and reference, so most of the noise cancels in the difference (right). The training curve is smoother, hyperparameters more forgiving.'
        }</div>`;
    } else {
      container.querySelector('.vrpo-naive-title').textContent = lang === 'zh' ? '朴素采样 — 每次新随机数' : 'Naive sampling — fresh randomness each call';
      container.querySelector('.vrpo-anti-title').textContent = lang === 'zh' ? '对偶采样 — 策略与参考共享噪声' : 'Antithetic — policy & reference share noise';
      container.querySelector('.vrpo-caption').textContent = lang === 'zh' ?
          '同一个真实 log-likelihood（虚线）下，朴素 ELBO 估计上下颠簸（左图）；对偶采样让策略和参考使用同一组随机 mask，差分时噪声大部分相消（右图）。后者训练曲线更稳，超参更宽容。' :
          'Same true log-likelihood (dashed line). Naive ELBO bounces around (left); antithetic sampling uses the same random masks for policy and reference, so most of the noise cancels in the difference (right). The training curve is smoother, hyperparameters more forgiving.';
    }

    drawTrace('vrpo-svg-naive', naive, target, step);
    drawTrace('vrpo-svg-anti', anti, target, step);
  }

  function drawTrace(id, data, target, upTo) {
    const svg = document.getElementById(id);
    if (!svg) return;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const W = 320, H = 160;
    const PAD = 16;
    const yMin = -2.2, yMax = -0.6;
    const xs = (i) => PAD + (i / (TOTAL - 1)) * (W - 2 * PAD);
    const ys = (v) => H - PAD - ((v - yMin) / (yMax - yMin)) * (H - 2 * PAD);

    // target line
    const css = getComputedStyle(document.documentElement);
    const muted = css.getPropertyValue('--text-muted').trim() || '#777';
    const accent = css.getPropertyValue('--accent').trim() || '#1677ff';

    const targetLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    targetLine.setAttribute('x1', PAD);
    targetLine.setAttribute('y1', ys(target));
    targetLine.setAttribute('x2', W - PAD);
    targetLine.setAttribute('y2', ys(target));
    targetLine.setAttribute('stroke', muted);
    targetLine.setAttribute('stroke-width', '1');
    targetLine.setAttribute('stroke-dasharray', '3,4');
    targetLine.setAttribute('opacity', '0.45');
    svg.appendChild(targetLine);

    // path
    const visible = Math.min(upTo, data.length);
    let path = '';
    for (let i = 0; i < visible; i++) {
      path += (i === 0 ? 'M' : 'L') + xs(i).toFixed(1) + ',' + ys(data[i]).toFixed(1) + ' ';
    }
    if (path) {
      const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      p.setAttribute('d', path);
      p.setAttribute('fill', 'none');
      p.setAttribute('stroke', accent);
      p.setAttribute('stroke-width', '1.8');
      p.setAttribute('stroke-linejoin', 'round');
      svg.appendChild(p);
      // dot at the current head
      if (visible > 0) {
        const cur = visible - 1;
        const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        c.setAttribute('cx', xs(cur));
        c.setAttribute('cy', ys(data[cur]));
        c.setAttribute('r', '3');
        c.setAttribute('fill', accent);
        svg.appendChild(c);
      }
    }
  }

  function play() {
    if (playing) return;
    playing = true;
    step = 0;
    render();
    const tick = () => {
      step++;
      if (step > TOTAL) { playing = false; return; }
      render();
      timer = setTimeout(tick, 70);
    };
    timer = setTimeout(tick, 200);
  }

  function reset() {
    if (timer) clearTimeout(timer);
    playing = false;
    step = 0;
    render();
  }

  function init() {
    const container = document.getElementById('vizX6');
    if (!container) return;
    document.getElementById('vizX6-play')?.addEventListener('click', play);
    document.getElementById('vizX6-reset')?.addEventListener('click', reset);
    window.addEventListener('langchange', () => { container.innerHTML = ''; render(); });
    window.addEventListener('palettechange', render);
    step = TOTAL;
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
