/* §14 — CAP confidence-threshold parallel decoding */
(function () {
  'use strict';

  let tau = 0.95;
  let playing = false;
  let timer = null;
  let target = null;
  let confidences = null;
  let stepLog = [];

  function setup() {
    target = DLM.pickSentence(3);
    confidences = DLM.makeConfidence(target, 53);
    stepLog = [];
  }

  function simulate() {
    setup();
    // Each step: commit all positions whose conf > tau
    // Then "re-evaluate" — confidences nudge upward for remaining positions because context improves
    let masked = target.map(() => true);
    let confSnapshot = [...confidences];
    let stepNum = 0;
    const maxSteps = 12;
    while (masked.some((m) => m) && stepNum < maxSteps) {
      stepNum++;
      const willCommit = masked.map((m, i) => m && confSnapshot[i] > tau);
      const committedThisStep = willCommit.filter((x) => x).length;
      stepLog.push({
        step: stepNum,
        committedThisStep,
        totalCommitted: masked.filter((m, i) => !m || willCommit[i]).length,
        masked: masked.map((m, i) => m && !willCommit[i]),
        confs: [...confSnapshot],
      });
      // Apply commits
      willCommit.forEach((c, i) => { if (c) masked[i] = false; });
      // If no positions committed (tau too high), force at least one to break deadlock
      if (committedThisStep === 0) {
        // commit the highest-confidence still-masked
        let maxI = -1, maxV = -1;
        masked.forEach((m, i) => { if (m && confSnapshot[i] > maxV) { maxV = confSnapshot[i]; maxI = i; } });
        if (maxI >= 0) masked[maxI] = false;
      }
      // Bump up remaining confidences (context grows)
      confSnapshot = confSnapshot.map((c, i) => masked[i] ? Math.min(0.99, c + 0.06) : c);
    }
  }

  function render(stepIdx) {
    const container = document.getElementById('viz14-1');
    if (!container) return;
    if (!target) setup();
    const lang = document.documentElement.getAttribute('data-lang') || 'en';

    container.innerHTML = '';

    // Threshold visualization line: confidence per position
    const confChart = document.createElement('div');
    confChart.style.display = 'flex';
    confChart.style.gap = '4px';
    confChart.style.alignItems = 'flex-end';
    confChart.style.height = '64px';
    confChart.style.marginBottom = '10px';
    target.forEach((tok, i) => {
      const bar = document.createElement('div');
      bar.style.flex = '1';
      const conf = stepLog.length > 0 ? stepLog[Math.min(stepIdx, stepLog.length - 1)].confs[i] : confidences[i];
      bar.style.height = (conf * 100) + '%';
      const isMasked = stepLog.length > 0 && stepLog[Math.min(stepIdx, stepLog.length - 1)].masked[i];
      bar.style.background = (!isMasked) ? '#8ec07c' : (conf > tau ? '#f5b54a' : '#3b4458');
      bar.style.borderRadius = '2px 2px 0 0';
      bar.title = `${tok}: ${conf.toFixed(2)}`;
      confChart.appendChild(bar);
    });

    // Tau line
    const tauLine = document.createElement('div');
    tauLine.style.position = 'relative';
    tauLine.style.height = '0';
    tauLine.style.borderTop = '2px dashed #e879a8';
    tauLine.style.transform = `translateY(${-tau * 64}px)`;
    tauLine.innerHTML = `<span style="position:absolute;right:0;top:-18px;font-family:JetBrains Mono,monospace;font-size:0.72rem;color:#e879a8">τ = ${tau.toFixed(2)}</span>`;

    const chartWrap = document.createElement('div');
    chartWrap.style.position = 'relative';
    chartWrap.appendChild(confChart);
    chartWrap.appendChild(tauLine);
    container.appendChild(chartWrap);

    // Tokens at this step
    const strip = document.createElement('div');
    strip.className = 'token-strip';
    strip.style.padding = '0';
    target.forEach((tok, i) => {
      const el = document.createElement('span');
      el.className = 'token';
      const isMasked = stepLog.length > 0 ? stepLog[Math.min(stepIdx, stepLog.length - 1)].masked[i] : true;
      const willCommit = stepLog.length > 0 && stepIdx < stepLog.length && stepLog[stepIdx].confs[i] > tau && stepLog[Math.max(0, stepIdx - 1)] && stepLog[Math.max(0, stepIdx - 1)].masked[i];
      if (!isMasked) {
        el.classList.add(willCommit ? 'commit' : 'committed');
        el.textContent = tok;
      } else {
        el.classList.add('mask');
        el.textContent = DLM.MASK;
      }
      strip.appendChild(el);
    });
    container.appendChild(strip);

    // Stats
    const stats = document.createElement('div');
    stats.style.fontFamily = 'JetBrains Mono, monospace';
    stats.style.fontSize = '0.84rem';
    stats.style.marginTop = '10px';
    stats.style.color = '#a9a8a3';
    const finalSteps = stepLog.length;
    const finalCommitted = stepLog.length > 0 ? target.length - stepLog[stepLog.length - 1].masked.filter((m) => m).length : 0;
    const tpf = finalSteps > 0 ? (finalCommitted / finalSteps).toFixed(2) : '0.00';
    // Quality penalty for low tau (heuristic) — model 70.15 at 0.95, dropping toward 60 at 0.5
    const quality = (60 + (tau - 0.5) * 20.3).toFixed(1);
    stats.innerHTML = `
      <span>${lang === 'zh' ? '总步骤' : 'steps'}: <b style="color:#fbfaf7">${finalSteps}</b></span>
      &nbsp;·&nbsp;
      <span>${lang === 'zh' ? '每前向 token' : 'tokens-per-forward'}: <b style="color:#f5b54a">${tpf}</b></span>
      &nbsp;·&nbsp;
      <span>${lang === 'zh' ? '模拟基准分' : 'simulated score'}: <b style="color:${parseFloat(quality) > 68 ? '#8ec07c' : (parseFloat(quality) > 64 ? '#f5b54a' : '#e879a8')}">${quality}</b></span>
    `;
    container.appendChild(stats);
  }

  let currentStep = 0;

  function play() {
    if (playing) return;
    simulate();
    playing = true;
    currentStep = 0;
    render(0);
    const tick = () => {
      currentStep++;
      if (currentStep >= stepLog.length) {
        playing = false;
        return;
      }
      render(currentStep);
      timer = setTimeout(tick, 600);
    };
    timer = setTimeout(tick, 500);
  }

  function reset() {
    if (timer) clearTimeout(timer);
    playing = false;
    currentStep = 0;
    setup();
    render(0);
  }

  function init() {
    const slider = document.getElementById('viz14-1-tau');
    const tval = document.getElementById('viz14-1-tval');
    if (slider) {
      slider.addEventListener('input', () => {
        tau = parseInt(slider.value) / 100;
        if (tval) tval.textContent = tau.toFixed(2);
        simulate();
        currentStep = stepLog.length - 1;
        render(currentStep);
      });
    }
    document.getElementById('viz14-1-play')?.addEventListener('click', play);
    document.getElementById('viz14-1-reset')?.addEventListener('click', reset);
    window.addEventListener('langchange', () => { setup(); render(0); });
    simulate();
    render(0);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
