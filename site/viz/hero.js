/* Hero — animated 4-step diffusion decoding */
(function () {
  'use strict';

  const STEPS = [
    // Step 0: all masked
    'mmmmmmmmmmmm',
    // Step 1: a couple commit
    'mmmmmmmmM01M',
    // Step 2: a few more
    'M0mmmM2M5M01M',
    // Step 3: most committed
    'M0M1m3M4M5M01M7M89',
    // Step 4: fully resolved
    'TARGET',
  ];

  function render(stepIdx) {
    const container = document.getElementById('hero-demo');
    if (!container) return;
    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    const target = DLM.pickSentence(0);
    container.innerHTML = '';

    // Decide which positions are committed at each step. Deterministic.
    const commitSchedule = computeCommitSchedule(target.length, 5);

    target.forEach((tok, i) => {
      const cell = document.createElement('span');
      cell.className = 'token';
      const committedAt = commitSchedule[i];
      if (stepIdx >= committedAt) {
        cell.classList.add(committedAt === stepIdx ? 'commit' : 'committed');
        cell.textContent = tok;
      } else {
        cell.classList.add('mask');
        cell.textContent = DLM.MASK;
      }
      cell.style.animationDelay = (i * 25) + 'ms';
      container.appendChild(cell);
    });

    const stepReadout = document.getElementById('heroStep');
    if (stepReadout) {
      const playLabel = lang === 'zh' ? '步骤' : 'step';
      const ofLabel = lang === 'zh' ? '/' : '/';
      stepReadout.textContent = `${playLabel} ${stepIdx} ${ofLabel} 4`;
    }
  }

  function computeCommitSchedule(n, steps) {
    // assign each position a step at which it commits (1..steps-1)
    // earlier positions tend to commit slightly earlier
    const result = new Array(n);
    const rng = DLM.makeRNG(42);
    for (let i = 0; i < n; i++) {
      // bias forward positions to commit earlier with some randomness
      const r = rng();
      const bias = (i / n) * 0.3;
      const val = r + bias;
      if (val < 0.20) result[i] = 1;
      else if (val < 0.50) result[i] = 2;
      else if (val < 0.80) result[i] = 3;
      else result[i] = 4;
    }
    return result;
  }

  let currentStep = 0;
  let playing = false;
  let timer = null;

  function play() {
    if (playing) return;
    playing = true;
    currentStep = 0;
    render(0);
    const step = () => {
      currentStep++;
      if (currentStep > 4) {
        playing = false;
        return;
      }
      render(currentStep);
      timer = setTimeout(step, 750);
    };
    timer = setTimeout(step, 600);
  }

  function reset() {
    if (timer) clearTimeout(timer);
    playing = false;
    currentStep = 0;
    render(0);
  }

  function init() {
    document.getElementById('heroPlay')?.addEventListener('click', play);
    document.getElementById('heroReset')?.addEventListener('click', reset);
    window.addEventListener('langchange', () => render(currentStep));
    render(0);
    // Auto-play once on load after a delay
    setTimeout(play, 800);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
