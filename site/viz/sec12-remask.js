/* §12 — Four remasking strategies side-by-side */
(function () {
  'use strict';

  const STRATEGIES = [
    { id: 'random', en: 'Random', zh: '随机', desc_en: 'Pick positions uniformly at random.', desc_zh: '均匀随机选取位置。', color: 'var(--text-muted)' },
    { id: 'lowconf', en: 'Low-confidence', zh: '低置信度', desc_en: 'Commit highest-confidence positions; remask the rest.', desc_zh: '提交最高置信度位置；其余重新掩码。', color: 'var(--accent-2)' },
    { id: 'threshold', en: 'Confidence-threshold (τ=0.7)', zh: '置信度阈值 (τ=0.7)', desc_en: 'Commit any position above τ.', desc_zh: '提交所有超过 τ 的位置。', color: 'var(--accent)' },
    { id: 'margin', en: 'Margin (top-1 − top-2)', zh: '差距 (top-1 − top-2)', desc_en: 'Commit positions with large margin between best two.', desc_zh: '提交两个最佳候选差距大的位置。', color: 'var(--accent-3)' },
  ];

  const N = 8;
  let target = null;
  let confidences = null;
  let margins = null;
  let steps = {};
  let currentStep = 0;
  let playing = false;
  let timer = null;

  function setup() {
    target = DLM.pickSentence(0);
    confidences = DLM.makeConfidence(target, 7);
    const rng = DLM.makeRNG(101);
    margins = target.map(() => 0.2 + rng() * 0.5);

    // For each strategy, compute commit-step per position
    steps = {};
    STRATEGIES.forEach((s) => {
      const order = computeOrder(s.id);
      const commitStep = new Array(target.length);
      const perStep = Math.max(1, Math.ceil(target.length / N));
      order.forEach((idx, rank) => {
        commitStep[idx] = Math.min(N, 1 + Math.floor(rank / perStep));
      });
      steps[s.id] = commitStep;
    });
  }

  function computeOrder(strategy) {
    if (strategy === 'random') {
      const rng = DLM.makeRNG(999);
      return target.map((_, i) => i).sort(() => rng() - 0.5);
    }
    if (strategy === 'lowconf') {
      return target.map((_, i) => i).sort((a, b) => confidences[b] - confidences[a]);
    }
    if (strategy === 'threshold') {
      const above = target.map((_, i) => i).filter((i) => confidences[i] >= 0.7).sort((a, b) => confidences[b] - confidences[a]);
      const below = target.map((_, i) => i).filter((i) => confidences[i] < 0.7).sort((a, b) => confidences[b] - confidences[a]);
      return above.concat(below);
    }
    if (strategy === 'margin') {
      return target.map((_, i) => i).sort((a, b) => margins[b] - margins[a]);
    }
    return target.map((_, i) => i);
  }

  function render() {
    const container = document.getElementById('viz12-1');
    if (!container) return;
    if (!target) setup();
    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    container.innerHTML = '';

    STRATEGIES.forEach((s) => {
      const pane = document.createElement('div');
      pane.className = 'remask-pane';
      const title = document.createElement('div');
      title.className = 'pane-title';
      title.style.color = s.color;
      title.style.borderBottomColor = s.color;
      title.textContent = lang === 'zh' ? s.zh : s.en;
      pane.appendChild(title);

      const desc = document.createElement('div');
      desc.style.fontFamily = 'Inter, sans-serif';
      desc.style.fontSize = '0.78rem';
      desc.style.color = 'var(--text-soft)';
      desc.style.marginBottom = '10px';
      desc.textContent = lang === 'zh' ? s.desc_zh : s.desc_en;
      pane.appendChild(desc);

      const strip = document.createElement('div');
      strip.className = 'token-strip';
      strip.style.padding = '0';
      strip.style.minHeight = '40px';
      target.forEach((tok, i) => {
        const el = document.createElement('span');
        el.className = 'token';
        const cs = steps[s.id][i];
        if (cs <= currentStep) {
          el.classList.add(cs === currentStep ? 'commit' : 'committed');
          el.textContent = tok;
        } else {
          el.classList.add('mask');
          el.textContent = DLM.MASK;
        }
        strip.appendChild(el);
      });
      pane.appendChild(strip);

      const stepInfo = document.createElement('div');
      stepInfo.className = 'pane-meta';
      const committed = target.filter((_, i) => steps[s.id][i] <= currentStep).length;
      stepInfo.textContent = `step ${currentStep} · ${committed}/${target.length}`;
      pane.appendChild(stepInfo);

      container.appendChild(pane);
    });
  }

  function play() {
    if (playing) return;
    playing = true;
    currentStep = 0;
    render();
    const tick = () => {
      currentStep++;
      render();
      if (currentStep < N) timer = setTimeout(tick, 700);
      else playing = false;
    };
    timer = setTimeout(tick, 500);
  }
  function reset() {
    if (timer) clearTimeout(timer);
    playing = false;
    currentStep = 0;
    setup();
    render();
  }

  function init() {
    document.getElementById('viz12-1-play')?.addEventListener('click', play);
    document.getElementById('viz12-1-reset')?.addEventListener('click', reset);
    window.addEventListener('langchange', () => { setup(); render(); });
    setup();
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
