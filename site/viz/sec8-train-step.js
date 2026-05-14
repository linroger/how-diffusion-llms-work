/* §8 — Training step in slow motion */
(function () {
  'use strict';

  let activeStep = 0;
  let t = 0.5;
  let seed = 17;

  const STEPS_EN = [
    { label: 'STEP 1', desc: 'Sample t ~ Uniform(0, 1)' },
    { label: 'STEP 2', desc: 'Corrupt: x_t[i] = [MASK] w.p. t' },
    { label: 'STEP 3', desc: 'Forward pass → logits at every position' },
    { label: 'STEP 4', desc: 'Cross-entropy on masked, × 1/t' },
    { label: 'STEP 5', desc: 'Backprop, optimizer step' },
  ];
  const STEPS_ZH = [
    { label: '步骤 1', desc: '采样 t ~ Uniform(0, 1)' },
    { label: '步骤 2', desc: '损坏：x_t[i] = [MASK] 以概率 t' },
    { label: '步骤 3', desc: '前向传播 → 每个位置的 logits' },
    { label: '步骤 4', desc: '仅 masked 位置的交叉熵, × 1/t' },
    { label: '步骤 5', desc: '反向传播、优化器步进' },
  ];

  function render() {
    const container = document.getElementById('viz8-1');
    if (!container) return;
    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    const steps = lang === 'zh' ? STEPS_ZH : STEPS_EN;
    const sentence = DLM.pickSentence(0);
    const rng = DLM.makeRNG(seed);

    container.innerHTML = '';

    // Step 1: sample t (slider/value)
    container.appendChild(makeRow(steps[0], activeStep >= 0, `t = ${t.toFixed(2)}`));

    // Step 2: corrupt
    const maskedState = sentence.map((tok) => {
      const masked = rng() < t;
      return { tok, masked };
    });
    const corrupted = document.createElement('div');
    corrupted.style.display = 'flex';
    corrupted.style.gap = '4px';
    corrupted.style.flexWrap = 'wrap';
    if (activeStep >= 1) {
      maskedState.forEach((c) => {
        const el = document.createElement('span');
        el.className = 'token ' + (c.masked ? 'mask' : 'committed');
        el.textContent = c.masked ? DLM.MASK : c.tok;
        corrupted.appendChild(el);
      });
    }
    container.appendChild(makeRow(steps[1], activeStep >= 1, corrupted));

    // Step 3: predictions
    const predictions = document.createElement('div');
    predictions.style.display = 'flex';
    predictions.style.gap = '4px';
    predictions.style.flexWrap = 'wrap';
    if (activeStep >= 2) {
      maskedState.forEach((c) => {
        const el = document.createElement('span');
        el.className = 'token ' + (c.masked ? 'predicted' : 'committed');
        el.style.opacity = c.masked ? '0.9' : '0.4';
        el.textContent = c.tok;
        predictions.appendChild(el);
      });
    }
    container.appendChild(makeRow(steps[2], activeStep >= 2, predictions));

    // Step 4: loss
    const lossView = document.createElement('div');
    lossView.style.display = 'flex';
    lossView.style.alignItems = 'center';
    lossView.style.gap = '12px';
    lossView.style.fontFamily = 'JetBrains Mono, monospace';
    lossView.style.fontSize = '0.86rem';
    if (activeStep >= 3) {
      const maskCount = maskedState.filter((c) => c.masked).length;
      const fakeLoss = (1.4 + 0.7 * Math.sin(seed)).toFixed(3);
      const weight = (1 / t).toFixed(2);
      const final = (parseFloat(fakeLoss) * (1 / t)).toFixed(3);
      lossView.innerHTML = `
        <span style="color:#a9a8a3">masked: <b style="color:#fbfaf7">${maskCount}</b></span>
        <span style="color:#5fb0c7">CE = ${fakeLoss}</span>
        <span style="color:#a9a8a3">×</span>
        <span style="color:#f5b54a">1/t = ${weight}</span>
        <span style="color:#a9a8a3">=</span>
        <span style="color:#fbfaf7;font-weight:600">L = ${final}</span>
      `;
    }
    container.appendChild(makeRow(steps[3], activeStep >= 3, lossView));

    // Step 5: backprop
    const back = document.createElement('div');
    back.style.fontFamily = 'JetBrains Mono, monospace';
    back.style.fontSize = '0.86rem';
    back.style.color = '#a9a8a3';
    if (activeStep >= 4) {
      back.innerHTML = '<span style="color:#8ec07c">✓ ∂L/∂θ computed</span>  <span style="color:#5fb0c7">→ AdamW step</span>  <span style="color:#a9a8a3">→ next batch</span>';
    }
    container.appendChild(makeRow(steps[4], activeStep >= 4, back));
  }

  function makeRow(stepInfo, active, body) {
    const row = document.createElement('div');
    row.className = 'train-row' + (active ? ' active' : '');
    const lbl = document.createElement('div');
    lbl.className = 'train-row-label';
    lbl.textContent = stepInfo.label;
    const bodyWrap = document.createElement('div');
    if (typeof body === 'string') bodyWrap.textContent = body;
    else if (body) bodyWrap.appendChild(body);
    else bodyWrap.innerHTML = '<span style="color:#a9a8a3">' + stepInfo.desc + '</span>';
    row.appendChild(lbl);
    row.appendChild(bodyWrap);
    return row;
  }

  function next() {
    activeStep = Math.min(4, activeStep + 1);
    render();
  }
  function reset() { activeStep = 0; render(); }
  function fresh() {
    seed = Math.floor(Math.random() * 100000);
    t = 0.2 + Math.random() * 0.6;
    activeStep = 0;
    render();
  }

  function init() {
    document.getElementById('viz8-1-next')?.addEventListener('click', next);
    document.getElementById('viz8-1-reset')?.addEventListener('click', reset);
    document.getElementById('viz8-1-newt')?.addEventListener('click', fresh);
    window.addEventListener('langchange', render);
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
