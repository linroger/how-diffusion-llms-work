/* §19 — Diffusion playground: pick sentence, block size, steps, strategy */
(function () {
  'use strict';

  let sentenceIdx = 0;
  let blockSize = 4;
  let totalSteps = 4;
  let strategy = 'lowconf';
  let activeBlock = 0;
  let intraStep = 0;
  let playing = false;
  let timer = null;

  function render() {
    const container = document.getElementById('viz19-1');
    if (!container) return;
    container.innerHTML = '';
    const lang = document.documentElement.getAttribute('data-lang') || 'en';

    // Controls row
    const controls = document.createElement('div');
    controls.style.display = 'grid';
    controls.style.gridTemplateColumns = 'repeat(auto-fit, minmax(180px, 1fr))';
    controls.style.gap = '14px';
    controls.style.marginBottom = '20px';
    controls.innerHTML = `
      <div>
        <div style="font-family:Inter,sans-serif;font-size:0.74rem;color:#a9a8a3;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px">${lang === 'zh' ? '句子' : 'Sentence'}</div>
        <select id="pg-sentence" style="width:100%;background:#1c2230;border:1px solid #2e3648;color:#e8e6e1;padding:6px 8px;border-radius:4px;font-family:JetBrains Mono,monospace;font-size:0.82rem">
          <option value="0">${lang === 'zh' ? '扩散模型生成…' : 'Diffusion models...'}</option>
          <option value="1">${lang === 'zh' ? '敏捷的棕色狐狸…' : 'The quick brown fox...'}</option>
          <option value="2">def add(a, b): return a + b</option>
          <option value="3">${lang === 'zh' ? '双向注意力…' : 'Bidirectional attention...'}</option>
        </select>
      </div>
      <div>
        <div style="font-family:Inter,sans-serif;font-size:0.74rem;color:#a9a8a3;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px">${lang === 'zh' ? '块大小' : 'Block size'}: ${blockSize}</div>
        <input type="range" id="pg-bs" min="2" max="8" value="${blockSize}" style="width:100%">
      </div>
      <div>
        <div style="font-family:Inter,sans-serif;font-size:0.74rem;color:#a9a8a3;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px">${lang === 'zh' ? '每块步数' : 'Steps per block'}: ${totalSteps}</div>
        <input type="range" id="pg-steps" min="1" max="6" value="${totalSteps}" style="width:100%">
      </div>
      <div>
        <div style="font-family:Inter,sans-serif;font-size:0.74rem;color:#a9a8a3;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px">${lang === 'zh' ? '策略' : 'Strategy'}</div>
        <select id="pg-strat" style="width:100%;background:#1c2230;border:1px solid #2e3648;color:#e8e6e1;padding:6px 8px;border-radius:4px;font-family:JetBrains Mono,monospace;font-size:0.82rem">
          <option value="random">${lang === 'zh' ? '随机' : 'Random'}</option>
          <option value="lowconf" selected>${lang === 'zh' ? '低置信度' : 'Low-confidence'}</option>
          <option value="threshold">${lang === 'zh' ? '阈值' : 'Threshold'}</option>
        </select>
      </div>
    `;
    container.appendChild(controls);

    // The actual viz area
    const vizArea = document.createElement('div');
    vizArea.style.minHeight = '120px';
    vizArea.style.padding = '20px';
    vizArea.style.background = '#11151f';
    vizArea.style.borderRadius = '6px';

    const tokens = DLM.pickSentence(sentenceIdx);
    const numBlocks = Math.ceil(tokens.length / blockSize);

    const row = document.createElement('div');
    row.style.display = 'flex';
    row.style.flexWrap = 'wrap';
    row.style.gap = '8px';

    for (let b = 0; b < numBlocks; b++) {
      const group = document.createElement('div');
      group.style.display = 'flex';
      group.style.gap = '3px';
      group.style.padding = '6px';
      group.style.border = '1.5px ' + (b === activeBlock ? 'solid #f5b54a' : (b < activeBlock ? 'solid #8ec07c' : 'dashed #2e3648')) + '';
      group.style.borderRadius = '6px';
      const start = b * blockSize;
      const end = Math.min(tokens.length, start + blockSize);

      for (let i = start; i < end; i++) {
        const el = document.createElement('span');
        el.className = 'token';
        const posInBlock = i - start;
        if (b < activeBlock) {
          el.classList.add('committed');
          el.textContent = tokens[i];
        } else if (b === activeBlock) {
          const commitAt = 1 + Math.floor(posInBlock / Math.max(1, (end - start) / totalSteps));
          if (intraStep >= commitAt) {
            el.classList.add(intraStep === commitAt ? 'commit' : 'committed');
            el.textContent = tokens[i];
          } else {
            el.classList.add('mask');
            el.textContent = DLM.MASK;
          }
        } else {
          el.classList.add('mask');
          el.textContent = DLM.MASK;
        }
        group.appendChild(el);
      }
      row.appendChild(group);
    }
    vizArea.appendChild(row);

    // status
    const status = document.createElement('div');
    status.style.marginTop = '14px';
    status.style.fontFamily = 'JetBrains Mono, monospace';
    status.style.fontSize = '0.82rem';
    status.style.color = '#a9a8a3';
    const totalForwards = activeBlock * totalSteps + Math.min(intraStep, totalSteps);
    status.innerHTML = `
      ${lang === 'zh' ? '总前向' : 'total forwards'}: <b style="color:#f5b54a">${totalForwards}</b>
      &nbsp;·&nbsp;
      ${lang === 'zh' ? '块' : 'block'}: <b>${Math.min(activeBlock, numBlocks)} / ${numBlocks}</b>
      &nbsp;·&nbsp;
      AR-equivalent: ${tokens.length} forwards
    `;
    vizArea.appendChild(status);

    container.appendChild(vizArea);

    // Action buttons
    const actions = document.createElement('div');
    actions.style.display = 'flex';
    actions.style.gap = '10px';
    actions.style.marginTop = '14px';
    actions.innerHTML = `
      <button class="btn" id="pg-play">▶ ${lang === 'zh' ? '播放' : 'Play'}</button>
      <button class="btn btn-ghost" id="pg-reset">↺ ${lang === 'zh' ? '重置' : 'Reset'}</button>
    `;
    container.appendChild(actions);

    // Wire up
    document.getElementById('pg-sentence').addEventListener('change', (e) => { sentenceIdx = parseInt(e.target.value); reset(); });
    document.getElementById('pg-bs').addEventListener('input', (e) => { blockSize = parseInt(e.target.value); reset(); });
    document.getElementById('pg-steps').addEventListener('input', (e) => { totalSteps = parseInt(e.target.value); reset(); });
    document.getElementById('pg-strat').addEventListener('change', (e) => { strategy = e.target.value; reset(); });
    document.getElementById('pg-play').addEventListener('click', play);
    document.getElementById('pg-reset').addEventListener('click', reset);
  }

  function play() {
    if (playing) return;
    playing = true;
    activeBlock = 0;
    intraStep = 0;
    render();

    const tokens = DLM.pickSentence(sentenceIdx);
    const numBlocks = Math.ceil(tokens.length / blockSize);

    const tick = () => {
      intraStep++;
      if (intraStep > totalSteps) {
        intraStep = 0;
        activeBlock++;
      }
      if (activeBlock >= numBlocks) {
        render();
        playing = false;
        return;
      }
      render();
      timer = setTimeout(tick, 450);
    };
    timer = setTimeout(tick, 400);
  }

  function reset() {
    if (timer) clearTimeout(timer);
    playing = false;
    activeBlock = 0;
    intraStep = 0;
    render();
  }

  function init() {
    render();
    window.addEventListener('langchange', render);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
