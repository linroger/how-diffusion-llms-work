/* §13 — Block diffusion animator: blocks resolve left-to-right with intra-block denoising */
(function () {
  'use strict';

  let blockSize = 8;
  let activeBlock = 0;
  let intraStep = 0;
  let playing = false;
  let timer = null;

  // For demo we use a longer sentence — 24 tokens
  function makeTokens() {
    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    if (lang === 'zh') {
      return ['扩散', '模型', '通过', '对', '一', '条', 'masked', 'token', '序列', '执行', '迭代', '去噪', '来', '生成', '文本', '与', '代码', '，', '这', '与', '自回归', '模型', '完全', '不同', '。'];
    }
    return ['Diffusion', 'models', 'generate', 'text', 'by', 'iteratively', 'denoising', 'a', 'tape', 'of', 'masked', 'tokens', ',', 'which', 'is', 'fundamentally', 'different', 'from', 'how', 'autoregressive', 'models', 'work', 'left', 'to', 'right'];
  }

  function render() {
    const container = document.getElementById('viz13-1');
    if (!container) return;
    container.innerHTML = '';

    const tokens = makeTokens();
    const numBlocks = Math.ceil(tokens.length / blockSize);
    const lang = document.documentElement.getAttribute('data-lang') || 'en';

    const row = document.createElement('div');
    row.className = 'block-row';

    for (let b = 0; b < numBlocks; b++) {
      const group = document.createElement('div');
      group.className = 'block-group';
      if (b < activeBlock) group.classList.add('committed');
      if (b === activeBlock) group.classList.add('active');

      const start = b * blockSize;
      const end = Math.min(tokens.length, start + blockSize);
      const totalIntra = 3; // 3 intra-block denoise steps per block

      for (let i = start; i < end; i++) {
        const el = document.createElement('span');
        el.className = 'token';
        let tokenShown = DLM.MASK;
        let cls = 'mask';

        if (b < activeBlock) {
          // committed
          tokenShown = tokens[i];
          cls = 'committed';
        } else if (b === activeBlock) {
          // intra-block — commit ~1/3 per intra step, by position in block
          const posInBlock = i - start;
          const commitAtIntra = 1 + Math.floor(posInBlock / Math.max(1, (end - start) / totalIntra));
          if (intraStep >= commitAtIntra) {
            tokenShown = tokens[i];
            cls = intraStep === commitAtIntra ? 'commit' : 'committed';
          }
        }

        el.classList.add(cls);
        el.textContent = tokenShown;
        group.appendChild(el);
      }

      // Block label
      const lbl = document.createElement('div');
      lbl.style.position = 'absolute';
      lbl.style.top = '-18px';
      lbl.style.left = '4px';
      lbl.style.fontFamily = 'JetBrains Mono, monospace';
      lbl.style.fontSize = '0.7rem';
      lbl.style.color = b === activeBlock ? 'var(--accent)' : (b < activeBlock ? 'var(--accent-4)' : 'var(--text-muted)');
      lbl.textContent = `block ${b}`;
      group.appendChild(lbl);

      row.appendChild(group);
    }
    container.appendChild(row);

    // Status line
    const status = document.createElement('div');
    status.style.fontFamily = 'JetBrains Mono, monospace';
    status.style.fontSize = '0.84rem';
    status.style.color = 'var(--text-soft)';
    status.style.marginTop = '20px';
    if (activeBlock >= numBlocks) {
      status.innerHTML = `<span style="color:var(--accent-4)">✓ ${lang === 'zh' ? '所有块解码完毕' : 'all blocks decoded'}</span>`;
    } else {
      status.innerHTML = `<span>${lang === 'zh' ? '活跃块' : 'active block'}: <b style="color:var(--accent)">${activeBlock}</b></span>  ·  <span>${lang === 'zh' ? '块内步骤' : 'intra-block step'}: <b style="color:var(--accent)">${intraStep}</b> / 3</span>  ·  <span style="color:var(--accent-4)">${activeBlock} ${lang === 'zh' ? '块已缓存' : 'blocks cached'}</span>`;
    }
    container.appendChild(status);

    // Legend
    const legend = document.createElement('div');
    legend.style.display = 'flex';
    legend.style.gap = '18px';
    legend.style.fontFamily = 'Inter, sans-serif';
    legend.style.fontSize = '0.76rem';
    legend.style.color = 'var(--text-soft)';
    legend.style.marginTop = '8px';
    legend.innerHTML = `
      <span><span style="display:inline-block;width:10px;height:10px;background:var(--accent-4);border-radius:2px;vertical-align:middle"></span> ${lang === 'zh' ? '已提交 + 缓存' : 'committed + cached'}</span>
      <span><span style="display:inline-block;width:10px;height:10px;background:var(--accent);border-radius:2px;vertical-align:middle"></span> ${lang === 'zh' ? '活跃 (双向)' : 'active (bidirectional)'}</span>
      <span><span style="display:inline-block;width:10px;height:10px;background:#2a2e3a;border-radius:2px;border:1px solid #4a5066;vertical-align:middle"></span> ${lang === 'zh' ? '未解码' : 'not yet decoded'}</span>
    `;
    container.appendChild(legend);
  }

  function play() {
    if (playing) return;
    playing = true;
    activeBlock = 0;
    intraStep = 0;
    render();

    const tick = () => {
      const tokens = makeTokens();
      const numBlocks = Math.ceil(tokens.length / blockSize);
      intraStep++;
      if (intraStep > 3) {
        intraStep = 0;
        activeBlock++;
      }
      if (activeBlock >= numBlocks) {
        render();
        playing = false;
        return;
      }
      render();
      timer = setTimeout(tick, 550);
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
    const slider = document.getElementById('viz13-1-bs');
    const bsval = document.getElementById('viz13-1-bsval');
    if (slider) {
      slider.addEventListener('input', () => {
        blockSize = parseInt(slider.value);
        if (bsval) bsval.textContent = blockSize;
        reset();
      });
    }
    document.getElementById('viz13-1-play')?.addEventListener('click', play);
    document.getElementById('viz13-1-reset')?.addEventListener('click', reset);
    window.addEventListener('langchange', render);
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
