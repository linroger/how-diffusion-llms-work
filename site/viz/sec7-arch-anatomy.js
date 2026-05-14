/* §7 — Architecture anatomy: AR Llama vs LLaDA-8B side-by-side */
(function () {
  'use strict';

  const LAYERS_EN = [
    { name: 'Embedding', ar: 'vocab 128k, dim 4096', diff: 'vocab 128k + [MASK], dim 4096', changed: true, why: 'A dedicated [MASK] token is added to the vocabulary as the absorbing state.' },
    { name: 'Position', ar: 'RoPE (base 500k)', diff: 'RoPE (base 500k)', changed: false, why: 'Position encoding is unchanged.' },
    { name: 'Attention', ar: 'Causal mask, GQA 32:8', diff: 'Bidirectional, no GQA (32 q-heads)', changed: true, why: 'The single biggest change. Bidirectional means every position attends every position. LLaDA also drops grouped-query attention.' },
    { name: 'Feed-forward', ar: 'SwiGLU, dim 14336', diff: 'SwiGLU, dim 14336', changed: false, why: 'Unchanged.' },
    { name: 'Norm', ar: 'RMSNorm', diff: 'RMSNorm', changed: false, why: 'Unchanged.' },
    { name: 'Time / t embed', ar: '—', diff: 'sinusoidal(t) → AdaLN scale/shift', changed: true, why: 'The mask rate t is injected as a conditioning signal, similar to image diffusion timestep conditioning.' },
    { name: 'Loss', ar: 'Next-token CE on all positions', diff: 'Weighted CE on masked only (× 1/t)', changed: true, why: 'Loss is the MD4 simplification — weighted MLM on masked positions.' },
  ];

  const LAYERS_ZH = [
    { name: '嵌入', ar: '词表 128k, 维度 4096', diff: '词表 128k + [MASK], 维度 4096', changed: true, why: '词表中加入专用 [MASK] 作为吸收态。' },
    { name: '位置编码', ar: 'RoPE (base 500k)', diff: 'RoPE (base 500k)', changed: false, why: '位置编码不变。' },
    { name: '注意力', ar: '因果 mask, GQA 32:8', diff: '双向, 不使用 GQA (32 查询头)', changed: true, why: '最大单点变化。双向 = 每个位置关注所有位置。LLaDA 也移除了分组查询注意力。' },
    { name: '前馈', ar: 'SwiGLU, 维度 14336', diff: 'SwiGLU, 维度 14336', changed: false, why: '不变。' },
    { name: '归一化', ar: 'RMSNorm', diff: 'RMSNorm', changed: false, why: '不变。' },
    { name: '时间 / t 嵌入', ar: '—', diff: 'sinusoidal(t) → AdaLN 缩放/平移', changed: true, why: 'mask 率 t 作为条件信号注入，类似图像扩散的时间步条件化。' },
    { name: '损失', ar: '所有位置的下一个 token CE', diff: '仅 masked 位置的加权 CE (× 1/t)', changed: true, why: '损失即 MD4 简化 —— 仅在 masked 位置上的加权 MLM。' },
  ];

  function render() {
    const container = document.getElementById('viz7-1');
    if (!container) return;
    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    const data = lang === 'zh' ? LAYERS_ZH : LAYERS_EN;
    container.innerHTML = '';

    // Header
    const header = document.createElement('div');
    header.className = 'arch-layer';
    header.style.opacity = '1';
    header.style.borderLeftColor = 'transparent';
    header.style.background = 'transparent';
    header.style.fontWeight = '600';
    header.style.color = '#a9a8a3';
    header.style.fontSize = '0.74rem';
    header.style.textTransform = 'uppercase';
    header.style.letterSpacing = '0.08em';
    header.innerHTML = `
      <div>${lang === 'zh' ? '组件' : 'Component'}</div>
      <div>${lang === 'zh' ? 'AR (Llama-3)' : 'AR (Llama-3)'}</div>
      <div>${lang === 'zh' ? '扩散 (LLaDA-8B)' : 'Diffusion (LLaDA-8B)'}</div>
    `;
    container.appendChild(header);

    data.forEach((layer) => {
      const row = document.createElement('div');
      row.className = 'arch-layer' + (layer.changed ? ' changed' : '');
      row.innerHTML = `
        <div class="arch-layer-name">${layer.name}</div>
        <div class="arch-layer-ar">${layer.ar}</div>
        <div class="arch-layer-diff">${layer.diff}</div>
      `;
      row.title = layer.why;
      container.appendChild(row);
    });
  }

  function init() {
    window.addEventListener('langchange', render);
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
