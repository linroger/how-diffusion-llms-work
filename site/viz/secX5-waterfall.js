/* Extra 5 — Ant Group's speedup waterfall
   Shows the cumulative speedup from each Ant Group inference optimization layer:
   base AR baseline → block parallel decoding → CAP confidence training → 
   KV cache reuse → dInfer framework → FP8 quantization → Alpha-MoE megakernel.
   Numbers from LLaDA2.0 / LLaDA2.1 ablations + LMSYS day-0 benchmarks. */
(function () {
  'use strict';

  const LAYERS_EN = [
    { name: 'AR baseline (Qwen3-30B SGLang)', value: 237, color: 'base', desc: '' },
    { name: '+ Block parallel decoding', delta: 146, value: 383, color: 'delta', desc: 'Bidirectional within block, causal between blocks. KV cache survives across blocks.' },
    { name: '+ CAP confidence training', delta: 117, value: 500, color: 'delta', desc: 'Auxiliary entropy loss on correctly-predicted tokens makes confidence trustworthy.' },
    { name: '+ dInfer framework', delta: 392, value: 892, color: 'delta', desc: 'Ant\u2019s optimized inference engine. Per-block FP8, KV reuse, fused kernels.' },
    { name: '+ FP8 quantization', delta: 200, value: 1092, color: 'delta', desc: 'Per-block FP8 with negligible quality loss.' },
    { name: '+ Alpha-MoE megakernel (mini)', delta: 495, value: 1587, color: 'delta', desc: 'Fused expert routing + activation in a single kernel launch.' },
  ];

  const LAYERS_ZH = [
    { name: 'AR 基线 (Qwen3-30B / SGLang)', value: 237, color: 'base', desc: '' },
    { name: '+ 块并行解码', delta: 146, value: 383, color: 'delta', desc: '块内双向、块间因果。KV 缓存在已解码块上有效。' },
    { name: '+ CAP 置信度训练', delta: 117, value: 500, color: 'delta', desc: '在"答对的 token"上加上熵损失，让置信度变得可信。' },
    { name: '+ dInfer 框架', delta: 392, value: 892, color: 'delta', desc: '蚂蚁的高优推理引擎。逐块 FP8、KV 复用、融合 kernel。' },
    { name: '+ FP8 量化', delta: 200, value: 1092, color: 'delta', desc: '逐块 FP8，质量近乎无损。' },
    { name: '+ Alpha-MoE megakernel (mini)', delta: 495, value: 1587, color: 'delta', desc: '专家路由 + 激活 融合到一个 kernel 启动。' },
  ];

  function render() {
    const container = document.getElementById('vizX5');
    if (!container) return;
    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    const layers = lang === 'zh' ? LAYERS_ZH : LAYERS_EN;
    const max = Math.max(...layers.map((l) => l.value));

    container.innerHTML = '';
    const finalTotal = layers[layers.length - 1].value;
    const totalSpeedup = (finalTotal / layers[0].value).toFixed(1);

    const intro = document.createElement('div');
    intro.style.fontFamily = 'var(--font-sans)';
    intro.style.fontSize = '0.92rem';
    intro.style.color = 'var(--text-soft)';
    intro.style.marginBottom = '14px';
    intro.style.padding = '10px 14px';
    intro.style.background = 'var(--bg-frame-2)';
    intro.style.borderRadius = '6px';
    intro.innerHTML = lang === 'zh' ?
      `<strong style="color:var(--text)">${totalSpeedup}×</strong> 相对 AR 基线的累计加速 (HumanEval+，TPS)。每一项是 Ant Group 在 LLaDA 系列上独立报告的贡献。` :
      `<strong style="color:var(--text)">${totalSpeedup}×</strong> cumulative speedup over the AR baseline (HumanEval+, TPS). Each row is a separately-reported contribution from Ant Group's LLaDA stack.`;
    container.appendChild(intro);

    layers.forEach((l, i) => {
      const row = document.createElement('div');
      row.className = 'waterfall-row';
      const prevValue = i === 0 ? 0 : layers[i - 1].value;
      const barW = (l.value / max) * 100;
      const deltaW = i === 0 ? 0 : ((l.value - prevValue) / max) * 100;
      const baseW = (prevValue / max) * 100;
      row.innerHTML = `
        <div class="waterfall-name">${l.name}${l.desc ? `<div style="font-family:var(--font-sans);font-size:0.78rem;color:var(--text-muted);font-weight:400;margin-top:3px;line-height:1.4">${l.desc}</div>` : ''}</div>
        <div class="waterfall-bar-track">
          ${baseW > 0 ? `<div class="waterfall-bar" style="left:0;width:${baseW}%;opacity:0.4"></div>` : ''}
          <div class="waterfall-bar delta" style="left:${baseW}%;width:${deltaW || barW}%"></div>
        </div>
        <div class="waterfall-value">${l.value} ${lang === 'zh' ? 'TPS' : 'TPS'}</div>
      `;
      row.style.animation = `wfFade 0.6s ${i * 80}ms backwards cubic-bezier(0.22,1,0.36,1)`;
      container.appendChild(row);
    });

    // Add keyframes once
    if (!document.getElementById('wf-keyframes')) {
      const s = document.createElement('style');
      s.id = 'wf-keyframes';
      s.textContent = `@keyframes wfFade { from { opacity: 0; transform: translateX(-12px) } to { opacity: 1; transform: translateX(0) } }`;
      document.head.appendChild(s);
    }
  }

  function init() {
    window.addEventListener('langchange', render);
    window.addEventListener('palettechange', render);
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
