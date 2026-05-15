/* §17 — DeepMind timeline + Gemini Diffusion radar */
(function () {
  'use strict';

  const TIMELINE_EN = [
    { date: 'Jan 2024', title: 'AR2Diff', desc: 'Three-stage recipe to convert an AR pretrained checkpoint into a diffusion model. Decoder-only + prefix-LM transfers best. Validated diffusion as a viable alternative on code and QA.', tags: ['Transfer recipe', '280M–1.7B', 'SUNDAE loss'] },
    { date: 'Jun 2024', title: 'MD4 (NeurIPS 2024)', desc: 'The simplification. Proves the continuous-time ELBO collapses to a weighted cross-entropy integral. Mean-parameterization removes the need for variance prediction.', tags: ['Math', 'JAX/Flax', 'github.com/google-deepmind/md4'] },
    { date: 'Jun 2024', title: 'GenMD4', desc: 'State-dependent (per-vocabulary-token) learnable masking schedules, trained via Rao-Blackwellized REINFORCE.', tags: ['Schedule generalization', 'RLOO REINFORCE'] },
    { date: 'May 2025', title: 'Gemini Diffusion', desc: 'Productization. Block-diffusion serving system demoed at Google I/O 2025. Vanilla transformer backbone, ~128-token blocks, bidirectional intra-block + causal inter-block.', tags: ['Block diffusion', '1479 tok/s', '0.84s TTFT', 'Experimental waitlist'] },
  ];

  const TIMELINE_ZH = [
    { date: '2024 年 1 月', title: 'AR2Diff', desc: '将 AR 预训练检查点转换为扩散模型的三阶段配方。Decoder-only + prefix-LM 迁移效果最佳。验证扩散在代码与问答上是一种可行的替代方案。', tags: ['迁移配方', '280M–1.7B', 'SUNDAE 损失'] },
    { date: '2024 年 6 月', title: 'MD4 (NeurIPS 2024)', desc: '简化。证明连续时间 ELBO 坍缩为加权交叉熵积分。Mean-parameterization 不再需要方差预测。', tags: ['数学', 'JAX/Flax', 'github.com/google-deepmind/md4'] },
    { date: '2024 年 6 月', title: 'GenMD4', desc: '每个 token 可学习的状态相关掩码调度，通过 Rao-Blackwellized REINFORCE 训练。', tags: ['调度泛化', 'RLOO REINFORCE'] },
    { date: '2025 年 5 月', title: 'Gemini Diffusion', desc: '产品化。Google I/O 2025 演示的块扩散服务系统。普通 Transformer 骨干、约 128 token 的块、块内双向 + 块间因果。', tags: ['块扩散', '1479 tok/s', '0.84s TTFT', '实验性候补'] },
  ];

  function renderTimeline() {
    const container = document.getElementById('viz17-1');
    if (!container) return;
    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    const data = lang === 'zh' ? TIMELINE_ZH : TIMELINE_EN;
    container.innerHTML = '';

    data.forEach((entry) => {
      const row = document.createElement('div');
      row.className = 'timeline-entry';
      row.innerHTML = `
        <div class="timeline-date">${entry.date}</div>
        <div class="timeline-body">
          <div class="timeline-title">${entry.title}</div>
          <div class="timeline-desc">${entry.desc}</div>
          <div class="timeline-tags">
            ${entry.tags.map((t) => `<span class="timeline-tag">${t}</span>`).join('')}
          </div>
        </div>
      `;
      container.appendChild(row);
    });
  }

  // Radar chart: Gemini Diffusion vs Flash-Lite, 8 benchmarks
  const BENCHMARKS_EN = [
    { name: 'HumanEval', diff: 89.6, ar: 90.2 },
    { name: 'MBPP', diff: 76.0, ar: 75.8 },
    { name: 'LiveCodeBench', diff: 30.9, ar: 28.5 },
    { name: 'BigCodeBench', diff: 45.4, ar: 45.8 },
    { name: 'AIME 2025', diff: 23.3, ar: 20.0 },
    { name: 'GPQA Diamond', diff: 40.4, ar: 56.5 },
    { name: 'BBH', diff: 15.0, ar: 21.0 },
    { name: 'Global MMLU', diff: 69.1, ar: 79.0 },
  ];

  function renderRadar() {
    const svg = d3.select('#viz17-2-svg');
    if (svg.empty()) return;
    svg.selectAll('*').remove();
    svg.attr('viewBox', '0 0 560 480');
    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    const W = 560, H = 480;
    const cx = W / 2, cy = H / 2;
    const radius = 160;
    const g = svg.append('g').attr('transform', `translate(${cx},${cy})`);

    const N = BENCHMARKS_EN.length;
    const angleSlice = (Math.PI * 2) / N;

    // Grid circles
    const levels = 5;
    for (let l = 1; l <= levels; l++) {
      const r = (radius / levels) * l;
      g.append('circle').attr('r', r).attr('fill', 'none').style('stroke', 'var(--bg-frame-2)').attr('stroke-width', 1);
      // labels
      if (l === levels) {
        g.append('text').attr('x', 4).attr('y', -r + 12).style('fill', 'var(--text-muted)').style('font-size', '9px').style('font-family', 'JetBrains Mono, monospace').text('100');
      } else if (l === levels / 2 || l === 2) {
        g.append('text').attr('x', 4).attr('y', -r + 12).style('fill', 'var(--text-muted)').style('font-size', '9px').style('font-family', 'JetBrains Mono, monospace').text((l * 20).toString());
      }
    }

    // Axes
    BENCHMARKS_EN.forEach((b, i) => {
      const angle = -Math.PI / 2 + angleSlice * i;
      const ex = Math.cos(angle) * radius;
      const ey = Math.sin(angle) * radius;
      g.append('line').attr('x1', 0).attr('y1', 0).attr('x2', ex).attr('y2', ey).style('stroke', 'var(--border-strong)').attr('stroke-width', 1);

      // Label
      const lx = Math.cos(angle) * (radius + 24);
      const ly = Math.sin(angle) * (radius + 24);
      const txt = g.append('text').attr('x', lx).attr('y', ly).attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
        .style('fill', 'var(--text-soft)').style('font-family', 'Inter, sans-serif').style('font-size', '11px').style('font-weight', '500').text(b.name);
      // adjust text-anchor for left/right halves
      if (Math.cos(angle) > 0.3) txt.attr('text-anchor', 'start');
      else if (Math.cos(angle) < -0.3) txt.attr('text-anchor', 'end');
    });

    // Two polygons
    const renderPolygon = (key, color, fillOpacity) => {
      const path = d3.lineRadial()
        .angle((d, i) => angleSlice * i)
        .radius((d) => (d[key] / 100) * radius)
        .curve(d3.curveLinearClosed);
      g.append('path').datum(BENCHMARKS_EN).attr('d', path).attr('fill', color).attr('fill-opacity', fillOpacity).attr('stroke', color).attr('stroke-width', 2);
      // Points
      BENCHMARKS_EN.forEach((b, i) => {
        const angle = -Math.PI / 2 + angleSlice * i;
        const r = (b[key] / 100) * radius;
        g.append('circle').attr('cx', Math.cos(angle) * r).attr('cy', Math.sin(angle) * r).attr('r', 3.5).attr('fill', color);
      });
    };

    const css = getComputedStyle(document.documentElement);
    const accent = css.getPropertyValue('--accent').trim() || 'var(--accent)';
    const accent2 = css.getPropertyValue('--accent-2').trim() || 'var(--accent-2)';
    renderPolygon('ar', accent2, 0.18);
    renderPolygon('diff', accent, 0.22);
  }

  function init() {
    renderTimeline();
    renderRadar();
    window.addEventListener('langchange', () => { renderTimeline(); renderRadar(); });
    window.addEventListener('palettechange', renderRadar);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
