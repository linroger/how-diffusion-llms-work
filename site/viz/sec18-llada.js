/* §18 — LLaDA timeline + LLaDA2.0 MoE routing visualization */
(function () {
  'use strict';

  const TIMELINE_EN = [
    { date: 'Feb 2025', title: 'LLaDA 8B', desc: 'First serious from-scratch 8B masked-diffusion LM. Bidirectional Llama-3-class transformer. 2.3T tokens, 0.13M H800 GPU-hours. Competitive with LLaMA3-8B on standard benchmarks. Breaks the reverse curse.', tags: ['8B dense', 'MIT', 'arxiv 2502.09992'] },
    { date: 'May 2025', title: 'LLaDA 1.5 (VRPO)', desc: 'Variance-reduced preference optimization. Tackles the high variance of ELBO-based likelihood estimates. <0.5% of pretraining cost yields +4.7 GSM8K, +3.0 HumanEval, +4.3 Arena-Hard.', tags: ['Preference optimization', 'arxiv 2505.19223'] },
    { date: 'Sep 2025', title: 'LLaDA-MoE-7B-A1B', desc: 'First MoE diffusion LM. 7B total / 1.4B active. 64 experts, top-8 routing. 16 layers, hidden 2048. 20T tokens. Competitive with Qwen2.5-3B-Instruct.', tags: ['MoE 64×top-8', '7B/1.4B', 'arxiv 2509.24389'] },
    { date: 'Dec 2025', title: 'LLaDA2.0 (mini / flash)', desc: 'The 100B-class moment. Flash: ~100B total, ~6.1B active, 256 routed experts + 1 shared, top-8. AR-to-diffusion conversion from Ling-flash-2.0 via WSD curriculum. 535 TPS (~2.1× over AR).', tags: ['100B MoE', 'WSD curriculum', 'CAP + EBPO', 'arxiv 2512.15745'] },
    { date: 'Apr 2026', title: 'LLaDA2.1', desc: 'T2T (Token-to-Token) editing + Multi-Block Editing. Speedy mode (892 TPS HumanEval+) and Quality mode share the same model. Speedy is 2.1× faster than LLaDA2.0 and quality-equivalent.', tags: ['T2T editing', 'MBE', 'S/Q modes', 'inclusionAI/LLaDA2.X'] },
  ];

  const TIMELINE_ZH = [
    { date: '2025 年 2 月', title: 'LLaDA 8B', desc: '首个严肃的从零训练 8B masked 扩散语言模型。双向 Llama-3 级 Transformer。2.3T tokens，0.13M H800 GPU 小时。与 LLaMA3-8B 在标准基准上可比。打破反转诅咒。', tags: ['8B dense', 'MIT 许可', 'arxiv 2502.09992'] },
    { date: '2025 年 5 月', title: 'LLaDA 1.5 (VRPO)', desc: '方差降低的偏好优化。解决基于 ELBO 的似然估计的高方差问题。预训练成本不到 0.5%，带来 +4.7 GSM8K、+3.0 HumanEval、+4.3 Arena-Hard。', tags: ['偏好优化', 'arxiv 2505.19223'] },
    { date: '2025 年 9 月', title: 'LLaDA-MoE-7B-A1B', desc: '首个 MoE 扩散语言模型。总 7B / 激活 1.4B。64 专家、top-8 路由。16 层、隐层 2048。20T tokens。与 Qwen2.5-3B-Instruct 可比。', tags: ['MoE 64×top-8', '7B/1.4B', 'arxiv 2509.24389'] },
    { date: '2025 年 12 月', title: 'LLaDA2.0 (mini / flash)', desc: '百亿规模的关键节点。Flash：总参 ~100B、激活 ~6.1B、256 路由专家 + 1 共享、top-8。通过 WSD 课程从 Ling-flash-2.0 完成 AR-to-diffusion 转换。535 TPS（约 2.1× AR）。', tags: ['100B MoE', 'WSD 课程', 'CAP + EBPO', 'arxiv 2512.15745'] },
    { date: '2026 年 4 月', title: 'LLaDA2.1', desc: 'T2T（Token-to-Token）编辑 + 多块编辑（MBE）。Speedy 模式（HumanEval+ 892 TPS）与 Quality 模式共用同一模型。Speedy 比 LLaDA2.0 快 2.1×，质量等价。', tags: ['T2T 编辑', 'MBE', 'S/Q 模式', 'inclusionAI/LLaDA2.X'] },
  ];

  function renderTimeline() {
    const container = document.getElementById('viz18-1');
    if (!container) return;
    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    const data = lang === 'zh' ? TIMELINE_ZH : TIMELINE_EN;
    container.innerHTML = '';

    data.forEach((entry, idx) => {
      const row = document.createElement('div');
      row.className = 'timeline-entry';
      row.style.cursor = 'pointer';
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

  // MoE routing: animated token-by-token through 256 experts + 1 shared
  function renderMoE() {
    const container = document.getElementById('viz18-2');
    if (!container) return;
    container.innerHTML = '';

    const lang = document.documentElement.getAttribute('data-lang') || 'en';

    // Token strip at top — 8 tokens
    const sentence = DLM.pickSentence(0).slice(0, 8);
    const tokenStrip = document.createElement('div');
    tokenStrip.style.display = 'flex';
    tokenStrip.style.gap = '8px';
    tokenStrip.style.marginBottom = '20px';
    tokenStrip.style.justifyContent = 'center';

    sentence.forEach((tok, i) => {
      const el = document.createElement('div');
      el.className = 'token committed';
      el.style.padding = '6px 10px';
      el.style.cursor = 'pointer';
      el.dataset.idx = i;
      el.textContent = tok;
      el.addEventListener('click', () => animateToken(i));
      el.addEventListener('mouseenter', () => animateToken(i));
      tokenStrip.appendChild(el);
    });
    container.appendChild(tokenStrip);

    // SVG with shared expert in center + ring of 256 experts
    const svg = d3.create('svg').attr('viewBox', '0 0 600 360').attr('style', 'width:100%;max-height:380px;display:block');
    container.appendChild(svg.node());
    const g = svg.append('g').attr('transform', 'translate(300, 180)');

    // Shared expert (center)
    g.append('circle').attr('r', 22).attr('fill', '#f5b54a').attr('stroke', '#1c2230').attr('stroke-width', 2);
    g.append('text').attr('text-anchor', 'middle').attr('dy', 4).attr('fill', '#1a1308').style('font-family', 'Inter, sans-serif').style('font-size', '10px').style('font-weight', '600').text('Shared');

    // Ring of 256 experts (we'll show 64 as dots arranged in a ring, representing the 256 logically)
    const ringRadius = 130;
    const visibleExperts = 64; // visual cap
    const experts = [];
    for (let i = 0; i < visibleExperts; i++) {
      const ang = (i / visibleExperts) * Math.PI * 2 - Math.PI / 2;
      const ex = Math.cos(ang) * ringRadius;
      const ey = Math.sin(ang) * ringRadius;
      const c = g.append('circle').attr('cx', ex).attr('cy', ey).attr('r', 5).attr('fill', '#2e3648').attr('stroke', '#1c2230').attr('stroke-width', 1).attr('class', `expert expert-${i}`);
      experts.push({ x: ex, y: ey, node: c });
    }
    // "Routed: 256, Active: 8" annotation
    svg.append('text').attr('x', 300).attr('y', 24).attr('text-anchor', 'middle').attr('fill', '#a9a8a3').style('font-family', 'Inter, sans-serif').style('font-size', '12px').style('font-weight', '500').text(lang === 'zh' ? '路由专家：256 (此处显示 64) · 每 token 激活：top-8' : 'Routed experts: 256 (showing 64) · Activated per token: top-8');
    svg.append('text').attr('x', 300).attr('y', 340).attr('text-anchor', 'middle').attr('fill', '#a9a8a3').style('font-family', 'Inter, sans-serif').style('font-size', '12px').text(lang === 'zh' ? '点击任意 token 查看其专家选择' : 'Hover any token to see its expert selection');

    // Active expert lines container
    const linesGroup = g.append('g').attr('class', 'routing-lines');

    function animateToken(tokenIdx) {
      // Deterministically pick 8 experts for this token
      const rng = DLM.makeRNG(1000 + tokenIdx * 13);
      const chosen = new Set();
      while (chosen.size < 8) chosen.add(Math.floor(rng() * visibleExperts));

      // Reset
      experts.forEach((e, i) => e.node.attr('fill', chosen.has(i) ? '#f5b54a' : '#2e3648').attr('r', chosen.has(i) ? 7 : 5));
      linesGroup.selectAll('*').remove();

      // Draw lines from center to chosen
      chosen.forEach((i) => {
        const e = experts[i];
        linesGroup.append('line').attr('x1', 0).attr('y1', 0).attr('x2', 0).attr('y2', 0)
          .attr('stroke', '#f5b54a').attr('stroke-width', 1.2).attr('opacity', 0)
          .transition().duration(400)
          .attr('x2', e.x).attr('y2', e.y).attr('opacity', 0.6);
      });
    }

    // Auto-animate first
    setTimeout(() => animateToken(0), 200);
  }

  function init() {
    renderTimeline();
    renderMoE();
    window.addEventListener('langchange', () => { renderTimeline(); renderMoE(); });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
