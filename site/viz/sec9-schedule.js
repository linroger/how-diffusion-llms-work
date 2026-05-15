/* §9 — Noise schedule comparison + induced weighting */
(function () {
  'use strict';

  let schedule = 'linear';

  const schedules = {
    linear: { alpha: (t) => 1 - t, weight: (t) => 1 / Math.max(t, 0.01) },
    cosine: { alpha: (t) => Math.cos(Math.PI * t / 2), weight: (t) => Math.tan(Math.PI * t / 2) * (Math.PI / 2) + 0.01 },
    poly2: { alpha: (t) => 1 - t * t, weight: (t) => (2 * t) / Math.max(1 - t * t, 0.01) },
    poly3: { alpha: (t) => 1 - t * t * t, weight: (t) => (3 * t * t) / Math.max(1 - t * t * t, 0.01) },
  };

  function drawAlpha() {
    const svg = d3.select('#viz9-1-alpha');
    if (svg.empty()) return;
    svg.selectAll('*').remove();
    const W = 240, H = 180;
    const m = { top: 26, right: 14, bottom: 28, left: 36 };
    const iw = W - m.left - m.right;
    const ih = H - m.top - m.bottom;
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const x = d3.scaleLinear().domain([0, 1]).range([0, iw]);
    const y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);

    // Axes
    g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(3))
      .selectAll('text').style('fill', 'var(--text-soft)').style('font-size', '9px').style('font-family', 'JetBrains Mono, monospace');
    g.append('g').call(d3.axisLeft(y).ticks(3))
      .selectAll('text').style('fill', 'var(--text-soft)').style('font-size', '9px').style('font-family', 'JetBrains Mono, monospace');
    g.selectAll('.domain, .tick line').style('stroke', 'var(--border-strong)');

    Object.keys(schedules).forEach((s) => {
      const data = d3.range(0, 1.01, 0.02).map((t) => ({ t, v: schedules[s].alpha(t) }));
      const line = d3.line().x((d) => x(d.t)).y((d) => y(d.v)).curve(d3.curveBasis);
      g.append('path').datum(data)
        .attr('fill', 'none')
        .attr('stroke', s === schedule ? 'var(--accent)' : 'var(--border-strong)')
        .attr('stroke-width', s === schedule ? 2.5 : 1.2)
        .attr('opacity', s === schedule ? 1 : 0.5)
        .attr('d', line);
    });

    g.append('text').attr('x', iw / 2).attr('y', -8).attr('text-anchor', 'middle')
      .style('fill', 'var(--text-soft)').style('font-size', '11px').style('font-family', 'JetBrains Mono, monospace')
      .text('α(t) — fraction unmasked');
    g.append('text').attr('x', iw / 2).attr('y', ih + 24).attr('text-anchor', 'middle')
      .style('fill', 'var(--text-muted)').style('font-size', '10px').style('font-family', 'JetBrains Mono, monospace').text('time t');
  }

  function drawWeight() {
    const svg = d3.select('#viz9-1-weight');
    if (svg.empty()) return;
    svg.selectAll('*').remove();
    const W = 240, H = 180;
    const m = { top: 26, right: 14, bottom: 28, left: 36 };
    const iw = W - m.left - m.right;
    const ih = H - m.top - m.bottom;
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const x = d3.scaleLinear().domain([0.02, 1]).range([0, iw]);
    const y = d3.scaleLog().domain([1, 100]).range([ih, 0]).clamp(true);

    g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(3))
      .selectAll('text').style('fill', 'var(--text-soft)').style('font-size', '9px').style('font-family', 'JetBrains Mono, monospace');
    g.append('g').call(d3.axisLeft(y).ticks(3))
      .selectAll('text').style('fill', 'var(--text-soft)').style('font-size', '9px').style('font-family', 'JetBrains Mono, monospace');
    g.selectAll('.domain, .tick line').style('stroke', 'var(--border-strong)');

    Object.keys(schedules).forEach((s) => {
      const data = d3.range(0.02, 1.0, 0.01).map((t) => ({ t, v: Math.min(100, schedules[s].weight(t)) }));
      const line = d3.line().x((d) => x(d.t)).y((d) => y(Math.max(1, d.v))).curve(d3.curveBasis);
      g.append('path').datum(data)
        .attr('fill', 'none')
        .attr('stroke', s === schedule ? 'var(--accent-2)' : 'var(--border-strong)')
        .attr('stroke-width', s === schedule ? 2.5 : 1.2)
        .attr('opacity', s === schedule ? 1 : 0.5)
        .attr('d', line);
    });

    g.append('text').attr('x', iw / 2).attr('y', -8).attr('text-anchor', 'middle')
      .style('fill', 'var(--text-soft)').style('font-size', '11px').style('font-family', 'JetBrains Mono, monospace')
      .text('w(t) — induced loss weight (log)');
    g.append('text').attr('x', iw / 2).attr('y', ih + 24).attr('text-anchor', 'middle')
      .style('fill', 'var(--text-muted)').style('font-size', '10px').style('font-family', 'JetBrains Mono, monospace').text('time t');
  }

  function init() {
    document.querySelectorAll('#viz9-1 .schedule-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#viz9-1 .schedule-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        schedule = btn.dataset.schedule;
        drawAlpha();
        drawWeight();
      });
    });
    drawAlpha();
    drawWeight();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
