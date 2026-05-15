/* §10 — WSD curriculum chart + linked attention-mask preview */
(function () {
  'use strict';

  // Warmup-Stable-Decay curriculum, plotted on a log y-axis
  // Pairs of (training_progress, L_B)
  const CURRICULUM = [
    { p: 0, lb: 1, phase: 'warmup' },
    { p: 0.05, lb: 1, phase: 'warmup' },
    { p: 0.08, lb: 4, phase: 'warmup' },
    { p: 0.12, lb: 32, phase: 'warmup' },
    { p: 0.16, lb: 64, phase: 'warmup' },
    { p: 0.22, lb: 4096, phase: 'warmup' },
    { p: 0.30, lb: 4096, phase: 'stable' },
    { p: 0.75, lb: 4096, phase: 'stable' },
    { p: 0.80, lb: 2048, phase: 'decay' },
    { p: 0.85, lb: 512, phase: 'decay' },
    { p: 0.90, lb: 128, phase: 'decay' },
    { p: 0.95, lb: 32, phase: 'decay' },
    { p: 1.00, lb: 32, phase: 'decay' },
  ];

  let progress = 0;
  let playing = false;
  let timer = null;

  function lerp(a, b, t) { return a + (b - a) * t; }

  function lookupLb(p) {
    for (let i = 0; i < CURRICULUM.length - 1; i++) {
      const cur = CURRICULUM[i], nxt = CURRICULUM[i + 1];
      if (p >= cur.p && p <= nxt.p) {
        const u = (p - cur.p) / Math.max(0.0001, nxt.p - cur.p);
        return { lb: Math.round(Math.exp(lerp(Math.log(cur.lb), Math.log(nxt.lb), u))), phase: cur.phase };
      }
    }
    return { lb: 32, phase: 'decay' };
  }

  function drawChart() {
    const svg = d3.select('#viz10-1-chart');
    if (svg.empty()) return;
    svg.selectAll('*').remove();
    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    const W = 720, H = 220;
    const m = { top: 20, right: 30, bottom: 36, left: 60 };
    const iw = W - m.left - m.right;
    const ih = H - m.top - m.bottom;
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const x = d3.scaleLinear().domain([0, 1]).range([0, iw]);
    const y = d3.scaleLog().domain([1, 8192]).range([ih, 0]);

    // Axes
    g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.format('.0%')))
      .selectAll('text').style('fill', 'var(--text-soft)').style('font-size', '10px').style('font-family', 'JetBrains Mono, monospace');
    g.append('g').call(d3.axisLeft(y).tickValues([1, 4, 32, 64, 512, 4096]).tickFormat(d3.format('d')))
      .selectAll('text').style('fill', 'var(--text-soft)').style('font-size', '10px').style('font-family', 'JetBrains Mono, monospace');
    g.selectAll('.domain, .tick line').style('stroke', 'var(--border-strong)');

    // Phase shading
    const phaseLabels = lang === 'zh' ? { warmup: '预热 Warmup', stable: '稳定 Stable', decay: '退火 Decay' } : { warmup: 'Warmup', stable: 'Stable', decay: 'Decay' };
    const phaseBounds = [
      { x0: 0, x1: 0.22, color: 'rgba(245, 181, 74, 0.06)', label: phaseLabels.warmup },
      { x0: 0.22, x1: 0.78, color: 'rgba(95, 176, 199, 0.06)', label: phaseLabels.stable },
      { x0: 0.78, x1: 1.0, color: 'rgba(232, 121, 168, 0.06)', label: phaseLabels.decay },
    ];
    phaseBounds.forEach((pb) => {
      g.append('rect').attr('x', x(pb.x0)).attr('y', 0).attr('width', x(pb.x1) - x(pb.x0)).attr('height', ih).attr('fill', pb.color);
      g.append('text').attr('x', x((pb.x0 + pb.x1) / 2)).attr('y', 14).attr('text-anchor', 'middle')
        .style('fill', 'var(--text-soft)').style('font-size', '10px').style('font-family', 'Inter, sans-serif').style('font-weight', '600')
        .style('letter-spacing', '0.1em').style('text-transform', 'uppercase').text(pb.label);
    });

    // Curve
    const data = d3.range(0, 1.001, 0.005).map((p) => ({ p, lb: lookupLb(p).lb }));
    const line = d3.line().x((d) => x(d.p)).y((d) => y(Math.max(1, d.lb))).curve(d3.curveMonotoneX);
    g.append('path').datum(data).attr('fill', 'none').style('stroke', 'var(--accent)').attr('stroke-width', 2).attr('d', line);

    // Current point
    const cur = lookupLb(progress);
    g.append('circle').attr('cx', x(progress)).attr('cy', y(Math.max(1, cur.lb))).attr('r', 6).style('fill', 'var(--accent)').style('stroke', 'var(--bg-frame-2)').attr('stroke-width', 2);
    g.append('line').attr('x1', x(progress)).attr('x2', x(progress)).attr('y1', 0).attr('y2', ih).style('stroke', 'var(--accent)').attr('stroke-width', 1).attr('stroke-dasharray', '3,3').attr('opacity', 0.5);

    // Axis labels
    g.append('text').attr('x', iw / 2).attr('y', ih + 28).attr('text-anchor', 'middle')
      .style('fill', 'var(--text-muted)').style('font-family', 'JetBrains Mono, monospace').style('font-size', '10px').text(lang === 'zh' ? '训练进度 →' : 'training progress →');
    g.append('text').attr('transform', `translate(-44, ${ih / 2}) rotate(-90)`).attr('text-anchor', 'middle')
      .style('fill', 'var(--text-muted)').style('font-family', 'JetBrains Mono, monospace').style('font-size', '10px').text(lang === 'zh' ? '块大小 L_B →' : 'block size L_B →');
  }

  function drawMaskPreview() {
    const cur = lookupLb(progress);
    const svg = d3.select('#viz10-1-mask');
    if (svg.empty()) return;
    svg.selectAll('*').remove();
    const W = 200, H = 200;
    const pad = 8;
    const N = 16;
    const cell = (W - pad * 2) / N;
    const g = svg.append('g').attr('transform', `translate(${pad},${pad})`);
    const bs = Math.min(cur.lb, N);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const bi = Math.floor(i / bs);
        const bj = Math.floor(j / bs);
        const allowed = bi === bj || bj < bi;
        g.append('rect').attr('x', j * cell).attr('y', i * cell).attr('width', cell - 0.5).attr('height', cell - 0.5).attr('fill', allowed ? 'var(--accent)' : 'var(--bg-frame-2)').attr('opacity', allowed ? 0.85 : 1);
      }
    }
    const lbEl = document.getElementById('viz10-1-lb');
    const phaseEl = document.getElementById('viz10-1-phase');
    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    const phaseTxt = lang === 'zh' ? { warmup: '预热', stable: '稳定', decay: '退火' } : { warmup: 'warmup', stable: 'stable', decay: 'decay' };
    if (lbEl) lbEl.textContent = cur.lb;
    if (phaseEl) phaseEl.textContent = phaseTxt[cur.phase] || cur.phase;
  }

  function tick() {
    progress = Math.min(1, progress + 0.005);
    drawChart();
    drawMaskPreview();
    if (progress < 1) timer = setTimeout(tick, 40);
    else playing = false;
  }

  function play() {
    if (playing) return;
    playing = true;
    progress = 0;
    tick();
  }
  function reset() {
    if (timer) clearTimeout(timer);
    playing = false;
    progress = 0;
    drawChart();
    drawMaskPreview();
  }

  function init() {
    document.getElementById('viz10-1-play')?.addEventListener('click', play);
    document.getElementById('viz10-1-reset')?.addEventListener('click', reset);
    drawChart();
    drawMaskPreview();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
