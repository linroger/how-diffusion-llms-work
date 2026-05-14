/* §16 — Speed/quality tradeoff scatter, with controlled-comparison points */
(function () {
  'use strict';

  // Each point is { name, tps, score, group: 'ar' | 'diff', tooltip }
  const POINTS = [
    { name: 'Qwen3-30B-A3B (AR)', tps: 237, score: 73.6, group: 'ar', size: 8 },
    { name: 'Ling-flash-2.0 (AR)', tps: 256, score: 72.8, group: 'ar', size: 8 },
    { name: 'gpt-oss-120B (AR)', tps: 263, score: 74.2, group: 'ar', size: 8 },
    { name: 'LLaDA2.0-flash', tps: 383, score: 73.2, group: 'diff', size: 9 },
    { name: 'LLaDA2.0-flash + CAP', tps: 535, score: 73.18, group: 'diff', size: 10 },
    { name: 'LLaDA2.0-flash-CAP (peak)', tps: 935, score: 70.5, group: 'diff', size: 8 },
    { name: 'LLaDA2.1-flash (S-Mode)', tps: 892, score: 73.0, group: 'diff', size: 10 },
    { name: 'LLaDA2.0-mini (FP8)', tps: 1587, score: 63.7, group: 'diff', size: 8 },
    { name: 'Mercury-Coder (Small)', tps: 1109, score: 70.2, group: 'diff', size: 8 },
    { name: 'Gemini Diffusion', tps: 1479, score: 70.5, group: 'diff', size: 9 },
    { name: 'Seed-Diffusion (preview)', tps: 2146, score: 67.4, group: 'diff', size: 8 },
  ];

  function draw() {
    const svg = d3.select('#viz16-1-chart');
    if (svg.empty()) return;
    svg.selectAll('*').remove();

    const W = 720, H = 400;
    const m = { top: 30, right: 40, bottom: 50, left: 60 };
    const iw = W - m.left - m.right;
    const ih = H - m.top - m.bottom;
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const x = d3.scaleLog().domain([200, 2500]).range([0, iw]);
    const y = d3.scaleLinear().domain([60, 76]).range([ih, 0]);

    // Grid
    const xGrid = g.append('g').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).ticks(6, '~s').tickSize(-ih).tickFormat(''));
    xGrid.selectAll('line').style('stroke', '#1c2230').style('stroke-dasharray', '2,3');
    xGrid.select('.domain').remove();
    const yGrid = g.append('g').call(d3.axisLeft(y).ticks(5).tickSize(-iw).tickFormat(''));
    yGrid.selectAll('line').style('stroke', '#1c2230').style('stroke-dasharray', '2,3');
    yGrid.select('.domain').remove();

    // Axes (labels)
    g.append('g').attr('transform', `translate(0,${ih})`)
      .call(d3.axisBottom(x).ticks(6, '~s'))
      .selectAll('text').style('fill', '#a9a8a3').style('font-size', '11px').style('font-family', 'JetBrains Mono, monospace');
    g.append('g')
      .call(d3.axisLeft(y).ticks(5))
      .selectAll('text').style('fill', '#a9a8a3').style('font-size', '11px').style('font-family', 'JetBrains Mono, monospace');
    g.selectAll('.domain, .tick line').style('stroke', '#2e3648');

    // Points
    const tooltip = d3.select('body').selectAll('.tradeoff-tooltip').data([0]);
    const tt = tooltip.enter().append('div').attr('class', 'tradeoff-tooltip')
      .style('position', 'fixed').style('pointer-events', 'none').style('background', '#11151f').style('border', '1px solid #2e3648')
      .style('border-radius', '4px').style('padding', '8px 12px').style('font-family', 'JetBrains Mono, monospace').style('font-size', '12px')
      .style('color', '#e8e6e1').style('display', 'none').style('z-index', '1000').merge(tooltip);

    POINTS.forEach((p) => {
      g.append('circle')
        .attr('cx', x(p.tps)).attr('cy', y(p.score)).attr('r', p.size)
        .attr('fill', p.group === 'ar' ? '#5fb0c7' : '#f5b54a')
        .attr('opacity', 0.85)
        .attr('stroke', p.group === 'ar' ? '#5fb0c7' : '#f5b54a').attr('stroke-width', 2)
        .style('cursor', 'pointer')
        .on('mouseenter', function (event) {
          d3.select(this).attr('r', p.size + 3).attr('opacity', 1);
          tt.style('display', 'block').html(`<b>${p.name}</b><br>TPS: ${p.tps}<br>Score: ${p.score}<br><span style="color:#a9a8a3">${p.group === 'ar' ? 'Autoregressive' : 'Diffusion'}</span>`);
        })
        .on('mousemove', function (event) {
          tt.style('left', (event.clientX + 12) + 'px').style('top', (event.clientY + 12) + 'px');
        })
        .on('mouseleave', function () {
          d3.select(this).attr('r', p.size).attr('opacity', 0.85);
          tt.style('display', 'none');
        });
    });

    // Labels
    g.append('text').attr('x', iw / 2).attr('y', ih + 40).attr('text-anchor', 'middle')
      .attr('fill', '#a9a8a3').style('font-family', 'JetBrains Mono, monospace').style('font-size', '12px').text('throughput (tokens/sec) — log scale');
    g.append('text').attr('transform', `translate(-44, ${ih / 2}) rotate(-90)`).attr('text-anchor', 'middle')
      .attr('fill', '#a9a8a3').style('font-family', 'JetBrains Mono, monospace').style('font-size', '12px').text('benchmark score (47-benchmark avg)');

    // Title
    g.append('text').attr('x', 0).attr('y', -10).attr('fill', '#fbfaf7').style('font-family', 'Source Serif 4, serif').style('font-size', '14px').style('font-weight', '600')
      .text('Controlled comparison · 8×H20, TP=8, SGLang');

    // Pareto frontier hint — draw a soft curve through the best-of-both
    const frontier = POINTS.filter((p) => p.group === 'diff' && p.score > 70).sort((a, b) => a.tps - b.tps);
    const line = d3.line().x((d) => x(d.tps)).y((d) => y(d.score)).curve(d3.curveMonotoneX);
    g.append('path').datum(frontier).attr('fill', 'none').attr('stroke', '#f5b54a').attr('stroke-width', 1.5).attr('stroke-dasharray', '4,4').attr('opacity', 0.4).attr('d', line);

    // Legend
    const legend = g.append('g').attr('transform', `translate(${iw - 200}, 8)`);
    legend.append('circle').attr('cx', 6).attr('cy', 0).attr('r', 6).attr('fill', '#5fb0c7');
    legend.append('text').attr('x', 18).attr('y', 4).attr('fill', '#a9a8a3').style('font-family', 'Inter, sans-serif').style('font-size', '12px').text('Autoregressive');
    legend.append('circle').attr('cx', 6).attr('cy', 18).attr('r', 6).attr('fill', '#f5b54a');
    legend.append('text').attr('x', 18).attr('y', 22).attr('fill', '#a9a8a3').style('font-family', 'Inter, sans-serif').style('font-size', '12px').text('Diffusion');
  }

  function init() {
    draw();
    window.addEventListener('resize', draw);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
