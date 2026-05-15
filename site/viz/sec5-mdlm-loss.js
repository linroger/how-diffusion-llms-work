/* §5 — MDLM loss visualization: live mask + 1/t curve */
(function () {
  'use strict';

  let t = 0.5;

  function renderSentence() {
    const container = document.getElementById('viz5-1-sentence');
    if (!container) return;
    const sentence = DLM.pickSentence(0);
    const rng = DLM.makeRNG(Math.round(t * 10000));
    container.innerHTML = '';
    sentence.forEach((tok, i) => {
      const el = document.createElement('span');
      el.className = 'token';
      const masked = rng() < t;
      if (masked) {
        el.classList.add('mask');
        el.style.boxShadow = '0 0 0 2px rgba(245, 181, 74, 0.3)';
        el.textContent = DLM.MASK;
      } else {
        el.classList.add('committed');
        el.style.opacity = '0.5';
        el.textContent = tok;
      }
      container.appendChild(el);
    });
  }

  function renderCurve() {
    const svg = d3.select('#viz5-1-curve');
    if (svg.empty()) return;
    svg.selectAll('*').remove();
    const W = 500, H = 160;
    const margin = { top: 20, right: 20, bottom: 30, left: 50 };
    const innerW = W - margin.left - margin.right;
    const innerH = H - margin.top - margin.bottom;
    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    // x-axis: t in [0.01, 1], y-axis: w(t) = 1/t in [1, 100]
    const xScale = d3.scaleLinear().domain([0.01, 1]).range([0, innerW]);
    const yScale = d3.scaleLog().domain([1, 100]).range([innerH, 0]);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format('.1f')))
      .selectAll('text').style('fill', 'var(--text-soft)').style('font-family', 'JetBrains Mono, monospace').style('font-size', '10px');
    g.selectAll('.domain, .tick line').style('stroke', 'var(--border-strong)');

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(4))
      .selectAll('text').style('fill', 'var(--text-soft)').style('font-family', 'JetBrains Mono, monospace').style('font-size', '10px');

    // Curve: 1/t
    const data = d3.range(0.01, 1.01, 0.01).map((tt) => ({ t: tt, w: 1 / tt }));
    const line = d3.line().x((d) => xScale(d.t)).y((d) => yScale(d.w)).curve(d3.curveBasis);
    g.append('path')
      .datum(data)
      .attr('fill', 'none')
      .style('stroke', 'var(--accent-2)')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Current-t marker
    const cx = xScale(t);
    const cy = yScale(Math.max(1, Math.min(100, 1 / t)));
    g.append('line')
      .attr('x1', cx).attr('x2', cx)
      .attr('y1', cy).attr('y2', innerH)
      .style('stroke', 'var(--accent)').attr('stroke-width', 1).attr('stroke-dasharray', '3,3');
    g.append('circle').attr('cx', cx).attr('cy', cy).attr('r', 5).style('fill', 'var(--accent)');

    // Labels
    g.append('text')
      .attr('x', innerW).attr('y', -6).attr('text-anchor', 'end')
      .style('fill', 'var(--text-soft)').style('font-family', 'JetBrains Mono, monospace').style('font-size', '11px')
      .text('w(t) = 1/t');
    g.append('text')
      .attr('x', innerW / 2).attr('y', innerH + 26).attr('text-anchor', 'middle')
      .style('fill', 'var(--text-muted)').style('font-family', 'JetBrains Mono, monospace').style('font-size', '10px')
      .text('mask rate t');
  }

  function init() {
    const slider = document.getElementById('viz5-1-t');
    const tval = document.getElementById('viz5-1-tval');
    const wval = document.getElementById('viz5-1-w');
    if (!slider) return;

    const update = () => {
      t = parseInt(slider.value) / 100;
      if (tval) tval.textContent = t.toFixed(2);
      if (wval) wval.textContent = (1 / t).toFixed(2);
      renderSentence();
      renderCurve();
    };
    slider.addEventListener('input', update);
    window.addEventListener('langchange', () => { renderSentence(); renderCurve(); });
    update();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
