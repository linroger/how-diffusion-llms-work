/* §6 — Three attention-mask heatmaps: causal, bidirectional, block */
(function () {
  'use strict';

  const N = 16; // sequence length to render
  let blockSize = 4;

  function drawMask(svgId, allowedFn) {
    const svg = d3.select('#' + svgId);
    if (svg.empty()) return;
    svg.selectAll('*').remove();
    const W = 200, H = 200;
    const pad = 8;
    const cell = (W - pad * 2) / N;
    const g = svg.append('g').attr('transform', `translate(${pad},${pad})`);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const allowed = allowedFn(i, j);
        g.append('rect')
          .attr('x', j * cell)
          .attr('y', i * cell)
          .attr('width', cell - 0.5)
          .attr('height', cell - 0.5)
          .attr('fill', allowed ? '#f5b54a' : '#1c2230')
          .attr('opacity', allowed ? 0.85 : 1);
      }
    }
    // axis hint
    g.append('text').attr('x', -4).attr('y', cell - 2).attr('fill', '#6c6d72').style('font-family', 'JetBrains Mono, monospace').style('font-size', '8px').attr('text-anchor', 'end').text('i');
    g.append('text').attr('x', 0).attr('y', N * cell + 10).attr('fill', '#6c6d72').style('font-family', 'JetBrains Mono, monospace').style('font-size', '8px').text('j →');
  }

  function renderAll() {
    drawMask('viz6-1-causal', (i, j) => j <= i);
    drawMask('viz6-1-bidir', () => true);
    drawMask('viz6-1-block', (i, j) => {
      const bi = Math.floor(i / blockSize);
      const bj = Math.floor(j / blockSize);
      // within block: any j; between blocks: j's block <= i's block
      return bi === bj || bj < bi;
    });
  }

  function init() {
    const slider = document.getElementById('viz6-1-bs');
    const bsval = document.getElementById('viz6-1-bsval');
    if (slider) {
      slider.addEventListener('input', () => {
        blockSize = parseInt(slider.value);
        if (bsval) bsval.textContent = blockSize;
        renderAll();
      });
    }
    renderAll();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
