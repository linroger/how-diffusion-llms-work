/* §3 — Pixel continuum vs token grid hover comparison */
(function () {
  'use strict';
  function __cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback || '#000';
  }
  function __resolveColor(s) {
    if (typeof s !== 'string') return s;
    return s.replace(/var\((--[\w-]+)\)/g, (_m, n) => __cssVar(n));
  }

  // Pixel pane: a gradient ramp; on hover, show value at mouse position.
  function initPixel() {
    const canvas = document.getElementById('viz3-1-pixel');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    // Color stripes — smooth gradient
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0, '#1e2233');
    grad.addColorStop(0.3, '#3b4d6e');
    grad.addColorStop(0.6, __resolveColor('var(--accent-2)'));
    grad.addColorStop(1, __resolveColor('var(--accent)'));
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    // Vertical gradient too
    const gradV = ctx.createLinearGradient(0, 0, 0, H);
    gradV.addColorStop(0, 'rgba(11,14,21,0)');
    gradV.addColorStop(1, 'rgba(11,14,21,0.55)');
    ctx.fillStyle = gradV;
    ctx.fillRect(0, 0, W, H);

    let hoverX = -1, hoverY = -1;

    canvas.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      hoverX = (e.clientX - rect.left) * (W / rect.width);
      hoverY = (e.clientY - rect.top) * (H / rect.height);
      redraw();
    });

    canvas.addEventListener('mouseleave', () => { hoverX = -1; redraw(); });

    function redraw() {
      // Re-paint the gradient
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = gradV;
      ctx.fillRect(0, 0, W, H);
      if (hoverX < 0) return;
      // Crosshair
      ctx.strokeStyle = 'rgba(255,255,255,0.6)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hoverX, 0); ctx.lineTo(hoverX, H);
      ctx.moveTo(0, hoverY); ctx.lineTo(W, hoverY);
      ctx.stroke();
      // Readout: r, g, b at hover
      const pix = ctx.getImageData(Math.round(hoverX), Math.round(hoverY), 1, 1).data;
      ctx.fillStyle = 'rgba(0,0,0,0.7)';
      const tx = Math.min(hoverX + 8, W - 90);
      const ty = Math.min(hoverY + 18, H - 8);
      ctx.fillRect(tx, ty - 14, 84, 18);
      ctx.fillStyle = __resolveColor('var(--text)');
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`(${pix[0]}, ${pix[1]}, ${pix[2]})`, tx + 4, ty - 1);
    }
  }

  // Token grid: 8x5 grid of "tokens" with random words; hover shows ID
  const VOCAB_WORDS = [
    'the', 'and', 'of', 'a', 'in', 'to', 'is', 'was',
    'dog', '##ing', '了', 'def', 'cat', '猫', 'hello', '$',
    'Llama', 'mask', '0x4a', '\\n', 'token', '##ed', 'attention', '——',
    'GPT', '了', '。', 'Antonio', 'fox', '##s', 'def', 'class',
    'return', 'self', 'lambda', 'import', 'if', 'else', '\\t', '我',
  ];

  function initTokenGrid() {
    const grid = document.getElementById('viz3-1-token');
    if (!grid) return;
    grid.innerHTML = '';
    for (let i = 0; i < 40; i++) {
      const cell = document.createElement('div');
      cell.className = 'token-grid-cell';
      cell.textContent = VOCAB_WORDS[i % VOCAB_WORDS.length];
      cell.dataset.id = i + 1;
      cell.addEventListener('mouseenter', () => {
        grid.querySelectorAll('.token-grid-cell').forEach((c) => c.classList.remove('active'));
        cell.classList.add('active');
        cell.title = `id #${cell.dataset.id}: ${cell.textContent}`;
      });
      grid.appendChild(cell);
    }
  }

  function init() {
    initPixel();
    initTokenGrid();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
