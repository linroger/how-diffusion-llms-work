/* §2 — Image diffusion forward/reverse on a canvas
   We render three frames side by side (clean / mid / noisy) and let the slider
   interpolate the middle one between clean and pure-noise. */

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

  const W = 480, H = 160;
  const TILE_W = 152, TILE_H = 120;
  const GAP = 12;
  const PAD_X = (W - TILE_W * 3 - GAP * 2) / 2;
  const PAD_Y = (H - TILE_H) / 2;

  // Generate a "clean" image: a simple gradient with shapes
  function generateClean() {
    const img = new Uint8ClampedArray(TILE_W * TILE_H * 4);
    for (let y = 0; y < TILE_H; y++) {
      for (let x = 0; x < TILE_W; x++) {
        const idx = (y * TILE_W + x) * 4;
        // background gradient
        const r0 = 30 + (x / TILE_W) * 90;
        const g0 = 40 + (y / TILE_H) * 100;
        const b0 = 80 + ((x + y) / (TILE_W + TILE_H)) * 100;
        // a "face" - circles
        const cx1 = TILE_W * 0.35, cy1 = TILE_H * 0.4, r = 18;
        const d1 = Math.hypot(x - cx1, y - cy1);
        const cx2 = TILE_W * 0.65, cy2 = TILE_H * 0.4;
        const d2 = Math.hypot(x - cx2, y - cy2);
        // smile arc
        const sx = TILE_W * 0.5, sy = TILE_H * 0.65;
        const ds = Math.abs(Math.hypot(x - sx, y - sy) - 24);
        const inSmile = ds < 3 && y > sy;
        let r1 = r0, g1 = g0, b1 = b0;
        if (d1 < r || d2 < r) {
          r1 = 240; g1 = 200; b1 = 80;
        } else if (inSmile) {
          r1 = 240; g1 = 200; b1 = 80;
        }
        img[idx] = r1;
        img[idx + 1] = g1;
        img[idx + 2] = b1;
        img[idx + 3] = 255;
      }
    }
    return img;
  }

  function noisedImage(clean, t) {
    // forward: x_t = sqrt(1-t) * x_0 + sqrt(t) * N(0, 255)
    const out = new Uint8ClampedArray(clean.length);
    const alpha = Math.sqrt(1 - t);
    const sigma = Math.sqrt(t) * 110;
    for (let i = 0; i < clean.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const noise = (Math.random() * 2 - 1) * sigma;
        out[i + c] = Math.max(0, Math.min(255, alpha * clean[i + c] + 128 * (1 - alpha) + noise));
      }
      out[i + 3] = 255;
    }
    return out;
  }

  let cleanImg = null;
  let showReverse = false;

  function draw(t) {
    const canvas = document.getElementById('viz2-1-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = __resolveColor('var(--bg-frame-2)');
    ctx.fillRect(0, 0, W, H);

    if (!cleanImg) cleanImg = generateClean();

    // Three tiles. Left = clean. Middle = at time t. Right = pure noise.
    const middleT = showReverse ? (1 - t) : t;
    const tiles = [
      { x: PAD_X, img: cleanImg, label: 't=0' },
      { x: PAD_X + TILE_W + GAP, img: noisedImage(cleanImg, middleT), label: `t=${middleT.toFixed(2)}` },
      { x: PAD_X + (TILE_W + GAP) * 2, img: noisedImage(cleanImg, 1.0), label: 't=1' },
    ];

    tiles.forEach((tile, i) => {
      const imageData = ctx.createImageData(TILE_W, TILE_H);
      imageData.data.set(tile.img);
      ctx.putImageData(imageData, tile.x, PAD_Y);
      // Border
      ctx.strokeStyle = i === 1 ? 'var(--accent)' : 'var(--border-strong)';
      ctx.lineWidth = i === 1 ? 2 : 1;
      ctx.strokeRect(tile.x - 0.5, PAD_Y - 0.5, TILE_W + 1, TILE_H + 1);
      // Label
      ctx.fillStyle = __resolveColor('var(--text-soft)');
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(tile.label, tile.x + TILE_W / 2, PAD_Y + TILE_H + 14);
    });

    // Arrows
    ctx.strokeStyle = __resolveColor('var(--text-muted)');
    ctx.lineWidth = 1;
    for (let i = 0; i < 2; i++) {
      const x1 = PAD_X + TILE_W + (TILE_W + GAP) * i;
      const x2 = x1 + GAP;
      const y = PAD_Y + TILE_H / 2;
      ctx.beginPath();
      ctx.moveTo(x1 + 2, y);
      ctx.lineTo(x2 - 2, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x2 - 5, y - 3);
      ctx.lineTo(x2 - 2, y);
      ctx.lineTo(x2 - 5, y + 3);
      ctx.stroke();
    }
  }

  function init() {
    const slider = document.getElementById('viz2-1-t');
    const tval = document.getElementById('viz2-1-tval');
    const toggle = document.getElementById('viz2-1-toggle');
    if (!slider) return;

    const update = () => {
      const t = parseInt(slider.value) / 100;
      if (tval) tval.textContent = t.toFixed(2);
      draw(t);
    };

    slider.addEventListener('input', update);
    toggle?.addEventListener('click', () => {
      showReverse = !showReverse;
      toggle.textContent = showReverse
        ? (document.documentElement.getAttribute('data-lang') === 'zh' ? '显示前向' : 'Show forward')
        : (document.documentElement.getAttribute('data-lang') === 'zh' ? '显示反向' : 'Show reverse');
      update();
    });

    // re-render frequently when slider moves (noise is stochastic)
    update();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
