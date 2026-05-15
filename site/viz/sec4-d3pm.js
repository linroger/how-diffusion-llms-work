/* §4 — D3PM three-variant corruption playground */
(function () {
  'use strict';

  let variant = 'absorbing';
  let t = 0;

  // For "uniform" and "gaussian" — when a token gets corrupted, what does it get replaced with?
  const RANDOM_TOKENS = ['the', 'dog', '##ing', '了', 'def', '猫', '0x4a', 'fox', 'attention', 'import', '。', 'class', '\\n', 'self'];
  // For "gaussian-over-tokens" we use a "nearby" replacement — same first letter or adjacent length
  function gaussianReplace(tok, rng) {
    // Just pick a similar-length string
    const candidates = RANDOM_TOKENS.filter((w) => Math.abs(w.length - tok.length) <= 1);
    return candidates[Math.floor(rng() * candidates.length)] || tok;
  }

  function render() {
    const container = document.getElementById('viz4-1-output');
    if (!container) return;
    const sentence = DLM.pickSentence(0);
    const rng = DLM.makeRNG(7 + Math.round(t * 1000));
    container.innerHTML = '';
    sentence.forEach((tok, i) => {
      const el = document.createElement('span');
      el.className = 'token';
      const corrupted = rng() < t;
      if (!corrupted) {
        el.classList.add('committed');
        el.textContent = tok;
      } else if (variant === 'absorbing') {
        el.classList.add('mask');
        el.textContent = DLM.MASK;
      } else if (variant === 'uniform') {
        el.style.background = 'rgba(232, 121, 168, 0.12)';
        el.style.borderColor = 'var(--accent-3)';
        el.style.color = 'var(--accent-3)';
        el.textContent = RANDOM_TOKENS[Math.floor(rng() * RANDOM_TOKENS.length)];
      } else {
        // gaussian
        el.style.background = 'rgba(95, 176, 199, 0.12)';
        el.style.borderColor = 'var(--accent-2)';
        el.style.color = 'var(--accent-2)';
        el.textContent = gaussianReplace(tok, rng);
      }
      el.style.animationDelay = (i * 18) + 'ms';
      container.appendChild(el);
    });
  }

  function init() {
    const slider = document.getElementById('viz4-1-t');
    const tval = document.getElementById('viz4-1-tval');
    const buttons = document.querySelectorAll('#viz4-1 .variant-btn');

    if (slider) {
      slider.addEventListener('input', () => {
        t = parseInt(slider.value) / 100;
        if (tval) tval.textContent = t.toFixed(2);
        render();
      });
    }
    buttons.forEach((btn) => {
      btn.addEventListener('click', () => {
        buttons.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        variant = btn.dataset.variant;
        render();
      });
    });
    window.addEventListener('langchange', render);
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
