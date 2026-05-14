/* §1 — AR vs diffusion side-by-side animation */
(function () {
  'use strict';

  let timer = null;
  let arStep = 0, diffStep = 0;
  let playing = false;

  function render() {
    const arEl = document.getElementById('viz1-1-ar');
    const diffEl = document.getElementById('viz1-1-diff');
    if (!arEl || !diffEl) return;

    const target = DLM.pickSentence(1);

    // AR: positions 0..arStep are committed (left to right), rest are not yet emitted
    const arState = target.map((tok, i) => {
      if (i < arStep) return { token: tok, state: 'committed' };
      if (i === arStep - 1) return { token: tok, state: 'commit' };
      return { token: '', state: 'mask' };
    });
    // Render AR with hidden tokens shown as empty mask-like cells
    arEl.innerHTML = '';
    arState.forEach((cell, i) => {
      const el = document.createElement('span');
      el.className = 'token';
      if (i >= arStep) {
        el.classList.add('mask');
        el.textContent = '·';
      } else if (i === arStep - 1) {
        el.classList.add('commit');
        el.textContent = cell.token;
      } else {
        el.classList.add('committed');
        el.textContent = cell.token;
      }
      arEl.appendChild(el);
    });

    // Diffusion: each step commits ~25% of positions; 4 steps total
    const commitSchedule = computeDiffSchedule(target.length, 4);
    diffEl.innerHTML = '';
    target.forEach((tok, i) => {
      const el = document.createElement('span');
      el.className = 'token';
      const committedAt = commitSchedule[i];
      if (diffStep >= committedAt) {
        el.classList.add(committedAt === diffStep ? 'commit' : 'committed');
        el.textContent = tok;
      } else {
        el.classList.add('mask');
        el.textContent = DLM.MASK;
      }
      diffEl.appendChild(el);
    });

    const arMeta = document.getElementById('viz1-1-ar-meta');
    const diffMeta = document.getElementById('viz1-1-diff-meta');
    if (arMeta) arMeta.textContent = `step ${arStep} / ${target.length}`;
    if (diffMeta) diffMeta.textContent = `step ${diffStep} / 4`;
  }

  function computeDiffSchedule(n, steps) {
    const result = new Array(n);
    const rng = DLM.makeRNG(99);
    for (let i = 0; i < n; i++) {
      const r = rng();
      if (r < 0.30) result[i] = 1;
      else if (r < 0.60) result[i] = 2;
      else if (r < 0.85) result[i] = 3;
      else result[i] = 4;
    }
    return result;
  }

  function getInterval() {
    const slider = document.getElementById('viz1-1-speed');
    const speed = slider ? parseInt(slider.value) : 5;
    // speed 1 = slow (1200ms), speed 10 = fast (150ms)
    return 1300 - speed * 110;
  }

  function play() {
    if (playing) return;
    playing = true;
    arStep = 0;
    diffStep = 0;
    render();
    const tick = () => {
      const target = DLM.pickSentence(1);
      const arDone = arStep >= target.length;
      const diffDone = diffStep >= 4;
      if (arDone && diffDone) {
        playing = false;
        return;
      }
      if (!arDone) arStep++;
      // diffusion runs at target.length / 4 frequency
      const diffEveryN = Math.max(1, Math.floor(target.length / 4));
      if (!diffDone && arStep % diffEveryN === 0) diffStep++;
      render();
      timer = setTimeout(tick, getInterval());
    };
    timer = setTimeout(tick, getInterval());
  }

  function reset() {
    if (timer) clearTimeout(timer);
    playing = false;
    arStep = 0;
    diffStep = 0;
    render();
  }

  function init() {
    document.getElementById('viz1-1-play')?.addEventListener('click', play);
    document.getElementById('viz1-1-reset')?.addEventListener('click', reset);
    window.addEventListener('langchange', render);
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
