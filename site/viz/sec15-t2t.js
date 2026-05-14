/* §15 — T2T (Token-to-Token) editing demo */
(function () {
  'use strict';

  // We script a small narrative trace where the model:
  //  step 1: commits some tokens, one of which is WRONG ("the" instead of "a")
  //  step 2: more context arrives, more tokens commit
  //  step 3: the model realizes the early "the" should be "a" — T2T edit triggers, replaces it
  //  step 4: final state
  // The user can adjust τ_mask / τ_edit. If τ_edit > 0.8 the edit may not trigger.

  let tmask = 0.7;
  let tedit = 0.5;
  let step = 0;
  let playing = false;
  let timer = null;

  // Hard-coded narrative — token, isWrongAtFirst (boolean), commitStep, editStep, confidenceWhenWrong, confidenceOfCorrect
  function getNarrative() {
    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    if (lang === 'zh') {
      return {
        prompt: '请把以下英文翻成中文：',
        finalSentence: ['请', '把', '这', '段', '英文', '翻译', '为', '中文', '。'],
        firstSentence:  ['请', '将', '这', '段', '英文', 'mask', 'mask', 'mask', 'mask'],
        secondSentence: ['请', '将', '这', '段', '英文', '翻译', '为', 'mask', 'mask'],
        thirdSentence:  ['请', '将', '这', '段', '英文', '翻译', '为', '中文', '。'],
        afterEdit:     ['请', '把', '这', '段', '英文', '翻译', '为', '中文', '。'],
        wrongIdx: 1, // position of '将' that gets edited to '把'
      };
    }
    return {
      prompt: 'Diffusion models',
      finalSentence: ['Diffusion', 'models', 'generate', 'text', 'by', 'denoising', 'a', 'tape', 'of', 'masks', '.'],
      firstSentence:  ['Diffusion', 'models', 'generate', 'text', 'by', 'mask', 'the', 'mask', 'mask', 'mask', '.'],
      secondSentence: ['Diffusion', 'models', 'generate', 'text', 'by', 'denoising', 'the', 'tape', 'mask', 'mask', '.'],
      thirdSentence:  ['Diffusion', 'models', 'generate', 'text', 'by', 'denoising', 'the', 'tape', 'of', 'masks', '.'],
      afterEdit:     ['Diffusion', 'models', 'generate', 'text', 'by', 'denoising', 'a', 'tape', 'of', 'masks', '.'],
      wrongIdx: 6,
    };
  }

  function render() {
    const container = document.getElementById('viz15-1');
    if (!container) return;
    container.innerHTML = '';

    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    const n = getNarrative();

    // Threshold check: edit triggers when tedit <= 0.65
    const editTriggers = tedit <= 0.65;

    const stepNames = ['Initial', 'Step 1', 'Step 2', 'Step 3', editTriggers ? 'Step 4: T2T edit' : 'Step 4: no edit'];
    const stepNamesZh = ['初始', '步骤 1', '步骤 2', '步骤 3', editTriggers ? '步骤 4：T2T 编辑' : '步骤 4：无编辑'];
    const names = lang === 'zh' ? stepNamesZh : stepNames;

    // Stages
    const stages = [
      n.finalSentence.map(() => DLM.MASK), // initial
      n.firstSentence,
      n.secondSentence,
      n.thirdSentence,
      editTriggers ? n.afterEdit : n.thirdSentence,
    ];

    // Render the current stage's sentence as tokens
    const stageTitle = document.createElement('div');
    stageTitle.style.fontFamily = 'JetBrains Mono, monospace';
    stageTitle.style.fontSize = '0.86rem';
    stageTitle.style.color = '#f5b54a';
    stageTitle.style.marginBottom = '12px';
    stageTitle.textContent = names[step];
    container.appendChild(stageTitle);

    const strip = document.createElement('div');
    strip.className = 'token-strip';
    strip.style.padding = '0';
    strip.style.minHeight = '40px';
    const prev = step > 0 ? stages[step - 1] : stages[0];
    const cur = stages[step];
    cur.forEach((tok, i) => {
      const el = document.createElement('span');
      el.className = 'token';
      const isMask = tok === DLM.MASK || tok === 'mask';
      const wasMask = prev[i] === DLM.MASK || prev[i] === 'mask';
      if (isMask) {
        el.classList.add('mask');
        el.textContent = DLM.MASK;
      } else if (wasMask) {
        el.classList.add('commit');
        el.textContent = tok;
      } else if (prev[i] !== tok) {
        // edited!
        el.classList.add('edited');
        el.textContent = tok;
        el.title = `T2T edit: "${prev[i]}" → "${tok}"`;
      } else {
        el.classList.add('committed');
        el.textContent = tok;
      }
      strip.appendChild(el);
    });
    container.appendChild(strip);

    // Δ_t / Γ_t indicators (only at the edit step)
    if (step === 4) {
      const sets = document.createElement('div');
      sets.style.marginTop = '14px';
      sets.style.padding = '12px';
      sets.style.background = '#1c2230';
      sets.style.borderRadius = '6px';
      sets.style.fontFamily = 'JetBrains Mono, monospace';
      sets.style.fontSize = '0.82rem';
      if (editTriggers) {
        const wrongTok = n.thirdSentence[n.wrongIdx];
        const rightTok = n.afterEdit[n.wrongIdx];
        sets.innerHTML = `
          <div style="color:#a9a8a3;margin-bottom:6px">${lang === 'zh' ? '编辑触发：' : 'Edit triggered:'}</div>
          <div style="color:#e879a8">Δ_t = { ${n.wrongIdx}: "${wrongTok}" → "${rightTok}",  conf = 0.78 > τ_edit = ${tedit.toFixed(2)} ✓ }</div>
        `;
      } else {
        sets.innerHTML = `
          <div style="color:#a9a8a3;margin-bottom:6px">${lang === 'zh' ? '编辑未触发 (τ_edit 太高)：' : 'Edit blocked (τ_edit too high):'}</div>
          <div style="color:#6c6d72">Δ_t = { },  conf = 0.78 < τ_edit = ${tedit.toFixed(2)}</div>
        `;
      }
      container.appendChild(sets);
    }

    // Step navigation
    const nav = document.createElement('div');
    nav.style.display = 'flex';
    nav.style.gap = '6px';
    nav.style.marginTop = '16px';
    for (let i = 0; i < 5; i++) {
      const dot = document.createElement('div');
      dot.style.width = '24px';
      dot.style.height = '24px';
      dot.style.borderRadius = '50%';
      dot.style.background = i === step ? '#f5b54a' : (i < step ? '#3b4458' : '#1c2230');
      dot.style.color = i === step ? '#1a1308' : '#a9a8a3';
      dot.style.display = 'flex';
      dot.style.alignItems = 'center';
      dot.style.justifyContent = 'center';
      dot.style.fontFamily = 'JetBrains Mono, monospace';
      dot.style.fontSize = '0.7rem';
      dot.style.cursor = 'pointer';
      dot.textContent = i;
      dot.addEventListener('click', () => { step = i; render(); });
      nav.appendChild(dot);
    }
    container.appendChild(nav);
  }

  function play() {
    if (playing) return;
    playing = true;
    step = 0;
    render();
    const tick = () => {
      step++;
      render();
      if (step < 4) timer = setTimeout(tick, 1000);
      else playing = false;
    };
    timer = setTimeout(tick, 800);
  }

  function reset() {
    if (timer) clearTimeout(timer);
    playing = false;
    step = 0;
    render();
  }

  function init() {
    const tmSlider = document.getElementById('viz15-1-tm');
    const tmval = document.getElementById('viz15-1-tmval');
    const teSlider = document.getElementById('viz15-1-te');
    const teval = document.getElementById('viz15-1-teval');
    if (tmSlider) tmSlider.addEventListener('input', () => { tmask = parseInt(tmSlider.value) / 100; if (tmval) tmval.textContent = tmask.toFixed(2); render(); });
    if (teSlider) teSlider.addEventListener('input', () => { tedit = parseInt(teSlider.value) / 100; if (teval) teval.textContent = tedit.toFixed(2); render(); });
    document.getElementById('viz15-1-play')?.addEventListener('click', play);
    document.getElementById('viz15-1-reset')?.addEventListener('click', reset);
    window.addEventListener('langchange', render);
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
