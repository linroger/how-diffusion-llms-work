/* Extra 8 — Coordination Problem animator
   A chain of 5 reasoning steps. AR commits step-by-step (no errors, slow).
   Diffusion commits all in parallel: if step 1 commits wrong, steps 2-5 cascade.
   Click "Step" to advance. Watch how each paradigm handles a multi-hop chain. */
(function () {
  'use strict';

  const STEPS_EN = [
    { text: 'all dogs are mammals' },
    { text: 'all mammals are vertebrates' },
    { text: 'all vertebrates have a spine' },
    { text: 'Fido is a dog' },
    { text: '∴ Fido has a spine' },
  ];

  const STEPS_ZH = [
    { text: '所有狗都是哺乳动物' },
    { text: '所有哺乳动物都是脊椎动物' },
    { text: '所有脊椎动物都有脊柱' },
    { text: 'Fido 是一只狗' },
    { text: '∴ Fido 有脊柱' },
  ];

  let mode = 'ar';
  let step = 0;
  let arError = false;
  let diffError = false;

  function render() {
    const container = document.getElementById('vizX8');
    if (!container) return;
    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    const steps = lang === 'zh' ? STEPS_ZH : STEPS_EN;
    container.innerHTML = '';

    const ctl = document.createElement('div');
    ctl.className = 'x8-mode';
    ctl.innerHTML = `
      <button class="x8-modebtn ${mode==='ar'?'active':''}" data-m="ar">${lang==='zh'?'自回归（每步一次提交）':'Autoregressive (1 step at a time)'}</button>
      <button class="x8-modebtn ${mode==='diff'?'active':''}" data-m="diff">${lang==='zh'?'扩散（全部并行提交）':'Diffusion (all in parallel)'}</button>
    `;
    container.appendChild(ctl);
    ctl.querySelectorAll('.x8-modebtn').forEach((b) => {
      b.addEventListener('click', () => { mode = b.dataset.m; step = 0; render(); });
    });

    const chain = document.createElement('div');
    chain.className = 'x8-chain';
    steps.forEach((s, i) => {
      const row = document.createElement('div');
      row.className = 'x8-row';
      let state = 'pending';
      if (mode === 'ar') {
        if (i < step) state = 'committed';
        else if (i === step - 1) state = 'commit';
      } else {
        // Diffusion: at step >= 1, all 5 commit at once
        if (step >= 1) {
          state = diffError && i >= 1 ? 'wrong' : 'committed';
          if (i === 0 && diffError) state = 'wrongseed';
        }
      }
      row.classList.add('x8-state-' + state);
      row.innerHTML = `
        <div class="x8-num">${i+1}</div>
        <div class="x8-text">${s.text}</div>
        <div class="x8-status">${
          state === 'committed' ? (lang==='zh'?'✓ 已确认':'✓ committed') :
          state === 'commit' ? (lang==='zh'?'⟳ 正在提交':'⟳ committing') :
          state === 'wrong' ? (lang==='zh'?'✗ 受错传染':'✗ cascaded error') :
          state === 'wrongseed' ? (lang==='zh'?'✗ 起点已错':'✗ wrong seed') :
          (lang==='zh'?'… 等待':'… pending')
        }</div>
      `;
      chain.appendChild(row);
    });
    container.appendChild(chain);

    const ctrls = document.createElement('div');
    ctrls.className = 'x8-ctrls';
    if (mode === 'ar') {
      ctrls.innerHTML = `
        <button class="btn" id="x8-next">${lang==='zh'?'下一步 →':'Next step →'}</button>
        <button class="btn btn-ghost" id="x8-reset">${lang==='zh'?'↺ 重置':'↺ Reset'}</button>
        <span class="x8-readout">${lang==='zh'?'步骤':'step'} ${step} / ${steps.length}</span>
      `;
    } else {
      ctrls.innerHTML = `
        <button class="btn" id="x8-fire" ${step>=1?'disabled':''}>${lang==='zh'?'并行提交 5 个 token':'Commit all 5 in parallel'}</button>
        <button class="btn btn-ghost" id="x8-error" ${step<1?'disabled':''}>${diffError?(lang==='zh'?'✓ 错误已注入':'✓ Error injected'):(lang==='zh'?'注入早期错误':'Inject step-1 error')}</button>
        <button class="btn btn-ghost" id="x8-reset">${lang==='zh'?'↺ 重置':'↺ Reset'}</button>
      `;
    }
    container.appendChild(ctrls);

    document.getElementById('x8-next')?.addEventListener('click', () => { step = Math.min(steps.length, step + 1); render(); });
    document.getElementById('x8-fire')?.addEventListener('click', () => { step = 1; render(); });
    document.getElementById('x8-error')?.addEventListener('click', () => { diffError = true; render(); });
    document.getElementById('x8-reset')?.addEventListener('click', () => { step = 0; arError = false; diffError = false; render(); });

    // Caption
    const cap = document.createElement('div');
    cap.className = 'x8-caption';
    cap.innerHTML = mode === 'ar' ?
      (lang === 'zh' ?
        `AR 顺序提交：每一步都基于上一步的精确结果。慢，但每个 token 都是在 <em>正确</em> 的前缀条件下生成的。多跳推理这种串行依赖任务上，AR 占优。` :
        `AR commits step by step, each conditioned on the exact text of the previous one. Slow, but every token is generated under a <em>correct</em> prefix. On multi-hop reasoning, this serial determinism is AR's structural advantage.`) :
      (lang === 'zh' ?
        (diffError ?
          `如果第 1 步在并行提交时落子错误，第 2–5 步全部建立在错误前提之上 —— 错误级联，无法挽回。这就是<strong>协调问题</strong>。T2T 与 MBE 编辑就是为了缓解这种情形（用代价：增加重新评估的开销）。` :
          `扩散一次提交所有 5 个 token —— 速度的来源。但每个 token 都必须 <em>独立可信</em>，因为它看不到其他 4 个的确切文本，只能看到当前的（部分掩码）上下文。点击"注入早期错误"看看会发生什么。`) :
        (diffError ?
          `Step 1 commits a wrong token under parallel commit. Steps 2–5 are now conditioned on garbage — and they were all committed simultaneously, so there's no per-step verification. This is the <strong>coordination problem</strong>. T2T and MBE editing are designed to mitigate it, at the cost of extra revisit passes.` :
          `Diffusion commits all 5 tokens in one parallel pass — that's where the speed comes from. But each token must be <em>independently plausible</em> without knowing the exact text of the other 4. Click "Inject step-1 error" to see what happens when that lie isn't innocent.`));
    container.appendChild(cap);
  }

  function init() {
    window.addEventListener('langchange', render);
    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
