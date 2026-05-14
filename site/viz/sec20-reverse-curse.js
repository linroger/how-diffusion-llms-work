/* §20 — Reverse curse demo */
(function () {
  'use strict';

  const PROMPTS_EN = [
    { forward: 'Olaf Scholz is the Chancellor of ____', back: 'The Chancellor of Germany is ____', forwardAns: 'Germany', backAns: 'Olaf Scholz', arForwardConf: 0.94, arBackConf: 0.31, diffForwardConf: 0.62, diffBackConf: 0.56 },
    { forward: 'The capital of France is ____', back: 'Paris is the capital of ____', forwardAns: 'Paris', backAns: 'France', arForwardConf: 0.97, arBackConf: 0.62, diffForwardConf: 0.59, diffBackConf: 0.55 },
    { forward: 'LLaDA was created by ____', back: '____ created LLaDA', forwardAns: 'Renmin/Ant', backAns: 'Renmin/Ant', arForwardConf: 0.71, arBackConf: 0.18, diffForwardConf: 0.58, diffBackConf: 0.54 },
  ];

  const PROMPTS_ZH = [
    { forward: '床前明月光，____', back: '____，疑是地上霜', forwardAns: '疑是地上霜', backAns: '床前明月光', arForwardConf: 0.83, arBackConf: 0.34, diffForwardConf: 0.55, diffBackConf: 0.50 },
    { forward: '春眠不觉晓，____', back: '____，处处闻啼鸟', forwardAns: '处处闻啼鸟', backAns: '春眠不觉晓', arForwardConf: 0.85, arBackConf: 0.42, diffForwardConf: 0.58, diffBackConf: 0.51 },
    { forward: '欲穷千里目，____', back: '____，更上一层楼', forwardAns: '更上一层楼', backAns: '欲穷千里目', arForwardConf: 0.88, arBackConf: 0.27, diffForwardConf: 0.55, diffBackConf: 0.50 },
  ];

  let promptIdx = 0;

  function render() {
    const container = document.getElementById('viz20-1');
    if (!container) return;
    container.innerHTML = '';
    const lang = document.documentElement.getAttribute('data-lang') || 'en';
    const prompts = lang === 'zh' ? PROMPTS_ZH : PROMPTS_EN;
    const p = prompts[promptIdx];

    // Prompt picker
    const picker = document.createElement('div');
    picker.style.display = 'flex';
    picker.style.gap = '8px';
    picker.style.marginBottom = '20px';
    prompts.forEach((pr, i) => {
      const btn = document.createElement('button');
      btn.style.padding = '6px 14px';
      btn.style.background = i === promptIdx ? '#f5b54a' : 'transparent';
      btn.style.color = i === promptIdx ? '#1a1308' : '#a9a8a3';
      btn.style.border = '1px solid ' + (i === promptIdx ? '#f5b54a' : '#2e3648');
      btn.style.borderRadius = '4px';
      btn.style.fontFamily = 'Inter, sans-serif';
      btn.style.fontSize = '0.82rem';
      btn.style.cursor = 'pointer';
      btn.textContent = lang === 'zh' ? `例 ${i + 1}` : `Example ${i + 1}`;
      btn.addEventListener('click', () => { promptIdx = i; render(); });
      picker.appendChild(btn);
    });
    container.appendChild(picker);

    // Two-column compare
    const cols = document.createElement('div');
    cols.style.display = 'grid';
    cols.style.gridTemplateColumns = '1fr 1fr';
    cols.style.gap = '20px';
    if (window.innerWidth < 720) cols.style.gridTemplateColumns = '1fr';

    cols.appendChild(renderCol(lang === 'zh' ? '正向（A → B）' : 'Forward (A → B)', p.forward, p.forwardAns, p.arForwardConf, p.diffForwardConf, lang));
    cols.appendChild(renderCol(lang === 'zh' ? '反向（B → A）' : 'Reverse (B → A)', p.back, p.backAns, p.arBackConf, p.diffBackConf, lang));
    container.appendChild(cols);

    // Conclusion
    const conclusion = document.createElement('div');
    conclusion.style.marginTop = '20px';
    conclusion.style.padding = '14px 18px';
    conclusion.style.background = '#1c2230';
    conclusion.style.borderRadius = '6px';
    conclusion.style.borderLeft = '3px solid #f5b54a';
    conclusion.style.fontFamily = 'Source Serif 4, serif';
    conclusion.style.fontSize = '0.95rem';
    conclusion.style.color = '#e8e6e1';
    conclusion.style.fontStyle = 'italic';
    if (lang === 'zh') {
      conclusion.innerHTML = '从左到右训练的自回归模型，从未把 "A 是 B" 视作 "B 是 A" 的证据。双向训练同时看到两个方向，因此其反向回答几乎与正向同样自信。';
      conclusion.style.fontFamily = 'Inter, sans-serif';
      conclusion.style.fontStyle = 'normal';
    } else {
      conclusion.innerHTML = '<em>The autoregressive model, trained left-to-right, never sees "A is B" as evidence for "B is A." The diffusion model, trained bidirectionally, sees both directions equally — its reverse answer is nearly as confident as its forward answer.</em>';
    }
    container.appendChild(conclusion);
  }

  function renderCol(title, prompt, answer, arConf, diffConf, lang) {
    const col = document.createElement('div');
    col.style.padding = '18px';
    col.style.background = '#11151f';
    col.style.borderRadius = '6px';
    col.style.border = '1px solid #2e3648';
    col.innerHTML = `
      <div style="font-family:Inter,sans-serif;font-size:0.74rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#a9a8a3;margin-bottom:12px">${title}</div>
      <div style="font-family:JetBrains Mono,monospace;font-size:0.94rem;color:#fbfaf7;margin-bottom:18px;line-height:1.4">${prompt}</div>
      <div style="margin-bottom:12px">
        <div style="display:flex;justify-content:space-between;font-family:Inter,sans-serif;font-size:0.78rem;color:#5fb0c7;margin-bottom:4px">
          <span>AR (GPT-style)</span>
          <span style="font-family:JetBrains Mono,monospace">${(arConf * 100).toFixed(1)}%</span>
        </div>
        <div style="height:8px;background:#1c2230;border-radius:4px;overflow:hidden">
          <div style="height:100%;width:${arConf * 100}%;background:#5fb0c7"></div>
        </div>
        <div style="font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#a9a8a3;margin-top:4px">→ "${answer}"</div>
      </div>
      <div>
        <div style="display:flex;justify-content:space-between;font-family:Inter,sans-serif;font-size:0.78rem;color:#f5b54a;margin-bottom:4px">
          <span>Diffusion (LLaDA-style)</span>
          <span style="font-family:JetBrains Mono,monospace">${(diffConf * 100).toFixed(1)}%</span>
        </div>
        <div style="height:8px;background:#1c2230;border-radius:4px;overflow:hidden">
          <div style="height:100%;width:${diffConf * 100}%;background:#f5b54a"></div>
        </div>
        <div style="font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#a9a8a3;margin-top:4px">→ "${answer}"</div>
      </div>
    `;
    return col;
  }

  function init() {
    render();
    window.addEventListener('langchange', render);
    window.addEventListener('resize', render);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
