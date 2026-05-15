# 扩散语言模型是怎么工作的

一份单页、中英双语的扩散语言模型讲解。开头讨论"为什么不能把图像扩散直接搬到文本上"，结尾大致落在 Gemini Diffusion 和蚂蚁集团的 LLaDA 系列上，中间穿插着十几个可以动手试的交互演示。

🌐 **在线版本：** [linroger.github.io/how-diffusion-llms-work](https://linroger.github.io/how-diffusion-llms-work/)

🇨🇳 中文 README · [🇬🇧 English README](README.md)

---

## 页面里有什么

整页分成六个部分。可以从头读到尾，也可以直接跳到自己卡住的那一段。

- **基础（Foundations）** —— 把像素换成 token 之后到底变了什么？为什么需要离散扩散？MD4 / MDLM 的那次"简化"为什么让整件事忽然就能训练了？
- **架构（Architecture）** —— 训成扩散模型之后，Transformer 哪里不一样？基本上只是注意力掩码而已。
- **训练（Training）** —— 慢镜头走一遍单个训练步骤、对比几种噪声调度、看看蚂蚁集团怎么把已有的 AR 检查点接着用，而不是从零开始。
- **推理（Inference）** —— 去噪循环、四种重新掩码策略、块扩散、置信度感知并行解码，以及 LLaDA 2.1 怎么修改自己已经写下的 token。
- **技术栈（The Stacks）** —— DeepMind 的路线（MD4 → AR2Diff → Gemini Diffusion）和蚂蚁集团的开源线（LLaDA → LLaDA 2.1）。
- **实验场（Playground）** —— 自己跑一次去噪循环，亲眼看一下 reverse curse 是怎么发生的。

每一节至少配一张可以动手试的图。所有东西都在浏览器里跑，背后没有任何服务端模型。

## 适合谁看

只要你用过 ChatGPT、还记得一点点概率、看到伪代码不会立刻关页面，就能看下去。每一节都附了一段小提示，把那一节会用到的前置概念（Markov 链、注意力掩码、KV 缓存等等）顺一遍，所以不需要带太多背景。对数学符号有一点耐心会更舒服。

## 本地跑起来

```bash
cd site
python3 -m http.server 8000
# 然后打开 http://localhost:8000
```

没有打包器，没有 `npm install`，没有构建步骤。一份静态 HTML、CSS、原生 JS，D3 从 CDN 直接拉。

## 目录结构

```
site/
├── index.html        页面本体
├── styles.css        设计系统，含深色 / 浅色 / 移动端
├── i18n.js           页面上所有可见文字，英 + 简体中文
├── app.js            语言切换、滚动联动目录、共用工具
└── viz/              每个可视化一个独立脚本（hero.js、sec1-…、sec2-…）
```

如果想看比可视化更细的内容，仓库根目录下的几个 markdown（`diffusion_llms_mechanistic_deepdive.md`、`diffusion_report.agent.final.md`、`research/`）就是页面背后的长文研究稿。

## 引用来源

我尽量把页面里每一个具体数字都对回到论文或对应代码上。完整清单在网页底部的 **Sources** 一节，下面这几篇是最常用到的：

- [MD4](https://arxiv.org/abs/2406.04329) —— Shi 等，NeurIPS 2024（[代码](https://github.com/google-deepmind/md4)）
- [LLaDA](https://arxiv.org/abs/2502.09992) —— Nie 等，2025 年 2 月（[代码](https://github.com/ML-GSAI/LLaDA)）
- [LLaDA 2.0](https://arxiv.org/abs/2512.15745) —— 蚂蚁集团，2025 年 12 月（[代码](https://github.com/inclusionAI/LLaDA2.X)）
- [BD3-LM（块扩散）](https://arxiv.org/abs/2503.09573) —— Arriola 等，ICLR 2025
- [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/)
- [LMSYS SGLang Day-0 LLaDA 2.0](https://www.lmsys.org/blog/2025-12-19-diffusion-llm/)

## 许可

MIT —— 见 [LICENSE](LICENSE)。

---

本项目作为作品集 / 讲解小站做着玩。欢迎在 issue 或 PR 里指出错误，或者只是吐槽哪一段没讲清楚。
