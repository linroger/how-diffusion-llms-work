# Diffusion LLMs for Text and Code — A Mechanistic, Layer-by-Layer Deep-Dive

**Scope.** This document explains how modern diffusion language models work at a mechanistic level, layer by layer: the math, the architecture, the training procedure, the inference loop, and the specific design choices that distinguish Google DeepMind's stack from Ant Group / inclusionAI's stack. It complements — rather than restates — the existing landscape report in this directory (`diffusion_report.agent.final.md`), which already covers benchmarks, commercial dynamics, and RL post-training.

**Reading order.** Each section assumes the previous. Skim §1–§3 if you already know D3PM / MDLM math; the interesting engineering content lives in §5–§11.

---

## 1. From DDPM to text: why continuous diffusion fails on tokens

DDPM ([Ho et al. 2020](https://arxiv.org/abs/2006.11239)) defines a fixed Gaussian forward Markov chain on a continuous variable $x_0 \in \mathbb{R}^d$:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\bigl(x_t;\sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I\bigr),\qquad q(x_t\mid x_0) = \mathcal{N}\bigl(x_t;\sqrt{\bar\alpha_t}\, x_0,\, (1-\bar\alpha_t) I\bigr),$$

with $\bar\alpha_t = \prod_{s\le t}(1-\beta_s)$. The reverse process $p_\theta(x_{t-1}\mid x_t) = \mathcal{N}(\mu_\theta, \Sigma_\theta)$ is trained via the simple denoising loss

$$L_{\text{simple}} = \mathbb{E}_{t,x_0,\varepsilon}\bigl[\,\lVert \varepsilon - \varepsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\varepsilon,\, t)\rVert^2\,\bigr].$$

This is mathematically elegant and empirically excellent for images. It also **does not directly work for text**. Tokens live in a discrete vocabulary $\mathcal{V}$ of size $K \in \{32\text{k}, …, 256\text{k}\}$. "Token + Gaussian noise" is not a token, and the local Euclidean geometry that makes Gaussian smoothing meaningful in pixel space has no analog in vocab space.

Two responses emerged:

1. **Embedding-space diffusion** ([Diffusion-LM, Li et al. 2022](https://arxiv.org/abs/2205.14217), and follow-ups like SSD-LM, CDCD). Embed tokens into $\mathbb{R}^d$, run Gaussian diffusion on the embeddings, round back to tokens at inference. Works, but rounding is lossy, the embedding manifold is a thin shell in $\mathbb{R}^d$, and most model capacity ends up learning the rounding boundary rather than language. Quality has consistently lagged autoregressive (AR) baselines at matched scale. It is now a historical branch.

2. **Token-space (discrete) diffusion.** Replace the Gaussian kernel with a categorical Markov chain on $\mathcal{V}$. This is the line that produced D3PM, SEDD, MDLM/MD4, and the current wave of diffusion LLMs (LLaDA, Gemini Diffusion, Mercury, Seed Diffusion). Everything below builds on it.

---

## 2. Discrete diffusion as a categorical Markov chain (D3PM)

[D3PM (Austin et al. 2021)](https://arxiv.org/abs/2107.03006) defines forward noise via right-stochastic transition matrices $Q_t \in \mathbb{R}^{K\times K}$. With $x_0$ as a one-hot row vector,

$$q(x_t \mid x_{t-1}) = \mathrm{Cat}(x_t;\, x_{t-1} Q_t), \qquad q(x_t \mid x_0) = \mathrm{Cat}(x_t;\, x_0\, \bar Q_t),\ \ \bar Q_t = Q_1 Q_2 \cdots Q_t.$$

The reverse posterior $q(x_{t-1}\mid x_t, x_0)$ is computable in closed form by Bayes; the model parameterizes $p_\theta(\tilde x_0 \mid x_t)$, plugs it in, and is trained with a categorical ELBO.

Three concrete choices of $Q_t$ matter:

- **Uniform.** $Q_t = (1-\beta_t)I + \beta_t \mathbf{1}\mathbf{1}^\top / K$. Each step mixes a little uniform noise.
- **Absorbing-state (MASK).** Add a $(K{+}1)$-th absorbing state $[\text{MASK}]$. Each token is left unchanged w.p. $1-\beta_t$ or sent to $[\text{MASK}]$ w.p. $\beta_t$; once masked, always masked.
- **Gaussian-over-tokens / nearest-neighbor.** Local-in-vocab-order; mostly used for ordinal data, not language.

The **absorbing variant is the one that matters for text.** Its forward process is exactly BERT-style masking; its reverse is "predict the masked tokens given the partial context." MDLM and MD4 (next section) are clean refactorings of this branch.

[SEDD (Score Entropy Discrete Diffusion, Lou/Meng/Ermon 2024)](https://arxiv.org/abs/2310.16834) is the bridge from D3PM to scalable training. SEDD replaces the categorical KL loss with a **score-entropy** loss that targets the discrete analog of $\nabla \log p_t$ — i.e., the ratios $p_t(y)/p_t(x_t)$ for neighbor states $y \neq x_t$. The loss is non-negative, convex in the predicted ratio, and avoids the numerical pathologies of training a model whose output is an unbounded ratio. SEDD generalizes D3PM (the absorbing-state instance reduces to a masked predictor) and was the first discrete-diffusion method competitive with GPT-2 on perplexity at matched scale. It is the conceptual ancestor of MDLM and MD4.

---

## 3. The simplification that made diffusion LLMs viable: MDLM / MD4

[MD4 — "Simplified and Generalized Masked Diffusion for Discrete Data"](https://arxiv.org/abs/2406.04329) (Shi, Han, Wang, Doucet, Titsias, NeurIPS 2024), with parallel work [MDLM](https://arxiv.org/abs/2406.07524), proved that the entire continuous-time ELBO for absorbing-state discrete diffusion collapses to a single weighted cross-entropy integral.

**Continuous-time forward.** Replace the integer index $t \in \{1,\dots,T\}$ with $t \in [0,1]$ and a monotone schedule $\alpha_t: [0,1]\to[0,1]$ with $\alpha_0 = 1$ (no masking), $\alpha_1 = 0$ (all masked). Per-token, independently:

$$q(x_t^{(i)} \mid x_0^{(i)}) = \alpha_t\, \delta_{x_0^{(i)}} + (1-\alpha_t)\, \delta_{[\text{MASK}]}.$$

So $x_t$ is just $x_0$ with each token independently replaced by $[\text{MASK}]$ with probability $1-\alpha_t$. This is D3PM-absorbing in continuous time.

**Mean parameterization.** The model $x_\theta(x_t, t)$ predicts a categorical distribution over clean tokens at every masked position. Unmasked positions are passed through; predictions there are ignored because the true posterior is a delta.

**The collapse.** After bookkeeping which transitions are possible (only $\text{token}\to[\text{MASK}]$ forward, only $[\text{MASK}]\to\text{token}$ reverse), the continuous-time ELBO simplifies to

$$L_{\text{MD4}} = \mathbb{E}_{x_0}\!\!\int_0^1 \!\! \frac{-\alpha_t'}{1-\alpha_t}\, \mathbb{E}_{x_t \mid x_0}\!\!\!\sum_{i: x_t^{(i)} = [\text{MASK}]}\!\! -\log x_\theta(x_t, t)^{(i)}_{x_0^{(i)}}\, dt,$$

i.e. $L = \int_0^1 w(t)\cdot\mathrm{CE}(x_\theta(x_t,t), x_0)\,dt$ summed only over masked positions, with weight $w(t) = -\alpha_t'/(1-\alpha_t)$. **No KL ladder, no auxiliary terms, no learned variance — just weighted MLM cross-entropy integrated over a continuous masking rate.** That is the full training objective.

The DeepMind reference implementation [`google-deepmind/md4`](https://github.com/google-deepmind/md4) makes the schedule choices explicit. From `md4/models/diffusion/md4.py::_alpha`:

```python
if self.schedule_fn_type == 'linear':
    return 1.0 - t
elif 'poly' in self.schedule_fn_type:
    exponent = float(self.schedule_fn_type.replace('poly', ''))
    return 1.0 - t**exponent
elif self.schedule_fn_type == 'cosine':
    return 1.0 - jax.lax.cos(math.pi / 2.0 * (1.0 - t))
```

The default is linear ($\alpha_t = 1 - t$), with $\epsilon$-clipping to avoid degenerate endpoints. Crucially the schedule is **parametric, not learned** in vanilla MD4; the GenMD4 variant (§11.2) makes it state-dependent and learnable.

**Why this matters.** This is literally "BERT with a continuous masking rate and an ELBO-derived weighting." Three differences from BERT:

1. Mask rate $1-\alpha_t$ is sampled from a continuous schedule covering $[0,1]$, not fixed at 15%.
2. Loss is computed *only* on masked positions, with weight $w(t)$ that balances the gradient contribution of heavy- and light-masking regimes per the ELBO.
3. The trained model is a valid generative model with a tractable likelihood upper bound and an iterative sampler — not just a representation learner.

LLaDA's training loss ([Eq. 3 of Nie et al. 2025](https://arxiv.org/abs/2502.09992)) is exactly the linear-schedule special case:

$$L_{\text{LLaDA}}(\theta) = -\mathbb{E}_{t,x_0,x_t}\!\left[\frac{1}{t}\,\sum_{i:\, x_t^{(i)}=[\text{M}]} \log p_\theta(x_0^{(i)} \mid x_t)\right],$$

where $t$ is the mask rate (so $\alpha_t = 1-t$ gives $w(t) = 1/t$ after the bookkeeping). Once you see MD4, you see that LLaDA's "1/t weighting" is not a heuristic — it is the linear-schedule instance of an ELBO.

**Scaling.** [SMDM (Nie et al., arxiv 2410.18514)](https://arxiv.org/abs/2410.18514) — the immediate predecessor of LLaDA — fit Chinchilla-style IsoFLOP curves to masked diffusion up to 1.1B parameters and showed that **MDMs scale at the same rate as ARMs, on a slightly worse intercept**. The gap shrinks with scale, which is the empirical license to attempt 8B and then 100B. SMDM also documented unsupervised classifier-free guidance (no paired data needed because conditioning is just "more unmasked context") and the first "reverse curse" results (a 1.1B MDM beats 13B LLaMA-2 / 175B GPT-3 on reverse poem completion).

*Note on source identification.* The DeepMind publications-page URL `https://deepmind.google/research/publications/93097/` and `https://arxiv.org/abs/2410.05364` both resolve to [D-MPC ("Diffusion Model Predictive Control", Zhou et al. 2024)](https://arxiv.org/abs/2410.05364) — a control-theory paper using continuous diffusion for action proposals + dynamics, *not* discrete-text diffusion. The most likely intended papers for those slots are MD4 (above) and SMDM (cited above); the discussion in this document follows that substitution.

**Status of the "true diffusion?" debate.** Critics call MDLM "BERT with extra steps." Defenders point out: (a) the absorbing-state CTMC *is* a diffusion in the Feller/Markov-process sense, (b) the loss is an honest ELBO not a heuristic, and (c) the continuous-time formulation gives genuine inference-time freedoms (variable step count, any-order generation, self-correction) that fixed-rate MLM cannot. The pragmatic stance: the math is sound, the label is marketing.

---

## 4. The transformer changes when you train it as a diffusion model

A diffusion LLM is a standard transformer with three small but load-bearing modifications relative to a Llama-class AR model. Reading the LLaDA source ([`ML-GSAI/LLaDA`](https://github.com/ML-GSAI/LLaDA)) makes them concrete; LLaDA-8B is a Llama-3-class decoder with these deltas:

| Component | AR Llama-3 | LLaDA-8B (diffusion) |
|---|---|---|
| Layers / hidden / heads | 32 / 4096 / 32 | 32 / 4096 / 32 (same) |
| FFN | SwiGLU 14336 | SwiGLU 14336 (same) |
| Norm | RMSNorm | RMSNorm (same) |
| Position | RoPE (base 500K) | RoPE (base 500K, same) |
| **Attention mask** | **Causal** | **Fully bidirectional** |
| **Loss** | Next-token CE on every position | $\frac{1}{t}$ · CE on **masked** positions only |
| **GQA** | Yes | **No (32 query heads, no grouping)** |
| Time/step conditioning | n/a | Sinusoidal embedding of $t$ injected via AdaLN-style FFN (config dependent) |

The single biggest architectural change is dropping the causal mask — and that change has consequences that propagate through everything else. With full bidirectional attention, every position can attend to every other, which is exactly what the training objective needs (the model must predict $x_0^{(i)}$ from a context in which any subset of positions is masked, including positions to its right). But it also means **the KV cache that makes AR fast no longer works** — when any position can be re-attended to from any other, you cannot freeze K/V for "earlier" tokens, because there is no "earlier." This is the structural reason block diffusion (§7) exists.

Most code-relevant: the time/step embedding $t$ is the *only* mechanism by which the model knows what mask rate it's operating at. In `md4/backward.py`, the `CondEmbedding` module produces a sinusoidal embedding of $t$ that's broadcast into AdaLN-style scale/shift parameters per layer (config flag `mlp_type='swiglu'`). LLaDA's open-source implementation passes a similar signal, though the public release does not emphasize it; in practice the model can be trained without an explicit $t$ input (the mask pattern in the input is already a strong cue), at the cost of slightly looser ELBO weighting.

---

## 5. Training the model

### 5.1 The forward (corruption) process at training time

Per batch element:
1. Sample $t \sim \mathcal{U}(0, 1)$.
2. For each position $i$, independently set $x_t^{(i)} \leftarrow [\text{MASK}]$ with probability $1 - \alpha_t$ (linear: $1-\alpha_t = t$), else $x_t^{(i)} \leftarrow x_0^{(i)}$.
3. Run the model on $x_t$, get logits at every position.
4. Compute cross-entropy on masked positions only, weighted by $w(t)$.

The result is essentially BERT MLM with a randomized mask rate covering the full $[0, 1]$ interval — but the gradient *per masked position* is the same as MLM's. This is why retrofitting a pretrained AR model to a diffusion model is much cheaper than training from scratch (see §5.4).

### 5.2 Noise schedule choices

The schedule $\alpha_t$ enters only through (a) what mask rates the loss emphasizes, via $w(t) = -\alpha_t'/(1-\alpha_t)$, and (b) the corresponding reverse-process step sizes at inference. Three common choices:

| Schedule | $\alpha_t$ | Effective weighting |
|---|---|---|
| Linear | $1 - t$ | $w(t) = 1/t$ — heaviest weight on light masking |
| Polynomial-$k$ | $1 - t^k$ | adjustable; $k>1$ shifts weight toward heavy masking |
| Cosine | $1 - \cos(\pi t / 2)$ | smoother near endpoints |

MD4 ships with all three. LLaDA uses linear. The schedule does not change the model architecture — only the loss weight per sampled $t$ and, by symmetry, the inference step distribution.

### 5.3 LLaDA training recipe (reference numbers)

From [LLaDA paper](https://arxiv.org/abs/2502.09992) and the [repo](https://github.com/ML-GSAI/LLaDA):

- **Pretraining:** 2.3T tokens, 0.13M H800 GPU-hours, 8B dense transformer. Crash at 1.2T tokens recovered by dropping LR 4e-4 → 1e-4.
- **SFT:** 4.5M instruction pairs, standard supervised fine-tuning with the same diffusion loss.
- **Alignment (LLaDA 1.5):** VRPO (Variance-Reduced Preference Optimization) — see §5.5.
- **Sampling at training time:** $t \sim \mathcal{U}(0, 1)$ with no curriculum; the model sees the full range of mask rates from epoch 1.

### 5.4 AR-to-diffusion conversion: AR2Diff and WSD

If you already have an expensive AR pretraining checkpoint, you do not have to throw it away. [AR2Diff (Han et al., arxiv 2401.17181)](https://arxiv.org/abs/2401.17181) showed that a three-stage recipe converts an AR decoder into a diffusion decoder:

1. AR-pretrain a decoder normally (causal mask, next-token CE).
2. Continue pretraining with **bidirectional attention** and the masked-diffusion loss, for a small fraction (10K–100K steps) of the original budget. AR2Diff specifically uses a SUNDAE-style multi-step unrolled loss in this phase so the model learns to refine its own previous-step outputs.
3. Fine-tune on the downstream task in diffusion mode.

Best architectural variant: decoder-only with a prefix-LM objective. AR2Diff shipped at 280M / 700M / 1.7B scales (the "270M" in some derivative reports appears to be a typo for ~700M).

LLaDA2.0 generalizes this conversion to 100B with a much more elaborate curriculum. From [arxiv 2512.15745v2](https://arxiv.org/abs/2512.15745), the **Warmup-Stable-Decay (WSD)** schedule operates on a *block size* parameter $L_B$ that controls the granularity of bidirectional attention:

- **Warmup.** $L_B = 1 \to 4 \to 32 \to 64 \to 4096$. At $L_B = 1$, the model is identical to AR (bidirectional attention over a single token is causal). As $L_B$ grows, blocks of size $L_B$ become internally bidirectional with causal attention between blocks. By $L_B = 4096$, the model is doing full-sequence MDLM.
- **Stable.** $L_B = 4096$ held constant. Full-sequence diffusion training.
- **Decay.** $L_B = 4096 \to 2048 \to \ldots \to 32$. Compress back down to the operational inference block size.

This curriculum smoothly anneals the model from AR to full-diffusion and back to the production block size, preserving the AR-pretrained knowledge throughout. Two stability hacks make it work:

1. **Gaussian noise injection on masked-token embeddings.** In AR training, the embedding of `[MASK]` (or any token never observed) decays to zero because it never receives gradient. When switching to diffusion training, the masked-token embeddings are near-zero, so the model's predictions at high mask rates collapse and gradients explode. LLaDA2.0 solves this by adding *independent Gaussian noise to the embedding-layer output for masked tokens* during the initial iterations — keeping the L2 norm of the masked-token embedding meaningful without re-initializing weights (which would cause catastrophic forgetting).

2. **Document-level attention mask.** Packed training concatenates multiple documents into the same sequence. The diffusion model must not attend across document boundaries. LLaDA2.0's mask is block-diagonal within the noisy sequence, block-causal in the clean sequence, and zero across documents. The exact formula from the paper (for a concatenated $[x_t; x_0]$ of length $2L$ packed at block size $L_B$, with block index $b(k) = \lfloor k/L_B \rfloor$):

   - $M_{ij} = 1$ iff $b(i) = b(j)$ when $i, j \in x_t$ (block-diagonal within noisy half)
   - $M_{ij} = 1$ iff $b(i) > b(j-L)$ when $i \in x_t, j \in x_0$ (offset block-causal noisy→clean)
   - $M_{ij} = 1$ iff $b(i-L) \ge b(j-L)$ when $i, j \in x_0$ (causal within clean half)
   - $M_{ij} = 0$ everywhere else (clean cannot attend to noisy)

   This mask is the operational definition of "block diffusion training" and is reused at inference unchanged.

### 5.5 RL / preference optimization

Standard DPO assumes a tractable sequence log-likelihood. Diffusion LMs do not have one — the marginal $\log p_\theta(x_0)$ requires integrating over all possible mask patterns. The ELBO is tractable but high-variance. Two responses:

- **VRPO ([LLaDA 1.5, arxiv 2505.19223](https://arxiv.org/abs/2505.19223)).** Variance-Reduced Preference Optimization. Formal analysis of ELBO-estimator variance in preference optimization gradients; introduces optimal Monte-Carlo budget allocation and antithetic sampling. Applied to LLaDA-8B-Instruct yields LLaDA 1.5 with +4.7 GSM8K, +3.0 HumanEval, +1.8 MBPP, +4.0 IFEval, +4.3 Arena-Hard over the SFT baseline at <0.5% of pretraining cost.

- **EBPO (LLaDA2.0/2.1).** ELBO-based block-level Policy Optimization. Uses block-conditional ELBO as a tractable proxy for sequence log-likelihood in a PPO-style clipped surrogate. Block-level rather than token-level (matches the inference granularity). First RL framework applied to a 100B-param diffusion model.

- **DPO with ELBO substitution (LLaDA2.0).** The advantage $\Delta_B(x|c) = B_{\text{BDLM}}(\theta, x|c) - B_{\text{BDLM}}(\theta_{\text{ref}}, x|c)$ uses block-diffusion ELBOs instead of log-likelihoods. Novel; weakly ablated in the paper; worth scrutiny.

---

## 6. Inference layer 1 — the basic denoising loop

Start with $x_T = [\text{MASK}]^L$. Choose $N$ denoising steps (typically $N \ll L$). At each step $k$:

```
logits   = f_theta(x_t_k, t_k)              # one forward pass over the full sequence
probs    = softmax(logits / temperature)    # categorical distribution per position
S_k      = choose_positions(probs, x_t_k, t_k)   # which masked slots to unmask
x_{k-1}  = commit(x_t_k, probs, S_k)        # write tokens at S_k; rest stay masked
```

Four position-selection policies dominate practice:

- **Random remasking.** Pick $k$ masked positions uniformly. Matches the training objective under uniform schedule. Quality is poor at fixed step budgets.
- **Low-confidence remasking (the workhorse).** Compute top-1 confidence at each masked position; unmask the top-$k$ by confidence; the rest roll forward. This is what LLaDA-8B-Instruct ships with — dominates random remasking on GSM8K (78.6 vs 69.4) and MATH (42.2 vs 31.9) at equal step budget.
- **Temperature / nucleus sampling at unmasked positions.** Decouples position selection (argmax-on-confidence) from token selection (sample-with-temperature). Both knobs are independent.
- **Confidence-threshold commitment.** Unmask every position whose top-1 probability exceeds $\tau$. Converts step count from a hyperparameter into an emergent quantity; the basis of Fast-dLLM and CAP.

The MD4 reference sampler does plain ancestral / top-$p$ with **no confidence remasking** (it commits per the marginal-preserving reverse process: each currently-masked position independently unmasks with probability $(\alpha_{s} - \alpha_{t})/(1-\alpha_t)$ between adjacent times $t > s$). LLaDA, Gemini Diffusion, and LLaDA2 all add confidence remasking on top — it is empirically a free win.

---

## 7. Inference layer 2 — block diffusion

Full-sequence bidirectional attention over $L = 4096$ tokens with $N$ denoising steps costs $N$ full $O(L^2)$ attention passes with *no reusable KV cache* (since any unmasked neighbor changing invalidates K/V for everyone). This is the structural bottleneck that made vanilla MDLM uncompetitive at scale.

**Block diffusion** ([BD3-LM, Arriola et al., ICLR 2025](https://arxiv.org/abs/2503.09573)) is the fix. Partition the sequence into contiguous blocks of size $B$ (typically 16–64). Attention is:
- **Causal between blocks** (block $k$ attends only to blocks $1, \dots, k$).
- **Bidirectional within a block** (all positions in block $k$ attend to each other).

Once block $k$ is fully decoded, its K/V is frozen and cached — exactly as in AR. The model emits blocks left-to-right; within a block it runs $T_{\text{block}}$ denoising steps with intra-block bidirectional attention. Variable-length generation is preserved because the model can emit an EOS block at any time.

**This is the structural backbone of Gemini Diffusion, LLaDA2.0/2.1, Mercury, and Seed Diffusion.** Block size 32 has emerged as a near-universal sweet spot empirically.

Fast-dLLM ([nvlabs.github.io/Fast-dLLM](https://nvlabs.github.io/Fast-dLLM/)) extends this with a two-level cache: block-level cache for committed blocks plus a sub-block "DualCache" that even reuses K/V *across denoising steps within an active block*, exploiting the fact that committed positions inside the active block don't move once they're committed.

---

## 8. Inference layer 3 — parallel decoding (Confidence-Aware Parallel)

Block diffusion alone still costs $T_{\text{block}}$ forward passes per block. The decisive speedup comes from **committing many positions per pass**. The dominant policy is *Confidence-Aware Parallel (CAP)*: in a single forward pass, commit every masked position in the active block whose top-1 confidence exceeds threshold $\tau$. Positions below $\tau$ stay masked, re-scored against the freshly committed context next step.

CAP works only if the model's confidence is well-calibrated and *sharpened* — uncalibrated softmax outputs lead to over-commitment and quality collapse. LLaDA2.0 trains for this with an **auxiliary confidence loss**:

$$L(\theta) = L_{\text{SFT}}(\theta) + \lambda \cdot L_{\text{conf}}(\theta),$$

where $L_{\text{conf}}$ selectively minimizes the entropy of $p_\theta(x_0 \mid x_t, c)$ **only on tokens that are correctly predicted** in the current step. The "only on correctly predicted" qualifier is critical — minimizing entropy unconditionally would push the model toward confident wrong answers; minimizing it on the correct-and-confident subset selectively sharpens the calibration where it's safe.

**Measured speedups (controlled, [LMSYS Day-0 SGLang post, Dec 2025](https://www.lmsys.org/blog/2025-12-19-diffusion-llm/)).** On 8×H20 (TP8):
- LLaDA2.0-flash-CAP: 535 TPS sustained (up to 935 in animated demos)
- LLaDA2.0-flash (no CAP): 383 TPS
- Ling-flash-2.0 (matched AR baseline): 256 TPS
- Qwen3-30B-A3B: 237 TPS

→ **~2.1× speedup over matched AR** under the same serving stack, on the same hardware, at small batch. The "10×" headline numbers in marketing material reflect unfavorable AR baselines (no continuous batching, smaller GPUs) or cherry-picked workloads. The honest small-batch number is ~2×.

Variance- or entropy-gated variants commit on margin (top-1 minus top-2) rather than absolute confidence, which is more robust under miscalibration. LLaDA2.0's ablation suggests $\tau = 0.95$ at block size 32 is the operational optimum; pushing $\tau$ down to 0.85 raises TPF from 2.55 to 3.31 but drops the score from 70.15 to 67.90 — an unacceptable quality cliff per the authors' own framing.

---

## 9. Inference layer 4 — self-correction (T2T editing, MBE, self-speculative)

CAP commits early and irreversibly. That's the structural weakness: an early-step argmax can be wrong, and the standard loop has no mechanism to revisit it. Three families of mechanisms address this.

### 9.1 Token-to-Token (T2T) editing — LLaDA2.1's contribution

LLaDA2.1 ([repo: `inclusionAI/LLaDA2.X`](https://github.com/inclusionAI/LLaDA2.X)) augments the standard Mask-to-Token (M2T) update with a **T2T** path. At step $t$, with current state $x_t$ and the model's top-1 candidate $v_t$ at every position:

- **Unmasking set** $\Gamma_t = \{i : x_t^{(i)} = [\text{M}]\ \text{AND}\ p_\theta(v_t^{(i)} | x_t) > \tau_{\text{mask}}\}$
- **Editing set** $\Delta_t = \{i : x_t^{(i)} \ne v_t^{(i)}\ \text{AND}\ x_t^{(i)} \ne [\text{M}]\ \text{AND}\ p_\theta(v_t^{(i)} | x_t) > \tau_{\text{edit}}\}$
- **Update:** $x_{t-1}^{(i)} = v_t^{(i)}$ if $i \in \Gamma_t \cup \Delta_t$, else $x_t^{(i)}$.

The inner loop terminates when both no masks remain AND no T2T edits trigger; then it advances to the next block. The architectural change is small — the model already outputs a categorical distribution at every position (masked or not), so the sampler simply starts trusting non-masked predictions when they disagree with the committed token *and* exceed a threshold. The training-side change is large — the SFT loss must teach the model to *propose replacements*, not just fill masks. LLaDA2.1's "unified Mixture of M2T and T2T training objective" mixes M2T (random masking) with T2T (random uniform-perturbation of tokens) examples throughout CPT and SFT.

Two operating modes:
- **Speedy (S-Mode):** $\tau_{\text{m2t}} = 0.5$, $\tau_{\text{t2t}} = 0.0$. Aggressive parallel commit; T2T cleans up.
- **Quality (Q-Mode):** $\tau_{\text{m2t}} = 0.7$, $\tau_{\text{t2t}} = 0.5$, max_post_steps ≥ 5–16.

S-Mode TPF on LLaDA2.1-flash is 5.93 vs 3.08 on LLaDA2.0 — the 2.1× speed gain over LLaDA2.0 comes from S-Mode being safe to use because T2T can repair its over-commits. Reported throughputs: 892 TPS on HumanEval+, 801 TPS on BigCodeBench, 663 TPS on LiveCodeBench.

**Known failure modes** (acknowledged in the LLaDA2.1 paper):
1. *Correction inertia.* Multimodal posterior; no single alternative crosses $\tau_{\text{edit}}$.
2. *Premature replacement.* Locally correct token gets replaced under polluted right-context.
3. *Positional lock-in.* T2T cannot expand a position's span — it can only swap one token for another at the same position.

A T2M follow-up (Lin Yao et al.) addresses *correction inertia* by allowing T2T to *remask* suspicious tokens (revert to `[MASK]`) instead of overwriting, then letting M2T re-fill. Reports +13.33 on AIME 2025, +8.56 CMATH, repairs 41.3% of "last-mile corruption" on CMATH.

### 9.2 Multi-Block Editing (MBE)

T2T as described is intra-block. **MBE** extends the scope: after block $k+1$ is decoded, re-evaluate (and via T2T, rewrite) positions in blocks $1, \dots, k$ whose newly-available right-context changes their top-1 prediction. Implementation is non-trivial — the causal-between-blocks attention pattern means revisiting an earlier block requires either (a) breaking the K/V cache for the touched block and re-running attention with that block in editable/remasked state, or (b) augmenting the cache with editability metadata. LLaDA2.1 reports MBE consistently improves benchmarks on reasoning and coding tasks where late-arriving evidence invalidates an early commitment.

### 9.3 Self-speculative decoding (SSD)

Because an MDLM already produces a joint over all remaining masked positions on every pass, the model is its own draft model — no auxiliary network. [SSD (Hong et al. 2025, arxiv 2510.04147)](https://arxiv.org/abs/2510.04147):

1. **Self-draft.** One forward pass produces draft tokens + confidence scores for every remaining masked position.
2. **Hierarchical verification tree.** A greedy linear-chain tree of length $N+1$ is built; a child node is accepted only when its parent token matches the step-wise generation result.
3. **Batched verify.** All verification nodes are checked in one further pass; walk the tree, accept the longest matching prefix.

Provably lossless (output is identical to plain stepwise decoding). Reports up to **3.46×** wall-clock speedup on Dream-Instruct and 50–70% step reduction across GSM8K / MATH / HumanEval / MBPP.

---

## 10. Inference layer 5 — variable-length generation

MDLMs natively operate on a fixed canvas $L$. Three solutions in production:

- **EOS-as-token.** Reserve `[EOS]`; truncate at the first committed EOS. LLaDA, Dream, Mercury.
- **Block-EOS.** Block-diffusion models can emit a whole block as "EOS block" and stop. BD3-LM.
- **Dynamic-length canvas (DreamOn).** [DreamOn (HKUNLP 2026)](https://hkunlp.github.io/blog/2025/dreamon/) adds two learned special states, `<|expand|>` and `<|delete|>`. In the reverse process, `<|expand|>` deterministically becomes two new `[MASK]` tokens at that position; `<|delete|>` is removed. The model grows or shrinks its own canvas in response to confidence, no architectural change beyond two extra vocabulary entries. Reports 26.4% improvement on variable-length tasks.

---

## 11. The two flagship stacks side-by-side

### 11.1 Google DeepMind — MD4 → GenMD4 → AR2Diff → Gemini Diffusion

| Layer | Artifact | What it provides |
|---|---|---|
| Math | [MD4](https://arxiv.org/abs/2406.04329) | Continuous-time ELBO collapses to weighted CE integral; reference JAX/Flax impl |
| Schedule | GenMD4 (same paper) | Learnable state-dependent masking schedule via RLOO REINFORCE |
| Transfer recipe | [AR2Diff](https://arxiv.org/abs/2401.17181) | 3-stage AR→diffusion conversion with SUNDAE-style multi-step loss |
| Productization | [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/) | Block-diffusion serving system; "experimental demo" since May 2025 |

**GenMD4 mechanism.** The schedule is no longer a scalar $\alpha_t$ but a per-vocab-token learnable power. From `md4/models/diffusion/genmd4.py`:

```python
self.w = self.param('w', utils.constant_init(w_init), [self.vocab_size])
self.power = nn.softplus(self.w)
```

Per-token mask rates → discrete sampling → non-differentiable schedule → use Rao-Blackwellized leave-one-out REINFORCE with two independent loss samples per timestep for low-variance gradient. Gains are small but consistent on text.

**Gemini Diffusion (publicly known mechanics).** Block-diffusion model with ~128-token blocks (per third-party reporting; not officially disclosed by DeepMind). Intra-block bidirectional + inter-block causal. ~16 denoising steps per block (third-party). Transformer backbone is described as "vanilla" — i.e., **not a DiT** — consistent with MD4's reference architecture. The "U-Net encoder-decoder with skip connections" claim that appears in some derivative reports is likely incorrect for the text backbone (sourced to a low-authority blog); a senior reader should ignore that claim.

**Reported benchmarks (Gemini Diffusion vs Gemini 2.0 Flash-Lite, from the official model page):**

| Benchmark | Gemini Diffusion | Flash-Lite |
|---|---|---|
| HumanEval | 89.6% | 90.2% |
| MBPP | 76.0% | 75.8% |
| LiveCodeBench v6 | 30.9% | 28.5% |
| BigCodeBench | 45.4% | 45.8% |
| AIME 2025 | 23.3% | 20.0% |
| GPQA Diamond | 40.4% | **56.5%** |
| BIG-Bench Hard | 15.0% | **21.0%** |
| Global MMLU (Lite) | 69.1% | **79.0%** |

**The pattern is the punchline.** Diffusion is within ~1 point of AR on code (HumanEval, MBPP, LiveCodeBench, BigCodeBench) and slightly ahead on short-form math (AIME). It collapses by 10–16 points on GPQA, BIG-Bench Hard, and Global MMLU. These are exactly the benchmarks dominated by multi-hop reasoning and long-chain knowledge retrieval. The mechanism is the **coordination problem**: when many positions in a block are committed in parallel, the model must commit to a *jointly consistent* hypothesis without conditioning each new token on the precise text of the previous one. AR sidesteps this — every new token is generated knowing the exact prefix. Block diffusion's inter-block causality partially recovers this, but intra-block parallel commitment is structurally weaker than per-token AR conditioning for long-chain reasoning. The DeepMind page does not acknowledge this trade-off; the numbers speak for themselves.

**Throughput.** 1,479 tok/s average, ~2,000 peak on code, 0.84s TTFT, "~5× faster than Flash-Lite" (the 5× figure is third-party framing, not on the DeepMind page).

**Stack opacity.** DeepMind does not officially state that Gemini Diffusion uses MD4. The architectural similarity (vanilla transformer, bidirectional intra-block, masked discrete diffusion, no DiT) plus shared authors (Kehang Han is on both MD4 and AR2Diff) make the connection obvious but inferential.

### 11.2 Ant Group / inclusionAI — LLaDA → LLaDA 1.5 → LLaDA-MoE → LLaDA2.0 → LLaDA2.1

| Generation | Artifact | What's new |
|---|---|---|
| LLaDA-8B-Base/Instruct (Feb 2025) | [arxiv 2502.09992](https://arxiv.org/abs/2502.09992), [repo](https://github.com/ML-GSAI/LLaDA) | First serious from-scratch 8B masked-diffusion LM; bidirectional Llama-3-class transformer; 2.3T tokens; SFT 4.5M pairs; semi-AR low-confidence remasking |
| LLaDA 1.5 (May 2025) | [arxiv 2505.19223](https://arxiv.org/abs/2505.19223) | VRPO preference optimization with variance-reduction; +4.7 GSM8K / +3.0 HumanEval / +4.3 Arena-Hard over LLaDA SFT, at <0.5% of pretraining cost |
| LLaDA-MoE-7B-A1B (Sep 2025) | [arxiv 2509.24389](https://arxiv.org/abs/2509.24389), [HF](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base) | First MoE diffusion LM. 7B total / 1.4B active; 64 experts, top-8 routing, 16 layers, 2048 hidden, 1024 expert dim; 20T training tokens (10T base + 10T math/code reweighted); competitive with Qwen2.5-3B-Instruct |
| LLaDA2.0-mini / flash (Dec 2025) | [arxiv 2512.15745v2](https://arxiv.org/abs/2512.15745), [repo](https://github.com/inclusionAI/LLaDA2.X) | First 100B-class diffusion LM. Flash: ~100B total, ~6.1B active MoE (256 routed + 1 shared expert, 8 activated/token). AR-to-diffusion conversion from Ling-flash-2.0 via WSD curriculum. CAP training. EBPO RL. Document-level attention mask. 535 TPS / ~2.1× over AR baseline |
| LLaDA2.1 | (paper pending public ID, repo: [LLaDA2.X](https://github.com/inclusionAI/LLaDA2.X)) | Adds T2T editing + Multi-Block Editing + S/Q dual modes. 892/801/663 TPS on HumanEval+/BigCodeBench/LiveCodeBench |

**Why this lineage matters.** Each generation isolates one variable:
- LLaDA proved a from-scratch 8B masked-diffusion LM is feasible and competitive with LLaMA-3-8B on standard benchmarks, and *breaks the reverse curse* on poem completion (forward 51.8% / reverse 45.6%, vs GPT-4o's 82.7% / 34.3%) — direct evidence that bidirectional MLE training removes the directional asymmetry that AR bakes in.
- LLaDA 1.5 proved that diffusion-aware preference optimization is tractable at a tiny fraction of pretraining cost.
- LLaDA-MoE proved that MoE works for diffusion LMs (no shared experts in this version; standard top-$k$ routing with auxiliary load-balancing).
- LLaDA2.0 proved that AR→diffusion conversion scales to 100B and that CAP + block diffusion + EBPO produces a production-grade serving system with measured ~2× speedup over matched AR at small batch.
- LLaDA2.1 proved that adding *editability* (T2T + MBE) lets you push throughput to ~2.1× over LLaDA2.0 without sacrificing benchmark quality — and that the model can fix its own mistakes mid-decode.

**Open-source stance.** Every checkpoint is MIT/Apache licensed and ships with the full inference stack (dInfer + customized SGLang). The closed→open gap that was real through mid-2025 (Mercury, Gemini Diffusion lead) is now measured in months, not generations.

---

## 12. Why code is the natural beachhead (and where diffusion still loses)

The pattern across DeepMind, Ant Group, ByteDance, Mercury, and Dream-Coder is consistent: **diffusion reaches parity, often surpasses AR, on structured-output tasks (code, math, tool calls); it lags slightly on broad knowledge QA.**

Mechanism:
- Code is locally constrained (function signatures, bracket matching, type schemas, multi-step tool-call structures) and benefits from bidirectional refinement — later tokens can condition global structure that AR has to commit greedily.
- The *editing* primitive of diffusion (any position can be revisited, then unmasked, then via T2T re-written) aligns naturally with how programmers actually write code — sketch, refine, fix. CanItEdit benchmark shows the largest single diffusion-over-AR delta: Stable-DiffCoder 60.0% vs Seed-Coder 50.5% (+18.8% relative).
- Long-context code (RepoQA) benefits from diffusion's superior length extrapolation: Mercury-Coder ~15% degradation at 64K vs AR ~30%.

Where it loses:
- **Multi-hop reasoning + knowledge retrieval** (GPQA, BIG-Bench Hard, Global MMLU). Block-parallel commitment can't condition each new token on the precise prefix; long chains of dependent inferences break. This is the structural ceiling.
- **Competitive programming** (LiveCodeBench): diffusion currently lags AR (Gemini Diffusion 30.9% vs Flash-Lite 28.5% is an exception, not the rule; most diffusion models are well below 25%). Sequential algorithmic reasoning is closer to multi-hop than to "fill in the bracket structure."

The [NAP paper (Feb 2026)](https://arxiv.org/abs/2602.xxxxx) frames this as the **AR-collapse hypothesis**: training data is overwhelmingly sequentially-ordered, so diffusion models learn AR-shaped behavior even with bidirectional architectures (Dream-7B's measured "ARness" with parallel decoding is ~0.92 per the report). The fix is presumably training-data restructuring or process-reward signals that explicitly reward non-sequential planning — open research.

---

## 13. The theoretical ceiling — and the convergence

[Feng et al.] proved that masked diffusion models require **linear-in-sequence-length steps** for low sequence-level error on reasoning tasks. This is the strongest theoretical result against diffusion's speed advantage in that task class — if you need $N = \Theta(L)$ steps to get low error on reasoning, you've eliminated the parallelism gain you started with. The practical response has been:

1. **Don't chase the linear bound for non-reasoning tasks.** Diffusion *is* fast on code and short-form generation where the conditional-independence assumption is approximately true.
2. **Plan-conditioning.** Use an external AR model to produce a plan, then use diffusion to fill it in. "Think First, Diffuse Fast" reports +11.6pp on GSM8K via this composition.
3. **Self-correction (T2T, MBE).** Reduce the per-step error rate by allowing revision, so fewer steps achieve the same end-state error.

**The convergence trend (late 2025).** The AR-vs-diffusion dichotomy is dissolving into a continuum parameterized by block size, attention causality, and unmasking-schedule confidence. LLaDA2.0's WSD curriculum literally moves the model from $L_B=1$ (pure AR) through $L_B=4096$ (pure diffusion) and back. LMSYS observes that block diffusion "bears a high degree of similarity to chunked-prefill" — the *serving primitives* are the same. Hybrid architectures (TiDAR, CALM, Projected Autoregression, A3, BD3-LM, SDAR) all sit on this continuum.

The pragmatic 12-month forecast: the question stops being "AR or diffusion" and becomes **"what block size and what unmasking threshold for this workload."** Code and structured generation → small block, aggressive confidence threshold, ~2× throughput win. Long-chain reasoning → block size 1 (effectively AR), no parallel commit, no speedup but no quality loss. The same model can do both, controlled by inference-time hyperparameters.

---

## 14. Honest gaps in this writeup

Where the public material is opaque or where I'm relying on indirect attribution:

1. **MD4 → Gemini Diffusion linkage is inferential, not officially confirmed by DeepMind.** Architectural similarity and shared authorship make it the obvious bet, but the DeepMind product page deliberately discloses only outermost numbers.
2. **Gemini Diffusion's exact step count, schedule, and sampler are not officially disclosed.** "~16 steps per block" and "~128-token blocks" come from third-party reporting.
3. **The "U-Net encoder-decoder" claim for Gemini Diffusion text backbone is likely wrong** — it appears in some derivative reports sourced to a low-authority blog. MD4's reference architecture is a plain bidirectional transformer with optional AdaLN time conditioning; no U-Net. Treat any "U-Net" claim for diffusion-LLM text backbones with skepticism.
4. **MD4 weight $w(t) = -\alpha_t'/(1-\alpha_t)$** — described in the paper as "SNR-related" but the explicit form above is the actual implementation in `md4` for linear $\alpha_t = 1 - t$, giving $w(t) = 1/t$.
5. **EBPO's exact clipped-surrogate equation** is sketched in the LLaDA2.0 paper but not reproduced here verbatim; for production implementation, read the paper directly.
6. **The DeepMind URL `/publications/93097/` and `arxiv 2410.05364`** both resolve to D-MPC (a control-theory paper), not a discrete-diffusion LM paper. I've substituted MD4 and SMDM as the likely intended references throughout.
7. **Numbers attributed to LLaDA2.1's repo** (S-Mode thresholds, MBE failure modes, T2M follow-up) should be cross-checked against the [`inclusionAI/LLaDA2.X`](https://github.com/inclusionAI/LLaDA2.X) source before implementation; some come from a third-party Moonlight review.
8. **"33× data-repetition robustness" claim** (DLM half-life ~500 epochs vs AR ~15) is single-source in the existing local research notes; verify against original paper before citing.

---

## 15. Sources, by section

**Foundations:**
- [DDPM, Ho et al. 2020](https://arxiv.org/abs/2006.11239)
- [Diffusion-LM, Li et al. 2022](https://arxiv.org/abs/2205.14217)
- [D3PM, Austin et al. 2021](https://arxiv.org/abs/2107.03006)
- [SEDD, Lou/Meng/Ermon 2024](https://arxiv.org/abs/2310.16834)
- [MD4, Shi et al. 2024](https://arxiv.org/abs/2406.04329), [code](https://github.com/google-deepmind/md4)
- [MDLM, Sahoo et al. 2024](https://arxiv.org/abs/2406.07524)
- [SMDM, Nie et al. 2024](https://arxiv.org/abs/2410.18514)
- [BERT, Devlin et al. 2018](https://arxiv.org/abs/1810.04805)

**Transformer adapters:**
- [The Illustrated Transformer (Alammar)](https://jalammar.github.io/illustrated-transformer/) — the canonical pre-diffusion mental model

**Google DeepMind:**
- [Gemini Diffusion model page](https://deepmind.google/models/gemini-diffusion/)
- [AR2Diff, Han et al. 2024](https://arxiv.org/abs/2401.17181)
- [MD4 paper / code (above)](https://github.com/google-deepmind/md4)

**Ant Group / inclusionAI:**
- [LLaDA paper, Nie et al. Feb 2025](https://arxiv.org/abs/2502.09992)
- [LLaDA repo / model card](https://github.com/ML-GSAI/LLaDA), [demo page](https://ml-gsai.github.io/LLaDA-demo/)
- [LLaDA 1.5 / VRPO, Zhu et al. May 2025](https://arxiv.org/abs/2505.19223)
- [LLaDA-MoE-7B-A1B paper, Sep 2025](https://arxiv.org/abs/2509.24389), [HF org](https://huggingface.co/inclusionAI)
- [LLaDA2.0 paper, Dec 2025](https://arxiv.org/abs/2512.15745), [repo](https://github.com/inclusionAI/LLaDA2.X)
- [SGLang Day-0 integration, LMSYS Dec 19 2025](https://www.lmsys.org/blog/2025-12-19-diffusion-llm/)

**Inference algorithms:**
- [BD3-LM / Block Diffusion, Arriola et al. ICLR 2025](https://arxiv.org/abs/2503.09573)
- [Fast-dLLM, NVIDIA/HKU/MIT](https://nvlabs.github.io/Fast-dLLM/)
- [Fast-dLLM v2](https://arxiv.org/html/2509.26328)
- [SSD, Hong et al. 2025](https://arxiv.org/abs/2510.04147)
- [DreamOn, HKUNLP 2026](https://hkunlp.github.io/blog/2025/dreamon/), [code](https://github.com/DreamLM/DreamOn)

**Local prior research:**
- `diffusion_report.agent.final.md` and the per-section files in this directory provide the commercial landscape, benchmark deep-dive, and RL post-training material that this document deliberately does not duplicate. See `diffusion_report.agent.outline.md` for the index.

---

*Document author: synthesis from six parallel research strands plus prior local research, May 2026.*
