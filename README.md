# cache-2-cache-lite

The mechanism comes from *Cache-to-Cache: Direct Semantic Communication Between
Large Language Models*, by Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan,
Guohao Dai, Wanli Ouyang, and Yu Wang, published at ICLR 2026
([arXiv:2510.03215](https://arxiv.org/abs/2510.03215)). The
reference implementation is released by Tsinghua's NICS group at
[thu-nics/C2C](https://github.com/thu-nics/C2C).

## The question

Two models reading the same text produce two different internal readings of it.
Cache-to-cache communication proposes that one model's reading can be handed
directly to the other, as tensors rather than as text, and that the receiving
model can use it.

The proposal has been demonstrated. What has not been demonstrated is how far it
reaches. The experiment that established a cache is convertible at all mapped
Qwen3-4B into Qwen3-0.6B: two models from one family, one generation, sharing
every architectural convention. The question here is whether the same mapping
survives when those conventions stop being shared.

## Why this pair

Qwen2.5-0.5B-Instruct produces the cache. Qwen3-0.6B consumes it. This is the
smallest pair the original work of cache-to-cache reports, and the two models are separated by a generation rather than by size.

Three differences follow from that separation, and each is a candidate reason
for the transfer to fail.

1. The receiver normalizes its keys per head before applying rotary embeddings.
The sharer has no such step, so its keys arrive with a magnitude distribution
the receiver's own keys never exhibit at any depth.

2. Keys carry position and values do not. Position is written into a key as a
rotation, in a basis fixed by the head dimension. If the two models size their
heads differently, a transferred key arrives encoded in a basis its reader does
not use, while a transferred value arrives with no such problem.

![Both rotary ladders on one axis. The Sharer's 32 inverse frequencies land
exactly on every second one of the Receiver's 64, so the coarse ladder is the
fine one sampled at stride two.](assets/c2c_rope_ladders.png)

*Measurement refines the difference rather than confirming it. The basis is not
foreign, it is half present: every frequency the Sharer uses is one the Receiver
also uses, and the Receiver has 32 more that the Sharer never writes. Rebuilt
from `results/contracts.json` and checked against the endpoints that file
recorded.*

3. The cache is stored at the key-value head count, not the attention head count.
The width the projection actually spans is that product, which for these two
models is expected to differ by close to an order of magnitude rather than the
fifteen percent their hidden sizes suggest.

![Two bands drawn to the same scale. The concatenation entering the projection
is 128 channels from the Sharer beside 1024 from the Receiver; the band below
shows the 896 against 1024 that hidden sizes would have suggested.](assets/c2c_channel_widths.png)

*The expectation held. The Sharer occupies 11.1 percent of the channels the
projection reads, not the 47 percent the hidden sizes imply. From
`results/contracts.json`.*

All three are cheap to check before anything is trained, which is what most of this repository does.

## What would count as an answer

Three outcomes are possible and all three are informative.

1. A learned map reaches the receiver's cache and the fused system reads better
than the receiver reading alone. The mechanism generalizes past the family it
was demonstrated in, and the architectural differences above are absorbable.

2. A learned map reaches the receiver's cache but the fused system reads no better
than the receiver alone. The geometry is bridgeable and the bridge is not
useful, which places the difficulty in how transferred information is combined
rather than in whether it can be carried.

3. A learned map does not reach the receiver's cache at all. The three differences
above become the candidate explanations, and each was already measured in
isolation before training began, so the search has somewhere to start.

## What it found

Of the three outcomes above, the first. A fused system built from scratch on
this pair scores above the Receiver reading alone, on 500 MMLU-Redux questions
scored by the same argmax under the same prompt.

| | Accuracy |
|---|---|
| Receiver alone | 0.382 |
| Fused, the Sharer's cache added under a gate | 0.458 |
| Replacement, the same module substituting instead of adding | 0.250 |
| The earlier projection, fitted on wikitext under mean squared error | 0.198 |

Against the Receiver alone the fused system gains 0.076, with a paired
bootstrap interval of [+0.038, +0.112] and McNemar at p = 7.66e-05.

Adding and substituting were then trained on the same corpus, under the same
loss, for the same steps, from the same weights at step zero, so the two differ
in the addition and nothing else. That single difference is worth 0.208.

Both substituting builds collapsed onto one letter, answering B on 497 of 500
and A on 498 of 500, and each scored the base rate of the letter it settled on.

![Answer counts over the four option letters, with the answer key drawn as a
hatched series beside the four conditions. Two conditions are a single tall bar;
the fused condition tracks the key across all four letters.](assets/c2c_answer_distributions.png)

*This is the figure that makes an accuracy readable. An accuracy near the base
rate of one letter and an accuracy earned across four are the same number, and
no interval or paired test tells them apart. Total variation distance from the
key: fused 0.118, Receiver alone 0.336, replacement 0.744, the earlier
projection 0.800. From `results/run_2026-08-07_n500.json`.*
The gate is what makes the difference: shut, it multiplies the Sharer's
contribution by exactly zero, so the fused system can always fall back to the
Receiver alone.

Cost is measured too. Fusion adds about eight percent to the Receiver's own
prefill and costs less than two tokens of Sharer decode at every length tried,
while the cache itself is 12,288 bytes per position against a few hundred bytes
for the text it replaces. Compute favours the exchange on this machine and
communication does not.

![Break-even link speed as a contour over prompt length and the length of the
response the Sharer would have written, with the four measured points
marked.](assets/c2c_ledger.png)

*The two sides of the exchange do not scale together: the payload grows with the
prompt and the decode saved grows with the response. At the four measured
points, all with a 16 token response, the link has to carry 8, 15, 31 and 61
megabits per second before the trade pays for itself. The arithmetic on top of
the measurements is written into the figure, as are the two terms left out and
the direction in which leaving them out moves the threshold. From
`results/ledger.json`.*

`FINDINGS.md` carries every figure with the file it was read from, the seven
predictions written before their runs, and the readings that measurement
overturned.

## What this is not

Not a benchmark run. The reference implementation exists, is public, and
publishes trained weights for this exact pairing. Those weights are deliberately
not used here. A mechanism reproduced from a checkpoint teaches nothing about
the mechanism.

Not a faster or better variant. Every architectural choice follows the original
unless a measurement here says otherwise, and where it does say otherwise, the
measurement is in `results/`.

Not settled. Every figure below rests on one training run, one seed, and one
sample of 500 questions. Nothing here has been repeated, and a result that has
not been repeated is a result that has not been tested for repeatability.

## Layout

```
c2c/          the mechanism, as pure functions
scripts/      everything that touches model weights or disk
tests/        the compliance gates
results/      contracts and measurements, tracked; tensors and weights not
assets/       figures, all of them regenerated by c2c_plot_sandbox.ipynb
TODO.md       the specification: what each gate claims and how it fails
FINDINGS.md   what the gates measured
```

The split between the first two directories is load-bearing rather than
cosmetic. Because nothing in `c2c/` loads a model, the entire first stage of
checks runs on a machine that has never downloaded one.

## Running it

One task per gate, in the order the gates close.

```sh
mise run contracts          # both models' geometry, from their live configs
mise run substrate          # an injected cache is read, and position ids are absolute
mise run geometry           # both caches at the contract shapes, and the null
mise run train              # a projection beats a constant predictor
mise run fuser-substrate    # a shut fuser is a no-op, and the loss reaches it
mise run fuser              # train one arm; set MODE in scripts/train_fuser.py
mise run analysis           # accuracy, once, over four conditions
mise run ledger             # what the exchange costs on this machine
```

Three probes are not gates and answer questions raised inside them.

```sh
mise run shift              # does a projection fitted on wikitext carry to MMLU
mise run parity             # does the key cache split along the Sharer's rotary ladder
mise run probe-letter-mass  # where a fused run's loss drop went
```

Each script writes a file the next one reads and refuses to start when its
input is missing. The ladder is enforced by that data dependency rather than by
the runner, so the order above is a convenience and not the mechanism.

Two scripts refuse to overwrite what they wrote before, because a training arm
costs hours and a graded run is the only accuracy number this repository
reports.

## Config contract

<!-- contracts:start -->
| field | sharer | receiver |
| --- | --- | --- |
| model | Qwen/Qwen2.5-0.5B-Instruct | Qwen/Qwen3-0.6B |
| role | Sharer | Receiver |
| n_layers | 24 | 28 |
| hidden_size | 896 | 1024 |
| n_q_heads | 14 | 16 |
| n_kv_heads | 2 | 8 |
| gqa_group_size | 7 | 2 |
| head_dim | 64 | 128 |
| hidden_size / n_q_heads | 64 | 64 |
| head_dim decoupled | False | True |
| kv_width | 128 | 1024 |
| rope_theta | 1000000.0 | 1000000.0 |
| rope_theta source | config.rope_parameters | config.rope_parameters |
| deprecated rope_theta attr | None | None |
| len(inv_freq) | 32 | 64 |
| q_norm | False | True |
| k_norm | False | True |
| vocab_size | 151936 | 151936 |

Concatenated key width entering the projection: 128 + 1024 = 1152. The Sharer contributes 11.1% of the channels.

Rotary ladders: theta match True, length match False. Nesting at stride 2: True.

Layer alignment (terminal): 24 paired target layers, target [0, 1, 2, 3] unpaired.

Token alignment: probe ids identical True, therefore align_tokens is out of scope.
<!-- contracts:end -->