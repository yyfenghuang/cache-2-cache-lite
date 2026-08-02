# c2c-lite

A from-scratch reconstruction of cache-to-cache communication between two
language models, built to find out whether a KV-cache can be made legible to a
model from a different generation.

The mechanism comes from *Cache-to-Cache: Direct Semantic Communication Between
Large Language Models*, by Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan,
Guohao Dai, Wanli Ouyang, and Yu Wang, published at ICLR 2026
([arXiv:2510.03215](https://arxiv.org/abs/2510.03215)). The authors span
Tsinghua University, Infinigence AI, the Chinese University of Hong Kong,
Shanghai Jiao Tong University, SLAI, and Shanghai AI Laboratory, and the
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
smallest pair the original work reports, and the two models are separated by a
generation rather than by size.

Three differences follow from that separation, and each is a candidate reason
for the transfer to fail.

The receiver normalizes its keys per head before applying rotary embeddings.
The sharer has no such step, so its keys arrive with a magnitude distribution
the receiver's own keys never exhibit at any depth.

Keys carry position and values do not. Position is written into a key as a
rotation, in a basis fixed by the head dimension. If the two models size their
heads differently, a transferred key arrives encoded in a basis its reader does
not use, while a transferred value arrives with no such problem.

The cache is stored at the key-value head count, not the attention head count.
The width the projection actually spans is that product, which for these two
models is expected to differ by close to an order of magnitude rather than the
fifteen percent their hidden sizes suggest.

None of these is fatal on paper. All three are cheap to check before anything is
trained, which is what most of this repository does.

## What would count as an answer

Three outcomes are possible and all three are informative.

A learned map reaches the receiver's cache and the fused system reads better
than the receiver reading alone. The mechanism generalizes past the family it
was demonstrated in, and the architectural differences above are absorbable.

A learned map reaches the receiver's cache but the fused system reads no better
than the receiver alone. The geometry is bridgeable and the bridge is not
useful, which places the difficulty in how transferred information is combined
rather than in whether it can be carried.

A learned map does not reach the receiver's cache at all. The three differences
above become the candidate explanations, and each was already measured in
isolation before training began, so the search has somewhere to start.

## What this is not

Not a benchmark run. The reference implementation exists, is public, and
publishes trained weights for this exact pairing. Those weights are deliberately
not used here. A mechanism reproduced from a checkpoint teaches nothing about
the mechanism.

Not a faster or better variant. Every architectural choice follows the original
unless a measurement here says otherwise, and where it does say otherwise, the
measurement is in `results/`.

Not finished. Every accuracy number this repository will eventually report sits
behind a single gate that has not been opened, and until it is, `results/` holds
measurements of geometry and cost only.

## Layout

```
c2c_lite/     the mechanism, as pure functions: no file I/O, no model loading
scripts/      everything that touches weights or disk
tests/        the gates
results/      contracts and measurements, tracked; tensors and weights, not
assets/       figures
```

The split between the first two directories is load-bearing rather than
cosmetic. Because nothing in `c2c_lite/` loads a model, the entire first stage of
checks runs on a machine that has never downloaded one.

## Running it

```sh
mise run setup       # once
mise run gates       # every check that does not require training
```

`gates` runs three stages in order: the configuration contract, which reads both
models' geometry from their live configs; the substrate checks, which establish
that an injected cache is read and that reading it has consequences; and the
geometric baseline, which measures how far apart the two caches are before any
mapping is trained. Each stage writes a file the next one reads, and each script
refuses to start when its input is missing.

Training runs afterward and only afterward.

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
