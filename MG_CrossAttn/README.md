# MG_CrossAttn

Graph attention transformer over the molecular glue, frozen ESM-2 embeddings
for the target protein, bidirectional cross-attention between the two, and a
binary classifier on top. Built for the `dc50` and `dmax` PatGlue datasets
under target- and scaffold-disjoint cross-validation.

```
SMILES ──► graph ──► GraphAttentionTransformer ──► atom embeddings   [B, Na, D]
                                                          │
gene ──► cached ESM-2 residues ──► linear projection ──► residue emb. [B, Nr, D]
                                                          │
                              CoAttentionStack  (bidirectional)
                            atoms ⇄ residues, n_blocks deep
                                                          │
                          masked attention pooling of both streams
                                                          │
                        [mol ; prot ; mol⊙prot] ──► MLP ──► 1 logit
```

## Design decisions worth knowing

**ESM runs exactly once, offline.** There are only **27 distinct proteins** in
the whole dataset, so per-residue embeddings are extracted once by
`src/esm_embed.py` and cached to disk. The training process never instantiates
ESM — the protein tower is a projection layer. This keeps 2.6 GB of weights
out of training and is why the model is trainable without ESM in the loop.
The full cache is ~52 MB (fp16) for all 27 proteins.

**Per-residue, not mean-pooled.** Cross-attention needs a residue axis to
attend over, so the cache stores `[L, 1280]` per protein, with row *i*
corresponding to residue *i* (tokenizer BOS/EOS are stripped and the row count
is asserted against the sequence length).

**ESM is frozen, deliberately, with no fine-tuning path.** With 27 distinct
proteins, fine-tuning a 650M-parameter protein model would memorise the target
set immediately. This is a data limitation, not a knob worth exposing.

**Paralog families are kept in the same fold.** This fixes a bug in the
baseline split — see [Splits](#splits).

## Setup

```bash
pip install -r requirements.txt
```

Extract the ESM cache once per checkpoint (~2.6 GB download, then a few
minutes on CPU for 27 sequences):

```bash
python -m src.esm_embed --config configs/dc50.yaml
```

Verify the architecture end to end on CPU, no training:

```bash
python -m src.smoke_test --config configs/dc50.yaml
```

Train (GPU strongly recommended; the script warns and continues on CPU):

```bash
python -m src.train --config configs/dc50.yaml
```

```bash
python -m src.train --config configs/dmax.yaml
```

Override any config leaf from the CLI, no code change needed:

```bash
python -m src.train --config configs/dc50.yaml --set model.gat.n_layers=6 train.optimizer.lr=1e-4
```

Run a subset of folds, e.g. while checking a change:

```bash
python -m src.train --config configs/dmax.yaml --folds 0 1
```

## Layout

```
configs/base.yaml       every model + training knob, documented inline
configs/dc50.yaml       dc50 overrides (inherits base via `defaults:`)
configs/dmax.yaml       dmax overrides
src/config.py           YAML inheritance, deep merge, dotted CLI overrides
src/data/featurize.py   SMILES -> attributed graph; feature blocks are config-selected
src/data/splits.py      Murcko scaffolds, paralog-aware target folds, leak audit
src/data/dataset.py     Dataset + collate (graph batching, protein padding/masking)
src/esm_embed.py        one-time ESM extraction + in-memory embedding store
src/models/gat.py       graph attention transformer blocks, dense-batch scatter
src/models/cross_attention.py   co-attention sublayers, stack, masked pooling
src/models/model.py     assembled model: towers + co-attention + classifier
src/metrics.py          binary metrics, fold aggregation
src/train.py            CV training loop; optimizer/scheduler/loss all from config
src/smoke_test.py       9-stage CPU verification
```

## Configuration

Everything is in YAML; no hyperparameter is written in a module. `base.yaml`
covers the full surface — a selection:

| block | controls |
|---|---|
| `featurizer.atom_features` / `bond_features` | which feature blocks exist; input dims are **derived** from these lists, so dropping a name is a complete ablation |
| `model.gat` | `n_layers`, `n_heads`, `conv_type` (gatv2/gat/transformer_conv), `ffn_mult`, `norm` (pre/post), `residual`, dropouts |
| `model.cross_attention` | `n_blocks`, `n_heads`, `direction` (bidirectional / mol_to_prot / prot_to_mol), norms, dropouts |
| `model.readout` | `mol_pool` / `prot_pool` (attention/mean/max/mean_max), `interactions` (product, difference) |
| `model.classifier` | `hidden_dims`, `activation`, `dropout`, `batch_norm` |
| `model.protein` | `projection` (linear/mlp), `pool_stride` for subsampling the residue axis |
| `esm` | `checkpoint`, `layer`, `embed_dim`, `cache_dtype`, `max_residues` |
| `train` | epochs, batch sizes, `grad_clip_norm`, `accumulate_grad_batches`, `amp` |
| `train.optimizer` | adamw/adam/sgd, lr, weight decay, betas |
| `train.scheduler` | cosine/linear/plateau/none, `warmup_epochs`, `min_lr` |
| `train.loss` | `class_weighting: balanced` sets `pos_weight` per fold; `label_smoothing` |
| `train.early_stopping` | `monitor`, `mode`, `patience`, `min_delta` |
| `split` | `n_folds`, `scaffold_disjoint`, `group_paralogs`, `paralog_families` |

The fully-resolved config is written to `runs/<name>/resolved_config.yaml` on
every run, so a result is always traceable to its exact settings.

## Splits

Targets are partitioned into folds by greedy largest-first packing on row
count. For fold *k*: test = fold *k*, val = fold *k+1*, train = the rest.
Scaffold-disjointness is then imposed — a Bemis–Murcko scaffold straddling two
splits goes to whichever split holds most of its rows, and its rows elsewhere
are dropped. `audit_fold` re-derives all pairwise overlaps and **raises** on
any shared target, scaffold, or SMILES.

### The paralogy fix

The baseline split packed *individual* targets, which scattered paralogs across
folds. Because paralogs share chemical series, the scaffold pass then destroyed
a **biased** subsample of the smaller partner's rows:

| | baseline (individual targets) | here (families grouped) |
|---|---|---|
| SMARCA2/SMARCA4 shared scaffolds | 235, split across folds | same fold |
| dc50 per-fold retention | 82.6 – 94.9% | **93.7 – 99.9%** |
| dc50 SMARCA2 rows surviving as test | 125 of 494 (25.3%) | not fragmented |

Worse than the row loss was its bias: what survived for SMARCA2 was precisely
its *non-shared* chemistry. `split.paralog_families` packs each family as one
unit, so a family's shared scaffolds never straddle a boundary.

This changes the question being asked, for the better: "can the model
generalise to a new target **family**" rather than "to a paralog of a target it
has already seen." It also makes folds coarser — grouping IKZF1/2/3/4 creates
one large block, so target counts per fold become uneven even though row counts
stay balanced.

`dmax` additionally groups `CDK2`/`GSPT1`, which share 101 scaffolds without
being paralogs; leaving them split cost CDK2 58% of its rows.

## Tested environment

Smoke-tested on macOS 15 (Darwin 25.1.0), Python 3.12.3, CPU only:

```
torch 2.5.0    torch-geometric 2.6.1    transformers 4.51.3
rdkit 2025.03.2    scikit-learn 1.9.0    numpy 1.26.4    pandas 2.3.1
```

**No model was trained on that machine** — it has no GPU. Verification was
limited to `src/smoke_test.py`, which confirms shapes, masking, gradient flow,
padding invariance, loss descent on a single batch, and checkpoint round-trip.
`src/train.py` auto-selects cuda → mps → cpu and logs a warning on CPU.

## What to expect

Set expectations from the baselines in `../BASELINE CHECK`. Under
target-disjoint CV with MACCS + AAC features, RandomForest reached
**0.722 ± 0.222** test ROC-AUC on dc50 and **0.625 ± 0.196** on dmax, with
pooled out-of-fold dmax at 0.458 — below chance. Per-target analysis showed the
models could not infer a held-out target's base rate from its sequence at all.

Cross-attention over ESM embeddings is a much richer target representation
than 20-dim amino-acid composition, so there is real headroom. But the binding
constraint is unchanged: **23 and 18 distinct targets**, which is a very small
sample in target space however it is encoded. Judge results by the **std across
folds**, not the mean — a ±0.20 spread on a [0,1] metric means the mean is not
estimating anything stable. Compare against the MACCS-only and AAC-only
ablations in `../BASELINE CHECK/outputs/cv_target_disjoint/`, not against the
0.97 figure from the molecule-disjoint split, which is a different and much
easier problem.
