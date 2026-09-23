# MG_CrossAttn

Graph attention transformer over the molecular glue, frozen ESM-2 embeddings
for the target protein, bidirectional cross-attention between the two, and a
binary classifier on top. Built for the `dc50` and `dmax` PatGlue datasets
under target-disjoint 5-fold cross-validation, 80/10/10 per fold, with fold
class ratios balanced by target assignment.

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

**Fold size is the val/test size.** val and test are the two halves of the
held-out fold, so the fold-size band in `split.balance` sets the split ratio
and the class-ratio freedom at the same time. The shipped configs favour the
ratio; widening the band buys balance back — see [Splits](#splits).

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
| `split.balance` | `min_fold_row_frac` / `max_fold_row_frac` (the fold-size band the class-ratio search works inside), `min_class_rows`, search effort |

The fully-resolved config is written to `runs/<name>/resolved_config.yaml` on
every run, so a result is always traceable to its exact settings.

## Splits

Each of the 5 splits is **80% train / 10% val / 10% test**. Targets are
partitioned into folds and every target lands in exactly one fold; for fold
*k*, fold *k* is held out whole and the other four are train. The held-out
fold is then cut in half by rows — stratified within each (target, label)
cell — to give val and test. `audit_fold` re-derives all pairwise overlaps and
**raises** on a target shared with train.

Two consequences of cutting one fold in half, both deliberate:

- **val and test share targets.** Cutting the held-out fold by *target*
  instead does not work here: dc50 fold 1 holds one target (GSPT1), so it
  cannot be halved at all, and where folds do hold several, the targets differ
  wildly in size and class rate — dc50 fold 0 is VAV1 (816 rows, 90% positive)
  plus CCNK (11 rows, 0%), which a target-cut would turn into an 816/11 split
  with one half single-class. The stratified row cut instead gives val and test
  the same size (to within one row), the same target mix and the same class
  ratio. The price is that early stopping is tuned on the same targets that are
  then scored. Both halves remain target-disjoint from train, which is the
  claim the CV rests on.
- **the 5 test sets cover half the dataset**, not all of it, since each is 10%.

### Why it is not exactly 80/10/10

It cannot be. val and test are each half a fold, so hitting 10%/10% needs every
fold at exactly 20% of the data — but the largest indivisible target group is
bigger than that:

| | largest group | share | forces val/test to at least |
|---|---|---|---|
| dc50 | SMARCA2+SMARCA4 | 1096 rows, 27.3% | 13.7% each |
| dmax | VAV1 (a single target) | 937 rows, 28.6% | 14.3% each |

And once one fold is that large, the other four share what is left, so they
fall *below* 20%. Asking for equal folds is therefore self-contradictory rather
than merely ambitious, and `assign_groups_to_folds` widens the requested band
at **both** ends to the tightest the data admits, logging what it settled on.
The shipped configs ask for `1.00`/`1.00` and get:

| | fold sizes | realised train/val/test |
|---|---|---|
| dc50 | 0.91 – 1.37× fair | 73–83% / 8.5–13.7% / 8.5–13.7% |
| dmax | 0.89 – 1.43× fair | 71–85% / 7.4–14.3% / 7.4–14.3% |

**Scaffold-disjointness is off** (`split.scaffold_disjoint: false`). A molecule
recurring across folds is paired with a *different* target in each, which is a
distinct example rather than a leak — the pair is the unit of this dataset, not
the molecule. Enforcing it on top of target-disjointness bought nothing and
deleted rows non-uniformly. It is still available; `retention` in the fold
report says what it costs when switched on.

### Fold class ratios

The label here is close to a function of the target:

| dc50 (66.9% positive) | | dmax (49.6% positive) | |
|---|---|---|---|
| CSNK1A1 | 736 rows, 92.5% | VAV1 | 937 rows, 90.3% |
| VAV1 | 816 rows, 90.3% | IKZF2 | 537 rows, 70.0% |
| SMARCA2+SMARCA4 | 1096 rows, 24.2% | GSPT1 | 414 rows, 6.5% |
| CDK4/ARNT/AR/CCNK | 80 rows, 0% | CDK2 | 387 rows, 1.3% |

So a fold's positive rate is just whatever its targets' base rates average to.
Packing folds by row count alone — what the baseline did — gave folds from 24%
to 93% positive on dc50 and 4% to 90% on dmax, which makes ROC-AUC and MCC on
the small end of that range close to meaningless.

`assign_groups_to_folds` therefore picks *which targets share a fold* to
optimise the class ratio, subject to fold size staying inside a band. It never
drops a row. Fold size is a constraint rather than a competing objective
because it has the larger dynamic range and, weighted against the ratio,
silently dominates it.

### The band trades split ratio against class balance — directly

Because fold size *is* val/test size, the band is a single knob with the split
ratio at one end and the class ratio at the other. **The shipped configs sit at
the ratio end, and that costs nearly all of the balancing**: at `1.00`/`1.00`
the balancer has so little freedom that it lands on the same partition
row-count packing alone would give.

| band | dc50 val/test | dc50 fold rates | dmax val/test | dmax fold rates |
|---|---|---|---|---|
| **1.00–1.00 (shipped)** | **8.5 – 13.7%** | **24 – 93%** | **7.4 – 14.3%** | **7 – 90%** |
| 0.70–1.50 | 7.0 – 14.8% | 30 – 88% | 7.0 – 14.6% | 20 – 86% |
| 0.60–1.60 | 6.0 – 16.0% | 30 – 88% | 6.1 – 15.5% | 17 – 83% |
| 0.50–1.80 | 5.0 – 18.0% | 40 – 88% | 5.4 – 16.2% | 17 – 80% |
| 0.40–2.00 | 4.0 – 19.2% | 43 – 89% | 4.0 – 20.2% | 26 – 64% |

Two rows worth noting at the shipped setting: dmax fold 2 is 7% positive, so
its val and test each hold **17 positives against ~227 negatives** — PR-AUC and
MCC there will be very noisy. dc50 fold 0 is 24% positive.

To buy the balance back, widen the band — it is one line:

```bash
python -m src.train --config configs/dmax.yaml \
    --set split.balance.min_fold_row_frac=0.4 split.balance.max_fold_row_frac=2.0
```

Every row is kept at any band (`retention` is 1.0 on every fold), and the
`min_class_rows` floor — applied per val/test split, so twice that at fold
level — keeps every split two-class, so ROC-AUC and MCC are always defined.

The search reaches the same partition as a 3000-restart reference on both
datasets from every seed tried, in 4–7 s.

### Balance can be cosmetic — read the `spread` column

A fold sitting on the base rate can be one of two very different things: a fold
whose targets are each near it, or a fold pairing a 90%-positive target with a
4%-positive one so they cancel. Both report the same fold ratio, but only the
first is a harder test — in the second the label is still readable straight off
target identity.

`fold_balance.mixing` measures this as `spread`, the row-weighted mean gap
between a fold's constituent target rates and the fold's own rate, and
`train.py` logs it per fold. It rises as the band widens (dc50 4.2% at the
shipped setting → 21.9% at 0.3–2.8), which is the real price of a tighter
headline ratio. **This is a property of
the data, not something the search can remove**: with a label this close to a
function of the target, cancellation is the only mechanism available to a
target-disjoint split.

### The paralogy fix

### The paralogy fix

The baseline split packed *individual* targets, which scattered paralogs across
folds. Because paralogs share chemical series, the scaffold pass then destroyed
a **biased** subsample of the smaller partner's rows — SMARCA2 kept 125 of its
494 dc50 rows (25.3%), and what survived was precisely its *non-shared*
chemistry.

That row loss is now moot, since the scaffold pass is off. The grouping stays
for the reason that outlives it: SMARCA2 and SMARCA4 are near-identical
proteins sharing 235 dc50 scaffolds, so splitting the family across folds would
put a near-identical (molecule, near-identical protein) pair in both train and
test. Grouping asks the better question — "can the model generalise to a new
target **family**" rather than "to a paralog of a target it has already seen."

It is not free, and now there is a price tag on it. Grouping IKZF1/2/3/4
creates one indivisible block, and every such block is one the ratio balancer
cannot break up. With `split.group_paralogs: false` the same balancer reaches
much tighter fold ratios:

| | grouped (default) | families split |
|---|---|---|
| dc50 fold rates | 43% – 89% | **62% – 71%** |
| dmax fold rates | 26% – 64% | **33% – 64%** |
| mean mixing spread | 14% / 23% | 26% / 30% |

That is a real trade, not a free win: the tighter ratios come with more
cancellation (see above), and splitting a family puts near-identical protein
pairs on both sides of the boundary. The default stays `true` — but if fold
class ratio is the binding problem, this is the largest single lever.

`dmax` used to group `CDK2`/`GSPT1` as well. They are **not** paralogs — that
grouping existed only to stop the scaffold pass costing CDK2 58% of its rows,
and with the pass off it had no remaining job. It was also the single most
expensive constraint on that dataset: together they are an indivisible 801-row
block at 4.0% positive, a quarter of dmax, which pinned one fold at a 4.8%
positive rate however the rest were arranged. Split apart (387 rows at 1.3%,
414 at 6.5%) the worst dmax fold improves from 4.8% to 25% positive.

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
