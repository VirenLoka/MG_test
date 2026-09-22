# BASELINE CHECK

RandomForest and XGBoost baselines for binary molecular-glue activity on the
base PatGlue datasets, evaluated under two split regimes of very different
difficulty.

| | |
|---|---|
| **Datasets** | `MG_data/patglue_dc50_clean.csv` (4010 rows), `MG_data/patglue_dmax_clean.csv` (3280 rows) |
| **Labels** | `dc50_label`, `dmax_label` — binary, taken as-is from the source CSVs |
| **Glue features** | 166 MACCS key bits (bit 0 dropped) |
| **Sequences** | joined from `MG_data/protein_structures_mapped.csv` on `Gene_Name` |
| **Models** | RandomForest, XGBoost, plus a majority-class reference |

Two evaluations, because they answer different questions:

| run | split | target features | question |
|---|---|---|---|
| `train_baselines.py` | 70/15/15 grouped on `canonical_smiles` | one-hot | can it rank molecules against *known* targets? |
| `train_cv.py` | 5-fold, target- **and** scaffold-disjoint | 20-dim AAC | can it generalise to *unseen* targets and scaffolds? |

The answers are **yes (0.97 ROC-AUC)** and **no (0.72 / 0.62, ±0.20)**
respectively. Details in [Results](#results-1-molecule-disjoint-one-hot) below.

## Running it

```bash
python "BASELINE CHECK/src/train_baselines.py"
```

```bash
python "BASELINE CHECK/src/train_cv.py"
```

```bash
python "BASELINE CHECK/src/diagnostics.py"
```

Options: `--datasets dc50`, `--seed 7`, `--folds 5`, `--config <path>`. All
paths in `config/baseline.yaml` resolve relative to the repo root.

## Layout

```
config/baseline.yaml      paths, split fractions, CV settings, hyperparameters
src/data.py               dataset loading + sequence join from the protein map
src/features.py           MACCS keys + target block (one-hot or AAC)
src/splits.py             molecule-grouped splitting + leakage audit
src/cv_splits.py          Murcko scaffolds + doubly-disjoint fold construction
src/train_baselines.py    molecule-disjoint entry point
src/train_cv.py           target+scaffold-disjoint CV entry point
src/diagnostics.py        ablation / analog-similarity / label-skew analysis
outputs/metrics.json      full record of the molecule-disjoint run
outputs/summary.csv       flat dataset x model x split metrics table
outputs/<ds>/split_*.csv  the exact rows in each split
outputs/<ds>/test_predictions_<model>.csv   per-row test probabilities
outputs/<ds>/models/*.joblib                fitted models + feature names
outputs/cv_target_disjoint/cv_metrics.json  full record of the CV run
outputs/cv_target_disjoint/cv_summary.csv   per-fold metrics table
outputs/cv_target_disjoint/<ds>/cv_per_target.csv              per-target scores
outputs/cv_target_disjoint/<ds>/cv_pooled_predictions_*.csv    out-of-fold preds
```

## Split design 1: molecule-disjoint

The requirement was that no glue–target pair appear in both train and test.
That constraint is **already satisfied by construction** — both source CSVs
contain exactly one row per `(canonical_smiles, target)` pair (4010 pairs in
4010 dc50 rows; 3280 in 3280 dmax rows), so any row-level split would satisfy
it while proving nothing.

What actually leaks under a row split is the *molecule*: 667 dc50 SMILES and
746 dmax SMILES appear against more than one target. A random split would
train on `A–VAV1` and test on `A–GSPT1`, letting the model recall molecule `A`.

So splits are grouped on `canonical_smiles`: every row sharing a structure
lands in one split. `splits.audit_split` re-derives the pairwise overlaps from
the finished splits and raises if any SMILES or pair crosses a boundary, so the
guarantee is enforced at runtime rather than assumed.

Grouping on the molecule does not control target coverage. In the dc50 split
the single `VCL` row lands in test, and `VCL` is absent from train — its
one-hot block encodes as all-zero. This is reported, not patched.

## Sequence lookup

Neither clean CSV carries a sequence, so all sequences come from the protein
map. Coverage is complete: all 23 dc50 targets and all 18 dmax targets resolve
to a mapped `FASTA_Sequence`, with zero rows dropped. Sequences and UniProt IDs
are carried into the saved splits.

Note that the sequence is **not** a model feature here — the target is
represented by its one-hot identity, as specified. The sequences are joined and
persisted so the same prepared splits can feed a sequence-based model later
without redoing the merge.

## Split design 2: target- and scaffold-disjoint CV

`train_cv.py` enforces both constraints at once. Targets are partitioned into
5 folds by greedy largest-first packing on row count; for fold *k*, test =
fold *k*, val = fold *k+1*, train = the rest, so the splits are target-disjoint
by construction. Scaffold-disjointness is then imposed on top: a Bemis-Murcko
scaffold straddling two splits is assigned to whichever split holds most of its
rows, and its rows elsewhere are dropped.

That costs data — per-fold retention runs 82.6–94.9% (dc50) and 78.9–90.0%
(dmax) — which is the price of the stronger claim. `audit_folds` verifies every
fold and raises on any crossing; all 10 folds report **0 shared targets, 0
shared scaffolds, 0 shared SMILES**.

The target block switches from one-hot to **20-dim amino-acid composition**,
because a one-hot vector cannot represent a target absent from training; it
would encode as all-zero and the evaluation would be vacuous. AAC is computed
from the FASTA sequence, so an unseen target still gets a real vector.

### Two structural caveats on this design

**Fold row balance is poor, unavoidably.** Target row counts are so skewed that
whole-target folds cannot be evened out: dmax fold 0 puts `VAV1` alone in test
(936 rows, 31.7% of the data), and some val folds hold a single target. Test
base rates swing from 0.137 to 0.903 against train rates of 0.25–0.70.

**XGBoost early stopping is disabled here** (`cv.xgb_early_stopping: false`).
With a single-target val fold whose base rate is nothing like train's, early
stopping halts at a meaningless point — it cost XGBoost 0.02–0.06 ROC-AUC and
turned the model comparison into a comparison of val-fold luck. Fixed rounds
instead, so both models see the same training signal.

## Results 1: molecule-disjoint, one-hot

Test-set figures from the molecule-disjoint split (seed 42):

| dataset | model | ROC-AUC | PR-AUC | bal. acc | F1 | MCC |
|---|---|---|---|---|---|---|
| dc50 | RandomForest | 0.974 | 0.984 | 0.902 | 0.927 | 0.796 |
| dc50 | XGBoost | 0.973 | 0.984 | 0.898 | 0.925 | 0.788 |
| dmax | RandomForest | 0.965 | 0.959 | 0.895 | 0.885 | 0.787 |
| dmax | XGBoost | 0.965 | 0.961 | 0.888 | 0.878 | 0.775 |

Train/val/test gaps are small (dc50 RF: 0.993 / 0.978 / 0.974 ROC-AUC), so
these are not overfit in the ordinary sense. The two models are
indistinguishable from each other on every metric.

### Read the molecule-disjoint results with these two caveats

`diagnostics.py` exists because a 0.97 test ROC-AUC on a "disjoint" split
deserves suspicion. It finds two shortcuts that inflate these numbers.

**1. The test molecules are not structurally novel.** Maximum Morgan (r=2)
Tanimoto similarity from each test molecule to the nearest training molecule:

| dataset | median | ≥0.9 | ≥0.7 |
|---|---|---|---|
| dc50 | 0.836 | 19.9% | 89.5% |
| dmax | 0.833 | 21.7% | 90.4% |

Roughly 90% of test molecules have a training analog above 0.7 Tanimoto. Exact
structures are disjoint; the underlying chemistry is not. Accuracy holds up
across similarity bands, but the genuinely distinct band (<0.5) contains 1 dc50
row and 15 dmax rows — far too few to support a claim about novel chemistry.

**2. On dmax, target identity alone is most of the signal.** Feature ablation,
RandomForest test ROC-AUC:

| features | dc50 | dmax |
|---|---|---|
| target one-hot only (18–22 feats) | 0.869 | 0.908 |
| MACCS only (166 feats) | 0.968 | 0.925 |
| MACCS + target | 0.974 | 0.965 |

For dmax, 18 one-hot columns with no structural information reach 0.908 — the
model largely reproduces per-target base rates, which are extreme (`CDK2` 1.3%
positive over 387 rows, `GSPT1` 6.5% over 414, `VAV1` 90.3% over 937). dc50
depends more on structure (MACCS alone 0.968 vs target alone 0.869), but its
label skew is also severe (`CSNK1A1` 92.5% positive, `BRAF` 95.6%).

**What this means.** These baselines are valid for the split as specified, and
the leakage audit is genuinely clean. They do **not** establish that the models
generalise to novel chemistry or to unseen targets. The next section tests
exactly that, and the answer is negative.

## Results 2: target- and scaffold-disjoint CV, AAC features

Mean ± std of the per-fold **test** ROC-AUC across 5 folds:

| dataset | model | ROC-AUC | range | MCC |
|---|---|---|---|---|
| dc50 | RandomForest | **0.722 ± 0.222** | 0.350 – 0.952 | 0.339 ± 0.287 |
| dc50 | XGBoost | 0.702 ± 0.187 | 0.396 – 0.949 | 0.300 ± 0.292 |
| dmax | RandomForest | **0.625 ± 0.196** | 0.420 – 0.870 | 0.131 ± 0.234 |
| dmax | XGBoost | 0.533 ± 0.195 | 0.245 – 0.781 | 0.082 ± 0.235 |

Against 0.97 on the molecule-disjoint split. Pooled out-of-fold RandomForest
gives 0.822 ROC-AUC / MCC 0.520 on dc50 and **0.458 ROC-AUC / MCC −0.048** on
dmax — the dmax model is worse than a coin flip once targets are held out.

**Read the std, not the mean.** A ±0.20 spread on a metric bounded in [0,1],
with folds ranging from 0.35 to 0.95, means the mean is not a stable estimate
of anything. Performance is determined by which targets happen to land in test,
not by what the model learned.

### Adding features makes it worse

Ablation, RandomForest test ROC-AUC across the same folds:

| features | dc50 | dmax |
|---|---|---|
| MACCS only (166) | **0.749 ± 0.182** | 0.429 ± 0.154 |
| AAC only (20) | 0.564 ± 0.208 | **0.656 ± 0.186** |
| MACCS + AAC (186) | 0.722 ± 0.222 | 0.625 ± 0.196 |

On both datasets the combined feature set is *beaten by one of its own halves*.
That is the signature of a model fitting target-specific shortcuts that do not
transfer, rather than learning transferable structure–activity relationships.

**The AAC-only column needs care in interpretation.** AAC is constant within a
target, so an AAC-only model emits one probability per target. Its ROC-AUC
inside a multi-target test fold measures only whether it ordered 3–6 held-out
targets correctly by base rate — closer to a coin-flip over a handful of items
than a real ranking. dmax fold 2 scoring 0.916 on three targets is luck, and
the ±0.19 std reflects that.

### Where it fails, per target

Pooled out-of-fold RandomForest, from `cv_per_target.csv`:

| dataset | target | n | true pos-rate | mean predicted | ROC-AUC |
|---|---|---|---|---|---|
| dc50 | `VAV1` | 816 | 0.903 | 0.624 | 0.350 |
| dc50 | `SMARCA4` | 357 | 0.137 | 0.238 | 0.917 |
| dmax | `CDK2` | 163 | 0.018 | 0.423 | 0.134 |
| dmax | `VAV1` | 936 | 0.903 | 0.433 | 0.421 |

The model cannot infer a held-out target's base rate from its amino-acid
composition. `CDK2` is 1.8% positive and receives a mean probability of 0.423;
`VAV1` is 90.3% positive and receives 0.433. Both get roughly the training
prior, and several targets land below 0.5 ROC-AUC — actively anti-correlated.

## Bottom line

These baselines rank molecules well against targets they were trained on, and
fail to generalise to new targets. With 23 and 18 distinct targets, 20-dim AAC
gives the model on the order of a dozen points in target space to learn from,
which is not enough to interpolate to an unseen protein — and no amount of
model tuning fixes a target-space sample size that small.

If cross-target generalisation is the goal, the constraint is the data, not the
model: either many more distinct targets, or a target representation carrying
real structural/binding information (pocket descriptors, a pretrained protein
language-model embedding, or interface features) rather than bulk composition.
