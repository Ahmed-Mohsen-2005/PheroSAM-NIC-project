# PheroSAM-3D — Week 7 results, explained

This document explains the results of the first full run of `PheroSAM3D_week7_baseline.ipynb` on Kaggle (T4 GPU), section by section: what each number means, what it tells us about the idea, and what to change for Phase 2. All numbers come from `week7_outputs.zip` and the notebook's printed output.

---

## 1. Summary

1. **The Phase 1 requirements are met.** The data loads, the baselines work, and the ant colony runs end to end on 12 volumes from 3 hospitals it has never seen.
2. **The ant colony (MMAS) behaves exactly as expected.**
   - It reaches 99.2% of the best possible score (the DP optimum) and its 3D Dice is the same as DP: 0.602 vs 0.603.
   - It is clearly better than random search under the same budget, and the difference is statistically significant.
   - It beats greedy propagation by 8 Dice points on average. That gap is not significant after correction with only 12 volumes.
3. **The most important finding:** with a perfect box, MedSAM reaches **0.888** Dice. With the proposer's boxes, the best any method can reach by *choosing* between them is **0.650**, and the methods reach about **0.60**.
   - So the main bottleneck is the **quality of the candidate boxes**, not the choice between them.
   - This makes **ACO_R box refinement** (moving and resizing the boxes) the top priority for Phase 2.
4. **The second finding: the masks run too far past the ends of the prostate.**
   - All selection methods produce about 2.4–2.8 separate pieces per volume, while the oracle produces about 1.
   - MedSAM produces a confident-looking mask wherever it is given a box, even on slices with no prostate. Deciding where the organ starts and ends must come from somewhere else: a better "no object" option and global 3D terms.
5. **The label-free score works in broad terms but needs improving.**
   - It ranks random paths well (median Spearman 0.84).
   - It does not rank the very best paths correctly: the oracle path scores *lower* than the DP path.
   - MedSAM's own confidence is only weakly linked to real quality (Spearman 0.33).

**Go/no-go:** **Go**, with the Phase 2 priorities re-ordered (Section 12).

---

## 2. What was run

| Item | Value |
| --- | --- |
| Data | Multi-site prostate T2 MRI, 6 hospitals (116 volumes in total) |
| Proposer training | RUNMC only: 24 training + 6 validation volumes |
| Test set | 12 volumes: 4 each from BMC, BIDMC, HK (hospitals never seen in training) |
| Foundation model | MedSAM ViT-B, frozen |
| Ant colony budget | 20 ants × 30 rounds = 600 path evaluations per volume; random search gets the same 600 |
| Seeds | 0, 1, 2 for the methods that use randomness |
| Fitness | Local terms only: MedSAM confidence + slice-to-slice smoothness |
| Hardware | Kaggle T4 GPU |

### Dataset summary (`dataset_summary.csv`)

| Site | Volumes | Slices (median) | Slice spacing × pixel size (mm) | Prostate slices (median) | Role |
| --- | --- | --- | --- | --- | --- |
| RUNMC | 30 | 20 | 3.6 × 0.52 | 14 | Source (training) |
| BMC | 30 | 32 | 3.0 × 0.42 | 13 | Test |
| I2CVB | 19 | 64 | 1.2 × 0.53 | 25 | Unused |
| UCL | 13 | 23 | 3.3 × 0.52 | 13 | Unused |
| BIDMC | 12 | 43 | 2.2 × 0.36 | 22 | Test |
| HK | 12 | 24 | 3.6 × 0.52 | 12 | Test |

**What this shows:** the hospitals differ a lot, not only in how images look (Figure 1) but also in **geometry**. BIDMC has twice as many slices, thinner slices (2.2 vs 3.6 mm) and smaller pixels than RUNMC. A model trained on RUNMC therefore sees a prostate that looks bigger and spans more slices at BIDMC. This explains much of why BIDMC is the hardest test site below.

### The MedSAM checkpoint — note for the report

The run loaded the checkpoint with SHA-256 `34b34b78c1d18cb8c6bf84cf9c00e135d6d6c965699f3c0e31ef1bc9dcb5be74`. This matches the community copy on Hugging Face (`dennistrujillo/MedSAMTest`), **not a file confirmed to come from the MedSAM authors**.

The model clearly works:
- the sanity check gives **2D Dice 0.956** with the true box;
- oracle boxes give **0.888** 3D Dice on average.

Still, write down which file you used, and if possible re-run once with the official file from the MedSAM Google Drive to confirm the numbers.

---

## 3. The proposer (Figure 2, `proposer_training.csv`)

| Epoch | 5 | 10 | 15 | 20 | 25 | **30** | 35 | 40 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Validation 3D Dice (RUNMC) | 0.733 | 0.747 | 0.699 | 0.811 | 0.786 | **0.844** | 0.835 | 0.835 |

On its own hospital, the small U-Net reaches **0.844** 3D Dice, which is good for a quick model. The best weights (epoch 30) were kept. Training took about 6 minutes.

On the unseen hospitals it drops sharply:

| Test site | Proposer 3D Dice | Drop from 0.844 |
| --- | --- | --- |
| HK | 0.836 | almost none |
| BMC | 0.603 | −0.24 |
| BIDMC | 0.358 | −0.49 |

**What this indicates:** this is the **domain shift** problem the project is built around, and it is real and large. HK is similar to RUNMC (same slice spacing, Figure 1). BIDMC has different geometry and contrast and is badly hurt. The test set therefore spans easy, medium and hard domains, which is useful for the paper.

---

## 4. Candidates and the slice graph

The graph-building step took **7 seconds** for all 12 volumes, using between 34 and 109 MedSAM decoder calls per volume. Each slice had **2.4–3.8 options** on average (including "no object").

**Candidate recall** is the share of prostate slices where at least one candidate box gives a mask with true Dice ≥ 0.7:

| Site | Volumes (recall) | Mean |
| --- | --- | --- |
| HK | 0.95, 0.71, 1.00, 0.73 | 0.85 |
| BMC | 0.64, 0.67, 0.38, 0.38 | 0.52 |
| BIDMC | 0.00, 0.38, 0.46, 0.09 | 0.23 |
| **All** | | **0.53** |

**What this indicates:**
- On about half of the prostate slices, **no candidate is good**, so no selection method (DP, ants or anything else) can produce a good mask there.
- On two BIDMC volumes (Case05, Case12) the proposer gives almost no usable box at all. Those volumes fail for every method.
- This one number explains most of the results below.

**The search window is too wide.** On many volumes the window covers almost the whole scan. For example, BMC Case14 searches slices 0–31 while the prostate occupies only 12 of them. The proposer gives probabilities above 0.3 on many non-prostate slices, so the search has to handle many slices where the right answer is "no object".

---

## 5. Main results (`table_main_results.csv`, Figure 3)

| Method | 3D Dice | HD95 (mm) | Pieces | Jumps | Fitness F |
| --- | --- | --- | --- | --- | --- |
| Proposer only | 0.599 ± 0.230 | 26.2 ± 16.6 | 2.17 | 4.67 | – |
| MedSAM best box | 0.595 ± 0.246 | 25.0 ± 17.4 | 2.17 | 5.92 | 1.213 |
| MedSAM greedy propagation | 0.520 ± 0.275 | 30.1 ± 14.4 | 1.00 | 1.42 | – |
| Random search | 0.587 ± 0.248 | 24.2 ± 17.0 | 3.22 | 4.17 | 1.188 |
| DP (Viterbi) | 0.603 ± 0.249 | 24.8 ± 17.7 | 2.75 | 3.08 | 1.256 |
| **MMAS** | **0.602 ± 0.251** | **24.6 ± 17.4** | **2.39** | **3.94** | **1.246** |
| MMAS + DP seed | 0.603 ± 0.249 | 24.8 ± 17.7 | 2.75 | 3.08 | 1.256 |
| *Oracle selection (upper bound)* | *0.650 ± 0.256* | *15.6 ± 13.9* | *1.08* | *2.67* | *1.137* |
| *Oracle boxes (upper bound)* | *0.888 ± 0.024* | *4.3 ± 1.3* | *1.00* | *1.67* | – |

(Pieces = separate parts of the mask along the slice axis, ideally 1. Jumps = neighbouring slices whose masks overlap with IoU < 0.5. ± is the standard deviation over the 12 volumes.)

### How to read each row

**Proposer only (0.599) vs MedSAM best box (0.595).** With the proposer's boxes, MedSAM adds almost nothing on average. This is not because MedSAM is weak: with true boxes it reaches 0.888. MedSAM is very good *when the box is right*; with a wrong box, it accurately outlines the wrong region. Better boxes are therefore the way to unlock MedSAM.

**Greedy propagation (0.520) is the worst real method.** This is the "error accumulation" problem from the proposal, now shown with real numbers.
- It has the most consistent masks (always 1 piece, only 1.4 jumps) but the lowest Dice and the worst HD95 (30.1 mm).
- It **lost the organ completely on two BIDMC volumes** (Dice 0.000 on Case05 and Case12). It also dropped to 0.647 on HK Case46, where the other methods get about 0.86.
- **Smooth is not the same as correct:** greedy propagation smoothly follows a mistake. This is a useful sentence for the paper's motivation.

**DP (0.603), MMAS (0.602) and MMAS + DP seed (0.603) are effectively equal.** This is the expected result for Phase 1.
- The fitness has only local terms, and DP finds the exact optimum for local terms. The colony therefore cannot do better than DP, only match it.
- MMAS + DP seed gave **exactly the same result as DP on every seed** (0.6026 each time): starting from the optimum, the colony found nothing better, which is correct behaviour.

**Random search (0.587).** It is clearly worse than the colony, and it gives the most fragmented masks (3.22 pieces). Under the same 600-evaluation budget, guided search matters.

**Oracle selection (0.650).** This is the best result possible by choosing among *these* candidates using the ground truth. DP and MMAS get about 0.60, so choosing better between the existing boxes could gain at most about 0.05 Dice.

**Oracle boxes (0.888).** This is the best result possible for MedSAM with perfect boxes. The gap between 0.650 and 0.888, about **0.24 Dice**, is lost because the candidate boxes themselves are wrong. That gap is five times bigger than the selection gap.

### The two gaps, per site (`table_dice_by_site.csv`)

| Site | DP | Oracle selection | Oracle boxes | Selection gap (oracle sel − DP) | Box gap (oracle boxes − oracle sel) |
| --- | --- | --- | --- | --- | --- |
| HK | 0.833 | 0.872 | 0.873 | 0.04 | **0.00** |
| BMC | 0.610 | 0.672 | 0.909 | 0.06 | **0.24** |
| BIDMC | 0.365 | 0.407 | 0.881 | 0.04 | **0.47** |

**What this indicates:**
- On the easy site (HK) the candidates are already as good as perfect boxes; everything is near the ceiling.
- On the harder sites, almost all of the missing accuracy comes from the boxes. On BIDMC, perfect boxes would more than double the Dice (0.365 → 0.881).
- **The harder the domain shift, the more box refinement matters.** This supports the project's main idea, provided Phase 2 adds refinement (ACO_R), not only selection.

### HD95 and pieces point to the second problem

- DP and MMAS have HD95 about **25 mm**, while oracle selection has 15.6 mm and oracle boxes 4.3 mm.
- HD95 measures the worst boundary errors, so 25 mm means some predicted pieces lie **far from the prostate**.
- Together with 2.4–2.8 pieces vs about 1 for the oracle, this shows the methods keep masks on slices **beyond the ends of the prostate** (Section 9 shows this directly).

---

## 6. Statistics (`table_wilcoxon_mmas.csv`)

Wilcoxon signed-rank test on per-volume Dice, MMAS vs each method, with Holm correction for 8 comparisons.

| Comparison | Mean Dice difference (MMAS − other) | p (raw) | p (Holm) | Significant at 0.05? |
| --- | --- | --- | --- | --- |
| MMAS vs Random search | **+0.015** | 0.001 | **0.006** | **Yes** |
| MMAS vs Greedy propagation | **+0.082** | 0.016 | 0.081 | No (borderline) |
| MMAS vs MedSAM best box | +0.007 | 0.064 | 0.256 | No |
| MMAS vs Proposer only | +0.003 | 0.470 | 1.0 | No |
| MMAS vs DP | −0.001 | 0.677 | 1.0 | No (they are equal) |
| MMAS vs MMAS + DP seed | −0.001 | 0.677 | 1.0 | No (they are equal) |
| MMAS vs Oracle selection | −0.048 | 0.0005 | 0.004 | Yes (oracle is better) |
| MMAS vs Oracle boxes | −0.286 | 0.0005 | 0.004 | Yes (oracle is better) |

**What this indicates:**
- **MMAS beats random search** consistently. That difference is small in Dice but present on almost every volume, so it is significant. The colony is a genuinely guided search.
- **MMAS vs greedy (+0.082)** is the largest gain over a real baseline. It does not survive correction with only 12 volumes, and much of it comes from the two BIDMC volumes where greedy collapsed.
  - Using all test volumes (54 instead of 12) is likely to make this clear one way or the other.
  - Until then, call it "a consistent trend, not yet significant".
- **MMAS = DP**, as expected (Section 5).
- With 12 volumes, the smallest possible p-value is 0.0005, reached when all 12 differences have the same sign. That is what the oracle comparisons show: the oracles beat MMAS on every single volume.

---

## 7. Search efficiency (Figure 5)

- **MMAS** reaches **99.2%** of the DP optimum score on average within 600 evaluations. Most of the progress happens in the first 50 evaluations (about 97%).
- **Random search** reaches only **94.6%** with the same 600 evaluations.
- **Stability across seeds:** MMAS's final score varies very little between seeds (average standard deviation 0.0035 per volume), and its Dice is 0.603 / 0.599 / 0.604 for seeds 0 / 1 / 2.
- **Speed:** the colony takes about **0.02 seconds per volume** on top of the cached MedSAM masks.

**What this indicates:**
- The pheromone mechanism works: the colony learns quickly and reliably which candidates combine well.
- It is cheap, so the real cost of the method is MedSAM, not the ants.
- This is the evidence for the "NIC approach designed and partially implemented" requirement.

**One cost to fix:** encoding the slices with MedSAM took **1011 seconds for 393 slices (about 2.6 s per slice)**. That is slow for a T4. Most of that time is probably the image resizing on the CPU before the encoder. Moving the resize to the GPU and processing slices in batches should make it several times faster, which matters before running all 54 test volumes.

---

## 8. Go/no-go checks (`table_go_no_go_preview.csv`, Figure 6)

| Check | Result | Target | Verdict |
| --- | --- | --- | --- |
| Headroom (oracle selection − best box) | **0.056** | ≥ 0.03 | Pass |
| Candidate recall | **0.53** | ≥ 0.80 | Fail |
| MedSAM confidence vs true slice Dice (Spearman) | **0.33** | ≥ 0.40 | Fail |
| Fitness F vs true 3D Dice (median Spearman) | **0.84** | ≥ 0.50 | Pass |

### Headroom — pass, but small
Choosing better among the existing candidates could add about 0.05 Dice. That is real but small. The bigger potential (0.24) is in better boxes (Section 5).

### Candidate recall — fail
This is the main bottleneck, explained in Section 4. The plan already said what to do if recall is weak: **"refinement must close the gap"**. ACO_R moves to the top of the Phase 2 list. A stronger proposer (nnU-Net) is the second lever.

### MedSAM confidence — fail
Figure 6 (left) shows two problems:
- **Narrow, shifted range.** MedSAM's predicted IoU only ranges from about 0.36 to 0.66, even for excellent masks. In the sanity check, a mask with true Dice 0.956 got a predicted IoU of only 0.57.
- **Confident about nothing.** There is a row of points at true Dice = 0 spread across the whole confidence range, up to 0.66. These are masks on slices with no prostate. **MedSAM cannot tell you "there is nothing here"**: give it a box and it confidently outlines whatever is inside.

The fix for Phase 2:
- try the stability score (`CONF_SOURCE = "stability"`), which was computed in this run but not used;
- let the "no object" decision rely on the proposer and on global 3D terms, not on MedSAM.

### Fitness vs true Dice — pass, with an important caveat
- The median is 0.84, so on most volumes paths with higher F really do have higher Dice.
- The two exceptions, **BIDMC Case05 (0.31)** and **BIDMC Case12 (−0.03)**, are the two volumes with almost no good candidates (recall 0.00 and 0.09). When every option is bad, the score cannot help.
- **Caveat:** this test uses mostly random paths, which are easy to rank. Among the *best* paths the score is wrong in a revealing way. The oracle-selection path has F = **1.137**, lower than the DP path (**1.256**), yet a much higher Dice (0.650 vs 0.603).
- So maximising F moves you *away* from the truth near the top. The current F prefers confident, smooth masks, including masks on slices without prostate, over correct ones.
- **This is the most important thing to fix in the score** (Section 12).

---

## 9. Qualitative examples

### Figure 4 — HK Case46 (apex, mid-gland, base)
This is the volume where the colony gains most over greedy propagation (0.856 vs 0.647).
- **Greedy propagation** (left): the mask grows too large at the apex and the base. Each slice inherits a slightly too-large box from the previous slice and the error accumulates.
- **DP and MMAS** (middle, right): the outlines follow the true boundary (green) much more closely on all three slices, because they choose among the proposer's boxes on each slice instead of copying forward.
- On the mid-gland slice all three methods are good. The differences are at the ends of the organ, where the anatomy is hardest.

### Figure 7 — HK Case49, pheromone disagreement and flags
The orange line is the true Dice of the MMAS result on each slice; the purple bars are pheromone entropy (how much the colony disagreed).
- On slices **15–23** the true Dice is **0**: MMAS predicted prostate on slices where there is none, past the base. This is the over-extension problem, seen directly.
- The 3 flagged slices (17, 19, 20) all fall **inside that wrong region**. In this example, the slices the colony would show a doctor are exactly the wrong ones.
- **Caveat 1:** entropy is high almost everywhere (0.77–0.95), so the colony is not very decisive. That follows from the narrow confidence range above: when all options look similar, the ants spread out.
- **Caveat 2:** this is one example. Whether flags find wrong slices *in general* must be measured over all volumes in Phase 2 (research question 5).
- It is still an encouraging first sign for the doctor-correction idea.

---

## 10. The global-term demo (`table_global_term_demo.csv`)

A penalty for fragmented masks (extra pieces) was switched on (`W_PEN = 0.2`). DP cannot use it; MMAS can.

| Result | Value |
| --- | --- |
| Volumes where MMAS found a better full score than DP | **7 / 12** |
| Pieces reduced on those 7 volumes | e.g. BIDMC Case06 4 → 2, HK Case49 4 → 2, BMC Case12 2 → 1, HK Case43 2 → 1 |
| Dice change on those 7 volumes | between −0.008 and +0.014 (essentially unchanged) |

**What this indicates:**
1. **The main claim works:** once the score contains a term that depends on the whole volume, the colony finds better solutions than DP, on 7 of 12 volumes here. That is the "why not just use DP?" argument, already visible in a first demo.
2. **This particular penalty does not improve accuracy.** Fewer pieces did not mean higher Dice: the removed pieces were small, and the larger wrong regions past the organ's ends (Section 9) stay. A fragmentation penalty is the wrong global term for the main error.
3. Phase 2 needs global terms that target **the organ's extent and size**: the 3D shape autoencoder, and an expected volume and slice-span range learned from the training masks.

---

## 11. What the results mean for the research questions

| Research question | What week 7 shows |
| --- | --- |
| RQ1: Does the colony beat baselines and exceed the best proposer box? | Beats random search (significant) and greedy propagation on average (not yet significant); equal to DP, as expected for local terms. Cannot exceed the proposer's boxes yet; that needs ACO_R. |
| RQ2: When is DP not enough? | First evidence: with one global term, the colony beats DP's score on 7/12 volumes. |
| RQ3: Competing colonies | Not tested (single organ). |
| RQ4: Corrections | Not tested yet. |
| RQ5: Does the label-free score work; do flags find wrong slices? | The score ranks paths broadly well (0.84) but prefers wrong paths near the top; MedSAM confidence is weak (0.33). One example where flags hit exactly the wrong slices. |
| RQ6: MedSAM2 plug-in | Not tested. |

---

## 12. What to change for Phase 2, in priority order

1. **Add ACO_R box refinement first.** The box gap (0.24 overall, 0.47 on BIDMC) is five times the selection gap (0.05). This is where the method can show a real gain over all baselines, including DP.
2. **Fix the organ's extent (the false slices).**
   - Make the "no object" option stronger: base its confidence on the proposer's predicted area or mean probability on that slice, not only its maximum.
   - Use a stricter search window (e.g., threshold 0.5).
   - Add global terms for expected organ volume and number of slices, plus the 3D shape autoencoder.
   - This should cut HD95 (about 25 mm now) and the number of pieces.
3. **Improve the confidence signal.** Rerun with `CONF_SOURCE = "stability"` and compare the Spearman values. Consider combining MedSAM confidence with the proposer's probability inside each mask.
4. **Fix the fitness at the top end.** After steps 2 and 3, check again that the oracle-selection path scores *higher* than the DP path. This is a simple, direct test of whether the score is right.
5. **Better proposer.** nnU-Net (as planned), or longer training with stronger intensity and scaling augmentation for the BIDMC geometry.
6. **More test volumes.** Use all BMC, BIDMC and HK volumes (`N_TEST_PER_SITE = None`, 54 volumes) so that differences like MMAS vs greedy can reach significance.
7. **Speed up MedSAM encoding** (GPU resize, batching) before step 6, and cache the embeddings to disk so they are computed only once.
8. **Confirm the checkpoint** with the official MedSAM file, or at least report the exact file used.

---

## 13. Suggested text for the Phase 1 report

> **Initial results.** We implemented the first version of PheroSAM-3D on multi-site prostate MRI. A 2D U-Net proposer trained only on RUNMC (validation 3D Dice 0.844) generated up to five candidate boxes per slice, which frozen MedSAM decoded into masks. On 12 volumes from three unseen hospitals (BMC, BIDMC, HK), the proposer dropped to 0.599 3D Dice, confirming a strong domain shift (0.836 on HK, 0.358 on BIDMC). MedSAM with the best box per slice reached 0.595 and greedy slice-to-slice propagation 0.520, failing completely on two BIDMC volumes due to error accumulation. A MAX–MIN ant colony over the slice graph reached 0.602, matching the exact dynamic-programming optimum (0.603) as expected for the local-only fitness used in this phase. It reached 99.2% of the optimal score within 600 evaluations, against 94.6% for random search under the same budget (Dice difference significant, Holm-corrected p = 0.006). With a global fragmentation term added, the colony found better solutions than dynamic programming on 7 of 12 volumes. Oracle analyses show that perfect selection among the current candidates would reach 0.650 and perfect boxes 0.888. The main limitation is therefore candidate box quality (candidate recall 0.53), which motivates the ACO_R box refinement and global 3D terms planned for Phase 2.

---

## 14. Limitations to state honestly

- **Small test set** (12 volumes): most differences between real methods are not yet statistically significant.
- **Local-only fitness:** the colony cannot beat DP on the main result yet; this is by design for Phase 1.
- **No box refinement yet:** no method can exceed the oracle-selection ceiling (0.650).
- **Weak, simple proposer:** a quick U-Net, to be replaced by nnU-Net.
- **MedSAM confidence is poorly calibrated** for this data, and MedSAM cannot signal "no organ on this slice".
- **Checkpoint source:** a community copy, not yet confirmed against the official file.
- **Data leakage not yet checked:** MedSAM may have seen some of these public prostate images during its training. This is planned for week 8.
- **One qualitative example** for the flags; a quantitative test is needed.

---

## Appendix — per-volume 3D Dice

| Site / case | Proposer | Best box | Greedy | DP | MMAS | Oracle sel. | Oracle boxes | Candidate recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BIDMC Case05 | 0.174 | 0.195 | 0.000 | 0.214 | 0.205 | 0.221 | 0.905 | 0.00 |
| BIDMC Case06 | 0.556 | 0.626 | 0.746 | 0.635 | 0.627 | 0.688 | 0.886 | 0.38 |
| BIDMC Case09 | 0.482 | 0.573 | 0.460 | 0.553 | 0.554 | 0.636 | 0.890 | 0.46 |
| BIDMC Case12 | 0.219 | 0.059 | 0.000 | 0.058 | 0.059 | 0.082 | 0.845 | 0.09 |
| BMC Case12 | 0.635 | 0.673 | 0.577 | 0.668 | 0.666 | 0.722 | 0.919 | 0.64 |
| BMC Case14 | 0.633 | 0.649 | 0.551 | 0.642 | 0.656 | 0.681 | 0.920 | 0.67 |
| BMC Case21 | 0.536 | 0.525 | 0.435 | 0.540 | 0.528 | 0.618 | 0.895 | 0.38 |
| BMC Case28 | 0.609 | 0.572 | 0.525 | 0.588 | 0.591 | 0.668 | 0.900 | 0.38 |
| HK Case42 | 0.892 | 0.869 | 0.881 | 0.891 | 0.891 | 0.910 | 0.898 | 0.95 |
| HK Case43 | 0.802 | 0.767 | 0.759 | 0.782 | 0.799 | 0.815 | 0.863 | 0.71 |
| HK Case46 | 0.867 | 0.855 | 0.647 | 0.860 | 0.856 | 0.890 | 0.860 | 1.00 |
| HK Case49 | 0.782 | 0.774 | 0.662 | 0.799 | 0.793 | 0.874 | 0.870 | 0.73 |

(MMAS is averaged over 3 seeds. Volumes with low candidate recall fail for every method, which confirms that the boxes, not the search, limit the result.)
