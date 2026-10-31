# PheroSAM-3D — Week 7 notebook, explained

This file explains what `PheroSAM3D_week7_baseline.ipynb` does, section by section: what each cell does, why it is there, which settings matter, and how to read what comes out. Section numbers match the headings inside the notebook.

---

## The big picture in one paragraph

A prostate MRI scan is a stack of 2D slices. MedSAM can outline the prostate on one slice very well, **but only if you give it a box** around the prostate on that slice. The notebook builds a small, honest version of the PheroSAM-3D pipeline: a small network (the *proposer*) guesses where the prostate is and suggests a few possible boxes on each slice. MedSAM turns every suggested box into a mask. Then different methods — simple baselines, dynamic programming, random search and an **ant colony (MMAS)** — each pick **one box per slice** so that the whole 3D result is good. All methods are scored against the real outlines (ground truth), which are used **only for evaluation**, never for choosing.

```
MRI volume ──► proposer (U-Net) ──► 3–5 candidate boxes per slice + "no object"
                                            │
                                            ▼
                    MedSAM (frozen): one mask + confidence per box (cached)
                                            │
                                            ▼
          slice graph: columns = slices, nodes = candidate masks, edges = neighbour slices
                                            │
           ┌───────────────┬────────────────┼───────────────┬──────────────┐
           ▼               ▼                ▼               ▼              ▼
       best box      greedy propagation   DP (Viterbi)   random search    MMAS ant colony
           └───────────────┴────────────────┴───────────────┴──────────────┘
                                            │
                                            ▼
                     3D mask per method ──► Dice, HD95, consistency, statistics, figures
```

What this notebook is **for**: the Phase 1 requirements "dataset is working + baseline model is working + the NIC approach has been designed/partially implemented", plus deliverables 8 (initial implementation), 9 (baseline model) and 10 (initial experimental results).

What it deliberately does **not** do yet (Phase 2): the 3D shape term, the ACO_R box refinement, doctor corrections, competing colonies for several organs, pheromone memory and MedSAM2 plug-in mode.

---

## Section 0 — Setup

### Cell: GPU check and installs
- `nvidia-smi -L` prints the GPU name. If it says "No GPU found", the run will be extremely slow; switch the accelerator to GPU.
- Installs three things that Kaggle/Colab do not always have:
  - `nibabel` — reads `.nii` / `.nii.gz` medical image files.
  - `gdown` — downloads files from Google Drive (both the dataset and the MedSAM checkpoint live there).
  - `segment-anything` — Meta's SAM code. MedSAM uses exactly the SAM ViT-B architecture with different weights, so this package is all we need to run MedSAM.

### Cell: CONFIG — the only cell you normally edit
Every setting of the experiment lives here, so the run is reproducible and easy to change. The important ones:

| Setting | Default | Meaning |
| --- | --- | --- |
| `WORK` | `/kaggle/working/pherosam` (Kaggle) | Where data, checkpoint and outputs go. Chosen automatically for Kaggle, Colab or a local PC. |
| `DATA_ROOT`, `CKPT_PATH` | inside `WORK` | Point these at `/kaggle/input/...` if you uploaded the data or checkpoint yourself. |
| `SOURCE_SITE` | `RUNMC` | The only hospital the proposer is trained on. |
| `TEST_SITES`, `N_TEST_PER_SITE` | BMC, BIDMC, HK × 4 | Unseen hospitals used for testing → 12 test volumes. `None` = all volumes of those sites. |
| `SEEDS` | 0, 1, 2 | Random seeds for the methods that use randomness (random search, MMAS). |
| `UNET_*` | 256 px, 40 epochs | Size and training length of the proposer. |
| `THRESHOLDS`, `MARGINS`, `K_MAX` | 0.5/0.3/0.7, 0/6 px, 5 | How candidate boxes are made. |
| `WINDOW_THR`, `WINDOW_EXTEND` | 0.3, 2 | Which slices are searched (see Section 4). |
| `CONF_SOURCE` | `iou_pred` | Which MedSAM confidence to use (`iou_pred` or `stability`). |
| `W_CONF`, `W_SMOOTH`, `W_PEN` | 1, 1, 0 | Weights of the label-free fitness. `W_PEN = 0` means local terms only. |
| `N_ANTS`, `N_ROUNDS` | 20, 30 | Ant colony budget = 600 path evaluations. Random search gets the same 600. |
| `ALPHA`, `BETA`, `GAMMA` | 1, 2, 1 | How much ants trust pheromone, MedSAM confidence, and smoothness. |
| `RHO`, `P_BEST` | 0.1, 0.05 | Evaporation rate; MMAS parameter used to compute the lower pheromone limit. |
| `GB_SWITCH`, `PATIENCE` | 10, 10 | When to switch from iteration-best to global-best deposit; early stopping. |
| `KAPPA_EPS`, `KAPPA_EMPTY` | 0.05, 0.5 | Smoothing so zero overlap is not impossible; neutral smoothness for "no object". |
| `TAU_SEED_FRAC` | 0.5 | Starting pheromone of non-DP edges in the DP-seeded colony. |
| `GREEDY_MARGIN` | 5 px | Box margin for greedy propagation. |

### Cell: imports and figure style
- Picks the device (GPU if available), creates the output folders, and fixes random seeds so results repeat.
- Sets a consistent, quiet figure style: light background, light grid, and **one accent colour (blue) for the ant colony**, grey for baselines, light grey for the ground-truth upper bounds. Every figure is saved as a PNG in `outputs/figures/`.

---

## Section 1 — Data: multi-site prostate MRI

### The dataset
The *multi-site prostate MRI* collection (Liu et al., MS-Net / SAML) contains T2-weighted prostate MRI from **six hospitals**, taken from three public datasets (NCI-ISBI 2013, I2CVB, PROMISE12). The authors already converted it to NIfTI and resized every slice to 384×384. Because each hospital uses a different scanner and protocol, the images look different — this is the **domain shift** the project is about.

### Cell: download
- Searches `DATA_ROOT` for folders named RUNMC, BMC, I2CVB, UCL, BIDMC or HK that contain `.nii` files.
- If they are not there, downloads the zip from Google Drive with `gdown` and unzips it.
- Stops with a clear message if the source hospital is missing.

### Cell: list and load cases
- `list_cases` pairs each image file with its mask file in each site folder. It recognises `Case00_segmentation.nii.gz` and `Case00_Segmentation.nii.gz` (both spellings exist).
- `load_case` reads one volume and:
  - turns it into the shape **(slices, height, width)**. The slice axis is detected as the **smallest dimension**, so it works whatever order the file stores;
  - turns the mask into 0/1 (some masks use other label values; anything above 0 counts as prostate);
  - reads the voxel spacing in millimetres (needed for HD95).
- `normalize_255` copies **MedSAM's own MRI preprocessing**: clip intensities to the 0.5th–99.5th percentile, then rescale to 0–255. Matching MedSAM's preprocessing matters; a mismatch silently hurts MedSAM.
- `get_vol` loads a volume once and keeps it in memory (only for the volumes actually used).
- The cell ends with a **dataset summary table** (`dataset_summary.csv`): volumes per site, slices, in-plane size, spacing, how many slices contain prostate, and each site's role (source / test / unused). This is your "dataset is working" evidence.

### Cell: Figure 1 (`fig1_sites.png`)
One middle prostate slice from each hospital with the true outline in orange. It shows visually that the hospitals look different. Good for the dataset section of the report.

---

## Section 2 — Proposer: small 2D U-Net trained on one hospital

### Why a proposer
MedSAM needs a box on every slice. In a real automatic pipeline, someone must suggest those boxes. The proposer is a segmentation network whose only job is to say roughly *where* the prostate is. We do **not** need it to be perfect; MedSAM and the search do the precise work.

### Why trained on one hospital only
The proposer sees only RUNMC. The test hospitals are never seen in training, so the test is a realistic "new hospital" scenario: the proposer will be weaker there, which is exactly when a smart box search should help.

### Cell: the U-Net
- A standard 2D U-Net: 4 down-sampling steps, 4 up-sampling steps with skip connections, base width 32 channels.
- Input: one slice, z-score normalised per volume, resized to 256×256. Output: a probability map (after sigmoid).
- RUNMC volumes are split 80% train / 20% validation **by volume** (never by slice, which would leak information between train and validation).

### Cell: training
- Loss = **binary cross-entropy + soft Dice loss** (the usual combination for medical segmentation; Dice handles the small foreground).
- Simple augmentation: random left–right flip and random brightness/contrast jitter.
- AdamW optimiser with cosine learning-rate schedule, 40 epochs.
- Every 5 epochs, 3D Dice on the RUNMC validation volumes is computed; the best weights are kept.
- `predict_volume` runs the trained U-Net on every slice of a volume and resizes the probabilities back to 384×384.
- Saved: `proposer_training.csv`.

### Cell: Figure 2 (`fig2_proposer_training.png`)
Training loss (should go down) and validation 3D Dice (should go up and level off). For the report: "the proposer reaches X validation Dice on its own hospital".

---

## Section 3 — Test volumes and MedSAM (frozen)

### Cell: choose test volumes
- From each test hospital, picks `N_TEST_PER_SITE` volumes with a fixed random seed (so everyone running the notebook gets the same volumes).
- Runs the proposer on each and records the proposer's own 3D Dice. Printed per site: this already shows how much the proposer drops on unseen hospitals.

### Cell: MedSAM
**How MedSAM works.** It has two parts:
1. A **heavy image encoder** (a Vision Transformer) that turns a 1024×1024 image into an "embedding" (256×64×64 numbers). Slow: run it **once per slice**.
2. A **light mask decoder** that takes the embedding plus a box and returns a mask. Fast: milliseconds per box.

Because the encoder runs only once per slice, trying many boxes is cheap. This is what makes search-based methods practical.

What the cell does:
- Downloads `medsam_vit_b.pth` (≈375 MB) if needed and loads it into the SAM ViT-B architecture. **MedSAM is never trained or changed** ("frozen").
- `encode_volume`: for every slice, copies MedSAM's preprocessing exactly (repeat the grey slice into 3 channels, resize to 1024×1024 with cubic interpolation, min–max scale to [0, 1]), runs the encoder, and stores the embedding in half precision on the CPU to save memory.
- `decode(embedding, boxes)`: converts boxes from image pixels to MedSAM's 1024 coordinates, runs the decoder for many boxes at once, and returns for each box:
  - the **mask** (probability > 0.5),
  - MedSAM's **predicted IoU** — the model's own guess of how good the mask is,
  - a **stability score** — how much the mask changes if the threshold on its raw output moves a little (stable masks are usually good). This is a backup confidence in case the predicted IoU turns out unreliable.
- A global counter `DECODER_CALLS` counts every decoder call, so the cost of each method can be reported.
- **Sanity check**: decodes the true box on one slice and prints the 2D Dice. If MedSAM is loaded correctly this should be high (roughly 0.85 or more). If it is very low, something is wrong with the checkpoint or preprocessing — stop and investigate before trusting anything else.

---

## Section 4 — Candidate boxes and the slice graph

### Search window
Not every slice contains prostate. The notebook only searches slices where the proposer's maximum probability is above 0.3, **plus 2 extra slices on each side** (`WINDOW_EXTEND`), so the search can extend the organ if the proposer cut it short. Slices outside the window are predicted empty by every selection method. If the proposer finds nothing at all in a volume, that volume has no window and all selection methods return an empty mask (this is recorded honestly, not hidden).

### Candidate boxes on each slice (`slice_candidates`)
1. Threshold the proposer map at 0.5, 0.3 and 0.7.
2. Keep the **largest connected component** of each thresholded mask (removes small false spots); skip it if smaller than 30 pixels.
3. Take its bounding box, once tight (0 px margin) and once with 6 px margin.
4. Remove near-duplicates (boxes overlapping more than 90%) and keep at most 5.
5. Add the **"no object"** option.

Each candidate keeps a *type* label such as `t0.5_m6` (threshold 0.5, margin 6), used later by the pheromone memory in Phase 2.

### The slice graph (`build_graph`)
Think of a grid: **columns = slices** in the window, **nodes = candidate masks** on that slice (node 0 is always "no object"). A complete answer = **one node per column** = a path through the grid.

For every node and every edge the notebook pre-computes:
- **Node confidence η** — MedSAM's predicted IoU of that mask (or stability score). For "no object" it is `1 − max proposer probability on that slice`: if the proposer is confident there is prostate, "no object" gets low confidence, and vice versa.
- **Edge smoothness κ** between consecutive slices — the IoU (overlap) of the two masks. Real organs change gradually, so neighbouring masks should overlap a lot. **Any transition involving "no object" gets a neutral 0.5** (`KAPPA_EMPTY`). This was a deliberate fix: if empty→empty scored a perfect 1.0, the methods could be tempted to predict nothing at all on hospitals where the proposer is unsure.
- Two **virtual empty nodes** before the first and after the last window slice, so every path has a defined start and end.
- Also stored, for evaluation only: the **true Dice of every candidate mask**.

Everything is stored in padded arrays (`C` for confidences, `K` for smoothness) so a whole batch of paths can be scored in one vectorised numpy operation — that is why 600 evaluations per volume take a fraction of a second.

The cell prints a table per test volume: window, nodes per slice, decoder calls, and **candidate recall** = share of prostate slices where at least one candidate has true Dice ≥ 0.7. Low recall means no selection method can do well on that volume — the proposer, not the search, is the bottleneck.

---

## Section 5 — Methods

### The label-free fitness (the score every search method maximises)

```
F(path) = W_CONF × mean confidence of the chosen nodes
        + W_SMOOTH × mean smoothness of the transitions along the path
        − W_PEN × Pen(path)
```

- **Conf** rewards masks MedSAM is sure about.
- **Smooth** rewards neighbouring slices that agree.
- **Pen** = number of *extra* separate pieces along the slice axis (a correct prostate is one piece). Pen depends on the **whole path**, not on neighbouring slices only — a *global* term. It is switched off (`W_PEN = 0`) in the main run and switched on in Section 10.

"Label-free" means F uses no ground truth; it can be computed for a new patient in a hospital.

### The nine methods compared

| Method | What it does | Role |
| --- | --- | --- |
| **Proposer only** | U-Net probability > 0.5, no MedSAM. | Shows what MedSAM adds. |
| **MedSAM best box** | On each slice independently, picks the node with the highest confidence. No consistency between slices. | Simplest automatic MedSAM pipeline. |
| **MedSAM greedy propagation** | Starts on the slice where the proposer is most confident, picks the best box there, then moves slice by slice up and down: the box for the next slice is the bounding box of the previous mask (+5 px). Stops when the mask becomes tiny. Uses new decoder calls. | The common "copy forward" approach; one bad slice spreads through the rest (error accumulation). |
| **Random search** | Draws 600 random paths (same budget as MMAS) and keeps the one with the best F. | Sanity check: a smart search must beat random. |
| **DP (Viterbi)** | Finds the **exactly best** path for the local fitness (Conf + Smooth) using dynamic programming. | Strongest non-nature-inspired baseline. |
| **MMAS** | Ant colony; details below. | The NIC method. |
| **MMAS + DP seed** | Same colony, but starting pheromone strongly favours the DP path. | Contribution 2 of the proposal (DP-seeded ACO). |
| **Oracle selection** | Picks, on each slice, the candidate with the highest **true** Dice. | Upper bound for any method that only selects among these candidates. |
| **Oracle boxes** | Gives MedSAM the box of the **true** mask on every slice. | Upper bound for any box-driven MedSAM method. |

The two oracles use the ground truth and are **not** real methods. They tell you how much room there is to improve.

### How DP (Viterbi) works
Going slice by slice, for each node DP keeps "the best score of any path that ends here" and which previous node achieved it. At the end it walks backwards to recover the best path. Because the local fitness is a sum of node scores and neighbour-edge scores, this is **exact** and very fast. DP cannot handle Pen or any other global term, because those depend on the whole path at once.

### How the MMAS ant colony works (the `mmas` function)
This follows the MAX–MIN Ant System of Stützle & Hoos (2000).

1. **Pheromone** τ sits on every edge between consecutive slices (stored as an array of shape slices × nodes × nodes). Initially every edge gets the maximum level τ_max (encourages exploration).
2. **Each round, 20 ants each build one path.** At each slice, an ant standing on node *i* moves to node *j* with probability proportional to

   ```
   τ_ij^α × η_j^β × (κ_ij + ε)^γ
   ```

   i.e. pheromone (what earlier ants learned) × MedSAM confidence (how good the box looks) × smoothness (how well it continues the previous mask). With α = 1, β = 2, γ = 1, confidence counts most.
3. **Score** every ant's path with F.
4. **Evaporation**: all pheromone is multiplied by (1 − ρ) = 0.9, so old information fades.
5. **Deposit**: only one ant adds pheromone, an amount equal to its F, on the edges of its path — the round's best ant for the first 10 rounds (more exploration), then the best ant found so far (more exploitation).
6. **MAX–MIN limits**: pheromone is clipped to [τ_min, τ_max]. τ_max = best F / ρ; τ_min is computed from τ_max with the standard MMAS formula (parameter p_best = 0.05). The lower limit stops any option from becoming impossible, which prevents the colony from locking onto a wrong path too early — important with noisy medical images.
7. **Stop** after 30 rounds, or earlier if the best F has not improved for 10 rounds.
8. **Pheromone entropy per slice**: at the end, for each slice, how spread out the pheromone is across its candidates (0 = all ants agree, 1 = no agreement). High entropy = the colony is unsure about that slice. This is the basis of the **doctor-review flags** in Phase 2.

The function also returns the **convergence curve** (best F after each evaluation), used in Figure 5.

**MMAS + DP seed**: before the first round, edges on the DP path start at τ_max and every other edge at 0.5 × τ_max, so the ants begin near the DP answer but can still leave it.

### `path_to_mask`
Turns a path (one node index per window slice) into a 3D mask by stacking the chosen MedSAM masks; slices outside the window and "no object" choices stay empty.

---

## Section 6 — Run every method on every test volume

### Metrics (computed per volume, per method)
| Metric | Meaning | Better |
| --- | --- | --- |
| **3D Dice** | Overlap between predicted and true 3D masks: 2·|A∩B| / (|A|+|B|). | Higher (1 = perfect) |
| **HD95 (mm)** | 95th percentile of distances between the two surfaces, in millimetres. Measures boundary errors and stray pieces. Undefined if a mask is empty (counted separately). | Lower |
| **Pieces** | Number of separate parts of the mask along the slice axis. A prostate should be 1. | 1 |
| **Jumps** | Neighbouring slices both predicted but overlapping with IoU < 0.5 (sudden changes). | Lower |
| **Fitness F** | The label-free score of the chosen path (selection methods only). | Higher |
| **Decoder calls** | MedSAM decoder calls used. Selection methods share the cached candidates. | Lower |
| **Seconds** | Time of the method itself (excluding the shared encoding). | Lower |

HD95 is computed on a crop around the masks to keep it fast.

### The main loop
For each test volume: proposer only, greedy propagation and oracle boxes run once (they are deterministic); best box, DP and oracle selection run once on the cached graph; random search, MMAS and MMAS + DP seed run once **per seed**. Everything goes into `results_per_volume_and_seed.csv`. One prediction per method (first seed) is kept, bit-packed, for the qualitative figure.

---

## Section 7 — Results table and statistics

- **Seed averaging**: stochastic methods are first averaged over seeds for each volume, so each volume counts once (`results_per_volume.csv`).
- **Main table** (`table_main_results.csv`): mean ± standard deviation over test volumes for every metric and method. This is the table for deliverable 10.
- **Dice by hospital** (`table_dice_by_site.csv`): shows which unseen hospitals are hardest.
- **Statistics** (`table_wilcoxon_mmas.csv`): the **Wilcoxon signed-rank test** compares MMAS with each other method on paired per-volume Dice (same volumes, different methods). Because several comparisons are made, p-values are adjusted with the **Holm correction**. With ~12 volumes the test has little power; report it as preliminary.

---

## Section 8 — Figures

| File | Shows | Use in report |
| --- | --- | --- |
| `fig3_dice_by_method.png` | One dot per test volume, black bar = mean, for each method; ant colony in blue, oracles light. | Main results figure. |
| `fig4_qualitative.png` | The volume where MMAS gains most over greedy propagation; apex, mid-gland and base slices; green = truth, orange = prediction; columns greedy / DP / MMAS. | Qualitative example of error accumulation vs consistent search. |
| `fig5_convergence.png` | Best F found vs number of evaluations, MMAS vs random search, as a fraction of the DP optimum (1.0). | Shows the colony searches efficiently under the same budget. |

---

## Section 9 — Early go/no-go checks (preview of week 8)

These answer whether the whole idea has room to work. The ground truth is used only to *measure*.

| Check | How | Rough target | If it fails |
| --- | --- | --- | --- |
| **Headroom** | Mean Dice of oracle selection − best box. | ≥ ~0.03 | Choosing better boxes cannot help much; refinement and corrections must carry the paper. |
| **Candidate recall** | Share of prostate slices with a candidate of true Dice ≥ 0.7. | ≥ ~0.80 | The proposer is the bottleneck on unseen hospitals; ACO_R refinement is needed. |
| **Confidence vs truth** | Spearman correlation between MedSAM confidence and true slice Dice, over every candidate. | ≥ ~0.40 | Switch `CONF_SOURCE` to `"stability"`. |
| **Fitness vs truth** | For each volume, 200 random paths + the DP, best-box and oracle paths: Spearman between F and true 3D Dice; median over volumes. | ≥ ~0.50 | The score points the search in the wrong direction; reweight or redesign F before Phase 2. |

Saved as `table_go_no_go_preview.csv`; plotted in `fig6_go_no_go_checks.png` (left: confidence vs truth scatter; right: per-volume fitness correlation).

`fig7_flags_preview.png` shows, for one volume, the **pheromone entropy per slice** (bars) next to the **true Dice of the MMAS result on that slice** (line), and marks the 3 most uncertain slices as "flag". If the flags tend to fall where the line dips, the pheromone map is a useful pointer for a doctor — that is research question 5.

---

## Section 10 — Why the colony matters once a global term is added

With only local terms, DP is exact, so the best MMAS can do is **match** it. This is expected and should be stated clearly in the report.

This small demo switches on the global penalty (`W_PEN_DEMO = 0.2`, penalising fragmented masks). DP cannot include it, so it keeps its old path. MMAS seeded with the DP path optimises the **full** score. The table (`table_global_term_demo.csv`) compares, per volume, the full score, the number of pieces and the Dice of DP vs DP-seeded MMAS, and prints how many volumes MMAS improved. Phase 2 replaces this toy penalty with the real global terms (3D shape autoencoder, volume range) — the core answer to "why not just use DP?".

---

## Section 11 — Summary and download

- Writes `summary.md`: a few sentences with the key numbers (test set, Dice of each method, upper bounds, go/no-go values) ready to adapt for the report.
- Writes `config_used.json`: every setting used in the run, for reproducibility.
- Zips the whole `outputs/` folder to `week7_outputs.zip` (download it from Kaggle's Output panel).

---

## What results to expect, and how to read them

- **Proposer only < MedSAM methods** on unseen hospitals is the usual pattern: MedSAM is more robust to scanner changes than a small network trained on one hospital.
- **Greedy propagation** can be good on easy volumes but collapses on some: when one slice goes wrong, every slice after it follows. Look for this in Figure 4 and in the spread of dots in Figure 3.
- **DP ≈ MMAS ≈ MMAS + DP seed** on F and Dice: expected with the local-only fitness. MMAS reaching about 1.0 in Figure 5 shows the colony works; random search staying below 1.0 shows the search is not trivial.
- **Oracle selection vs best box** is the headroom: how much better boxes would help. **Oracle boxes** is the ceiling for any box-driven method.
- If the confidence or fitness correlation is weak, that is an important *finding*, not a failure: it tells you what to fix in week 8.

## Suggested wording for the Phase 1 report (fill in the numbers)

> We implemented the first version of PheroSAM-3D on multi-site prostate MRI. A 2D U-Net proposer trained only on RUNMC (validation Dice X) produced up to five candidate boxes per slice; frozen MedSAM decoded each into a mask. On 12 volumes from three unseen hospitals, MedSAM with the best box per slice reached a 3D Dice of X and greedy propagation X. A MAX–MIN ant colony over the slice graph reached X, matching the exact DP optimum (X), as expected for the local-only fitness used in this phase; random search under the same 600-evaluation budget reached X. The oracle upper bounds (X selection, X boxes) show headroom of X Dice. When a global fragmentation penalty is added, the colony improves the full objective on N of 12 volumes where DP cannot. Phase 2 adds the 3D shape term, ACO_R box refinement, uncertainty flags and simulated corrections.

## Known limitations of this first version (be honest about these)

- Small test set (12 volumes): statistics are preliminary.
- The proposer is a quick U-Net, not nnU-Net; it will be replaced in Phase 2.
- Fitness has local terms only, so the colony cannot yet beat DP on the main result.
- No box refinement yet: no method can do better than the best candidate box (the oracle-selection ceiling).
- MedSAM may have seen some of these public images during its own training; the leakage check is planned for week 8.

## If something goes wrong

| Problem | Likely cause | Fix |
| --- | --- | --- |
| "No GPU found" / very slow | Accelerator off | Kaggle: Settings → Accelerator → GPU T4 x2. |
| Download fails (quota / access) | Google Drive limit | Download by hand, upload as a Kaggle dataset, set `DATA_ROOT` / `CKPT_PATH`. |
| `Source site RUNMC not found` | Data unzipped somewhere else | Point `DATA_ROOT` at the folder that contains the site folders. |
| Sanity-check Dice is low | Wrong checkpoint or preprocessing | Make sure the file is `medsam_vit_b.pth` from the MedSAM page. |
| CUDA out of memory | Too many test volumes at once | Lower `N_TEST_PER_SITE` or restart the session. |
| A volume has `window = None` | Proposer found no prostate there | Expected on hard hospitals; it is reported in candidate recall. Train the proposer longer if it is frequent. |
