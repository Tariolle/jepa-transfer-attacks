# Phase 10 Plan: Explain the JEPA Continuation Gain

Audience: lower-end coding agent working in this repository.

Goal: explain the small Phase 9 matched gain from adding I-JEPA to the official dSVA continuation. Do not try to prove a new SOTA result yet. Phase 10 should answer which victims improve, whether the gains are complementary failures, and whether JEPA is helping with DINO, with MAE, or only with both.

Current known result from `RESULTS.md`:

- Phase 9 used the released dSVA checkpoint.
- Each seed compared a matched DINO+MAE continuation control against normalized DINO+MAE+I-JEPA.
- Training budget: one continuation epoch on 1000 Imagenette train images.
- Evaluation: full Imagenette validation, `eval_limit=5000`.
- Seeds: `0`, `1`, `2`.
- Mean transfer improved from `67.73% +/- 1.08` to `68.39% +/- 1.38`.
- Matched gain: `+0.66 +/- 0.35` points.

## Rules

- Keep comparisons matched: same seed, same init checkpoint, same train subset limit, same eval subset, same LR, same epochs, same output mode.
- Do not compare JEPA ablations against an uncontinued official checkpoint unless it is clearly labeled as an extra reference.
- Use `--normalize-loss-weights` whenever multiple objectives are active, so adding or removing JEPA does not silently change total objective scale.
- Keep generated CSVs and checkpoints under `results/phase10_*`. Result files are gitignored; summarize final conclusions in `RESULTS.md`.
- Prefer extending existing scripts over creating a new parallel training pipeline.

## Step 1: Inventory Phase 9 Artifacts

Find the Phase 9 output directory produced by `notebooks/phase9_dsva_jepa_validation_colab.ipynb`. Expected shape:

```text
results/phase9_.../
  seed_0/
    phase9_seed0_summary.csv
    phase9_seed0_control_lr5e-05_eval.csv
    phase9_seed0_jw0p05_lr5e-05_eval.csv
  seed_1/
  seed_2/
```

If the Phase 9 per-seed CSVs are not present locally, rerun or request the notebook outputs before doing analysis. Do not infer per-victim behavior from only the aggregate table in `RESULTS.md`.

Deliverable:

- A short note in `RESULTS.md` naming the exact Phase 9 artifact directory used for Phase 10 analysis.

## Step 2: Add a Phase 10 CSV Analyzer

Create `scripts/analyze_phase10_results.py`.

Inputs:

- `--phase9-root`: directory containing `seed_*/phase9_seed*_summary.csv` and eval CSVs.
- `--output-csv`: default `results/phase10_analysis/phase9_per_victim_summary.csv`.

Required behavior:

- Read each seed summary.
- Locate the matched control row with `run_type=control` and `lr=5e-05`.
- Locate the matched JEPA row with `run_type=jepa`, `jepa_weight=0.05`, and `lr=5e-05`.
- Read both eval CSVs.
- Compute per-victim matched gain for every victim present in the eval CSVs.
- Also compute the seed-level mean gain and the across-seed average/std for each victim.

Expected eval CSV columns likely include:

- `model`
- `clean_acc`
- `adv_acc`
- `attack_success_rate`

Check the real files before coding. If column names differ, use the real names and keep the script simple.

Output CSV columns:

```text
victim,seed,control_attack_success,jepa_attack_success,matched_gain
```

Also write a compact aggregate CSV:

```text
victim,control_mean,jepa_mean,gain_mean,gain_std,num_seeds
```

Suggested aggregate path:

```text
results/phase10_analysis/phase9_per_victim_aggregate.csv
```

Deliverables:

- `scripts/analyze_phase10_results.py`
- `results/phase10_analysis/phase9_per_victim_summary.csv`
- `results/phase10_analysis/phase9_per_victim_aggregate.csv`
- `RESULTS.md` subsection summarizing which victim(s) produced the gain.

## Step 3: Extend the Sweep Wrapper for Objective Ablations

The trainer already supports:

- `--disable-dino`
- `--disable-mae`
- `--enable-jepa`
- `--jepa-weight`
- `--normalize-loss-weights`

But `scripts/sweep_dsva_jepa_finetune.py` does not expose the disable flags. Extend it with:

- `--disable-dino`
- `--disable-mae`
- `--disable-jepa-controls` or equivalent naming only if needed
- `--extra-train-args`, optional passthrough list if useful

Keep the old default behavior unchanged.

Add the active objective names to the summary CSV. For example:

```text
objectives,run_type,jepa_weight,lr,mean_transfer_success,checkpoint,train_log,eval_csv
```

Objective labels should be deterministic:

- `dino_mae`
- `dino_mae_jepa`
- `dino_jepa`
- `mae_jepa`
- `jepa_only`
- `dino_only`
- `mae_only`

Do not break existing Phase 8/9 commands.

Deliverable:

- Updated `scripts/sweep_dsva_jepa_finetune.py`.

Quick check:

```powershell
python scripts/sweep_dsva_jepa_finetune.py `
  --dry-run `
  --output-dir results/phase10_dry_run `
  --run-prefix phase10_dry `
  --configs 0.05:0.00005 `
  --normalize-loss-weights `
  --disable-mae
```

The dry-run command should show training commands that include `--disable-mae`.

## Step 4: Run Core Objective Ablations

Use the same core Phase 9 settings:

- `--init-checkpoint`: released official dSVA checkpoint path used in Phase 9.
- `--limit 1000`
- `--eval-limit 5000`
- `--epochs 1`
- `--lr 5e-5`
- `--jepa-weight 0.05`
- `--epsilon 16/255`
- `--output-mode scaled-delta`
- `--token-loss`
- `--normalize-loss-weights`
- seeds `0`, `1`, `2`
- victims `resnet50 convnext_tiny vit_b_16`

Run these ablations:

1. `DINO+MAE` control: existing Phase 9 control. Reuse if already available.
2. `DINO+MAE+JEPA`: existing Phase 9 JEPA. Reuse if already available.
3. `DINO+JEPA`: add `--disable-mae`.
4. `MAE+JEPA`: add `--disable-dino`.
5. `JEPA-only`: add `--disable-dino --disable-mae`.
6. Optional if time allows: `DINO-only` and `MAE-only`, to separate "JEPA complements one objective" from "the removed objective was the useful part."

Recommended output layout:

```text
results/phase10_objective_ablations/
  seed_0/
  seed_1/
  seed_2/
```

For each seed, run one objective set at a time so failures are easy to resume.

Template command for `DINO+JEPA`:

```powershell
python scripts/sweep_dsva_jepa_finetune.py `
  --train-root imagenette2-320/train `
  --val-root imagenette2-320/val `
  --init-checkpoint PATH_TO_OFFICIAL_DSVA_CHECKPOINT `
  --output-dir results/phase10_objective_ablations/seed_0 `
  --run-prefix phase10_seed0_dino_jepa `
  --configs 0.05:0.00005 `
  --limit 1000 `
  --eval-limit 5000 `
  --epochs 1 `
  --batch-size 1 `
  --eval-batch-size 8 `
  --grad-accum-steps 8 `
  --epsilon 0.06274509803921569 `
  --output-mode scaled-delta `
  --victims resnet50 convnext_tiny vit_b_16 `
  --device cuda `
  --seed 0 `
  --normalize-loss-weights `
  --disable-mae `
  --skip-existing
```

Change only:

- seed number
- output prefix
- disable flags

Deliverables:

- Per-run summary CSVs under `results/phase10_objective_ablations`.
- A new aggregate CSV combining all ablations:

```text
results/phase10_analysis/objective_ablation_aggregate.csv
```

Required aggregate columns:

```text
objectives,seed,mean_transfer_success,resnet50,convnext_tiny,vit_b_16
```

## Step 5: Complementarity Analysis

If the eval CSVs only contain aggregate per-victim metrics, stop at per-victim analysis.

If the eval CSVs contain per-image rows or image identifiers, compute overlap:

- images fooled by control only
- images fooled by JEPA variant only
- images fooled by both
- images fooled by neither

Do this per victim and per seed.

Output:

```text
results/phase10_analysis/complementarity.csv
```

Columns:

```text
objectives,victim,seed,control_only,jepa_only,both,neither,jepa_unique_rate,control_unique_rate
```

Interpretation:

- If JEPA mostly increases `jepa_only`, it adds complementary failures.
- If JEPA mostly shifts the same images and lowers others, the mean gain may be noise or optimization variance.
- If no image-level rows are available, write in `RESULTS.md` that complementarity cannot be measured from current eval artifacts.

## Step 6: Update Documentation

Update `RESULTS.md` with a new section:

```markdown
## Phase 10: Objective Ablations and Per-Victim Analysis
```

Include:

- artifact paths used
- per-victim Phase 9 gains
- objective ablation table
- complementarity result if available
- decision: scale JEPA, revise objective, or stop JEPA continuation line

Update `ROADMAP.md`:

- Mark Phase 10 complete only after all required ablations are summarized.
- Add the next phase based on the result.

Do not overstate. If the gain is small, say it is small.

## Decision Criteria

Scale JEPA if at least one of these holds:

- `DINO+MAE+JEPA` improves all or most victims across seeds.
- `DINO+JEPA` or `MAE+JEPA` is close to or better than `DINO+MAE`, showing JEPA can replace one objective.
- Image-level overlap shows meaningful JEPA-unique failures.

Do not scale JEPA yet if:

- The gain is isolated to one victim and unstable across seeds.
- `JEPA-only`, `DINO+JEPA`, and `MAE+JEPA` are all clearly worse than `DINO+MAE`.
- The Phase 9 gain disappears after re-running with the same settings.

## Final Handoff Summary Format

When done, report:

```text
Phase 10 completed.

Artifacts:
- ...

Main finding:
- ...

Per-victim finding:
- ...

Objective ablation finding:
- ...

Complementarity:
- ...

Recommendation:
- ...
```
