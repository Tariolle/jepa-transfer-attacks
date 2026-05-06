# Phase 11 Plan: Scale the JEPA Continuation Signal

Audience: Colab runner.

Goal: test whether the small Phase 9/10 I-JEPA continuation gain grows with higher JEPA weights or longer continuation budgets. Keep the experiment matched against DINO+MAE controls with the same seed, LR, epoch count, train subset, and eval subset.

Colab entry point:

```text
notebooks/phase11_jepa_scale_colab.ipynb
```

## Experiment Matrix

Use the released dSVA checkpoint and the same core settings as Phase 9/10:

- `limit=1000`
- `eval_limit=5000`
- `lr=5e-5`
- `epsilon=16/255`
- `output_mode=scaled-delta`
- victims: `resnet50 convnext_tiny vit_b_16`
- seeds: `0, 1, 2`
- `--token-loss`
- `--normalize-loss-weights`

Run:

| experiment | configs | epochs |
| --- | --- | ---: |
| weight sweep | `0.05:0.00005`, `0.1:0.00005`, `0.2:0.00005` | 1 |
| longer baseline JEPA | `0.05:0.00005` | 2 |
| longer baseline JEPA | `0.05:0.00005` | 3 |

The sweep wrapper also runs matched DINO+MAE controls unless `--no-controls` is passed. Do not pass `--no-controls` for Phase 11.

## Outputs

Expected Colab outputs:

```text
results/phase11_jepa_scale/
  seed_0/
  seed_1/
  seed_2/
results/phase11_analysis/jepa_scale_detail.csv
results/phase11_analysis/jepa_scale_aggregate.csv
results/phase11_csv_artifacts.tar.gz
```

Download `results/phase11_csv_artifacts.tar.gz` from Colab and copy the CSVs back into this repo if you want local documentation updates.

## Decision Criteria

Scale JEPA if:

- `jepa_weight=0.1` or `0.2` improves mean transfer over `0.05` without worsening ViT-B/16 materially.
- 2-3 epochs improve CNN victims while preserving or improving the matched DINO+MAE control delta.
- Gains are present in most seeds rather than coming from one outlier seed.

Stop or revise the JEPA continuation line if:

- Higher weights improve CNNs but consistently degrade ViT-B/16 enough to lower mean transfer.
- Longer continuation mostly improves the DINO+MAE control too, leaving no matched JEPA-specific gain.
- The Phase 9/10 gain disappears under the repeated Phase 11 controls.

## Final Handoff Summary Format

When Colab finishes, report:

```text
Phase 11 Colab run completed.

Artifacts:
- ...

Best run:
- ...

Matched control comparison:
- ...

Per-victim result:
- ...

Recommendation:
- ...
```
