# Results

All results below use the first 100 images from `imagenette2-320/val`, `epsilon=8/255`, `step_size=2/255`, and 10 PGD steps unless noted otherwise.

Generated CSV files are ignored by git; this file records the reusable summary.

These are vanilla diagnostic runs. They compare objectives under a simple PGD engine; they do not decide SOTA potential.

## Phase 1: Supervised Transfer Baseline

### ResNet-50 Surrogate

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 1.00% | 98.98% |
| convnext_tiny | 98.00% | 89.00% | 9.18% |
| vit_b_16 | 99.00% | 97.00% | 2.02% |

Mean non-surrogate transfer success: **5.60%**

### ViT-B/16 Surrogate

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| vit_b_16 | 99.00% | 0.00% | 100.00% |
| resnet50 | 98.00% | 93.00% | 5.10% |
| convnext_tiny | 98.00% | 87.00% | 11.22% |

Mean non-surrogate transfer success: **8.16%**

## Phase 2a: Naive DINOv2 Feature Disruption

### DINOv2 Small CLS

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 96.00% | 2.04% |
| convnext_tiny | 98.00% | 94.00% | 4.08% |
| vit_b_16 | 99.00% | 98.00% | 1.01% |

Mean transfer success: **2.38%**

### DINOv2 Small Patch Mean

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 94.00% | 4.08% |
| convnext_tiny | 98.00% | 94.00% | 4.08% |
| vit_b_16 | 99.00% | 98.00% | 1.01% |

Mean transfer success: **3.06%**

### DINOv2 Base Tokens

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 96.00% | 2.04% |
| convnext_tiny | 98.00% | 94.00% | 4.08% |
| vit_b_16 | 99.00% | 97.00% | 2.02% |

Mean transfer success: **2.71%**

## Phase 3a: Naive I-JEPA Encoder Feature Disruption

Model: `facebook/ijepa_vith14_1k`, token-level cosine feature disruption.

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 93.00% | 5.10% |
| convnext_tiny | 98.00% | 87.00% | 11.22% |
| vit_b_16 | 99.00% | 95.00% | 4.04% |

Mean transfer success: **6.79%**

## Phase 3b Proxy: Masked-Context I-JEPA Inconsistency

This used the Hugging Face I-JEPA encoder with masked target tokens. It is not the full trained Meta predictor objective.

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 98.00% | 0.00% |
| convnext_tiny | 98.00% | 98.00% | 0.00% |
| vit_b_16 | 99.00% | 98.00% | 1.01% |

Mean transfer success: **0.34%**

## Phase 3c: Full I-JEPA Predictor Objective

This used Meta's original full I-JEPA ViT-H/14 ImageNet-1K checkpoint with the trained predictor objective.

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 97.00% | 1.02% |
| convnext_tiny | 98.00% | 96.00% | 2.04% |
| vit_b_16 | 99.00% | 97.00% | 2.02% |

Mean transfer success: **1.69%**

## CE ResNet-50, Strong Engine

Supervised ResNet-50 surrogate with `momentum=1.0`, `input_diversity_prob=0.7`, and `translation_kernel_size=5`.

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 48.00% | 51.02% |
| convnext_tiny | 98.00% | 80.00% | 18.37% |
| vit_b_16 | 99.00% | 87.00% | 12.12% |

Mean non-surrogate transfer success: **15.24%**

## CE ViT-B/16, Strong Engine

Supervised ViT-B/16 surrogate with `momentum=1.0`, `input_diversity_prob=0.7`, and `translation_kernel_size=5`.

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| vit_b_16 | 99.00% | 37.00% | 62.63% |
| resnet50 | 98.00% | 85.00% | 13.27% |
| convnext_tiny | 98.00% | 85.00% | 13.27% |

Mean non-surrogate transfer success: **13.27%**

## DINOv2 Base Tokens, Strong Engine

DINOv2 base token disruption with `momentum=1.0`, `input_diversity_prob=0.7`, and `translation_kernel_size=5`.

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 88.00% | 10.20% |
| convnext_tiny | 98.00% | 91.00% | 7.14% |
| vit_b_16 | 99.00% | 93.00% | 6.06% |

Mean transfer success: **7.80%**

## dSVA-Style DINO K-Facet Proxy, Strong Engine

Intermediate DINOv2 base block-10 key-facet token disruption with `momentum=1.0`, `input_diversity_prob=0.7`, and `translation_kernel_size=5`. This is a proxy only; it does not include dSVA generator training, DINO+MAE joint training, or attention regularization.

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 88.00% | 10.20% |
| convnext_tiny | 98.00% | 91.00% | 7.14% |
| vit_b_16 | 99.00% | 94.00% | 5.05% |

Mean transfer success: **7.47%**

## Released dSVA Checkpoint

Released dSVA generator checkpoint from Hugging Face, evaluated directly with the paper's `epsilon=16/255` budget. This is not directly comparable to the `8/255` diagnostics above.

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 26.00% | 73.47% |
| convnext_tiny | 98.00% | 44.00% | 55.10% |
| vit_b_16 | 99.00% | 7.00% | 92.93% |

Mean transfer success: **73.83%**

## Retrained dSVA-Style Generator, Epsilon 16/255

Our dSVA-style DINO+MAE generator trained for one epoch on 1000 Imagenette train images with DINO key facet, MAE query facet, and `0.5/0.5` weighting.

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 40.00% | 59.18% |
| convnext_tiny | 98.00% | 75.00% | 23.47% |
| vit_b_16 | 99.00% | 82.00% | 17.17% |

Mean transfer success: **33.27%**

## Retrained dSVA + I-JEPA Generator, Epsilon 16/255

Same 1000-image dSVA-style DINO+MAE generator run with I-JEPA encoder token disruption added at `jepa_weight=0.25`.

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 73.00% | 25.51% |
| convnext_tiny | 98.00% | 92.00% | 6.12% |
| vit_b_16 | 99.00% | 97.00% | 2.02% |

Mean transfer success: **11.22%**

## dSVA Generator Fine-Tuned With I-JEPA, Epsilon 16/255

Started from the 1000-image dSVA-style DINO+MAE checkpoint, then fine-tuned for one epoch with I-JEPA encoder token disruption at `jepa_weight=0.05` and `lr=2e-5`.

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 23.00% | 76.53% |
| convnext_tiny | 98.00% | 64.00% | 34.69% |
| vit_b_16 | 99.00% | 74.00% | 25.25% |

Mean transfer success: **45.49%**

## Official dSVA + Normalized I-JEPA Fine-Tune, Epsilon 16/255

Started from the released dSVA checkpoint, continued for one epoch on 1000 Imagenette train images, and evaluated on 1000 Imagenette val images. The control also gets the same extra training steps; loss weights are normalized so adding JEPA does not inflate total loss scale.

| run | resnet50 | convnext_tiny | vit_b_16 | mean |
| --- | ---: | ---: | ---: | ---: |
| DINO+MAE control, `lr=5e-5` | 84.54% | 74.77% | 94.75% | 84.69% |
| DINO+MAE+I-JEPA, `jepa_weight=0.05`, `lr=5e-5` | 85.47% | 76.75% | 94.53% | 85.58% |

Matched gain: **+0.89 points**. Treat as a positive but small signal until full-set or repeated-seed validation.

## Phase 9: Official dSVA + I-JEPA Full-Val Repeated Seeds, Epsilon 16/255

Started from the released dSVA checkpoint and compared matched one-epoch continuations on 1000 Imagenette train images. Evaluation uses full Imagenette validation coverage via `eval_limit=5000`. Each seed compares the same extra training budget: DINO+MAE continuation control vs normalized DINO+MAE+I-JEPA with `jepa_weight=0.05` and `lr=5e-5`.

| seed | DINO+MAE control mean | DINO+MAE+I-JEPA mean | matched gain |
| --- | ---: | ---: | ---: |
| 0 | 68.96% | 69.90% | +0.94 pts |
| 1 | 66.92% | 67.19% | +0.27 pts |
| 2 | 67.33% | 68.09% | +0.77 pts |

Across seeds: control **67.73% +/- 1.08**, I-JEPA **68.39% +/- 1.38**, matched gain **+0.66 +/- 0.35 points**.

This confirms a small positive I-JEPA continuation signal against a proper extra-training control. The absolute full-val score is lower than the earlier 1000-image validation result, so the earlier `84.69% -> 85.58%` read should be treated as an optimistic subset estimate rather than the headline result.

## Phase 10: Objective Ablations and Per-Victim Analysis

Artifacts: `results/phase9_dsva_official_jepa_validation/` and `results/phase10_objective_ablations/` from the updated Colab notebook.

### Per-Victim Phase 9 Gains

Matched control vs DINO+MAE+I-JEPA (`jepa_weight=0.05`, `lr=5e-5`, 3 seeds, full-val `eval_limit=5000`):

| victim | control mean | JEPA mean | matched gain | gain std |
| --- | ---: | ---: | ---: | ---: |
| resnet50 | 70.93% | 72.41% | **+1.49%** | 0.11% |
| convnext_tiny | 54.00% | 55.29% | **+1.29%** | 0.35% |
| vit_b_16 | 78.25% | 77.52% | **-0.72%** | 0.41% |

The aggregate gain (+0.69 points) is driven by ResNet-50 and ConvNeXt-Tiny. ViT-B/16 drops slightly across all three seeds.

### Objective Ablation Table

Across-seed mean transfer success for each continuation objective:

| objectives | mean transfer | resnet50 | convnext_tiny | vit_b_16 |
| --- | ---: | ---: | ---: | ---: |
| DINO-only | 66.78% | 69.82% | 52.07% | 78.46% |
| DINO+JEPA | 67.91% | 71.84% | 53.42% | 78.46% |
| MAE-only | 63.37% | 64.29% | 50.73% | 75.09% |
| MAE+JEPA | 64.93% | 66.88% | 51.56% | 76.34% |
| JEPA-only | 64.99% | 67.96% | 50.67% | 76.34% |
| DINO+MAE (control) | 67.73% | 70.93% | 54.00% | 78.25% |
| DINO+MAE+JEPA | 68.41% | 72.41% | 55.29% | 77.52% |

### Interpretation

- **JEPA complements DINO**: DINO+JEPA (67.91%) is close to DINO+MAE (67.73%) and adds +1.13 points over DINO-only. This suggests JEPA can partially substitute for MAE when DINO is present.
- **JEPA also helps MAE**: MAE+JEPA (64.93%) adds +1.56 points over MAE-only, but the absolute level remains well below DINO-based combinations.
- **JEPA-only is insufficient**: 64.99% is below DINO-only and only slightly above MAE-only.
- **DINO+MAE+JEPA is still best**: 68.41% edges out DINO+JEPA (67.91%), so the third signal still adds marginal value.
- **Complementarity**: Cannot be measured from current eval artifacts because `run_dsva_checkpoint_attack.py` outputs only per-victim aggregates, not per-image correctness flags.

### Recommendation

The gain is small, but it is consistent across seeds and concentrated on CNN victims (ResNet-50, ConvNeXt-Tiny). DINO+JEPA nearly matches DINO+MAE, which is a stronger signal than the original DINO+MAE+JEPA delta alone. Do not scale JEPA as a MAE replacement yet, but continue exploring higher JEPA weights or longer continuation budgets before closing the line.

## Phase 11: JEPA Weight and Continuation-Budget Scaling

Completed locally with the Phase 11 notebook.

Entry point: `notebooks/phase11_jepa_scale.ipynb`.

Settings:

| test | configs | epochs | seeds |
| --- | --- | ---: | --- |
| JEPA weight sweep | `0.05`, `0.1`, `0.2` at `lr=5e-5` | 1 | `0,1,2` |
| longer continuation | `0.05` at `lr=5e-5` | 2, 3 | `0,1,2` |

All runs start from the released dSVA checkpoint, train on 1000 Imagenette train images, evaluate with full Imagenette validation coverage via `eval_limit=5000`, and compare against a matched DINO+MAE continuation control with the same seed, LR, epoch count, and training budget. Training uses normalized loss weights.

### Aggregate Results

| run | control mean | JEPA mean | matched gain |
| --- | ---: | ---: | ---: |
| `jepa_weight=0.05`, 1 epoch | 67.53% | 68.03% | +0.50 pts |
| `jepa_weight=0.1`, 1 epoch | 67.53% | 68.62% | **+1.09 pts** |
| `jepa_weight=0.2`, 1 epoch | 67.53% | 68.61% | **+1.09 pts** |
| `jepa_weight=0.05`, 2 epochs | 65.68% | 66.47% | +0.79 pts |
| `jepa_weight=0.05`, 3 epochs | 66.55% | 67.40% | +0.85 pts |

### Per-Victim Matched Gains

| run | resnet50 | convnext_tiny | vit_b_16 |
| --- | ---: | ---: | ---: |
| `jepa_weight=0.05`, 1 epoch | +1.19 pts | +1.07 pts | -0.75 pts |
| `jepa_weight=0.1`, 1 epoch | **+1.78 pts** | **+2.07 pts** | -0.57 pts |
| `jepa_weight=0.2`, 1 epoch | +1.51 pts | +1.93 pts | -0.17 pts |
| `jepa_weight=0.05`, 2 epochs | +0.77 pts | +0.94 pts | +0.66 pts |
| `jepa_weight=0.05`, 3 epochs | +0.41 pts | +0.18 pts | +1.95 pts |

### Interpretation

Phase 11 strengthens the continuation story. Higher JEPA weights roughly double the one-epoch matched gain, and the gain is positive for all three seeds. The best mean result is `jepa_weight=0.1`, while `jepa_weight=0.2` is effectively tied and has the smallest ViT-B/16 penalty.

The architectural pattern from Phase 10 still matters: `jepa_weight=0.1` is driven by ResNet-50 and ConvNeXt-Tiny improvements, while ViT-B/16 slightly drops. Longer `jepa_weight=0.05` continuation stays positive, but it does not beat the tuned one-epoch `0.1`/`0.2` runs.

Recommendation: continue to Phase 12. The paper-worthy claim is narrow but alive: I-JEPA encoder disruption appears to be a complementary predictive-representation objective for dSVA-style generator training, especially for CNN victim transfer. The next test should tune `jepa_weight=0.1` and `0.2` at longer continuation budgets and add broader victims before scaling to larger ImageNet validation.

## Phase 12: Tuned Continuation and Broader Victims

Completed locally with `notebooks/phase11_jepa_scale.ipynb`.

Important methodology note: the aggregate CSV combines the original three-victim tuned runs with the winner-only five-victim broader panel under the same config. The clean broader-panel read below separates only `phase12_seed*_broader_victims_ep2_jw0p2_*` summaries.

### Full-Val Untouched Released Baseline

The untouched released dSVA checkpoint was also evaluated on the full Imagenette validation protocol for the original three victims.

| victim | attack success |
| --- | ---: |
| resnet50 | 68.93% |
| convnext_tiny | 55.41% |
| vit_b_16 | 77.33% |

Mean transfer success: **67.22%**.

This changes the interpretation of continuation experiments: our plain DINO+MAE continuation can underperform the untouched checkpoint, so matched continuation gains alone are not sufficient evidence that JEPA improves the released model.

### Tuned Longer Continuation

Across seeds `0,1,2`, `jepa_weight=0.2` was the best tested setting. It was also the highest weight tested.

| run | control mean | JEPA mean | matched gain |
| --- | ---: | ---: | ---: |
| `jepa_weight=0.1`, 2 epochs | 65.64% | 67.21% | +1.57 pts |
| `jepa_weight=0.2`, 2 epochs | 65.64% | 68.02% | **+2.38 pts** |
| `jepa_weight=0.1`, 3 epochs | 66.56% | 67.45% | +0.89 pts |
| `jepa_weight=0.2`, 3 epochs | 66.56% | 68.15% | **+1.59 pts** |

The best tuned run is about **+0.80 to +0.93 points** above the untouched three-victim released checkpoint, but the plain continuation control is worse than untouched.

### Winner-Only Broader Victim Panel

The best tuned setting, `jepa_weight=0.2` for 2 epochs, was rerun with the original victims plus `efficientnet_b0` and `swin_t`.

| run | mean transfer | std |
| --- | ---: | ---: |
| DINO+MAE control, 2 epochs | 68.05% | 1.06 |
| DINO+MAE+I-JEPA `0.2`, 2 epochs | **70.25%** | 0.68 |

Matched gain: **+2.20 +/- 0.46 points**.

| victim | control | JEPA `0.2` | matched gain |
| --- | ---: | ---: | ---: |
| resnet50 | 70.81% | 71.60% | +0.79 pts |
| convnext_tiny | 52.20% | 54.45% | +2.25 pts |
| vit_b_16 | 73.88% | 77.39% | +3.51 pts |
| efficientnet_b0 | 94.81% | 94.93% | +0.12 pts |
| swin_t | 48.54% | 52.87% | +4.33 pts |

Seed-level matched gains were all positive: `+2.72`, `+1.84`, and `+2.04` points.

### Interpretation

Phase 12 is promising, but it exposed a baseline problem. JEPA clearly improves over our matched DINO+MAE continuation control, and the winner remains slightly above the untouched released checkpoint on the original three overlapping victims. However, because plain continuation can damage the released checkpoint, the main claim must be judged against an untouched released baseline on the exact same victim panel.

Next: evaluate untouched released dSVA on the same five broader victims, then run a JEPA-heavy weight ablation against that untouched baseline before considering full ImageNet.

## Phase 13: Methodology-Clean JEPA Weight Ablation

Completed locally with `notebooks/phase13_methodology_clean.ipynb`.

Phase 13 fixes the Phase 12 baseline concern by using the untouched released dSVA checkpoint as the primary baseline on the exact same five-victim panel. Matched DINO+MAE continuation is retained as a diagnostic, but not as the headline baseline.

Settings:

| item | value |
| --- | --- |
| train root | `imagenette2-320/train` |
| eval root | `imagenette2-320/val` |
| train limit | `1000` |
| eval limit | `5000` full Imagenette val coverage |
| epochs | `2` |
| LR | `5e-5` |
| seeds | `0,1,2` |
| victims | `resnet50`, `convnext_tiny`, `vit_b_16`, `efficientnet_b0`, `swin_t` |
| JEPA weights | `0.2`, `0.3`, `0.5`, `1.0` |

### Untouched Released dSVA Baseline

| victim | attack success |
| --- | ---: |
| resnet50 | 68.90% |
| convnext_tiny | 55.44% |
| vit_b_16 | 77.36% |
| efficientnet_b0 | 94.61% |
| swin_t | 50.89% |

Mean transfer success: **69.44%**.

### JEPA Weight Ablation

| run | mean transfer | gain vs untouched | gain vs DINO+MAE control |
| --- | ---: | ---: | ---: |
| DINO+MAE control | 68.04% | -1.41 pts | -- |
| DINO+MAE+I-JEPA `0.2` | 70.22% | +0.78 pts | +2.19 pts |
| DINO+MAE+I-JEPA `0.3` | **70.65%** | **+1.21 pts** | **+2.62 pts** |
| DINO+MAE+I-JEPA `0.5` | 70.64% | +1.20 pts | +2.61 pts |
| DINO+MAE+I-JEPA `1.0` | 69.87% | +0.43 pts | +1.84 pts |

The best setting is `jepa_weight=0.3`, with `0.5` effectively tied. The gain falls at `1.0`, so the current evidence supports moderate JEPA continuation rather than "more JEPA is always better."

### Best Run Per-Victim Gains vs Untouched

| victim | untouched | JEPA `0.3` | gain |
| --- | ---: | ---: | ---: |
| resnet50 | 68.90% | 71.59% | +2.68 pts |
| convnext_tiny | 55.44% | 54.89% | -0.55 pts |
| vit_b_16 | 77.36% | 78.38% | +1.02 pts |
| efficientnet_b0 | 94.61% | 95.15% | +0.54 pts |
| swin_t | 50.89% | 53.27% | +2.38 pts |

### Interpretation

Phase 13 is the first methodology-clean positive result. JEPA continuation beats the untouched released dSVA checkpoint on the same five-victim panel, not just a degraded matched continuation control. The effect is still modest, and ConvNeXt-Tiny drops slightly versus untouched, but the aggregate gain clears the pre-set `+1` point threshold.

Next: run a single-config larger validation, not another broad grid. Use untouched released dSVA, matched DINO+MAE continuation, and DINO+MAE+I-JEPA `jepa_weight=0.3`, 2 epochs on full ImageNet validation or the largest available ImageNet-style validation subset.

## Phase 14: Full ImageNet-Val Validation

Completed locally with `notebooks/phase14_full_imagenet_validation.ipynb`.

Settings: Kaggle ImageNet localization data prepared into `imagenet_phase14/`, 1000 balanced train images, full 50000-image validation, victims `resnet50`, `convnext_tiny`, `vit_b_16`, `efficientnet_b0`, `swin_t`, `epsilon=16/255`.

| run | mean transfer | conclusion |
| --- | ---: | --- |
| untouched released dSVA | 68.92% | primary baseline |
| DINO+MAE continuation | **70.77%** | strongest run |
| DINO+MAE+I-JEPA `0.3` | 69.98% | +1.06 pts vs untouched, -0.79 pts vs control |

Per-victim JEPA `0.3` gains vs untouched:

| victim | gain |
| --- | ---: |
| resnet50 | +0.49 pts |
| convnext_tiny | -2.43 pts |
| vit_b_16 | +2.73 pts |
| efficientnet_b0 | +1.23 pts |
| swin_t | +3.27 pts |

Interpretation: full ImageNet validation supports a workshop/short-paper claim that JEPA-guided continuation improves the released dSVA checkpoint over doing nothing. It does not support claiming JEPA is better than a well-performing DINO+MAE continuation recipe, because the matched control is higher on this larger validation.

## CE ResNet-50, Strong Engine, Epsilon 16/255

Same strong CE baseline as above, but with `epsilon=16/255` and `step_size=4/255`.

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 24.00% | 75.51% |
| convnext_tiny | 98.00% | 69.00% | 29.59% |
| vit_b_16 | 99.00% | 78.00% | 21.21% |

Mean non-surrogate transfer success: **25.40%**

## I-JEPA Encoder, Strong Engine, Epsilon 16/255

I-JEPA encoder token disruption with the same strong engine, `epsilon=16/255`, and `step_size=4/255`.

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 63.00% | 35.71% |
| convnext_tiny | 98.00% | 71.00% | 27.55% |
| vit_b_16 | 99.00% | 66.00% | 33.33% |

Mean transfer success: **32.20%**

## I-JEPA Encoder Generator, Epsilon 16/255

dSVA ResNet generator architecture trained for one epoch on 1000 Imagenette train images with I-JEPA encoder token disruption, batch size 1 with gradient accumulation.

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 61.00% | 37.76% |
| convnext_tiny | 98.00% | 80.00% | 18.37% |
| vit_b_16 | 99.00% | 95.00% | 4.04% |

Mean transfer success: **20.05%**

## I-JEPA Encoder, Strong Engine

I-JEPA encoder token disruption with `momentum=1.0`, `input_diversity_prob=0.7`, and `translation_kernel_size=5`.

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 88.00% | 10.20% |
| convnext_tiny | 98.00% | 89.00% | 9.18% |
| vit_b_16 | 99.00% | 88.00% | 11.11% |

Mean transfer success: **10.17%**

## Hybrid CE + I-JEPA Encoder, Strong Engine

ResNet-50 CE surrogate plus I-JEPA encoder token disruption. Uses `momentum=1.0`, `input_diversity_prob=0.7`, and `translation_kernel_size=5`.

| model | clean acc | adv acc | attack success |
| --- | ---: | ---: | ---: |
| resnet50 | 98.00% | 49.00% | 50.00% |
| convnext_tiny | 98.00% | 82.00% | 16.33% |
| vit_b_16 | 99.00% | 89.00% | 10.10% |

Mean non-surrogate transfer success: **13.21%**

## Takeaways

- The best JEPA signal so far is the I-JEPA encoder, not predictor inconsistency.
- Naive I-JEPA encoder disruption beat naive DINOv2 feature disruption, but generator-scale dSVA is the real comparison point.
- Heavy `DINO+MAE+JEPA` from scratch hurt transfer; low-weight normalized continuation is the promising path.
- Official dSVA + normalized I-JEPA continuation first improved repeated-seed full-val mean transfer from `67.73%` to `68.41%`.
- Phase 11 tuning improves the repeated-seed matched gain to about `+1.09` points at `jepa_weight=0.1` and `0.2`.
- Phase 12 improves the broader-panel matched continuation gain to about `+2.20` points at `jepa_weight=0.2`, but the matched DINO+MAE continuation control is not a faithful strong baseline because it can underperform the untouched released checkpoint.
- Phase 13 resolves that baseline concern on Imagenette: `jepa_weight=0.3` improves the untouched released dSVA five-victim mean from `69.44%` to `70.65%`.
- Phase 14 validates the effect on full ImageNet val: `jepa_weight=0.3` improves untouched dSVA from `68.92%` to `69.98%`, but is below the DINO+MAE continuation control at `70.77%`.
- The earlier 1000-image official dSVA continuation result (`84.69% -> 85.58%`) was directionally consistent but optimistic in absolute score.
- Per-victim: gain comes from ResNet-50 (+1.49) and ConvNeXt-Tiny (+1.29); ViT-B/16 drops slightly (-0.72).
- DINO+JEPA (67.91%) is close to DINO+MAE (67.73%), suggesting JEPA can partially substitute for MAE, but DINO+MAE+JEPA (68.41%) remains the best combination.
- JEPA-only is insufficient (64.99%). Do not replace DINO with JEPA.
- Next: diagnose why DINO+MAE continuation is now strong on ImageNet while JEPA helps transformer-family victims more than ConvNeXt; then write a short, carefully scoped result.
