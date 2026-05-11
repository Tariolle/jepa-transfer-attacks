# JEPA Transfer Attacks

Negative result from testing whether I-JEPA features improve black-box adversarial transfer when added to a released dSVA generator.

## Hypothesis

I-JEPA representations might add a useful predictive-representation signal to dSVA-style generator training. The main test was whether continuing a released dSVA checkpoint with `DINO+MAE+I-JEPA` improves transfer beyond both:

- the untouched released dSVA checkpoint
- a matched `DINO+MAE` continuation baseline

## Method

The experiments start from the released dSVA checkpoint and continue training the generator on small ImageNet-style subsets. Evaluation uses transfer success across multiple victim architectures:

- `resnet50`
- `convnext_tiny`
- `vit_b_16`
- `efficientnet_b0`
- `swin_t`

Main comparisons:

| run | role |
| --- | --- |
| untouched released dSVA | do-nothing baseline |
| `DINO+MAE` continuation | matched continuation control |
| `DINO+MAE+I-JEPA` continuation | tested hypothesis |

## Findings

Early Imagenette-scale experiments looked promising. Plain `DINO+MAE` continuation sometimes degraded the released checkpoint, while `DINO+MAE+I-JEPA` improved over both the matched continuation control and the untouched checkpoint. That suggests I-JEPA may act as a stabilizing or more robust continuation signal under narrow-data continuation.

The full ImageNet-val result did not support the stronger hypothesis:

| run | mean transfer |
| --- | ---: |
| untouched released dSVA | 68.92% |
| `DINO+MAE` continuation | **70.77%** |
| `DINO+MAE+I-JEPA`, `jepa_weight=0.3` | 69.98% |

`DINO+MAE+I-JEPA` improves over the untouched checkpoint by **+1.06 points**, but it is **0.79 points worse** than the matched `DINO+MAE` continuation.

Per-victim gains of `DINO+MAE+I-JEPA` vs untouched dSVA on full ImageNet-val:

| victim | gain |
| --- | ---: |
| resnet50 | +0.49 pts |
| convnext_tiny | -2.43 pts |
| vit_b_16 | +2.73 pts |
| efficientnet_b0 | +1.23 pts |
| swin_t | +3.27 pts |

## Takeaway

The main hypothesis is **not supported**: adding I-JEPA to the best tested `DINO+MAE` continuation did not improve full ImageNet transfer.

The narrower signal is still interesting: I-JEPA continuation can improve the released checkpoint over doing nothing, and it appears more helpful for transformer-family victims than for ConvNeXt. This is best treated as a mixed/negative result with a possible robustness or architecture-specific transfer signal, not as a SOTA attack.

## Reproducibility

Dense result logs are in [RESULTS.md](RESULTS.md). The current project direction and caveats are in [ROADMAP.md](ROADMAP.md).

Main notebooks:

- [Phase 11/12 JEPA scaling](notebooks/phase11_jepa_scale.ipynb)
- [Phase 13 methodology-clean ablation](notebooks/phase13_methodology_clean.ipynb)
- [Phase 14 full ImageNet validation](notebooks/phase14_full_imagenet_validation.ipynb)

Prepare the Kaggle ImageNet localization archive with:

```powershell
python scripts/prepare_kaggle_imagenet_subset.py `
  --zip imagenet-object-localization-challenge.zip `
  --output-root imagenet_phase14 `
  --train-per-class 1 `
  --skip-existing
```
