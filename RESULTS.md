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
- Official dSVA + normalized I-JEPA continuation improves repeated-seed full-val mean transfer from `67.73%` to `68.39%`.
- The gain is positive for all three seeds and compared against a matched extra-training control, but it is small.
- The earlier 1000-image official dSVA continuation result (`84.69% -> 85.58%`) was directionally consistent but optimistic in absolute score.
- Next: inspect per-victim gains and ablate whether JEPA complements DINO, MAE, or only their combination.
