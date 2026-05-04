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

- The best JEPA result so far is encoder feature disruption, not predictor inconsistency.
- I-JEPA encoder disruption beats the matched DINOv2 base token baseline in both vanilla and strong-engine diagnostics.
- The first DINO key-facet proxy does not improve over final-token DINO; it should not be treated as a dSVA reproduction.
- The released dSVA checkpoint is dramatically stronger at its paper budget of `16/255`; do not compare it directly to `8/255` runs.
- The local dSVA-style retraining run reaches I-JEPA PGD territory but remains far below the released ImageNet-trained dSVA checkpoint.
- At `16/255`, I-JEPA encoder PGD beats CE ResNet-50 on mean transfer but remains far below the released dSVA generator.
- The first I-JEPA generator is weaker than I-JEPA PGD, so generator architecture alone is not enough.
- The strong-engine CE baseline is now the best result; the first CE + I-JEPA hybrid did not beat CE-only.
- I-JEPA encoder disruption does improve with the stronger engine, but still trails CE-only here.
- ResNet-50 CE remains the best strong-engine supervised surrogate on the current mean-transfer metric.
- The full predictor objective was weak in its first vanilla setup, so it should be treated as an ablation, not the whole project.
- Next comparisons must give CE, DINO/MAE, JEPA encoder, JEPA predictor, and hybrids the same stronger attack engine before making scale-up decisions.
