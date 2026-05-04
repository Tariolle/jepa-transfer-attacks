# Results

All results below use the first 100 images from `imagenette2-320/val`, `epsilon=8/255`, `step_size=2/255`, and 10 PGD steps unless noted otherwise.

Generated CSV files are ignored by git; this file records the reusable summary.

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

## Phase 3a: Naive I-JEPA Feature Disruption

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

## Takeaways

- White-box supervised PGD works as expected.
- Vanilla cross-architecture supervised transfer is weak but nonzero.
- Naive DINOv2 feature disruption is weaker than supervised PGD in these runs.
- Naive I-JEPA feature disruption is the strongest diagnostic SSL result so far.
- The masked-context I-JEPA proxy is too weak to stand in for the true trained predictor objective.
- The full trained I-JEPA predictor objective transfers weakly in this first run and does not beat naive I-JEPA feature disruption.
