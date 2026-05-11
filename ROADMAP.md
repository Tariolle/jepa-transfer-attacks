# Roadmap

## Archive Status

The main hypothesis is closed as a negative result.

Tested hypothesis: adding I-JEPA to a released dSVA generator continuation would improve black-box transfer beyond a matched DINO+MAE continuation baseline.

Final full ImageNet-val result:

| run | mean transfer |
| --- | ---: |
| untouched released dSVA | 68.92% |
| DINO+MAE continuation | **70.77%** |
| DINO+MAE+I-JEPA `0.3` | 69.98% |

Conclusion: I-JEPA continuation improves over the untouched checkpoint, but it does not beat the relevant DINO+MAE continuation control. The original claim is not supported.

## Remaining Signal

The only remaining signal is narrower:

- On Imagenette, I-JEPA appeared to stabilize narrow-data continuation when DINO+MAE degraded the released checkpoint.
- On full ImageNet-val, I-JEPA helped ViT-B/16, Swin-T, EfficientNet-B0, and ResNet-50 versus untouched dSVA, but hurt ConvNeXt-Tiny.

This may motivate a future per-architecture or complementarity study, but it is a new hypothesis, not a continuation of the original one.
