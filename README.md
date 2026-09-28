# JEPA Transfer Attacks

**Closed research prototype. Preserved for reference; no longer maintained.**

## Question

Do image perturbations that disrupt I-JEPA's context-to-target predictions transfer to other image classifiers better than ordinary feature attacks?

The project later shifted to adding I-JEPA **encoder-feature disruption** to a released dSVA generator. The final experiment therefore tested a different, narrower hypothesis—not the predictive objective itself.

## Recorded result

Evaluation on 50,000 ImageNet validation images, five classifiers, and three training seeds; continuation used 1,000 training images and `epsilon=16/255`.

| Continuation objective | Mean transfer success |
| --- | ---: |
| DINO+MAE control | 70.77% |
| DINO+MAE+I-JEPA (`jepa_weight=0.3`) | 69.98% |

Adding I-JEPA did not improve the tested recipe: **−0.79 percentage points** against the matched control.

## Limitations

- [MAE feature extraction](src/models/ssl_encoders.py) does not enforce clean/adversarial token correspondence.
- [Early subset selection](src/data/imagenet_subset.py) takes class-ordered prefixes rather than representative samples.
- Untouched and continued generators use different output interpretations, confounding comparisons against the untouched checkpoint.

These issues remain unresolved. **The result does not establish that the original predictive-attack idea fails—or that correcting the implementation would make it succeed.**

[Results](RESULTS.md) and the [final notebook](notebooks/phase14_full_imagenet_validation.ipynb) preserve the experimental record. The caveats above supersede earlier interpretations.
