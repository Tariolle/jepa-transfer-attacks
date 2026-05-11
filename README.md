# JEPA Transfer Attacks

JEPA-guided continuation for improving black-box transfer of released dSVA adversarial generators.

The project is not predictor-only. We test JEPA as an attack family:

- `JEPA-Encoder`: disrupt I-JEPA encoder features.
- `JEPA-Predictor`: break context-to-target prediction.
- `JEPA-Hybrid`: combine encoder, predictor, SSL, or supervised losses.

The practical goal is honest transferability gains from JEPA-guided continuation of strong generator checkpoints. Ablations decide whether the useful signal comes from the encoder, the predictor, or their combination.

## Fair Comparison

Separate the attack objective from the attack engine.

| Layer | Examples |
| --- | --- |
| Objective | CE, DINO/MAE features, JEPA encoder, JEPA predictor, hybrids |
| Engine | PGD, momentum, input diversity, translation/scale tricks, ensembles, generators |

JEPA should be compared against DINO/MAE and supervised baselines under the same budget and attack engine. A vanilla JEPA run should not be judged against heavily optimized baselines.

## Current Status

- Naive I-JEPA encoder disruption beat naive DINOv2 feature disruption in early diagnostics.
- Predictor-style JEPA objectives were weak in first vanilla setups.
- The current best signal is I-JEPA encoder disruption added to released dSVA continuation, judged against the untouched released checkpoint.

Headline result: full ImageNet-val validation improved the untouched released dSVA checkpoint from **68.92%** to **69.98%** mean transfer across five victims at `jepa_weight=0.3`, a **+1.06 point** gain. The matched DINO+MAE continuation reached **70.77%**, so the honest conclusion is that JEPA improves the released checkpoint, but does not yet beat the best continuation control.

The project remains promising, but the claim is narrow: JEPA is not a standalone replacement for DINO/MAE. It appears to be a complementary predictive-representation objective or stabilizing continuation objective for released dSVA-style generators.

See [RESULTS.md](RESULTS.md) and [ROADMAP.md](ROADMAP.md).

See [SOTA_BASELINE.md](SOTA_BASELINE.md) for the external baseline target. The current DINOv2 token attack is a diagnostic, not a SOTA-grade dSVA reproduction.

Notebook entry points:

- Phase 9/10 validation and objective ablations: [notebooks/phase9_dsva_jepa_validation_colab.ipynb](notebooks/phase9_dsva_jepa_validation_colab.ipynb)
- Phase 11/12 JEPA weight, continuation-budget, and broader-victim scaling, local GPU runner: [notebooks/phase11_jepa_scale.ipynb](notebooks/phase11_jepa_scale.ipynb)
- Phase 13 methodology-clean baseline and JEPA-heavy ablation, local GPU runner: [notebooks/phase13_methodology_clean.ipynb](notebooks/phase13_methodology_clean.ipynb)
- Phase 14 full-ImageNet validation of the best Phase 13 setting, local GPU runner: [notebooks/phase14_full_imagenet_validation.ipynb](notebooks/phase14_full_imagenet_validation.ipynb)

The Kaggle ImageNet localization zip can be prepared with:

```powershell
python scripts/prepare_kaggle_imagenet_subset.py `
  --zip imagenet-object-localization-challenge.zip `
  --output-root imagenet_phase14 `
  --train-per-class 1 `
  --skip-existing
```

## Run Examples

Add the same transfer-engine flags to any attack script when comparing stronger runs:

```powershell
--momentum 1.0 --input-diversity-prob 0.7 --translation-kernel-size 5
```

Naive DINO/SSL feature baseline:

```powershell
python scripts/run_ssl_feature_attack.py `
  --data-root D:\path\to\imagenette2-320\val `
  --limit 100 `
  --ssl-model vit_base_patch14_dinov2 `
  --feature-mode tokens `
  --token-loss `
  --victims resnet50 convnext_tiny vit_b_16 `
  --device cuda
```

Naive I-JEPA encoder attack:

```powershell
python scripts/run_ijepa_feature_attack.py `
  --data-root D:\path\to\imagenette2-320\val `
  --limit 100 `
  --feature-mode tokens `
  --token-loss `
  --victims resnet50 convnext_tiny vit_b_16 `
  --device cuda
```

Full Meta I-JEPA predictor attack:

```powershell
python scripts/run_ijepa_full_predictor_attack.py `
  --data-root D:\path\to\imagenette2-320\val `
  --ijepa-repo D:\path\to\ijepa `
  --checkpoint D:\path\to\IN1K-vit.h.14-300e.pth.tar `
  --limit 100 `
  --target-block-size 7 `
  --victims resnet50 convnext_tiny vit_b_16 `
  --device cuda
```

Hybrid CE + I-JEPA encoder attack:

```powershell
python scripts/run_hybrid_ce_ijepa_attack.py `
  --data-root D:\path\to\imagenette2-320\val `
  --limit 100 `
  --surrogate resnet50 `
  --token-loss `
  --victims resnet50 convnext_tiny vit_b_16 `
  --device cuda
```
