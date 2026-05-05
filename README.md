# JEPA Transfer Attacks

Research prototype for testing whether JEPA-trained representations can improve black-box adversarial transfer across CNN and ViT classifiers.

The project is not predictor-only. We test JEPA as an attack family:

- `JEPA-Encoder`: disrupt I-JEPA encoder features.
- `JEPA-Predictor`: break context-to-target prediction.
- `JEPA-Hybrid`: combine encoder, predictor, SSL, or supervised losses.

The practical goal is SOTA transferability. Ablations decide whether the useful signal comes from the encoder, the predictor, or their combination.

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
- The current best signal is low-weight I-JEPA added to an official dSVA continuation.

Headline result: matched full-val repeated-seed dSVA continuation improved from **67.73%** to **68.39%** mean transfer. The gain is small but positive for all three seeds, and it is compared against the same extra training budget without JEPA.

See [RESULTS.md](RESULTS.md) and [ROADMAP.md](ROADMAP.md).

See [SOTA_BASELINE.md](SOTA_BASELINE.md) for the external baseline target. The current DINOv2 token attack is a diagnostic, not a SOTA-grade dSVA reproduction.

Colab entry point for Phase 9 validation: [notebooks/phase9_dsva_jepa_validation_colab.ipynb](notebooks/phase9_dsva_jepa_validation_colab.ipynb).

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
