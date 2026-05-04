# JEPA Transfer Attacks

Research prototype for testing whether JEPA-style predictive latent inconsistency produces adversarial perturbations that transfer better across architectures than standard supervised or SSL feature attacks.

## Research Question

Do perturbations that break latent context-to-target prediction transfer better to black-box CNN/ViT classifiers than perturbations that attack:

- a supervised classifier boundary;
- ordinary SSL features from models such as DINO/DINOv2 or MAE;
- I-JEPA features without using the predictive objective?

The key comparison is not JEPA versus supervised attacks alone. JEPA must beat or complement strong SSL feature-disruption baselines.

## Hypothesis

Supervised attacks can overfit to one model's decision boundary. SSL feature attacks may transfer better because they disrupt reusable visual representations. I-JEPA is interesting only if its predictive context-to-target structure adds transferability beyond plain feature disruption.

## Roadmap

See [ROADMAP.md](ROADMAP.md).

## Metrics

For each attack and victim model, report clean accuracy, adversarial accuracy, accuracy drop, attack success rate on originally correct samples, and mean transfer success across non-surrogate victims.

## Go / No-Go

The project is promising if I-JEPA predictive inconsistency beats plain I-JEPA feature disruption, beats or complements DINO/MAE, or improves CNN-to-ViT transfer. It is weak if DINO/MAE dominate, predictive JEPA behaves like plain feature disruption, or the perturbations do not affect downstream classifier decisions.

## Status

Phase 1 established the supervised PGD transfer baseline. Phase 2a adds naive DINOv2 feature-disruption attacks as a diagnostic SSL baseline. Phase 3a adds naive I-JEPA feature disruption. The roadmap separates diagnostic feature disruption from stronger transfer baselines.

Run the current SSL baseline:

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

Run the current I-JEPA diagnostic:

```powershell
python scripts/run_ijepa_feature_attack.py `
  --data-root D:\path\to\imagenette2-320\val `
  --limit 100 `
  --ssl-model facebook/ijepa_vith14_1k `
  --feature-mode tokens `
  --token-loss `
  --victims resnet50 convnext_tiny vit_b_16 `
  --device cuda
```
