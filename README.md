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

## Results

See [RESULTS.md](RESULTS.md) for the current 100-image Imagenette baseline summaries.

## Metrics

For each attack and victim model, report clean accuracy, adversarial accuracy, accuracy drop, attack success rate on originally correct samples, and mean transfer success across non-surrogate victims.

## Go / No-Go

The project is promising if I-JEPA predictive inconsistency beats plain I-JEPA feature disruption, beats or complements DINO/MAE, or improves CNN-to-ViT transfer. It is weak if DINO/MAE dominate, predictive JEPA behaves like plain feature disruption, or the perturbations do not affect downstream classifier decisions.

## Status

Phase 1 established the supervised PGD transfer baseline. Phase 2a adds naive DINOv2 feature-disruption attacks as a diagnostic SSL baseline. Phase 3a adds naive I-JEPA feature disruption. Phase 3b-proxy tests masked-context inconsistency with the available Hugging Face I-JEPA encoder, but it is not the full trained Meta predictor objective.

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

Run the masked-context I-JEPA proxy:

```powershell
python scripts/run_ijepa_predictive_attack.py `
  --data-root D:\path\to\imagenette2-320\val `
  --limit 100 `
  --ijepa-model facebook/ijepa_vith14_1k `
  --target-block-size 8 `
  --victims resnet50 convnext_tiny vit_b_16 `
  --device cuda
```

Run the full Meta I-JEPA predictor objective:

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

The true predictor objective requires Meta's original full I-JEPA checkpoint, which includes a trained predictor and is about 10.36 GB for the ViT-H/14 ImageNet-1K checkpoint.
