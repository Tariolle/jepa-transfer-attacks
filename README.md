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

## Phase 1

Run the supervised baseline on a labeled ImageNet-style folder:

```powershell
python scripts/run_supervised_baseline.py `
  --data-root D:\path\to\imagenet_like_subset `
  --limit 100 `
  --surrogate resnet50 `
  --victims resnet50 convnext_tiny vit_b_16 `
  --device cuda
```

Images should be stored in class folders. Folder names can be ImageNet class ids such as `207`, ImageNette WordNet ids such as `n02102040`, or labels supplied through `--class-map`.

The attack uses an `L_inf` budget of `8/255`, step size `2/255`, and 10 PGD steps by default.
Pretrained weights are cached under `.torch_cache/` by default.
