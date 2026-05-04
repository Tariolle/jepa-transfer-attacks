# SOTA Baseline

Scope: untargeted black-box transfer for ImageNet-style classification under an `L_inf` budget, with CNN and ViT victims and no victim queries.

## Downloaded Assets

- Model: `external/models/dSVA/model.pth`
- Paper PDF: `external/papers/dsva_2506.21046.pdf`
- Paper source: `external/papers/dsva_2506.21046_source.tar`
- Official repo snapshot: `external/repos/dSVA`

## Target Reference

Use dSVA as the main SSL-feature SOTA reference:

- Paper: `Boosting Generative Adversarial Transferability with Self-supervised Vision Transformer Features`, ICCV 2025.
- Core idea: DINO captures contrastive/global structure, MAE captures masked-image/local texture, and the attack exploits intermediate ViT Q/K/V facets plus attention guidance.
- Reported setup: generator trained on ImageNet train, evaluated on the NeurIPS 2017 ImageNet-compatible 1000-image set.
- Status: the official GitHub currently exposes only README/license and says the implementation will be updated later, so full faithful reproduction depends on code or checkpoints becoming available.

## What Our Current Baselines Are Not

`DINOv2 final tokens + cosine PGD` and the current `DINO block-10 K-facet + cosine PGD` proxy are useful diagnostics, but they are not dSVA and should not be called SOTA DINO baselines.

## Reproduction Plan

1. Track official dSVA code/checkpoints.
2. Add the closest runnable proxy: intermediate DINO facets, MAE hidden states, and DINO+MAE joint feature disruption. Treat weak proxy results as evidence that the missing dSVA machinery matters, not as evidence against dSVA.
3. Add attention-guided weighting if the paper details are enough to implement it cleanly.
4. Add generator training only after the proxy is validated or official code is available.
5. Compare JEPA against the strongest reproduced SSL baseline under the same data, victims, epsilon, and compute budget.

## Decision Rule

JEPA is interesting if it beats the reproduced SSL baseline at the same epsilon, improves CE or DINO/MAE when combined with them, or transfers to complementary victims/samples that the SOTA baseline misses.

## Current Proxy Command

```powershell
python scripts/run_dsva_proxy_attack.py `
  --data-root .\imagenette2-320\val `
  --limit 100 `
  --dino-model vit_base_patch14_dinov2 `
  --dino-block-index 10 `
  --dino-facet k `
  --feature-mode tokens `
  --token-loss `
  --momentum 1.0 `
  --input-diversity-prob 0.7 `
  --translation-kernel-size 5 `
  --victims resnet50 convnext_tiny vit_b_16 `
  --device cuda
```

Add `--mae-model facebook/vit-mae-base` when that checkpoint is available locally or network access is allowed.

## Released Checkpoint Command

```powershell
python scripts/run_dsva_checkpoint_attack.py `
  --data-root .\imagenette2-320\val `
  --checkpoint external\models\dSVA\model.pth `
  --limit 100 `
  --epsilon 0.06274509803921569 `
  --victims resnet50 convnext_tiny vit_b_16 `
  --device cuda `
  --output-csv results\phase4_dsva_checkpoint.csv
```

For direct JEPA comparison to dSVA, use the same `epsilon=16/255` budget and report those results in a separate table from the older `8/255` diagnostics.

## dSVA Retraining Command

Closest paper-style retraining path: ResNet generator, DINO layer-10 key facet, MAE layer-10 query facet, lambda-style `0.5/0.5` weighting, Adam `2e-4`, and `epsilon=16/255`.

The first 1000-image Imagenette retraining run reached `33.27%` mean transfer, close to I-JEPA PGD at `32.20%` but far below the released ImageNet-trained dSVA checkpoint at `73.83%`.

```powershell
python scripts/train_dsva_generator.py `
  --data-root .\imagenette2-320\train `
  --limit 1000 `
  --batch-size 1 `
  --grad-accum-steps 8 `
  --epochs 1 `
  --token-loss `
  --amp `
  --epsilon 0.06274509803921569 `
  --device cuda `
  --output-checkpoint results\dsva_retrained_eps16.pth
```

If Hugging Face DINO is unavailable locally, use the timm DINOv2 fallback:

```powershell
python scripts/train_dsva_generator.py `
  --data-root .\imagenette2-320\train `
  --limit 1000 `
  --batch-size 1 `
  --grad-accum-steps 8 `
  --epochs 1 `
  --dino-backend timm `
  --dino-model vit_base_patch14_dinov2 `
  --disable-mae `
  --token-loss `
  --amp `
  --epsilon 0.06274509803921569 `
  --device cuda `
  --output-checkpoint results\dsva_dino_only_retrained_eps16.pth
```

## dSVA + JEPA Command

Compare this directly against `results\dsva_retrained_eps16.pth`. Start with a light JEPA weight because the first CE+JEPA hybrid suggested the JEPA loss can conflict when over-weighted.

```powershell
python scripts/train_dsva_generator.py `
  --data-root .\imagenette2-320\train `
  --limit 1000 `
  --batch-size 1 `
  --grad-accum-steps 8 `
  --epochs 1 `
  --token-loss `
  --amp `
  --enable-jepa `
  --jepa-weight 0.25 `
  --epsilon 0.06274509803921569 `
  --device cuda `
  --output-checkpoint results\dsva_jepa_retrained_eps16.pth
```

On an 8 GB GPU, `DINO+MAE+JEPA` may be too large. If it OOMs, first test `DINO+JEPA`:

```powershell
python scripts/train_dsva_generator.py `
  --data-root .\imagenette2-320\train `
  --limit 1000 `
  --batch-size 1 `
  --grad-accum-steps 8 `
  --epochs 1 `
  --disable-mae `
  --token-loss `
  --amp `
  --enable-jepa `
  --jepa-weight 0.25 `
  --epsilon 0.06274509803921569 `
  --device cuda `
  --output-checkpoint results\dsva_dino_jepa_retrained_eps16.pth
```

## JEPA Generator Command

This trains the same ResNet generator architecture with an I-JEPA encoder-disruption loss. Start with a small `--limit` smoke run, then scale to a larger training split.

The first 1000-image Imagenette run underperformed I-JEPA PGD, so the next upgrades should add dSVA-like training scale, joint DINO/MAE/JEPA objectives, and attention or intermediate-facet guidance.

```powershell
python scripts/train_jepa_generator.py `
  --data-root .\imagenette2-320\train `
  --limit 1000 `
  --batch-size 8 `
  --epochs 1 `
  --token-loss `
  --epsilon 0.06274509803921569 `
  --device cuda `
  --output-checkpoint results\jepa_generator_eps16.pth
```

Evaluate the trained generator with:

```powershell
python scripts/run_dsva_checkpoint_attack.py `
  --data-root .\imagenette2-320\val `
  --checkpoint results\jepa_generator_eps16.pth `
  --output-mode scaled-delta `
  --limit 100 `
  --epsilon 0.06274509803921569 `
  --victims resnet50 convnext_tiny vit_b_16 `
  --device cuda `
  --output-csv results\jepa_generator_eps16_eval.csv
```
