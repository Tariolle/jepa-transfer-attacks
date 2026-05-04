# Roadmap

## Next Step

Build the SOTA-grade SSL baseline path from [SOTA_BASELINE.md](SOTA_BASELINE.md): dSVA tracking, intermediate DINO facets, MAE hidden states, and DINO+MAE joint disruption.

## Phases

- ~~Phase 1: supervised PGD transfer baseline~~
- ~~Phase 2a: naive DINOv2 feature disruption~~
- ~~Phase 3a: naive I-JEPA encoder feature disruption~~
- ~~Phase 3b: masked-context predictor proxy~~
- ~~Phase 3c: full Meta I-JEPA predictor objective~~
- Phase 4: stronger SSL baseline, especially dSVA-style DINO/MAE feature attacks: retraining script added
- Phase 5: stronger transfer engine shared by all objectives: started with momentum, input diversity, and translation smoothing
- Phase 6: JEPA hybrids, especially `JEPA encoder + CE`: first strong-engine run did not beat CE-only
- Phase 7: JEPA generator trained with dSVA architecture: first 1000-image run is below JEPA PGD
- Phase 8: larger randomized/balanced evaluation set

## Decision Rule

Scale JEPA if it improves best transfer success, improves cross-family transfer, or adds complementary failures when combined with CE or DINO/MAE. Do not require the predictor-only objective to win by itself.
