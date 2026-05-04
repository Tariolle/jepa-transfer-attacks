# Roadmap

## Next Step

Validate the normalized official dSVA + low-weight I-JEPA continuation signal on a larger eval set or repeated seeds. The current 1000-image check is positive but small: `84.69% -> 85.58%`.

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
- Phase 8: official dSVA + normalized I-JEPA continuation: initial 1000-image eval gives a small positive matched gain
- Phase 9: larger randomized/balanced evaluation set

## Decision Rule

Scale JEPA if the official dSVA continuation gain survives larger validation, improves cross-family transfer, or adds complementary failures. Do not require the predictor-only objective to win by itself.
