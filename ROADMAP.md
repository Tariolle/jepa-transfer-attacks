# Roadmap

## Current Framing

The main branch is no longer "separate JEPA attack vs dSVA". Use dSVA as the SOTA-grade generator scaffold, then test which representation objectives belong inside it:

- `DINO + MAE`: dSVA control
- `JEPA + MAE`: tests whether JEPA can replace DINO's semantic role
- `DINO + JEPA`: tests whether JEPA can complement or replace MAE
- `DINO + MAE + JEPA`: tests whether JEPA is a third complementary signal

dSVA is not a neutral benchmark: intermediate ViT facets, attention guidance, and DINO/MAE feature taps are partly tuned for DINO/MAE. A weak JEPA drop-in result would not prove JEPA is weak; it would only prove that this dSVA configuration does not favor it.

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

## Day-One Read

Promising: I-JEPA encoder features beat naive DINO features in early diagnostics, and normalized low-weight JEPA continuation slightly improved the released dSVA checkpoint on 1000 validation images.

Not proven: JEPA replacing DINO, JEPA-only sufficiency, predictor-based attacks, and full dSVA reproduction fidelity.

Not promising so far: heavy `DINO+MAE+JEPA` from-scratch weighting, JEPA-only generator training, and predictor-only objectives.
