# Roadmap

## Current Framing

The main branch is no longer "separate JEPA attack vs dSVA". Use dSVA as the SOTA-grade generator scaffold, then test which representation objectives belong inside it:

- `DINO + MAE`: dSVA control
- `JEPA + MAE`: tests whether JEPA can replace DINO's semantic role
- `DINO + JEPA`: tests whether JEPA can complement or replace MAE
- `DINO + MAE + JEPA`: tests whether JEPA is a third complementary signal

dSVA is not a neutral benchmark: intermediate ViT facets, attention guidance, and DINO/MAE feature taps are partly tuned for DINO/MAE. A weak JEPA drop-in result would not prove JEPA is weak; it would only prove that this dSVA configuration does not favor it.

## Next Step

Run Phase 12 as a paper-kill experiment. Phase 11 showed that increasing the JEPA continuation weight from `0.05` to `0.1`/`0.2` roughly doubles the matched mean gain, with strongest improvements on CNN victims. The next question is whether tuned JEPA weights survive longer continuation and broader victim coverage, or whether the gain is an Imagenette/victim-set artifact.

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
- ~~Phase 8: official dSVA + normalized I-JEPA continuation: initial 1000-image eval gives a small positive matched gain~~
- ~~Phase 9: larger randomized/balanced evaluation set: full-val repeated-seed gain is positive but small, `67.73% -> 68.41%`~~
- ~~Phase 10: per-architecture and objective ablations~~
- ~~Phase 11: scale JEPA weight or continuation budget to test if the CNN gain can be enlarged: `jepa_weight=0.1` and `0.2` both improve matched mean transfer by about +1.09 points~~
- Phase 12: validate tuned JEPA continuation with longer budgets and broader victims

## Decision Rule

Scale JEPA if tuned continuation (`jepa_weight=0.1` or `0.2`) survives longer validation, improves cross-family transfer on additional CNN/Transformer victims, or adds complementary per-image failures. Do not require the predictor-only objective to win by itself.

## Phase 12 Protocol

Use the local Phase 11 notebook as the runner and extend the matrix rather than repeating completed baselines:

| experiment | configs | epochs | decision target |
| --- | --- | ---: | --- |
| tuned longer continuation | `0.1:0.00005`, `0.2:0.00005` | 2 | test whether Phase 11 gains scale |
| tuned longer continuation | `0.1:0.00005`, `0.2:0.00005` | 3 | test over-continuation risk |
| broader victims | best tuned config | 1-2 | test whether CNN gains generalize |

Continue toward a paper if the tuned JEPA run preserves at least a +1 point matched mean gain, keeps CNN gains around +2 points or better, avoids a mean-erasing Transformer penalty, and shows complementary per-image failures. Downscope if the gain vanishes under longer controls, broader victims, or per-image analysis.

## Day-One Read

Promising: I-JEPA encoder features beat naive DINO features in early diagnostics, normalized low-weight JEPA continuation slightly improved the released dSVA checkpoint, and Phase 11 tuning grew the matched full-val repeated-seed gain to about +1.09 points at `jepa_weight=0.1`/`0.2`.

Not proven: JEPA replacing DINO, JEPA-only sufficiency, predictor-based attacks, and full dSVA reproduction fidelity.

Not promising so far: heavy `DINO+MAE+JEPA` from-scratch weighting, JEPA-only generator training, and predictor-only objectives.
