# Roadmap

## Current Framing

The main branch is no longer "separate JEPA attack vs dSVA". Use dSVA as the SOTA-grade generator scaffold, then test which representation objectives belong inside it:

- `DINO + MAE`: dSVA control
- `JEPA + MAE`: tests whether JEPA can replace DINO's semantic role
- `DINO + JEPA`: tests whether JEPA can complement or replace MAE
- `DINO + MAE + JEPA`: tests whether JEPA is a third complementary signal

dSVA is not a neutral benchmark: intermediate ViT facets, attention guidance, and DINO/MAE feature taps are partly tuned for DINO/MAE. A weak JEPA drop-in result would not prove JEPA is weak; it would only prove that this dSVA configuration does not favor it.

## Next Step

Write up the result as a narrow workshop/short-paper candidate and run one diagnostic pass. Full ImageNet validation shows `jepa_weight=0.3` improves the untouched released dSVA checkpoint from `68.92%` to `69.98%`, but the matched DINO+MAE continuation control is stronger at `70.77%`. The claim should be about JEPA-guided continuation improving a released checkpoint, not beating all continuation recipes.

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
- ~~Phase 12: validate tuned JEPA continuation with longer budgets and broader victims: `jepa_weight=0.2`, 2 epochs gave a +2.20 point matched broader-panel gain, but exposed a continuation-baseline flaw~~
- ~~Phase 13: methodology-clean untouched baseline and JEPA-heavy weight ablation: `jepa_weight=0.3` improved untouched dSVA by +1.21 points on five victims~~
- ~~Phase 14: full-ImageNet validation: `jepa_weight=0.3` beats untouched dSVA by +1.06 points, but trails DINO+MAE continuation by -0.79 points~~

## Decision Rule

Proceed toward a short writeup if the claim stays narrow: JEPA-guided continuation improves a released dSVA checkpoint over the untouched baseline on full ImageNet val. Do not claim SOTA, do not claim JEPA beats DINO+MAE continuation, and keep the matched control as an important caveat.

## Immediate Experiments

Run only diagnostics that clarify the story:

- Per-victim and per-image overlap between untouched, DINO+MAE continuation, and JEPA continuation.
- A hybrid/control sanity check: DINO+MAE continuation plus lower JEPA weights around `0.2` to `0.5` only if compute is cheap.
- A reproduction check of DINO+MAE continuation settings, because the control is strong on ImageNet but was unstable on Imagenette.

## Day-One Read

Promising: I-JEPA encoder features beat naive DINO features in early diagnostics, Phase 13 showed a methodology-clean +1.21 point gain over untouched released dSVA on Imagenette, and Phase 14 preserved a +1.06 point gain over untouched dSVA on full ImageNet val.

Not proven: JEPA replacing DINO, JEPA-only sufficiency, predictor-based attacks, full dSVA reproduction fidelity, and JEPA beating DINO+MAE continuation.

Not promising so far: heavy `DINO+MAE+JEPA` from-scratch weighting, JEPA-only generator training, predictor-only objectives, and overstating matched-control gains when the untouched baseline is the fair comparison.
