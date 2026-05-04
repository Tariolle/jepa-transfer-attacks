from __future__ import annotations

import argparse


def add_transfer_attack_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--momentum", type=float, default=0.0, help="MI-FGSM decay; 0 keeps vanilla PGD.")
    parser.add_argument(
        "--input-diversity-prob",
        type=float,
        default=0.0,
        help="Probability of random resize-and-pad before loss evaluation.",
    )
    parser.add_argument(
        "--input-diversity-min-resize",
        type=int,
        default=None,
        help="Minimum random resize size for input diversity; defaults to 90%% of image size.",
    )
    parser.add_argument(
        "--translation-kernel-size",
        type=int,
        default=1,
        help="Odd kernel size for TI-FGSM-style gradient smoothing; 1 disables it.",
    )
