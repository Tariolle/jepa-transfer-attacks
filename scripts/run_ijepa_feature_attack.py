from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ssl_feature_attack import main


if __name__ == "__main__":
    if "--encoder-backend" not in sys.argv:
        sys.argv.extend(["--encoder-backend", "ijepa"])
    if "--ssl-model" not in sys.argv:
        sys.argv.extend(["--ssl-model", "facebook/ijepa_vith14_1k"])
    main()
