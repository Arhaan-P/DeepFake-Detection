"""Extract only individually audited publisher artifacts; original bytes immutable."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gait_verification.backends.checkpoint_extract import extract_model_only


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", required=True)
    a = p.parse_args()
    if Path(a.manifest).exists():
        raise FileExistsError(a.manifest)
    result = extract_model_only(a.checkpoint, a.output, a.sha256)
    with open(a.manifest, "x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
