"""Source-checkout entry point for the packaged isolated preprocessing worker."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gait_verification.backends.fastpose_worker import main

if __name__ == "__main__":
    main()
