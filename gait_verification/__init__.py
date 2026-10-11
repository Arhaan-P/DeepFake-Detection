"""V2 claimed-identity verification; isolated from historical E0 code."""

from .contracts import Status, VerificationResult
from .verifier import ClaimVerifier

__all__ = ["ClaimVerifier", "Status", "VerificationResult"]
