"""ClaimRequest -> frozen E0 score; no actual query identity/label enters the model."""

import numpy as np

from .contracts import Status, valid_claim
from .dataset_processing import final_test_denied, source_issue
from .e0 import validate_features
from .pipeline import CacheIntegrityError
from .splits import validate_split


class E0TrialScorer:
    """Resolve exactly the provided query and claimed subject's reference originals.

    Only validated managed feature artifacts can be scored. Preparation is explicit
    through process_e0; missing work is never silently substituted with legacy caches.
    No policy parameter exists: successful inference is always POLICY_NOT_READY.
    """

    def __init__(self, registry, split, data_root, features, backend_factory):
        validate_split(split, registry.subjects())
        self.registry, self.split, self.data_root = registry, split, data_root
        self.features, self.backend_factory = features, backend_factory
        self.backend = None
        self.recipes, self.runtime, self.lookups = set(), {}, []

    def __call__(self, request):
        quality = False
        try:
            r = request
            if not valid_claim(r.claimed_subject_id):
                return Status.INVALID_INPUT, "INVALID_CLAIM", None, None, False
            query = self.registry.asset(r.query_asset_id)
            if final_test_denied(query, self.split):
                return Status.INVALID_INPUT, "LOCKED_FINAL_DENIED", None, None, False
            self.registry.recording(
                query.asset_id, self.split["assignments"][query.subject_id]
            )
            if (query.original_take_id, query.variant_id) != (
                r.query_original_take_id,
                r.query_variant_id,
            ):
                return (
                    Status.INVALID_INPUT,
                    "QUERY_PROVENANCE_MISMATCH",
                    None,
                    None,
                    False,
                )
            wanted = tuple(r.reference_original_take_ids)
            if not wanted or len(set(wanted)) != len(wanted):
                return Status.INVALID_INPUT, "INVALID_REFERENCE_SET", None, None, False
            if query.original_take_id in wanted:
                return Status.INVALID_INPUT, "SHARED_ORIGINAL", None, None, False
            if not query.is_original:
                return Status.INVALID_INPUT, "E0_ORIGINALS_ONLY", None, None, False
            self.lookups.append(r.claimed_subject_id)
            claimed = {
                a.original_take_id: a
                for a in self.registry.originals(r.claimed_subject_id)
            }
            if not claimed:
                return Status.CLAIM_NOT_ENROLLED, "NO_REFERENCES", None, None, False
            if not set(wanted) <= claimed.keys():
                return (
                    Status.CLAIM_NOT_ENROLLED,
                    "REFERENCE_NOT_IN_CLAIM",
                    None,
                    None,
                    False,
                )
            refs = [claimed[t] for t in sorted(wanted)]
            for ref in refs:
                if final_test_denied(ref, self.split):
                    return (
                        Status.INVALID_INPUT,
                        "LOCKED_FINAL_DENIED",
                        None,
                        None,
                        False,
                    )
                role = self.split["assignments"][r.claimed_subject_id]
                self.registry.recording(ref.asset_id, role)
                if role != self.split["assignments"][query.subject_id]:
                    return (
                        Status.INVALID_INPUT,
                        "CROSS_ROLE_REFERENCE",
                        None,
                        None,
                        False,
                    )
                if self.registry.lineage_ids(
                    query.asset_id
                ) & self.registry.lineage_ids(ref.asset_id):
                    return Status.INVALID_INPUT, "SHARED_ORIGINAL", None, None, False
            for asset in [query, *refs]:
                issue = source_issue(self.registry, asset, self.data_root)
                if issue:
                    return Status.INVALID_INPUT, issue, None, None, False
            q = self.features.get(query)
            if q is None:
                return (
                    Status.PROCESSING_ERROR,
                    "QUERY_FEATURE_MISSING",
                    None,
                    None,
                    False,
                )
            self.runtime[query.asset_id] = (q["runtime_sec"], None)
            if q["status"] != "FEATURED":
                return Status.INSUFFICIENT_GAIT, q["reason"], None, None, False
            quality = True
            values = []
            for ref in refs:
                f = self.features.get(ref)
                if f is None:
                    return (
                        Status.CLAIM_NOT_ENROLLED,
                        "REFERENCE_FEATURE_MISSING",
                        None,
                        None,
                        quality,
                    )
                if f["status"] != "FEATURED":
                    return (
                        Status.INSUFFICIENT_GAIT,
                        "REFERENCE_INSUFFICIENT_GAIT",
                        None,
                        None,
                        quality,
                    )
                values.append(validate_features(f["values"]))
            if self.backend is None:
                self.backend = self.backend_factory()
            score, _ = self.backend.score(validate_features(q["values"]), values)
            if not isinstance(score, (int, float)) or not np.isfinite(score):
                raise ValueError("NONFINITE_E0_SCORE")
            # Recheck immutable inputs after computation, including cached paths.
            if any(
                source_issue(self.registry, a, self.data_root) for a in [query, *refs]
            ):
                return (
                    Status.INVALID_INPUT,
                    "SOURCE_CHANGED_DURING_SCORE",
                    None,
                    None,
                    quality,
                )
            self.recipes.add(
                self.backend.model_id + ":features=" + self.features.recipe_id
            )
            return (
                Status.POLICY_NOT_READY,
                "NO_AUTHORIZED_POLICY",
                float(score),
                None,
                True,
            )
        except CacheIntegrityError:
            return Status.PROCESSING_ERROR, "E0_FEATURE_INTEGRITY", None, None, quality
        except FileNotFoundError:
            return Status.PROCESSING_ERROR, "E0_RESOURCE_MISSING", None, None, quality
        except Exception:
            return Status.PROCESSING_ERROR, "E0_PROCESSING_ERROR", None, None, quality
