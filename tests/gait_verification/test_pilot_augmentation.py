"""Real FFmpeg transformations on generated patterns; these contain no human gait."""

from dataclasses import asdict
import json

import pytest

from gait_verification.capture_registry import CaptureRegistry
from gait_verification.dataset_processing import derive_variants
from gait_verification.media import probe_video, sha256_file
from gait_verification.stabilization import AugmentationConfig
from test_phase2 import build_world
from test_phase1d import locked_split


@pytest.mark.parametrize(
    "kind,value",
    [("brightness", 0.04), ("contrast", 1.08), ("blur", 0.6), ("compression", 28)],
)
def test_single_recipe_provenance_and_resume(tmp_path, kind, value):
    world = build_world(tmp_path)
    source = world.registry.asset(world.ids[0])
    before = sha256_file(world.registry.path_of(source, world.data))
    config = AugmentationConfig(kind, value)
    variant = config.variant_id
    assert variant == AugmentationConfig(**asdict(config)).variant_id
    results, rows = derive_variants(
        world.registry,
        world.data,
        [source.asset_id],
        [variant],
        recipes={variant: config},
        split=locked_split(
            world.registry,
            counts={"development": 2, "calibration": 0, "locked_final": 0},
        ),
    )
    assert results[0]["status"] == "CREATED"
    registry = CaptureRegistry(
        dict(
            schema_version="capture-registry-v1",
            assets=world.registry.document["assets"] + rows,
        )
    )
    child = registry.asset(rows[0]["asset_id"])
    assert child.declared_split_role == source.declared_split_role
    assert registry.original(child.asset_id).asset_id == source.asset_id
    manifest = json.loads((world.data / child.transformation_manifest_ref).read_text())
    assert manifest["config"] == asdict(config)
    assert manifest["input"]["sha256"] == before
    assert manifest["output"]["sha256"] == child.sha256
    assert manifest["commands"] and manifest["ffmpeg_version"]
    assert len(manifest["input"]["timestamps"]) == len(manifest["output"]["timestamps"])
    assert manifest["timing_transform"]["max_relative_pts_error_sec"] <= 0.0001
    assert sha256_file(world.registry.path_of(source, world.data)) == before
    rerun, _ = derive_variants(
        world.registry,
        world.data,
        [source.asset_id],
        [variant],
        recipes={variant: config},
        split=locked_split(
            world.registry,
            counts={"development": 2, "calibration": 0, "locked_final": 0},
        ),
    )
    assert rerun[0]["status"] == "REUSED"
    assert probe_video(registry.path_of(child, world.data)).sha256 == child.sha256
    registry.path_of(child, world.data).write_bytes(b"corrupt test derivative")
    corrupt, _ = derive_variants(
        registry,
        world.data,
        [source.asset_id],
        [variant],
        recipes={variant: config},
        split=locked_split(
            world.registry,
            counts={"development": 2, "calibration": 0, "locked_final": 0},
        ),
    )
    assert corrupt[0]["status"] == "FAILED"


@pytest.mark.parametrize(
    "kind,value",
    [
        ("flip", 1),
        ("brightness", 0.9),
        ("contrast", 4),
        ("blur", 5),
        ("compression", 51),
        ("blur", float("nan")),
    ],
)
def test_unsafe_or_unknown_recipe_refused(kind, value):
    with pytest.raises(ValueError):
        AugmentationConfig(kind, value).validate()


def test_variant_parameters_cannot_be_relabelled(tmp_path):
    w = build_world(tmp_path)
    a = AugmentationConfig("brightness", 0.04)
    b = AugmentationConfig("brightness", -0.04)
    assert a.variant_id != b.variant_id
    with pytest.raises(ValueError):
        derive_variants(
            w.registry, w.data, w.ids[:1], [a.variant_id], recipes={a.variant_id: b}
        )
