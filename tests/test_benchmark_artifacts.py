"""Artifact schema, JSON, and manifest round-trip tests."""

from __future__ import annotations

from chem_operator._benchmark.artifacts import (
    ArtifactManifest,
    CheckpointBundle,
    load_torch,
    read_json,
)


def test_checkpoint_bundle_and_manifest_round_trip(tmp_path) -> None:
    bundle = CheckpointBundle(
        model_state={
            "backend": "dummy",
            "state_dict": {},
        },
        preprocessing_state={"normalizer": {"type": "identity"}},
        problem_spec={"name": "problem"},
        model_spec={"name": "model"},
        best_config={"width": 8},
        dataset_fingerprint="abc123",
    )
    paths = bundle.save(tmp_path)
    assert load_torch(paths["model"])["backend"] == "dummy"
    assert (
        load_torch(paths["preprocessing"])["normalizer"]["type"]
        == "identity"
    )

    manifest = ArtifactManifest.create(
        run_id="run",
        problem="problem",
        model="model",
        seed=1,
        dataset_fingerprint="abc123",
    )
    manifest.files = {"model": "model.pt"}
    path = manifest.save(tmp_path / "manifest.json")
    restored = read_json(path)
    assert restored["schema"] == "chem-operator-benchmark-v1"
    assert restored["files"] == {"model": "model.pt"}
