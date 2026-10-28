from darius.pipeline.jobs import STAGE_NAMES, ready_stages, stages_from


def names(stages):
    return [stage.name for stage in stages]


def test_a_fresh_job_starts_with_ingest_only():
    assert names(ready_stages(set(), set())) == ["ingest"]


def test_reconstruction_and_segmentation_run_side_by_side():
    assert names(ready_stages({"ingest"}, set())) == ["reconstruct", "segment"]
    assert names(ready_stages({"ingest", "reconstruct"}, {"segment"})) == []


def test_physics_and_splats_run_side_by_side_and_package_waits_for_both():
    done = {"ingest", "reconstruct", "segment", "lift"}
    assert names(ready_stages(done, set())) == ["physics", "splat"]
    assert names(ready_stages(done | {"physics"}, {"splat"})) == []
    assert names(ready_stages(done | {"physics", "splat"}, set())) == ["package"]


def test_rerunning_a_stage_reruns_only_what_depends_on_it():
    assert names(stages_from(None)) == STAGE_NAMES
    assert names(stages_from("segment")) == ["segment", "lift", "physics", "splat", "package"]
    assert names(stages_from("physics")) == ["physics", "package"]
    assert names(stages_from("splat")) == ["splat", "package"]
