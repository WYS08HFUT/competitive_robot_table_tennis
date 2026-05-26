from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.env_cfg import DatasetCfg
from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.utils.serve_dataset import (
    filter_serves,
    load_serves,
)


def test_load_serves_schema() -> None:
    samples = load_serves("serves.json")
    assert len(samples) == 2704
    sample = samples[0]
    assert sample.id == 0
    assert sample.pos_y > 0.0
    assert sample.vel_y < 0.0


def test_filter_serves_buckets_are_nonempty() -> None:
    samples = load_serves("serves.json")
    dataset_cfg = DatasetCfg()
    easy_ids = filter_serves(
        samples,
        difficulty="easy",
        dataset_cfg=dataset_cfg,
        workspace_x=(-0.60, 0.60),
        workspace_y=(-1.25, -0.05),
        workspace_z=(0.05, 0.60),
    )
    base_ids = filter_serves(
        samples,
        difficulty="base",
        dataset_cfg=dataset_cfg,
        workspace_x=(-0.60, 0.60),
        workspace_y=(-1.25, -0.05),
        workspace_z=(0.05, 0.60),
    )
    assert easy_ids
    assert base_ids
    assert set(easy_ids).issubset(set(base_ids))
