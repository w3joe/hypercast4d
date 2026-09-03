from pathlib import Path

import pandas as pd
import pytest
import yaml
from sklearn.model_selection import train_test_split

from hypercast4d.data import load_paper_data
from hypercast4d.models import parameter_count
from hypercast4d.reproduction import (
    _build_candidate,
    _effective_model_name,
    candidate_grids,
    prepare_paper_windows,
)


ROOT = Path(__file__).parents[1]


def reproduction_config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "paper_reproduction.yaml").read_text(encoding="utf-8")
    )


def timed_subset_config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "paper_reproduction_20min.yaml").read_text(encoding="utf-8")
    )


def test_complete_search_spaces_match_supplementary_notebook() -> None:
    grids = candidate_grids(reproduction_config()["search_spaces"])
    assert len(grids["cnn"]) == 160
    assert len(grids["lstm"]) == 160
    assert len(grids["hyper"]) == 576
    assert sum(map(len, grids.values())) == 896
    assert grids["cnn"][78] == {
        "activation": "linear",
        "dense_before_pool": True,
        "dense_after_pool": True,
        "units": 128,
        "dense_units": 32,
    }
    assert grids["lstm"][78] == {
        "activation": "linear",
        "dense_before_pool": True,
        "dense_after_pool": True,
        "dense_units": 64,
        "units": 64,
    }
    assert grids["hyper"][282] == {
        "activation": "linear",
        "declared_algebra": "coquaternion",
        "dense_before_pool": True,
        "dense_after_pool": True,
        "units": 16,
        "dense_units": 32,
    }


def test_timed_subset_is_680_fits_and_stays_in_full_grid() -> None:
    complete = candidate_grids(reproduction_config()["search_spaces"])
    subset_config = timed_subset_config()
    subset = candidate_grids(subset_config["search_spaces"])
    assert {model: len(grid) for model, grid in subset.items()} == {
        "cnn": 16,
        "lstm": 16,
        "hyper": 36,
    }
    assert (
        sum(map(len, subset.values())) * subset_config["cross_validation"]["folds"]
        == 680
    )
    for model in subset:
        assert all(candidate in complete[model] for candidate in subset[model])
    assert subset_config["training"]["epochs"] == 50
    assert subset_config["data"]["windows"] == [10]
    assert subset_config["data"]["horizons"] == [1]


def test_global_scaling_uses_named_target_column() -> None:
    frame = pd.DataFrame(
        {
            "Copper": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "FCX": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "CLP": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "SCCO": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        }
    )
    prepared = prepare_paper_windows(
        frame,
        input_columns=["FCX", "CLP", "SCCO", "Copper"],
        target_column="Copper",
        window=2,
        horizon=1,
        feature_range=(0.0, 1.0),
    )
    assert prepared.features.shape == (4, 2, 4)
    assert prepared.targets[:, 0].tolist() == pytest.approx([0.4, 0.6, 0.8, 1.0])


def test_paper_w10_h1_split_has_published_sample_counts() -> None:
    frame = load_paper_data(ROOT / "data" / "raw" / "paper_data.xlsx")
    prepared = prepare_paper_windows(
        frame,
        input_columns=["Copper", "FCX", "CLP", "SCCO"],
        target_column="Copper",
        window=10,
        horizon=1,
        feature_range=(0.0, 1.0),
    )
    split = train_test_split(
        prepared.features,
        prepared.targets,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )
    train_features, holdout_features, _, _ = split
    assert len(train_features) == 1598
    assert len(holdout_features) == 400


def test_notebook_bug_compatibility_is_explicit() -> None:
    candidate = {"declared_algebra": "cl11"}
    model_name, declared, effective = _effective_model_name(
        "hyper", candidate, "force_quaternion", "quaternion"
    )
    assert (model_name, declared, effective) == (
        "hyper_quaternion",
        "cl11",
        "quaternion",
    )
    intended = _effective_model_name("hyper", candidate, "declared", "quaternion")
    assert intended == ("hyper_cl11", "cl11", "cl11")


def test_published_w10_h1_parameter_counts() -> None:
    config = reproduction_config()
    architecture = config["architecture"]
    cases = {
        "cnn": (
            {
                "units": 128,
                "dense_before_pool": True,
                "dense_after_pool": True,
                "dense_units": 32,
                "activation": "linear",
            },
            10977,
        ),
        "lstm": (
            {
                "units": 64,
                "dense_before_pool": True,
                "dense_after_pool": True,
                "dense_units": 64,
                "activation": "linear",
            },
            42433,
        ),
        "hyper": (
            {
                "units": 16,
                "dense_before_pool": True,
                "dense_after_pool": True,
                "dense_units": 32,
                "activation": "linear",
                "declared_algebra": "coquaternion",
            },
            7393,
        ),
    }
    for model_type, (candidate, expected) in cases.items():
        model, _, _ = _build_candidate(
            model_type,
            candidate,
            architecture,
            "force_quaternion",
            "quaternion",
            window=10,
            horizon=1,
        )
        assert parameter_count(model) == expected
