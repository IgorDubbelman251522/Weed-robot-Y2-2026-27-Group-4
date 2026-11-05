import numpy as np

from src.eval import condition_table, metrics_by_condition

CLASSES = ["weed", "moss", "clean_surface"]


def test_precision_and_recall_per_condition():
    rows = [{"surface": "tiles"}, {"surface": "tiles"}, {"surface": "tiles"}, {"surface": "gravel"}, {"surface": ""}]
    targets = np.array([0, 0, 2, 0, 1])
    preds = np.array([0, 2, 0, 0, 1])
    tiles = metrics_by_condition(rows, preds, targets, CLASSES, columns=("surface",))["surface"]["tiles"]
    assert tiles["n"] == 3
    assert tiles["per_class"]["weed"] == {"n": 2, "precision": 0.5, "recall": 0.5}
    assert tiles["per_class"]["moss"] == {"n": 0, "precision": None, "recall": None}
    assert "surface=tiles" in condition_table({"surface": {"tiles": tiles}})


def test_untagged_rows_give_a_clear_message():
    out = metrics_by_condition([{"surface": ""}], np.array([0]), np.array([0]), CLASSES)
    assert out == {"lighting": {}, "weather": {}, "surface": {}}
    assert "no lighting/weather/surface tags" in condition_table(out)
