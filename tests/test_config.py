from pathlib import Path

import yaml


def test_classes_yaml_has_five_classes():
    path = Path(__file__).parents[1] / "configs" / "classes.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert len(cfg["classes"]) == 5