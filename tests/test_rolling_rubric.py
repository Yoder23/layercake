import json

from layercake.rolling.gates import MinMetricGate
from layercake.rolling.rubric import TrainingRubric


def test_rubric_hash_is_stable(tmp_path):
    path = tmp_path / "rubric.json"
    path.write_text(
        json.dumps(
            {
                "rubric_id": "r1",
                "name": "Rubric 1",
                "branch": "main",
                "max_steps": 1,
                "metric": "score",
                "target": 0.5,
                "direction": "min",
                "trainable_modules": ["toy_model"],
                "frozen_modules": [],
                "protected_capabilities": ["abi_hash"],
                "gates": [{"name": "score_gate", "type": "min_metric", "metric": "score", "threshold": 0.5}],
            }
        ),
        encoding="utf-8",
    )
    rubric = TrainingRubric.from_file(path)
    assert rubric.rubric_id == "r1"
    assert rubric.compute_hash() == TrainingRubric.from_json(rubric.to_json()).compute_hash()


def test_rubric_accepts_gate_objects():
    rubric = TrainingRubric(rubric_id="r2", description="Rubric 2", gates=[MinMetricGate("g", "score", 0.5)])
    assert rubric.gates[0].run({"score": 1.0}).passed
