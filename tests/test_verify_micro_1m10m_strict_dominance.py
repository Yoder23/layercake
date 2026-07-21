import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path("scripts/verify_micro_1m10m_strict_dominance.py")
    spec = importlib.util.spec_from_file_location("verify_micro_1m10m", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_current_micro_frontier_does_not_promote_strict_dominance(tmp_path):
    module = load_module()
    artifact = Path("results/micro_scale_curriculum_frontier_v2.json")
    output = tmp_path / "strict.json"
    result = module.verify(
        artifact=artifact,
        output=output,
        min_train_bytes=1_000_000,
        min_eval_bytes=100_000,
        min_steps=500,
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    assert result == written
    assert result["status"] == "FAIL"
    assert "all_scales_strict_pass" in result["failed"]
    for scale in result["scales"]:
        assert scale["status"] == "FAIL"
        assert "bpb_strictly_lower" in scale["failed"]
        assert "generation_quality_noninferior" in scale["failed"]
