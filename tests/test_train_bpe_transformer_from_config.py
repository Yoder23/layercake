import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.train_bpe_transformer_from_config import _load_mixed_bytes


def test_bpe_mixed_loader_interleaves_weighted_sources(tmp_path):
    lm_path = tmp_path / "lm.txt"
    task_path = tmp_path / "task.jsonl"
    lm_path.write_text("language replay segment\n" * 512, encoding="utf-8")
    task_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "text": (
                        "Question: Convert XML <item id=\"7\">ok</item> "
                        "to JSON. Answer:{\"item\":{\"id\":\"7\","
                        "\"text\":\"ok\"}}\n###\n"
                    )
                }
            )
            for _ in range(128)
        )
        + "\n",
        encoding="utf-8",
    )

    payload, summary = _load_mixed_bytes(
        root=ROOT,
        data_mix=[
            {
                "name": "lm_replay",
                "weight": 2.0,
                "include_suffixes": [".txt"],
                "data_roots": [str(lm_path)],
            },
            {
                "name": "grounded_tasks",
                "weight": 1.0,
                "include_suffixes": [".jsonl"],
                "row_preserve_jsonl_examples": True,
                "data_roots": [str(task_path)],
            },
        ],
        include_suffixes={".txt", ".jsonl"},
        max_bytes=4096,
        read_block_bytes=64,
    )

    text = payload.decode("utf-8", errors="replace")
    assert summary["mode"] == "weighted_mix"
    assert summary["schedule"] == [0, 1, 0]
    assert [component["name"] for component in summary["components"]] == [
        "lm_replay",
        "grounded_tasks",
    ]
    shares = [component["realized_byte_share"] for component in summary["components"]]
    assert 0.64 <= shares[0] <= 0.69
    assert 0.31 <= shares[1] <= 0.36
    assert sum(component["realized_bytes"] for component in summary["components"]) == len(
        payload
    )
    assert "language replay segment" in text
    assert "Question:" in text
