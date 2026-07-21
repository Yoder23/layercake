import json

import torch

from scripts.prepare_game_domain_data import prepare_game_domain_data
from scripts.train_250m_english_core import EnglishCorpusDataset, build_250m_model


def test_english_corpus_dataset_streams_fixed_byte_chunks(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(
            [
                json.dumps({"text": "abc"}),
                "not-json",
                json.dumps({"content": "defgh"}),
                json.dumps(["ignored"]),
            ]
        ),
        encoding="utf-8",
    )

    chunks = [bytes(tensor.tolist()) for tensor in EnglishCorpusDataset(corpus, 4)]

    assert chunks == [b"abcd", b"efgh"]


def test_game_domain_preparation_excludes_its_own_output(tmp_path):
    game_dir = tmp_path / "game"
    game_dir.mkdir()
    (game_dir / "quest.txt").write_text("Quest objective", encoding="utf-8")
    (game_dir / "records.json").write_text(
        json.dumps([{"item": "sword"}, {"location": "inn"}]),
        encoding="utf-8",
    )
    output = game_dir / "prepared.jsonl"

    count = prepare_game_domain_data(game_dir, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert count == 3
    assert len(rows) == 3
    assert {row["doc_id"] for row in rows} == {0, 1, 2}
    assert any(row["type"] == "quest" for row in rows)


def test_nominal_250m_model_configuration_is_valid():
    with torch.device("meta"):
        model = build_250m_model(torch.device("meta"))

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 240_000_000 <= parameter_count <= 260_000_000
