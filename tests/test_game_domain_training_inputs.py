from argparse import Namespace
import importlib.util
from pathlib import Path
import sys


def load_training_script():
    path = Path("scripts/train_portable_domain_decoder.py")
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("train_portable_domain_decoder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_eval_script():
    path = Path("scripts/eval_lossless_domain_decoder.py")
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("eval_lossless_domain_decoder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_bpe_adapter_script():
    path = Path("scripts/train_bpe_adapter.py")
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("train_bpe_adapter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_portable_domain_training_accepts_game_domain_files(tmp_path):
    module = load_training_script()
    game_file = tmp_path / "npc_dialogue.jsonl"
    game_file.write_text(
        "\n".join(
            [
                '{"speaker":"guard","text":"The north gate is sealed."}',
                '{"speaker":"merchant","text":"Bring me wolf pelts."}',
                '{"speaker":"healer","text":"Rest here before the ruins."}',
            ]
            * 32
        ),
        encoding="utf-8",
    )
    args = Namespace(
        domain_file=[str(game_file)],
        domain_bytes=20_000,
        seq=32,
        batch=4,
    )
    stream = module.load_domain_stream(args, Path("."))
    assert stream.ndim == 1
    assert stream.numel() > args.seq * 4
    assert b"north gate" in bytes(stream.tolist())


def test_lossless_domain_eval_accepts_game_eval_files(tmp_path):
    module = load_eval_script()
    game_file = tmp_path / "quest_lore.txt"
    game_file.write_text(
        "The moon bell opens the north gate.\nThe river stone wakes the shrine.\n",
        encoding="utf-8",
    )
    args = Namespace(
        eval_file=[str(game_file)],
        eval_bytes=512,
        eval_root=None,
        domain_limit=512,
    )
    stream = module.load_eval_stream(args, Path("."))
    assert stream.numel() == 512
    assert b"river stone" in bytes(stream.tolist())


def test_bpe_adapter_training_accepts_domain_files(tmp_path):
    module = load_bpe_adapter_script()
    game_file = tmp_path / "adapter_domain.txt"
    game_file.write_text(
        "quest status active\nnpc dialogue branch accepted\n" * 32,
        encoding="utf-8",
    )
    args = Namespace(
        domain_file=[str(game_file)],
        domain_bytes=20_000,
        seq=32,
    )
    stream = module.load_domain_stream(args, Path("."))
    assert stream.ndim == 1
    assert b"dialogue branch" in bytes(stream.tolist())
