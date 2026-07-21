import importlib.util
import json
from pathlib import Path
import sys
import torch


def load_train_module():
    path = Path("scripts/train_byte_core_from_config.py")
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("train_byte_core_from_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_train_byte_core_writes_speed_metrics(tmp_path):
    module = load_train_module()
    corpus = tmp_path / "tiny.txt"
    corpus.write_text(
        "LayerCake training speed smoke. Bytes only. " * 64,
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"
    config = {
        "name": "tiny_speed_metrics",
        "model": {
            "patch_size": 2,
            "d_byte": 8,
            "d_model": 32,
            "d_abi": 16,
            "layers": 1,
            "heads": 4,
            "max_patches": 16,
            "direct_global_context": True,
            "local_decoder": "window_transformer",
            "local_layers": 1,
            "local_width": 32,
            "modern_blocks": True,
            "fused_attention": True,
            "local_window": 8,
            "dropout": 0.0,
            "qk_norm": False,
            "patch_prediction": True,
            "patch_prediction_mode": "factorized",
            "patch_prediction_context": "global",
        },
        "training": {
            "device": "cpu",
            "seq_len": 16,
            "micro_batch_size": 1,
            "grad_accum_steps": 1,
            "steps": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "patch_prediction_loss_weight": 0.1,
            "teacher_local_decoder": "window_transformer",
            "distill_loss_weight": 0.1,
            "distill_interval": 1,
            "distill_until_step_ratio": 1.0,
            "log_interval": 1,
            "save_interval": 1,
            "keep_last_n": 1,
            "save_optimizer": False,
            "include_suffixes": [".txt"],
            "data_roots": [str(corpus)],
            "out_dir": str(out_dir),
        },
    }
    module._train(config)
    metrics = json.loads((out_dir / "training_metrics.json").read_text())
    assert metrics["status"] == "COMPLETE"
    latest = metrics["latest"]
    for key in [
        "steps_per_second",
        "gib_per_hour",
        "projected_total_hours",
        "data_seconds_per_step",
        "forward_backward_seconds_per_step",
        "optimizer_seconds_per_step",
    ]:
        assert key in latest
        assert latest[key] >= 0.0


def test_copy_alignment_labels_pick_previous_matching_byte():
    module = load_train_module()
    x = torch.tensor([[ord("a"), ord("b"), ord("c"), ord("b")]])
    y = torch.tensor([[ord("b"), ord("c"), ord("z"), ord("b")]])
    labels = module._copy_alignment_labels(x, y, source_len=4)
    assert labels.tolist() == [[-100, -100, -100, 3]]


def test_copy_alignment_labels_can_be_prompt_bounded():
    module = load_train_module()
    x = torch.tensor([[ord("a"), ord("b"), ord("c"), ord("b")]])
    y = torch.tensor([[ord("b"), ord("c"), ord("z"), ord("b")]])
    labels = module._copy_alignment_labels(
        x,
        y,
        source_len=4,
        source_end_positions=torch.tensor([2]),
    )
    assert labels.tolist() == [[-100, -100, -100, 1]]


def test_copy_alignment_at_positions_can_be_prompt_bounded():
    module = load_train_module()
    x = torch.tensor([[ord("a"), ord("b"), ord("c"), ord("b")]])
    targets = torch.tensor([[ord("b")]])
    positions = torch.tensor([[3]])

    unbounded = module._copy_alignment_labels_at_positions(
        x,
        targets,
        positions,
        source_len=4,
    )
    bounded = module._copy_alignment_labels_at_positions(
        x,
        targets,
        positions,
        source_len=4,
        source_end_positions=torch.tensor([2]),
    )

    assert unbounded.tolist() == [[3]]
    assert bounded.tolist() == [[1]]


def test_weighted_mixed_dataset_interleaves_components(tmp_path):
    module = load_train_module()
    lm_path = tmp_path / "lm.txt"
    task_path = tmp_path / "task.jsonl"
    lm_path.write_text("language-model-replay " * 8, encoding="utf-8")
    task_path.write_text(
        json.dumps({"text": "Question: Name the color. Answer: blue\n###\n"})
        + "\n",
        encoding="utf-8",
    )

    dataset = module.WeightedMixedByteDataset(
        [
            module.MixedByteComponent(
                name="lm_replay",
                weight=2.0,
                dataset=module.ByteCorpusDataset([lm_path], seq_len=32),
                files=[lm_path],
                row_preserve_jsonl_examples=False,
            ),
            module.MixedByteComponent(
                name="grounded_tasks",
                weight=1.0,
                dataset=module.JsonlRowByteDataset(
                    [task_path],
                    seq_len=32,
                    patch_size=2,
                    answer_marker=b"Answer:",
                ),
                files=[task_path],
                row_preserve_jsonl_examples=True,
            ),
        ]
    )

    assert dataset.schedule == [0, 1, 0]
    iterator = iter(dataset)
    decoded = [
        bytes(int(value) for value in next(iterator).tolist()).decode(
            "utf-8",
            errors="replace",
        )
        for _ in range(6)
    ]
    assert sum("Question:" in sample for sample in decoded) == 2
    assert sum("Question:" not in sample for sample in decoded) == 4


def test_train_byte_core_accepts_weighted_data_mix(tmp_path):
    module = load_train_module()
    lm_path = tmp_path / "lm.txt"
    task_path = tmp_path / "task.jsonl"
    lm_path.write_text(
        "LayerCake needs fluent replay while it learns grounded answers. " * 32,
        encoding="utf-8",
    )
    task_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "text": (
                            "Question: Convert XML <item id=\"7\">ok</item> "
                            "to JSON. Answer:{\"item\":{\"id\":\"7\","
                            "\"text\":\"ok\"}}\n###\n"
                        )
                    }
                ),
                json.dumps(
                    {
                        "text": (
                            "Question: Move the Save button to the top right. "
                            "Answer: move save button to top right\n###\n"
                        )
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "mixed_run"
    config = {
        "name": "tiny_weighted_mix",
        "model": {
            "patch_size": 2,
            "d_byte": 8,
            "d_model": 32,
            "d_abi": 16,
            "layers": 1,
            "heads": 4,
            "max_patches": 16,
            "direct_global_context": True,
            "local_decoder": "window_transformer",
            "local_layers": 1,
            "local_width": 32,
            "modern_blocks": True,
            "fused_attention": True,
            "local_window": 8,
            "dropout": 0.0,
            "qk_norm": False,
            "patch_prediction": True,
            "patch_prediction_mode": "factorized",
            "patch_prediction_context": "global",
        },
        "training": {
            "device": "cpu",
            "seq_len": 16,
            "micro_batch_size": 2,
            "grad_accum_steps": 1,
            "steps": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "patch_prediction_loss_weight": 0.1,
            "answer_loss_weight": 2.0,
            "answer_marker": "Answer:",
            "log_interval": 1,
            "save_interval": 1,
            "keep_last_n": 1,
            "save_optimizer": False,
            "data_mix": [
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
            "out_dir": str(out_dir),
        },
    }

    module._train(config)

    metrics = json.loads((out_dir / "training_metrics.json").read_text())
    assert metrics["status"] == "COMPLETE"
    summary = metrics["data_source_summary"]
    assert summary["mode"] == "weighted_mix"
    assert summary["schedule"] == [0, 1, 0]
    assert [component["name"] for component in summary["components"]] == [
        "lm_replay",
        "grounded_tasks",
    ]


def test_answer_position_weights_select_answer_bytes():
    module = load_train_module()
    row = torch.tensor(
        [list(b"Question: x Answer: yes\n###    ")],
        dtype=torch.long,
    )
    weights = module._answer_position_weights(
        row,
        torch.tensor([18, 19, 20, 21, 22]),
        answer_weight=5.0,
        base_weight=1.0,
        answer_marker=b"Answer:",
    )

    assert weights.tolist() == [[1.0, 5.0, 5.0, 5.0, 5.0]]


def test_answer_start_positions_locate_deployment_boundary():
    module = load_train_module()
    rows = torch.tensor(
        [
            list(b"Question: x Answer: yes   "),
            list(b"No marker here            "),
        ],
        dtype=torch.long,
    )

    starts = module._answer_start_positions(rows, answer_marker=b"Answer: ")

    assert starts.tolist() == [20, -1]


def test_answer_position_weights_support_row_specific_positions():
    module = load_train_module()
    rows = torch.tensor(
        [
            list(b"Q Answer: yes\n###".ljust(24)),
            list(b"Question Answer: no\n###".ljust(24)),
        ],
        dtype=torch.long,
    )
    positions = torch.tensor([[10, 11, 12], [17, 18, 19]])

    weights = module._answer_position_weights(
        rows,
        positions,
        answer_weight=3.0,
        base_weight=0.0,
        answer_marker=b"Answer: ",
    )

    assert weights.tolist() == [[3.0, 3.0, 3.0], [3.0, 3.0, 0.0]]


def test_train_byte_core_can_filter_to_patch_prediction_heads(tmp_path):
    module = load_train_module()
    corpus = tmp_path / "task.jsonl"
    corpus.write_text(
        json.dumps({"text": "Question: Name the color. Answer: blue\n###\n"})
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "filtered_run"
    config = {
        "name": "tiny_patch_head_filter",
        "model": {
            "patch_size": 2,
            "d_byte": 8,
            "d_model": 32,
            "d_abi": 16,
            "layers": 1,
            "heads": 4,
            "max_patches": 16,
            "direct_global_context": True,
            "local_decoder": "window_transformer",
            "local_layers": 1,
            "local_width": 32,
            "modern_blocks": True,
            "fused_attention": True,
            "local_window": 8,
            "dropout": 0.0,
            "qk_norm": False,
            "patch_prediction": True,
            "patch_prediction_mode": "factorized",
            "patch_prediction_context": "global",
        },
        "training": {
            "device": "cpu",
            "seq_len": 16,
            "micro_batch_size": 1,
            "grad_accum_steps": 1,
            "steps": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "patch_prediction_loss_weight": 1.0,
            "patch_prediction_answer_loss_weight": 4.0,
            "answer_marker": "Answer:",
            "trainable_parameter_patterns": ["patch_prediction_heads.*"],
            "row_preserve_jsonl_examples": True,
            "log_interval": 1,
            "save_interval": 1,
            "keep_last_n": 1,
            "save_optimizer": False,
            "include_suffixes": [".jsonl"],
            "data_roots": [str(corpus)],
            "out_dir": str(out_dir),
        },
    }

    module._train(config)

    metrics = json.loads((out_dir / "training_metrics.json").read_text())
    parameter_filter = metrics["parameter_filter"]
    assert parameter_filter["enabled"] is True
    assert parameter_filter["trainable_params_after_filter"] < parameter_filter["total_params"]
    assert parameter_filter["trainable_parameter_names"]
    assert all(
        name.startswith("patch_prediction_heads.")
        for name in parameter_filter["trainable_parameter_names"]
    )


def test_scale_training_configs_have_throughput_guards():
    config_paths = [
        Path("configs/byte_225m_core_phaseA_curriculum.json"),
        Path("configs/byte_225m_core_phaseB_polish.json"),
        Path("configs/byte_500m_core_quickrun.json"),
        Path("configs/byte_500m_sparse_state_quickrun.json"),
        Path("configs/byte_500m_core.json"),
        Path("configs/byte_500m_core_phase1_fluency.json"),
        Path("configs/byte_500m_core_phase2_gameblend.json"),
    ]
    for path in config_paths:
        config = json.loads(path.read_text())
        training = config["training"]
        guard = training.get("throughput_guard")
        assert training.get("metrics_path") == "training_metrics.json", path
        assert training.get("optimizer_fused") is True, path
        assert training.get("matmul_precision") == "high", path
        assert training.get("pin_memory") is True, path
        assert training.get("dataloader_workers", 0) > 0, path
        assert guard, path
        assert guard["warmup_steps"] > 0, path
        assert guard["max_projected_hours"] > 0, path
        if "quickrun" not in path.name:
            assert guard["abort_on_fail"] is True, path


def test_train_byte_core_sparse_cake_optimizer_records_active_fraction(tmp_path):
    module = load_train_module()
    corpus = tmp_path / "tiny.txt"
    corpus.write_text("Sparse routed LayerCake training. " * 64, encoding="utf-8")
    out_dir = tmp_path / "routed_run"
    config = {
        "name": "tiny_sparse_routed_cake",
        "model": {
            "patch_size": 2,
            "d_byte": 8,
            "d_model": 32,
            "d_abi": 16,
            "layers": 1,
            "heads": 4,
            "max_patches": 16,
            "direct_global_context": True,
            "local_decoder": "parallel_patch",
            "local_width": 32,
            "modern_blocks": True,
            "fused_attention": True,
            "routed_cake_experts": 3,
            "dropout": 0.0,
        },
        "training": {
            "device": "cpu",
            "seq_len": 16,
            "micro_batch_size": 1,
            "grad_accum_steps": 1,
            "steps": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "cake_route": 1,
            "cake_sparse_optimizer": True,
            "log_interval": 1,
            "save_interval": 1,
            "keep_last_n": 1,
            "save_optimizer": False,
            "include_suffixes": [".txt"],
            "data_roots": [str(corpus)],
            "out_dir": str(out_dir),
        },
    }

    module._train(config)

    metrics = json.loads((out_dir / "training_metrics.json").read_text())
    routing = metrics["cake_routing"]
    assert routing["route"] == 1
    assert routing["sparse_optimizer"] is True
    assert routing["optimizer_params"] < routing["trainable_params"]
    assert 0.0 < routing["optimizer_fraction_of_trainable"] < 1.0
    checkpoint = torch.load(out_dir / "latest.pt", map_location="cpu")
    assert checkpoint["cake_routing"] == routing
