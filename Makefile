.PHONY: test smoke-train smoke-transfer smoke-byte-patch smoke-infer benchmark docs-check rolling-demo rolling-smoke rolling-benchmark preview-demo preview-benchmark dominance-smoke verify verify-research verify-scale verify-scale15 verify-lossless verify-mobile-domain verify-general-frontier

test:
	pytest -q

smoke-train:
	python scripts/train_abi_aligned.py --steps 10

smoke-transfer:
	python scripts/eval_lossless_copy.py

smoke-byte-patch:
	python scripts/smoke_byte_patch.py

smoke-infer:
	python scripts/benchmark_byte_patch.py

benchmark:
	python scripts/benchmark_training_cost.py
	python scripts/benchmark_domain_routing.py

rolling-demo:
	python scripts/demo_rolling_training.py --smoke

rolling-smoke:
	python -m layercake.rolling.cli --help
	pytest tests/test_rolling_cli.py tests/test_rolling_rubric.py tests/test_rolling_trainer.py tests/test_model_commit.py tests/test_module_registry.py tests/test_dataset_manifest.py tests/test_gates.py tests/test_rollback.py tests/test_branching.py tests/test_cherrypick.py tests/test_bisect.py -q

rolling-benchmark:
	python scripts/benchmark_rolling_training.py
	python scripts/benchmark_rollback_cost.py
	python scripts/benchmark_cherrypick_transfer.py

preview-demo:
	python scripts/demo_preview_guided_layercake_training.py --smoke

preview-benchmark:
	python scripts/benchmark_preview_guided_training.py
	python scripts/benchmark_curriculum_modes.py

dominance-smoke:
	python scripts/run_dominance_gates.py --run-id smoke
	python scripts/benchmark_tier1_dominance.py --steps 4
	python scripts/verify_tier1_dominance.py
	python scripts/verify_tier1_local_frontier.py

verify:
	python scripts/verify_northstar_mobile.py

docs-check:
	python -c "from pathlib import Path; required=['RUBRIC.md','BYTE_PATCH_LAYERCAKE.md','BENCHMARKS.md','ORCHESTRATION.md','TOKENIZER_FREE.md','ROADMAP.md','NEXT_STEPS.md','ROLLING_TRAINING.md','MODEL_COMMITS.md','RUBRIC_TRAINING.md','SEMANTIC_CI.md','ROLLBACK.md','BRANCHING_AND_CHERRYPICK.md','PREVIEW_GUIDED_TRAINING.md','SCALING_PROTOCOL.md','DOMINANCE_GATES.md']; assert all(Path(p).exists() for p in required)"

verify-research:
	python scripts/verify_research_gates.py

verify-scale:
	python scripts/verify_scale5m_results.py

verify-scale15:
	python scripts/verify_scale15m_results.py

verify-lossless:
	python scripts/verify_lossless_domain_results.py

verify-mobile-domain:
	python scripts/verify_mobile_domain_win.py

verify-general-frontier:
	python scripts/verify_general_core_frontier.py
