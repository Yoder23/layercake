.PHONY: test smoke-train smoke-transfer smoke-byte-patch smoke-infer benchmark docs-check verify-research verify-scale verify-scale15 verify-lossless verify-mobile-domain verify-general-frontier

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

docs-check:
	python -c "from pathlib import Path; required=['RUBRIC.md','BYTE_PATCH_LAYERCAKE.md','BENCHMARKS.md','ORCHESTRATION.md','TOKENIZER_FREE.md','ROADMAP.md','NEXT_STEPS.md']; assert all(Path(p).exists() for p in required)"

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
