from __future__ import annotations

from pathlib import Path

import torch

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import build_package, tensor_specs
from layercake.cake.signing import generate_keypair
from layercake.models.direct_cake_host import DirectCakeHost
from layercake.models.portable_decoder import (
    portable_token_plan_manifest_architecture,
)
from layercake.portable_token_plan import (
    EOS_ID,
    GENERIC_TOKENIZER_FORMAT,
    LosslessLexemePointerTokenizer,
    PortableTokenPlan,
    build_token_plan_artifact,
)
from layercake.training.generic_domain import (
    CONFIG_FORMAT,
    evaluate_generated,
    render_dataset,
)


ABI_VERSION = "lc-direct-neural-decoder/1"
ABI_HASH = (
    "de765899700aefe22bfe6c9d00ed5b0c1f87a7ef864cf7211aa8aa4491a0742a"
)


def _config() -> dict:
    return {
        "format": CONFIG_FORMAT,
        "domain_id": "sql",
        "description": "test SQL capability",
        "data_seed": 71,
        "rows_per_family": {
            "train": 2,
            "validation": 1,
            "test": 1,
        },
        "families": [
            {
                "id": "select_value",
                "prompt_templates": [
                    (
                        "Return SQL only. Select exact column {column} "
                        "from exact table {table}."
                    )
                ],
                "response_template": (
                    "SELECT {column} FROM {table};"
                ),
                "copy_slots": ["table", "column"],
                "evaluation": {
                    "kind": "sqlite_query",
                    "setup": [
                        "CREATE TABLE {table} ({column} INTEGER)",
                        "INSERT INTO {table} VALUES (7)",
                    ],
                    "expected_rows": [[7]],
                },
            }
        ],
    }


def test_generic_dataset_is_split_disjoint_and_functional():
    rows = render_dataset(_config())
    assert len(rows) == 4
    by_split = {
        split: {
            value
            for row in rows
            if row["split"] == split
            for value in row["copy_lexemes"]
        }
        for split in ("train", "validation", "test")
    }
    assert by_split["train"].isdisjoint(by_split["validation"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["validation"].isdisjoint(by_split["test"])
    for row in rows:
        passed, _ = evaluate_generated(row["response"], row)
        assert passed


def test_generic_tokenizer_uses_multiple_lossless_pointers():
    rows = render_dataset(_config())
    tokenizer = LosslessLexemePointerTokenizer.build_generic(rows)
    assert tokenizer.format_version == GENERIC_TOKENIZER_FORMAT
    source_ids, source = tokenizer.encode_source(rows[0]["prompt"])
    actions = tokenizer.encode_target(
        rows[0]["response"],
        copy_lexemes=rows[0]["copy_lexemes"],
        source_lexemes=source,
    )
    assert sum(
        action >= tokenizer.vocab_size for action in actions
    ) == 2
    assert tokenizer.decode_actions(actions, source) == rows[0][
        "response"
    ].encode()
    assert len(source_ids) == len(source)


def test_regex_evaluator_checks_behavior_not_exact_text():
    row = {
        "format": "layercake-generic-domain-dataset/1",
        "id": "regex-test",
        "domain_id": "regex",
        "family": "literal",
        "split": "test",
        "prompt": "Use exact literal alpha_regex_test_1.",
        "response": "^alpha_regex_test_1$",
        "copy_lexemes": ["alpha_regex_test_1"],
        "evaluation": {
            "kind": "regex_fullmatch",
            "must_match": ["alpha_regex_test_1"],
            "must_not_match": ["xalpha_regex_test_1", "beta"],
        },
        "generation_index": 0,
    }
    passed, _ = evaluate_generated(
        "(?:alpha_regex_test_1)", row
    )
    assert passed
    failed, _ = evaluate_generated(".*", row)
    assert not failed


def _package(
    path: Path,
    *,
    domain: str,
    private: bytes,
    key_id: str,
) -> Path:
    row = {
        "prompt": f"Use {domain}_dynamic_value once.",
        "response": f"{domain}_dynamic_value",
        "copy_lexemes": [f"{domain}_dynamic_value"],
    }
    tokenizer = LosslessLexemePointerTokenizer.build_generic([row])
    model = PortableTokenPlan(
        fixed_vocab_size=tokenizer.vocab_size,
        model_width=24,
        attention_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        feedforward_width=48,
        pointer_width=12,
        dropout=0.0,
        maximum_source_lexemes=24,
        maximum_target_actions=8,
    ).eval().bind_tokenizer(tokenizer)
    with torch.no_grad():
        model.fixed_output.weight.zero_()
        model.fixed_output.bias.fill_(-10.0)
        model.fixed_output.bias[EOS_ID] = 10.0
        model.pointer_gate.weight.zero_()
        model.pointer_gate.bias.fill_(-10.0)
    artifact = build_token_plan_artifact(
        model, tokenizer, domain_id=domain
    )
    manifest = CakeManifest(
        schema_version="1",
        cake_id=f"{domain}-token-plan",
        name=f"{domain} test cake",
        description="test direct neural cake",
        version="1.0.0",
        publisher={"id": "test", "name": "Test", "key_id": key_id},
        abi_version=ABI_VERSION,
        abi_hash=ABI_HASH,
        cake_type="portable_decoder",
        input_contract={
            "mode": "direct_selected_portable_decoder",
            "external": "UTF-8 bytes",
        },
        output_contract={
            "external": "UTF-8 bytes",
            "composition": "direct_selected_one_cake_no_router",
        },
        architecture=portable_token_plan_manifest_architecture(
            artifact["spec"]
        ),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={
            "features": ["byte_input", "safe_tensors", "incremental"]
        },
        tensor_payload_hash="",
        tensor_shapes=tensor_specs(artifact["state_dict"]),
        package_hash="",
        training_data_provenance={"dataset": "test"},
        evaluation_evidence={"status": "TEST"},
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": key_id},
        domains=(domain,),
    )
    return build_package(
        path,
        manifest,
        artifact["state_dict"],
        private_key=private,
    )


def test_direct_host_physically_skips_nonselected_cakes(tmp_path):
    private, public, key_id = generate_keypair()
    sql = _package(
        tmp_path / "sql.cake",
        domain="sql",
        private=private,
        key_id=key_id,
    )
    regex = _package(
        tmp_path / "regex.cake",
        domain="regex",
        private=private,
        key_id=key_id,
    )
    host = DirectCakeHost(
        tmp_path / "registry",
        abi_version=ABI_VERSION,
        abi_hash=ABI_HASH,
        trust_store={key_id: public},
    )
    host.install(sql)
    host.install(regex)
    host.reset_telemetry()
    result = host.generate(
        "sql-token-plan", "Use sql_dynamic_value once."
    )
    assert result.output == b""
    telemetry = host.telemetry()
    assert telemetry["sql-token-plan"]["prefill_calls"] == 1
    assert telemetry["sql-token-plan"]["decode_step_calls"] == 1
    assert telemetry["regex-token-plan"]["prefill_calls"] == 0
    assert telemetry["regex-token-plan"]["decode_step_calls"] == 0
    assert telemetry["regex-token-plan"]["module_load_calls"] == 0
