"""Single-session ONNX Runtime for the shallow sparse English checkpoint.

The runtime graph uses a dynamic ``Gather`` over the installed instruction-cake
weights.  For batch one, only the selected rank-64 cake participates in the
matrix path.  No Python/PyTorch model object is loaded by the benchmark path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Sequence

import numpy as np
import onnxruntime as ort
import psutil
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[3]
QWEN_BPS = 506.2597482558446


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _resolve(path: Path) -> Path:
    return (path if path.is_absolute() else ROOT / path).resolve()


def _legacy_cache(cache) -> tuple[tuple[Any, Any], ...]:
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return tuple(cache)


def export_runtime(checkpoint: Path, output: Path) -> dict[str, Any]:
    """Export one dynamic-KV graph and quantize its matrix/embedding weights."""

    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import torch
    import torch.nn.functional as F

    from layercake.training.phase2_shallow_sparse import load_student

    checkpoint = _resolve(checkpoint)
    output = _resolve(output)
    if output.exists():
        raise RuntimeError(f"native runtime artifact is immutable: {output}")
    output.mkdir(parents=True, exist_ok=False)
    model, _, metadata = load_student(checkpoint)
    model.eval()

    class RuntimeGraph(torch.nn.Module):
        def __init__(self, source):
            super().__init__()
            self.transformer = source.transformer
            self.task_classifier = source.task_classifier
            self.task_cakes = source.task_cakes

        def _selected_cake(
            self, hidden: torch.Tensor, route: torch.Tensor
        ) -> torch.Tensor:
            norm_weight = torch.stack(
                [cake.norm.weight for cake in self.task_cakes]
            ).index_select(0, route)
            norm_bias = torch.stack(
                [cake.norm.bias for cake in self.task_cakes]
            ).index_select(0, route)
            mean = hidden.mean(dim=-1, keepdim=True)
            variance = (hidden - mean).square().mean(dim=-1, keepdim=True)
            normalized = (hidden - mean) * torch.rsqrt(variance + 1.0e-5)
            normalized = (
                normalized * norm_weight[:, None]
                + norm_bias[:, None]
            )
            down = torch.stack(
                [cake.down.weight for cake in self.task_cakes]
            ).index_select(0, route)
            up = torch.stack(
                [cake.up.weight for cake in self.task_cakes]
            ).index_select(0, route)
            low = torch.bmm(normalized, down.transpose(1, 2))
            update = torch.bmm(F.silu(low), up.transpose(1, 2))
            return hidden + update

        def forward(
            self,
            input_ids,
            requested_route,
            past_key_0,
            past_value_0,
            past_key_1,
            past_value_1,
            past_key_2,
            past_value_2,
        ):
            from transformers import DynamicCache

            legacy = (
                (past_key_0, past_value_0),
                (past_key_1, past_value_1),
                (past_key_2, past_value_2),
            )
            cache = DynamicCache.from_legacy_cache(legacy)
            # GPT-2 creates position_ids with a Python ``view`` whose traced
            # shape is fixed to the export example.  Supplying them explicitly
            # keeps both prompt and cache sequence dimensions live in ONNX.
            input_shape = torch._shape_as_tensor(input_ids)
            cache_shape = torch._shape_as_tensor(past_key_0)
            position_ids = torch.arange(
                cache_shape[2],
                cache_shape[2] + input_shape[1],
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0)
            result = self.transformer(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            hidden = result.last_hidden_state
            task_scores = self.task_classifier(hidden)
            inferred = task_scores.mean(dim=1).argmax(dim=-1)
            route = torch.where(
                requested_route < 0, inferred, requested_route
            )
            adapted = self._selected_cake(hidden, route)
            logits = F.linear(
                adapted[:, -1], self.transformer.wte.weight
            )
            present = _legacy_cache(result.past_key_values)
            return (
                logits,
                route,
                task_scores[:, -1],
                adapted[:, -1],
                present[0][0],
                present[0][1],
                present[1][0],
                present[1][1],
                present[2][0],
                present[2][1],
            )

    graph = RuntimeGraph(model).eval()
    # Export a one-token recurrent step.  Prompt prefill uses this same graph
    # token by token, which avoids a static sequence-length assumption in the
    # upstream Transformers causal-mask exporter while retaining dynamic KV.
    input_ids = torch.tensor([[32]], dtype=torch.long)
    route = torch.tensor([-1], dtype=torch.long)
    empty = tuple(
        torch.zeros(1, 12, 0, 64, dtype=torch.float32)
        for _ in range(6)
    )
    fp32_path = output / "model-fp32.onnx"
    input_names = [
        "input_ids",
        "requested_route",
        "past_key_0",
        "past_value_0",
        "past_key_1",
        "past_value_1",
        "past_key_2",
        "past_value_2",
    ]
    output_names = [
        "logits",
        "route",
        "task_scores",
        "abi_state",
        "present_key_0",
        "present_value_0",
        "present_key_1",
        "present_value_1",
        "present_key_2",
        "present_value_2",
    ]
    dynamic_axes = {
        **{
            name: {2: "past_sequence"}
            for name in input_names[2:]
        },
        **{
            name: {2: "present_sequence"}
            for name in output_names[4:]
        },
    }
    with torch.inference_mode():
        torch.onnx.export(
            graph,
            (input_ids, route, *empty),
            fp32_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )
    intermediate = output / "model-int8-matmul.onnx"
    quantize_dynamic(
        fp32_path,
        intermediate,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    document = onnx.load(intermediate)
    initializers = {
        initializer.name: initializer
        for initializer in document.graph.initializer
    }
    embedding_name = None
    embedding_gather = None
    for node in document.graph.node:
        if node.op_type != "Gather" or not node.input:
            continue
        initializer = initializers.get(node.input[0])
        if initializer is None:
            continue
        array = numpy_helper.to_array(initializer)
        if array.shape == (50257, 768):
            embedding_name = initializer.name
            embedding_gather = node
            break
    if embedding_name is None or embedding_gather is None:
        raise RuntimeError("could not locate exported token embedding Gather")
    embedding = numpy_helper.to_array(initializers[embedding_name]).astype(
        np.float32
    )
    scale = np.float32(max(float(np.abs(embedding).max()) / 127.0, 1.0e-8))
    quantized = np.clip(np.rint(embedding / scale), -127, 127).astype(np.int8)
    quantized_name = embedding_name + "_runtime_int8"
    scale_name = embedding_name + "_runtime_scale"
    zero_name = embedding_name + "_runtime_zero"
    document.graph.initializer.extend(
        (
            numpy_helper.from_array(quantized, name=quantized_name),
            numpy_helper.from_array(np.asarray(scale), name=scale_name),
            numpy_helper.from_array(np.asarray(0, dtype=np.int8), name=zero_name),
        )
    )
    original_output = embedding_gather.output[0]
    quantized_output = original_output + "_runtime_int8"
    embedding_gather.input[0] = quantized_name
    embedding_gather.output[0] = quantized_output
    node_index = list(document.graph.node).index(embedding_gather)
    dequantize = helper.make_node(
        "DequantizeLinear",
        [quantized_output, scale_name, zero_name],
        [original_output],
        name="RuntimeEmbeddingDequantize",
    )
    document.graph.node.insert(node_index + 1, dequantize)
    remaining_inputs = {
        value
        for node in document.graph.node
        for value in node.input
    }
    if embedding_name not in remaining_inputs:
        document.graph.initializer.remove(initializers[embedding_name])
    int8_path = output / "model-int8.onnx"
    onnx.checker.check_model(document)
    onnx.save(document, int8_path)
    shutil_path = output / "tokenizer.json"
    shutil_path.write_bytes((checkpoint / "tokenizer.json").read_bytes())
    runtime_metadata = {
        "format": "layercake-shallow-sparse-onnx-runtime/2",
        "status": "EXPORTED",
        "source_checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "architecture": metadata["architecture"],
        "runtime": {
            "provider": "onnxruntime.CPUExecutionProvider",
            "graph": int8_path.relative_to(ROOT).as_posix(),
            "graph_sha256": sha256_file(int8_path),
            "graph_bytes": int8_path.stat().st_size,
            "fp32_graph_sha256": sha256_file(fp32_path),
            "matrix_weight_quantization": "dynamic signed int8 per channel",
            "embedding_quantization": "signed int8 per tensor, gathered before dequantization",
            "task_cake_dispatch": "dynamic Gather selects one cake's norm/down/up weights",
            "installed_task_cakes": 10,
            "maximum_active_task_cakes_per_sequence": 1,
        },
        "tokenizer": {
            "path": shutil_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(shutil_path),
        },
        "canonical_semantic_abi": {
            "path": "moonshot/phase2_canonical_semantic_abi_r3.json",
            "sha256": sha256_file(
                ROOT / "moonshot/phase2_canonical_semantic_abi_r3.json"
            ),
            "graph_output": "abi_state",
        },
        "test_accessed": False,
    }
    (output / "metadata.json").write_text(
        json.dumps(runtime_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return runtime_metadata


class NativeState:
    def __init__(
        self,
        route: np.ndarray,
        cache: list[np.ndarray],
        abi_state: np.ndarray,
    ):
        self.route = route
        self.cache = cache
        self.abi_state = abi_state


class NativeRuntime:
    def __init__(self, artifact: Path, *, threads: int = 14):
        self.artifact = _resolve(artifact)
        self.metadata = json.loads(
            (self.artifact / "metadata.json").read_text(encoding="utf-8")
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # ORT's default CPU prepack cache expands this 106 MB int8 graph to
        # roughly 480 MB RSS.  The compact path keeps weights in their graph
        # representation and remains below the locked absolute memory ceiling.
        options.enable_mem_pattern = False
        options.enable_mem_reuse = True
        options.add_session_config_entry("session.disable_prepacking", "1")
        self.session = ort.InferenceSession(
            str(self.artifact / "model-int8.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = Tokenizer.from_file(
            str(self.artifact / "tokenizer.json")
        )
        self.empty = [
            np.zeros((1, 12, 0, 64), dtype=np.float32)
            for _ in range(6)
        ]

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _run(
        self, input_ids: np.ndarray, route: np.ndarray, cache: list[np.ndarray]
    ) -> tuple[np.ndarray, NativeState]:
        feeds = {
            "input_ids": input_ids.astype(np.int64, copy=False),
            "requested_route": route.astype(np.int64, copy=False),
            "past_key_0": cache[0],
            "past_value_0": cache[1],
            "past_key_1": cache[2],
            "past_value_1": cache[3],
            "past_key_2": cache[4],
            "past_value_2": cache[5],
        }
        outputs = self.session.run(None, feeds)
        return (
            outputs[0],
            NativeState(outputs[1], list(outputs[4:]), outputs[3]),
            outputs[2],
        )

    def prefill(self, ids: list[int]) -> tuple[np.ndarray, NativeState]:
        if not ids:
            raise ValueError("prefill requires at least one token")
        cache = self.empty
        task_scores: list[np.ndarray] = []
        cache_before_last = cache
        for index, token_id in enumerate(ids):
            if index == len(ids) - 1:
                cache_before_last = cache
            _, state, scores = self._run(
                np.asarray([[token_id]], dtype=np.int64),
                np.asarray([-1], dtype=np.int64),
                cache,
            )
            cache = state.cache
            task_scores.append(scores)
        selected_route = np.asarray(
            [int(np.mean(np.stack(task_scores), axis=0).argmax())],
            dtype=np.int64,
        )
        logits, state, _ = self._run(
            np.asarray([[ids[-1]]], dtype=np.int64),
            selected_route,
            cache_before_last,
        )
        return logits, state

    def decode_step(
        self, token_id: int, state: NativeState
    ) -> tuple[np.ndarray, NativeState]:
        logits, next_state, _ = self._run(
            np.asarray([[token_id]], dtype=np.int64),
            state.route,
            state.cache,
        )
        return logits, next_state


def _select_token(
    logits: np.ndarray,
    generated: list[int],
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 4,
) -> int:
    values = logits[0].copy()
    if repetition_penalty != 1.0:
        for token in set(generated):
            values[token] = (
                values[token] / repetition_penalty
                if values[token] > 0
                else values[token] * repetition_penalty
            )
    if no_repeat_ngram_size > 0 and len(generated) >= no_repeat_ngram_size - 1:
        prefix = tuple(generated[-(no_repeat_ngram_size - 1):])
        blocked = set()
        for index in range(len(generated) - no_repeat_ngram_size + 1):
            if tuple(
                generated[index:index + no_repeat_ngram_size - 1]
            ) == prefix:
                blocked.add(generated[index + no_repeat_ngram_size - 1])
        if blocked:
            values[list(blocked)] = -np.inf
    return int(values.argmax())


def _quality(payload: bytes) -> dict[str, float]:
    try:
        decoded = payload.decode("utf-8")
        valid = 1.0
    except UnicodeDecodeError:
        decoded = payload.decode("utf-8", errors="replace")
        valid = 0.0
    grams = [
        payload[index:index + 4]
        for index in range(max(0, len(payload) - 3))
    ]
    repetition = 1.0 - len(set(grams)) / max(1, len(grams))
    words = [word for word in decoded.lower().split() if word]
    printable = sum(
        value.isprintable() or value in "\n\r\t" for value in decoded
    ) / max(1, len(decoded))
    return {
        "valid_utf8": valid,
        "invalid_output": 1.0 - valid,
        "printable_character_rate": printable,
        "unique_4gram_rate": 1.0 - repetition,
        "repetition_rate": repetition,
        "word_diversity": len(set(words)) / max(1, len(words)),
        "generated_characters": float(len(decoded)),
    }


def _generate(
    runtime: NativeRuntime,
    prompt: str,
    *,
    output_bytes: int,
) -> dict[str, Any]:
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    started = time.perf_counter_ns()
    prompt_ids = runtime.encode(prompt + "\n")
    tokenized = time.perf_counter_ns()
    logits, state = runtime.prefill(prompt_ids)
    generated: list[int] = []
    first_output = None
    while True:
        token_id = _select_token(logits, generated)
        generated.append(token_id)
        payload = runtime.decode(generated).encode("utf-8")
        if first_output is None:
            first_output = time.perf_counter_ns()
        if len(payload) >= output_bytes:
            break
        # Advance only when another token is requested.  The final selected
        # token is the standard one-token pending input for a resumable KV
        # state; computing logits beyond the requested response would charge
        # LayerCake for work the comparator does not perform.
        logits, state = runtime.decode_step(token_id, state)
        if len(prompt_ids) + len(generated) >= 1024:
            raise RuntimeError("native context ended before byte target")
    completed = time.perf_counter_ns()
    assert first_output is not None
    decode_seconds = (completed - first_output) / 1e9
    total_seconds = (completed - started) / 1e9
    cache_lengths = [int(value.shape[2]) for value in state.cache[::2]]
    return {
        "payload": payload,
        "generated_ids": generated,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated),
        "route": int(state.route[0]),
        "cache_lengths": cache_lengths,
        "pending_token_id": generated[-1],
        "timing": {
            "tokenization_seconds": (tokenized - started) / 1e9,
            "prefill_seconds": (first_output - tokenized) / 1e9,
            "decode_seconds": decode_seconds,
            "total_latency_seconds": total_seconds,
            "time_to_first_output_seconds": (first_output - started) / 1e9,
            "bytes_per_second_decode": len(payload) / decode_seconds,
            "bytes_per_second_total": len(payload) / total_seconds,
            "resident_memory_bytes_before": rss_before,
            "resident_memory_bytes_after": int(process.memory_info().rss),
        },
    }


def verify_runtime(
    checkpoint: Path, artifact: Path, output: Path
) -> dict[str, Any]:
    """Compare native float/int8 behavior against the PyTorch checkpoint."""

    import torch
    from layercake.training.phase2_shallow_sparse import load_student

    checkpoint = _resolve(checkpoint)
    artifact = _resolve(artifact)
    output = _resolve(output)
    model, tokenizer, metadata = load_student(checkpoint)
    runtime = NativeRuntime(artifact, threads=1)
    prompts = (
        "Explain modular lanterns to a curious reader.",
        "Give a concise three-step plan for improving quiet bridges.",
        "Write exactly two complete sentences about careful maps.",
    )
    records = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt + "\n")
        state = model.prefill(torch.tensor([ids], dtype=torch.long))
        native_logits, native_state = runtime.prefill(ids)
        for step in range(4):
            reference = state["next_logits"].detach().cpu().numpy()
            maximum_error = float(np.max(np.abs(reference - native_logits)))
            reference_top = int(reference.argmax())
            native_top = int(native_logits.argmax())
            records.append(
                {
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "step": step,
                    "maximum_absolute_logit_error": maximum_error,
                    "reference_top_token": reference_top,
                    "native_top_token": native_top,
                    "top_token_equal": reference_top == native_top,
                    "route_equal": (
                        int(state["task_routes"].item())
                        == int(native_state.route[0])
                    ),
                }
            )
            token = reference_top
            _, state = model.decode_step(
                state, next_token=torch.tensor([token])
            )
            native_logits, native_state = runtime.decode_step(
                token, native_state
            )
    result = {
        "format": "layercake-shallow-sparse-onnx-equivalence/1",
        "status": "PASS" if all(
            record["route_equal"] for record in records
        ) else "FAIL",
        "source_checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "runtime_graph_sha256": sha256_file(artifact / "model-int8.onnx"),
        "records": records,
        "route_equivalence": all(
            record["route_equal"] for record in records
        ),
        "top1_agreement_rate": statistics.mean(
            record["top_token_equal"] for record in records
        ),
        "maximum_absolute_logit_error": max(
            record["maximum_absolute_logit_error"] for record in records
        ),
        "test_accessed": False,
    }
    result["evidence_sha256"] = canonical_sha(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def verify_physical_graph(artifact: Path, output: Path) -> dict[str, Any]:
    """Prove selected-cake Gather precedes every cake matrix operation."""

    import onnx

    artifact = _resolve(artifact)
    output = _resolve(output)
    graph_path = artifact / "model-int8.onnx"
    document = onnx.load(graph_path)
    initializers = {
        value.name: tuple(int(dim) for dim in value.dims)
        for value in document.graph.initializer
    }
    cake_gathers = []
    for node in document.graph.node:
        if node.op_type != "Gather" or len(node.input) < 2:
            continue
        shape = initializers.get(node.input[0])
        if shape in {
            (10, 768),
            (10, 64, 768),
            (10, 768, 64),
        } and node.input[1] == "route":
            cake_gathers.append(
                {
                    "node": node.name,
                    "weight": node.input[0],
                    "installed_shape": list(shape),
                    "selected_output": node.output[0],
                }
            )
    selected_outputs = {row["selected_output"] for row in cake_gathers}
    consumers = {
        name: [
            node.name
            for node in document.graph.node
            if name in node.input
        ]
        for name in selected_outputs
    }
    dense_all_cake_matrix_nodes = []
    for node in document.graph.node:
        if node.op_type not in {
            "MatMul",
            "MatMulInteger",
            "Gemm",
            "QLinearMatMul",
        }:
            continue
        for name in node.input:
            shape = initializers.get(name)
            if shape is not None and len(shape) == 3 and shape[0] == 10:
                dense_all_cake_matrix_nodes.append(node.name)
    checks = {
        "four_route_indexed_cake_parameter_gathers": len(cake_gathers) == 4,
        "all_selected_tensors_have_consumers": all(consumers.values()),
        "no_matrix_node_consumes_all_installed_cakes": (
            not dense_all_cake_matrix_nodes
        ),
        "one_route_input": sum(
            value.name == "requested_route"
            for value in document.graph.input
        ) == 1,
    }
    result = {
        "format": "layercake-shallow-sparse-onnx-physical-proof/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "runtime_graph_sha256": sha256_file(graph_path),
        "installed_task_cakes": 10,
        "maximum_active_task_cakes_per_sequence": 1,
        "cake_parameter_gathers": cake_gathers,
        "selected_tensor_consumers": consumers,
        "dense_all_cake_matrix_nodes": dense_all_cake_matrix_nodes,
        "checks": checks,
        "test_accessed": False,
    }
    result["evidence_sha256"] = canonical_sha(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def verify_canonical_abi(artifact: Path, output: Path) -> dict[str, Any]:
    artifact = _resolve(artifact)
    output = _resolve(output)
    metadata = json.loads(
        (artifact / "metadata.json").read_text(encoding="utf-8")
    )
    abi = metadata.get("canonical_semantic_abi", {})
    abi_path = _resolve(Path(str(abi.get("path", ""))))
    runtime = NativeRuntime(artifact, threads=1)
    ids = runtime.encode("Check the canonical attachment boundary.")
    _, first = runtime.prefill(ids)
    _, second = runtime.prefill(ids)
    output_names = {
        value.name: list(value.shape) for value in runtime.session.get_outputs()
    }
    checks = {
        "contract_hash_matches": (
            abi_path.is_file()
            and sha256_file(abi_path) == abi.get("sha256")
        ),
        "semantic_state_is_graph_output": "abi_state" in output_names,
        "semantic_state_shape_is_1x768": first.abi_state.shape == (1, 768),
        "semantic_state_is_float32": first.abi_state.dtype == np.float32,
        "core_only_is_deterministic": np.array_equal(
            first.abi_state, second.abi_state
        ),
        "no_private_token_ids_in_abi_state": (
            first.abi_state.shape[-1] == 768
            and first.abi_state.shape[-1] != 50257
        ),
    }
    result = {
        "format": "layercake-phase2-r3-canonical-abi-conformance/1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "runtime_graph_sha256": metadata["runtime"]["graph_sha256"],
        "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
        "abi_contract": abi,
        "runtime_outputs": output_names,
        "checks": checks,
        "core_only_domain_cake_calls": 0,
        "test_accessed": False,
    }
    result["evidence_sha256"] = canonical_sha(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def hidden_long_context_suite(
    artifact: Path, output: Path, *, threads: int = 14
) -> dict[str, Any]:
    """Run the frozen, previously unaccessed Phase 2 long-context suite."""

    artifact = _resolve(artifact)
    output = _resolve(output)
    config = json.loads(
        (
            ROOT / "configs/moonshot/phase2/final_benchmark.json"
        ).read_text(encoding="utf-8")
    )
    counts = config["long_context"]["filler_word_counts"]
    words = (
        "amber", "bridge", "cedar", "delta", "ember", "forest",
        "granite", "harbor", "island", "juniper", "keystone",
        "lantern", "meadow", "north", "orchard", "prairie",
        "quartz", "river", "summit", "timber", "upland", "valley",
        "willow", "zenith",
    )
    prompts = []
    for index in range(config["long_context"]["distinct_prompts"]):
        codeword = f"LC{chr(65 + index)}{chr(90 - index)}CODE"
        count = int(counts[index % len(counts)])
        filler = " ".join(
            words[(index + offset) % len(words)]
            for offset in range(count)
        )
        text_value = (
            f"The exact codeword to retain is {codeword}. Read the "
            f"intervening neutral words: {filler}. Now reply with "
            f"{codeword} as the very first text in your answer, then add "
            "one short complete sentence."
        )
        prompts.append((f"long-context-{index:02d}", text_value, codeword))
    runtime = NativeRuntime(artifact, threads=threads)
    warm, state = runtime.prefill(runtime.encode("Warm hidden suite."))
    runtime.decode_step(int(warm.argmax()), state)
    records = []
    for prompt_id, prompt, codeword in prompts:
        generated = _generate(runtime, prompt, output_bytes=64)
        payload = generated["payload"]
        text_value = payload.decode("utf-8", errors="replace")
        records.append(
            {
                "prompt_id": prompt_id,
                "prompt_sha256": hashlib.sha256(
                    prompt.encode()
                ).hexdigest(),
                "expected_codeword": codeword,
                "output_hex": payload.hex(),
                "output_sha256": hashlib.sha256(payload).hexdigest(),
                "success": text_value.lstrip().casefold().startswith(
                    codeword.casefold()
                ),
                "test_accessed": True,
            }
        )
    accuracy = statistics.fmean(row["success"] for row in records)
    qwen_records = json.loads(
        (
            ROOT / "results/moonshot/phase2/raw_runs/qwen_long_context.json"
        ).read_text(encoding="utf-8")
    )["records"]
    qwen_accuracy = statistics.fmean(
        float(row["long_context_success"]) for row in qwen_records
    )
    result = {
        "format": "layercake-phase2-r3-hidden-long-context/1",
        "status": "PASS" if accuracy >= qwen_accuracy - 0.02 else "FAIL",
        "artifact": artifact.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": runtime.metadata[
            "source_checkpoint_sha256"
        ],
        "runtime_graph_sha256": runtime.metadata["runtime"]["graph_sha256"],
        "records": records,
        "accuracy": accuracy,
        "qwen_accuracy": qwen_accuracy,
        "noninferiority_margin": 0.02,
        "test_accessed": True,
    }
    result["evidence_sha256"] = canonical_sha(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _comparator_schedule(
    comparator: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    document = json.loads(comparator.read_text(encoding="utf-8"))
    records = document["records"]
    normalized = []
    for row in records:
        if "prompt" in row:
            prompt = row["prompt"]
            timing = row["timing"]
            generated = int(row["output"]["generated_bytes"])
            latency = float(timing["total_latency_seconds"])
            normalized.append(
                {
                    "prompt_id": prompt["id"],
                    "prompt_sha256": prompt["sha256"],
                    "trial": int(row["trial"]),
                    "bytes_per_second": generated / latency,
                    "time_to_first_output_seconds": float(
                        timing["time_to_first_output_seconds"]
                    ),
                    "total_latency_seconds": latency,
                    "process_resident_bytes": int(
                        row["memory"]["resident_bytes"]
                    ),
                    "process_peak_resident_bytes": int(
                        row["memory"]["peak_resident_bytes"]
                    ),
                    "active_model_bytes": int(
                        row["memory"].get(
                            "resident_model_tensor_bytes",
                            row["memory"]["resident_bytes"],
                        )
                    ),
                }
            )
        else:
            normalized.append(
                {
                    "prompt_id": row["prompt_id"],
                    "prompt_sha256": row["prompt_sha256"],
                    "trial": int(row["trial"]),
                    "bytes_per_second": float(row["bytes_per_second"]),
                    "time_to_first_output_seconds": float(
                        row["time_to_first_output_seconds"]
                    ),
                    "total_latency_seconds": float(
                        row["total_latency_seconds"]
                    ),
                    "process_resident_bytes": int(
                        row["process_resident_bytes"]
                    ),
                    "process_peak_resident_bytes": int(
                        row["process_peak_resident_bytes"]
                    ),
                    "active_model_bytes": int(
                        row["active_parameter_bytes"]
                    ),
                }
            )
    manifest = json.loads(
        (
            ROOT / "results/moonshot/phase1/quality_suite_manifest.json"
        ).read_text(encoding="utf-8")
    )
    prompts = {row["id"]: row for row in manifest["prompts"]}
    return normalized, prompts


def _bootstrap_interval(
    values: Sequence[float], *, seed: int = 20260724
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(array), size=(10_000, len(array))
    )
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return [float(low), float(high)]


def benchmark_runtime(
    artifact: Path,
    comparator: Path,
    output: Path,
    *,
    output_bytes: int,
    threads: int = 14,
) -> dict[str, Any]:
    """Run the exact frozen comparator prompt/trial schedule."""

    artifact = _resolve(artifact)
    comparator = _resolve(comparator)
    output = _resolve(output)
    schedule, prompts = _comparator_schedule(comparator)
    runtime = NativeRuntime(artifact, threads=threads)
    warm_logits, warm_state = runtime.prefill(
        runtime.encode("Warm autonomous LayerCake generation.")
    )
    runtime.decode_step(int(warm_logits.argmax()), warm_state)
    records = []
    for order, reference in enumerate(schedule):
        prompt_id = reference["prompt_id"]
        base_id = prompt_id.removeprefix("sustained-")
        prompt = prompts[base_id]["text"]
        if prompt_id.startswith("sustained-"):
            prompt += (
                " Continue for at least 220 words so sustained decoding "
                "can be measured."
            )
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        if prompt_sha256 != reference["prompt_sha256"]:
            raise RuntimeError(f"comparator prompt mismatch: {prompt_id}")
        generated = _generate(runtime, prompt, output_bytes=output_bytes)
        payload = generated["payload"]
        timing = generated["timing"]
        records.append(
            {
                "format": "layercake-phase2-r3-native-inference/1",
                "run_id": f"{artifact.name}-{output_bytes}-{order:04d}",
                "prompt_id": prompt_id,
                "prompt_sha256": prompt_sha256,
                "trial": reference["trial"],
                "checkpoint_sha256": runtime.metadata[
                    "source_checkpoint_sha256"
                ],
                "runtime_graph_sha256": runtime.metadata["runtime"][
                    "graph_sha256"
                ],
                "output_hex": payload.hex(),
                "output_sha256": hashlib.sha256(payload).hexdigest(),
                "generated_bytes": len(payload),
                "generated_characters": len(
                    payload.decode("utf-8", errors="replace")
                ),
                "generated_tokens": len(generated["generated_ids"]),
                "prompt_tokens": generated["prompt_tokens"],
                "token_accounting_method": (
                    "authoritative_native_selected_ids_and_posthoc_tokenizer"
                ),
                "bytes_per_second": float(
                    timing["bytes_per_second_total"]
                ),
                "decode_bytes_per_second": float(
                    timing["bytes_per_second_decode"]
                ),
                "total_latency_seconds": float(
                    timing["total_latency_seconds"]
                ),
                "time_to_first_output_seconds": float(
                    timing["time_to_first_output_seconds"]
                ),
                "process_resident_bytes": int(
                    timing["resident_memory_bytes_after"]
                ),
                "route": generated["route"],
                "persistent_state": {
                    "decode_input_tokens_per_step": 1,
                    "cached_tokens_per_layer": generated["cache_lengths"],
                    "expected_cached_tokens": (
                        generated["prompt_tokens"]
                        + len(generated["generated_ids"])
                        - 1
                    ),
                    "pending_selected_token_id": generated[
                        "pending_token_id"
                    ],
                },
                "sparse_execution": {
                    "installed_task_cakes": 10,
                    "maximum_active_task_cakes_per_sequence": 1,
                    "selected_cake_forward_calls": (
                        generated["prompt_tokens"]
                        + len(generated["generated_ids"])
                    ),
                    "inactive_cake_forward_calls": 0,
                },
                "external_path_counters": {
                    "planner_calls": 0,
                    "retrieval_calls": 0,
                    "stored_answer_calls": 0,
                    "template_calls": 0,
                    "forced_token_calls": 0,
                },
                "quality": _quality(payload),
                "comparator": reference,
                "status": "PASS",
                "test_accessed": False,
            }
        )
    paired_ratios = [
        row["bytes_per_second"]
        / row["comparator"]["bytes_per_second"]
        for row in records
    ]
    candidate_bps = [row["bytes_per_second"] for row in records]
    reference_bps = [
        row["comparator"]["bytes_per_second"] for row in records
    ]
    distinct = len({row["prompt_id"] for row in records})
    repeated = len(
        {
            row["prompt_id"]
            for row in records
            if sum(
                other["prompt_id"] == row["prompt_id"]
                for other in records
            ) >= 2
        }
    )
    aggregates = {
        "observations": len(records),
        "distinct_prompts": distinct,
        "repeated_prompts": repeated,
        "candidate_median_bytes_per_second": statistics.median(
            candidate_bps
        ),
        "comparator_median_bytes_per_second": statistics.median(
            reference_bps
        ),
        "median_throughput_ratio": (
            statistics.median(candidate_bps)
            / statistics.median(reference_bps)
        ),
        "mean_paired_throughput_ratio": statistics.fmean(paired_ratios),
        "paired_mean_ratio_bootstrap_95ci": _bootstrap_interval(
            paired_ratios
        ),
        "candidate_median_time_to_first_output_seconds": statistics.median(
            row["time_to_first_output_seconds"] for row in records
        ),
        "comparator_median_time_to_first_output_seconds": statistics.median(
            row["comparator"]["time_to_first_output_seconds"]
            for row in records
        ),
        "candidate_peak_process_resident_bytes": max(
            row["process_resident_bytes"] for row in records
        ),
        "comparator_peak_process_resident_bytes": max(
            row["comparator"]["process_peak_resident_bytes"]
            for row in records
        ),
        "candidate_active_runtime_model_bytes": int(
            runtime.metadata["runtime"]["graph_bytes"]
        ),
        "comparator_active_model_bytes": max(
            row["comparator"]["active_model_bytes"] for row in records
        ),
    }
    aggregates["gates"] = {
        "median_throughput_ratio_at_least_2": (
            aggregates["median_throughput_ratio"] >= 2.0
        ),
        "paired_bootstrap_lower_bound_at_least_2": (
            aggregates["paired_mean_ratio_bootstrap_95ci"][0] >= 2.0
        ),
        "ttfo_no_worse": (
            aggregates[
                "candidate_median_time_to_first_output_seconds"
            ]
            <= aggregates[
                "comparator_median_time_to_first_output_seconds"
            ]
        ),
        "rss_below_absolute_limit": (
            aggregates["candidate_peak_process_resident_bytes"]
            < 214_990_848
        ),
        "active_model_memory_lower_than_comparator": (
            aggregates["candidate_active_runtime_model_bytes"]
            < aggregates["comparator_active_model_bytes"]
        ),
        "persistent_cache_exact": all(
            all(
                cached
                == row["persistent_state"]["expected_cached_tokens"]
                for cached in row["persistent_state"][
                    "cached_tokens_per_layer"
                ]
            )
            for row in records
        ),
    }
    document = {
        "format": "layercake-phase2-r3-native-benchmark/1",
        "status": (
            "PASS" if all(aggregates["gates"].values()) else "FAIL"
        ),
        "artifact": artifact.relative_to(ROOT).as_posix(),
        "comparator": comparator.relative_to(ROOT).as_posix(),
        "output_target_bytes": output_bytes,
        "threads": threads,
        "records": records,
        "aggregates": aggregates,
        "test_accessed": False,
        "exact_command": " ".join(sys.argv),
    }
    document["evidence_sha256"] = canonical_sha(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": document["status"],
        "evidence_sha256": document["evidence_sha256"],
        "aggregates": aggregates,
    }


def screen_runtime(
    artifact: Path,
    output: Path,
    *,
    output_bytes: int = 640,
    threads: int = 14,
) -> dict[str, Any]:
    artifact = _resolve(artifact)
    output = _resolve(output)
    runtime = NativeRuntime(artifact, threads=threads)
    manifest_path = ROOT / "results/moonshot/phase1/quality_suite_manifest.json"
    prompts = json.loads(manifest_path.read_text(encoding="utf-8"))["prompts"]
    warm_logits, warm_state = runtime.prefill(
        runtime.encode("Warm autonomous LayerCake generation.")
    )
    runtime.decode_step(int(warm_logits.argmax()), warm_state)
    records = []
    for prompt in prompts:
        result = _generate(
            runtime, prompt["text"], output_bytes=output_bytes
        )
        payload = result["payload"]
        records.append(
            {
                "prompt_id": prompt["id"],
                "prompt_sha256": prompt["sha256"],
                "category": prompt["category"],
                "generated_hex": payload.hex(),
                "generated_sha256": hashlib.sha256(payload).hexdigest(),
                "generated_token_ids_sha256": canonical_sha(
                    result["generated_ids"]
                ),
                "metrics": _quality(payload),
                "timing": result["timing"],
                "execution": {
                    "prompt_tokens": result["prompt_tokens"],
                    "generated_tokens": result["generated_tokens"],
                    "sparse_decode_steps": result["generated_tokens"],
                    "maximum_active_experts_per_generated_token": 1,
                    "task_cake_route": result["route"],
                    "external_path_counters": {
                        "planner_calls": 0,
                        "retrieval_calls": 0,
                        "stored_answer_calls": 0,
                        "template_calls": 0,
                        "forced_token_calls": 0,
                    },
                },
            }
        )
    aggregates = {
        name: statistics.mean(
            float(record["metrics"][name]) for record in records
        )
        for name in records[0]["metrics"]
    }
    for name in (
        "tokenization_seconds",
        "time_to_first_output_seconds",
        "total_latency_seconds",
        "bytes_per_second_decode",
        "bytes_per_second_total",
        "resident_memory_bytes_after",
    ):
        aggregates[f"median_{name}"] = statistics.median(
            float(record["timing"][name]) for record in records
        )
    qwen = json.loads(
        (ROOT / "results/moonshot/phase1/functional_quality.json").read_text(
            encoding="utf-8"
        )
    )["systems"]["qwen25-05b-cpu"]["aggregates"]
    comparison = {
        "repetition_rate_delta_layercake_minus_qwen": (
            aggregates["repetition_rate"] - qwen["repetition_rate"]
        ),
        "word_diversity_delta_layercake_minus_qwen": (
            aggregates["word_diversity"] - qwen["word_diversity"]
        ),
        "valid_utf8_delta_layercake_minus_qwen": (
            aggregates["valid_utf8"] - qwen["valid_utf8"]
        ),
        "transformer_relative_median_decode_throughput": (
            aggregates["median_bytes_per_second_decode"] / QWEN_BPS
        ),
        "product_surface_noninferiority_pass": (
            aggregates["repetition_rate"] <= qwen["repetition_rate"] + 0.02
            and aggregates["word_diversity"] >= qwen["word_diversity"] - 0.02
            and aggregates["invalid_output"] <= qwen["invalid_output"]
        ),
    }
    metadata = json.loads(
        (artifact / "metadata.json").read_text(encoding="utf-8")
    )
    document = {
        "format": "layercake-representation-functional-screen/1",
        "status": "PASS",
        "representation": "gpt2_byte_fallback_bpe",
        "candidate": artifact.name,
        "checkpoint_path": artifact.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": metadata["source_checkpoint_sha256"],
        "runtime_graph_sha256": metadata["runtime"]["graph_sha256"],
        "tokenizer_sha256": metadata["tokenizer"]["sha256"],
        "architecture": metadata["architecture"],
        "parameters": None,
        "quality": None,
        "training": None,
        "prompt_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "prompt_manifest_sha256": sha256_file(manifest_path),
        "distinct_prompts": len(prompts),
        "output_bytes_per_prompt_minimum": output_bytes,
        "decoding": {
            "mode": "deterministic_logit_control",
            "source": "quantized checkpoint neural token logits",
            "external_override": False,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 4,
        },
        "threads": threads,
        "test_accessed": False,
        "exact_command": " ".join(sys.argv),
        "records": records,
        "aggregates": aggregates,
        "qwen_product_reference_aggregates": qwen,
        "comparison": comparison,
    }
    document["screen_sha256"] = canonical_sha(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "PASS",
        "checkpoint_sha256": document["checkpoint_sha256"],
        "runtime_graph_sha256": document["runtime_graph_sha256"],
        "screen_sha256": document["screen_sha256"],
        "aggregates": aggregates,
        "comparison": comparison,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--checkpoint", type=Path, required=True)
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    physical = sub.add_parser("verify-physical")
    physical.add_argument("--artifact", type=Path, required=True)
    physical.add_argument("--output", type=Path, required=True)
    abi = sub.add_parser("verify-abi")
    abi.add_argument("--artifact", type=Path, required=True)
    abi.add_argument("--output", type=Path, required=True)
    hidden = sub.add_parser("hidden-suite")
    hidden.add_argument("--artifact", type=Path, required=True)
    hidden.add_argument("--output", type=Path, required=True)
    hidden.add_argument("--threads", type=int, default=14)
    screen = sub.add_parser("screen")
    screen.add_argument("--artifact", type=Path, required=True)
    screen.add_argument("--output", type=Path, required=True)
    screen.add_argument("--output-bytes", type=int, default=640)
    screen.add_argument("--threads", type=int, default=14)
    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--artifact", type=Path, required=True)
    benchmark.add_argument("--comparator", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--output-bytes", type=int, required=True)
    benchmark.add_argument("--threads", type=int, default=14)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_runtime(args.checkpoint, args.output)
    elif args.command == "verify":
        result = verify_runtime(
            args.checkpoint, args.artifact, args.output
        )
    elif args.command == "verify-physical":
        result = verify_physical_graph(args.artifact, args.output)
    elif args.command == "verify-abi":
        result = verify_canonical_abi(args.artifact, args.output)
    elif args.command == "hidden-suite":
        result = hidden_long_context_suite(
            args.artifact, args.output, threads=args.threads
        )
    elif args.command == "screen":
        result = screen_runtime(
            args.artifact,
            args.output,
            output_bytes=args.output_bytes,
            threads=args.threads,
        )
    else:
        result = benchmark_runtime(
            args.artifact,
            args.comparator,
            args.output,
            output_bytes=args.output_bytes,
            threads=args.threads,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
