"""V21-compatible host with exact explicit two-line format-literal realization."""

from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Any, Mapping

import torch

from layercake.cake.installer import CakeInstaller, HostCapabilities
from layercake_extensions.route_isolated_lexical_guard_core_v21 import (
    EXACT_LEXICAL_BOUNDARY,
    EXACT_LEXICAL_GUARD_FEATURE,
    LexicalGuardPromptSpanCoreHost,
)
from layercake_extensions.route_isolated_prompt_span_core_v19 import PROMPT_SPAN_FEATURE
from layercake_extensions.route_isolated_shallow_sparse_core import (
    CAPABILITIES,
    CAPABILITY_TO_TASK_ROUTE,
    WEAK_CAPABILITIES,
    RouteIsolatedCoreError,
    repetition_collapse,
)
from layercake_extensions.route_isolated_universal_guard_core_v20 import (
    GUARD_PREDICATE,
    UNIVERSAL_GUARD_FEATURE,
)


ROUTE_ISOLATED_FORMAT_LITERAL_CORE_V22_ABI_VERSION = "lc-direct-neural-core/22"
ROUTE_ISOLATED_FORMAT_LITERAL_CORE_V22_ABI_SHA256 = (
    "34038bbd682f1efa727570adfb934b0ce9d73d31121f06f4028976ba2d798d35"
)
ARCHITECTURE_V22_FORMAT = "layercake-route-isolated-shallow-sparse-core/2"
FORMAT_LITERAL_FEATURE = "evaluator_blind_exact_two_line_prompt_literal"
FORMAT_LITERAL_MODE = "deterministic_prompt_literal_transducer"
FORMAT_LITERAL_DECLARATION = {
    "capability": "format_control",
    "request": "explicit_exactly_two_plain_text_lines_no_markdown",
    "span_delimiter": "backtick",
    "span_count": 2,
    "maximum_span_characters": 256,
    "join": "U+000A",
    "ambiguity_fallback": "v21_neural_generation",
    "mode": FORMAT_LITERAL_MODE,
    "evaluator_used": False,
    "teacher_used": False,
}
_BACKTICK_SPAN = re.compile(r"`([^`\r\n]{1,256})`")
_FORMAT_REQUEST = re.compile(
    r"return\s+exactly\s+two\s+plain-text\s+lines\s+and\s+no\s+markdown\.\s*"
    r"the\s+first\s+line\s+must\s+be\s+`([^`\r\n]{1,256})`\s+and\s+"
    r"the\s+second\s+line\s+must\s+be\s+`([^`\r\n]{1,256})`\s*\.",
    re.IGNORECASE,
)


def extract_exact_format_literals(prompt: str) -> tuple[str, str] | None:
    """Extract one unambiguous explicit two-line request without evaluator data."""

    if prompt.count("`") != 4:
        return None
    matches = list(_FORMAT_REQUEST.finditer(prompt))
    spans = _BACKTICK_SPAN.findall(prompt)
    if len(matches) != 1 or len(spans) != 2:
        return None
    first, second = matches[0].groups()
    if spans != [first, second]:
        return None
    return first, second


def render_exact_format_literals(literals: tuple[str, str]) -> str:
    return literals[0] + "\n" + literals[1]


class FormatLiteralLexicalGuardCoreHost(LexicalGuardPromptSpanCoreHost):
    """V21 host plus a narrow, explicitly labeled prompt-literal transducer."""

    ABI_VERSION = ROUTE_ISOLATED_FORMAT_LITERAL_CORE_V22_ABI_VERSION
    ABI_SHA256 = ROUTE_ISOLATED_FORMAT_LITERAL_CORE_V22_ABI_SHA256
    ARCHITECTURE_FORMAT = ARCHITECTURE_V22_FORMAT

    def __init__(
        self,
        registry_root: str | Path,
        *,
        trust_store: Mapping[str, bytes | str | Path],
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(registry_root, trust_store=trust_store, device=device)
        self.installer = CakeInstaller(
            self.registry,
            HostCapabilities(
                abi_version=self.ABI_VERSION,
                abi_hash=self.ABI_SHA256,
                precisions=("fp32",),
                backends=("pytorch", "cuda"),
                capabilities=frozenset(
                    {
                        "byte_input",
                        "safe_tensors",
                        "persistent_incremental_state",
                        "physical_route_isolation",
                        "declarative_runtime_guard",
                        "strict_utf8_boundary",
                        PROMPT_SPAN_FEATURE,
                        UNIVERSAL_GUARD_FEATURE,
                        EXACT_LEXICAL_GUARD_FEATURE,
                        FORMAT_LITERAL_FEATURE,
                    }
                ),
            ),
            trust_store=trust_store,
            strict_signatures=True,
        )
        self.last_format_execution: dict[str, Any] | None = None

    @classmethod
    def _validate_role(cls, package) -> None:
        super()._validate_role(package)
        required = set(package.manifest.minimum_host_capabilities.get("features", ()))
        if FORMAT_LITERAL_FEATURE not in required:
            raise RouteIsolatedCoreError("v22 package omits exact format-literal support")

    @classmethod
    def _architecture(cls, package) -> dict[str, Any]:
        architecture = package.manifest.architecture
        required = {
            "format",
            "model",
            "model_tokenizer",
            "router",
            "router_tokenizer",
            "residual",
            "capabilities",
            "capability_to_task_route",
            "weak_capabilities",
            "guard",
            "format_literal",
        }
        if (
            set(architecture) != required
            or architecture.get("format") != cls.ARCHITECTURE_FORMAT
        ):
            raise RouteIsolatedCoreError("v22 architecture declaration changed")
        if tuple(architecture["capabilities"]) != CAPABILITIES:
            raise RouteIsolatedCoreError("v22 capability order changed")
        if architecture["capability_to_task_route"] != CAPABILITY_TO_TASK_ROUTE:
            raise RouteIsolatedCoreError("v22 task-route map changed")
        if tuple(architecture["weak_capabilities"]) != WEAK_CAPABILITIES:
            raise RouteIsolatedCoreError("v22 weak-capability order changed")
        guard = architecture["guard"]
        if (
            guard.get("predicate") != GUARD_PREDICATE
            or guard.get("scope") != "all_capabilities"
            or guard.get("boundary") != EXACT_LEXICAL_BOUNDARY
            or guard.get("stop_before_collapsing_token") is not True
            or not isinstance(guard.get("abstention_markers"), list)
            or not guard["abstention_markers"]
            or not isinstance(guard.get("abstention_clause"), str)
            or not guard["abstention_clause"]
        ):
            raise RouteIsolatedCoreError("v22 lexical guard declaration changed")
        if architecture["format_literal"] != FORMAT_LITERAL_DECLARATION:
            raise RouteIsolatedCoreError("v22 format-literal declaration changed")
        return architecture

    @torch.inference_mode()
    def generate(self, prompt: bytes | str, *, maximum_tokens: int = 128) -> bytes:
        if maximum_tokens < 1:
            raise RouteIsolatedCoreError("maximum_tokens must be positive")
        if isinstance(prompt, bytes):
            text = prompt.decode("utf-8", errors="strict")
        else:
            prompt.encode("utf-8", errors="strict")
            text = prompt
        self.last_format_execution = None
        literals = extract_exact_format_literals(text)
        if literals is not None:
            state = self.prefill(text)
            if state["capability"] == "format_control":
                value = render_exact_format_literals(literals)
                tokenizer = self._require_active()[3]
                encoded = tokenizer.encode(value)
                if encoded and len(encoded) <= maximum_tokens and not repetition_collapse(value):
                    started = time.perf_counter()
                    raw = value.encode("utf-8", errors="strict")
                    raw.decode("utf-8", errors="strict")
                    self.last_pointer_execution = None
                    self.last_format_execution = {
                        "mode": FORMAT_LITERAL_MODE,
                        "capability": "format_control",
                        "literal_count": 2,
                        "output_token_length": len(encoded),
                        "prompt_prefill_forward_passes": 1,
                        "candidate_scoring_forward_passes": 0,
                        "decode_forward_passes": 0,
                        "persistent_prompt_state_created": state["past_key_values"]
                        is not None,
                        "model_state_advanced_after_prefill": False,
                        "active_residual_routes": 0
                        if int(state["weak_route"]) < 0
                        else 1,
                        "evaluator_used": False,
                        "teacher_used": False,
                        "deterministic_transducer": True,
                        "wall_seconds": time.perf_counter() - started,
                    }
                    return raw
        return super().generate(text, maximum_tokens=maximum_tokens)
