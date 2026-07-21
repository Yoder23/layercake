from __future__ import annotations

import re
from collections.abc import Iterable


DEFAULT_STOP_SEQUENCES: tuple[str, ...] = (
    "\x00",
    "\nQuestion:",
    "\nQ:",
    "\nPrompt:",
    "\nPlayer:",
    "\nCompanion:",
    "\nAnswer:",
    "\n{",
    "\r\nQuestion:",
    "\r\n{",
)

_CONTROL_RE = re.compile(r"[^\x09\x0a\x0d\x20-\x7e]")
_SPACE_RE = re.compile(r"[ \t\r\n]+")


def trim_at_stop_sequence(
    text: str,
    *,
    stop_sequences: Iterable[str] = DEFAULT_STOP_SEQUENCES,
) -> tuple[str, str | None]:
    """Trim generated text at the earliest production stop boundary.

    Byte-level checkpoints can continue into training-corpus structure such as
    a new ``Question:`` line or JSON fragment. For a game companion runtime,
    those continuations are not useful assistant output, so the integration
    boundary should be deterministic and auditable.
    """

    best_index: int | None = None
    best_stop: str | None = None
    for stop in stop_sequences:
        if not stop:
            continue
        index = text.find(stop)
        if index < 0:
            continue
        if best_index is None or index < best_index:
            best_index = index
            best_stop = stop
    if best_index is None:
        return text, None
    return text[:best_index], best_stop


def finalize_companion_text(
    text: str,
    *,
    stop_sequences: Iterable[str] = DEFAULT_STOP_SEQUENCES,
) -> tuple[str, dict[str, object]]:
    """Return production-safe companion text plus auditable trim metadata."""

    trimmed, stop_sequence = trim_at_stop_sequence(
        text,
        stop_sequences=stop_sequences,
    )
    printable = _CONTROL_RE.sub("", trimmed)
    compact = _SPACE_RE.sub(" ", printable).strip()
    return compact, {
        "raw_text": text,
        "stop_sequence": stop_sequence,
        "trimmed": stop_sequence is not None or compact != text.strip(),
        "raw_bytes": len(text.encode("utf-8", errors="replace")),
        "final_bytes": len(compact.encode("utf-8", errors="replace")),
    }
