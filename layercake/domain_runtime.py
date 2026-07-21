from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable


_QA_RE = re.compile(
    r"Question:\s*(?P<question>.*?)\s*Answer:\s*(?P<answer>.*)",
    flags=re.IGNORECASE,
)

_WORD_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a",
    "about",
    "after",
    "again",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "do",
    "does",
    "for",
    "from",
    "get",
    "give",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "me",
    "my",
    "next",
    "of",
    "on",
    "or",
    "player",
    "plan",
    "response",
    "should",
    "the",
    "then",
    "to",
    "up",
    "what",
    "when",
    "with",
}

_SYNONYMS = {
    "alive": "survive",
    "archers": "archer",
    "ambushed": "ambush",
    "attack": "combat",
    "battle": "combat",
    "bosses": "boss",
    "brutes": "brute",
    "cooldowns": "cooldown",
    "down": "downed",
    "downed": "downed",
    "enemies": "threat",
    "enemy": "threat",
    "foe": "threat",
    "foes": "threat",
    "heal": "mend",
    "healing": "mend",
    "hit": "damage",
    "hits": "damage",
    "mistakes": "mistake",
    "opening": "step",
    "positioning": "position",
    "recovering": "recover",
    "safe": "safest",
    "stability": "stabilize",
    "stable": "stabilize",
    "threats": "threat",
}


@dataclass(frozen=True)
class InstructionAlias:
    question: str
    answer: str
    tokens: frozenset[str]


@dataclass(frozen=True)
class InstructionAliasMatch:
    alias: InstructionAlias
    score: float
    overlap: int
    query_tokens: frozenset[str]


@dataclass(frozen=True)
class PortableDomainChunk:
    source: str
    chunk_id: int
    text: str
    tokens: frozenset[str]


@dataclass(frozen=True)
class PortableDomainMatch:
    chunk: PortableDomainChunk
    score: float
    overlap: int
    query_tokens: frozenset[str]


def _stem(token: str) -> str:
    if token in _SYNONYMS:
        return _SYNONYMS[token]
    if len(token) > 5 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 6 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 5 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 4 and token.endswith("s"):
        token = token[:-1]
    return _SYNONYMS.get(token, token)


def extract_question(text: str) -> str:
    match = _QA_RE.search(text)
    if match:
        return match.group("question").strip()
    return text.strip()


def normalize_instruction_tokens(text: str) -> frozenset[str]:
    question = extract_question(text).lower()
    tokens = {
        _stem(token)
        for token in _WORD_RE.findall(question)
        if token not in _STOPWORDS and (len(token) > 1 or token.isdigit())
    }
    return frozenset(token for token in tokens if token and token not in _STOPWORDS)


def normalize_domain_tokens(text: str) -> frozenset[str]:
    return normalize_instruction_tokens(text)


def parse_instruction_aliases(text: str) -> list[InstructionAlias]:
    aliases: list[InstructionAlias] = []
    seen: set[tuple[str, str]] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _QA_RE.fullmatch(stripped)
        if not match:
            continue
        question = match.group("question").strip()
        answer = match.group("answer").strip()
        key = (question.casefold(), answer.casefold())
        if key in seen:
            continue
        tokens = normalize_instruction_tokens(question)
        if not tokens:
            continue
        seen.add(key)
        aliases.append(InstructionAlias(question=question, answer=answer, tokens=tokens))
    return aliases


def load_instruction_aliases(paths: Iterable[Path]) -> list[InstructionAlias]:
    aliases: list[InstructionAlias] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for alias in parse_instruction_aliases(Path(path).read_text(encoding="utf-8")):
            key = (alias.question.casefold(), alias.answer.casefold())
            if key in seen:
                continue
            seen.add(key)
            aliases.append(alias)
    return aliases


def instruction_alias_score(
    query_tokens: frozenset[str],
    candidate_tokens: frozenset[str],
) -> tuple[float, int]:
    if not query_tokens or not candidate_tokens:
        return 0.0, 0
    overlap = len(query_tokens & candidate_tokens)
    if overlap == 0:
        return 0.0, 0
    overlap_coeff = overlap / max(min(len(query_tokens), len(candidate_tokens)), 1)
    jaccard = overlap / max(len(query_tokens | candidate_tokens), 1)
    return (0.70 * overlap_coeff) + (0.30 * jaccard), overlap


def match_instruction_alias(
    prompt: str,
    aliases: Iterable[InstructionAlias],
    *,
    threshold: float = 0.34,
    min_overlap: int = 1,
) -> InstructionAliasMatch | None:
    query_tokens = normalize_instruction_tokens(prompt)
    best: InstructionAliasMatch | None = None
    for alias in aliases:
        score, overlap = instruction_alias_score(query_tokens, alias.tokens)
        if overlap < min_overlap or score < threshold:
            continue
        match = InstructionAliasMatch(
            alias=alias,
            score=float(score),
            overlap=int(overlap),
            query_tokens=query_tokens,
        )
        if best is None or match.score > best.score:
            best = match
    return best


def render_instruction_alias_answer(
    prompt: str,
    aliases: Iterable[InstructionAlias],
    *,
    max_new_bytes: int,
    threshold: float = 0.34,
    min_overlap: int = 1,
) -> tuple[str, InstructionAliasMatch] | None:
    match = match_instruction_alias(
        prompt,
        aliases,
        threshold=threshold,
        min_overlap=min_overlap,
    )
    if match is None:
        return None
    answer = (" " + match.alias.answer).encode("utf-8", errors="replace")[:max_new_bytes]
    return answer.decode("utf-8", errors="replace"), match


def _split_portable_domain_text(text: str, *, max_chunk_chars: int) -> list[str]:
    units: list[str] = []
    current: list[str] = []
    current_len = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if current:
                units.append(" ".join(current).strip())
                current = []
                current_len = 0
            continue
        if line.startswith("#"):
            if current:
                units.append(" ".join(current).strip())
                current = []
                current_len = 0
            current.append(line.lstrip("#").strip())
            current_len = len(current[-1])
            continue
        sentence_parts = re.split(r"(?<=[.!?])\s+", line)
        for sentence in sentence_parts:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current and current_len + 1 + len(sentence) > max_chunk_chars:
                units.append(" ".join(current).strip())
                current = []
                current_len = 0
            current.append(sentence)
            current_len += len(sentence) + 1
    if current:
        units.append(" ".join(current).strip())
    return [unit for unit in units if unit]


def parse_portable_domain_chunks(
    text: str,
    *,
    source: str = "memory",
    max_chunk_chars: int = 180,
) -> list[PortableDomainChunk]:
    chunks: list[PortableDomainChunk] = []
    for index, chunk_text in enumerate(
        _split_portable_domain_text(text, max_chunk_chars=max_chunk_chars)
    ):
        tokens = normalize_domain_tokens(chunk_text)
        if not tokens:
            continue
        chunks.append(
            PortableDomainChunk(
                source=source,
                chunk_id=index,
                text=chunk_text,
                tokens=tokens,
            )
        )
    return chunks


def load_portable_domain_chunks(
    paths: Iterable[Path],
    *,
    max_chunk_chars: int = 180,
) -> list[PortableDomainChunk]:
    chunks: list[PortableDomainChunk] = []
    seen_text: set[str] = set()
    for path in paths:
        resolved = Path(path)
        parsed = parse_portable_domain_chunks(
            resolved.read_text(encoding="utf-8"),
            source=str(resolved),
            max_chunk_chars=max_chunk_chars,
        )
        for chunk in parsed:
            key = chunk.text.casefold()
            if key in seen_text:
                continue
            seen_text.add(key)
            chunks.append(chunk)
    return chunks


def portable_domain_score(
    query_tokens: frozenset[str],
    candidate_tokens: frozenset[str],
) -> tuple[float, int]:
    if not query_tokens or not candidate_tokens:
        return 0.0, 0
    overlap = len(query_tokens & candidate_tokens)
    if overlap == 0:
        return 0.0, 0
    overlap_coeff = overlap / max(min(len(query_tokens), len(candidate_tokens)), 1)
    jaccard = overlap / max(len(query_tokens | candidate_tokens), 1)
    coverage = overlap / max(len(query_tokens), 1)
    return (0.50 * overlap_coeff) + (0.30 * coverage) + (0.20 * jaccard), overlap


def match_portable_domain_chunk(
    prompt: str,
    chunks: Iterable[PortableDomainChunk],
    *,
    threshold: float = 0.30,
    min_overlap: int = 2,
) -> PortableDomainMatch | None:
    query_tokens = normalize_domain_tokens(prompt)
    best: PortableDomainMatch | None = None
    for chunk in chunks:
        score, overlap = portable_domain_score(query_tokens, chunk.tokens)
        if overlap < min_overlap or score < threshold:
            continue
        match = PortableDomainMatch(
            chunk=chunk,
            score=float(score),
            overlap=int(overlap),
            query_tokens=query_tokens,
        )
        if best is None or match.score > best.score:
            best = match
    return best


def render_portable_domain_answer(
    prompt: str,
    chunks: Iterable[PortableDomainChunk],
    *,
    max_new_bytes: int,
    threshold: float = 0.30,
    min_overlap: int = 2,
) -> tuple[str, PortableDomainMatch] | None:
    match = match_portable_domain_chunk(
        prompt,
        chunks,
        threshold=threshold,
        min_overlap=min_overlap,
    )
    if match is None:
        return None
    text = match.chunk.text.strip()
    if not text.endswith((".", "!", "?")):
        text += "."
    answer = (" " + text).encode("utf-8", errors="replace")[:max_new_bytes]
    return answer.decode("utf-8", errors="replace"), match
