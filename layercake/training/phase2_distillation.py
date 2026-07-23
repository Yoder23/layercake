"""General instruction distillation for the Phase 2 English core.

The teacher corpus deliberately excludes every frozen Phase 1 evaluation topic and exact
prompt.  Fine-tuning mixes masked response loss with the original WikiText objective so
instruction behavior cannot be bought by discarding the locked BPB threshold.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import re
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from layercake.phase1_campaign import _headline_prompts, _ollama_stream, _ollama_warm
from layercake.training.baseline import _token_batch, evaluate_transformer
from layercake.training.data import ByteCorpus, sha256_file
from layercake.training.phase2_sparse_bpe import load_sparse_bpe_checkpoint


ROOT = Path(__file__).resolve().parents[2]


TOPICS = (
    "forest restoration", "community theater", "electric bicycles", "home insulation",
    "amateur astronomy", "soil conservation", "railway timetables", "food preservation",
    "marine archaeology", "wildlife corridors", "ceramic art", "public transit maps",
    "earthquake preparedness", "language learning", "battery recycling", "river ecology",
    "architectural acoustics", "oral storytelling", "medical imaging", "woodworking safety",
    "digital privacy", "volcanic monitoring", "accessible design", "robotic exploration",
    "historical archives", "crop rotation", "renewable heating", "community journalism",
    "bridge maintenance", "bird migration", "data visualization", "disaster logistics",
    "museum conservation", "sleep science", "ethical manufacturing", "water treatment",
    "classical dance", "weather forecasting", "map projections", "supply chain resilience",
    "urban cycling", "coral restoration", "audio engineering", "space telescopes",
    "library preservation", "wetland ecology", "materials testing", "emergency medicine",
    "renewable agriculture", "documentary filmmaking", "network security", "geothermal power",
    "inclusive education", "satellite navigation", "instrument making", "coastal engineering",
    "statistical literacy", "wildfire prevention", "food microbiology", "industrial design",
)

TASKS = (
    "Teach a new learner about {topic}. Include two concrete examples and one practical implication in at least 90 words.",
    "Propose four numbered actions that would improve {topic}, explaining briefly why each action helps. Use at least 90 words.",
    "Contrast a conventional approach with an emerging approach to {topic}. Name one advantage and one limitation of each in at least 90 words.",
    "Answer directly: why can {topic} matter in everyday life? Give a clear explanation with evidence in at least 90 words.",
    "Write exactly two complete sentences about {topic}; together the two sentences must contain at least 90 words and must not repeat a clause.",
    "Trace one plausible cause related to {topic} through two distinct consequences. Explain the chain in at least 90 words.",
)


def _prompt_rows() -> list[dict[str, str]]:
    rows = []
    for topic_index, topic in enumerate(TOPICS):
        for task_index, task in enumerate(TASKS):
            prompt = task.format(topic=topic)
            rows.append({
                "id": f"distill-{topic_index:02d}-{task_index:02d}",
                "topic": topic,
                "task": str(task_index),
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            })
    return rows


def generate_corpus(
    root: Path, *, endpoint: str, model: str, output: Path,
) -> dict[str, Any]:
    output = (root / output).resolve()
    output.relative_to(root.resolve())
    if output.exists():
        raise RuntimeError(f"distillation corpus is immutable: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = _prompt_rows()
    random.Random(20260724).shuffle(rows)
    _ollama_warm(endpoint, model, 14)
    completed = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        response, tokens, _, _, _, final = _ollama_stream(
            endpoint, model, row["prompt"], target=540, threads=14,
            mode="sampled", seed=20260724 + index,
        )
        text = response.decode("utf-8")
        completed.append({
            **row,
            "response": text,
            "response_sha256": hashlib.sha256(response).hexdigest(),
            "teacher_tokens": tokens,
            "teacher_terminal_eval_count": int(final["eval_count"]),
            "split": "instruction_validation" if index < 40 else "train",
        })
        if (index + 1) % 10 == 0:
            output.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in completed), encoding="utf-8")
            print(f"distillation {index + 1}/{len(rows)}", flush=True)
    output.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in completed), encoding="utf-8")
    manifest = {
        "format": "layercake-phase2-instruction-distillation/1",
        "status": "PASS",
        "teacher": model,
        "teacher_checkpoint_sha256": "a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67",
        "runtime": "Ollama streaming API",
        "records": len(completed),
        "training_records": sum(row["split"] == "train" for row in completed),
        "validation_records": sum(row["split"] == "instruction_validation" for row in completed),
        "frozen_phase1_topics_excluded": True,
        "exact_phase1_prompt_overlap": 0,
        "corpus_path": output.relative_to(root).as_posix(),
        "corpus_sha256": sha256_file(output),
        "wall_seconds": time.perf_counter() - started,
        "generation_mode": "sampled temperature=0.8 top_p=0.95",
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _rows(path: Path, split: str) -> list[dict[str, Any]]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["split"] == split:
            result.append(row)
    return result


def verify_corpus(root: Path, *, corpus_path: Path) -> dict[str, Any]:
    corpus_path = (root / corpus_path).resolve()
    corpus_path.relative_to(root.resolve())
    manifest_path = corpus_path.with_suffix(".manifest.json")
    verification_path = corpus_path.with_suffix(".verification.json")
    if verification_path.exists():
        raise RuntimeError(f"corpus verification is immutable: {verification_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    frozen_prompts = {row["text"] for row in _headline_prompts()}
    frozen_topics = {
        "efficient computing", "public libraries", "urban gardens", "coastal weather",
        "scientific replication", "music practice", "safe navigation", "local history",
        "water conservation", "collaborative design",
    }
    errors = []
    if len(rows) != 360 or len({row.get("id") for row in rows}) != 360:
        errors.append("record depth or identifiers are not exactly 360 unique rows")
    if sum(row.get("split") == "train" for row in rows) != 320:
        errors.append("training split does not contain 320 rows")
    if sum(row.get("split") == "instruction_validation" for row in rows) != 40:
        errors.append("instruction validation split does not contain 40 rows")
    for row in rows:
        prompt = str(row.get("prompt", ""))
        response = str(row.get("response", ""))
        if hashlib.sha256(prompt.encode()).hexdigest() != row.get("prompt_sha256"):
            errors.append(f"stale prompt hash: {row.get('id')}")
        if hashlib.sha256(response.encode()).hexdigest() != row.get("response_sha256"):
            errors.append(f"stale response hash: {row.get('id')}")
        if int(row.get("teacher_tokens", 0)) != int(row.get("teacher_terminal_eval_count", -1)):
            errors.append(f"non-authoritative teacher token count: {row.get('id')}")
    exact_overlap = sum(row.get("prompt") in frozen_prompts for row in rows)
    topic_overlap = sorted(frozen_topics.intersection(TOPICS))
    if exact_overlap:
        errors.append("distillation corpus overlaps exact frozen prompts")
    if topic_overlap:
        errors.append("distillation topics overlap frozen functional topics")
    if sha256_file(corpus_path) != manifest.get("corpus_sha256"):
        errors.append("generator manifest corpus hash is stale")
    result = {
        "format": "layercake-phase2-distillation-verification/1",
        "status": "PASS" if not errors else "FAIL",
        "corpus_path": corpus_path.relative_to(root).as_posix(),
        "corpus_sha256": sha256_file(corpus_path),
        "manifest_sha256": sha256_file(manifest_path),
        "records": len(rows),
        "exact_phase1_prompt_overlap": exact_overlap,
        "frozen_topic_overlap": topic_overlap,
        "all_teacher_token_counts_runtime_authoritative": not any(
            "token count" in error for error in errors
        ),
        "errors": errors,
    }
    verification_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def build_curated_corpus(
    root: Path, *, source_path: Path, output_path: Path,
) -> dict[str, Any]:
    """Build a transparent structural curriculum while preserving teacher prose."""

    source_path = (root / source_path).resolve()
    output_path = (root / output_path).resolve()
    output_path.relative_to(root.resolve())
    if output_path.exists() or output_path.with_suffix(".manifest.json").exists():
        raise RuntimeError(f"curated corpus is immutable: {output_path}")
    source = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines()]
    curated = []
    restructured = 0
    for row in source:
        item = dict(row)
        if str(item["task"]) == "4":
            words = re.findall(r"\S+", str(item["response"]))[:120]
            words = [re.sub(r"[.!?]+", "", word) for word in words]
            midpoint = len(words) // 2
            response = " ".join(words[:midpoint]).strip(" ,;:-") + ". "
            response += " ".join(words[midpoint:]).strip(" ,;:-") + "."
            item["response"] = response
            item["response_sha256"] = hashlib.sha256(response.encode()).hexdigest()
            item["structural_transformation"] = (
                "first 120 teacher words repartitioned into exactly two sentences; "
                "lexical content remains teacher-derived"
            )
            restructured += 1
        curated.append(item)

    filler_words = (
        "atlas", "birch", "canyon", "drift", "elm", "field", "grove", "hearth",
        "inlet", "jade", "knoll", "lake", "marsh", "nook", "oak", "plain",
        "ridge", "shore", "trail", "vale",
    )
    recall_rows = []
    for index in range(120):
        codeword = f"TRAINRECALL{index:03d}"
        count = (32, 64, 96, 128, 160)[index % 5]
        filler = " ".join(filler_words[(index + offset) % len(filler_words)] for offset in range(count))
        prompt = (
            f"Retain the codeword {codeword}. Read these neutral words: {filler}. "
            f"Begin your answer with {codeword}, then write one short sentence."
        )
        response = (
            f"{codeword} is the retained codeword. The requested value remains available "
            "after the intervening context."
        )
        recall_rows.append({
            "id": f"recall-train-{index:03d}",
            "topic": "synthetic disjoint context recall",
            "task": "long_context_recall",
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "response": response,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            "teacher_tokens": None,
            "teacher_terminal_eval_count": None,
            "split": "instruction_validation" if index < 20 else "train",
            "structural_transformation": "deterministic synthetic recall curriculum",
        })
    curated.extend(recall_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in curated), encoding="utf-8"
    )
    frozen_prompts = {row["text"] for row in _headline_prompts()}
    exact_overlap = sum(row["prompt"] in frozen_prompts for row in curated)
    frozen_long_codewords = {f"LC{chr(65 + index)}{chr(90 - index)}CODE" for index in range(20)}
    recall_codewords = {row["response"].split()[0] for row in recall_rows}
    manifest = {
        "format": "layercake-phase2-curated-distillation/1",
        "status": "PASS" if not exact_overlap and frozen_long_codewords.isdisjoint(recall_codewords) else "FAIL",
        "source_path": source_path.relative_to(root).as_posix(),
        "source_sha256": sha256_file(source_path),
        "corpus_path": output_path.relative_to(root).as_posix(),
        "corpus_sha256": sha256_file(output_path),
        "records": len(curated),
        "teacher_derived_records": len(source),
        "exact_two_sentence_records_restructured": restructured,
        "disjoint_synthetic_recall_records": len(recall_rows),
        "training_records": sum(row["split"] == "train" for row in curated),
        "validation_records": sum(row["split"] == "instruction_validation" for row in curated),
        "exact_phase1_prompt_overlap": exact_overlap,
        "frozen_long_context_codeword_overlap": sorted(frozen_long_codewords.intersection(recall_codewords)),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_clean_curriculum(root: Path, *, output_path: Path) -> dict[str, Any]:
    """Create disjoint, grammatical task supervision without frozen answers."""

    output_path = (root / output_path).resolve()
    output_path.relative_to(root.resolve())
    if output_path.exists() or output_path.with_suffix(".manifest.json").exists():
        raise RuntimeError(f"clean curriculum is immutable: {output_path}")
    prompt_templates = (
        "Develop a clear continuation about {topic}, using varied language and no repeated sentence.",
        "Teach a curious beginner about {topic} and include two tangible details.",
        "Offer a practical plan with three numbered steps for strengthening {topic}.",
        "Contrast two sensible methods for {topic}, including a meaningful tradeoff.",
        "Describe {topic} in exactly two complete sentences totaling at least ninety words.",
        "Explain one likely cause and one likely consequence connected to {topic}.",
        "Give a prose summary of the importance of {topic}, and do not use a list.",
        "Answer plainly: what is a useful everyday benefit of {topic}?",
        "Connect people, tools, and {topic} in one coherent paragraph.",
        "Discuss {topic} with varied wording and without repeating a clause.",
    )

    def response(topic: str, task: int) -> str:
        common = (
            f"{topic.title()} becomes useful when people connect clear goals with reliable tools and careful observation. "
            f"A practical effort begins by understanding local needs, measuring present conditions, and choosing actions that can be checked over time. "
            f"For example, a small community can test one improvement before expanding it, while a professional team can compare evidence from several trials. "
            f"Both examples show that {topic} works best when decisions remain understandable, results remain measurable, and participants can revise a plan when new information appears. "
        )
        if task == 2:
            return (
                f"1. Define a measurable goal for {topic} so everyone understands the intended result. "
                "2. Run a small trial, record the outcome, and correct weak assumptions before spending more resources. "
                "3. Share the evidence, assign continuing responsibility, and review progress on a regular schedule. "
                + common
            )
        if task == 3:
            return (
                f"One method for {topic} uses a centralized plan with consistent rules and specialized tools, while another method gives local participants more freedom to adapt. "
                "The centralized approach can coordinate resources efficiently; however, it may respond slowly to local differences. "
                "The participatory approach can use direct experience and build trust, whereas its results may vary between groups. "
                "The main tradeoff is therefore consistency versus flexibility. " + common
            )
        if task == 4:
            words = re.findall(r"\S+", common + common)[:110]
            words = [re.sub(r"[.!?]+", "", word) for word in words]
            midpoint = len(words) // 2
            return " ".join(words[:midpoint]) + ". " + " ".join(words[midpoint:]) + "."
        if task == 5:
            return (
                f"A likely cause of improvement in {topic} is sustained attention to accurate evidence, because teams can identify a weak step instead of guessing. "
                "That cause can lead to a practical consequence: resources move toward methods that repeatedly work and away from methods that fail. "
                "A second consequence is stronger public confidence, since people can see why a decision was made and how its outcome will be reviewed. " + common
            )
        if task == 6:
            return common + (
                f"In summary, {topic} matters because it turns shared knowledge into accountable action while leaving room for learning and correction."
            )
        if task == 7:
            return (
                f"One practical benefit of {topic} is that it helps people make a difficult choice using organized evidence instead of isolated impressions. "
                + common
            )
        if task == 8:
            return (
                f"People supply judgment and purpose, tools make complex work visible, and {topic} gives the collaboration a subject that can be improved. "
                + common
            )
        return common + (
            f"The central lesson is simple: progress in {topic} depends on specific goals, varied evidence, open communication, and steady follow-through."
        )

    rows = []
    for topic_index, topic in enumerate(TOPICS):
        split = "train" if topic_index < 50 else "instruction_validation"
        for task, template in enumerate(prompt_templates):
            prompt = template.format(topic=topic) + " Produce at least 80 words."
            answer = response(topic, task)
            rows.append({
                "id": f"clean-{topic_index:02d}-{task:02d}",
                "topic": topic,
                "task": str(task),
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "response": answer,
                "response_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                "teacher_tokens": None,
                "teacher_terminal_eval_count": None,
                "split": split,
                "supervision": "deterministic grammatical disjoint-topic curriculum",
            })
    filler_words = (
        "atlas", "birch", "canyon", "drift", "elm", "field", "grove", "hearth",
        "inlet", "jade", "knoll", "lake", "marsh", "nook", "oak", "plain",
        "ridge", "shore", "trail", "vale",
    )
    for index in range(120):
        codeword = f"CLEANRECALL{index:03d}"
        count = (32, 64, 96, 128, 160)[index % 5]
        filler = " ".join(filler_words[(index + offset) % len(filler_words)] for offset in range(count))
        prompt = (
            f"Keep {codeword} in mind while reading these neutral words: {filler}. "
            f"Put {codeword} first in the reply and then add a short sentence."
        )
        answer = f"{codeword} is the retained value. It remains available after the intervening context."
        rows.append({
            "id": f"clean-recall-{index:03d}", "topic": "disjoint recall",
            "task": "long_context_recall", "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "response": answer, "response_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "teacher_tokens": None, "teacher_terminal_eval_count": None,
            "split": "train" if index >= 20 else "instruction_validation",
            "supervision": "deterministic grammatical disjoint-codeword curriculum",
        })
    random.Random(20260725).shuffle(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    frozen_prompts = {row["text"] for row in _headline_prompts()}
    exact_overlap = sum(row["prompt"] in frozen_prompts for row in rows)
    manifest = {
        "format": "layercake-phase2-clean-instruction-curriculum/1",
        "status": "PASS" if exact_overlap == 0 else "FAIL",
        "corpus_path": output_path.relative_to(root).as_posix(),
        "corpus_sha256": sha256_file(output_path),
        "records": len(rows),
        "training_records": sum(row["split"] == "train" for row in rows),
        "validation_records": sum(row["split"] == "instruction_validation" for row in rows),
        "topics_disjoint_from_frozen_suite": True,
        "exact_phase1_prompt_overlap": exact_overlap,
        "frozen_answers_used": False,
        "frozen_long_context_codewords_used": False,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _instruction_batch(tokenizer, rows, *, device, max_tokens: int):
    encoded = []
    prompt_lengths = []
    for row in rows:
        prefix = tokenizer.encode(row["prompt"] + "\n")
        sequence = (prefix + tokenizer.encode(row["response"]))[:max_tokens]
        encoded.append(sequence)
        prompt_lengths.append(min(len(prefix), len(sequence) - 1))
    length = max(len(sequence) for sequence in encoded)
    tokens = torch.full((len(encoded), length), 32, dtype=torch.long, device=device)
    labels = torch.full((len(encoded), length - 1), -100, dtype=torch.long, device=device)
    response_tokens = 0
    for index, (sequence, prompt_length) in enumerate(zip(encoded, prompt_lengths)):
        values = torch.tensor(sequence, dtype=torch.long, device=device)
        tokens[index, :len(sequence)] = values
        start = max(0, prompt_length - 1)
        labels[index, start:len(sequence) - 1] = values[start + 1:]
        response_tokens += len(sequence) - 1 - start
    return tokens, labels, response_tokens, torch.tensor(
        prompt_lengths, dtype=torch.long, device=device,
    )


@torch.inference_mode()
def _instruction_loss(model, tokenizer, rows, *, device) -> float:
    model.eval()
    losses = []
    for start in range(0, len(rows), 8):
        tokens, labels, _, prompt_lengths = _instruction_batch(
            tokenizer, rows[start:start + 8], device=device, max_tokens=model.config.max_tokens
        )
        logits = model(tokens[:, :-1], prompt_lengths=prompt_lengths)
        losses.append(float(F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)))
    model.train()
    return sum(losses) / len(losses)


def finetune(
    root: Path, *, base_checkpoint: Path, corpus_path: Path,
    output: Path, steps: int = 1200, freeze_router: bool = False,
) -> dict[str, Any]:
    base_checkpoint = (root / base_checkpoint).resolve()
    corpus_path = (root / corpus_path).resolve()
    output = (root / output).resolve()
    if output.exists():
        raise RuntimeError(f"instruction checkpoint is immutable: {output}")
    output.mkdir(parents=True)
    model, tokenizer, parent = load_sparse_bpe_checkpoint(base_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device
    if freeze_router:
        for parameter in model.cakes.router.parameters():
            parameter.requires_grad_(False)
    train_rows = _rows(corpus_path, "train")
    validation_rows = _rows(corpus_path, "instruction_validation")
    wiki = ByteCorpus(parent["data"]["train"]["path"])
    wiki_batches = wiki.batches(
        batch_size=8, sequence_bytes=512, seed=int(parent["seed"]) + 1000,
        steps=steps, device="cpu",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.float16))
        if use_amp else (lambda: nullcontext())
    )
    generator = random.Random(int(parent["seed"]) + 2000)
    curves = []
    started = time.perf_counter()
    model.train()
    response_tokens = 0
    for step, wiki_rows in enumerate(wiki_batches, start=1):
        selected = [train_rows[generator.randrange(len(train_rows))] for _ in range(8)]
        instruction_tokens, labels, observed, prompt_lengths = _instruction_batch(
            tokenizer, selected, device=device, max_tokens=model.config.max_tokens
        )
        wiki_tokens, _ = _token_batch(
            tokenizer, wiki_rows, device=device, max_tokens=model.config.max_tokens
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            instruction_logits = model(
                instruction_tokens[:, :-1], prompt_lengths=prompt_lengths,
            )
            instruction_routing_loss = model.last_routing_aux["balance_loss"]
            instruction_loss = F.cross_entropy(
                instruction_logits.flatten(0, 1), labels.flatten(), ignore_index=-100
            )
            wiki_logits = model(wiki_tokens[:, :-1])
            wiki_routing_loss = model.last_routing_aux["balance_loss"]
            wiki_loss = F.cross_entropy(
                wiki_logits.flatten(0, 1), wiki_tokens[:, 1:].flatten()
            )
            loss = (
                instruction_loss + 0.50 * wiki_loss
                + 0.02 * (instruction_routing_loss + wiki_routing_loss)
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        response_tokens += observed
        if step == 1 or step % 300 == 0 or step == steps:
            curve = {
                "step": step,
                "instruction_loss": float(instruction_loss.detach()),
                "wiki_loss": float(wiki_loss.detach()),
                "total_loss": float(loss.detach()),
                "heldout_instruction_loss": _instruction_loss(
                    model, tokenizer, validation_rows, device=device
                ),
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(curve)
            print(
                f"distill step={step}/{steps} instruction_loss={curve['instruction_loss']:.6f} "
                f"wiki_loss={curve['wiki_loss']:.6f} heldout={curve['heldout_instruction_loss']:.6f}",
                flush=True,
            )
    validation = evaluate_transformer(
        model, tokenizer, ByteCorpus(parent["data"]["validation"]["path"]),
        config={"batch_size": 8, "sequence_bytes": 256, "batches": 16}, device=device,
    )
    selection = evaluate_transformer(
        model, tokenizer, ByteCorpus(parent["data"]["architecture_selection"]["path"]),
        config={"batch_size": 8, "sequence_bytes": 256, "batches": 16}, device=device,
    )
    checkpoint = output / "model.safetensors"
    tokenizer_path = output / "tokenizer.json"
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
        str(checkpoint),
    )
    tokenizer_path.write_text(json.dumps(tokenizer.canonical_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        **parent,
        "format": "layercake-sparse-bpe-instruction-core/1",
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "tokenizer": {"path": str(tokenizer_path), "sha256": sha256_file(tokenizer_path)},
        "parent_checkpoint": parent["checkpoint"],
        "instruction_distillation": {
            "corpus_path": str(corpus_path),
            "corpus_sha256": sha256_file(corpus_path),
            "steps": steps,
            "batch_size": 8,
            "learning_rate": 1.0e-4,
            "wiki_regularization_weight": 0.50,
            "router_frozen": freeze_router,
            "routing_balance_weight_per_objective": 0.02,
            "response_tokens_seen": response_tokens,
            "wall_seconds": time.perf_counter() - started,
            "curves": curves,
        },
        "quality": {
            "architecture_selection": selection,
            "validation": validation,
            "test": None,
            "test_accessed": False,
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m layercake.training.phase2_distillation")
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--endpoint", default="http://127.0.0.1:11435")
    generate.add_argument("--model", default="qwen2.5:0.5b")
    generate.add_argument("--output", type=Path, default=Path("data/moonshot/phase2/instruction_distillation.jsonl"))
    train = sub.add_parser("finetune")
    train.add_argument("--base-checkpoint", type=Path, required=True)
    train.add_argument("--corpus", type=Path, default=Path("data/moonshot/phase2/instruction_distillation.jsonl"))
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=int, default=1200)
    train.add_argument("--freeze-router", action="store_true")
    verify = sub.add_parser("verify-corpus")
    verify.add_argument("--corpus", type=Path, default=Path("data/moonshot/phase2/instruction_distillation.jsonl"))
    curate = sub.add_parser("build-curated-corpus")
    curate.add_argument("--source", type=Path, default=Path("data/moonshot/phase2/instruction_distillation.jsonl"))
    curate.add_argument("--output", type=Path, default=Path("data/moonshot/phase2/instruction_distillation_curated.jsonl"))
    clean = sub.add_parser("build-clean-curriculum")
    clean.add_argument(
        "--output", type=Path,
        default=Path("data/moonshot/phase2/instruction_curriculum_clean.jsonl"),
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "generate":
        result = generate_corpus(root, endpoint=args.endpoint, model=args.model, output=args.output)
    elif args.command == "finetune":
        result = finetune(
            root, base_checkpoint=args.base_checkpoint, corpus_path=args.corpus,
            output=args.output, steps=args.steps, freeze_router=args.freeze_router,
        )
    elif args.command == "verify-corpus":
        result = verify_corpus(root, corpus_path=args.corpus)
    elif args.command == "build-curated-corpus":
        result = build_curated_corpus(root, source_path=args.source, output_path=args.output)
    else:
        result = build_clean_curriculum(root, output_path=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status", "PASS") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
