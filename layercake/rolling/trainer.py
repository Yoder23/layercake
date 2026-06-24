from __future__ import annotations

from pathlib import Path
import time
from dataclasses import replace

import torch
from torch import nn

from .certificates import SemanticCertificate
from .commit import ModelCommit
from .common import stable_hash, write_json
from .gates import GateResult, run_gates
from .preview import run_preview
from .registry import ModuleRegistry
from .rollback import rollback_failed_stage
from .syllabus import compile_syllabus


class RollingTrainer:
    def __init__(
        self,
        registry: ModuleRegistry,
        *,
        root: str | Path = "artifacts/commits",
        model_family_id: str = "toy",
        abi_hash: str = "toy-abi",
        input_interface_hash: str = "toy-input",
        byte_patch_hash: str = "toy-byte-patch",
    ):
        self.registry = registry
        self.root = Path(root)
        self.model_family_id = model_family_id
        self.abi_hash = abi_hash
        self.input_interface_hash = input_interface_hash
        self.byte_patch_hash = byte_patch_hash
        self.commits: list[ModelCommit] = []
        self.root.mkdir(parents=True, exist_ok=True)

    def create_commit(self, parent, rubric, message: str, status: str = "candidate") -> ModelCommit:
        module_hashes = self.registry.module_hashes()
        artifacts = {}
        commit = ModelCommit.create(
            parent_commit_id=parent.commit_id if parent else None,
            branch=rubric.branch,
            status=status,
            model_family_id=self.model_family_id,
            abi_hash=self.abi_hash,
            input_interface_hash=self.input_interface_hash,
            byte_patch_hash=self.byte_patch_hash,
            module_hashes=module_hashes,
            artifact_paths={},
            rubric_hash=rubric.compute_hash(),
            message=message,
        )
        for name in self.registry.list_modules():
            artifacts[name] = self.registry.save_module(
                name, self.root / f"{commit.commit_id}_{name}.pt"
            )
        commit = ModelCommit.create(
            **{**commit.to_dict(), "commit_id": "", "artifact_paths": artifacts}
        )
        commit.save(self.root)
        self.commits.append(commit)
        return commit

    def run_rubric(
        self,
        rubric,
        parent_commit=None,
        *,
        train_step=None,
        metrics: dict | None = None,
        certificate_path: str | Path | None = None,
    ) -> tuple[ModelCommit, SemanticCertificate, dict | None]:
        for name in rubric.frozen_modules:
            if name in self.registry.modules:
                self.registry.freeze(name)
        for name in rubric.trainable_modules:
            if name in self.registry.modules:
                self.registry.unfreeze(name)
        if train_step:
            for _ in range(max(rubric.max_steps, 1)):
                train_step()
        commit = self.create_commit(parent_commit, rubric, f"run {rubric.rubric_id}")
        context = {
            "score": 1.0,
            "commit": commit.to_dict(),
            "parent_commit": parent_commit.to_dict() if parent_commit else commit.to_dict(),
            "parent": metrics.get("parent", {}) if metrics else {},
        }
        if metrics:
            context.update(metrics)
        gate_results = run_gates(rubric.gates, context)
        passed = all(result.passed for result in gate_results)
        commit = commit.mark_passed() if passed else commit.mark_failed()
        commit.save(self.root)
        rollback_report = None
        if not passed and parent_commit and rubric.rollback_policy.get("on_failure", True):
            rollback_report = rollback_failed_stage(commit, parent_commit, self.registry)
        cert = SemanticCertificate.create(
            commit.commit_id,
            parent_commit.commit_id if parent_commit else None,
            rubric.rubric_id,
            gate_results,
            regression_summary={"rollback": rollback_report} if rollback_report else {},
        )
        cert.save(certificate_path or Path("results/certificates") / f"{rubric.rubric_id}.json")
        return commit, cert, rollback_report

    def run_preview_guided(
        self,
        rubric,
        dataset_path,
        *,
        model=None,
        parent_commit=None,
        train_step=None,
        metrics: dict | None = None,
        mode: str | None = None,
        certificate_path: str | Path | None = None,
    ):
        preview = run_preview(rubric, dataset_path, model=model, parent_commit=parent_commit)
        syllabus = compile_syllabus(rubric, preview, mode=mode)
        loss_history = []
        early_stop_report = None
        started = time.perf_counter()
        for name in syllabus.frozen_modules:
            if name in self.registry.modules:
                self.registry.freeze(name)
        for name in syllabus.trainable_modules:
            if name in self.registry.modules:
                self.registry.unfreeze(name)
        if train_step:
            from .early_stop import EarlyStopper

            stopper = EarlyStopper(
                patience=syllabus.early_stop_rules[0].get("patience", 2),
                min_delta=syllabus.early_stop_rules[0].get("min_delta", 0.0),
            )
            for step in range(max(rubric.max_steps, 1)):
                value = train_step()
                if value is None:
                    value = metrics.get("loss", 1.0) if metrics else 1.0
                loss_history.append(float(value))
                decision = stopper.update(float(value), step=step)
                if decision.should_stop:
                    early_stop_report = decision.__dict__
                    break
        elapsed = time.perf_counter() - started
        merged_metrics = {
            "preview_id": preview.preview_id,
            "syllabus_id": syllabus.syllabus_id,
            "training_seconds": elapsed,
            "steps": len(loss_history) or max(rubric.max_steps, 1),
            "loss_history": loss_history,
            "trainable_params": self.registry.trainable_parameter_count(),
            "bpb": preview.current_model_bpb if preview.current_model_bpb is not None else 0.0,
            "preview": preview.to_dict(),
            "syllabus": syllabus.to_dict(),
        }
        if metrics:
            merged_metrics.update(metrics)
        commit, cert, rollback_report = self.run_rubric(
            rubric,
            parent_commit,
            train_step=None,
            metrics=merged_metrics,
            certificate_path=certificate_path,
        )
        if early_stop_report and parent_commit:
            commit = commit.mark_failed()
            commit.save(self.root)
            rollback_report = rollback_failed_stage(commit, parent_commit, self.registry)
            gate_results = list(cert.gate_results) + [{
                "gate_name": "early_stop",
                "passed": False,
                "metric_name": early_stop_report["metric"],
                "value": early_stop_report["value"],
                "threshold": None,
                "comparison": "triggered",
                "details": early_stop_report,
                "artifact_path": None,
            }]
            cert = replace(
                cert,
                gate_results=gate_results,
                passed=False,
                regression_summary={"early_stop": early_stop_report, "rollback": rollback_report},
            )
            cert.save(certificate_path or Path("results/certificates") / f"{rubric.rubric_id}.json")
        return commit, cert, rollback_report, preview, syllabus

    def run_sequence(self, rubrics: list, train_steps: list | None = None):
        parent = self.commits[-1] if self.commits else None
        results = []
        for index, rubric in enumerate(rubrics):
            step = train_steps[index] if train_steps else None
            commit, cert, rollback_report = self.run_rubric(rubric, parent, train_step=step)
            results.append((commit, cert, rollback_report))
            if cert.passed:
                parent = commit
        return results


class TinyToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


def toy_train_step(model: nn.Module, *, lr: float = 0.01, damage: bool = False):
    with torch.no_grad():
        for parameter in model.parameters():
            if damage:
                parameter.add_(10.0)
            else:
                parameter.mul_(0.99)
