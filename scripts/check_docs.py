"""Fail-closed checks for LayerCake's canonical documentation surface."""

from __future__ import annotations

import re
from pathlib import Path
import subprocess
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DOCUMENTS = (
    "README.md",
    "DEPLOYMENT_QUICKSTART.md",
    "ARCHITECTURE.md",
    "CLAIMS.md",
    "CONTRIBUTING.md",
    "ROADMAP.md",
    "SECURITY.md",
    "docs/README.md",
    "docs/CONCEPTS.md",
    "docs/PROJECT_STATUS.md",
    "docs/REPOSITORY_MAP.md",
    "docs/ABI_HANDOFF_STATUS.md",
    "docs/CAKE_AUTHORING.md",
    "docs/CAKE_REGISTRY_SPEC.md",
    "docs/CAKE_THREAT_MODEL.md",
    "docs/MOONSHOT_ARCHITECTURE.md",
    "docs/PHASE_STATUS.md",
    "docs/VERIFICATION_AND_LIMITS.md",
)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def local_link_target(document: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().strip("<>")
    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return None
    path_part = unquote(target.split("#", 1)[0])
    if not path_part:
        return None
    return (document.parent / path_part).resolve()


def check() -> list[str]:
    errors: list[str] = []
    canonical_documents: list[Path] = []
    for relative in CANONICAL_DOCUMENTS:
        document = ROOT / relative
        if not document.is_file():
            errors.append(f"missing canonical document: {relative}")
            continue
        canonical_documents.append(document)

    tracked_markdown = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    link_documents = {
        *canonical_documents,
        *(ROOT / relative for relative in tracked_markdown),
    }

    for document in canonical_documents:
        text = document.read_text(encoding="utf-8")
        if not text.startswith("# "):
            errors.append(f"missing level-one heading: {document.relative_to(ROOT)}")

    for document in sorted(link_documents):
        text = document.read_text(encoding="utf-8")
        for match in LINK.finditer(text):
            target = local_link_target(document, match.group(1))
            if target is not None and not target.exists():
                errors.append(
                    f"broken local link in {document.relative_to(ROOT)}: {match.group(1)}"
                )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "DEPLOYMENT_QUICKSTART.md",
        "docs/README.md",
        "docs/PROJECT_STATUS.md",
        "docs/REPOSITORY_MAP.md",
        "docs/VERIFICATION_AND_LIMITS.md",
    ):
        if required not in readme:
            errors.append(f"README discovery path is missing: {required}")

    quickstart = (ROOT / "DEPLOYMENT_QUICKSTART.md").read_text(encoding="utf-8")
    for required_command in (
        "python -m pip install -e .",
        "python -m layercake --help",
        "python -m layercake cake",
        "python -m layercake run",
    ):
        if required_command not in quickstart:
            errors.append(f"quickstart command is missing: {required_command}")

    return errors


def main() -> int:
    errors = check()
    if errors:
        for error in errors:
            print(f"DOCS_ERROR: {error}")
        return 1
    print(
        f"DOCS_OK: {len(CANONICAL_DOCUMENTS)} canonical documents and all tracked Markdown links checked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
