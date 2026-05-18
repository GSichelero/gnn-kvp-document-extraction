"""Training-data derived category vocabulary for optional invoice scripts.

The auxiliary local pipeline used to keep a fixed dictionary of utility-bill
fields here. To keep the repository dataset-driven, categories are
now inferred from ground-truth category assignment files when those files are
available locally.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent


def default_gt_base() -> Path:
    """Return the local ground-truth base directory, if configured."""
    configured = os.environ.get("INVOICE_GT_BASE")
    if configured:
        return Path(configured)

    project_root = Path(os.environ.get("DOCUMENT_KVP_PROJECT_ROOT", REPO_ROOT)).resolve()
    return Path(
        next(
            (str(path) for path in project_root.glob("*_output/gt_annotation_images") if path.exists()),
            str(project_root / "invoice_output" / "gt_annotation_images"),
        )
    )


def _normalize_label(label: str) -> str:
    return re.sub(r"[_\s]+", " ", str(label)).strip().lower()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _add_terms(bucket: Counter, text: Optional[str]) -> None:
    if not text:
        return
    cleaned = str(text).strip()
    if not cleaned or cleaned == "[table_cell]":
        return
    bucket[cleaned.lower()] += 2
    for token in _tokenize(cleaned):
        if len(token) > 2:
            bucket[token] += 1


def iter_assignment_files(gt_base: Optional[Path] = None) -> Iterable[Path]:
    """Yield category assignment files from the local GT layout."""
    base = Path(gt_base or default_gt_base())
    if not base.exists():
        return

    for dist_dir in base.iterdir():
        if not dist_dir.is_dir() or dist_dir.name in {"xlsx_items", "category_gt", "annotations"}:
            continue
        cat_dir = dist_dir / "annotations" / "category_assignments"
        if cat_dir.exists():
            yield from sorted(cat_dir.glob("*.json"))


def build_category_vocabulary(
    gt_base: Optional[Path] = None,
    min_examples: int = 1,
    max_terms_per_category: int = 12,
) -> Dict[str, List[str]]:
    """Infer category terms from local training/annotation assignments."""
    terms_by_category: DefaultDict[str, Counter] = defaultdict(Counter)
    examples_by_category: Counter = Counter()

    for assignment_file in iter_assignment_files(gt_base):
        try:
            data = json.loads(assignment_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        assignments = data.get("assignments", {})
        if not isinstance(assignments, dict):
            continue

        for category, info in assignments.items():
            if not category:
                continue
            bucket = terms_by_category[str(category)]
            examples_by_category[str(category)] += 1
            bucket[_normalize_label(str(category))] += 1

            if isinstance(info, dict):
                _add_terms(bucket, info.get("entity_text"))
                _add_terms(bucket, info.get("key_text"))
                _add_terms(bucket, info.get("value_text"))

    vocabulary = {}
    for category, counter in sorted(terms_by_category.items()):
        if examples_by_category[category] < min_examples:
            continue
        terms = [term for term, _ in counter.most_common(max_terms_per_category)]
        if terms:
            vocabulary[category] = terms

    return vocabulary


# Kept for compatibility with older optional scripts. The value is now empty
# unless local GT category assignment files are present.
INVOICE_CATEGORIES = build_category_vocabulary()
