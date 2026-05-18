#!/usr/bin/env python3
"""
Train semantic normalization category embeddings using GT category assignments,
with optional table context enrichment:
- value text from assigned entity
- column header text above table cell
- row identifier text from left cell in same row

Outputs:
  outputs/models/semantic_norm_finetuned.pt
  outputs/models/semantic_norm_context_examples.json
  outputs/models/semantic_norm_context_summary.json
"""

import json
import os
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from invoice_categories import build_category_vocabulary

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
CONTRIB_ROOT = SCRIPTS_DIR.parent
PROJECT_ROOT = Path(os.environ.get("DOCUMENT_KVP_PROJECT_ROOT", CONTRIB_ROOT)).resolve()
GT_BASE = Path(
    os.environ.get(
        "INVOICE_GT_BASE",
        next(
            (str(path) for path in PROJECT_ROOT.glob("*_output/gt_annotation_images") if path.exists()),
            str(PROJECT_ROOT / "invoice_output" / "gt_annotation_images"),
        ),
    )
)
MODELS_DIR = Path(os.environ.get("INVOICE_MODELS_DIR", CONTRIB_ROOT / "outputs" / "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def overlap_ratio(inner, outer):
    a = area(inner)
    if a <= 0:
        return 0.0
    return intersection(inner, outer) / a


def center(box):
    return (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))


def collect_ocr_text_in_box(ocr_entities, box, thr=0.3):
    hits = []
    for e in ocr_entities:
        ebox = e.get("box")
        if not ebox:
            continue
        ov = overlap_ratio(ebox, box)
        if ov >= thr:
            cx, cy = center(ebox)
            txt = (e.get("text") or "").strip()
            if txt:
                hits.append((cy, cx, txt))
    hits.sort()
    if not hits:
        return ""
    return " ".join(t[2] for t in hits)


def normalize_spaces(text: str) -> str:
    return " ".join((text or "").strip().split())


def build_table_context(entity, entities_by_id, ocr_entities, table_headers, table_cells):
    cell_box = entity.get("box")
    if not cell_box:
        return {"value": "", "header": "", "row": ""}

    value_text = collect_ocr_text_in_box(ocr_entities, cell_box, thr=0.3)

    header_text = ""
    cx, cy = center(cell_box)
    best_header = None
    best_dist = 1e18
    for h in table_headers:
        hbox = h.get("box")
        if not hbox:
            continue
        hcx, hcy = center(hbox)
        x_overlap = intersection([cell_box[0], 0, cell_box[2], 1], [hbox[0], 0, hbox[2], 1])
        w_min = max(1.0, min(cell_box[2] - cell_box[0], hbox[2] - hbox[0]))
        x_ov_ratio = x_overlap / w_min
        if hcy < cy and x_ov_ratio > 0.25:
            dist = abs(cy - hcy)
            if dist < best_dist:
                best_dist = dist
                best_header = h
    if best_header is not None:
        header_text = collect_ocr_text_in_box(ocr_entities, best_header["box"], thr=0.2)

    row_text = ""
    best_left_cell = None
    best_left_x = -1e18
    row_tol = max(10.0, 0.5 * (cell_box[3] - cell_box[1]))
    for c in table_cells:
        cbox = c.get("box")
        if not cbox:
            continue
        if c.get("id") == entity.get("id"):
            continue
        ccx, ccy = center(cbox)
        same_row = abs(ccy - cy) <= row_tol
        is_left = cbox[2] <= cell_box[0]
        if same_row and is_left and cbox[2] > best_left_x:
            best_left_x = cbox[2]
            best_left_cell = c
    if best_left_cell is not None:
        row_text = collect_ocr_text_in_box(ocr_entities, best_left_cell["box"], thr=0.2)

    return {
        "value": value_text,
        "header": header_text,
        "row": row_text,
    }


def build_kvp_context(entity, entities_by_id):
    """
    Build KVP context around an assigned OCR entity using GT linking.

    Returns:
      {
        "self_text": str,
        "linked_texts": [str, ...],
        "pairs": [str, ...]   # combined variants (self + linked)
      }
    """
    self_text = normalize_spaces(entity.get("text", ""))
    linked_ids = set()

    # Outgoing links
    for target in entity.get("linking", []):
        if isinstance(target, str):
            linked_ids.add(target)

    # Incoming links (some files may encode relations from the other endpoint)
    self_id = entity.get("id")
    if self_id:
        for other in entities_by_id.values():
            for target in other.get("linking", []):
                if isinstance(target, str) and target == self_id:
                    oid = other.get("id")
                    if oid:
                        linked_ids.add(oid)

    linked_texts = []
    for lid in sorted(linked_ids):
        linked_ent = entities_by_id.get(lid)
        if not linked_ent:
            continue
        txt = normalize_spaces(linked_ent.get("text", ""))
        if txt:
            linked_texts.append(txt)

    pairs = []
    for lt in linked_texts:
        if self_text and lt:
            pairs.append(f"{lt} {self_text}")
            pairs.append(f"{self_text} {lt}")

    return {
        "self_text": self_text,
        "linked_texts": linked_texts,
        "pairs": pairs,
    }


def iter_assignment_files():
    if not GT_BASE.exists():
        return
    for dist_dir in GT_BASE.iterdir():
        if not dist_dir.is_dir() or dist_dir.name in {"xlsx_items", "category_gt", "annotations"}:
            continue
        cat_dir = dist_dir / "annotations" / "category_assignments"
        ann_dir = dist_dir / "annotations"
        if not cat_dir.exists() or not ann_dir.exists():
            continue
        for cf in cat_dir.glob("*.json"):
            yield dist_dir.name, cf, ann_dir


def gather_examples():
    examples = []
    n_table_examples = 0
    n_kvp_examples = 0

    for distributor, cat_file, ann_dir in iter_assignment_files():
        cat_data = json.loads(cat_file.read_text(encoding="utf-8"))
        image_stem = cat_data.get("image_stem")
        assignments = cat_data.get("assignments", {})
        if not image_stem or not assignments:
            continue

        ann_file = ann_dir / f"{image_stem}.json"
        if not ann_file.exists():
            continue

        ann_data = json.loads(ann_file.read_text(encoding="utf-8"))
        form = ann_data.get("form", [])
        entities_by_id = {e.get("id"): e for e in form if e.get("id")}

        ocr_entities = [e for e in form if str(e.get("id", "")).startswith("ocr_")]
        table_headers = [e for e in form if e.get("label") == "table_header"]
        table_cells = [e for e in form if e.get("label") == "table_cell"]

        for category, info in assignments.items():
            entity_id = info.get("entity_id")
            entity_text = (info.get("entity_text") or "").strip()
            entity_label = info.get("entity_label")
            entity = entities_by_id.get(entity_id, {})

            variants = set()
            if entity_text and entity_text != "[table_cell]":
                variants.add(entity_text)

            if entity_label == "table_cell" and entity:
                ctx = build_table_context(entity, entities_by_id, ocr_entities, table_headers, table_cells)
                value_text = (ctx["value"] or "").strip()
                header_text = (ctx["header"] or "").strip()
                row_text = (ctx["row"] or "").strip()

                if value_text:
                    variants.add(value_text)
                if header_text and value_text:
                    variants.add(f"{header_text} {value_text}")
                if row_text and value_text:
                    variants.add(f"{row_text} {value_text}")
                if row_text and header_text and value_text:
                    variants.add(f"{row_text} {header_text} {value_text}")

                if value_text or header_text or row_text:
                    n_table_examples += 1

            # KVP context for OCR entities linked via GT relations
            if entity_label != "table_cell" and entity:
                kv = build_kvp_context(entity, entities_by_id)
                if kv["self_text"]:
                    variants.add(kv["self_text"])
                for lt in kv["linked_texts"]:
                    variants.add(lt)
                for pair_text in kv["pairs"]:
                    variants.add(pair_text)

                if kv["linked_texts"]:
                    n_kvp_examples += 1

            if entity_text and entity_text == "[table_cell]" and not variants:
                # fallback: at least category name as a weak example
                variants.add(category)

            for text in sorted(v for v in variants if v and len(v.strip()) > 0):
                examples.append({
                    "category": category,
                    "text": text,
                    "source": "table_context" if entity_label == "table_cell" else ("kvp_context" if entity else "assignment"),
                    "distributor": distributor,
                    "image_stem": image_stem,
                    "entity_id": entity_id,
                })

    return examples, {
        "n_table_context_assignments": n_table_examples,
        "n_kvp_context_assignments": n_kvp_examples,
    }


def train_embeddings_from_examples(examples):
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    category_examples = build_category_vocabulary(GT_BASE)
    by_cat = defaultdict(list)
    for ex in examples:
        c = ex["category"]
        t = ex["text"]
        if c and t:
            by_cat[c].append(t)

    cat_names = sorted(set(category_examples) | set(by_cat))
    if not cat_names:
        return [], torch.empty((0, 384)), by_cat

    cat_syn_texts = [
        " ".join(category_examples.get(c) or [c.replace("_", " ")])
        for c in cat_names
    ]
    base_embs = st_model.encode(cat_syn_texts, convert_to_tensor=True, show_progress_bar=False)
    base_embs = F.normalize(base_embs.float(), p=2, dim=1)

    adjusted = base_embs.clone()
    for i, cat in enumerate(cat_names):
        texts = by_cat.get(cat, [])
        if not texts:
            continue
        text_embs = st_model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        text_mean = F.normalize(text_embs.float(), p=2, dim=1).mean(dim=0)
        text_mean = F.normalize(text_mean, p=2, dim=0)

        # More GT evidence -> trust GT examples more
        n = len(texts)
        alpha_base = max(0.2, min(0.6, 1.0 / (1.0 + 0.12 * n)))
        merged = alpha_base * base_embs[i] + (1.0 - alpha_base) * text_mean
        adjusted[i] = F.normalize(merged, p=2, dim=0)

    return cat_names, adjusted, by_cat


def train_from_gt_context(verbose=True):
    if verbose:
        print("=" * 70)
        print("Semantic Normalization Training with GT Context")
        print("=" * 70)

    examples, context_stats = gather_examples()
    if not examples:
        if verbose:
            print("No GT category assignment examples found.")
        return {
            "ok": False,
            "reason": "no_examples",
        }

    examples_path = MODELS_DIR / "semantic_norm_context_examples.json"
    examples_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    cat_names, adjusted_embs, by_cat = train_embeddings_from_examples(examples)

    save_data = {
        "cat_names": cat_names,
        "cat_embeddings": adjusted_embs.cpu(),
        "n_examples": len(examples),
        "n_categories_adjusted": sum(1 for c in cat_names if len(by_cat.get(c, [])) > 0),
        "examples_per_category": {c: len(by_cat.get(c, [])) for c in cat_names},
        "context_stats": context_stats,
    }

    out_pt = MODELS_DIR / "semantic_norm_finetuned.pt"
    torch.save(save_data, out_pt)

    summary = {
        "n_examples": len(examples),
        "n_categories_adjusted": save_data["n_categories_adjusted"],
        "examples_per_category": save_data["examples_per_category"],
        "context_stats": context_stats,
        "files": {
            "finetuned_embeddings": str(out_pt),
            "examples": str(examples_path),
        },
    }
    out_summary = MODELS_DIR / "semantic_norm_context_summary.json"
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if verbose:
        print(f"Examples: {len(examples)}")
        print(f"Categories adjusted: {summary['n_categories_adjusted']}")
        print(f"Table-context assignments: {context_stats['n_table_context_assignments']}")
        print(f"KVP-context assignments:   {context_stats['n_kvp_context_assignments']}")
        print(f"Saved: {out_pt}")
        print(f"Saved: {examples_path}")
        print(f"Saved: {out_summary}")

    return {
        "ok": True,
        "summary": summary,
        "out_pt": str(out_pt),
        "out_examples": str(examples_path),
        "out_summary": str(out_summary),
    }


def main():
    train_from_gt_context(verbose=True)


if __name__ == "__main__":
    main()
