#!/usr/bin/env python3
"""
Runner script for pipeline testing with the new semantic normalization changes.

What it does:
1) Re-trains semantic normalization embeddings with GT context (table + KVP)
2) Runs evaluation pipeline (optional full mode)
3) Runs visualizations (best-vs-GT and dashboard)
4) Writes a consolidated run report

Usage:
  python run_pipeline_with_new_semantic_context.py
  python run_pipeline_with_new_semantic_context.py --full-eval
  python run_pipeline_with_new_semantic_context.py --skip-visuals
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
CONTRIB_ROOT = SCRIPT_DIR.parent
PROJECT = Path(os.environ.get("DOCUMENT_KVP_PROJECT_ROOT", CONTRIB_ROOT)).resolve()
sys.path.insert(0, str(SCRIPT_DIR))
PY = sys.executable


def run_cmd(cmd, cwd=None):
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd or PROJECT, capture_output=True, text=True)
    dt = time.time() - t0
    return {
        "cmd": " ".join(str(x) for x in cmd),
        "returncode": proc.returncode,
        "elapsed_sec": round(dt, 2),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-40:]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-eval", action="store_true", help="Run the invoice evaluation script after retraining")
    parser.add_argument("--eval-script", default=os.environ.get("INVOICE_EVAL_SCRIPT", "invoice_eval_only.py"))
    parser.add_argument("--skip-visuals", action="store_true", help="Skip visualization scripts")
    args = parser.parse_args()

    report = {
        "timestamp": datetime.now().isoformat(),
        "steps": [],
        "ok": True,
    }

    # Step 1: semantic training with context
    print("[1/3] Training semantic normalization with GT context...")
    t0 = time.time()
    from train_semantic_norm_with_gt_context import train_from_gt_context
    train_result = train_from_gt_context(verbose=True)
    report["steps"].append({
        "step": "train_semantic_with_gt_context",
        "elapsed_sec": round(time.time() - t0, 2),
        "result": train_result,
        "ok": bool(train_result.get("ok")),
    })
    if not train_result.get("ok"):
        report["ok"] = False

    # Step 2: evaluation
    if args.full_eval:
        print(f"[2/3] Running full evaluation ({args.eval_script})...")
        step_eval = run_cmd([PY, args.eval_script])
        step_eval["step"] = "full_eval"
        step_eval["ok"] = step_eval["returncode"] == 0
        report["steps"].append(step_eval)
        if step_eval["returncode"] != 0:
            report["ok"] = False
    else:
        print("[2/3] Skipping full evaluation (use --full-eval to enable).")
        report["steps"].append({
            "step": "full_eval",
            "ok": True,
            "skipped": True,
        })

    # Step 3: visualizations
    if not args.skip_visuals:
        print("[3/3] Generating visualizations...")

        vis1 = run_cmd([PY, "generate_best_inference_gt_visualizations.py"])
        vis1["step"] = "visualize_best_vs_gt"
        vis1["ok"] = vis1["returncode"] == 0
        report["steps"].append(vis1)
        if vis1["returncode"] != 0:
            report["ok"] = False

        vis2 = run_cmd([PY, "visualize_pipeline_with_gt_dashboard.py"])
        vis2["step"] = "visualize_gt_dashboard"
        vis2["ok"] = vis2["returncode"] == 0
        report["steps"].append(vis2)
        if vis2["returncode"] != 0:
            report["ok"] = False
    else:
        print("[3/3] Skipping visualizations (--skip-visuals).")
        report["steps"].append({"step": "visualizations", "ok": True, "skipped": True})

    out = CONTRIB_ROOT / "outputs" / "pipeline_new_semantic_context_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDone.")
    print(f"Overall status: {'OK' if report['ok'] else 'FAILED'}")
    print(f"Report: {out}")

    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
