"""
collate_results.py
------------------
Scans eval_results_qwen_only/ and produces:
  1. results.csv        — per-model per-perturbation ASR + counts
  2. summary.txt        — human-readable findings table
  3. failures/          — copies of red-bar (attack failed) images, organised by model/pert

Works on partial results — skips any folder with 0 images.
Safe to re-run after finishing the 32B models tomorrow.
"""

import os
import shutil
import csv
from pathlib import Path
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT         = Path("eval_results_qwen_only")   # where your result images live
FAILURES_DIR = Path("failures")                  # where to copy failure images
CSV_PATH     = ROOT / "results.csv"
SUMMARY_PATH = ROOT / "summary.txt"

# These are the surrogate models — used to train each perturbation.
# We flag them so we can read ASR with the right expectation
# (surrogate ASR is inflated vs transfer ASR).
SURROGATES = {
    "pert_3b": "qwen25-vl-3b",
    "pert_8b": "qwen3-vl-8b",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_folder(folder: Path) -> dict:
    """
    Given a result folder (e.g. eval_results_qwen_only/qwen25-vl-7b/pert_3b/),
    count successes and failures by reading filenames.

    Filename convention from save_result():
        success → 0042_DOG.jpg        (contains 'DOG')
        failure → 0042_CAT.jpg        (does NOT contain 'DOG')

    Returns a dict with keys: total, hits, misses, asr
    Returns None if folder is empty or doesn't exist.
    """
    if not folder.exists():
        return None

    images = list(folder.glob("*.jpg"))
    if not images:
        return None

    hits   = sum(1 for f in images if "_DOG." in f.name.upper())
    total  = len(images)
    misses = total - hits

    return {
        "total":  total,
        "hits":   hits,
        "misses": misses,
        "asr":    hits / total if total > 0 else 0.0,
        "images": images,       # keep reference for copying failures
    }


def copy_failures(images: list, dest: Path):
    """
    Copy all failure images (no _DOG_ in filename) to dest folder.
    Creates dest if it doesn't exist.
    Skips copy if file already exists (safe to re-run).
    """
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for img_path in images:
        if "_DOG." not in img_path.name.upper():   # failure = not predicted as dog
            out_path = dest / img_path.name
            if not out_path.exists():
                shutil.copy2(img_path, out_path)
            copied += 1
    return copied


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Discover all (model, pert) pairs that have any results
    rows = []

    # Walk model dirs in sorted order for stable output
    model_dirs = sorted([d for d in ROOT.iterdir() if d.is_dir()])

    for model_dir in model_dirs:
        model_name = model_dir.name

        # Walk perturbation subdirs (pert_3b, pert_8b)
        pert_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()])

        for pert_dir in pert_dirs:
            pert_name = pert_dir.name

            stats = parse_folder(pert_dir)
            if stats is None:
                # Folder empty or missing — model not evaluated yet
                print(f"  SKIP (no results): {model_name}/{pert_name}")
                continue

            # Is this the surrogate model for this perturbation?
            is_surrogate = (SURROGATES.get(pert_name) == model_name)

            rows.append({
                "model":        model_name,
                "perturbation": pert_name,
                "total":        stats["total"],
                "hits":         stats["hits"],
                "misses":       stats["misses"],
                "asr_pct":      round(stats["asr"] * 100, 1),
                "is_surrogate": is_surrogate,
            })

            # Copy failure images to failures/{model}/{pert}/
            failures_dest = FAILURES_DIR / model_name / pert_name
            n_copied = copy_failures(stats["images"], failures_dest)
            print(f"  {model_name:30s} | {pert_name} | "
                  f"ASR {stats['hits']}/{stats['total']} = {stats['asr']*100:.1f}% | "
                  f"{n_copied} failures → {failures_dest}")

    if not rows:
        print("\nNo results found. Have you run the eval yet?")
        return

    # ── Write CSV ──────────────────────────────────────────────────────────────
    fieldnames = ["model", "perturbation", "total", "hits", "misses", "asr_pct", "is_surrogate"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved → {CSV_PATH}")

    # ── Write summary table ────────────────────────────────────────────────────
    # Pivot: rows = models, cols = pert_3b / pert_8b
    models_seen = sorted(set(r["model"] for r in rows))
    perts_seen  = sorted(set(r["perturbation"] for r in rows))

    # Build lookup: (model, pert) -> row
    lookup = {(r["model"], r["perturbation"]): r for r in rows}

    col_w = 14   # column width for ASR cells

    lines = []
    lines.append("Universal Adversarial Perturbation — Transfer Attack Results")
    lines.append("=" * 70)
    lines.append(f"{'Model':<35}" + "".join(f"{p:>{col_w}}" for p in perts_seen))
    lines.append("-" * 70)

    for model in models_seen:
        line = f"{model:<35}"
        for pert in perts_seen:
            entry = lookup.get((model, pert))
            if entry is None:
                cell = "—"          # not evaluated yet
            else:
                tag  = " *" if entry["is_surrogate"] else ""
                cell = f"{entry['asr_pct']:.1f}%{tag}"
            line += f"{cell:>{col_w}}"
        lines.append(line)

    lines.append("-" * 70)
    lines.append("* = surrogate (perturbation was optimised against this model)")
    lines.append("")

    # Key findings: best transfer target per perturbation
    lines.append("Key findings:")
    for pert in perts_seen:
        pert_rows = [r for r in rows if r["perturbation"] == pert and not r["is_surrogate"]]
        if not pert_rows:
            continue
        best  = max(pert_rows, key=lambda r: r["asr_pct"])
        worst = min(pert_rows, key=lambda r: r["asr_pct"])
        lines.append(f"  {pert}: best transfer → {best['model']} ({best['asr_pct']}%)  |  "
                     f"worst → {worst['model']} ({worst['asr_pct']}%)")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    with open(SUMMARY_PATH, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSummary saved → {SUMMARY_PATH}")
    print(f"Failure images → {FAILURES_DIR}/")


if __name__ == "__main__":
    main()