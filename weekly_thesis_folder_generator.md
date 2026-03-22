# Weekly Thesis Folder Generator
# ================================
# Fill in the two variables below, then paste this entire block into Claude.
# Claude will build the folder, copy files, and report back.
#
# FOLDER_NAME : date in DDMMYY format, e.g. 260326
# SERVER_ROOT : top-level directory to search, e.g. /home/arun
# OUTPUT_DIR  : where to create the folder, e.g. /home/arun
# DATE_FROM   : start of the week, e.g. 2026-03-19
# DATE_TO     : end of the week, e.g. 2026-03-26

---

## Prompt (paste into Claude Code)

```
I want to build a weekly thesis progress folder. Here are the parameters:

FOLDER_NAME = 260326
SERVER_ROOT = /home/arun
OUTPUT_DIR  = /home/arun
DATE_FROM   = 2026-03-19
DATE_TO     = 2026-03-26

---

## Step 1 — Find all relevant files

Run this on the server:

```bash
find SERVER_ROOT -type f \
  -newermt "DATE_FROM 00:00:00" \
  ! -newermt "DATE_TO 23:59:59" \
  ! -path "*/.git/*" \
  ! -path "*/__pycache__/*" \
  ! -path "*/venv*" \
  ! -path "*/.venv*" \
  ! -path "*/node_modules/*" \
  ! -path "*/hub/models*" \
  ! -path "*/.cache/*" \
  2>/dev/null | sort
```

From the results, classify every file as one of:
- **code**    : .py .sh .yaml .yml .json .toml .cfg .ipynb
- **log**     : .log .txt (output logs only, not READMEs)
- **result**  : .csv .tsv .html (under 50MB only)
- **plot**    : .png .jpg .jpeg .pdf (under 50MB only)
- **checkpoint** : .pt .pkl .npy .h5 (almost certainly over 50MB — log to large_files.txt)
- **ignore**  : anything in .git, __pycache__, venv, .cache, node_modules

For each file note: full path, filename, size in KB/MB.

If zero files are found, stop and tell me — I may need to adjust SERVER_ROOT or dates.

---

## Step 2 — Build the folder structure

Create OUTPUT_DIR/FOLDER_NAME/ with this layout:

```
FOLDER_NAME/
├── README.md          ← placeholder only (I will write this separately)
├── scripts/           ← all code files found
├── logs/              ← all .log and output .txt files
├── results/           ← .csv, .tsv, small .html files (under 50MB)
├── plots/             ← .png, .jpg, .pdf files (under 50MB)
└── large_files.txt    ← full path + size of every file over 50MB
```

Rules:
- Copy files into the correct subfolder. Preserve original filenames.
- If a filename already exists in the destination, prefix with the parent folder name to avoid collisions, e.g. COCO_UAP__train_coco_tv.py
- If a file is over 50MB, do NOT copy it. Write its full path and size to large_files.txt instead.
- If large_files.txt would be empty, still create it with the text "No large files this week."

---

## Step 3 — Write a placeholder README.md

Write OUTPUT_DIR/FOLDER_NAME/README.md with exactly this content,
filling in FOLDER_NAME and the file lists you found:

---
# Week FOLDER_NAME

**Dates:** DATE_FROM – DATE_TO
**Researcher:** Arun Joseph Raj

---

## Table of Contents
1. [Week Overview](#1-week-overview)
2. [System & Models](#2-system--models)
3. [Experiments](#3-experiments)
4. [Key Findings & Takeaways](#4-key-findings--takeaways)
5. [Issues & Blockers](#5-issues--blockers)
6. [Next Steps](#6-next-steps)

---

## 1. Week Overview

> TODO — fill in after reviewing results.

---

## 2. System & Models

| Component | Details |
|-----------|---------|
| Hardware | |
| Models | |
| Dataset | |
| Epsilon | |
| Optimizer | |

---

## 3. Experiments

> TODO — add one section per experiment. Reference scripts by filename.

---

## 4. Key Findings & Takeaways

| Finding | Evidence |
|---------|---------|
| | |

---

## 5. Issues & Blockers

- TODO

---

## 6. Next Steps

1. TODO

---

## File Index

### Scripts
LIST_EVERY_SCRIPT_COPIED_HERE

### Logs
LIST_EVERY_LOG_COPIED_HERE

### Results
LIST_EVERY_RESULT_COPIED_HERE

### Plots
LIST_EVERY_PLOT_COPIED_HERE

### Large files (not copied)
See large_files.txt
---

---

## Step 4 — Report back to me with:

1. The output of: `tree OUTPUT_DIR/FOLDER_NAME/`
2. Total files copied vs skipped
3. Any filename collisions that were renamed
4. Any files that looked ambiguous (couldn't classify the type)
5. Confirm: is large_files.txt populated or empty?
```
