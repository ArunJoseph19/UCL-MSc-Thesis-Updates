# THESIS_STATE.md — COCO_UAP Repository

> **Last updated:** 2026-03-28
> **Purpose:** Complete audit of every file, experiment, and result in this repository for MSc AI thesis (UCL) on adversarial attacks against Vision-Language Models.

---

## Project Overview

This repository implements **Universal Adversarial Perturbations (UAPs)** targeting Qwen3-VL multimodal language models. A UAP is a single, image-agnostic perturbation tensor that, when added to any input image, causes a Vision-Language Model to output a specific target label regardless of what the image actually contains. The thesis investigates (1) data efficiency of UAP training (how many images are needed), and (2) cross-model transferability (does a perturbation trained against an 8B surrogate fool 2B, 4B, and 32B variants). Two targeted mislabelling attacks are implemented: one causing the model to say "hair drier" (multi-token target, uses CW+CE combined loss), and one causing the model to say "TV" (single-token target, uses CW-only loss). All training uses COCO images as the source distribution. The surrogate model throughout is `Qwen/Qwen3-VL-8B-Instruct`.

---

## Repository Structure

```
COCO_UAP/
├── train_coco_uap_all.py            Training script — data efficiency ablation (hair drier target, 4 training-set sizes)
├── train_coco_tv.py                 Training script — single UAP training run (TV target, 1000-image training set)
├── eval_coco_tv.py                  Evaluation script — transfer attack across all 4 Qwen3-VL model sizes
├── cat72_sample_0.jpg               Sample COCO image from category 72 (TV/living room), used for visual verification
│
├── checkpoints/                     Saved perturbation tensors (PyTorch .pt files)
│   ├── coco_hairdrier_pert_final_n50.pt     Final perturbation from hair drier experiment, n=50 training images
│   ├── coco_hairdrier_pert_final_n200.pt    Final perturbation from hair drier experiment, n=200 training images
│   ├── coco_hairdrier_pert_final_n1000.pt   Final perturbation from hair drier experiment, n=1000 training images
│   ├── coco_tv_pert_best.pt                 Best-validation-ASR checkpoint from TV UAP training
│   └── coco_tv_pert_final.pt                Final-epoch checkpoint from TV UAP training
│
├── eval_results_coco_tv/            Annotated evaluation images + summary CSV
│   ├── results.csv                  ASR numbers for all 4 model sizes (200 images each)
│   ├── qwen3-vl-2b/                 200 annotated JPEGs, each labelled with model prediction (e.g. 0001_DOG.jpg)
│   ├── qwen3-vl-4b/                 200 annotated JPEGs (same structure)
│   ├── qwen3-vl-8b/                 200 annotated JPEGs — surrogate model, 139 labelled TV.jpg
│   └── qwen3-vl-32b/                200 annotated JPEGs
│
├── failures/                        Mirror of eval_results_coco_tv/ containing only failed-attack images
│   ├── qwen3-vl-2b/                 199 images (all non-TV predictions for 2B model)
│   ├── qwen3-vl-4b/                 198 images
│   ├── qwen3-vl-8b/                 61 images (misses for surrogate)
│   └── qwen3-vl-32b/                193 images
│
└── successes/                       Mirror containing only successful-attack images (TV predicted)
    ├── qwen3-vl-2b/                 1 image
    ├── qwen3-vl-4b/                 2 images
    ├── qwen3-vl-8b/                 139 images
    └── qwen3-vl-32b/                7 images

── MISSING (expected but not present) ──────────────────────────────────────────
checkpoints/coco_hairdrier_pert_best_n50.pt      Expected by train_coco_uap_all.py — never saved (see Known Issues)
checkpoints/coco_hairdrier_pert_best_n200.pt     Expected — never saved
checkpoints/coco_hairdrier_pert_best_n1000.pt    Expected — never saved
checkpoints/coco_hairdrier_pert_best_n5000.pt    Expected — n=5000 experiment never ran (see Known Issues)
checkpoints/coco_hairdrier_pert_final_n5000.pt   Expected — n=5000 experiment never ran
results/ablation_summary.csv                     Expected output of train_coco_uap_all.py — never written
```

---

## Python Scripts — Full Reference

### `train_coco_uap_all.py`

**Purpose:** Batch wrapper that runs four data-efficiency experiments sequentially overnight, training a UAP for each of four training-set sizes (50, 200, 1000, 5000 images) and saving a summary CSV.

**Usage:**
```bash
python train_coco_uap_all.py
# No CLI arguments — all settings are hardcoded constants at the top of the file.
# Requires HF_TOKEN environment variable for HuggingFace authentication.
```

**Key hyperparameters (hardcoded):**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `MODEL_ID` | `Qwen/Qwen3-VL-8B-Instruct` | Surrogate model |
| `H, W` | `448, 448` | Input resolution |
| `EPSILON` | `32/255 = 0.1255` | L-inf perturbation bound |
| `NUM_EPOCHS` | `30` | Max epochs per experiment |
| `LR` | `0.003` | Adam learning rate |
| `GRAD_ACCUM_STEPS` | `4` | Gradient accumulation |
| `CW_MARGIN` | `5.0` | Carlini-Wagner margin |
| `CW_WEIGHT` | `0.8` | Weight on CW loss component |
| `CE_WEIGHT` | `0.2` | Weight on cross-entropy loss |
| `PATIENCE` | `6` | Early stopping: epochs without improvement |
| `TARGET_CAPTION` | `"A hair drier."` | Multi-token target string |
| `QUESTION` | `"What object is in this picture? Just answer in one word or phrase."` | VLM prompt |
| `VAL_SIZE` | `200` | Total validation images |
| `ASR_CHECK_SIZE` | `50` | Validation subset for fast ASR checks |
| `EXPERIMENTS` | `[50, 200, 1000, 5000]` | Training set sizes to sweep |

**Logic walkthrough:**

1. Authenticates with HuggingFace via `HF_TOKEN` env variable.
2. Loads `Qwen3-VL-8B-Instruct` once into CUDA memory with `bfloat16` precision; freezes all parameters.
3. Extracts `image_mean` and `image_std` from the model's `AutoProcessor` image processor to use for normalisation.
4. Tokenises `TARGET_CAPTION = "A hair drier."` — gets token IDs and identifies `hair_id` (the first token) for CW loss targeting.
5. Loads `detection-datasets/coco` (train split) from HuggingFace datasets and filters out any image containing category ID 89 (hair drier) in its `objects.category` list. This prevents training on images that already show the target object.
6. Creates a fixed val set of 200 images and a 50-image ASR subset, **shared across all experiments** so comparisons are fair.
7. For each `num_train` in `[50, 200, 1000, 5000]`:
   - Samples `num_train` random training images from the filtered dataset.
   - Initialises a zero tensor of shape `(3, 448, 448)` as the perturbation, with `requires_grad=True`.
   - Optimises with Adam + cosine LR scheduler.
   - For each training image, calls `pil_to_patches()` to manually construct the patch representation that Qwen3-VL's vision encoder expects (bypassing the processor's normal image handling so the perturbation gradient flows through).
   - Computes **CW loss**: pushes the logit for `hair_id` above all competing logits by a margin of 5.0. Uses the logit at position `-(T+1)` in the output sequence (the first token of the assistant response).
   - Computes **CE loss**: full-sequence cross-entropy against the tokenised target caption, over positions `-(T+1):-1`.
   - Combined loss: `0.8 x CW + 0.2 x CE`, divided by `GRAD_ACCUM_STEPS=4` before `.backward()`.
   - After every 4 steps, calls `optimizer.step()` and clamps the perturbation to `[-EPSILON, EPSILON]`.
   - Every 2 epochs, runs `compute_asr()` on the 50-image ASR subset: generates text with `max_new_tokens=15, do_sample=False` and checks for `"hair"` AND `"drier"` in the decoded output.
   - Saves best checkpoint when ASR improves by >1%. Applies early stopping after `PATIENCE=6` consecutive check-epochs without improvement.
   - Saves final checkpoint regardless.
8. After all experiments complete, writes `results/ablation_summary.csv` with columns: `num_train`, `best_asr`, `checkpoint`.

**Inputs:** COCO train split (streamed from HuggingFace).
**Outputs:**
- `checkpoints/coco_hairdrier_pert_best_n{50,200,1000,5000}.pt` — best checkpoint per experiment
- `checkpoints/coco_hairdrier_pert_final_n{50,200,1000,5000}.pt` — final checkpoint per experiment
- `results/ablation_summary.csv` — summary table

**`pil_to_patches()` function:** The critical custom preprocessing function shared across all three scripts. Manually implements Qwen3-VL's vision encoder preprocessing to enable gradient flow through the perturbation:
1. Converts PIL image to `(3, H, W)` float tensor.
2. Adds the perturbation tensor and clamps to `[0, 1]`.
3. Normalises with the model's mean/std.
4. Repeats along time dimension (`temporal_patch_size=2`) to shape `(2, 3, H, W)`.
5. Reshapes into patches of size 16x16 to shape `(ph*pw, temporal_patch_size * C * patch_size^2)`.
6. Returns this as `inputs["pixel_values"]`, overriding what the processor would normally produce.

**`compute_asr()` function:** Runs the model in inference mode (`torch.no_grad()`). For each validation image: applies the perturbation via `pil_to_patches()`, generates up to 15 tokens greedily, decodes, and checks if both `"hair"` and `"drier"` appear in the lowercased output.

---

### `train_coco_tv.py`

**Purpose:** Trains a single UAP on 1000 COCO images to cause Qwen3-VL-8B-Instruct to respond "TV" when asked to identify any object. Evolved version of the hair drier attack with tuned hyperparameters.

**Usage:**
```bash
python train_coco_tv.py
# No CLI arguments — all settings are hardcoded.
# Requires HF_TOKEN environment variable.
```

**Key hyperparameters (hardcoded):**

| Parameter | Value | Notes vs. hair drier script |
|-----------|-------|---------------------------|
| `MODEL_ID` | `Qwen/Qwen3-VL-8B-Instruct` | Same surrogate |
| `H, W` | `448, 448` | Same |
| `EPSILON` | `64/255 = 0.251` | **Doubled** from hair drier (32 to 64) |
| `NUM_EPOCHS` | `30` | Same |
| `LR` | `0.01` | **3x higher** than hair drier (0.003 to 0.01) |
| `GRAD_ACCUM_STEPS` | `4` | Same |
| `CW_MARGIN` | `2.0` | **Lowered** from 5.0 (TV is 1 token, easier) |
| `CW_WEIGHT` | `1.0` | CW only |
| `CE_WEIGHT` | `0.0` | **CE dropped** entirely |
| `PATIENCE` | `6` | Same |
| `NUM_TRAIN` | `1000` | Single experiment |
| `VAL_SIZE` | `200` | Same |
| `ASR_CHECK_SIZE` | `50` | Same |
| `TARGET_CAPTION` | `"TV."` | Single-token target |
| `TARGET_WORD` | `"tv"` | ASR check: substring match |
| `TV_CAT_ID` | `72` | COCO category ID for TV/monitor |
| `QUESTION` | `"What object is in this picture? Answer in one word."` | Slightly shorter than hair drier prompt |

**Rationale for hyperparameter changes (from docstring comments):**
- LR `0.003 to 0.01`: faster convergence
- EPSILON `32/255 to 64/255`: stronger perturbation signal
- CW_MARGIN `5.0 to 2.0`: TV is semantically distant from typical COCO objects, easier to push
- CE_WEIGHT `0.2 to 0.0`: CE was "fighting the signal" for a semantically distant single-token target
- `"TV"` tokenises to a **single token** (ID 15653) in Qwen3-VL-8B, unlike `"A hair drier."` which is multi-token. Only CW on the single first token position is needed.

**Logic walkthrough:**

1. Same HF auth, model loading, processor extraction as the hair drier script.
2. Tokenises `"TV."` and verifies token structure. Confirms `tv_id` for CW loss.
3. Loads COCO train split and filters out images containing category ID 72 (TV/monitor).
4. Splits into train (1000) and val (200) sets; takes 50-image ASR check subset.
5. Initialises zero perturbation `(3, 448, 448)`, Adam + cosine scheduler.
6. For each training image: builds the chat-template input with an assistant turn containing `"TV."`, calls `pil_to_patches()`, runs the model forward pass.
7. Computes **CW loss only**: `clamp(logit_competitor - logit_tv + 2.0, min=0)` at the first response token position.
8. Applies grad accumulation (every 4 steps), clamps perturbation to `[-64/255, 64/255]`.
9. Every 2 epochs: ASR check (substring "tv" in decoded output), saves best checkpoint, applies early stopping.
10. Saves final checkpoint at end.

**Inputs:** COCO train split.
**Outputs:**
- `checkpoints/coco_tv_pert_best.pt` — best checkpoint by validation ASR
- `checkpoints/coco_tv_pert_final.pt` — final checkpoint

---

### `eval_coco_tv.py`

**Purpose:** Loads the trained TV perturbation and evaluates its attack success rate across all four Qwen3-VL model size variants (2B, 4B, 8B, 32B) on 200 non-TV COCO validation images. Generates annotated result images and a CSV summary.

**Usage:**
```bash
# Full evaluation (all 4 models, 200 images each):
python eval_coco_tv.py

# Subset of models:
python eval_coco_tv.py --models 2b 4b

# Override HuggingFace cache directory:
python eval_coco_tv.py --hf_cache /scratch/hf

# Delete each model from HF cache after evaluating it (saves disk space):
python eval_coco_tv.py --delete_after_eval

# Evaluate on fewer images:
python eval_coco_tv.py --n_eval 100
```

**CLI Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--models` | `nargs="+"` | `["2b","4b","8b","32b"]` | Which model sizes to evaluate |
| `--hf_cache` | `str` | `None` | Override `HF_HOME` and `TRANSFORMERS_CACHE` |
| `--delete_after_eval` | flag | `False` | Delete model weights from disk after each model |
| `--n_eval` | `int` | `200` | Number of non-TV COCO val images to use |

**Model ID mapping (hardcoded):**

| Key | HuggingFace model ID |
|-----|---------------------|
| `"2b"` | `Qwen/Qwen3-VL-2B-Instruct` |
| `"4b"` | `Qwen/Qwen3-VL-4B-Instruct` |
| `"8b"` | `Qwen/Qwen3-VL-8B-Instruct` (surrogate) |
| `"32b"` | `Qwen/Qwen3-VL-32B-Instruct` |

**Logic walkthrough:**

1. Parses args; optionally overrides HF cache dir.
2. Loads perturbation from `checkpoints/coco_tv_pert_best.pt` onto CUDA; reports L-inf norm and shape.
3. Loads COCO **val** split (not train), filters out category 72 (TV) images, takes first `n_eval` images.
4. Pre-loads all eval images into memory as `(448, 448)` RGB PIL images.
5. For each model in `args.models`:
   - Loads model and processor with `bfloat16`.
   - Confirms `"tv"` is a single token in this model's tokeniser (asserts `len(tv_ids) == 1`).
   - For each of the 200 images, calls `infer()`:
     - Runs the model forward pass with the perturbed image.
     - Reads logits at the last position to get `tv_prob` (softmax probability of the TV token) and the actual top token.
     - Runs `model.generate()` with `max_new_tokens=10` to get the full text reply.
     - Returns `(reply, tv_prob, top_token, top_prob)`.
   - Calls `save_result()` for each image:
     - Creates a JPEG with a 54px coloured banner at the bottom (green = success, red = failure).
     - Banner text: `pred: "..."  |  P(tv)=X%  |  top: "..." Y%`
     - Saves to `eval_results_coco_tv/{model_short}/{idx:04d}_{LABEL}.jpg`.
     - Mirrors to `successes/{model_short}/` or `failures/{model_short}/`.
   - Counts `hits` (where `"tv"` appears in the lowercased reply).
   - Appends row to CSV: `model, total, hits, misses, asr_pct, is_surrogate, elapsed_min`.
   - Deletes model from GPU memory; optionally deletes from HF cache.
6. Writes `eval_results_coco_tv/results.csv`.

**Inputs:**
- `checkpoints/coco_tv_pert_best.pt`
- COCO val split (from HuggingFace)

**Outputs:**
- `eval_results_coco_tv/results.csv`
- `eval_results_coco_tv/{model}/NNNN_{LABEL}.jpg` (200 annotated images per model)
- `successes/{model}/` and `failures/{model}/` mirror directories

---

## Experiments Completed

### Experiment 1 — Hair Drier UAP: Data Efficiency Ablation

**Script:** `train_coco_uap_all.py`
**Status:** PARTIALLY COMPLETE — n=50, 200, 1000 ran; n=5000 did not run
**Surrogate model:** `Qwen/Qwen3-VL-8B-Instruct`
**Dataset:** COCO train (filtered: hair drier / category 89 removed)
**Target:** `"A hair drier."` (multi-token)
**Question prompt:** `"What object is in this picture? Just answer in one word or phrase."`
**Hyperparameters:** epsilon=32/255, LR=0.003, epochs=30, CW_MARGIN=5.0, CW_WEIGHT=0.8, CE_WEIGHT=0.2, patience=6, grad_accum=4

**What ran:**
- n=50: final checkpoint saved (`coco_hairdrier_pert_final_n50.pt`); best checkpoint not saved
- n=200: final checkpoint saved (`coco_hairdrier_pert_final_n200.pt`); best checkpoint not saved
- n=1000: final checkpoint saved (`coco_hairdrier_pert_final_n1000.pt`); best checkpoint not saved
- n=5000: **did not run** — neither final nor best checkpoint exists

**What is missing:** `coco_hairdrier_pert_best_n*.pt` files (see Known Issues), n=5000 data, `results/ablation_summary.csv`

**Key result:** No ASR numbers are recoverable. The script measured ASR on a 50-image subset every 2 epochs but only printed to stdout — no log files were saved.

---

### Experiment 2 — TV UAP Training

**Script:** `train_coco_tv.py`
**Status:** COMPLETE
**Surrogate model:** `Qwen/Qwen3-VL-8B-Instruct`
**Dataset:** COCO train (filtered: TV / category 72 removed)
**Target:** `"TV."` (single token, ID 15653)
**Question prompt:** `"What object is in this picture? Answer in one word."`
**Hyperparameters:** epsilon=64/255, LR=0.01, epochs=30, CW_MARGIN=2.0, CW only (CE=0), patience=6, grad_accum=4, NUM_TRAIN=1000

**Checkpoints saved:**
- `checkpoints/coco_tv_pert_best.pt` — best validation ASR checkpoint
- `checkpoints/coco_tv_pert_final.pt` — final epoch checkpoint

**Key result:** Training ASR not preserved (no log file). Held-out evaluation (Experiment 3) shows 69.5% ASR on the 8B surrogate.

---

### Experiment 3 — TV UAP Transfer Evaluation

**Script:** `eval_coco_tv.py`
**Status:** COMPLETE
**Perturbation used:** `checkpoints/coco_tv_pert_best.pt`
**Dataset:** COCO val split (non-TV, category 72 filtered)
**N eval:** 200 images per model
**Models evaluated:** Qwen3-VL-2B, 4B, 8B, 32B (all Instruct variants)

---

## Key Results

All numerical results from `eval_results_coco_tv/results.csv`:

| Model | Total | Hits | Misses | ASR | Surrogate? | Eval time |
|-------|-------|------|--------|-----|-----------|-----------|
| qwen3-vl-2b | 200 | 1 | 199 | **0.5%** | No | 2.7 min |
| qwen3-vl-4b | 200 | 2 | 198 | **1.0%** | No | 3.8 min |
| qwen3-vl-8b | 200 | 139 | 61 | **69.5%** | Yes | 10.5 min |
| qwen3-vl-32b | 200 | 7 | 193 | **3.5%** | No | 27.6 min |

**Interpretation:** The TV UAP achieves 69.5% ASR on the 8B surrogate it was trained against. Transfer to other model sizes is near-zero: 0.5%, 1.0%, 3.5% for 2B, 4B, 32B respectively. For context, random chance of outputting "tv" across ~80 COCO categories is roughly 1.25%. The 2B result is below chance; 4B and 32B are at or just above chance. The attack does not transfer within the Qwen3-VL model family.

**Note:** No numerical results exist for the hair drier experiments.

---

## Models Used

| HuggingFace ID | Role | Used in |
|---------------|------|---------|
| `Qwen/Qwen3-VL-2B-Instruct` | Transfer target | `eval_coco_tv.py` |
| `Qwen/Qwen3-VL-4B-Instruct` | Transfer target | `eval_coco_tv.py` |
| `Qwen/Qwen3-VL-8B-Instruct` | **Surrogate (all training)** | all three scripts |
| `Qwen/Qwen3-VL-32B-Instruct` | Transfer target | `eval_coco_tv.py` |

All require `HF_TOKEN` authentication. All use `bfloat16` precision during both training and evaluation.

---

## Datasets Used

| Dataset | HuggingFace ID | Split | Filter |
|---------|---------------|-------|--------|
| COCO 2017 | `detection-datasets/coco` | `train` | Remove category 89 (hair drier) for Exp 1; category 72 (TV) for Exp 2 |
| COCO 2017 | `detection-datasets/coco` | `val` | Remove category 72 (TV) for Exp 3 |

Filtered via `dataset.filter(..., num_proc=4)`. Each example has an `image` (PIL) and `objects.category` (list of COCO category IDs).

`cat72_sample_0.jpg` is a locally-saved COCO image from category 72 used to visually confirm the category ID mapping.

---

## Checkpoints Index

| File | Produced by | Contents | Status |
|------|-------------|----------|--------|
| `checkpoints/coco_hairdrier_pert_final_n50.pt` | `train_coco_uap_all.py` | Final-epoch `(3,448,448)` bfloat16 perturbation, n=50 training images, target "A hair drier.", epsilon=32/255 | EXISTS |
| `checkpoints/coco_hairdrier_pert_final_n200.pt` | `train_coco_uap_all.py` | Same, n=200 | EXISTS |
| `checkpoints/coco_hairdrier_pert_final_n1000.pt` | `train_coco_uap_all.py` | Same, n=1000 | EXISTS |
| `checkpoints/coco_hairdrier_pert_best_n50.pt` | `train_coco_uap_all.py` | Best-ASR checkpoint, n=50 | MISSING |
| `checkpoints/coco_hairdrier_pert_best_n200.pt` | `train_coco_uap_all.py` | Best-ASR checkpoint, n=200 | MISSING |
| `checkpoints/coco_hairdrier_pert_best_n1000.pt` | `train_coco_uap_all.py` | Best-ASR checkpoint, n=1000 | MISSING |
| `checkpoints/coco_hairdrier_pert_final_n5000.pt` | `train_coco_uap_all.py` | Final checkpoint, n=5000 | MISSING — experiment never ran |
| `checkpoints/coco_hairdrier_pert_best_n5000.pt` | `train_coco_uap_all.py` | Best-ASR checkpoint, n=5000 | MISSING — experiment never ran |
| `checkpoints/coco_tv_pert_best.pt` | `train_coco_tv.py` | Best-validation-ASR `(3,448,448)` bfloat16 perturbation, target "TV.", epsilon=64/255, n=1000. **Used by eval_coco_tv.py.** | EXISTS |
| `checkpoints/coco_tv_pert_final.pt` | `train_coco_tv.py` | Final-epoch checkpoint for TV target | EXISTS |

All `.pt` files contain a single tensor saved with `torch.save(perturbation.detach().cpu().to(torch.bfloat16), path)`. The tensor is a pixel-space additive perturbation in range `[-epsilon, epsilon]`, unnormalised (normalisation is applied inside `pil_to_patches()`).

---

## Known Issues

### Issue 1: Best Checkpoints Not Saved for Hair Drier Experiments

**Evidence:** All three `coco_hairdrier_pert_best_n*.pt` files are absent. The `_final_` files exist, confirming training ran to completion, but the best-checkpoint save never triggered.

**Root cause (inferred):** The save condition is `if asr > best_val_asr + 0.01`. If the hair drier UAP never achieved meaningful ASR (e.g., stayed near 0% throughout), `best_val_asr` remains 0.0 and the save never executes. This is consistent with hair drier being a harder multi-token target at lower epsilon (32/255) vs. the TV experiment (64/255).

**Implication:** Only final-epoch weights are available for hair drier. No ASR numbers are recoverable.

### Issue 2: n=5000 Experiment Never Ran

**Evidence:** Neither `coco_hairdrier_pert_final_n5000.pt` nor `results/ablation_summary.csv` exists.

**Root cause (inferred):** The script runs experiments sequentially. The most likely cause is job timeout or manual interruption after n=1000 completed. The `results/` directory does not exist at all (it would be created by the script at the end).

**Implication:** Data efficiency ablation is incomplete — only 3 of 4 planned data points exist, and none have ASR numbers.

### Issue 3: No Log Files

**Evidence:** No `.txt`, `.log`, or similar files anywhere in the repository.

**Implication:** All training metrics (epoch losses, live ASR values, convergence curves) are permanently lost. Training progress was only printed to terminal.

### Issue 4: `successes/` Mirror Path Depends on Working Directory

**Evidence:** `eval_coco_tv.py` uses `Path("successes")` — a relative path. If the script was not run from `COCO_UAP/`, the mirror images were written to a different location on disk.

### Design Decisions (Documented in Code Comments)

- **CE loss dropped for TV target:** `"CE_WEIGHT 0.0 (CW only — CE was fighting the signal)"` — CE interfered when target token is semantically distant from common COCO labels.
- **"TV" is a single token (id=15653):** Verified in docstring; dynamically asserted in both training and evaluation scripts.
- **cat_id=72 = tv — confirmed visually:** `cat72_sample_0.jpg` documents this.
- **`token_type_ids` removed:** `inputs.pop("token_type_ids", None)` appears in all three scripts — handles cases where the processor includes this field, which Qwen3-VL does not accept.
- **`pixel_values` overridden after processor call:** The processor builds the tokenised text input normally, but its `pixel_values` output is replaced by the manually-patched version from `pil_to_patches()`. Required to allow gradient flow through the perturbation.

---

## What is Done vs. What is In Progress

### Complete

- [x] TV UAP training — both `coco_tv_pert_best.pt` and `coco_tv_pert_final.pt` exist
- [x] TV UAP evaluation — 800 annotated images, `results.csv` written, all 4 model sizes covered
- [x] Hair drier UAP training for n=50, 200, 1000 — final checkpoints exist
- [x] `pil_to_patches()` implementation — correctly implements Qwen3-VL patch encoding with differentiable perturbation; consistent across all three scripts
- [x] COCO dataset filtering — category 72 and 89 filters implemented and confirmed
- [x] Single-token target verification — "TV" confirmed as token ID 15653 with runtime assertion

### Incomplete

- [ ] Hair drier UAP training n=5000 — not run, no checkpoint
- [ ] Hair drier ASR numbers — no log files, all metrics lost
- [ ] Hair drier best checkpoints — `_best_` files absent for all n values
- [ ] `results/ablation_summary.csv` — never written; would need complete re-run
- [ ] Hair drier transfer evaluation — no eval script exists for hair drier perturbations
- [ ] Quantitative training curves — no logging infrastructure (no wandb/tensorboard/log files)

### Planned But Not Implemented

The printed "next step" in `train_coco_uap_all.py` references a script that does not exist:
```
python eval_coco.py --pert checkpoints/coco_hairdrier_pert_best_n1000.pt
    --models 3b 7b 2b 4b 8b --delete_after_eval
```
The `--models 3b 7b` flags also reference sizes not in the Qwen3-VL family, suggesting these were from an older plan.

---

## Reproduction Instructions

### Prerequisites
```bash
# Inferred from imports — no requirements.txt exists in the repo
pip install torch torchvision transformers datasets huggingface_hub tqdm Pillow

export HF_TOKEN=<your_token>
```

### Re-run TV UAP training:
```bash
cd /home/arun/COCO_UAP
python train_coco_tv.py
```

### Re-run TV UAP evaluation:
```bash
cd /home/arun/COCO_UAP
python eval_coco_tv.py
python eval_coco_tv.py --models 8b                 # surrogate only
python eval_coco_tv.py --delete_after_eval          # save disk space
```

### Complete the hair drier ablation (n=5000 only):
```bash
# Edit train_coco_uap_all.py line: EXPERIMENTS = [5000]
python train_coco_uap_all.py
```

---

## Notes for Thesis Write-Up

1. **Transfer attack context:** Random chance of outputting "tv" with ~80 COCO categories is ~1.25%. The 2B result (0.5%) is below chance; 4B (1.0%) is at chance; 32B (3.5%) is marginally above. None represent meaningful transfer. The attack is currently white-box only.

2. **Epsilon scale:** TV attack uses epsilon=64/255 (~0.25 L-inf). Imperceptible perturbations are typically <= 16/255. The hair drier attack uses 32/255. Neither has been evaluated for perceptual quality (SSIM, LPIPS, or human study).

3. **Missing ablation:** The data efficiency experiment (hair drier, n=50/200/1000/5000) is incomplete and has no ASR numbers. Cannot be reported without re-running.

4. **No reproducibility record:** No `requirements.txt`, conda env file, or Docker image. Exact package versions are unknown.

5. **Single surrogate:** All experiments use Qwen3-VL-8B as the sole surrogate. Cross-architecture transfer (e.g., to LLaVA, InternVL, or a non-Qwen VLM) has not been attempted.
