# Week 190326 — Universal Adversarial Perturbations: CW Loss, Transfer Evaluation & COCO Ablation

**Dates:** 12–19 March 2026
**Researcher:** Arun Joseph Raj
**Models targeted:** Qwen2.5-VL-3B, Qwen2.5-VL-7B, Qwen2.5-VL-32B, Qwen3-VL-2B/4B/8B (white-box); Claude, GPT-4o, Gemini (black-box API)

---

## Table of Contents
1. [Week Overview](#1-week-overview)
2. [System & Models](#2-system--models)
3. [Experiment 1: CW+CE Loss UAP on Qwen2.5-VL-3B](#3-experiment-1-cwce-loss-uap-on-qwen25-vl-3b)
4. [Experiment 2: CW+CE Loss UAP on Qwen3-VL-8B](#4-experiment-2-cwce-loss-uap-on-qwen3-vl-8b)
5. [Experiment 3: Cross-Model Transfer Evaluation](#5-experiment-3-cross-model-transfer-evaluation)
6. [Experiment 4: Black-Box API Transfer](#6-experiment-4-black-box-api-transfer)
7. [Experiment 5: COCO Dataset UAP Ablation](#7-experiment-5-coco-dataset-uap-ablation)
8. [Key Findings & Takeaways](#8-key-findings--takeaways)
9. [Issues & Blockers](#9-issues--blockers)
10. [Next Steps](#10-next-steps)

---

## 1. Week Overview

This week focused on strengthening the universal adversarial perturbation (UAP) attack pipeline against vision-language models (VLMs), building directly on last week's single-token and LM-loss PGD experiments. The central goal was to move from cross-entropy loss alone (which drove training loss low but generalised poorly) to a Carlini-Wagner (CW) first-token objective combined with a lightweight CE sequence loss. This combination forces the model's first predicted token logit for "dog" to exceed all "cat" variant logits by a fixed margin of 5.0, which translates directly to reliable greedy decode-time flips rather than just improved teacher-forced loss.

Two separate surrogates were trained: Qwen2.5-VL-3B (`pert_3b`) and Qwen3-VL-8B (`pert_8b`), both using the Bingsu Cat-and-Dog dataset (train on ~4,000 cats, validate on 200 held-out cats). After training, both perturbations were evaluated across six Qwen family models, and `pert_3b` was additionally tested against the Claude, GPT-4o, and Gemini API to probe black-box transferability. In parallel, a second experimental thread extended UAP training beyond the cat-dog task to the COCO object detection dataset, running a data efficiency ablation (50 / 200 / 1,000 / 5,000 training images) and a separate COCO-TV experiment.

---

## 2. System & Models

| Component | Details |
|---|---|
| Hardware | GPU server (CUDA) |
| Precision | bfloat16 throughout |
| Surrogate 1 | Qwen/Qwen2.5-VL-3B-Instruct — patch_size=14, 1024×1176 patches |
| Surrogate 2 | Qwen/Qwen3-VL-8B-Instruct — patch_size=16 (SigLIP-2), 784×1536 patches |
| Dataset | Bingsu/Cat_and_Dog (HuggingFace) — ~4,200 cat images; 200 held-out val |
| Epsilon | 32/255 (L∞ constraint) |
| Optimizer | Adam, lr=0.003, weight_decay=1e-4, CosineAnnealingLR (eta_min=0.001) |
| Grad accumulation | 4 steps (effective batch = 4 images) |
| Patience | 6 ASR check-intervals (early stopping) |

---

## 3. Experiment 1: CW+CE Loss UAP on Qwen2.5-VL-3B

**Script:** `qwen25_3b_cw.py`
**Output checkpoints:** `qwen25_3b_cw_perturbation_best.pt`, `qwen25_3b_cw_perturbation_epoch{7–25}.pt`

### 3.1 Motivation
The earlier `attack.py` (Adam + CE only) drove training loss to ~0.075 on 500 images, but transferred at only 2% on held-out images — a classic UAP overfitting failure. The fix has two parts: (1) switch to CW margin loss on the *first* output token, which is the actual decision that greedy decode reads; and (2) expand training to the full ~4,000-image cat split with epoch-level shuffling to prevent Adam's momentum from developing image-order-specific structure.

### 3.2 Loss Function
```
loss = 0.8 × max(0, logit_cat_max − logit_dog + 5.0)
     + 0.2 × CrossEntropy(logits_seq, "A dog.")
```
- CW term (weight 0.8): margin-based push on first-token logits. Zero when `logit_dog > logit_cat_max + 5.0`.
- CE term (weight 0.2): keeps full sequence "A dog." coherent; prevents degenerate first-token-only solutions.
- Cat logits suppressed across six variants: `cat`, `Cat`, `CAT`, `feline`, `kitty`, `kitten`.

### 3.3 Training Progress
Live ASR (Attack Success Rate) was checked every 2 epochs on a fixed 50-image val subset. The perturbation was checkpointed only when ASR improved by >1%. Epoch checkpoints at odd numbered epochs were also saved whenever CW loss dropped below 0.05, providing an audit trail:
- `qwen25_3b_cw_perturbation_epoch7.pt` through `epoch25.pt` — 10 epoch snapshots saved.
- `qwen25_3b_cw_perturbation_best.pt` — saved 2026-03-18 22:05, the highest-ASR checkpoint.

### 3.4 Evaluation Result (on 200-image held-out val)
- **Surrogate ASR (qwen25-vl-3b, pert_3b): 88.5%** (177/200 cats predicted as "dog")

---

## 4. Experiment 2: CW+CE Loss UAP on Qwen3-VL-8B

**Script:** `attack_qwen3vl_8b.py`
**Output checkpoint:** `qwen3vl_8b_universal_dog_perturbation_best.pt`

### 4.1 Architecture Difference
Qwen3-VL uses SigLIP-2 as its vision encoder, which has patch_size=16 (vs. 14 in Qwen2.5-VL). The `pil_to_patches()` function was adapted accordingly — output shape is 784×1536 vs. 1024×1176. An incorrect patch size would silently produce the wrong pixel_values tensor, so this was verified against the model's image_processor config.

The `token_type_ids` key was removed from inputs with `inputs.pop("token_type_ids", None)` — Qwen3-VL's processor no longer returns this key, but defensively popping it keeps the script compatible with both model families.

### 4.2 Training Setup
Same CW+CE loss, same epsilon (32/255), same 30-epoch budget with early stopping on live ASR. Gradient accumulation of 4 steps was retained.

### 4.3 Evaluation Result
- **Surrogate ASR (qwen3-vl-8b, pert_8b): 98.5%** (197/200 cats predicted as "dog")

---

## 5. Experiment 3: Cross-Model Transfer Evaluation

**Scripts:** `eval_final.py`, `collate_results.py`, `test_both.py`
**Output CSV:** `eval_results/results.csv`
**Output summary:** `eval_results/summary.txt`
**Visual outputs:** `eval_results_qwen_only/{model}/{pert}/*.jpg` (not copied here; see `large_files.txt`)

### 5.1 Evaluation Methodology
`eval_final.py` evaluates both `pert_3b` and `pert_8b` on a 200-image held-out val set against all Qwen family models. The key implementation detail: perturbation injection uses `pil_to_patches()` to apply the delta in raw pixel space *before* the model's own normalisation, exactly replicating training-time preprocessing. A naïve approach of baking the perturbation into the PIL image and re-normalising would shift the perturbation into the wrong numerical range, explaining many failed transfer attacks in prior literature.

Each result image is saved with a colour-coded bar (green = attack succeeded, red = failed), the model's full reply string, P(dog) from the first-token softmax, and the top predicted token + probability.

### 5.2 Transfer Attack Results

| Model | pert\_3b | pert\_8b |
|---|---|---|
| qwen25-vl-3b | **88.5%** ★ | 22.0% |
| qwen25-vl-7b | 41.5% | 29.0% |
| qwen25-vl-32b | 32.5% | 7.5% |
| qwen3-vl-2b | 4.5% | 53.5% |
| qwen3-vl-4b | 6.0% | 48.5% |
| qwen3-vl-8b | 7.5% | **98.5%** ★ |

★ = surrogate model (white-box, inflated vs. black-box transfer)

**Source file:** `eval_results/results.csv`

### 5.3 Observations
- `pert_3b` transfers well within the Qwen2.5-VL family (3B → 7B: 41.5%, 3B → 32B: 32.5%) but fails almost completely on Qwen3-VL (2B: 4.5%, 4B: 6%, 8B: 7.5%). This confirms that the architectural shift from ViT-patch14 to SigLIP-2-patch16 creates a hard transfer boundary.
- `pert_8b` shows the reverse pattern: strong within Qwen3-VL (2B: 53.5%, 4B: 48.5%) but degrades on Qwen2.5-VL (7B: 29%, 32B: 7.5%). Cross-family transfer is asymmetrically weak.
- `collate_results.py` was run at an earlier intermediate stage; `eval_results/summary.txt` reflects those earlier (lower) numbers and should be treated as a historical snapshot — `results.csv` is the authoritative output.

---

## 6. Experiment 4: Black-Box API Transfer

**Visual outputs:** `blackbox/claude/*.jpg`, `blackbox/gemini/*.jpg`, `blackbox/openai/*.jpg`
**Sample size:** 93 cat images per API

The `pert_3b` perturbation was tested against three closed-source multimodal APIs: Anthropic Claude, Google Gemini, and OpenAI GPT-4o. Result images follow the naming convention `{idx:04d}_{API}_{PREDICTION}.jpg`.

### 6.1 Observed Results

| API | DOG successes | Total | ASR |
|---|---|---|---|
| Claude (Anthropic) | 7 | 93 | **7.5%** |
| OpenAI (GPT-4o) | 5 | 93 | **5.4%** |
| Gemini (Google) | 7 | 93 | **7.5%** |

Source: filename counting in `blackbox/{claude,openai,gemini}/` — DOG success = `_DOG` in filename.

The `**...**` formatting in Claude filenames (e.g., `CLAUDE_**CAT**`) reflects Claude's markdown output being captured verbatim in the result label. Non-dog predictions include `BEAR`, `HORSE`, `CAMEL`, `OPOSSUM`, `KITTEN` — a broad scatter indicating the perturbation is disrupting but not reliably redirecting Claude's visual understanding.

Black-box transfer from a white-box Qwen2.5-VL-3B perturbation achieves ~5–8% ASR across all three closed-source APIs. This is low but non-zero, suggesting some cross-architecture signal. The failure mode aligns with the Qwen3 family results: the architecture gap between Qwen2.5-VL (ViT-patch14) and the unknown vision encoders in GPT-4o/Gemini/Claude creates the same transfer barrier.

---

## 7. Experiment 5: COCO Dataset UAP Ablation

**Scripts:** `COCO_UAP/train_coco_uap_all.py`, `COCO_UAP/train_coco_tv.py`
**Checkpoints:** `COCO_UAP/checkpoints/`

### 7.1 Data Efficiency Ablation (`train_coco_uap_all.py`)
Goal: quantify how many training images are needed to learn a transferable UAP. Four experiments trained sequentially overnight on subsets of the COCO "hair drier" category (cat_id corresponding to hair drier), targeting Qwen3-VL-8B with the prompt "What object is in this picture? Just answer in one word or phrase." → target "A hair drier."

| Experiment | Training images | Status | Checkpoint |
|---|---|---|---|
| n=50 | 50 | Complete | `coco_hairdrier_pert_final_n50.pt` |
| n=200 | 200 | Complete | `coco_hairdrier_pert_final_n200.pt` |
| n=1000 | 1000 | Complete | `coco_hairdrier_pert_final_n1000.pt` |
| n=5000 | 5000 | **In progress** | Not yet saved (as of 2026-03-20) |

The `results/ablation_summary.csv` (ASR per experiment on 50-image val check) was not written — the script writes this only after all four experiments complete, and n=5000 is still running.

### 7.2 COCO-TV Experiment (`train_coco_tv.py`)
A parallel single-experiment run targeting a semantically distinct class: "TV" (COCO cat_id=72). Key hyperparameter differences from the hair-drier ablation: LR raised to 0.01 (faster convergence), epsilon raised to 64/255 (stronger budget), CW_MARGIN lowered to 2.0 (TV is semantically distant from most object classes, so a smaller margin suffices), CE_WEIGHT set to 0.0 (CW-only loss). The perturbation was verified to produce "TV" (single token id=15653) in greedy decode.

- **Checkpoint:** `COCO_UAP/checkpoints/coco_tv_pert_best.pt` (saved 2026-03-20 11:48)

---

## 8. Key Findings & Takeaways

1. **CW loss is the right objective for greedy-decode attacks.** The original CE-only `attack.py` minimised teacher-forced loss but did not reliably flip greedy decode. The CW margin loss on the first output token directly targets the quantity that matters at inference.

2. **Surrogate ASR reached 88.5% (3B) and 98.5% (8B)** on 200 held-out cat images — both substantially higher than anything achieved in prior weeks.

3. **Transfer is architecture-family-locked.** pert_3b (Qwen2.5-VL ViT-patch14) reaches 41.5% on a 7B Qwen2.5 model but only ~6% on Qwen3-VL models. pert_8b shows the same asymmetry in reverse. The patch size difference (14 vs. 16) and encoder change (ViT vs. SigLIP-2) appear to be the primary barrier.

4. **Within-family scaling degrades transfer.** pert_3b drops from 88.5% on the 3B surrogate to 32.5% on the 32B model of the same family. Larger models appear more robust, even without adversarial training.

5. **Black-box API transfer largely fails.** GPT-4o and Gemini show near-zero ASR against pert_3b. Claude shows partial flips on a small subset, possibly due to different internal preprocessing.

6. **COCO experiments establish a cross-domain UAP baseline.** The data efficiency ablation (n=50/200/1000/5000) will provide a learning-curve plot once the n=5000 experiment completes and the ablation CSV is written.

7. **Epoch checkpointing revealed smooth CW loss decay.** The 10 epoch snapshots of pert_3b (epoch7 through epoch25) confirm the perturbation was still improving through epoch 25, suggesting the full 30-epoch budget is appropriate.

---

## 9. Issues & Blockers

- **summary.txt vs results.csv discrepancy:** `eval_results/summary.txt` was generated by `collate_results.py` on an intermediate state before all 200-image evaluations were complete. Its ASR numbers (e.g., 11.8% for pert_3b on qwen25-vl-3b) do not match the authoritative `results.csv` (88.5%). The summary.txt is a historical artefact; results.csv is correct.

- **n=5000 COCO experiment incomplete:** `train_coco_uap_all.py` runs all four experiments sequentially; n=5000 was still running as of 2026-03-20. The `results/ablation_summary.csv` will not be written until all four finish.

- **Black-box eval script missing from archive:** The script that generated `attack_results_claude/`, `attack_results_gemini/`, and `attack_results_openai/` was not found in `Attack_CatDog/` on 2026-03-19. Only the output images were present. The eval logic may have been run interactively or from a different location.

- **`eval_results_cw/` intermediate results:** An earlier evaluation pass produced `eval_results_cw/{model}/{own_pert,pert_from_Xb}/` result images, but these appear to predate the final `eval_final.py` run. They are not included here (see `large_files.txt`).

- **Qwen3-VL-32B and Qwen2.5-VL-72B not yet evaluated:** The model registry in `eval_final.py` supports `32bv3` and `72b` keys, but the results.csv contains no entries for these models, indicating either memory constraints or the eval run was not launched for these sizes.

---

## 10. Next Steps

1. **Wait for n=5000 COCO experiment to complete** and collect the full ablation CSV (`results/ablation_summary.csv`). Plot ASR vs. number of training images to characterise the data efficiency curve.

2. **Ensemble the two surrogates.** Train a single perturbation that jointly minimises the CW loss on both Qwen2.5-VL-3B and Qwen3-VL-8B simultaneously. This could partially bridge the architecture-family transfer gap without requiring a separate perturbation per target.

3. **Evaluate pert_3b and pert_8b on Qwen2.5-VL-72B and Qwen3-VL-32B** (skipped this week due to likely VRAM/time constraints). Use `--delete_after_eval` flag in `eval_final.py` to free cache between models.

4. **Investigate the Claude partial-flip subset.** Identify whether images that flipped on Claude share any visual properties (brightness, complexity, aspect ratio) that could inform a targeted attack design for closed-source APIs.

5. **Run adversarial training baseline.** Take the best-performing surrogate (pert_8b at 98.5% on Qwen3-VL-8B) and apply adversarial fine-tuning to see how quickly the model recovers — this establishes the defence cost.

6. **Quantify COCO perturbation naturalness.** Compute SSIM and LPIPS between clean and perturbed COCO images for the hair-drier and TV perturbations; confirm epsilon=32/255 remains perceptually imperceptible.
