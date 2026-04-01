# THESIS_STATE.md

> Auto-generated exhaustive documentation of this repository.
> Generated: 2026-03-28. Working directory: `/UCL-MSc-Thesis-Updates/`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Experiments Completed](#3-experiments-completed)
4. [Key Results — Numerical Table](#4-key-results--numerical-table)
5. [Models Used](#5-models-used)
6. [Datasets Used](#6-datasets-used)
7. [Scripts Reference](#7-scripts-reference)
8. [Checkpoints Index](#8-checkpoints-index)
9. [Known Issues and Bugs Fixed](#9-known-issues-and-bugs-fixed)
10. [Done vs. In Progress](#10-done-vs-in-progress)

---

## 1. Project Overview

This repository documents the practical experimental work underpinning an MSc AI thesis at University College London, provisionally titled **"Beyond Model-Level Performance: Evaluating and Interpreting Agentic Failures in Vision-Language Models."** The thesis asks six research questions spanning (RQ1) benchmarking VLMs on composite task suites, (RQ2) constructing novel composite benchmarks, (RQ3) attacking VLMs with white-box and transferable adversarial perturbations, and (ERQ4–6) mechanistic interpretability of the attack vectors. The repository is organised by week (`120326`, `190326`, `260326`) and progresses from exploratory white-box image attacks on Qwen2.5-VL → universal adversarial perturbations (UAP) trained with Carlini-Wagner + cross-entropy loss → ensemble multi-surrogate UAP training → a novel **agentic attack** (Attack 3) that forces web-agent VLMs to hallucinate task completion on VisualWebArena screenshots. Core findings include: (i) CW-margin loss dramatically outperforms CE-only loss for greedy-decode flipping; (ii) architecture families (ViT-patch14 vs SigLIP-2-patch16) create hard transfer walls; (iii) equal-weight ensemble training across 2B+4B+8B Qwen3-VL surrogates achieves a 4.4× transfer improvement to the unseen 32B model; and (iv) Attack 3 reaches 100 % ASR on VisualWebArena screenshots with a visually imperceptible perturbation.

---

## 2. Repository Structure

```
UCL-MSc-Thesis-Updates/
│
├── README.md                              # Thesis abstract, six research questions, high-level plan
├── THESIS_STATE.md                        # This file — exhaustive state documentation
├── weekly_thesis_folder_generator.md      # Template / instructions for creating weekly work folders
│
├── 120326/                                # Week 1 (12 March 2026) — exploratory white-box attacks
│   ├── README.md                          # 587-line experiment log: CLIP transfer (failed), LM-loss PGD,
│   │                                      #   CAPTCHA bypass (100%), Amazon redirect (100%)
│   ├── CLIP_ET_CLIP_Ensemble_Transfer_Attack_on_Qwen2_5_VL.ipynb
│   │                                      # MI-FGSM CLIP ensemble transfer — failed to fool Qwen
│   ├── CLIP_ET_CLIP_Ensemble_Transfer_Attack_on_Qwen2_5_VL_(2).ipynb
│   │                                      # Second CLIP+Qwen dual-target variant — also failed
│   ├── Adverserial Amazon Search.ipynb    # Amazon product bounding-box redirection attack (100% ASR)
│   ├── Captcha/                           # CAPTCHA bypass images (numeric + Kaggle alphanumeric)
│   │   └── [result PNGs]
│   ├── WebAgent/                          # Web agent redirection attack images
│   │   └── [result PNGs]
│   └── plots/                             # All result visualisations from week 1
│       └── [PNG plots]
│
├── 190326/                                # Week 2 (19 March 2026) — UAP training + transfer evaluation
│   ├── README.md                          # 218-line experiment log: CW+CE loss, 88.5%/98.5% ASR,
│   │                                      #   cross-model transfer table, COCO ablation status
│   ├── attack.py                          # Baseline UAP trainer (Adam + CE loss only) — 2% transfer
│   ├── qwen25_3b_cw.py                    # CW+CE UAP for Qwen2.5-VL-3B → 88.5% ASR
│   ├── attack_qwen3vl_8b.py              # CW+CE UAP for Qwen3-VL-8B  → 98.5% ASR
│   ├── eval_final.py                      # Multi-model evaluation across full Qwen family
│   ├── test.py                            # 100-image black-box eval (Claude/GPT-4o/Gemini APIs)
│   ├── test_both.py                       # 100-image cross-API eval — second variant
│   ├── collate_results.py                 # Aggregates eval images → CSV + pivot table
│   ├── large_files.txt                    # List of result folders >50 MB (not committed to git)
│   │
│   ├── qwen_universal_dog_perturbation_best.pt       # 1.1 MB — baseline CE-only pert (2% transfer)
│   ├── qwen3vl_8b_universal_dog_perturbation_best.pt # 1.1 MB — Qwen3-VL-8B pert, 98.5% ASR
│   ├── qwen25_3b_cw_perturbation_best.pt             # 1.1 MB — Qwen2.5-VL-3B CW pert, 88.5% ASR
│   │
│   ├── eval_results/
│   │   ├── results.csv                    # 12-row table: model × pert × ASR (final authoritative)
│   │   └── summary.txt                    # Older intermediate snapshot (superseded by results.csv)
│   │
│   └── COCO_UAP/                          # COCO dataset experiments
│       ├── train_coco_uap_all.py          # Sequential 4-run ablation: n_train ∈ {50,200,1000,5000}
│       ├── train_coco_tv.py               # Single-run COCO TV category UAP (CW-only, LR=0.01)
│       └── checkpoints/
│           ├── coco_hairdrier_pert_final_n50.pt   # 1.1 MB — n=50 hair drier pert
│           ├── coco_hairdrier_pert_final_n200.pt  # 1.1 MB — n=200 hair drier pert
│           ├── coco_hairdrier_pert_final_n1000.pt # 1.1 MB — n=1000 hair drier pert
│           └── coco_tv_pert_best.pt               # 1.1 MB — TV pert, 69.5% ASR on Qwen3-VL-8B
│
└── 260326/                                # Week 3 (26 March 2026) — agentic attacks + ensemble UAP
    ├── README.md                          # 368-line log: Attack 3 (100% ASR), ensemble (15.5% on 32B),
    │                                      #   ablation: equal_avg > weighted > max_loss > grad_alignment
    │
    ├── scripts/
    │   ├── train_attack3_stop.py          # Attack 3 trainer: VWA screenshots → "stop" token, 100% ASR
    │   ├── train_coco_tv_ensemble.py      # Ensemble UAP: Qwen3-VL-2B+4B+8B joint CW loss
    │   ├── eval_coco_tv.py                # Transfer eval across all Qwen3-VL sizes
    │   ├── demo_attack3.py                # Single-step demo on 5 held-out VWA screenshots
    │   ├── demo_trajectory.py             # Multi-step 3-task trajectory demo (HTML output)
    │   ├── visualise_attack3.py           # Perturbation ×1/×10/×50 magnification visualisation
    │   ├── train_coco_tv_original.py      # Variant of TV training (alternate hyperparams)
    │   ├── eval_coco_tv_original.py       # Eval for original TV variant
    │   ├── train_ensemble_ablation.py     # 4-strategy aggregation ablation (200 train / 100 val)
    │   ├── extract_screenshots.py         # Parses VWA HTML trajectories → PNG screenshots
    │   └── quick_test.py                  # Smoke test: load pert, run 3 images, verify output
    │
    ├── checkpoints/
    │   ├── attack3_stop_best.pt           # 1.1 MB — Attack 3, epoch 6, 100% ASR, L-inf=0.2520
    │   ├── attack3_stop_final.pt          # 1.1 MB — Attack 3, epoch 18, early stop
    │   ├── coco_tv_ensemble_best.pt       # 1.1 MB — Ensemble (2B+4B+8B), epoch 12, 50% ASR on 8B
    │   ├── coco_tv_ensemble_final.pt      # 1.1 MB — Ensemble, epoch 30 (final)
    │   └── coco_tv_pert_best.pt           # 1.1 MB — Copy of 190326 single-surrogate TV baseline
    │
    ├── logs/
    │   └── attack3_stop.log               # 40-line training log: CW loss per epoch, live ASR checks
    │
    └── results/
        ├── results.csv                    # Single-surrogate COCO TV eval: 4 models, ASR per model
        ├── trajectory_demo.html           # Self-contained HTML: 3 tasks × 5 steps, clean vs attacked
        └── attack3_visualisation.html     # Perturbation imperceptibility demo, L-inf=0.2520
```

---

## 3. Experiments Completed

### Experiment 1 — CLIP Ensemble Transfer (MI-FGSM)
- **Attack**: MI-FGSM gradient attack targeting CLIP embedding similarity across an ensemble of CLIP model variants
- **Surrogate**: CLIP ViT-B/32, ViT-L/14 ensemble
- **Target**: Qwen2.5-VL-3B
- **Dataset**: Single adversarial image
- **Hyperparameters**: Standard MI-FGSM (μ=1.0, ε=16/255, steps=50)
- **Result**: FAILED — CLIP embedding alignment did not transfer to Qwen vision encoder
- **Files**: `120326/CLIP_ET_CLIP_Ensemble_Transfer_Attack_on_Qwen2_5_VL.ipynb` (two variants)

### Experiment 2 — White-Box LM-Loss PGD (Multi-Stage)
- **Attack**: Direct gradient backpropagation through quantized Qwen2.5-VL with CrossEntropy loss on LM head logits, applied to single images
- **Surrogate**: Qwen2.5-VL-3B (white-box)
- **Dataset**: Single images (cat, giraffe, various)
- **Hyperparameters**: PGD with ε=8–64/255, momentum=0.9, steps=50–200
  - Stage 1: semantic disruption → 47% loss reduction
  - Stage 2: targeted caption attack → 57% total loss reduction
  - Single-token "dog" attack: loss = 0.0000 by step 50
  - Local "giraffe" patch attack: model outputs full giraffe scene with hallucinated bow tie
- **Result**: White-box attacks succeed fully on single images; semantic collapse demonstrated
- **Files**: `120326/*.ipynb` notebooks

### Experiment 3 — CAPTCHA Bypass
- **Attack**: White-box PGD targeting VLM-predicted character token; local patch, ε=8–16/255
- **Surrogate**: Qwen2.5-VL-3B (white-box)
- **Dataset**: Numeric CAPTCHA (multicolorcaptcha), Kaggle Version 2 alphanumeric
- **Result**: 100% success rate on first-digit/first-character recognition
- **Files**: `120326/Captcha/`, `120326/*.ipynb`

### Experiment 4 — Amazon Web Agent Product Redirection
- **Attack**: PGD on bounding box prediction; forces correct product detection to adversarial product
- **Surrogate**: Qwen2.5-VL-3B (white-box)
- **Dataset**: Amazon product page screenshot
- **Result**: 100% success, model outputs adversarial bounding box
- **Files**: `120326/Adverserial Amazon Search.ipynb`, `120326/WebAgent/`

### Experiment 5 — Baseline UAP (CE-only, Adam)
- **Attack**: Universal adversarial perturbation optimised with Adam + CE loss to output "A dog." on cat images
- **Surrogate**: Qwen2.5-VL-3B (white-box)
- **Dataset**: Bingsu/Cat_and_Dog (~4,200 cat images)
- **Hyperparameters**: LR=0.003, ε=16/255, no momentum, CE-only loss, grad_accum=4
- **Result**: Training loss 0.075 but only 2% generalisation — CE-only loss identified as insufficient for greedy-decode flipping
- **Checkpoint**: `190326/qwen_universal_dog_perturbation_best.pt` (1.1 MB)
- **Files**: `190326/attack.py`

### Experiment 6 — UAP CW+CE Loss on Qwen2.5-VL-3B
- **Attack**: UAP with dual CW-margin loss (0.8×CW + 0.2×CE); suppresses all cat token variants
- **Surrogate**: Qwen2.5-VL-3B (white-box)
- **Target**: First-token "dog" (with "cat"/"Cat"/"CAT"/"feline"/"kitty"/"kitten" as suppressed tokens)
- **Dataset**: Bingsu/Cat_and_Dog (~4,200 cat images)
- **Hyperparameters**: LR=0.003, ε=16/255, CW_MARGIN=5.0, patience=6, grad_accum=4, epoch checkpoint if CW<0.05 after epoch 5
- **Result**: 88.5% ASR on 200-image val set; 41.5% transfer to Qwen2.5-VL-7B; 32.5% to 32B; ~4–7.5% to Qwen3-VL family (hard architecture boundary)
- **Checkpoint**: `190326/qwen25_3b_cw_perturbation_best.pt` (1.1 MB); epoch snapshots epoch 7–25
- **Files**: `190326/qwen25_3b_cw.py`

### Experiment 7 — UAP CW+CE Loss on Qwen3-VL-8B
- **Attack**: Same CW+CE formulation adapted for Qwen3-VL-8B (SigLIP-2 encoder, patch_size=16)
- **Surrogate**: Qwen3-VL-8B (white-box)
- **Target**: First-token "dog"
- **Dataset**: Bingsu/Cat_and_Dog
- **Hyperparameters**: Same as Exp. 6; token_type_ids defensively popped (Qwen3 doesn't return them); output shape (784, 1536) vs Qwen2.5 (1024, 1176)
- **Result**: 98.5% ASR on 200-image val set; 53.5% transfer to Qwen3-VL-2B; 48.5% to 4B; 7.5% to 32B; ~22–7.5% transfer to Qwen2.5-VL family (reverse cross-architecture barrier)
- **Checkpoint**: `190326/qwen3vl_8b_universal_dog_perturbation_best.pt` (1.1 MB)
- **Files**: `190326/attack_qwen3vl_8b.py`

### Experiment 8 — COCO Data-Efficiency Ablation (Hair Drier, n=50/200/1000/5000)
- **Attack**: UAP on COCO training images, CW+CE, target "A hair drier."
- **Surrogate**: Qwen3-VL-8B (white-box)
- **Dataset**: COCO train split; hair-drier images excluded (cat_id=89); n ∈ {50, 200, 1000, 5000}
- **Hyperparameters**: LR=0.003, fresh random seed per run
- **Status**: n=50/200/1000 complete as of 2026-03-20; n=5000 still running (overnight GPU job)
- **Checkpoints**: `190326/COCO_UAP/checkpoints/coco_hairdrier_pert_final_n50.pt`, `_n200.pt`, `_n1000.pt`
- **Files**: `190326/COCO_UAP/train_coco_uap_all.py`

### Experiment 9 — COCO TV Category UAP (Single Surrogate Baseline)
- **Attack**: CW-only UAP on COCO TV category
- **Surrogate**: Qwen3-VL-8B (white-box)
- **Target**: First-token "TV" (token id=15653)
- **Dataset**: COCO train; TV images excluded (cat_id=72); 500 train / 200 val / 50 ASR check
- **Hyperparameters**: LR=0.01, ε=64/255, CW_MARGIN=2.0, CE_WEIGHT=0.0 (CW-only)
- **Result**: 69.5% ASR on Qwen3-VL-8B surrogate; 3.5% transfer to Qwen3-VL-32B
- **Checkpoint**: `190326/COCO_UAP/checkpoints/coco_tv_pert_best.pt` (1.1 MB)
- **Files**: `190326/COCO_UAP/train_coco_tv.py`

### Experiment 10 — Transfer Evaluation Across Full Qwen Family
- **Attack**: Evaluate pert_3b and pert_8b on all 8 Qwen models
- **Models evaluated**: Qwen2.5-VL (3B, 7B, 32B, 72B) + Qwen3-VL (2B, 4B, 8B, 32B)
- **Dataset**: 200 cat images from Bingsu/Cat_and_Dog val split
- **Key detail**: Perturbation injected via `pil_to_patches()` BEFORE processor normalisation (naive PIL-bake approach fails); result images colour-coded green/red for ASR visualisation
- **Result**: See Key Results table (Section 4); architecture families create hard transfer wall
- **Files**: `190326/eval_final.py`, `190326/eval_results/results.csv`

### Experiment 11 — Black-Box API Transfer (Claude, GPT-4o, Gemini)
- **Attack**: Forward-pass black-box transfer — send perturbed images to closed-source API models
- **Surrogate**: Qwen2.5-VL-3B (pert_3b)
- **Targets**: Anthropic Claude (unspecified version), OpenAI GPT-4o, Google Gemini
- **Dataset**: 100 cat images
- **Result**: Claude 7.5%, GPT-4o 5.4%, Gemini 7.5% — effectively near-zero practical transfer
- **Files**: `190326/test.py`, `190326/test_both.py`

### Experiment 12 — Ensemble UAP Training (2B+4B+8B Equal-Weight)
- **Attack**: Joint CW loss across three Qwen3-VL surrogate models simultaneously; equal-weight gradient averaging
- **Surrogates**: Qwen3-VL-2B, 4B, 8B (all frozen, all in VRAM simultaneously; 28.6 GB / 121.7 GB peak)
- **Target**: First-token "TV"
- **Dataset**: COCO train (TV excluded); 500 train / 200 val / 50 ASR check
- **Hyperparameters**: Loss = (CW_2B + CW_4B + CW_8B) / 3; LR=0.01, ε=64/255, CW_MARGIN=2.0; training time: 686 minutes
- **Result**: 50% ASR on 8B (surrogate); 15.5% transfer to Qwen3-VL-32B (vs 3.5% single-surrogate = 4.4× improvement); surrogate ASR tradeoff: 69.5% → 50% (multi-task optimisation cost)
- **Checkpoint**: `260326/checkpoints/coco_tv_ensemble_best.pt` (epoch 12), `coco_tv_ensemble_final.pt` (epoch 30)
- **Files**: `260326/scripts/train_coco_tv_ensemble.py`

### Experiment 13 — Ensemble Aggregation Ablation (4 Strategies)
- **Attack**: Same as Exp. 12 but comparing 4 gradient aggregation strategies
- **Strategies tested**:
  1. `equal_avg` — average CW losses equally → **15.5% on 32B** (BEST)
  2. `weighted` — 0.2/0.2/0.6 weights for 2B/4B/8B → ~3% on 32B
  3. `max_loss` — backprop through max(CW_2B, CW_4B, CW_8B) → 54% on 8B, 4% on 32B
  4. `grad_alignment` — strict sign-agreement mask across surrogates → 0% everywhere
- **Dataset**: Smaller subset: 200 train / 100 val
- **Files**: `260326/scripts/train_ensemble_ablation.py`

### Experiment 14 — Attack 3: Agentic Task Completion Hallucination
- **Attack**: UAP that forces a web-agent VLM to output the `stop` action (task completion) regardless of actual page state; model confabulates reasons
- **Surrogate**: Qwen3-VL-8B (white-box, as web agent)
- **Target token**: "stop" (token id=9495); full target: "stop [Task completed successfully]"
- **Dataset**: 1,913 VWA (VisualWebArena) screenshots from GPT-4V trajectories; 14-task pool; split 800 train / 150 val / 50 ASR check
- **Hyperparameters**: CW loss at logit position `-9` (teacher-forced sequence length), ε=64/255, LR=0.01
- **System prompt**: Simulates web agent with valid action set (click, type, scroll, goto, stop, etc.)
- **Result**: 100% ASR (50-image eval) by epoch 6 in 70.6 minutes; L-inf=0.2520; model hallucinates false narratives (corrupted page, scanned document, Airstream RV listing, etc.); training continued to epoch 18 (219.8 min total) with early stop
- **Checkpoint**: `260326/checkpoints/attack3_stop_best.pt` (epoch 6), `attack3_stop_final.pt` (epoch 18)
- **Files**: `260326/scripts/train_attack3_stop.py`, `260326/logs/attack3_stop.log`

### Experiment 15 — Attack 3 Demo: Single-Step and Multi-Step Trajectories
- **Attack**: Load `attack3_stop_best.pt`; apply to held-out VWA screenshots (not seen during training)
- **Result**: Single-step demo: 100% success (5/5 screenshots); multi-step trajectory demo: 100% success (3 tasks × 5 steps; agents halt early with hallucinated content)
- **Output files**: `260326/results/trajectory_demo.html`, `260326/results/attack3_visualisation.html`
- **Files**: `260326/scripts/demo_attack3.py`, `260326/scripts/demo_trajectory.py`, `260326/scripts/visualise_attack3.py`

---

## 4. Key Results — Numerical Table

### 4.1 UAP White-Box (Surrogate) ASR

| Model | Target Label | Dataset | ASR (%) | Checkpoint |
|---|---|---|---|---|
| Qwen2.5-VL-3B | "dog" (CE-only) | Cat_and_Dog | 2.0 | `190326/qwen_universal_dog_perturbation_best.pt` |
| Qwen2.5-VL-3B | "dog" (CW+CE) | Cat_and_Dog | **88.5** | `190326/qwen25_3b_cw_perturbation_best.pt` |
| Qwen3-VL-8B | "dog" (CW+CE) | Cat_and_Dog | **98.5** | `190326/qwen3vl_8b_universal_dog_perturbation_best.pt` |
| Qwen3-VL-8B | "TV" (CW-only) | COCO | **69.5** | `190326/COCO_UAP/checkpoints/coco_tv_pert_best.pt` |
| Qwen3-VL-2B+4B+8B (ensemble) | "TV" | COCO | 50.0 (8B) | `260326/checkpoints/coco_tv_ensemble_best.pt` |
| Qwen3-VL-8B | "stop" (agentic) | VWA | **100.0** | `260326/checkpoints/attack3_stop_best.pt` |

### 4.2 Transfer Attack ASR (pert_3b = Qwen2.5-VL-3B CW, pert_8b = Qwen3-VL-8B CW)

| Source pert | Target model | Total imgs | Hits | ASR (%) | Is surrogate |
|---|---|---|---|---|---|
| pert_3b | Qwen2.5-VL-3B | 200 | 177 | **88.5** | Yes |
| pert_3b | Qwen2.5-VL-7B | 200 | 83 | **41.5** | No |
| pert_3b | Qwen2.5-VL-32B | 200 | 65 | 32.5 | No |
| pert_3b | Qwen3-VL-2B | 200 | 9 | 4.5 | No |
| pert_3b | Qwen3-VL-4B | 200 | 12 | 6.0 | No |
| pert_3b | Qwen3-VL-8B | 200 | 15 | 7.5 | No |
| pert_8b | Qwen2.5-VL-3B | 200 | 44 | 22.0 | No |
| pert_8b | Qwen2.5-VL-7B | 200 | 58 | 29.0 | No |
| pert_8b | Qwen2.5-VL-32B | 200 | 15 | 7.5 | No |
| pert_8b | Qwen3-VL-2B | 200 | 107 | **53.5** | No |
| pert_8b | Qwen3-VL-4B | 200 | 97 | 48.5 | No |
| pert_8b | Qwen3-VL-8B | 200 | 197 | **98.5** | Yes |

### 4.3 COCO TV Transfer: Single Surrogate vs. Ensemble

| Method | Qwen3-VL-2B | Qwen3-VL-4B | Qwen3-VL-8B | Qwen3-VL-32B |
|---|---|---|---|---|
| Single (8B surrogate) | 0.5% | 1.0% | 69.5% | 3.5% |
| Ensemble (2B+4B+8B) | ? | ? | 50.0% | **15.5%** |
| Improvement (32B) | — | — | −19.5 pp | **+12.0 pp (4.4×)** |

### 4.4 Ensemble Aggregation Ablation (32B transfer target)

| Strategy | 8B ASR (%) | 32B ASR (%) | Notes |
|---|---|---|---|
| equal_avg | ~50 | **15.5** | Best generalisation |
| weighted (0.2/0.2/0.6) | ? | ~3 | 8B dominates; no diversity |
| max_loss | 54 | ~4 | Strong single-surrogate, poor transfer |
| grad_alignment | 0 | 0 | Mask kills too many updates |

### 4.5 Black-Box API Transfer (pert_3b, 100 cat images)

| Target API | ASR (%) |
|---|---|
| Anthropic Claude | 7.5 |
| OpenAI GPT-4o | 5.4 |
| Google Gemini | 7.5 |

### 4.6 Attack 3 Training Log (Key Epochs)

| Epoch | CW Loss | LR | Elapsed (min) | Live ASR (%) |
|---|---|---|---|---|
| 1 | 1.7002 | 0.00998 | 11.3 | — |
| 2 | 0.8712 | 0.00990 | 22.5 | 94.0 (new best) |
| 6 | 0.0984 | 0.00914 | 70.6 | **100.0 (new best)** |
| 18 | — | — | 219.8 | early stop |

### 4.7 White-Box Single-Image Attacks

| Attack | Success metric | Result |
|---|---|---|
| LM-loss PGD Stage 1 (cat image) | Loss reduction | 47% (semantic disruption) |
| LM-loss PGD Stage 2 | Loss reduction | 57% total (semantic collapse, multilingual gibberish) |
| Single-token "dog" attack | CW loss | 0.0000 by step 50 |
| Local patch "giraffe" attack | Output caption | Contains "giraffe" + hallucinated scene |
| CAPTCHA numeric (first digit) | 1-char accuracy | 100% success |
| CAPTCHA alphanumeric (Kaggle) | 1-char accuracy | 100% success |
| Amazon product redirect | BBox label flip | 100% success |

---

## 5. Models Used

All models are loaded from HuggingFace (cached locally). None are fine-tuned — all used frozen for eval or as white-box surrogates via gradient backprop.

| Model ID | Family | Params | Vision Encoder | Patch Size | Used In |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-VL-3B-Instruct` | Qwen2.5-VL | 3B | ViT (custom) | 14 | Exp 1–6, 10–11 |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Qwen2.5-VL | 7B | ViT (custom) | 14 | Exp 10 (transfer) |
| `Qwen/Qwen2.5-VL-32B-Instruct` | Qwen2.5-VL | 32B | ViT (custom) | 14 | Exp 10 (transfer) |
| `Qwen/Qwen2.5-VL-72B-Instruct` | Qwen2.5-VL | 72B | ViT (custom) | 14 | Exp 10 (transfer, rarely run) |
| `Qwen/Qwen3-VL-2B-Instruct` | Qwen3-VL | 2B | SigLIP-2 | 16 | Exp 12–13 (surrogate), Exp 10 (transfer) |
| `Qwen/Qwen3-VL-4B-Instruct` | Qwen3-VL | 4B | SigLIP-2 | 16 | Exp 12–13 (surrogate), Exp 10 (transfer) |
| `Qwen/Qwen3-VL-8B-Instruct` | Qwen3-VL | 8B | SigLIP-2 | 16 | Exp 7, 9, 12–15 (surrogate) |
| `Qwen/Qwen3-VL-32B-Instruct` | Qwen3-VL | 32B | SigLIP-2 | 16 | Exp 10, 12, 13 (transfer target) |
| CLIP ViT-B/32 | CLIP | — | ViT-B/32 | — | Exp 1 (failed) |
| CLIP ViT-L/14 | CLIP | — | ViT-L/14 | — | Exp 1 (failed) |
| Anthropic Claude (API) | Closed-source | — | — | — | Exp 11 (black-box transfer) |
| OpenAI GPT-4o (API) | Closed-source | — | — | — | Exp 11 (black-box transfer) |
| Google Gemini (API) | Closed-source | — | — | — | Exp 11 (black-box transfer) |

**Critical architectural difference**: Qwen2.5-VL uses a ViT with patch_size=14 (output: (1024, 1176) patches), while Qwen3-VL uses SigLIP-2 with patch_size=16 (output: (784, 1536) patches). This mismatch creates a hard transfer wall between the two families.

---

## 6. Datasets Used

### Bingsu/Cat_and_Dog (HuggingFace)
- **Source**: `Bingsu/Cat_and_Dog` on HuggingFace Datasets
- **Size**: ~4,200 cat images (label=0), ~4,000+ dog images (label=1)
- **Resolution**: 448×448
- **Used for**: UAP training (cat images as input, "dog" as target output); val split for ASR evaluation (200 images)
- **Used in**: Exp 5, 6, 7, 10, 11

### detection-datasets/coco (HuggingFace)
- **Source**: `detection-datasets/coco` on HuggingFace Datasets
- **Split used**: train (118K images)
- **Category filters applied**:
  - Hair drier: cat_id=89 (excluded from training set; used as ablation target)
  - TV: cat_id=72 (excluded from training set; used as attack target)
- **Used for**: Data-efficiency ablation (Exp 8), TV category UAP (Exp 9), ensemble training (Exp 12–13)
- **Subset sizes**: 50, 200, 500, 1000, 5000 train images depending on experiment

### VisualWebArena (VWA) Screenshots
- **Source**: Extracted from GPT-4V+SoM HTML trajectory files (VisualWebArena benchmark)
- **Size**: 1,913 PNG screenshots
- **Environments**: Classifieds, shopping, Reddit
- **Task pool**: 14 task types (find bicycle, post comment, search jobs, buy product, etc.)
- **Split**: 800 train / 150 val / 50 ASR check
- **Used for**: Attack 3 training (Exp 14) and demos (Exp 15)
- **Extraction script**: `260326/scripts/extract_screenshots.py`

### CAPTCHA Datasets
- **Numeric CAPTCHA**: Generated with `multicolorcaptcha` Python library (custom)
- **Kaggle CAPTCHA Version 2**: Fournierp dataset, 600 alphanumeric samples
- **Used for**: Exp 3 (CAPTCHA bypass)

---

## 7. Scripts Reference

### `190326/attack.py` — Baseline CE-Only UAP Trainer

**Purpose**: First attempt at universal adversarial perturbation; uses Adam optimiser with CrossEntropy loss only. Serves as baseline and diagnostic for why CE-only fails.

**Usage**:
```bash
python 190326/attack.py
```
(No command-line arguments; all config is hardcoded.)

**Key hyperparameters (hardcoded)**:
- LR = 0.003
- ε = 16/255
- No momentum
- Grad accumulation = 4
- Loss = CrossEntropy(logits_seq, "A dog.")
- Dataset = Bingsu/Cat_and_Dog (cat images only)

**Logic**:
1. Loads Qwen2.5-VL-3B-Instruct (frozen, bfloat16)
2. Initialises perturbation as zero tensor, shape matching image patch space
3. For each cat image, calls `pil_to_patches()` to convert PIL → differentiable patch tensor
4. Forward pass through Qwen LM head; CE loss on full "A dog." sequence
5. Backward through perturbation; Adam step; clamp to [-ε, ε]
6. Every 2 epochs: compute ASR on 50-image subset

**Inputs**: Bingsu/Cat_and_Dog cat images
**Outputs**: `qwen_universal_dog_perturbation_best.pt`

**Key finding**: Achieves training loss 0.075 but only 2% generalisation — CE loss drives loss down without reliably flipping greedy-decode first token.

---

### `190326/qwen25_3b_cw.py` — CW+CE UAP for Qwen2.5-VL-3B

**Purpose**: Production UAP trainer for Qwen2.5-VL-3B using Carlini-Wagner margin loss plus auxiliary CE loss. Achieves 88.5% ASR.

**Usage**:
```bash
python 190326/qwen25_3b_cw.py
```
(No command-line arguments; all config hardcoded.)

**Key hyperparameters (hardcoded)**:
- LR = 0.003
- ε = 16/255
- CW_MARGIN = 5.0
- CE_WEIGHT = 0.2 (CW_WEIGHT = 0.8)
- Grad accumulation = 4 steps
- Patience = 6 (eval checks)
- Live ASR check every 2 epochs on 50-image subset
- Epoch checkpoint saved if CW loss < 0.05 after epoch 5

**Suppressed tokens** (all cat variants): "cat", "Cat", "CAT", "feline", "kitty", "kitten"

**Logic**:
1. Loads Qwen2.5-VL-3B (frozen, bfloat16)
2. Initialises zero perturbation
3. For each image: `pil_to_patches()` → forward pass → CW loss = max(0, max_cat_logit − dog_logit + MARGIN)
4. Combined loss = 0.8×CW + 0.2×CE, divided by grad_accum_steps
5. Cosine LR decay; Adam; ε-clamp
6. Saves 10 epoch snapshots + best checkpoint

**Inputs**: Bingsu/Cat_and_Dog cat images
**Outputs**: `qwen25_3b_cw_perturbation_best.pt`, epoch snapshots `_epoch{7,9,...,25}.pt`

---

### `190326/attack_qwen3vl_8b.py` — CW+CE UAP for Qwen3-VL-8B

**Purpose**: UAP trainer for Qwen3-VL-8B. Functionally identical to `qwen25_3b_cw.py` but adapted for Qwen3-VL architecture differences.

**Usage**:
```bash
python 190326/attack_qwen3vl_8b.py
```

**Key differences from `qwen25_3b_cw.py`**:
- `patch_size = 16` (vs 14) → output tensor shape (784, 1536) vs (1024, 1176)
- `token_type_ids` defensively popped from processor output (Qwen3 doesn't return them; crash if passed)
- Otherwise: LR=0.003, ε=16/255, CW_MARGIN=5.0, same cat token suppression list

**Inputs**: Bingsu/Cat_and_Dog cat images
**Outputs**: `qwen3vl_8b_universal_dog_perturbation_best.pt`

---

### `190326/eval_final.py` — Multi-Model Transfer Evaluation

**Purpose**: Evaluate two perturbations (pert_3b, pert_8b) across all 8 Qwen family models. Generates colour-coded result images and CSV.

**Usage**:
```bash
python 190326/eval_final.py
```

**Key arguments** (hardcoded):
- Models: Qwen2.5-VL (3B, 7B, 32B, 72B) + Qwen3-VL (2B, 4B, 8B, 32B)
- Perturbation files: `qwen25_3b_cw_perturbation_best.pt`, `qwen3vl_8b_universal_dog_perturbation_best.pt`
- Val set: 200 cat images

**Logic**:
1. For each (model, perturbation) pair:
   a. Load model (bfloat16, auto device map)
   b. For each image: inject perturbation via `pil_to_patches()` BEFORE normalisation (critical — naive PIL-bake fails)
   c. Run `model.generate()` (greedy, max_new_tokens=10)
   d. Check if "dog"/"Dog" in output → hit
   e. Save result image with green (hit) or red (miss) bar
2. Write `eval_results/results.csv` with columns: model, perturbation, total, hits, misses, asr_pct, is_surrogate

**Inputs**: 200 cat images, 2 `.pt` checkpoint files
**Outputs**: `eval_results/results.csv`, result image folders, `successes/`+`failures/` subfolders

---

### `190326/test.py` — Black-Box API Transfer Evaluation

**Purpose**: Test if pert_3b fools closed-source API models (Claude, GPT-4o, Gemini).

**Usage**:
```bash
python 190326/test.py
```

**Logic**:
1. Load Qwen2.5-VL-3B and 7B (local)
2. For 100 cat images: compute perturbed PIL = `TF.to_tensor() + pert`
3. When local Qwen models succeed: forward perturbed image to all three API models
4. Record whether API output contains "dog"; save confidence scores in result image bar

**Inputs**: 100 cat images, `qwen25_3b_cw_perturbation_best.pt`
**Outputs**: Result images with confidence scores; cross-model statistics

---

### `190326/test_both.py` — Black-Box API Transfer (Variant 2)

**Purpose**: Second variant of `test.py` with minor differences. Uses 100 images; same three APIs.

**Usage**:
```bash
python 190326/test_both.py
```

---

### `190326/collate_results.py` — Result Image Aggregation

**Purpose**: Aggregate eval result images into summary CSV and human-readable pivot table.

**Usage**:
```bash
python 190326/collate_results.py
```

**Logic**:
1. Scan result image directories
2. Count filenames containing "_DOG" → success; else → failure
3. Copy failed images to `failures/{model}/{pert}/`
4. Write CSV + pivot table (models × perturbations)

**Inputs**: Result image directories
**Outputs**: Aggregated CSV, failures folder, pivot table

---

### `190326/COCO_UAP/train_coco_uap_all.py` — COCO Data-Efficiency Ablation

**Purpose**: Sequentially run 4 independent UAP training experiments with n_train ∈ {50, 200, 1000, 5000} images from COCO, targeting "A hair drier." output.

**Usage**:
```bash
python 190326/COCO_UAP/train_coco_uap_all.py
```

**Key hyperparameters**:
- Target model: Qwen3-VL-8B
- Target label: "A hair drier."
- Dataset: COCO train; hair-drier images (cat_id=89) excluded
- LR = 0.003, ε = 16/255, CW+CE loss
- Fresh random seed per run

**Logic**: For each n in [50, 200, 1000, 5000]: load COCO, sample n images, train UAP from scratch, save `best_n{n}.pt`; write `ablation_summary.csv` only after all 4 complete (so n=5000 being incomplete means CSV may not exist).

**Inputs**: COCO train split
**Outputs**: `checkpoints/coco_hairdrier_pert_final_n{50,200,1000,5000}.pt`, `ablation_summary.csv` (if complete)

---

### `190326/COCO_UAP/train_coco_tv.py` — COCO TV Category UAP

**Purpose**: Train CW-only UAP on COCO general images targeting "TV" output on Qwen3-VL-8B. Generates baseline for ensemble comparison.

**Usage**:
```bash
python 190326/COCO_UAP/train_coco_tv.py
```

**Key hyperparameters**:
- LR = 0.01 (faster)
- ε = 64/255 (stronger)
- CW_MARGIN = 2.0
- CE_WEIGHT = 0.0 (CW-only)
- Target token: "TV" (id=15653)
- 500 train / 200 val / 50 ASR check

**Inputs**: COCO train (TV category excluded)
**Outputs**: `checkpoints/coco_tv_pert_best.pt`

---

### `260326/scripts/train_attack3_stop.py` — Attack 3: Agentic Task Hallucination

**Purpose**: Train a universal adversarial perturbation that causes a Qwen3-VL-8B web agent to output the `stop` action (task completion) regardless of actual page state. Achieves 100% ASR.

**Usage**:
```bash
python 260326/scripts/train_attack3_stop.py
```

**Key hyperparameters (hardcoded)**:
- Surrogate: Qwen3-VL-8B (as web agent)
- Target token: "stop" (token id=9495)
- Target sequence: "stop [Task completed successfully]"
- ε = 64/255
- LR = 0.01
- CW loss at logit position -9 (teacher-forced sequence length)
- Dataset: 1,913 VWA screenshots; 800 train / 150 val / 50 ASR check
- 14-task pool with web-agent system prompt

**Logic**:
1. Load VWA screenshots and task prompts
2. For each training step: randomly sample (screenshot, task) pair
3. Apply perturbation via `pil_to_patches()` → forward pass
4. CW loss on first token of response (target: "stop" token)
5. Backprop; Adam; ε-clamp
6. Every 2 epochs: 50-image live ASR check; save if improved

**Inputs**: 1,913 VWA screenshots, task pool
**Outputs**: `260326/checkpoints/attack3_stop_best.pt`, `attack3_stop_final.pt`, `260326/logs/attack3_stop.log`

---

### `260326/scripts/train_coco_tv_ensemble.py` — Ensemble UAP Training

**Purpose**: Train a single UAP perturbation across three Qwen3-VL surrogates (2B, 4B, 8B) simultaneously using equal-weight CW loss averaging. Achieves 4.4× transfer improvement to 32B.

**Usage**:
```bash
python 260326/scripts/train_coco_tv_ensemble.py
```

**Key hyperparameters**:
- Surrogates: Qwen3-VL-2B, 4B, 8B (all frozen; peak VRAM 28.6 GB / 121.7 GB)
- Loss = (CW_2B + CW_4B + CW_8B) / 3
- LR = 0.01, ε = 64/255, CW_MARGIN = 2.0
- 500 train / 200 val / 50 ASR check; training time: 686 minutes (11.4 hours)

**Inputs**: COCO train (TV excluded)
**Outputs**: `260326/checkpoints/coco_tv_ensemble_best.pt` (epoch 12), `coco_tv_ensemble_final.pt` (epoch 30)

---

### `260326/scripts/eval_coco_tv.py` — Transfer Evaluation

**Purpose**: Evaluate single-surrogate and ensemble checkpoints on all Qwen3-VL model sizes.

**Usage**:
```bash
python 260326/scripts/eval_coco_tv.py \
  --models qwen3-vl-2b qwen3-vl-4b qwen3-vl-8b qwen3-vl-32b \
  --hf_cache /path/to/cache \
  --delete_after_eval \
  --n_eval 200
```

**Arguments**:
- `--models`: Space-separated list of model IDs to evaluate
- `--hf_cache`: Path to HuggingFace model cache directory
- `--delete_after_eval`: Delete model from VRAM after eval (saves memory)
- `--n_eval`: Number of evaluation images (default 200)

**Inputs**: COCO val images, `.pt` checkpoint files
**Outputs**: `260326/results/results.csv`

---

### `260326/scripts/demo_attack3.py` — Single-Step Agentic Attack Demo

**Purpose**: Load Attack 3 perturbation and run clean vs. attacked comparison on 5 held-out VWA screenshots.

**Usage**:
```bash
python 260326/scripts/demo_attack3.py
```

**Inputs**: 5 held-out VWA screenshots, `260326/checkpoints/attack3_stop_best.pt`
**Outputs**: Comparison images (clean action vs attacked "stop" output); all 5 succeed

---

### `260326/scripts/demo_trajectory.py` — Multi-Step Trajectory Demo

**Purpose**: Simulate 3 web-agent tasks (buy coffee maker, post comment, find software engineer job) over 5 steps each. Perturbation injected starting at step 3.

**Usage**:
```bash
python 260326/scripts/demo_trajectory.py
```

**Inputs**: VWA screenshots for 3 task sequences
**Outputs**: `260326/results/trajectory_demo.html` — self-contained HTML with base64-embedded images; shows step-by-step clean vs. attacked comparison

---

### `260326/scripts/visualise_attack3.py` — Perturbation Visualisation

**Purpose**: Visualise Attack 3 perturbation at ×1, ×10, ×50 magnification; prove imperceptibility.

**Usage**:
```bash
python 260326/scripts/visualise_attack3.py
```

**Inputs**: `attack3_stop_best.pt`, 5 sample VWA screenshots
**Outputs**: `260326/results/attack3_visualisation.html`; L-inf = 0.2520 reported

---

### `260326/scripts/train_ensemble_ablation.py` — Aggregation Strategy Ablation

**Purpose**: Compare 4 gradient aggregation strategies for ensemble UAP training; identifies equal_avg as optimal.

**Usage**:
```bash
python 260326/scripts/train_ensemble_ablation.py
```

**Strategies**: `equal_avg`, `weighted`, `max_loss`, `grad_alignment`
**Dataset**: 200 train / 100 val (smaller than full run)

---

### `260326/scripts/extract_screenshots.py` — VWA Screenshot Extraction

**Purpose**: Parse VisualWebArena GPT-4V+SoM HTML trajectory files and extract PNG screenshots for Attack 3 training.

**Usage**:
```bash
python 260326/scripts/extract_screenshots.py --input_dir /path/to/vwa_html/ --output_dir /path/to/screenshots/
```

**Inputs**: HTML trajectory files from VWA GPT-4V evaluation run
**Outputs**: 1,913 PNG screenshots

---

### `260326/scripts/quick_test.py` — Smoke Test

**Purpose**: Minimal smoke test to verify attack3_stop_best.pt loads correctly and produces "stop" output on 3 sample images.

**Usage**:
```bash
python 260326/scripts/quick_test.py
```

---

### `260326/scripts/train_coco_tv_original.py` and `eval_coco_tv_original.py` — Original TV Variant

**Purpose**: Alternate hyperparameter variant of COCO TV training/eval. Exact differences from primary scripts not documented in logs; likely an earlier iteration before final hyperparameter tuning.

---

## 8. Checkpoints Index

All checkpoints are PyTorch tensors stored with `torch.save()`, bfloat16 dtype on disk, approximately 1.1 MB each. All represent the trained perturbation tensor δ (same shape as the model's internal image patch representation after `pil_to_patches()`).

| File | Size | Producing Script | Description |
|---|---|---|---|
| `190326/qwen_universal_dog_perturbation_best.pt` | 1.1 MB | `attack.py` | CE-only UAP for Qwen2.5-VL-3B; 2% transfer baseline |
| `190326/qwen25_3b_cw_perturbation_best.pt` | 1.1 MB | `qwen25_3b_cw.py` | CW+CE UAP for Qwen2.5-VL-3B; 88.5% surrogate ASR |
| `190326/qwen3vl_8b_universal_dog_perturbation_best.pt` | 1.1 MB | `attack_qwen3vl_8b.py` | CW+CE UAP for Qwen3-VL-8B; 98.5% surrogate ASR |
| `190326/COCO_UAP/checkpoints/coco_hairdrier_pert_final_n50.pt` | 1.1 MB | `train_coco_uap_all.py` | COCO hair-drier UAP trained on 50 images |
| `190326/COCO_UAP/checkpoints/coco_hairdrier_pert_final_n200.pt` | 1.1 MB | `train_coco_uap_all.py` | COCO hair-drier UAP trained on 200 images |
| `190326/COCO_UAP/checkpoints/coco_hairdrier_pert_final_n1000.pt` | 1.1 MB | `train_coco_uap_all.py` | COCO hair-drier UAP trained on 1000 images |
| `190326/COCO_UAP/checkpoints/coco_tv_pert_best.pt` | 1.1 MB | `train_coco_tv.py` | COCO TV UAP, single surrogate (8B); 69.5% surrogate ASR; 3.5% on 32B |
| `260326/checkpoints/attack3_stop_best.pt` | 1.1 MB | `train_attack3_stop.py` | Attack 3 best (epoch 6); 100% surrogate ASR; L-inf=0.2520 |
| `260326/checkpoints/attack3_stop_final.pt` | 1.1 MB | `train_attack3_stop.py` | Attack 3 final (epoch 18, early stop) |
| `260326/checkpoints/coco_tv_ensemble_best.pt` | 1.1 MB | `train_coco_tv_ensemble.py` | Ensemble UAP best (epoch 12); 50% on 8B, 15.5% on 32B |
| `260326/checkpoints/coco_tv_ensemble_final.pt` | 1.1 MB | `train_coco_tv_ensemble.py` | Ensemble UAP final (epoch 30) |
| `260326/checkpoints/coco_tv_pert_best.pt` | 1.1 MB | (copy of 190326 TV checkpoint) | Baseline reference copy for 260326 comparisons |

**Note**: Epoch snapshots for `qwen25_3b_cw.py` training (`_epoch7.pt` through `_epoch25.pt`) are stored locally in the compute environment but not committed to the repository. They are listed in `190326/large_files.txt`.

---

## 9. Known Issues and Bugs Fixed

### Bug: Naive PIL-bake perturbation approach (FIXED)
- **Issue**: Initial approach of adding perturbation to PIL image and re-normalising inside the Qwen processor failed — perturbation was absorbed into normalisation and had no effect on patch features.
- **Fix**: Implemented `pil_to_patches()` which injects the perturbation as a differentiable tensor step AFTER PIL→tensor conversion but BEFORE normalisation, exactly replicating Qwen's internal pipeline. Gradients flow to raw pixel values.
- **Location**: `eval_final.py`, `qwen25_3b_cw.py`, `attack_qwen3vl_8b.py`

### Bug: Qwen3-VL token_type_ids crash (FIXED)
- **Issue**: Passing `token_type_ids` to Qwen3-VL-8B forward() caused a crash because Qwen3 doesn't use or return this key.
- **Fix**: Defensive `.pop('token_type_ids', None)` added to processor output before forward().
- **Location**: `attack_qwen3vl_8b.py`

### Bug: Float16 gradient overflow with momentum + large epsilon (DOCUMENTED, WORKAROUND)
- **Issue**: Using momentum (μ=0.9) with ε > ~28/255 causes float16 NaN logits, collapsing training.
- **Workaround**: Limited ε to ≤28/255 when using momentum; used bfloat16 which has larger dynamic range.
- **Location**: Documented in `120326/README.md`

### Issue: CE-only loss insufficient for greedy-decode flipping (RESOLVED BY DESIGN)
- **Issue**: CrossEntropy loss on full sequence reached 0.075 training loss but only 2% greedy-decode ASR — loss can be driven down without the first token being the target.
- **Resolution**: Switched to CW margin loss (logit_dog > max_cat_logit + MARGIN) which directly targets the greedy-decode decision.
- **Location**: `attack.py` (failed), `qwen25_3b_cw.py` (fixed)

### Issue: CLIP transfer attack failed entirely (EXPECTED, RESOLVED BY DESIGN)
- **Issue**: MI-FGSM attack on CLIP ensemble achieved CLIP embedding alignment but did not transfer to Qwen2.5-VL.
- **Root cause**: Qwen uses a different vision encoder; CLIP feature space does not align.
- **Resolution**: Switched to direct white-box LM-loss attacks.

### Issue: n=5000 COCO experiment incomplete as of 2026-03-20 (PENDING)
- **Status**: `train_coco_uap_all.py` runs all 4 experiments sequentially; n=5000 was still running overnight. `ablation_summary.csv` may not exist yet.

### Issue: Ensemble training interrupted twice by system preemption
- **Status**: Workaround: GPU reservation required for 11.4-hour training jobs.
- **Location**: `260326/README.md`

### Issue: Mac demo infeasible (DOCUMENTED)
- **Causes**: Python 3.13 incompatibility with some dependencies; 16 GB RAM OOM; MPS (Metal) segfault.
- **Status**: All experiments run on remote GPU cluster only.

### Issue: 50-image ASR eval set too noisy for 32B transfer estimates
- **Issue**: At 3.5% ASR, a single hit on the 50-image set = 2% variance, making estimates unreliable.
- **Planned fix**: Increase to 100+ image eval set for 32B experiments.

### Issue: Weighted ensemble strategy anomaly
- **Issue**: Weighted strategy showed ~38% on one run and ~3% on another; explained as 50-image eval set sampling noise when true ASR is very low (~3%).

---

## 10. Done vs. In Progress

### Completed

- [x] White-box single-image LM-loss PGD attacks on Qwen2.5-VL-3B (all variants: semantic disruption, single-token, local patch, CAPTCHA, Amazon redirect)
- [x] CLIP ensemble transfer attack (executed and confirmed as failed approach)
- [x] Differentiable `pil_to_patches()` preprocessing pipeline reimplementation
- [x] CW+CE loss formulation design and validation
- [x] Baseline CE-only UAP training (confirmed as insufficient)
- [x] UAP training for Qwen2.5-VL-3B with CW+CE loss (88.5% ASR)
- [x] UAP training for Qwen3-VL-8B with CW+CE loss (98.5% ASR)
- [x] Full transfer evaluation across 8 Qwen models (results.csv authoritative)
- [x] Black-box API transfer evaluation (Claude, GPT-4o, Gemini) — confirmed near-zero
- [x] COCO hair-drier ablation n=50, 200, 1000 (n=5000 incomplete)
- [x] COCO TV category single-surrogate UAP (69.5% ASR on 8B, 3.5% on 32B)
- [x] Ensemble UAP training (2B+4B+8B, equal-weight, 15.5% on 32B)
- [x] Aggregation strategy ablation (equal_avg confirmed as best)
- [x] Attack 3 design and training (100% ASR on VWA, epoch 6)
- [x] Attack 3 single-step demo (5 held-out screenshots, 100% success)
- [x] Attack 3 multi-step trajectory demo (3 tasks × 5 steps, 100% success)
- [x] Attack 3 perturbation imperceptibility visualisation (L-inf=0.2520)
- [x] All result CSVs and HTML visualisations generated
- [x] Weekly README documentation for all three weeks

### In Progress / Incomplete

- [ ] COCO hair-drier n=5000 ablation (was running as of 2026-03-20; status unknown)
- [ ] `ablation_summary.csv` (only written after all 4 COCO runs complete)
- [ ] Attack 3 transfer evaluation on Qwen3-VL-2B and 32B (scripts exist, eval not run)
- [ ] Attack 2 (SoM Set-of-Mark ID confusion attack) — planned, not started
- [ ] Live VisualWebArena Docker environment setup for end-to-end trajectory evaluation
- [ ] Increase 32B eval set to 100+ images for reliable transfer estimates
- [ ] Thesis Chapter 3 writing: transferability analysis
- [ ] Thesis Chapter 4 writing: agentic attacks
- [ ] Mechanistic interpretability analysis (ERQ4–6) — not yet started

### Planned (Not Yet Started)

- [ ] Attack 2: SoM ID confusion (confuse model about which UI element ID to interact with)
- [ ] End-to-end VisualWebArena Docker evaluation pipeline
- [ ] Larger ensemble experiments (include 32B as surrogate if VRAM allows)
- [ ] Composite benchmark construction (RQ1/RQ2 — separate work stream)
- [ ] Mechanistic analysis of attack feature representations (ERQ4–6)

---

*End of THESIS_STATE.md*
