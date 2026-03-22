"""
train_coco_tv.py
----------------
Trains a UAP on COCO images to make Qwen3-VL-8B mislabel
ANY object as "TV".

Changes from v1:
  - LR 0.003 → 0.01 (faster convergence)
  - EPSILON 32/255 → 64/255 (stronger signal)
  - CW_MARGIN 5.0 → 2.0 (easier target, TV is semantically distant)
  - CE_WEIGHT 0.0 (CW only — CE was fighting the signal)

Verified:
  - "TV" is a single token (id=15653) in Qwen3-VL-8B tokenizer
  - cat_id=72 = tv in detection-datasets/coco (confirmed visually)

Usage (from COCO_UAP/ folder):
  python train_coco_tv.py

Outputs:
  checkpoints/coco_tv_pert_best.pt
  checkpoints/coco_tv_pert_final.pt
"""

import os
import random
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import time

login(token=os.environ["HF_TOKEN"])

# ── Dirs ───────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID         = "Qwen/Qwen3-VL-8B-Instruct"
H, W             = 448, 448
EPSILON          = 64.0 / 255.0   # increased from 32 — stronger perturbation signal
LR               = 0.01           # increased from 0.003 — faster convergence
NUM_EPOCHS       = 30
GRAD_ACCUM_STEPS = 4
CW_MARGIN        = 2.0            # lowered from 5.0 — TV is semantically distant
CW_WEIGHT        = 1.0            # CW only
CE_WEIGHT        = 0.0            # dropped — CE was fighting CW for distant target
PATIENCE         = 6
NUM_TRAIN        = 1000
VAL_SIZE         = 200
ASR_CHECK_SIZE   = 50

TARGET_CAPTION   = "TV."
TARGET_WORD      = "tv"
TV_CAT_ID        = 72             # confirmed visually
QUESTION         = "What object is in this picture? Answer in one word."


# ── pil_to_patches ─────────────────────────────────────────────────────────────
def pil_to_patches(pil_image, perturbation, mean, std,
                   patch_size=16, temporal_patch_size=2, device="cuda"):
    img_tensor = TF.to_tensor(pil_image).to(device=device, dtype=torch.bfloat16)
    perturbed  = (img_tensor + perturbation).clamp(0.0, 1.0)
    normalized = (perturbed - mean) / std
    img        = normalized.unsqueeze(0).repeat(temporal_patch_size, 1, 1, 1)
    t, c, h, w = img.shape
    ph, pw     = h // patch_size, w // patch_size
    img        = img.reshape(t, c, ph, patch_size, pw, patch_size)
    img        = img.permute(2, 4, 0, 1, 3, 5).contiguous()
    img        = img.reshape(ph * pw, t * c * patch_size * patch_size)
    return img


# ── ASR check ──────────────────────────────────────────────────────────────────
def compute_asr(model, processor, val_subset, perturbation, mean, std, device):
    hits = 0
    with torch.no_grad():
        for data in val_subset:
            raw_image = data['image'].convert("RGB").resize((H, W))
            messages  = [{"role": "user", "content": [
                {"type": "image", "image": raw_image},
                {"type": "text",  "text": QUESTION},
            ]}]
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text_prompt], images=[raw_image],
                padding=True, return_tensors="pt",
            ).to(device)
            inputs.pop("token_type_ids", None)
            inputs["pixel_values"] = pil_to_patches(
                raw_image, perturbation.to(torch.bfloat16), mean, std,
                patch_size=16, temporal_patch_size=2, device=device
            )
            out_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            decoded = processor.tokenizer.decode(
                out_ids[0], skip_special_tokens=True
            ).lower()
            if TARGET_WORD in decoded:
                hits += 1
    return hits / len(val_subset)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"Loading {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model     = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    img_proc = processor.image_processor
    mean = torch.tensor(img_proc.image_mean, dtype=torch.bfloat16, device=device).view(3,1,1)
    std  = torch.tensor(img_proc.image_std,  dtype=torch.bfloat16, device=device).view(3,1,1)

    # ── Token setup ────────────────────────────────────────────────────────────
    target_ids    = processor.tokenizer.encode(TARGET_CAPTION, add_special_tokens=False)
    target_tensor = torch.tensor(target_ids, device=device)
    T             = len(target_ids)
    tv_id         = target_tensor[0]

    print(f"\nTarget caption : '{TARGET_CAPTION}'")
    print(f"Token IDs      : {target_ids}  (length {T})")
    for i, tid in enumerate(target_ids):
        print(f"  [{i}] id={tid}  -> '{processor.tokenizer.decode([tid])}'")
    print(f"CW loss target : id={tv_id.item()} "
          f"-> '{processor.tokenizer.decode([tv_id.item()])}'\n")
    print(f"Hyperparameters: LR={LR}  EPSILON={EPSILON*255:.0f}/255  "
          f"CW_MARGIN={CW_MARGIN}  CW only (CE dropped)\n")

    # ── Dataset ────────────────────────────────────────────────────────────────
    print("Loading COCO train split...")
    dataset = load_dataset("detection-datasets/coco", split="train")

    def is_not_tv(example):
        if example.get("objects") is None:
            return True
        return TV_CAT_ID not in example["objects"].get("category", [])

    print("Filtering TV images from training set...")
    filtered = dataset.filter(is_not_tv, num_proc=4)
    print(f"COCO after filter: {len(filtered)} images available")

    all_indices   = list(range(len(filtered)))
    random.shuffle(all_indices)
    train_indices = all_indices[:NUM_TRAIN]
    val_indices   = all_indices[NUM_TRAIN:NUM_TRAIN + VAL_SIZE]

    train_set  = filtered.select(train_indices)
    val_set    = filtered.select(val_indices)
    asr_subset = val_set.select(range(ASR_CHECK_SIZE))
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | ASR check: {len(asr_subset)}\n")

    # ── Perturbation + optimiser ───────────────────────────────────────────────
    perturbation = torch.zeros(3, H, W, dtype=torch.float32,
                               device=device, requires_grad=True)
    optimizer    = torch.optim.Adam([perturbation], lr=LR, weight_decay=1e-4)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.1
    )
    best_val_asr     = 0.0
    no_improve_count = 0
    start_time       = time.time()

    print("Starting optimisation...\n")

    for epoch in range(NUM_EPOCHS):
        epoch_cw_loss = 0.0

        idxs           = list(range(len(train_set)))
        random.shuffle(idxs)
        shuffled_train = train_set.select(idxs)
        optimizer.zero_grad()

        for step, data in enumerate(tqdm(shuffled_train,
                                         desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            raw_image = data['image'].convert("RGB").resize((H, W))

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": raw_image},
                    {"type": "text",  "text": QUESTION},
                ]},
                {"role": "assistant", "content": TARGET_CAPTION},
            ]
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            inputs = processor(
                text=[text_prompt], images=[raw_image],
                padding=True, return_tensors="pt",
            ).to(device)
            inputs.pop("token_type_ids", None)
            inputs["pixel_values"] = pil_to_patches(
                raw_image, perturbation.to(torch.bfloat16), mean, std,
                patch_size=16, temporal_patch_size=2, device=device
            )

            outputs = model(**inputs)

            # ── CW loss only ───────────────────────────────────────────────────
            first_token_logits   = outputs.logits[:, -(T + 1), :]
            logit_target         = first_token_logits[0, tv_id]
            competing            = first_token_logits[0].clone()
            competing[tv_id]     = -1e9
            logit_competitor     = competing.max()
            cw_loss = torch.clamp(logit_competitor - logit_target + CW_MARGIN, min=0.0)

            loss = cw_loss / GRAD_ACCUM_STEPS
            epoch_cw_loss += cw_loss.item()
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    perturbation.clamp_(-EPSILON, EPSILON)

        if len(train_set) % GRAD_ACCUM_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                perturbation.clamp_(-EPSILON, EPSILON)

        scheduler.step()
        elapsed = (time.time() - start_time) / 60
        print(f"Epoch {epoch+1:3d}  "
              f"CW: {epoch_cw_loss/len(train_set):.4f}  "
              f"lr: {optimizer.param_groups[0]['lr']:.5f}  "
              f"elapsed: {elapsed:.1f}min")

        # ── ASR check every 2 epochs ───────────────────────────────────────────
        if (epoch + 1) % 2 == 0:
            asr = compute_asr(model, processor, asr_subset,
                              perturbation, mean, std, device)
            print(f"  -> Live ASR (50 val): {asr*100:.1f}%")

            if asr > best_val_asr + 0.01:
                best_val_asr     = asr
                no_improve_count = 0
                ckpt = CKPT_DIR / "coco_tv_pert_best.pt"
                torch.save(perturbation.detach().cpu().to(torch.bfloat16), ckpt)
                print(f"  -> New best ({best_val_asr*100:.1f}%)! Saved → {ckpt}")
            else:
                no_improve_count += 1
                print(f"  -> No improvement ({no_improve_count}/{PATIENCE})")
                if no_improve_count >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}.")
                    break

    final_ckpt = CKPT_DIR / "coco_tv_pert_final.pt"
    torch.save(perturbation.detach().cpu().to(torch.bfloat16), final_ckpt)
    total_time = (time.time() - start_time) / 60
    print(f"\nDone. Best ASR: {best_val_asr*100:.1f}%  Total time: {total_time:.1f}min")
    print(f"Best  → {CKPT_DIR / 'coco_tv_pert_best.pt'}")
    print(f"Final → {final_ckpt}")


if __name__ == "__main__":
    main()