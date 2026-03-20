"""
train_coco_uap_all.py
---------------------
Runs all 4 data efficiency experiments sequentially overnight.
Each experiment trains a UAP with a different number of training images,
saves checkpoints independently, then moves on to the next.

Experiments: num_train = 50, 200, 1000, 5000

Usage (from COCO_UAP/ folder):
  python train_coco_uap_all.py

Outputs:
  checkpoints/coco_hairdrier_pert_best_n50.pt
  checkpoints/coco_hairdrier_pert_best_n200.pt
  checkpoints/coco_hairdrier_pert_best_n1000.pt
  checkpoints/coco_hairdrier_pert_best_n5000.pt
  results/ablation_summary.csv   — ASR per experiment on 50-image val check
"""

import os
import csv
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

# ── Directories ────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent   # COCO_UAP/
CKPT_DIR  = BASE_DIR / "checkpoints"
RESULT_DIR = BASE_DIR / "results"
CKPT_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# ── Experiments to run ─────────────────────────────────────────────────────────
EXPERIMENTS = [50, 200, 1000, 5000]

# ── Shared hyperparameters ─────────────────────────────────────────────────────
MODEL_ID         = "Qwen/Qwen3-VL-8B-Instruct"
H, W             = 448, 448
EPSILON          = 32.0 / 255.0
NUM_EPOCHS       = 30
LR               = 0.003
GRAD_ACCUM_STEPS = 4
CW_MARGIN        = 5.0
CW_WEIGHT        = 0.8
CE_WEIGHT        = 0.2
PATIENCE         = 6
TARGET_CAPTION   = "A hair drier."
QUESTION         = "What object is in this picture? Just answer in one word or phrase."
VAL_SIZE         = 200
ASR_CHECK_SIZE   = 50


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
            out_ids = model.generate(**inputs, max_new_tokens=15, do_sample=False)
            decoded = processor.tokenizer.decode(
                out_ids[0], skip_special_tokens=True
            ).lower()
            if "hair" in decoded and "drier" in decoded:
                hits += 1
    return hits / len(val_subset)


# ── Dataset (loaded once, shared across experiments) ───────────────────────────
def load_coco_filtered():
    print("Loading COCO dataset (this may take a few minutes)...")
    dataset = load_dataset("detection-datasets/coco", split="train")

    def is_not_hair_drier(example):
        if example.get("objects") is None:
            return True
        categories = example["objects"].get("category", [])
        return not any(
            (isinstance(c, str) and "hair" in c.lower()) or c == 89
            for c in categories
        )

    print("Filtering hair drier images...")
    filtered = dataset.filter(is_not_hair_drier, num_proc=4)
    print(f"COCO filtered: {len(filtered)} images available\n")
    return filtered


# ── Single experiment ──────────────────────────────────────────────────────────
def run_experiment(num_train, filtered_dataset, model, processor,
                   mean, std, target_tensor, hair_id, T,
                   device, asr_subset):
    """
    Trains a UAP for one num_train setting.
    Returns best_asr achieved and path to saved checkpoint.
    """
    tag       = f"n{num_train}"
    best_ckpt = CKPT_DIR / f"coco_hairdrier_pert_best_{tag}.pt"
    final_ckpt = CKPT_DIR / f"coco_hairdrier_pert_final_{tag}.pt"

    print(f"\n{'='*65}")
    print(f"  EXPERIMENT: num_train={num_train}  [{tag}]")
    print(f"{'='*65}")

    # Sample training indices fresh for each experiment
    all_indices   = list(range(len(filtered_dataset)))
    random.shuffle(all_indices)
    train_indices = all_indices[:num_train]
    train_set     = filtered_dataset.select(train_indices)

    perturbation = torch.zeros(3, H, W, dtype=torch.float32,
                               device=device, requires_grad=True)
    optimizer    = torch.optim.Adam([perturbation], lr=LR, weight_decay=1e-4)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=0.001
    )
    loss_fn = nn.CrossEntropyLoss()

    best_val_asr     = 0.0
    no_improve_count = 0
    start_time       = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_cw_loss = 0.0
        epoch_ce_loss = 0.0

        idxs           = list(range(len(train_set)))
        random.shuffle(idxs)
        shuffled_train = train_set.select(idxs)

        optimizer.zero_grad()

        for step, data in enumerate(tqdm(shuffled_train,
                                         desc=f"  [{tag}] Epoch {epoch+1}/{NUM_EPOCHS}")):
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

            # CW loss — dynamic suppression of top competing token
            first_token_logits            = outputs.logits[:, -(T + 1), :]
            logit_target                  = first_token_logits[0, hair_id]
            competing_logits              = first_token_logits[0].clone()
            competing_logits[hair_id]     = -1e9
            logit_competitor              = competing_logits.max()
            cw_loss = torch.clamp(logit_competitor - logit_target + CW_MARGIN, min=0.0)

            # CE loss — full sequence "A hair drier."
            logits_seq  = outputs.logits[:, -(T+1):-1, :]
            logits_flat = logits_seq.reshape(-1, logits_seq.size(-1))
            ce_loss     = loss_fn(logits_flat, target_tensor.reshape(-1))

            loss = (CW_WEIGHT * cw_loss + CE_WEIGHT * ce_loss) / GRAD_ACCUM_STEPS
            epoch_cw_loss += cw_loss.item()
            epoch_ce_loss += ce_loss.item()
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
        avg_cw     = epoch_cw_loss / len(train_set)
        avg_ce     = epoch_ce_loss / len(train_set)
        elapsed    = (time.time() - start_time) / 60
        print(f"  [{tag}] Epoch {epoch+1:3d}  "
              f"CW: {avg_cw:.4f}  CE: {avg_ce:.4f}  "
              f"lr: {optimizer.param_groups[0]['lr']:.5f}  "
              f"elapsed: {elapsed:.1f}min")

        # ASR check every 2 epochs
        if (epoch + 1) % 2 == 0:
            asr = compute_asr(model, processor, asr_subset,
                              perturbation, mean, std, device)
            print(f"  [{tag}] -> Live ASR (50 val): {asr*100:.1f}%")

            if asr > best_val_asr + 0.01:
                best_val_asr     = asr
                no_improve_count = 0
                torch.save(perturbation.detach().cpu().to(torch.bfloat16), best_ckpt)
                print(f"  [{tag}] -> New best ({best_val_asr*100:.1f}%)! Saved → {best_ckpt}")
            else:
                no_improve_count += 1
                print(f"  [{tag}] -> No improvement ({no_improve_count}/{PATIENCE})")
                if no_improve_count >= PATIENCE:
                    print(f"  [{tag}] Early stopping at epoch {epoch+1}.")
                    break

    torch.save(perturbation.detach().cpu().to(torch.bfloat16), final_ckpt)
    total_time = (time.time() - start_time) / 60
    print(f"  [{tag}] Done. Best ASR: {best_val_asr*100:.1f}%  "
          f"Total time: {total_time:.1f}min")
    print(f"  [{tag}] Best  → {best_ckpt}")
    print(f"  [{tag}] Final → {final_ckpt}")

    return best_val_asr, str(best_ckpt)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiments to run: {EXPERIMENTS}\n")

    # Load model once — shared across all experiments
    print(f"Loading {MODEL_ID} (loaded once, reused for all experiments)...")
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

    # Target tokens
    target_ids    = processor.tokenizer.encode(TARGET_CAPTION, add_special_tokens=False)
    target_tensor = torch.tensor(target_ids, device=device)
    T             = len(target_ids)
    hair_id       = target_tensor[0]
    print(f"Target: '{TARGET_CAPTION}' -> token IDs {target_ids}  (length {T})")
    print(f"First token: '{processor.tokenizer.decode([hair_id.item()])}' (id {hair_id.item()})\n")

    # Load dataset once
    filtered = load_coco_filtered()

    # Fixed val/ASR subset — same across all experiments for fair comparison
    all_indices  = list(range(len(filtered)))
    random.shuffle(all_indices)
    val_indices  = all_indices[-VAL_SIZE:]          # last 200 as val
    val_set      = filtered.select(val_indices)
    asr_subset   = val_set.select(range(ASR_CHECK_SIZE))
    print(f"Fixed val set: {VAL_SIZE} images | ASR check subset: {ASR_CHECK_SIZE} images\n")

    # Run all experiments
    summary_rows = []
    overall_start = time.time()

    for num_train in EXPERIMENTS:
        best_asr, ckpt_path = run_experiment(
            num_train, filtered, model, processor,
            mean, std, target_tensor, hair_id, T,
            device, asr_subset
        )
        summary_rows.append({
            "num_train":  num_train,
            "best_asr":   round(best_asr * 100, 1),
            "checkpoint": ckpt_path,
        })

        # Clear CUDA cache between experiments
        torch.cuda.empty_cache()

    # Write ablation summary CSV
    total_time   = (time.time() - overall_start) / 60
    csv_path     = RESULT_DIR / "ablation_summary.csv"
    csv_fields   = ["num_train", "best_asr", "checkpoint"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n{'='*65}")
    print(f"ALL EXPERIMENTS COMPLETE  (total: {total_time:.1f}min)")
    print(f"{'='*65}")
    print(f"\n{'num_train':>12}  {'best_asr_pct':>14}")
    print("-" * 30)
    for row in summary_rows:
        print(f"{row['num_train']:>12}  {row['best_asr']:>13.1f}%")
    print(f"\nAblation summary → {csv_path}")
    print(f"\nNext step — eval transfer across all models:")
    print(f"  python eval_coco.py --pert checkpoints/coco_hairdrier_pert_best_n1000.pt "
          f"--models 3b 7b 2b 4b 8b --delete_after_eval")


if __name__ == "__main__":
    main()