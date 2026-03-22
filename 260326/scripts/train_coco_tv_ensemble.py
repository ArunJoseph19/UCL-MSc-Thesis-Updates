"""
train_coco_tv_ensemble.py
-------------------------
Contribution 1: Ensemble UAP training across Qwen3-VL 2B + 4B + 8B.
Goal: lift 32B transfer ASR from 3.5% to >20%.

Usage:
    python scripts/train_coco_tv_ensemble.py
"""

import os, random, time, torch
import torchvision.transforms.functional as TF
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

login(token=os.environ["HF_TOKEN"])

BASE_DIR = Path(__file__).parent.parent
CKPT_DIR = BASE_DIR / "checkpoints"
LOG_DIR  = BASE_DIR / "logs"
CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

SURROGATE_IDS = {
    "2b": "Qwen/Qwen3-VL-2B-Instruct",
    "4b": "Qwen/Qwen3-VL-4B-Instruct",
    "8b": "Qwen/Qwen3-VL-8B-Instruct",
}
H, W             = 448, 448
EPSILON          = 64.0 / 255.0
LR               = 0.01
NUM_EPOCHS       = 30
GRAD_ACCUM_STEPS = 4
CW_MARGIN        = 2.0
PATIENCE         = 6
NUM_TRAIN        = 500
VAL_SIZE         = 200
ASR_CHECK_SIZE   = 50
TV_CAT_ID        = 72
TARGET_CAPTION   = "TV."
TARGET_WORD      = "tv"
QUESTION         = "What object is in this picture? Answer in one word."


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load all three surrogates
    models, processors, means, stds, tv_ids = {}, {}, {}, {}, {}

    for key, mid in SURROGATE_IDS.items():
        print(f"Loading {mid}...")
        processors[key] = AutoProcessor.from_pretrained(mid)
        models[key]     = Qwen3VLForConditionalGeneration.from_pretrained(
            mid, torch_dtype=torch.bfloat16
        ).to(device).eval()
        for p in models[key].parameters():
            p.requires_grad = False

        img_proc   = processors[key].image_processor
        means[key] = torch.tensor(img_proc.image_mean, dtype=torch.bfloat16, device=device).view(3,1,1)
        stds[key]  = torch.tensor(img_proc.image_std,  dtype=torch.bfloat16, device=device).view(3,1,1)

        tv_id_list  = processors[key].tokenizer.encode(TARGET_WORD, add_special_tokens=False)
        tv_ids[key] = torch.tensor(tv_id_list[0], device=device)
        print(f"  '{TARGET_WORD}' token id = {tv_ids[key].item()} "
              f"→ '{processors[key].tokenizer.decode([tv_ids[key].item()])}'")

    # Check total VRAM used
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nGPU memory after loading models: {used:.1f}GB / {total:.1f}GB\n")

    # COCO dataset
    print("Loading COCO train split...")
    dataset  = load_dataset("detection-datasets/coco", split="train")
    filtered = dataset.filter(
        lambda ex: TV_CAT_ID not in (ex.get("objects") or {}).get("category", []),
        num_proc=4
    )
    print(f"COCO after filter: {len(filtered)} images")

    idx       = list(range(len(filtered)))
    random.shuffle(idx)
    train_set = filtered.select(idx[:NUM_TRAIN])
    val_set   = filtered.select(idx[NUM_TRAIN:NUM_TRAIN + VAL_SIZE])
    asr_sub   = val_set.select(range(ASR_CHECK_SIZE))
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | ASR: {len(asr_sub)}\n")

    perturbation = torch.zeros(3, H, W, dtype=torch.float32,
                               device=device, requires_grad=True)
    optimizer    = torch.optim.Adam([perturbation], lr=LR, weight_decay=1e-4)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LR * 0.1
    )

    best_asr, no_improve = 0.0, 0
    log_lines = []
    start     = time.time()

    print(f"Surrogates: {list(SURROGATE_IDS.keys())}")
    print(f"Epsilon: {EPSILON*255:.0f}/255  LR: {LR}  CW margin: {CW_MARGIN}\n")

    for epoch in range(NUM_EPOCHS):
        epoch_loss = {k: 0.0 for k in SURROGATE_IDS}
        idxs       = list(range(len(train_set)))
        random.shuffle(idxs)
        optimizer.zero_grad()

        for step, data in enumerate(
                tqdm(train_set.select(idxs), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):

            raw_pil   = data["image"].convert("RGB").resize((H, W))
            step_loss = torch.tensor(0.0, device=device)

            for key in SURROGATE_IDS:
                proc = processors[key]
                T    = len(proc.tokenizer.encode(TARGET_CAPTION, add_special_tokens=False))

                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": raw_pil},
                        {"type": "text",  "text": QUESTION},
                    ]},
                    {"role": "assistant", "content": TARGET_CAPTION},
                ]
                text_prompt = proc.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False)
                inputs = proc(text=[text_prompt], images=[raw_pil],
                              padding=True, return_tensors="pt").to(device)
                inputs.pop("token_type_ids", None)
                inputs["pixel_values"] = pil_to_patches(
                    raw_pil, perturbation.to(torch.bfloat16),
                    means[key], stds[key],
                    patch_size=16, temporal_patch_size=2, device=device
                )

                outputs   = models[key](**inputs)
                logits    = outputs.logits[:, -(T + 1), :]
                tv_id     = tv_ids[key]
                logit_t   = logits[0, tv_id]
                competing = logits[0].clone()
                competing[tv_id] = -1e9
                cw        = torch.clamp(competing.max() - logit_t + CW_MARGIN, min=0.0)
                step_loss = step_loss + cw / len(SURROGATE_IDS)
                epoch_loss[key] += cw.item()

            (step_loss / GRAD_ACCUM_STEPS).backward()

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
        elapsed  = (time.time() - start) / 60
        loss_str = "  ".join(
            f"CW_{k}: {epoch_loss[k]/len(train_set):.4f}" for k in SURROGATE_IDS)
        line = f"Epoch {epoch+1:3d}  {loss_str}  elapsed: {elapsed:.1f}min"
        print(line)
        log_lines.append(line)

        # ASR check on 8B every 2 epochs
        if (epoch + 1) % 4 == 0:
            hits = 0
            with torch.no_grad():
                for data in asr_sub:
                    raw_pil = data["image"].convert("RGB").resize((H, W))
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": raw_pil},
                        {"type": "text",  "text": QUESTION},
                    ]}]
                    text   = processors["8b"].apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                    inputs = processors["8b"](text=[text], images=[raw_pil],
                                              padding=True, return_tensors="pt").to(device)
                    inputs.pop("token_type_ids", None)
                    inputs["pixel_values"] = pil_to_patches(
                        raw_pil, perturbation.to(torch.bfloat16),
                        means["8b"], stds["8b"],
                        patch_size=16, temporal_patch_size=2, device=device
                    )
                    gen     = models["8b"].generate(**inputs, max_new_tokens=10, do_sample=False)
                    new_tok = gen[0][inputs.input_ids.shape[1]:]
                    decoded = processors["8b"].tokenizer.decode(
                        new_tok, skip_special_tokens=True).lower()
                    if TARGET_WORD in decoded:
                        hits += 1

            asr  = hits / ASR_CHECK_SIZE
            line = f"  -> Live ASR 8B ({ASR_CHECK_SIZE} val): {asr*100:.1f}%"
            print(line)
            log_lines.append(line)

            if asr > best_asr + 0.01:
                best_asr, no_improve = asr, 0
                ckpt = CKPT_DIR / "coco_tv_ensemble_best.pt"
                torch.save(perturbation.detach().cpu().to(torch.bfloat16), ckpt)
                line = f"  -> New best ({best_asr*100:.1f}%)! Saved → {ckpt}"
            else:
                no_improve += 1
                line = f"  -> No improvement ({no_improve}/{PATIENCE})"
                if no_improve >= PATIENCE:
                    print(line)
                    log_lines.append(line)
                    print("Early stopping.")
                    break
            print(line)
            log_lines.append(line)

    final = CKPT_DIR / "coco_tv_ensemble_final.pt"
    torch.save(perturbation.detach().cpu().to(torch.bfloat16), final)
    total = (time.time() - start) / 60
    for line in [f"\nDone. Best ASR: {best_asr*100:.1f}%  Total: {total:.1f}min",
                 f"Best  → {CKPT_DIR/'coco_tv_ensemble_best.pt'}",
                 f"Final → {final}"]:
        print(line)
        log_lines.append(line)

    (LOG_DIR / "coco_tv_ensemble.log").write_text("\n".join(log_lines))
    print(f"Log → {LOG_DIR/'coco_tv_ensemble.log'}")


if __name__ == "__main__":
    main()
