import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm
import random

login(token=os.environ["HF_TOKEN"])

GRAD_ACCUM_STEPS = 4


def pil_to_patches(pil_image, perturbation, mean, std,
                   patch_size=16, temporal_patch_size=2, device="cuda"):
    """
    Differentiable replication of Qwen3-VL's image preprocessing.
    Qwen3-VL uses SigLIP-2 with patch_size=16 (was 14 in Qwen2.5-VL).
    Output shape: (784, 1536) for a 448x448 image.
      - 784  = (448/16)^2 = 28x28 patches
      - 1536 = temporal_patch_size(2) * channels(3) * 16 * 16
    """
    img_tensor = TF.to_tensor(pil_image).to(device=device, dtype=torch.bfloat16)
    perturbed  = (img_tensor + perturbation).clamp(0.0, 1.0)
    normalized = (perturbed - mean) / std
    img = normalized.unsqueeze(0).repeat(temporal_patch_size, 1, 1, 1)
    t, c, h, w = img.shape
    ph, pw = h // patch_size, w // patch_size
    img = img.reshape(t, c, ph, patch_size, pw, patch_size)
    img = img.permute(2, 4, 0, 1, 3, 5).contiguous()
    img = img.reshape(ph * pw, t * c * patch_size * patch_size)
    return img


def compute_asr(model, processor, val_subset, perturbation, mean, std,
                target_word="dog", device="cuda", H=448, W=448):
    hits = 0
    with torch.no_grad():
        for data in val_subset:
            raw_image = data['image'].convert("RGB").resize((H, W))
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": raw_image},
                    {"type": "text",
                     "text": "What animal is in this picture? Just answer in one word."},
                ],
            }]
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
            if target_word in decoded:
                hits += 1
    return hits / len(val_subset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    print("Loading model and processor...")
    model_id  = "Qwen/Qwen3-VL-8B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model     = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    img_proc = processor.image_processor
    mean = torch.tensor(img_proc.image_mean, dtype=torch.bfloat16, device=device).view(3,1,1)
    std  = torch.tensor(img_proc.image_std,  dtype=torch.bfloat16, device=device).view(3,1,1)

    # Pre-compute cat token IDs once
    cat_token_ids  = processor.tokenizer.encode(
        "cat Cat CAT feline kitty kitten", add_special_tokens=False
    )
    cat_ids_tensor = torch.tensor(cat_token_ids, device=device)
    print(f"Cat token IDs to suppress: {cat_token_ids}")

    print("Downloading and preparing dataset...")
    dataset     = load_dataset("Bingsu/Cat_and_Dog", split="train")
    cat_dataset = dataset.filter(lambda ex: ex['labels'] == 0)
    n_total   = len(cat_dataset)
    n_val     = 200
    n_train   = n_total - n_val
    train_set = cat_dataset.select(range(n_train))
    val_set   = cat_dataset.select(range(n_train, n_total))
    asr_check_subset = val_set.select(range(50))
    print(f"Train: {n_train} cats | Val: {n_val} cats | ASR check: 50 cats")

    target_caption = "A dog."
    epsilon        = 32.0 / 255.0
    num_epochs     = 30
    H, W           = 448, 448

    target_ids    = processor.tokenizer.encode(target_caption, add_special_tokens=False)
    target_tensor = torch.tensor(target_ids, device=device)
    T             = len(target_ids)
    dog_id        = target_tensor[0]
    print(f"Target: '{target_caption}' -> token IDs {target_ids}  (length {T})")
    print(f"Dog first-token ID: {dog_id.item()}")

    perturbation = torch.zeros(3, H, W, dtype=torch.float32,
                               device=device, requires_grad=True)

    optimizer = torch.optim.Adam(
        [perturbation],
        lr=0.003,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0.001
    )
    loss_fn = nn.CrossEntropyLoss()

    CW_MARGIN = 5.0
    CW_WEIGHT = 0.8
    CE_WEIGHT = 0.2

    best_val_asr     = 0.0
    patience         = 6
    no_improve_count = 0

    print("Starting optimization loop...")
    for epoch in range(num_epochs):
        epoch_cw_loss = 0.0
        epoch_ce_loss = 0.0

        indices = list(range(n_train))
        random.shuffle(indices)
        shuffled_train = train_set.select(indices)

        model.train(False)
        optimizer.zero_grad()

        for step, data in enumerate(tqdm(shuffled_train,
                                         desc=f"Epoch {epoch+1}/{num_epochs} [train]")):
            raw_image = data['image'].convert("RGB").resize((H, W))

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": raw_image},
                        {"type": "text",
                         "text": "What animal is in this picture? Just answer in one word."},
                    ],
                },
                {"role": "assistant", "content": target_caption},
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

            # ── CW first-token loss ────────────────────────────────────
            first_token_logits = outputs.logits[:, -(T + 1), :]   # (1, vocab)
            logit_dog = first_token_logits[0, dog_id]
            logit_cat = first_token_logits[0, cat_ids_tensor].max()
            cw_loss   = torch.clamp(logit_cat - logit_dog + CW_MARGIN, min=0.0)

            # ── CE sequence loss ───────────────────────────────────────
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
                    perturbation.clamp_(-epsilon, epsilon)

        if n_train % GRAD_ACCUM_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                perturbation.clamp_(-epsilon, epsilon)

        scheduler.step()
        avg_cw     = epoch_cw_loss / n_train
        avg_ce     = epoch_ce_loss / n_train
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}  CW: {avg_cw:.4f}  CE: {avg_ce:.4f}  lr: {current_lr:.5f}")

        # ── Live ASR check every 2 epochs ─────────────────────────────
        if (epoch + 1) % 2 == 0:
            asr = compute_asr(
                model, processor, asr_check_subset,
                perturbation, mean, std,
                target_word="dog", device=device
            )
            print(f"  -> Live ASR (50 val images): {asr*100:.1f}%")

            if asr > best_val_asr + 0.01:
                best_val_asr     = asr
                no_improve_count = 0
                torch.save(perturbation.detach().cpu().to(torch.bfloat16),
                           "qwen3vl_8b_universal_dog_perturbation_best.pt")
                print(f"  -> New best ASR ({best_val_asr*100:.1f}%)! Checkpoint saved.")
            else:
                no_improve_count += 1
                print(f"  -> No ASR improvement ({no_improve_count}/{patience})")
                if no_improve_count >= patience:
                    print("Early stopping.")
                    break

    torch.save(perturbation.detach().cpu().to(torch.bfloat16),
               "qwen3vl_8b_universal_dog_perturbation.pt")
    print("Best : qwen3vl_8b_universal_dog_perturbation_best.pt")
    print("Final: qwen3vl_8b_universal_dog_perturbation.pt")


if __name__ == "__main__":
    main()
