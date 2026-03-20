import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm
import random


def pil_to_patches(pil_image, perturbation, mean, std,
                   patch_size=14, temporal_patch_size=2, device="cuda"):
    """
    Differentiable replication of Qwen2.5-VL's image preprocessing.
    Gradients flow back to `perturbation` through this function.
    Output shape: (1024, 1176) for a 448x448 image.
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    print("Loading model and processor...")
    model_id  = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model     = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    img_proc = processor.image_processor
    mean = torch.tensor(img_proc.image_mean, dtype=torch.bfloat16, device=device).view(3,1,1)
    std  = torch.tensor(img_proc.image_std,  dtype=torch.bfloat16, device=device).view(3,1,1)

    # ------------------------------------------------------------------
    # DATASET — use ALL available cat images, not just 500
    # ------------------------------------------------------------------
    # The previous run used 500 images and Adam drove loss to 0.075,
    # which sounds great but means the perturbation memorized those exact
    # 500 images (2% success on held-out images confirmed this).
    #
    # A universal perturbation must generalize across the full distribution
    # of the target class. Using all ~4000 cats forces the optimizer to find
    # a perturbation direction that works for ALL of them, not one that
    # perfectly fits a small subset.
    print("Downloading and preparing dataset...")
    dataset     = load_dataset("Bingsu/Cat_and_Dog", split="train")
    cat_dataset = dataset.filter(lambda ex: ex['labels'] == 0)
    # Reserve last 200 for validation so we can track generalization live
    n_total   = len(cat_dataset)
    n_val     = 200
    n_train   = n_total - n_val
    train_set = cat_dataset.select(range(n_train))
    val_set   = cat_dataset.select(range(n_train, n_total))
    print(f"Train: {n_train} cats | Val: {n_val} cats")

    # ------------------------------------------------------------------
    # ATTACK PARAMETERS
    # ------------------------------------------------------------------
    target_caption = "A dog."
    epsilon    = 32.0 / 255.0
    num_epochs = 30
    H, W       = 448, 448

    target_ids    = processor.tokenizer.encode(target_caption, add_special_tokens=False)
    target_tensor = torch.tensor(target_ids, device=device)
    T = len(target_ids)
    print(f"Target: '{target_caption}' -> token IDs {target_ids}  (length {T})")

    # ------------------------------------------------------------------
    # PERTURBATION + OPTIMIZER
    # ------------------------------------------------------------------
    perturbation = torch.zeros(3, H, W, dtype=torch.float32, device=device,
                               requires_grad=True)

    optimizer = torch.optim.Adam(
        [perturbation],
        lr=0.003,           # lower than before (was 0.005) — slower but less overfit
        weight_decay=1e-4,  # L2 regularization on the perturbation itself.
                            # Pulls perturbation values toward zero, preventing
                            # any single pixel from growing too extreme and
                            # image-specifically. Helps generalization.
    )

    # Cosine annealing but with a higher eta_min floor — don't let lr decay
    # all the way to near-zero, which caused the overfit in the last run
    # (the very low lr in epochs 27-30 let Adam make tiny, image-specific
    # adjustments that didn't generalize).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0.001
    )

    loss_fn = nn.CrossEntropyLoss()

    best_val_loss    = float('inf')
    patience         = 6
    no_improve_count = 0

    print("Starting optimization loop...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # ------------------------------------------------------------------
        # SHUFFLE TRAINING SET EACH EPOCH
        # ------------------------------------------------------------------
        # This is critical for universality. If images are always seen in the
        # same order, the momentum terms in Adam develop image-order-specific
        # structure. Shuffling forces the gradient signal to be consistent
        # across the whole distribution rather than order-dependent.
        indices = list(range(n_train))
        random.shuffle(indices)
        shuffled_train = train_set.select(indices)

        model.train(False)  # keep model in eval mode
        for data in tqdm(shuffled_train, desc=f"Epoch {epoch+1}/{num_epochs} [train]"):
            raw_image = data['image'].convert("RGB").resize((H, W))

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": raw_image},
                        {"type": "text",  "text": "What animal is in this picture? Just answer in one word."},
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

            inputs["pixel_values"] = pil_to_patches(
                raw_image, perturbation.to(torch.bfloat16), mean, std,
                patch_size=14, temporal_patch_size=2, device=device
            )

            outputs      = model(**inputs)
            logits_seq   = outputs.logits[:, -(T+1):-1, :]
            logits_flat  = logits_seq.reshape(-1, logits_seq.size(-1))
            targets_flat = target_tensor.reshape(-1)

            loss = loss_fn(logits_flat, targets_flat)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                perturbation.clamp_(-epsilon, epsilon)

        scheduler.step()
        avg_train_loss = epoch_loss / n_train

        # ------------------------------------------------------------------
        # VALIDATION LOSS — track generalization, not training loss
        # ------------------------------------------------------------------
        # This is what we checkpoint on. A universal perturbation that
        # generalizes should have low VAL loss, not just low train loss.
        # If train loss is 0.1 but val loss is 8.0, the attack is overfit.
        val_loss = 0.0
        with torch.no_grad():
            for data in tqdm(val_set, desc=f"Epoch {epoch+1}/{num_epochs} [val]", leave=False):
                raw_image = data['image'].convert("RGB").resize((H, W))
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": raw_image},
                            {"type": "text",  "text": "What animal is in this picture? Just answer in one word."},
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
                inputs["pixel_values"] = pil_to_patches(
                    raw_image, perturbation.to(torch.bfloat16), mean, std,
                    patch_size=14, temporal_patch_size=2, device=device
                )
                outputs      = model(**inputs)
                logits_seq   = outputs.logits[:, -(T+1):-1, :]
                logits_flat  = logits_seq.reshape(-1, logits_seq.size(-1))
                targets_flat = target_tensor.reshape(-1)
                val_loss += loss_fn(logits_flat, targets_flat).item()

        avg_val_loss = val_loss / n_val
        current_lr   = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}  Train: {avg_train_loss:.4f}  Val: {avg_val_loss:.4f}  lr: {current_lr:.5f}")

        # Checkpoint on VAL loss — this is the generalization metric
        if avg_val_loss < best_val_loss - 0.005:
            best_val_loss    = avg_val_loss
            no_improve_count = 0
            torch.save(perturbation.detach().cpu().to(torch.bfloat16),
                       "qwen_universal_dog_perturbation_best.pt")
            print(f"  -> New best val ({best_val_loss:.4f})! Checkpoint saved.")
        else:
            no_improve_count += 1
            print(f"  -> No improvement ({no_improve_count}/{patience})")
            if no_improve_count >= patience:
                print("Early stopping.")
                break

    torch.save(perturbation.detach().cpu().to(torch.bfloat16),
               "qwen_universal_dog_perturbation.pt")
    print("Best : qwen_universal_dog_perturbation_best.pt")
    print("Final: qwen_universal_dog_perturbation.pt")


if __name__ == "__main__":
    main()