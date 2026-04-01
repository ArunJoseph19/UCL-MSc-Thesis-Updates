"""
leverage_scoring.py
-------------------
Autoregressive Amplification Attack (AAA) — Checkpoint 1
Scores candidate pivot tokens by autoregressive leverage:
  leverage(t) = mean CLIP cosine distance between clean caption
                and free-form output when t is forced as first token

Usage:
    HF_TOKEN=xxx python scripts/leverage_scoring.py \
        --image_dir coco/images/val2017 \
        --captions_file coco/annotations/captions_val2017.json \
        --n_images 5 \
        --output results/leverage_scores.csv
"""

import os
import json
import argparse
import time
import random
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from huggingface_hub import login

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir",     default="coco/images/val2017")
parser.add_argument("--captions_file", default="coco/annotations/captions_val2017.json")
parser.add_argument("--n_images",      type=int, default=5)
parser.add_argument("--output",        default="results/leverage_scores.csv")
parser.add_argument("--epsilon",       type=float, default=64/255)
parser.add_argument("--lr",            type=float, default=0.01)
parser.add_argument("--epochs",        type=int,   default=15)
parser.add_argument("--cw_margin",     type=float, default=2.0)
parser.add_argument("--max_new_tokens",type=int,   default=60)
parser.add_argument("--seed",          type=int,   default=42)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

# ── Candidate pivot tokens ─────────────────────────────────────────────────────
# Covers: sentiment, negation, category, uncertainty, modal, filler
PIVOT_TOKENS = [
    # Sentiment / failure
    "unfortunately", "sadly", "regrettably", "warning",
    # Negation / absence
    "no", "nothing", "none", "empty",
    # Category redirect
    "dog", "car", "fire",
    # Uncertainty / corruption
    "unclear", "blurry", "damaged", "corrupted",
    # Modal / hedge
    "possibly", "maybe",
    # Filler / neutral (expected low leverage — control group)
    "the", "this", "a",
]

# ── Setup ──────────────────────────────────────────────────────────────────────
login(token=os.environ["HF_TOKEN"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

H, W = 448, 448
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # swap to Qwen3-VL-8B if available

print(f"Loading {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16
).to(device).eval()
for p in model.parameters():
    p.requires_grad = False

img_proc = processor.image_processor
mean = torch.tensor(img_proc.image_mean, dtype=torch.bfloat16, device=device).view(3, 1, 1)
std  = torch.tensor(img_proc.image_std,  dtype=torch.bfloat16, device=device).view(3, 1, 1)

# ── CLIP for semantic distance ─────────────────────────────────────────────────
print("Loading CLIP...")
import open_clip
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai"
)
clip_model = clip_model.to(device).eval()
clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

def clip_text_embed(text):
    """Returns normalised CLIP text embedding."""
    tokens = clip_tokenizer([text]).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens)
    return F.normalize(emb, dim=-1)

def clip_distance(text_a, text_b):
    """Cosine distance (1 - cosine similarity) between two strings."""
    ea = clip_text_embed(text_a)
    eb = clip_text_embed(text_b)
    sim = (ea * eb).sum().item()
    return 1.0 - sim  # distance: 0 = identical, 2 = opposite

# ── pil_to_patches ─────────────────────────────────────────────────────────────
def pil_to_patches(pil_image, perturbation, patch_size=16, temporal_patch_size=2):
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

# ── Load COCO images + first caption ──────────────────────────────────────────
print("Loading COCO captions...")
with open(args.captions_file) as f:
    coco = json.load(f)

# Map image_id -> first caption
id_to_caption = {}
for ann in coco["annotations"]:
    if ann["image_id"] not in id_to_caption:
        id_to_caption[ann["image_id"]] = ann["caption"]

# Map image_id -> filename
id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

# Filter to images that exist on disk
image_dir = Path(args.image_dir)
valid = [
    (iid, id_to_file[iid], id_to_caption[iid])
    for iid in id_to_caption
    if (image_dir / id_to_file[iid]).exists()
]
random.shuffle(valid)
selected = valid[:args.n_images]
print(f"Selected {len(selected)} images")

# Load PIL images
images = []
for iid, fname, caption in selected:
    pil = Image.open(image_dir / fname).convert("RGB").resize((H, W))
    images.append({"pil": pil, "caption": caption, "fname": fname})

# ── Generate clean captions from model ────────────────────────────────────────
QUESTION = "Describe this image in one sentence."

def get_model_caption(pil_image, perturbation=None):
    """Generate a free-form caption. If perturbation given, inject it."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text",  "text": QUESTION},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt").to(device)
    inputs.pop("token_type_ids", None)

    if perturbation is not None:
        inputs["pixel_values"] = pil_to_patches(pil_image, perturbation.to(torch.bfloat16))

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    decoded = processor.tokenizer.decode(
        gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    return decoded

print("\nGenerating clean captions...")
for item in images:
    item["clean_caption"] = get_model_caption(item["pil"])
    print(f"  {item['fname']}: {item['clean_caption']}")

# ── Verify pivot token is single token ────────────────────────────────────────
def get_token_id(word):
    ids = processor.tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        return None  # skip multi-token words
    return ids[0]

# ── Train perturbation for a given pivot token ─────────────────────────────────
def train_perturbation(pivot_token_id):
    """Train a UAP to force pivot_token_id as first generated token."""
    perturbation = torch.zeros(3, H, W, dtype=torch.float32,
                               device=device, requires_grad=True)
    optimizer = torch.optim.Adam([perturbation], lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    pivot_id = torch.tensor(pivot_token_id, device=device)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        random.shuffle(images)
        optimizer.zero_grad()

        for step, item in enumerate(images):
            pil = item["pil"]
            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text",  "text": QUESTION},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[text], images=[pil], return_tensors="pt").to(device)
            inputs.pop("token_type_ids", None)
            inputs["pixel_values"] = pil_to_patches(
                pil, perturbation.to(torch.bfloat16)
            )

            outputs = model(**inputs)
            # First generated position logits
            first_logits = outputs.logits[0, -1, :]
            logit_target = first_logits[pivot_id]
            competing = first_logits.clone()
            competing[pivot_id] = -1e9
            logit_competitor = competing.max()
            cw_loss = torch.clamp(
                logit_competitor - logit_target + args.cw_margin, min=0.0
            )

            loss = cw_loss / len(images)
            epoch_loss += cw_loss.item()
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            perturbation.clamp_(-args.epsilon, args.epsilon)
        scheduler.step()

    return perturbation.detach()

# ── Score a pivot token ────────────────────────────────────────────────────────
def score_token(pivot_word):
    token_id = get_token_id(pivot_word)
    if token_id is None:
        print(f"  [{pivot_word}] — multi-token, skipping")
        return None

    print(f"\n  Training perturbation for '{pivot_word}' (id={token_id})...")
    t0 = time.time()
    perturbation = train_perturbation(token_id)
    elapsed = (time.time() - t0) / 60

    # Generate attacked captions and compute distances
    distances = []
    attacked_captions = []
    for item in images:
        attacked = get_model_caption(item["pil"], perturbation)
        dist = clip_distance(item["clean_caption"], attacked)
        distances.append(dist)
        attacked_captions.append(attacked)

    mean_dist = sum(distances) / len(distances)
    # Check how often first token is actually the pivot
    pivot_word_lower = pivot_word.lower()
    asr = sum(1 for c in attacked_captions if c.lower().startswith(pivot_word_lower)) / len(attacked_captions)

    print(f"    Leverage score: {mean_dist:.4f} | Token ASR: {asr*100:.0f}% | Time: {elapsed:.1f}min")
    for item, attacked in zip(images, attacked_captions):
        print(f"      Clean:   {item['clean_caption']}")
        print(f"      Attacked:{attacked}")
        print()

    return {
        "pivot_token": pivot_word,
        "token_id": token_id,
        "leverage_score": round(mean_dist, 4),
        "token_asr": round(asr, 4),
        "training_time_min": round(elapsed, 2),
        "n_images": len(images),
    }

# ── Main loop ─────────────────────────────────────────────────────────────────
Path(args.output).parent.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}")
print(f"Scoring {len(PIVOT_TOKENS)} pivot tokens on {args.n_images} images")
print(f"Epochs: {args.epochs} | Epsilon: {args.epsilon:.3f} | LR: {args.lr}")
print(f"{'='*60}\n")

results = []
for pivot in PIVOT_TOKENS:
    result = score_token(pivot)
    if result:
        results.append(result)

# Sort by leverage score descending
results.sort(key=lambda x: x["leverage_score"], reverse=True)

# Write CSV
fieldnames = ["rank", "pivot_token", "token_id", "leverage_score", "token_asr", "training_time_min", "n_images"]
with open(args.output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i, r in enumerate(results, 1):
        writer.writerow({"rank": i, **r})

# Print final ranking
print(f"\n{'='*60}")
print("LEVERAGE RANKING")
print(f"{'='*60}")
print(f"{'Rank':<6}{'Token':<18}{'Leverage':<12}{'ASR':<10}")
print("-" * 46)
for i, r in enumerate(results, 1):
    print(f"{i:<6}{r['pivot_token']:<18}{r['leverage_score']:<12.4f}{r['token_asr']*100:<10.0f}%")

print(f"\nResults saved to {args.output}")
