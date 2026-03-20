import os
import csv
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# ── Argument parsing ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["3b"],
                    help="Which models to eval: 3b 7b 32b 72b 2b 4b 8b 32bv3")
parser.add_argument("--hf_cache", type=str, default=None,
                    help="Override HuggingFace cache dir (use a large scratch partition)")
parser.add_argument("--delete_after_eval", action="store_true",
                    help="Delete model weights from HF cache after evaluating each model")
args = parser.parse_args()

if args.hf_cache:
    os.environ["HF_HOME"] = args.hf_cache
    os.environ["TRANSFORMERS_CACHE"] = args.hf_cache
    print(f"HF cache → {args.hf_cache}")

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, W     = 448, 448
QUESTION = "What animal is in this picture? Answer in one word."
ROOT     = "eval_results_qwen_only"
CSV_PATH = os.path.join(ROOT, "results.csv")

# ── Model registry ─────────────────────────────────────────────────────────────
ALL_MODELS = {
    "3b":    ("Qwen/Qwen2.5-VL-3B-Instruct",  "qwen25", 14),
    "7b":    ("Qwen/Qwen2.5-VL-7B-Instruct",  "qwen25", 14),
    "32b":   ("Qwen/Qwen2.5-VL-32B-Instruct", "qwen25", 14),
    "72b":   ("Qwen/Qwen2.5-VL-72B-Instruct", "qwen25", 14),
    "2b":    ("Qwen/Qwen3-VL-2B-Instruct",    "qwen3",  16),
    "4b":    ("Qwen/Qwen3-VL-4B-Instruct",    "qwen3",  16),
    "8b":    ("Qwen/Qwen3-VL-8B-Instruct",    "qwen3",  16),
    "32bv3": ("Qwen/Qwen3-VL-32B-Instruct",   "qwen3",  16),
}

SURROGATES = {
    "pert_3b": "qwen25-vl-3b",
    "pert_8b": "qwen3-vl-8b",
}

MODELS = [(v[0], v[1], v[2]) for k, v in ALL_MODELS.items() if k in args.models]
if not MODELS:
    raise ValueError(f"No valid model keys. Choose from: {list(ALL_MODELS.keys())}")

print(f"Will evaluate {len(MODELS)} model(s): {[m[0] for m in MODELS]}")

os.makedirs(ROOT, exist_ok=True)


# ── Disk space check ───────────────────────────────────────────────────────────
def check_disk_space(path, min_gb=20):
    import shutil
    stat    = shutil.disk_usage(path)
    free_gb = stat.free / (1024 ** 3)
    if free_gb < min_gb:
        print(f"⚠️  WARNING: Only {free_gb:.1f} GB free on {path}.")
    else:
        print(f"✓ Disk space OK: {free_gb:.1f} GB free")

check_disk_space(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))


# ── Delete model cache ─────────────────────────────────────────────────────────
def delete_model_cache(model_id):
    import shutil
    folder_name = "models--" + model_id.replace("/", "--")
    cache_root  = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_path  = os.path.join(cache_root, "hub", folder_name)
    if os.path.exists(cache_path):
        size_gb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk(cache_path) for f in files
        ) / (1024 ** 3)
        shutil.rmtree(cache_path)
        print(f"  🗑  Deleted {cache_path} ({size_gb:.1f} GB freed)")


# ── Short model name ───────────────────────────────────────────────────────────
def model_short(model_id):
    return model_id.split("/")[-1].replace("-Instruct", "").replace(".", "").lower()

for mid, _, _ in MODELS:
    for pert in ["pert_3b", "pert_8b"]:
        os.makedirs(os.path.join(ROOT, model_short(mid), pert), exist_ok=True)


# ── pil_to_patches: matches training script exactly ───────────────────────────
def pil_to_patches(pil_image, perturbation, mean, std,
                   patch_size=16, temporal_patch_size=2, device="cuda"):
    """
    Replicates training-time preprocessing exactly:
      1. Convert PIL → float tensor
      2. Add perturbation and clamp to [0, 1]
      3. Normalise with model mean/std
      4. Patchify into (num_patches, patch_dim) matching model input format

    This MUST be used instead of baking the perturbation into the PIL image,
    because the processor's own normalisation would shift the perturbation
    into the wrong numerical space otherwise.
    """
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


# ── Image saving ───────────────────────────────────────────────────────────────
def save_result(pil_image, reply, dog_prob, top_token, top_prob, out_dir, idx):
    success = "dog" in reply.lower()
    bg      = (34, 139, 34) if success else (180, 50, 50)
    label   = "DOG" if success else reply[:12].replace(" ", "_").upper()
    fname   = f"{idx:04d}_{label}.jpg"
    bar_h   = 54
    out_img = Image.new("RGB", (pil_image.width, pil_image.height + bar_h), color=bg)
    out_img.paste(pil_image, (0, 0))
    draw    = ImageDraw.Draw(out_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    caption = f'pred: "{reply}"  |  P(dog)={dog_prob*100:.1f}%  |  top: "{top_token}" {top_prob*100:.1f}%'
    bbox    = draw.textbbox((0, 0), caption, font=font)
    tw, th  = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((pil_image.width - tw) // 2, pil_image.height + (bar_h - th) // 2),
              caption, fill=(255, 255, 255), font=font)
    out_img.save(os.path.join(out_dir, fname))

    # Also save to successes/ or failures/ mirror folder
    mirror_root = "successes" if success else "failures"
    mirror_dir  = os.path.join(mirror_root, os.path.relpath(out_dir, ROOT))
    os.makedirs(mirror_dir, exist_ok=True)
    out_img.save(os.path.join(mirror_dir, fname))

    return success


# ── Inference using pil_to_patches injection ───────────────────────────────────
def qwen_infer(raw_pil, perturbation, mean, std, patch_size,
               model, processor, dog_token_id):
    """
    raw_pil     — original unperturbed PIL image (processor uses this for
                  everything except pixel_values, which we override below)
    perturbation — raw float32 tensor (3, H, W), NOT yet normalised
    """
    messages = [{"role": "user", "content": [
        {"type": "image", "image": raw_pil},
        {"type": "text",  "text": QUESTION},
    ]}]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[raw_pil], return_tensors="pt").to(DEVICE)
    inputs.pop("token_type_ids", None)

    # ── Key fix: override pixel_values with correctly injected patches ─────────
    inputs["pixel_values"] = pil_to_patches(
        raw_pil, perturbation.to(torch.bfloat16), mean, std,
        patch_size=patch_size, temporal_patch_size=2, device=DEVICE
    )

    with torch.no_grad():
        out          = model(**inputs)
        first_logits = out.logits[0, -1, :].float()
        probs        = F.softmax(first_logits, dim=-1)
        dog_prob     = probs[dog_token_id].item()
        top_id       = probs.argmax().item()
        top_token    = processor.tokenizer.decode([top_id]).strip()
        top_prob     = probs[top_id].item()
        gen          = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        reply        = processor.tokenizer.decode(
            gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

    return reply, dog_prob, top_token, top_prob


# ── Load perturbations (keep as float32 — cast inside pil_to_patches) ──────────
print("Loading perturbations...")
pert_3b = torch.load("qwen25_3b_cw_perturbation_best.pt",              map_location=DEVICE).float()
pert_8b = torch.load("qwen3vl_8b_universal_dog_perturbation_best.pt",  map_location=DEVICE).float()
print(f"  pert_3b L-inf: {pert_3b.abs().max():.4f}")
print(f"  pert_8b L-inf: {pert_8b.abs().max():.4f}")

# ── Load dataset ───────────────────────────────────────────────────────────────
print("Loading dataset...")
dataset     = load_dataset("Bingsu/Cat_and_Dog", split="train")
cat_dataset = dataset.filter(lambda ex: ex['labels'] == 0)
n_total     = len(cat_dataset)
val_set     = cat_dataset.select(range(n_total - 200, n_total))
raw_images  = [data['image'].convert("RGB").resize((H, W)) for data in val_set]
print(f"Val set: {len(raw_images)} cats (held-out)\n")

# ── CSV setup ──────────────────────────────────────────────────────────────────
csv_rows   = []
csv_fields = ["model", "perturbation", "total", "hits", "misses", "asr_pct", "is_surrogate"]

# ── Evaluate each model ────────────────────────────────────────────────────────
for model_id, class_key, patch_size in MODELS:
    short = model_short(model_id)
    print(f"\n{'='*60}")
    print(f"Loading {model_id}  (patch_size={patch_size})...")

    processor = AutoProcessor.from_pretrained(model_id)

    if class_key == "qwen25":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16).to(DEVICE).eval()
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16).to(DEVICE).eval()

    for param in model.parameters():
        param.requires_grad = False

    # Per-model normalisation constants (important: each model may differ)
    img_proc = processor.image_processor
    mean = torch.tensor(img_proc.image_mean, dtype=torch.bfloat16, device=DEVICE).view(3, 1, 1)
    std  = torch.tensor(img_proc.image_std,  dtype=torch.bfloat16, device=DEVICE).view(3, 1, 1)

    dog_token_id = processor.tokenizer.encode("dog", add_special_tokens=False)[0]

    for pert_name, perturbation in [("pert_3b", pert_3b), ("pert_8b", pert_8b)]:
        hits    = 0
        out_dir = os.path.join(ROOT, short, pert_name)

        for idx, raw_pil in enumerate(tqdm(raw_images, desc=f"  {short} | {pert_name}"), start=1):
            reply, dog_prob, top_token, top_prob = qwen_infer(
                raw_pil, perturbation, mean, std, patch_size,
                model, processor, dog_token_id
            )
            hit   = save_result(raw_pil, reply, dog_prob, top_token, top_prob, out_dir, idx)
            hits += hit

        total        = len(raw_images)
        misses       = total - hits
        asr_pct      = round(hits / total * 100, 1)
        is_surrogate = (SURROGATES.get(pert_name) == short)

        csv_rows.append({
            "model":        short,
            "perturbation": pert_name,
            "total":        total,
            "hits":         hits,
            "misses":       misses,
            "asr_pct":      asr_pct,
            "is_surrogate": is_surrogate,
        })

        print(f"  {short:30s} | {pert_name} | "
              f"ASR {hits}/{total} = {asr_pct:.1f}%"
              + (" [surrogate]" if is_surrogate else ""))

    del model
    torch.cuda.empty_cache()

    if args.delete_after_eval:
        delete_model_cache(model_id)

# ── Write CSV ──────────────────────────────────────────────────────────────────
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\nCSV saved → {CSV_PATH}")

# ── Print summary table ────────────────────────────────────────────────────────
models_seen = sorted(set(r["model"] for r in csv_rows))
perts_seen  = sorted(set(r["perturbation"] for r in csv_rows))
lookup      = {(r["model"], r["perturbation"]): r for r in csv_rows}
col_w       = 14

print(f"\n{'Model':<35}" + "".join(f"{p:>{col_w}}" for p in perts_seen))
print("-" * (35 + col_w * len(perts_seen)))
for m in models_seen:
    line = f"{m:<35}"
    for p in perts_seen:
        entry = lookup.get((m, p))
        if entry is None:
            cell = "—"
        else:
            tag  = " *" if entry["is_surrogate"] else ""
            cell = f"{entry['asr_pct']:.1f}%{tag}"
        line += f"{cell:>{col_w}}"
    print(line)
print("* = surrogate model")
print(f"\nResults    → {CSV_PATH}")
print(f"Successes  → successes/")
print(f"Failures   → failures/")