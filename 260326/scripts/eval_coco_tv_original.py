"""
eval_coco_tv.py
---------------
Evaluates coco_tv_pert_best.pt against all four Qwen3-VL models
on COCO val images (non-TV, same filter as training).

Usage:
    python eval_coco_tv.py
    python eval_coco_tv.py --models 2b 4b          # subset
    python eval_coco_tv.py --hf_cache /scratch/hf  # override cache
    python eval_coco_tv.py --delete_after_eval      # free disk between models
    python eval_coco_tv.py --n_eval 200             # number of val images
"""

import os, csv, argparse, shutil, time
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

login(token=os.environ["HF_TOKEN"])

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=["2b", "4b", "8b", "32b"],
                    choices=["2b", "4b", "8b", "32b"])
parser.add_argument("--hf_cache", type=str, default=None)
parser.add_argument("--delete_after_eval", action="store_true")
parser.add_argument("--n_eval", type=int, default=200,
                    help="How many non-TV COCO val images to evaluate on")
args = parser.parse_args()

if args.hf_cache:
    os.environ["HF_HOME"]          = args.hf_cache
    os.environ["TRANSFORMERS_CACHE"] = args.hf_cache
    print(f"HF cache → {args.hf_cache}")

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, W           = 448, 448
TV_CAT_ID      = 72          # COCO category id for "tv" — same as training
TARGET_WORD    = "tv"
QUESTION       = "What object is in this picture? Answer in one word."
PERT_PATH      = Path("checkpoints/coco_tv_pert_best.pt")
ROOT           = Path("eval_results_coco_tv")
CSV_PATH       = ROOT / "results.csv"

ALL_MODELS = {
    "2b":  "Qwen/Qwen3-VL-2B-Instruct",
    "4b":  "Qwen/Qwen3-VL-4B-Instruct",
    "8b":  "Qwen/Qwen3-VL-8B-Instruct",   # ← surrogate
    "32b": "Qwen/Qwen3-VL-32B-Instruct",
}
SURROGATE_KEY = "8b"

MODELS = [(k, ALL_MODELS[k]) for k in args.models]
ROOT.mkdir(exist_ok=True)

# ── Disk space check ───────────────────────────────────────────────────────────
def check_disk(min_gb=20):
    path    = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    free_gb = shutil.disk_usage(path).free / 1024**3
    status  = "✓" if free_gb >= min_gb else "⚠️ WARNING:"
    print(f"{status} {free_gb:.1f} GB free at {path}")

check_disk()

# ── Delete model from HF cache ─────────────────────────────────────────────────
def delete_model_cache(model_id):
    folder     = "models--" + model_id.replace("/", "--")
    cache_root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    path       = os.path.join(cache_root, "hub", folder)
    if os.path.exists(path):
        size_gb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk(path) for f in files
        ) / 1024**3
        shutil.rmtree(path)
        print(f"  🗑  Deleted {path} ({size_gb:.1f} GB freed)")

# ── pil_to_patches: identical to training ─────────────────────────────────────
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

# ── Inference ──────────────────────────────────────────────────────────────────
def infer(raw_pil, perturbation, mean, std,
          model, processor, tv_token_id):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": raw_pil},
        {"type": "text",  "text": QUESTION},
    ]}]
    text   = processor.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    inputs = processor(text=[text], images=[raw_pil],
                       return_tensors="pt").to(DEVICE)
    inputs.pop("token_type_ids", None)
    inputs["pixel_values"] = pil_to_patches(
        raw_pil, perturbation.to(torch.bfloat16), mean, std,
        patch_size=16, temporal_patch_size=2, device=DEVICE
    )

    with torch.no_grad():
        out          = model(**inputs)
        first_logits = out.logits[0, -1, :].float()
        probs        = F.softmax(first_logits, dim=-1)
        tv_prob      = probs[tv_token_id].item()
        top_id       = probs.argmax().item()
        top_token    = processor.tokenizer.decode([top_id]).strip()
        top_prob     = probs[top_id].item()
        gen          = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        reply        = processor.tokenizer.decode(
            gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

    return reply, tv_prob, top_token, top_prob

# ── Save annotated image ───────────────────────────────────────────────────────
def save_result(pil_image, reply, tv_prob, top_token, top_prob,
                out_dir: Path, idx: int) -> bool:
    success = TARGET_WORD in reply.lower()
    bg      = (34, 139, 34) if success else (180, 50, 50)
    label   = "TV" if success else reply[:12].replace(" ", "_").upper()
    fname   = f"{idx:04d}_{label}.jpg"

    bar_h   = 54
    out_img = Image.new("RGB", (pil_image.width, pil_image.height + bar_h), color=bg)
    out_img.paste(pil_image, (0, 0))
    draw    = ImageDraw.Draw(out_img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    caption = (f'pred: "{reply}"  |  '
               f'P(tv)={tv_prob*100:.1f}%  |  '
               f'top: "{top_token}" {top_prob*100:.1f}%')
    bbox    = draw.textbbox((0, 0), caption, font=font)
    tw, th  = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((pil_image.width - tw) // 2, pil_image.height + (bar_h - th) // 2),
        caption, fill=(255, 255, 255), font=font
    )
    out_img.save(str(out_dir / fname))

    # Mirror into successes/ or failures/
    mirror = Path("successes" if success else "failures") / out_dir.relative_to(ROOT)
    mirror.mkdir(parents=True, exist_ok=True)
    out_img.save(str(mirror / fname))

    return success

# ── Load perturbation ──────────────────────────────────────────────────────────
print(f"Loading perturbation from {PERT_PATH}...")
assert PERT_PATH.exists(), f"Not found: {PERT_PATH}"
perturbation = torch.load(PERT_PATH, map_location=DEVICE).float()
print(f"  L-inf: {perturbation.abs().max():.4f}  "
      f"shape: {tuple(perturbation.shape)}")

# ── Load COCO val (non-TV) ─────────────────────────────────────────────────────
print("\nLoading COCO validation split...")
coco_val = load_dataset("detection-datasets/coco", split="val")

def is_not_tv(example):
    if example.get("objects") is None:
        return True
    return TV_CAT_ID not in example["objects"].get("category", [])

print("Filtering out TV images...")
filtered = coco_val.filter(is_not_tv, num_proc=4)
print(f"  {len(coco_val)} total val → {len(filtered)} non-TV images available")

n_eval   = min(args.n_eval, len(filtered))
eval_set = filtered.select(range(n_eval))
raw_images = [d['image'].convert("RGB").resize((H, W)) for d in eval_set]
print(f"  Using {len(raw_images)} images for eval\n")

# ── Evaluate ───────────────────────────────────────────────────────────────────
csv_rows   = []
csv_fields = ["model", "total", "hits", "misses", "asr_pct",
              "is_surrogate", "elapsed_min"]

print(f"Evaluating {len(MODELS)} model(s)...\n")

for key, model_id in MODELS:
    short        = model_id.split("/")[-1].replace("-Instruct", "").lower()
    is_surrogate = (key == SURROGATE_KEY)
    out_dir      = ROOT / short
    out_dir.mkdir(exist_ok=True)

    print(f"{'='*60}")
    print(f"Model : {model_id}{'  [SURROGATE]' if is_surrogate else ''}")

    processor = AutoProcessor.from_pretrained(model_id)
    model     = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    img_proc = processor.image_processor
    mean = torch.tensor(img_proc.image_mean,
                        dtype=torch.bfloat16, device=DEVICE).view(3, 1, 1)
    std  = torch.tensor(img_proc.image_std,
                        dtype=torch.bfloat16, device=DEVICE).view(3, 1, 1)

    # "tv" tokenises to a single token in the Qwen3 tokenizer — confirm here
    tv_ids = processor.tokenizer.encode(TARGET_WORD, add_special_tokens=False)
    assert len(tv_ids) == 1, \
        f"Expected 'tv' to be 1 token, got {tv_ids} — update TARGET_WORD logic"
    tv_token_id = tv_ids[0]
    print(f"  'tv' token id : {tv_token_id}  "
          f"(decoded: '{processor.tokenizer.decode([tv_token_id])}')")

    hits     = 0
    t0       = time.time()

    for idx, raw_pil in enumerate(
            tqdm(raw_images, desc=f"  {short}"), start=1):
        reply, tv_prob, top_token, top_prob = infer(
            raw_pil, perturbation, mean, std,
            model, processor, tv_token_id
        )
        if save_result(raw_pil, reply, tv_prob, top_token, top_prob,
                       out_dir, idx):
            hits += 1

    total       = len(raw_images)
    asr_pct     = round(hits / total * 100, 1)
    elapsed_min = round((time.time() - t0) / 60, 1)

    csv_rows.append({
        "model":        short,
        "total":        total,
        "hits":         hits,
        "misses":       total - hits,
        "asr_pct":      asr_pct,
        "is_surrogate": is_surrogate,
        "elapsed_min":  elapsed_min,
    })
    print(f"  ASR: {hits}/{total} = {asr_pct:.1f}%"
          + ("  [surrogate]" if is_surrogate else "")
          + f"  ({elapsed_min:.1f} min)")

    del model
    torch.cuda.empty_cache()

    if args.delete_after_eval:
        delete_model_cache(model_id)

# ── CSV + summary ──────────────────────────────────────────────────────────────
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\nCSV → {CSV_PATH}")

print(f"\n{'Model':<40} {'ASR':>8} {'Hits':>6} {'Total':>6}  Surrogate?")
print("-" * 68)
for r in csv_rows:
    tag = "  ← surrogate" if r["is_surrogate"] else ""
    print(f"{r['model']:<40} {r['asr_pct']:>7.1f}% "
          f"{r['hits']:>6} {r['total']:>6}{tag}")

print(f"\nAnnotated images → {ROOT}/")
print(f"Successes        → successes/")
print(f"Failures         → failures/")