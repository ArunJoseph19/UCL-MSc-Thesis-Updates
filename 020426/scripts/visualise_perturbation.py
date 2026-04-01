"""
visualise_perturbation.py
-------------------------
Trains a perturbation for one pivot token and saves:
  - clean image
  - perturbed image (what the model sees)
  - perturbation amplified x10 (to see the noise pattern)
  - side by side HTML with clean vs attacked captions

Usage:
    HF_TOKEN=xxx python scripts/visualise_perturbation.py \
        --image_dir coco/images/val2017 \
        --captions_file coco/annotations/captions_val2017.json \
        --pivot_token the \
        --n_images 5 \
        --output_dir results/visualisations
"""

import os
import json
import argparse
import random
from pathlib import Path

import torch
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from huggingface_hub import login

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir",     default="coco/images/val2017")
parser.add_argument("--captions_file", default="coco/annotations/captions_val2017.json")
parser.add_argument("--pivot_token",   default="the")
parser.add_argument("--n_images",      type=int,   default=5)
parser.add_argument("--epsilon",       type=float, default=64/255)
parser.add_argument("--lr",            type=float, default=0.01)
parser.add_argument("--epochs",        type=int,   default=30)
parser.add_argument("--cw_margin",     type=float, default=2.0)
parser.add_argument("--max_new_tokens",type=int,   default=60)
parser.add_argument("--output_dir",    default="results/visualisations")
parser.add_argument("--seed",          type=int,   default=42)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

# ── Setup ──────────────────────────────────────────────────────────────────────
login(token=os.environ["HF_TOKEN"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

H, W = 448, 448
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

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

# ── Load images ────────────────────────────────────────────────────────────────
print("Loading COCO...")
with open(args.captions_file) as f:
    coco = json.load(f)

id_to_caption = {}
for ann in coco["annotations"]:
    if ann["image_id"] not in id_to_caption:
        id_to_caption[ann["image_id"]] = ann["caption"]

id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
image_dir  = Path(args.image_dir)

valid = [
    (iid, id_to_file[iid], id_to_caption[iid])
    for iid in id_to_caption
    if (image_dir / id_to_file[iid]).exists()
]
random.shuffle(valid)
selected = valid[:args.n_images]

images = []
for iid, fname, caption in selected:
    pil = Image.open(image_dir / fname).convert("RGB").resize((H, W))
    images.append({"pil": pil, "fname": fname, "gt_caption": caption})

# ── Caption helper ─────────────────────────────────────────────────────────────
QUESTION = "Describe this image in one sentence."

def get_caption(pil_image, perturbation=None):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text",  "text": QUESTION},
    ]}]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt").to(device)
    inputs.pop("token_type_ids", None)
    if perturbation is not None:
        inputs["pixel_values"] = pil_to_patches(pil_image, perturbation.to(torch.bfloat16))
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(
        gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

# ── Generate clean captions ────────────────────────────────────────────────────
print("Generating clean captions...")
for item in images:
    item["clean_caption"] = get_caption(item["pil"])
    print(f"  {item['fname']}: {item['clean_caption']}")

# ── Verify pivot token ─────────────────────────────────────────────────────────
token_ids = processor.tokenizer.encode(args.pivot_token, add_special_tokens=False)
assert len(token_ids) == 1, f"'{args.pivot_token}' is multi-token: {token_ids}"
pivot_id = torch.tensor(token_ids[0], device=device)
print(f"\nPivot token: '{args.pivot_token}' (id={pivot_id.item()})")

# ── Train perturbation ─────────────────────────────────────────────────────────
print(f"\nTraining perturbation for '{args.pivot_token}'...")
perturbation = torch.zeros(3, H, W, dtype=torch.float32,
                           device=device, requires_grad=True)
optimizer = torch.optim.Adam([perturbation], lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
)

for epoch in range(args.epochs):
    epoch_loss = 0.0
    random.shuffle(images)
    optimizer.zero_grad()

    for item in images:
        pil = item["pil"]
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text",  "text": QUESTION},
        ]}]
        text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil], return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)
        inputs["pixel_values"] = pil_to_patches(pil, perturbation.to(torch.bfloat16))

        outputs      = model(**inputs)
        first_logits = outputs.logits[0, -1, :]
        logit_target = first_logits[pivot_id]
        competing    = first_logits.clone()
        competing[pivot_id] = -1e9
        cw_loss = torch.clamp(competing.max() - logit_target + args.cw_margin, min=0.0)

        loss = cw_loss / len(images)
        epoch_loss += cw_loss.item()
        loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        perturbation.clamp_(-args.epsilon, args.epsilon)
    scheduler.step()

    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/{args.epochs}  loss: {epoch_loss/len(images):.4f}  "
              f"L-inf: {perturbation.abs().max().item():.4f}")

pert_detached = perturbation.detach()
print(f"\nFinal L-inf: {pert_detached.abs().max().item():.4f}")

# ── Save images ────────────────────────────────────────────────────────────────
out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# Save the raw perturbation amplified x10 so you can see the noise pattern
pert_cpu    = pert_detached.cpu().float()
pert_amp    = (pert_cpu * 10 + 0.5).clamp(0, 1)   # amplify x10, centre at 0.5
pert_pil    = TF.to_pil_image(pert_amp)
pert_pil.save(out_dir / f"perturbation_{args.pivot_token}_10x.png")
print(f"Saved amplified perturbation → {out_dir}/perturbation_{args.pivot_token}_10x.png")

# Save per-image comparisons
html_rows = []
for i, item in enumerate(images):
    # Clean image
    clean_path = out_dir / f"{i:02d}_clean.png"
    item["pil"].save(clean_path)

    # Perturbed image (what the model actually sees)
    img_tensor  = TF.to_tensor(item["pil"]).float()
    pert_added  = (img_tensor + pert_cpu).clamp(0, 1)
    perturbed_pil = TF.to_pil_image(pert_added)
    attacked_path = out_dir / f"{i:02d}_attacked.png"
    perturbed_pil.save(attacked_path)

    # Difference image amplified x10
    diff        = (pert_added - img_tensor).abs()
    diff_amp    = (diff * 10).clamp(0, 1)
    diff_pil    = TF.to_pil_image(diff_amp)
    diff_path   = out_dir / f"{i:02d}_diff_10x.png"
    diff_pil.save(diff_path)

    # Generate attacked caption
    attacked_caption = get_caption(item["pil"], pert_detached)
    item["attacked_caption"] = attacked_caption

    print(f"\n[{i}] {item['fname']}")
    print(f"  Clean:   {item['clean_caption']}")
    print(f"  Attacked:{attacked_caption}")
    starts_with_pivot = attacked_caption.lower().strip().startswith(args.pivot_token.lower())
    print(f"  Starts with '{args.pivot_token}': {starts_with_pivot}")

    html_rows.append({
        "clean_path":    clean_path.name,
        "attacked_path": attacked_path.name,
        "diff_path":     diff_path.name,
        "clean_caption": item["clean_caption"],
        "attacked_caption": attacked_caption,
        "fname":         item["fname"],
        "starts_pivot":  starts_with_pivot,
    })

# ── Save HTML report ───────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Leverage Visualisation — pivot: {args.pivot_token}</title>
<style>
  body {{ font-family: Arial, sans-serif; background: #f5f5f5; padding: 20px; }}
  h1 {{ color: #2C5F8A; }}
  .meta {{ background: #e8f0f8; padding: 10px; border-radius: 6px; margin-bottom: 20px; font-size: 13px; }}
  .row {{ background: white; border-radius: 8px; padding: 16px; margin-bottom: 20px;
          box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  .imgs {{ display: flex; gap: 16px; margin-bottom: 12px; }}
  .imgs figure {{ margin: 0; text-align: center; }}
  .imgs img {{ width: 200px; height: 200px; object-fit: cover; border-radius: 4px;
               border: 2px solid #ddd; }}
  .caption {{ font-size: 13px; margin-top: 6px; color: #555; max-width: 200px; }}
  .label {{ font-weight: bold; font-size: 12px; color: #888; text-transform: uppercase; }}
  .clean-cap {{ color: #2C5F8A; font-size: 13px; margin: 4px 0; }}
  .attacked-cap {{ color: #c0392b; font-size: 13px; margin: 4px 0; }}
  .tag-ok  {{ background: #27ae60; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; }}
  .tag-no  {{ background: #e74c3c; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; }}
</style>
</head>
<body>
<h1>Perturbation Visualisation</h1>
<div class="meta">
  <b>Pivot token:</b> {args.pivot_token} &nbsp;|&nbsp;
  <b>Epsilon:</b> {args.epsilon:.3f} ({round(args.epsilon*255)}/255) &nbsp;|&nbsp;
  <b>Epochs:</b> {args.epochs} &nbsp;|&nbsp;
  <b>Images:</b> {args.n_images}
</div>
<p><b>Amplified perturbation (10x):</b><br>
<img src="perturbation_{args.pivot_token}_10x.png" style="width:200px;height:200px;border-radius:4px;border:2px solid #ddd;"></p>
"""

for i, r in enumerate(html_rows):
    tag = f'<span class="tag-ok">✓ starts with "{args.pivot_token}"</span>' \
          if r["starts_pivot"] else f'<span class="tag-no">✗ does not start with "{args.pivot_token}"</span>'
    html += f"""
<div class="row">
  <div class="label">Image {i} — {r['fname']} &nbsp; {tag}</div>
  <div class="imgs">
    <figure>
      <img src="{r['clean_path']}">
      <div class="caption">Clean</div>
    </figure>
    <figure>
      <img src="{r['attacked_path']}">
      <div class="caption">Attacked (what model sees)</div>
    </figure>
    <figure>
      <img src="{r['diff_path']}">
      <div class="caption">Difference ×10</div>
    </figure>
  </div>
  <p class="clean-cap"><b>Clean caption:</b> {r['clean_caption']}</p>
  <p class="attacked-cap"><b>Attacked caption:</b> {r['attacked_caption']}</p>
</div>
"""

html += "</body></html>"
html_path = out_dir / f"report_{args.pivot_token}.html"
html_path.write_text(html)
print(f"\nHTML report saved → {html_path}")
print(f"\nAll files in {out_dir}:")
for f in sorted(out_dir.iterdir()):
    print(f"  {f.name}")
