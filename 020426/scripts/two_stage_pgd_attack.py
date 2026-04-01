"""
two_stage_pgd_attack.py
-----------------------
Two-stage PGD captioning attack on Qwen2.5-VL-7B.

Adapted from UMK (Wang et al., ACM MM 2024, arxiv:2405.17894) and
SEA (arxiv:2508.01741) for targeted image captioning rather than jailbreaking.

STAGE 1 — Semantic injection (no text prompt)
    Show the model ONLY the perturbed image, no question.
    Optimise the perturbation so the model's free generation
    matches your target caption even with zero text context.
    Loss: cross-entropy over target caption tokens, image-only input.

STAGE 2 — Output forcing (with caption question)
    Warm-start from Stage 1 perturbation.
    Now add the actual caption question to the input.
    Fine-tune the perturbation so the model answers the question
    with your target caption.
    Loss: cross-entropy over target caption tokens, image+question input.

Optimiser: PGD (project after every step) — NOT Adam.
    PGD takes one step in the gradient direction then clips to epsilon ball.
    This is more aggressive than Adam and converges faster per-image.

Usage:
    HF_TOKEN=xxx python scripts/two_stage_pgd_attack.py \\
        --image_dir coco/images/val2017 \\
        --captions_file coco/annotations/captions_val2017.json \\
        --n_images 5 \\
        --target_caption "A dog sitting on a sofa." \\
        --output_dir results/two_stage_pgd \\
        --epsilon 0.063 \\
        --stage1_steps 500 \\
        --stage2_steps 500 \\
        --step_size 0.004

Key hyperparameters:
    --epsilon       L-inf budget. 0.063 = 16/255. Start here.
                    Try 0.031 (8/255) for more imperceptible.
                    Try 0.125 (32/255) if convergence is slow.
    --stage1_steps  PGD steps for semantic injection. 500 is standard.
    --stage2_steps  PGD steps for output forcing. 500 is standard.
    --step_size     PGD step size. Rule of thumb: epsilon / 40.
                    For epsilon=0.063: step_size=0.004 (16/255 / 40 ≈ 0.004).
"""

import os
import json
import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from huggingface_hub import login

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir",      default="coco/images/val2017")
parser.add_argument("--captions_file",  default="coco/annotations/captions_val2017.json")
parser.add_argument("--n_images",       type=int,   default=5,
                    help="Number of COCO images to attack")
parser.add_argument("--target_caption", default="A dog sitting on a sofa.",
                    help="The caption the model should output instead of the truth")
parser.add_argument("--output_dir",     default="results/two_stage_pgd")
parser.add_argument("--epsilon",        type=float, default=16/255,
                    help="L-inf perturbation budget (default 16/255 ≈ 0.063)")
parser.add_argument("--stage1_steps",   type=int,   default=500,
                    help="PGD steps for stage 1 (semantic injection, no text)")
parser.add_argument("--stage2_steps",   type=int,   default=500,
                    help="PGD steps for stage 2 (output forcing, with question)")
parser.add_argument("--step_size",      type=float, default=4/255,
                    help="PGD step size per iteration (default 4/255 ≈ 0.016)")
parser.add_argument("--log_every",      type=int,   default=50,
                    help="Print loss every N steps")
parser.add_argument("--seed",           type=int,   default=42)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

# ── Model setup ───────────────────────────────────────────────────────────────
login(token=os.environ["HF_TOKEN"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Epsilon: {args.epsilon:.4f} ({round(args.epsilon*255)}/255)")
print(f"Step size: {args.step_size:.4f} ({round(args.step_size*255)}/255)")
print(f"Stage 1 steps: {args.stage1_steps} | Stage 2 steps: {args.stage2_steps}")
print(f"Target caption: '{args.target_caption}'")

H, W = 448, 448
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

print(f"\nLoading {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16
).to(device).eval()

# Freeze all model parameters — we only optimise the pixel perturbation
for p in model.parameters():
    p.requires_grad = False

# Image normalisation constants from Qwen's image processor
img_proc = processor.image_processor
mean = torch.tensor(img_proc.image_mean, dtype=torch.float32, device=device).view(3, 1, 1)
std  = torch.tensor(img_proc.image_std,  dtype=torch.float32, device=device).view(3, 1, 1)

print("Model loaded.\n")

# ── Core preprocessing: pil_to_patches ───────────────────────────────────────
# This replicates Qwen's internal image pipeline differentiably.
# Critically this must match training exactly — wrong patch size or
# normalisation order means gradients flow to the wrong pixel values.
#
# Qwen2.5-VL uses:
#   patch_size = 14 (ViT-patch14)
#   temporal_patch_size = 2 (two temporal frames per image)
#   image processed to 448x448 grid = 32x32 patches = 1024 patch tokens

def pil_to_patches(pil_image, perturbation, patch_size=14, temporal_patch_size=2):
    """
    Apply perturbation to a PIL image and convert to Qwen patch format.

    Steps:
    1. Convert PIL to float32 tensor [3, H, W] in [0, 1]
    2. Add perturbation and clamp to valid pixel range [0, 1]
    3. Normalise with Qwen's mean/std
    4. Expand to temporal dimension (2 frames, identical content)
    5. Reshape into patch tokens expected by Qwen's vision encoder

    Returns:
        pixel_values: [n_patches, temporal * C * patch_h * patch_w] as bfloat16
    """
    # Convert to float32 for gradient stability
    img_tensor = TF.to_tensor(pil_image).to(device=device, dtype=torch.float32)

    # Add perturbation and clamp — this is where gradients flow through
    perturbed  = (img_tensor + perturbation).clamp(0.0, 1.0)

    # Normalise with Qwen's ImageNet-style mean/std
    normalized = (perturbed - mean) / std

    # Expand to 2 temporal frames (Qwen requires this even for single images)
    img = normalized.unsqueeze(0).repeat(temporal_patch_size, 1, 1, 1)
    # img shape: [2, 3, 448, 448]

    t, c, h, w = img.shape
    ph = h // patch_size   # = 32
    pw = w // patch_size   # = 32

    # Reshape into patch tokens
    img = img.reshape(t, c, ph, patch_size, pw, patch_size)
    img = img.permute(2, 4, 0, 1, 3, 5).contiguous()
    # img shape: [32, 32, 2, 3, 14, 14]

    img = img.reshape(ph * pw, t * c * patch_size * patch_size)
    # img shape: [1024, 2*3*14*14] = [1024, 1176]

    # Qwen expects bfloat16 for the vision encoder
    return img.to(torch.bfloat16)


# ── Loss function ─────────────────────────────────────────────────────────────
def compute_ce_loss(outputs, target_token_ids):
    """
    Compute cross-entropy loss over target caption tokens.

    The model outputs logits of shape [1, seq_len, vocab_size].
    We want the loss over only the target caption tokens, not the
    prompt tokens.

    The way teacher-forcing works in an autoregressive model:
    - The model sees [prompt_tokens ... target_token_1 target_token_2 ...]
    - At each position, it predicts the NEXT token
    - So logits[:, -(T+1):-1, :] predicts positions of target tokens
    - And target_token_ids are the labels at those positions

    Args:
        outputs:          model output from a forward pass
        target_token_ids: tensor of shape [n_target_tokens] — the tokenised target

    Returns:
        scalar CE loss averaged over target token positions
    """
    logits = outputs.logits  # [1, full_seq_len, vocab_size]
    T = len(target_token_ids)

    # Extract logits at positions that predict each target token
    # Position -(T+1) predicts target_token_ids[0],
    # Position -T     predicts target_token_ids[1], etc.
    # Using -(T+1):-1 selects exactly the T prediction positions
    target_logits = logits[0, -(T + 1):-1, :]
    # target_logits shape: [T, vocab_size]

    target_ids = target_token_ids.to(device)
    loss = F.cross_entropy(target_logits, target_ids)
    return loss


# ── PGD step ──────────────────────────────────────────────────────────────────
def pgd_step(perturbation, grad, step_size, epsilon):
    """
    One PGD update:
    1. Step in the negative gradient direction (minimise CE loss)
    2. Project back onto the epsilon ball (L-inf constraint)
    3. Return updated perturbation (detached, no gradient history)

    PGD is different from Adam:
    - No momentum, no adaptive learning rate
    - Hard clamp to epsilon after every step
    - Faster convergence for per-image attacks

    Args:
        perturbation: current perturbation tensor [3, H, W]
        grad:         gradient of loss w.r.t. perturbation
        step_size:    PGD step size alpha
        epsilon:      L-inf budget

    Returns:
        new perturbation tensor [3, H, W], requires_grad=True
    """
    with torch.no_grad():
        # Take a step in the negative gradient direction
        # (we're minimising CE loss, so we move opposite to gradient)
        new_pert = perturbation - step_size * grad.sign()

        # Project onto L-inf epsilon ball
        new_pert = new_pert.clamp(-epsilon, epsilon)

    # Re-attach gradient tracking for next iteration
    new_pert = new_pert.detach().requires_grad_(True)
    return new_pert


# ── Stage 1: Semantic injection (no text input) ───────────────────────────────
def run_stage1(pil_image, target_token_ids):
    """
    Stage 1: Force the target caption using image-only input.

    The model receives NO text question — just the perturbed image.
    This forces semantic content into the visual features directly.
    The language model prior has nothing to anchor to, so the visual
    tokens must carry the entire target meaning.

    Why this works: if the perturbation succeeds here, the image
    LOOKS like the target caption to the model's vision encoder.
    Stage 2 then only needs to refine this for the specific question format.

    Args:
        pil_image:        PIL image resized to 448x448
        target_token_ids: tensor of tokenised target caption

    Returns:
        perturbation: trained stage 1 perturbation [3, H, W]
        losses:       list of loss values per step
    """
    print("  [Stage 1] Semantic injection — image only, no question")

    # Initialise perturbation at zero (clean image)
    perturbation = torch.zeros(3, H, W, dtype=torch.float32,
                               device=device, requires_grad=True)

    # Stage 1 prompt: no question, just a minimal chat template
    # We pass an empty user message so the model generates freely
    messages_stage1 = [
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            # Empty text — model must rely entirely on visual features
            {"type": "text", "text": ""},
        ]}
    ]
    text_stage1 = processor.apply_chat_template(
        messages_stage1, tokenize=False, add_generation_prompt=True
    )

    losses = []
    t0 = time.time()

    for step in range(args.stage1_steps):
        # Build inputs with current perturbation injected
        inputs = processor(
            text=[text_stage1], images=[pil_image],
            return_tensors="pt"
        ).to(device)
        inputs.pop("token_type_ids", None)

        # Inject perturbation into pixel values via our differentiable pipeline
        inputs["pixel_values"] = pil_to_patches(pil_image, perturbation)

        # Forward pass through full model (vision encoder → projector → LLM)
        # The gradient will flow back through pil_to_patches to perturbation
        outputs = model(**inputs)

        # CE loss over target caption tokens
        loss = compute_ce_loss(outputs, target_token_ids)
        losses.append(loss.item())

        # Compute gradient of loss w.r.t. perturbation
        loss.backward()

        # PGD step: move in gradient direction, project to epsilon ball
        perturbation = pgd_step(
            perturbation, perturbation.grad, args.step_size, args.epsilon
        )

        if (step + 1) % args.log_every == 0:
            elapsed = (time.time() - t0) / 60
            print(f"    Step {step+1:4d}/{args.stage1_steps} | "
                  f"Loss: {loss.item():.4f} | "
                  f"L-inf: {perturbation.abs().max().item():.4f} | "
                  f"Time: {elapsed:.1f}min")

    print(f"  Stage 1 complete. Final loss: {losses[-1]:.4f}")
    return perturbation.detach(), losses


# ── Stage 2: Output forcing (with caption question) ───────────────────────────
def run_stage2(pil_image, perturbation_stage1, target_token_ids):
    """
    Stage 2: Refine the perturbation with the actual caption question.

    Warm-starts from Stage 1 perturbation. Now adds the real question
    ("Describe this image in one sentence.") to the input.

    The model now has both visual and textual context, so it's closer
    to real inference conditions. This stage forces the model to
    answer the specific question with the target caption.

    Args:
        pil_image:            PIL image resized to 448x448
        perturbation_stage1:  trained stage 1 perturbation [3, H, W]
        target_token_ids:     tensor of tokenised target caption

    Returns:
        perturbation: trained stage 2 perturbation [3, H, W]
        losses:       list of loss values per step
    """
    print("  [Stage 2] Output forcing — image + caption question")

    # Warm-start from stage 1 result
    perturbation = perturbation_stage1.clone().requires_grad_(True)

    # Stage 2 prompt: actual captioning question
    CAPTION_QUESTION = "Describe this image in one sentence."
    messages_stage2 = [
        {"role": "user", "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": CAPTION_QUESTION},
        ]},
        # Teacher-force the target caption as the assistant response
        {"role": "assistant", "content": args.target_caption},
    ]
    text_stage2 = processor.apply_chat_template(
        messages_stage2, tokenize=False, add_generation_prompt=False
    )

    losses = []
    t0 = time.time()

    for step in range(args.stage2_steps):
        inputs = processor(
            text=[text_stage2], images=[pil_image],
            return_tensors="pt"
        ).to(device)
        inputs.pop("token_type_ids", None)
        inputs["pixel_values"] = pil_to_patches(pil_image, perturbation)

        outputs = model(**inputs)
        loss = compute_ce_loss(outputs, target_token_ids)
        losses.append(loss.item())

        loss.backward()
        perturbation = pgd_step(
            perturbation, perturbation.grad, args.step_size, args.epsilon
        )

        if (step + 1) % args.log_every == 0:
            elapsed = (time.time() - t0) / 60
            print(f"    Step {step+1:4d}/{args.stage2_steps} | "
                  f"Loss: {loss.item():.4f} | "
                  f"L-inf: {perturbation.abs().max().item():.4f} | "
                  f"Time: {elapsed:.1f}min")

    print(f"  Stage 2 complete. Final loss: {losses[-1]:.4f}")
    return perturbation.detach(), losses


# ── Caption generation ────────────────────────────────────────────────────────
def generate_caption(pil_image, perturbation=None, max_new_tokens=80):
    """
    Generate a free-form caption from the model.
    If perturbation is given, inject it. Otherwise use clean image.
    """
    CAPTION_QUESTION = "Describe this image in one sentence."
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text",  "text": CAPTION_QUESTION},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt").to(device)
    inputs.pop("token_type_ids", None)

    if perturbation is not None:
        inputs["pixel_values"] = pil_to_patches(pil_image, perturbation)

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # Greedy decode — deterministic
        )

    # Decode only the newly generated tokens (not the prompt)
    new_tokens = gen[0][inputs.input_ids.shape[1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Save visualisation images ─────────────────────────────────────────────────
def save_comparison(pil_image, perturbation, out_dir, idx):
    """
    Save: clean image, attacked image, and amplified difference.
    The difference image x10 lets you see the noise pattern.
    If attacked image looks visually corrupted to human eye,
    epsilon is too large.
    """
    clean_path   = out_dir / f"{idx:02d}_clean.png"
    attacked_path= out_dir / f"{idx:02d}_attacked.png"
    diff_path    = out_dir / f"{idx:02d}_diff_10x.png"

    pil_image.save(clean_path)

    img_t    = TF.to_tensor(pil_image).float()
    pert_cpu = perturbation.cpu().float()
    attacked = (img_t + pert_cpu).clamp(0, 1)
    TF.to_pil_image(attacked).save(attacked_path)

    diff = (attacked - img_t).abs()
    TF.to_pil_image((diff * 10).clamp(0, 1)).save(diff_path)

    return clean_path.name, attacked_path.name, diff_path.name


# ── Load COCO images ──────────────────────────────────────────────────────────
print("Loading COCO annotations...")
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
for iid, fname, gt_caption in selected:
    pil = Image.open(image_dir / fname).convert("RGB").resize((H, W))
    images.append({"pil": pil, "fname": fname, "gt_caption": gt_caption})

print(f"Loaded {len(images)} images.\n")

# ── Tokenise target caption ───────────────────────────────────────────────────
# We need the token IDs of the target caption for CE loss computation.
# add_special_tokens=False because the chat template adds them already.
target_token_ids = torch.tensor(
    processor.tokenizer.encode(args.target_caption, add_special_tokens=False),
    dtype=torch.long,
    device=device
)
print(f"Target caption: '{args.target_caption}'")
print(f"Target token IDs: {target_token_ids.tolist()} "
      f"({len(target_token_ids)} tokens)\n")

# ── Generate clean captions ───────────────────────────────────────────────────
print("Generating clean captions...")
for item in images:
    item["clean_caption"] = generate_caption(item["pil"])
    print(f"  [{item['fname']}] {item['clean_caption']}")
print()

# ── Main attack loop ──────────────────────────────────────────────────────────
out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

results = []
total_start = time.time()

for i, item in enumerate(images):
    print(f"{'='*60}")
    print(f"Image {i+1}/{len(images)}: {item['fname']}")
    print(f"  Clean caption:  {item['clean_caption']}")
    print(f"  Target caption: {args.target_caption}")
    print(f"{'='*60}")

    t_img_start = time.time()

    # ── Stage 1: semantic injection ──
    pert_s1, losses_s1 = run_stage1(item["pil"], target_token_ids)

    # Check what stage 1 produces (image-only, no question)
    caption_after_s1_noquestion = generate_caption(item["pil"], pert_s1)
    print(f"  After Stage 1 (no question): {caption_after_s1_noquestion}")

    # Also check with the caption question to see where we are
    caption_after_s1 = generate_caption(item["pil"], pert_s1)
    print(f"  After Stage 1 (with question): {caption_after_s1}")

    # ── Stage 2: output forcing ──
    pert_s2, losses_s2 = run_stage2(item["pil"], pert_s1, target_token_ids)

    # Final attacked caption
    caption_after_s2 = generate_caption(item["pil"], pert_s2)
    print(f"  After Stage 2 (final):        {caption_after_s2}")

    # ── Check success ──
    # Exact match: does output exactly equal target?
    exact_match = caption_after_s2.strip().lower() == args.target_caption.strip().lower()

    # Contains match: does output contain the key words of the target?
    target_words = set(args.target_caption.lower().split())
    output_words = set(caption_after_s2.lower().split())
    overlap = len(target_words & output_words) / max(len(target_words), 1)
    contains_match = overlap > 0.5  # >50% word overlap

    print(f"\n  Results:")
    print(f"    Exact match:    {'✓' if exact_match else '✗'}")
    print(f"    Word overlap:   {overlap*100:.0f}% ({'✓' if contains_match else '✗'})")
    print(f"    Stage 1 final loss: {losses_s1[-1]:.4f}")
    print(f"    Stage 2 final loss: {losses_s2[-1]:.4f}")
    print(f"    L-inf: {pert_s2.abs().max().item():.4f}")

    # ── Save images ──
    c, a, d = save_comparison(item["pil"], pert_s2, out_dir, i)

    # ── Save perturbation checkpoint ──
    pert_path = out_dir / f"{i:02d}_perturbation.pt"
    torch.save(pert_s2.cpu().to(torch.bfloat16), pert_path)

    elapsed_img = (time.time() - t_img_start) / 60
    print(f"    Time for this image: {elapsed_img:.1f}min")

    results.append({
        "idx": i,
        "fname": item["fname"],
        "gt_caption": item["gt_caption"],
        "clean_caption": item["clean_caption"],
        "after_s1_caption": caption_after_s1,
        "attacked_caption": caption_after_s2,
        "target_caption": args.target_caption,
        "exact_match": exact_match,
        "word_overlap": round(overlap, 3),
        "stage1_final_loss": round(losses_s1[-1], 4),
        "stage2_final_loss": round(losses_s2[-1], 4),
        "linf": round(pert_s2.abs().max().item(), 4),
        "time_min": round(elapsed_img, 2),
    })

# ── Save HTML report ──────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Two-Stage PGD Attack Results</title>
<style>
  body {{ font-family: Arial, sans-serif; background: #f5f5f5; padding: 24px; }}
  h1 {{ color: #2C5F8A; }}
  .meta {{ background: #e8f0f8; padding: 12px; border-radius: 6px;
           margin-bottom: 24px; font-size: 13px; line-height: 1.6; }}
  .card {{ background: white; border-radius: 8px; padding: 20px;
           margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  .imgs {{ display: flex; gap: 16px; margin: 12px 0; }}
  .imgs figure {{ margin: 0; text-align: center; }}
  .imgs img {{ width: 200px; height: 200px; object-fit: cover;
               border-radius: 4px; border: 2px solid #ddd; }}
  .caption {{ font-size: 12px; color: #666; max-width: 200px; margin-top: 4px; }}
  .gt     {{ color: #555; font-size: 13px; margin: 4px 0; }}
  .clean  {{ color: #2C5F8A; font-size: 13px; margin: 4px 0; }}
  .s1     {{ color: #8e44ad; font-size: 13px; margin: 4px 0; }}
  .target {{ color: #27ae60; font-size: 13px; margin: 4px 0; font-weight: bold; }}
  .result {{ color: #c0392b; font-size: 13px; margin: 4px 0; font-weight: bold; }}
  .tag-ok  {{ background: #27ae60; color: white; padding: 2px 8px;
              border-radius: 10px; font-size: 11px; margin-left: 6px; }}
  .tag-no  {{ background: #e74c3c; color: white; padding: 2px 8px;
              border-radius: 10px; font-size: 11px; margin-left: 6px; }}
  .stats  {{ font-size: 12px; color: #888; margin-top: 8px; }}
</style>
</head>
<body>
<h1>Two-Stage PGD Attack — Results</h1>
<div class="meta">
  <b>Target caption:</b> {args.target_caption}<br>
  <b>Epsilon:</b> {args.epsilon:.4f} ({round(args.epsilon*255)}/255) &nbsp;|&nbsp;
  <b>Step size:</b> {args.step_size:.4f} ({round(args.step_size*255)}/255)<br>
  <b>Stage 1 steps:</b> {args.stage1_steps} &nbsp;|&nbsp;
  <b>Stage 2 steps:</b> {args.stage2_steps}<br>
  <b>Images attacked:</b> {len(results)} &nbsp;|&nbsp;
  <b>Exact match ASR:</b> {sum(r['exact_match'] for r in results)}/{len(results)} &nbsp;|&nbsp;
  <b>Word overlap ASR (&gt;50%):</b>
    {sum(r['word_overlap'] > 0.5 for r in results)}/{len(results)}
</div>
"""

for r in results:
    exact_tag = '<span class="tag-ok">✓ exact match</span>' \
                if r["exact_match"] else '<span class="tag-no">✗ no exact match</span>'
    overlap_tag = f'<span class="tag-ok">✓ {r["word_overlap"]*100:.0f}% overlap</span>' \
                  if r["word_overlap"] > 0.5 \
                  else f'<span class="tag-no">✗ {r["word_overlap"]*100:.0f}% overlap</span>'

    html += f"""
<div class="card">
  <b>{r['fname']}</b> {exact_tag} {overlap_tag}
  <div class="imgs">
    <figure>
      <img src="{r['idx']:02d}_clean.png">
      <div class="caption">Clean</div>
    </figure>
    <figure>
      <img src="{r['idx']:02d}_attacked.png">
      <div class="caption">Attacked (what model sees)</div>
    </figure>
    <figure>
      <img src="{r['idx']:02d}_diff_10x.png">
      <div class="caption">Difference ×10</div>
    </figure>
  </div>
  <p class="gt"><b>Ground truth:</b> {r['gt_caption']}</p>
  <p class="clean"><b>Clean caption:</b> {r['clean_caption']}</p>
  <p class="s1"><b>After Stage 1:</b> {r['after_s1_caption']}</p>
  <p class="target"><b>Target:</b> {r['target_caption']}</p>
  <p class="result"><b>Final attacked:</b> {r['attacked_caption']}</p>
  <p class="stats">
    Stage 1 loss: {r['stage1_final_loss']} &nbsp;|&nbsp;
    Stage 2 loss: {r['stage2_final_loss']} &nbsp;|&nbsp;
    L-inf: {r['linf']} &nbsp;|&nbsp;
    Time: {r['time_min']}min
  </p>
</div>
"""

html += "</body></html>"
html_path = out_dir / "report.html"
html_path.write_text(html)

# ── Final summary ─────────────────────────────────────────────────────────────
total_time = (time.time() - total_start) / 60
exact_asr  = sum(r["exact_match"] for r in results)
overlap_asr= sum(r["word_overlap"] > 0.5 for r in results)

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Images attacked:        {len(results)}")
print(f"Exact match ASR:        {exact_asr}/{len(results)}")
print(f"Word overlap ASR (>50%): {overlap_asr}/{len(results)}")
print(f"Avg Stage 1 final loss: {sum(r['stage1_final_loss'] for r in results)/len(results):.4f}")
print(f"Avg Stage 2 final loss: {sum(r['stage2_final_loss'] for r in results)/len(results):.4f}")
print(f"Total time:             {total_time:.1f}min")
print(f"\nHTML report: {html_path}")
print(f"Perturbation checkpoints: {out_dir}/*.pt")

print("\nNOTES:")
print("  - If Stage 2 loss is near 0, attack almost certainly succeeded.")
print("  - If loss stalls above 5, try larger epsilon or more steps.")
print("  - Check attacked images — if visually corrupted, epsilon is too large.")
print("  - Per-image PGD converges much faster than universal UAP training.")
