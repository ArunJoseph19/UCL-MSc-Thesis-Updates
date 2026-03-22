import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import os
import io
import base64
import anthropic
import openai
from google import genai


# ------------------------------------------------------------------
# API KEYS
# ------------------------------------------------------------------
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
CLAUDE_API_KEY = os.environ["CLAUDE_API_KEY"]


gemini_client = genai.Client(api_key=GEMINI_API_KEY)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)


# ------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------
device = torch.device("cuda")
H, W   = 448, 448

MODEL_3B = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"

OUT_QWEN_3B = "attack_results_3b"
OUT_QWEN_7B = "attack_results_7b"
OUT_GEMINI  = "attack_results_gemini"
OUT_OPENAI  = "attack_results_openai"
OUT_CLAUDE  = "attack_results_claude"
for d in [OUT_QWEN_3B, OUT_QWEN_7B, OUT_GEMINI, OUT_OPENAI, OUT_CLAUDE]:
    os.makedirs(d, exist_ok=True)

QUESTION = "What animal is in this picture? Answer in one word."


print("Loading Qwen 3B model...")
processor_3b = AutoProcessor.from_pretrained(MODEL_3B)
model_3b     = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_3B, torch_dtype=torch.bfloat16).to(device)
model_3b.eval()

print("Loading Qwen 7B model...")
processor_7b = AutoProcessor.from_pretrained(MODEL_7B)
model_7b     = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_7B, torch_dtype=torch.bfloat16).to(device)
model_7b.eval()


perturbation = torch.load("qwen_universal_dog_perturbation_best.pt").to(device=device, dtype=torch.bfloat16)
print(f"Perturbation loaded. L-inf norm: {perturbation.abs().max().item():.4f}")


print("Loading dataset...")
dataset = load_dataset("Bingsu/Cat_and_Dog", split="test")
cats    = dataset.filter(lambda ex: ex['labels'] == 0)
print(f"Evaluating on {len(cats)} held-out test cats")

dog_token_id_3b = processor_3b.tokenizer.encode("dog", add_special_tokens=False)[0]
dog_token_id_7b = processor_7b.tokenizer.encode("dog", add_special_tokens=False)[0]


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def pil_to_base64(pil_image, fmt="JPEG"):
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_image_with_caption(pil_image, caption, filepath, bg_color):
    bar_h   = 54
    out_img = Image.new("RGB", (pil_image.width, pil_image.height + bar_h), color=bg_color)
    out_img.paste(pil_image, (0, 0))
    draw = ImageDraw.Draw(out_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    bbox   = draw.textbbox((0, 0), caption, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx     = (pil_image.width - tw) // 2
    ty     = pil_image.height + (bar_h - th) // 2
    draw.text((tx, ty), caption, fill=(255, 255, 255), font=font)
    out_img.save(filepath)


def save_result(pil_image, reply, out_dir, idx, label=""):
    is_dog  = "dog" in reply.lower()
    bg      = (34, 139, 34) if is_dog else (180, 50, 50)
    tag     = "DOG" if is_dog else reply[:10].replace(" ", "_").upper()
    fname   = f"{idx:04d}_{label}_{tag}.jpg"
    caption = f'pred: "{reply}"'
    save_image_with_caption(pil_image, caption, os.path.join(out_dir, fname), bg)
    return is_dog


def already_tested(out_dir, idx):
    """Returns True if any file for this image index already exists in out_dir."""
    prefix = f"{idx:04d}_"
    return any(f.startswith(prefix) for f in os.listdir(out_dir))


# ------------------------------------------------------------------
# QWEN INFERENCE (shared function, pass model + processor + dog_token)
# ------------------------------------------------------------------
def qwen_reply_and_prob(pil_image, model, processor, dog_token_id):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text",  "text": QUESTION}
    ]}]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[pil_image], return_tensors="pt").to(device)

    with torch.no_grad():
        out          = model(**inputs)
        first_logits = out.logits[0, -1, :]
        probs        = F.softmax(first_logits.float(), dim=-1)
        dog_prob     = probs[dog_token_id].item()
        top_id       = probs.argmax().item()
        top_prob     = probs[top_id].item()
        top_token    = processor.tokenizer.decode([top_id]).strip()
        gen          = model.generate(**inputs, max_new_tokens=10)
        reply        = processor.decode(
            gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

    return reply, dog_prob, top_token, top_prob


# ------------------------------------------------------------------
# GEMINI INFERENCE
# ------------------------------------------------------------------
def gemini_reply(pil_image):
    try:
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG")
        buf.seek(0)
        img_part = genai.types.Part.from_bytes(data=buf.read(), mime_type="image/jpeg")
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[img_part, QUESTION]
        )
        return response.text.strip()
    except Exception as e:
        return f"ERROR: {e}"


# ------------------------------------------------------------------
# OPENAI INFERENCE
# ------------------------------------------------------------------
def openai_reply(pil_image):
    try:
        b64 = pil_to_base64(pil_image)
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": QUESTION}
                ]
            }],
            max_tokens=20
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


# ------------------------------------------------------------------
# CLAUDE INFERENCE
# ------------------------------------------------------------------
def claude_reply(pil_image):
    try:
        b64 = pil_to_base64(pil_image)
        response = claude_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=20,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": QUESTION}
                ]
            }]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"ERROR: {e}"


# ------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------
total              = 0
qwen_3b_success    = 0
qwen_7b_success    = 0
either_success     = 0
cross_model_fooled = {"gemini": 0, "openai": 0, "claude": 0}


for data in tqdm(cats, desc="Evaluating"):
    raw   = data['image'].convert("RGB").resize((H, W))
    img_t = TF.to_tensor(raw).to(device, dtype=torch.bfloat16)

    perturbed_pil = TF.to_pil_image((img_t + perturbation).clamp(0, 1).float().cpu())
    total += 1

    # ---- Qwen 3B ----
    r3, dp3, tt3, tp3 = qwen_reply_and_prob(perturbed_pil, model_3b, processor_3b, dog_token_id_3b)
    hit_3b = "dog" in r3.lower()
    qwen_3b_success += hit_3b

    caption_3b = f'pred: "{r3}"  |  P(dog)={dp3*100:.1f}%  |  top: "{tt3}" {tp3*100:.1f}%'
    bg_3b      = (34, 139, 34) if hit_3b else (180, 50, 50)
    fname_3b   = f"{total:04d}_{'DOG' if hit_3b else r3[:10].replace(' ', '_').upper()}.jpg"
    save_image_with_caption(perturbed_pil, caption_3b, os.path.join(OUT_QWEN_3B, fname_3b), bg_3b)

    # ---- Qwen 7B ----
    r7, dp7, tt7, tp7 = qwen_reply_and_prob(perturbed_pil, model_7b, processor_7b, dog_token_id_7b)
    hit_7b = "dog" in r7.lower()
    qwen_7b_success += hit_7b

    caption_7b = f'pred: "{r7}"  |  P(dog)={dp7*100:.1f}%  |  top: "{tt7}" {tp7*100:.1f}%'
    bg_7b      = (34, 139, 34) if hit_7b else (180, 50, 50)
    fname_7b   = f"{total:04d}_{'DOG' if hit_7b else r7[:10].replace(' ', '_').upper()}.jpg"
    save_image_with_caption(perturbed_pil, caption_7b, os.path.join(OUT_QWEN_7B, fname_7b), bg_7b)

    tqdm.write(f"[{total:4d}] 3B: '{r3}' P(dog)={dp3*100:.1f}% | 7B: '{r7}' P(dog)={dp7*100:.1f}% | ", end="")

    # ---- Cross-model eval: only when at least one Qwen model succeeded ----
    if hit_3b or hit_7b:
        either_success += 1

        # Skip APIs that already have a result for this image index
        g_reply = "SKIPPED"
        o_reply = "SKIPPED"
        c_reply = "SKIPPED"

        if not already_tested(OUT_GEMINI, total):
            g_reply = gemini_reply(perturbed_pil)
            if "dog" in g_reply.lower():
                cross_model_fooled["gemini"] += 1
            save_result(perturbed_pil, g_reply, OUT_GEMINI, total, "GEMINI")

        if not already_tested(OUT_OPENAI, total):
            o_reply = openai_reply(perturbed_pil)
            if "dog" in o_reply.lower():
                cross_model_fooled["openai"] += 1
            save_result(perturbed_pil, o_reply, OUT_OPENAI, total, "OPENAI")

        if not already_tested(OUT_CLAUDE, total):
            c_reply = claude_reply(perturbed_pil)
            if "dog" in c_reply.lower():
                cross_model_fooled["claude"] += 1
            save_result(perturbed_pil, c_reply, OUT_CLAUDE, total, "CLAUDE")

        tqdm.write(f"Gemini: '{g_reply}' | OpenAI: '{o_reply}' | Claude: '{c_reply}'")
    else:
        tqdm.write("")


print(f"\n{'='*60}")
print(f"Total evaluated                      : {total}")
print(f"Qwen 3B attack success (→ dog)       : {qwen_3b_success}/{total} = {qwen_3b_success/total*100:.1f}%")
print(f"Qwen 7B attack success (→ dog)       : {qwen_7b_success}/{total} = {qwen_7b_success/total*100:.1f}%")
print(f"Either Qwen fooled                   : {either_success}/{total} = {either_success/total*100:.1f}%")
print(f"\n-- Cross-model transferability on {either_success} images where either Qwen succeeded --")
for name, count in cross_model_fooled.items():
    print(f"  {name.capitalize():8s} also fooled (→ dog) : {count}/{either_success} = {count/max(either_success,1)*100:.1f}%")
print(f"\nImages saved to:")
print(f"  Qwen 3B → ./{OUT_QWEN_3B}/")
print(f"  Qwen 7B → ./{OUT_QWEN_7B}/")
print(f"  Gemini  → ./{OUT_GEMINI}/")
print(f"  OpenAI  → ./{OUT_OPENAI}/")
print(f"  Claude  → ./{OUT_CLAUDE}/")
