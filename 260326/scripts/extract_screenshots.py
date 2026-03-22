import base64
import re
import argparse
from pathlib import Path
from PIL import Image
import io

parser = argparse.ArgumentParser()
parser.add_argument("--traj_dir", type=str, default="vwa_repo/trajectories")
parser.add_argument("--out_dir",  type=str, default="vwa_screenshots")
parser.add_argument("--max",      type=int, default=2000)
args = parser.parse_args()

traj_dir = Path(args.traj_dir)
out_dir  = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

html_files = sorted(traj_dir.glob("**/*.html"))
print(f"Found {len(html_files)} HTML files in {traj_dir}")

count   = 0
skipped = 0

for html_file in html_files:
    if count >= args.max:
        break
    try:
        content = html_file.read_text(errors="ignore")
    except Exception:
        continue

    matches = re.findall(
        r'data:image/png;base64,([A-Za-z0-9+/=]{1000,})',
        content
    )

    for i, b64_str in enumerate(matches):
        if count >= args.max:
            break
        try:
            img_bytes = base64.b64decode(b64_str)
            img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if img.width < 400 or img.height < 300:
                skipped += 1
                continue
            out_path = out_dir / f"{html_file.stem}_{i:03d}.png"
            img.save(out_path, "PNG", optimize=True)
            count += 1
            if count % 100 == 0:
                print(f"  Saved {count} screenshots...")
        except Exception:
            skipped += 1
            continue

print(f"\nDone. Saved: {count} | Skipped: {skipped}")
print(f"Screenshots saved to: {out_dir.resolve()}")

sample = list(out_dir.glob("*.png"))[:5]
print(f"\nSample files:")
for p in sample:
    img = Image.open(p)
    print(f"  {p.name} — {img.size[0]}x{img.size[1]}")
