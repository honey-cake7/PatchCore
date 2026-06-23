#!/bin/bash
# Download both PVTv2-B2 checkpoints and print their real key structure, so the
# backbone loader's prefix-stripping is validated against fact, not assumed.
#
#   - ImageNet PVTv2-B2  -> models/pvt_v2_b2.pth  (whai362/PVT release, direct)
#   - Polyp-PVT trained  -> models/PolypPVT.pth   (Google Drive folder, via gdown)
set -e
cd "$(dirname "$0")"
mkdir -p models

# 1. ImageNet PVTv2-B2 backbone — direct download
if [ ! -f models/pvt_v2_b2.pth ]; then
  wget -O models/pvt_v2_b2.pth \
    https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth
fi

# 2. Polyp-PVT fine-tuned checkpoint — Google Drive folder 1xC5Opwu5Afz4xiK5O9v4NnQOZY0A9-2j
if [ ! -f models/PolypPVT.pth ]; then
  pip install -q gdown
  gdown --folder 1xC5Opwu5Afz4xiK5O9v4NnQOZY0A9-2j -O models/polyp_pvt_dl || true
  find models/polyp_pvt_dl -name "*.pth" -exec cp {} models/PolypPVT.pth \; || true
fi

# 3. Inspect both checkpoints — container type, key prefixes, sample keys
python - <<'PY'
import collections
import torch

for path in ("models/pvt_v2_b2.pth", "models/PolypPVT.pth"):
    try:
        ck = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"\n== {path}: FAILED to load: {e}")
        continue
    print(f"\n== {path}  (type={type(ck).__name__})")
    if isinstance(ck, dict) and ck and all(
        not torch.is_tensor(v) for v in list(ck.values())[:3]
    ):
        print("  top-level keys:", list(ck.keys())[:10])
        for k in ("model", "state_dict"):
            if k in ck:
                ck = ck[k]
                print(f"  -> unwrapped '{k}'")
                break
    prefixes = collections.Counter(k.split(".")[0] for k in ck.keys())
    print("  #keys:", len(ck), "| top-level prefixes:", dict(prefixes))
    print("  sample:", list(ck.keys())[:4])
PY
