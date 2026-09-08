---
license: apache-2.0
pipeline_tag: tabular-regression
tags:
  - tabular-regression
  - tabular-foundation-model
  - in-context-learning
  - tabdpt
base_model: Layer6/TabDPT
---

# TabDPT v1.2 Foundation Model Weights

This directory documents and stages the model weights for TabDPT v1.2 (TabDPT-Turbo).

## Weight Artifact Details

- **Filename:** `tabdpt1_2.safetensors`
- **File size:** 254,098,072 bytes
- **SHA-256:** `06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd`
- **Upstream Hugging Face Repository:** [`Layer6/TabDPT`](https://huggingface.co/Layer6/TabDPT)
- **Pinned Revision:** `4462ffbd1d8dea25d4862d30beed4b70cd596ae5`
- **Upstream Commit:** `9cfb05e0a6bc380ae6c99c08adc8d50dacd4f246`
- **License:** Apache-2.0

## Offline Staging Instructions

In air-gapped or network-restricted environments, stage the verified safetensors file locally or mount it into the DIMER container:

```bash
# Option 1: Download via huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Layer6/TabDPT', filename='tabdpt1_2.safetensors', revision='4462ffbd1d8dea25d4862d30beed4b70cd596ae5', local_dir='weights')"

# Option 2: Direct URL download
curl -L -o weights/tabdpt1_2.safetensors https://huggingface.co/Layer6/TabDPT/resolve/4462ffbd1d8dea25d4862d30beed4b70cd596ae5/tabdpt1_2.safetensors
```

## Integrity Verification

Verify the SHA-256 digest before running inference or baking into images:

```python
import hashlib
from pathlib import Path

EXPECTED_SHA256 = "06680220fd66c4524051706b98c1c659a674d19d3a766cd0bb276505e99faccd"

def verify_weight(path: Path) -> bool:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    assert observed == EXPECTED_SHA256, f"Checksum mismatch: {observed} != {EXPECTED_SHA256}"
    print(f"Verified {path.name}: {observed}")
    return True

verify_weight(Path("weights/tabdpt1_2.safetensors"))
```
