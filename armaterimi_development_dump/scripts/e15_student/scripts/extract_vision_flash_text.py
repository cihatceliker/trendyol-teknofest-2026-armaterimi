"""Extract the Trendyol-Vision-Flash TEXT tower (Qwen3-0.6B, language_model.model.*) from the local
InternVL snapshot into a plain HF directory, using m6a2's verified routine (zero missing/unexpected
tensors). Output: e15_student/backbones/vision_flash_text/ (Qwen3Model + tokenizer, bf16)."""
import sys, json
from pathlib import Path
import torch
sys.path.insert(0, '/home/jupyter/trendyol-teknofest-work/asya/gcp_jobs/m6a2_vision_flash_masked_inbatch_v1')
from src.modeling import load_vision_flash_text
SNAP = Path('/home/jupyter/trendyol-teknofest-work/asya/gcp_jobs/m6a2_vision_flash_masked_inbatch_v1/model_source/vision_flash_snapshot')
OUT = Path('/home/jupyter/trendyol-teknofest-work/asya/s2_exp/e15_student/backbones/vision_flash_text')
enc, tok, meta = load_vision_flash_text(SNAP, dtype=torch.bfloat16)
assert not meta['missing_keys'] and not meta['unexpected_keys'], meta
enc.backbone.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
json.dump(meta, open(OUT / 'extraction_meta.json', 'w'), indent=1)
print(json.dumps({k: meta[k] for k in ('architecture', 'parameter_count', 'num_hidden_layers', 'hidden_size', 'excluded_vision_tensor_count', 'excluded_lm_head')}, indent=1))
