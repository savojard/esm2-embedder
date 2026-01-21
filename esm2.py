#!/usr/bin/env python
"""
ESM2 embedding generator (Hugging Face)

Features:
- Writes ONE .npz PER SEQUENCE into --out-dir
- Splits sequences longer than --max-len into tokenizer-aware chunks and merges embeddings

Example:
  python esm2_embed_perseq.py input.fasta \
    --model-id esm2_t33_650M_UR50D \
    --device cuda:0 --dtype bf16 \
    --pooling mean \
    --out-dir ./esm2_out

Offline:
  python esm2_embed_perseq.py input.fasta \
    --model-dir /models/esm2_t33_650M_UR50D \
    --tokenizer-dir /models/esm2_t33_650M_UR50D \
    --device cuda:0 --dtype bf16 \
    --pooling none --save-per-residue \
    --out-dir ./esm2_out
"""
import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
import torch
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel

def read_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    records = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seq = str(rec.seq).replace(" ", "").replace("\n", "")
        if not seq:
            continue
        records.append((rec.id, seq))
    if not records:
        raise ValueError("No sequences found in FASTA.")
    return records

def pick_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    p = argparse.ArgumentParser(description="Generate ESM2 embeddings (HF) with per-sequence .npz outputs.")
    p.add_argument("fasta", help="Input FASTA (one or many sequences).")
    # Model sources (no 'facebook/' prefix)
    p.add_argument("--model-id", default="esm2_t33_650M_UR50D",
                   help="HF model id (without 'facebook/'). Ignored if --model-dir is set.")
    p.add_argument("--model-dir", default=None,
                   help="Local directory with pre-downloaded model (offline).")
    p.add_argument("--tokenizer-dir", default=None,
                   help="Local directory for tokenizer (defaults to model source).")
    # Inference controls
    p.add_argument("--device", default=None,
                   help='Torch device, e.g. "cuda:0" or "cpu". Default auto.')
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="fp32",
                   help="Computation dtype for inference.")
    p.add_argument("--max-len", type=int, default=1024,
                   help="Max tokenizer length (tokens including specials).")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="Optional model-internal chunk size (if supported) to reduce VRAM.")
    # Output & pooling
    p.add_argument("--pooling", choices=["mean", "cls", "none"], default="mean",
                   help="Per-sequence pooling: mean/cls or none (save per-residue only). "
                        "NOTE: if the sequence is chunked, 'cls' falls back to 'mean'.")
    p.add_argument("--save-per-residue", action="store_true",
                   help="Also save per-residue embeddings in each .npz file.")
    p.add_argument("--out-dir", required=True,
                   help="Output directory to store one .npz per sequence.")
    return p.parse_args()

def tokenizer_payload_len(tokenizer, max_len: int) -> int:
    """Compute how many *non-special* tokens can fit given max_len."""
    try:
        specials = tokenizer.num_special_tokens_to_add(pair=False)
    except Exception:
        # Fallback heuristic: assume 2 special tokens (BOS/EOS or CLS/SEP)
        specials = 2
    payload = max_len - specials
    return max(payload, 1)

def mask_special_tokens(input_ids: torch.Tensor, attn_mask: torch.Tensor, tokenizer) -> torch.Tensor:
    """Build a boolean mask of valid (non-special) tokens where attention_mask==1."""
    mask = attn_mask.bool()
    for attr in ("cls_token_id", "bos_token_id", "eos_token_id", "sep_token_id", "pad_token_id"):
        tok_id = getattr(tokenizer, attr, None)
        if tok_id is not None:
            mask &= (input_ids != tok_id)
    return mask

def embed_chunk(model, tokenizer, seq_chunk: str, device, torch_dtype, max_len: int):
    """Embed a single sequence chunk -> returns (per_residue_embeddings [L, H])."""
    enc = tokenizer(
        seq_chunk,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
    last_hidden = outputs.last_hidden_state  # [1, T, H]

    valid_mask = mask_special_tokens(input_ids, attention_mask, tokenizer)[0]  # [T]
    emb_valid = last_hidden[0][valid_mask]  # [L, H] (only non-special tokens)
    return emb_valid  # still on device

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = pick_device(args.device)
    use_bf16 = (args.dtype == "bf16") and (device.type == "cuda")
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

    seqs = read_fasta(args.fasta)

    # Model/tokenizer sources
    local_only = args.model_dir is not None or args.tokenizer_dir is not None
    # If model-dir not provided, build HF repo id as "facebook/<model-id>"
    model_src = args.model_dir or f"facebook/{args.model_id}"
    tok_src = args.tokenizer_dir or args.model_dir or f"facebook/{args.model_id}"

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, local_files_only=local_only)
    model = AutoModel.from_pretrained(
        model_src,
        local_files_only=local_only,
        torch_dtype=torch_dtype
    ).to(device)
    model.eval()
    torch.set_grad_enabled(False)

    # Optional model-internal chunk size (not all ESM2 builds expose this)
    if args.chunk_size is not None:
        try:
            if hasattr(model, "set_chunk_size"):
                model.set_chunk_size(args.chunk_size)
                print(f"🔹 Using model chunk size = {args.chunk_size}")
            elif hasattr(model, "esm") and hasattr(model.esm, "set_chunk_size"):
                model.esm.set_chunk_size(args.chunk_size)
                print(f"🔹 Using ESM chunk size = {args.chunk_size}")
        except Exception as e:
            print(f"⚠️ Could not set model chunk size: {e}")

    payload_len = tokenizer_payload_len(tokenizer, args.max_len)

    # autocast only on CUDA BF16
    autocast_cm = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if use_bf16 else torch.autocast("cuda", enabled=False))

    with autocast_cm:
        for seq_id, seq in seqs:
            # Split long sequences into tokenizer-aware chunks
            chunks = [seq[i:i + payload_len] for i in range(0, len(seq), payload_len)]

            per_residue_parts = []
            for ch in chunks:
                emb_valid = embed_chunk(model, tokenizer, ch, device, torch_dtype, args.max_len)
                # move to CPU fp32 for accumulation
                per_residue_parts.append(emb_valid.detach().to(torch.float32).cpu().numpy())

            # Merge per-residue embeddings across chunks
            per_residue = np.concatenate(per_residue_parts, axis=0)  # [L_total, H]

            # Compute pooled vector (if requested)
            pooled = None
            if args.pooling != "none":
                if len(chunks) > 1 and args.pooling == "cls":
                    # CLS pooling across chunks is ill-defined; fall back to global mean
                    pooled = per_residue.mean(axis=0, keepdims=False)
                elif args.pooling == "mean":
                    pooled = per_residue.mean(axis=0, keepdims=False)
                else:
                    # Single-chunk + 'cls': reuse first token of that single chunk
                    # But since we already stripped specials, we don't have CLS here.
                    # So for consistency, use mean as it's robust.
                    pooled = per_residue.mean(axis=0, keepdims=False)

            # Save per-sequence .npz
            out_path = os.path.join(args.out_dir, f"{seq_id}.npz")
            save_dict = {"id": np.array(seq_id, dtype=object)}
            if args.pooling != "none" and pooled is not None:
                save_dict["pooled"] = pooled.astype(np.float32)
            if args.save_per_residue:
                save_dict["token_embeddings"] = per_residue.astype(np.float32)
            np.savez_compressed(out_path, **save_dict)

            print(f"✅ Saved {out_path}  "
                  f"(pooled={'yes' if 'pooled' in save_dict else 'no'}, "
                  f"per_residue={'yes' if 'token_embeddings' in save_dict else 'no'}, "
                  f"L={per_residue.shape[0]}, H={per_residue.shape[1] if per_residue.ndim==2 else 'NA'})")

    print(f"\nDone. device={device.type}, dtype={'bf16' if use_bf16 else 'fp32'}, "
          f"max_len={args.max_len}, payload_len={payload_len}, pooling={args.pooling}")
    print(f"Outputs in: {args.out_dir}")

if __name__ == "__main__":
    main()

