"""
Pre-compute T5 + CLIP text embeddings for the empty prompt used during
ControlNet training.  Run once; saved to disk and loaded by train script.

Usage:
    python scripts/precompute_flux_embeddings.py \
        --model_id black-forest-labs/FLUX.1-Fill-dev \
        --out /data/hohs2/outputs/flux_controlnet/text_embeddings.pt
"""

import argparse
import os
import torch
from transformers import T5EncoderModel, CLIPTextModel, CLIPTokenizer, T5Tokenizer

PROMPT = ""   # empty prompt — ControlNet handles the conditioning


def encode(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading T5 tokenizer + encoder …")
    t5_tok = T5Tokenizer.from_pretrained(args.model_id, subfolder="tokenizer_2")
    t5_enc = T5EncoderModel.from_pretrained(
        args.model_id, subfolder="text_encoder_2",
        torch_dtype=torch.bfloat16
    ).to(device)

    print("Loading CLIP tokenizer + encoder …")
    clip_tok = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    clip_enc = CLIPTextModel.from_pretrained(
        args.model_id, subfolder="text_encoder",
        torch_dtype=torch.bfloat16
    ).to(device)

    with torch.no_grad():
        # T5 embedding  (1, seq_len, 4096)
        t5_ids = t5_tok(
            [PROMPT], padding="max_length", max_length=512,
            truncation=True, return_tensors="pt"
        ).input_ids.to(device)
        t5_emb = t5_enc(t5_ids).last_hidden_state   # (1, 512, 4096)

        # CLIP pooled embedding  (1, 768)
        clip_ids = clip_tok(
            [PROMPT], padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(device)
        clip_emb = clip_enc(clip_ids).pooler_output  # (1, 768)

    result = {
        "prompt_embeds":        t5_emb.cpu(),    # (1, 512, 4096) bfloat16
        "pooled_prompt_embeds": clip_emb.cpu(),  # (1, 768)       bfloat16
    }
    torch.save(result, args.out)
    print(f"Saved embeddings → {args.out}")
    print(f"  prompt_embeds:        {result['prompt_embeds'].shape}")
    print(f"  pooled_prompt_embeds: {result['pooled_prompt_embeds'].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-Fill-dev")
    parser.add_argument("--out", default="/data/hohs2/outputs/flux_controlnet/text_embeddings.pt")
    encode(parser.parse_args())
