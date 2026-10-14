"""
Script 1: Extracting hidden states
=====================================
Loads a local CSV file, runs each text through an LLM
and saves hidden states together with metadata in .pt files.

Expected CSV columns:
  title, lang, source_name, format, category, cefr_level, license, text

Output:
  {OUT}_full_layer{LAYER}.pt

Pooling:
  Default: mean pooling (average of all non-padding tokens)
  Disable with --no_pooling -> falls back to last_token

Usage:
  python 1_extract.py --csv data/cefr.csv --model aya --layer 16 --out states/cefr
  python 1_extract.py --csv data/cefr.csv --model qwen --layer 4 --out states/cefr
"""

import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

MODELS = {
    "aya":  "CohereLabs/aya-expanse-8b",
    "qwen": "Qwen/Qwen3.5-9B",
}

MAX_LENGTH = 4096

COL_TEXT  = "text"
COL_LABEL = "cefr_level"
COL_LANG  = "lang"


# ─────────────────────────────────────────────
# Load CSV
# ─────────────────────────────────────────────

def load_csv(csv_path, language_filter=None):
    df = pd.read_csv(csv_path)

    required = [COL_TEXT, COL_LABEL, COL_LANG]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Present: {list(df.columns)}")

    if language_filter:
        df = df[df[COL_LANG] == language_filter].reset_index(drop=True)
        print(f"Language filter '{language_filter}': {len(df)} rows remaining")

    before = len(df)
    df = df.dropna(subset=[COL_TEXT, COL_LABEL]).reset_index(drop=True)
    if len(df) < before:
        print(f"Warning: {before - len(df)} rows with missing text/label removed")

    return df


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────

def load_model(model_key):
    model_id = MODELS[model_key]
    print(f"Loading model: {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded. Number of layers: {model.config.num_hidden_layers}")
    return tokenizer, model, model_id


# ─────────────────────────────────────────────
# Pooling
# ─────────────────────────────────────────────

def pool_hidden(hidden, attention_mask, pooling):
    if pooling == "mean":
        mask = attention_mask.clone().float()
        mask[:, 0] = 0.0  # exclude BOS (padding_side must be "right"!)
        mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (hidden * mask).sum(dim=1) / denom
    elif pooling == "last_token":
        seq_lengths = attention_mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0)), seq_lengths]
    else:
        raise ValueError(f"Unknown pooling: {pooling}")


# ─────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────

def extract_single_layer(texts, tokenizer, model, layer, pooling, batch_size):
    """Returns tensor [N, hidden_dim]."""
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_LENGTH,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[layer]
        vecs = pool_hidden(hidden, inputs["attention_mask"], pooling)
        all_vectors.append(vecs.cpu().float())
        print(f"  [layer {layer}] {min(i + batch_size, len(texts))}/{len(texts)}")

    return torch.cat(all_vectors, dim=0)


def extract_all_layers(texts, tokenizer, model, pooling, batch_size):
    """Returns tensor [n_layers+1, N, hidden_dim]."""
    n_layers = model.config.num_hidden_layers + 1
    per_layer = [[] for _ in range(n_layers)]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_LENGTH,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer_idx in range(n_layers):
            hidden = outputs.hidden_states[layer_idx]
            vecs = pool_hidden(hidden, inputs["attention_mask"], pooling)
            per_layer[layer_idx].append(vecs.cpu().float())

        print(f"  [all layers] {min(i + batch_size, len(texts))}/{len(texts)}")

    return torch.stack([torch.cat(v, dim=0) for v in per_layer], dim=0)


# ─────────────────────────────────────────────
# Saving
# ─────────────────────────────────────────────

def save_subset(states, df, subset_name, out_prefix, layer, pooling, model_id):
    out_path = f"{out_prefix}_{subset_name}_layer{layer}.pt"
    torch.save({
        "states":      states,
        "texts":       list(df[COL_TEXT]),
        "cefr_levels": list(df[COL_LABEL]),
        "languages":   list(df[COL_LANG]),
        "layer":       layer,
        "pooling":     pooling,
        "model_id":    model_id,
        "subset":      subset_name,
    }, out_path)
    print(f"Saved: {out_path}  (shape: {tuple(states.shape)})")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        required=True,
                        help="Path to the CSV file")
    parser.add_argument("--model",      required=True,
                        choices=["aya", "qwen"],
                        help="Model: 'aya' or 'qwen'")
    parser.add_argument("--language",   default=None,
                        help="Language filter, e.g. 'en' or 'de'")
    parser.add_argument("--layer",      type=int, default=-1,
                        help="Layer index (-1 = last)")
    parser.add_argument("--no_pooling", action="store_true",
                        help="Disable pooling -> last_token")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--all_layers", action="store_true",
                        help="Additionally save all layers")
    parser.add_argument("--out",        required=True,
                        help="Output prefix, e.g. states/cefr")
    args = parser.parse_args()

    pooling = "last_token" if args.no_pooling else "mean"
    print(f"Pooling: {pooling}")
    print(f"Model:   {args.model} ({MODELS[args.model]})")

    # Load CSV
    df = load_csv(args.csv, language_filter=args.language)
    print(f"CSV loaded: {len(df)} rows | Languages: {sorted(df[COL_LANG].unique())}")
    print(f"CEFR levels: {sorted(df[COL_LABEL].unique())}")

    # Load model
    tokenizer, model, model_id = load_model(args.model)

    # Extraction + saving
    texts = list(df[COL_TEXT])
    print(f"\n--- Extraction: {len(texts)} texts ---")

    states = extract_single_layer(
        texts, tokenizer, model, args.layer, pooling, args.batch_size
    )
    save_subset(df=df, states=states, subset_name="full",
                out_prefix=args.out, layer=args.layer,
                pooling=pooling, model_id=model_id)

    if args.all_layers:
        print(f"\nExtracting all layers...")
        states_all = extract_all_layers(
            texts, tokenizer, model, pooling, args.batch_size
        )
        out_all = f"{args.out}_full_all_layers.pt"
        torch.save({
            "states":      states_all,
            "texts":       list(df[COL_TEXT]),
            "cefr_levels": list(df[COL_LABEL]),
            "languages":   list(df[COL_LANG]),
            "pooling":     pooling,
            "model_id":    model_id,
            "subset":      "full",
        }, out_all)
        print(f"Saved: {out_all}  (shape: {tuple(states_all.shape)})")


if __name__ == "__main__":
    main()