#!/usr/bin/env python3
"""
Test HF Transformers loading of a model: useful to differentiate between HF-loading issues and vLLM-specific issues.
"""
import argparse
import sys
import traceback

def main(model: str):
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoModel,
        )
        import torch
    except Exception as e:
        print("Import error:", e)
        traceback.print_exc()
        return 2

    print("Attempting to load tokenizer for", model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        print("Tokenizer OK")
    except Exception as e:
        print("Tokenizer failed:", e)
        traceback.print_exc()
        return 3

    # Try a few different HF classes to load multimodal / text gen models
    # Read model_type / architectures from config.json if present to pick appropriate loader
    import os
    config_file = os.path.join(model, "config.json") if os.path.isdir(model) else None
    architectures = []
    model_type = None
    if config_file and os.path.exists(config_file):
        import json
        cfg = json.load(open(config_file))
        architectures = cfg.get("architectures", []) or []
        model_type = cfg.get("model_type")
        print("Detected architectures:", architectures, "model_type:", model_type)

    loaders = [
        ("AutoModelForCausalLM", AutoModelForCausalLM),
        ("AutoModelForSeq2SeqLM", AutoModelForSeq2SeqLM),
        ("AutoModel", AutoModel),
    ]

    # Prefer Seq2Seq loader if architecture name contains "ConditionalGeneration" or model_type suggests seq2seq
    if any("ConditionalGeneration" in a for a in architectures) or (model_type and "vl" in model_type):
        loaders = [("AutoModelForSeq2SeqLM", AutoModelForSeq2SeqLM), ("AutoModel", AutoModel), ("AutoModelForCausalLM", AutoModelForCausalLM)]
    model_obj = None
    for name, loader in loaders:
        try:
            print(f"Attempting to load model with {name} (device_map='auto')")
            model_obj = loader.from_pretrained(model, trust_remote_code=True, device_map="auto")
            print(f"Model load OK with {name}, device_map:", model_obj.hf_device_map if hasattr(model_obj, 'hf_device_map') else 'N/A')
            break
        except Exception as e:
            print(f"Model failed to load with {name}: {e}")
            # Print a shorter excinfo to keep logs manageable
            # traceback.print_exc()
            continue
    if model_obj is None:
        print("Failed to load model with available loaders; try loading on CPU via device_map='cpu' or check model `config.json`")
        return 4

    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.model))
