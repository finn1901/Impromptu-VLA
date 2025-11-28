#!/usr/bin/env python3
"""
Merge a LoRA adapter into a base model using PEFT and Transformers APIs; saves merged model locally.
"""
import argparse


def main(base: str, adapter: str, outdir: str):
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
            AutoModel,
            AutoProcessor
        )
        from peft import PeftModel
        import torch
    except Exception as e:
        print("Missing package:", e)
        return 2

    print("Loading base model", base)
    # Try loading with a few plausible HF classes for multimodal / text gen models
    model = None
    load_kwargs = dict(trust_remote_code=True, device_map="cpu")
    try:
        model = AutoModelForCausalLM.from_pretrained(base, **load_kwargs)
        print("Loaded base with AutoModelForCausalLM")
    except Exception:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(base, **load_kwargs)
            print("Loaded base with AutoModelForSeq2SeqLM")
        except Exception:
            try:
                model = AutoModel.from_pretrained(base, **load_kwargs)
                print("Loaded base with AutoModel")
            except Exception as e:
                print("Failed to load base model with AutoModel/AutoModelForSeq2SeqLM/AutoModelForCausalLM:", e)
                return 3
    print("Applying adapter from", adapter)
    # Some multimodal base models may not provide the HF generation helper method
    # prepare_inputs_for_generation which PEFT expects. Provide a minimal fallback
    # wrapper that returns the input dict so PEFT can attach generation helpers.
    if not hasattr(model, "prepare_inputs_for_generation"):
        import types
        def _prepare_inputs_for_generation(self, input_ids, **kwargs):
            # Minimal hygiene: return inputs required by generation pipelines
            return {"input_ids": input_ids, **kwargs}
        # Attach method to model instance
        model.prepare_inputs_for_generation = types.MethodType(_prepare_inputs_for_generation, model)

    peft_model = PeftModel.from_pretrained(model, adapter, device_map="cpu")
    merged = peft_model.merge_and_unload()
    print("Saving merged model to", outdir)
    merged.save_pretrained(outdir)
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    tokenizer.save_pretrained(outdir)

    # Try to save processor if it exists (needed for vision-language models)
    try:
        processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
        processor.save_pretrained(outdir)
        print("Saved processor to", outdir)
    except Exception as e:
        print(f"Note: Could not load/save processor (this is fine for text-only models): {e}")

    print("Saved merged model and tokenizer to", outdir)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.base, args.adapter, args.outdir))
