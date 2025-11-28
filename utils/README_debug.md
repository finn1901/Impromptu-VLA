# Debugging vLLM initialization and LO-RA merging

This small README provides helper commands and usage for the debug scripts in `utils/`.

Files:
- `debug_vllm_init.py` — instantiate a vLLM engine with configurable `--gpu_mem` and `--dtype`.
- `test_transformers_load.py` — tries to load a model with HF Transformers to isolate vLLM vs HF issues.
- `merge_adapter.py` — uses PEFT to merge a LoRA adapter into the base model and saves the merged model locally.

Examples:

1. Test vLLM with low memory:
```bash
CUDA_VISIBLE_DEVICES=0 python utils/debug_vllm_init.py --model Qwen/Qwen2.5-VL-3B-Instruct --gpu_mem 0.5 --dtype float16
```

2. Test HF Transformers load (offloads automatically if needed):
```bash
python utils/test_transformers_load.py --model Qwen/Qwen2.5-VL-3B-Instruct
```

3. Merge an adapter into a base model:
```bash
python utils/merge_adapter.py --base Qwen/Qwen2.5-VL-3B-Instruct --adapter trainer_output --outdir merged_model
```

Notes:
- If vLLM fails while HF can load a model, vLLM is likely incompatible with the model or needs a pinned version that works (vLLM <= 0.6.5 is known to be reliable for some models, though this repo contains vLLM 0.11.2 in environment.yaml now).
- For `merge_adapter.py`, you must have `peft` and `transformers` installed; merging will combine the adapter into base weights and produce a directory `merged_model` for inference.
