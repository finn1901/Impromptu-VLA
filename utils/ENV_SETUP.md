# How to create a fresh GPU-capable conda environment for inference

This repo includes two pre-made conda environment YAML examples; pick one depending on your inference engine:

- `envs/env_vllm_gpu.yml` — for vLLM (v0.11.2) + CUDA 12.8
- `envs/env_sglang_gpu.yml` — for SGLang (0.5.5) + CUDA 12.8

Important notes
- Use a fresh conda environment to avoid breaking your main environment.
- Use a PyTorch wheel that supports your GPU compute capability (RTX 5090 → sm_120). For CUDA 12.8 you'll likely need a dev/nightly wheel for PyTorch (e.g., `torch==2.10.0.dev...+cu128`).
- `vllm` versions may require specific `torch` builds. For vLLM 0.6.x `torch==2.5.1` was required in older setups; for the modern RTX 5090 and vLLM 0.11.x you should use the latest nightly or a stable torch wheel that support CUDA 12.x.

Example steps (recommended: start with `env_vllm_gpu.yml`):

1. Create the environment
```bash
conda env create -f envs/env_vllm_gpu.yml -n vllm-gpu
conda activate vllm-gpu
```

2. Install PyTorch wheel for CUDA 12.8 (choose exact wheel matching your driver). Example:
```bash
# Install a CUDA wheel for PyTorch; this example uses the cu128 wheel
pip install --pre --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

Note: If you prefer, use the helper script `envs/setup_vllm.sh` which performs the steps above that create & install the wheel for you.

3. Now install the rest of pip packages in the env (If you created the env from the YAML, pip packages were already installed — check `pip list` and ensure torch is correct). If you need to change torch version, reinstall `pip install` accordingly and re-check `pip` listed packages.

4. Minimal validation
```bash
python - <<'PY'
import torch
print('Torch', torch.__version__)
print('CUDA available', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
PY

5. Test vLLM or SGLang initialization (examples)
```bash
# vLLM quick test
python utils/debug_vllm_init.py --model gpt2 --gpu_mem 0.6 --dtype float16 --suppress_del_error

# SGLang load/test
python train/inference_scripts/sglang_infer.py --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct --dataset QA_nuscenes_test_b2_exp11 --template qwen2_vl --max_samples 1
```

6. If you run into architecture or model-name mismatches with vLLM (like `Qwen2_5_VLForConditionalGeneration`), either:
  - Use `sglang_infer.py` or `internVL_lmdeploy.py` instead of vLLM for inference; or
  - Snapshot the model locally and rename its `config.json` `architectures` entry to `Qwen2VLForConditionalGeneration` and attempt `vllm` again (risky, only if code compatibility is likely).

Notes about torch nightly vs stable
- The RTX 5090 (sm_120) needs a torch wheel that includes sm_120 kernels. A PyTorch nightly (or a recent stable for CUDA 12.8) should be used and installed from PyTorch's cu128 index as above. Experiment with Pytorch nightly only in a fresh environment to avoid dependency conflicts.

If you want, I can provide a sample command that creates `sglang-gpu` env and another for `vllm-gpu` environment with pinned versions (exact wheel commands for torch). Tell me which engine you prefer.
