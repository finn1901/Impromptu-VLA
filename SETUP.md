# Workstation Setup Guide

This guide helps you set up an identical environment on a new workstation.

## Prerequisites

- CUDA-capable GPU
- Conda/Miniconda installed
- Git installed

## Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd Impromptu-VLA
```

## Step 2: Set Up Conda Environment

Create and activate the vllm-gpu environment:

```bash
conda env create -f envs/vllm-gpu.yml  # If you exported your environment
# OR manually create it:
conda create -n vllm-gpu python=3.10
conda activate vllm-gpu
```

Install required packages:
```bash
# Install vLLM
pip install vllm==0.6.5

# Install LLaMA-Factory (if not already installed)
cd /path/to/LLaMA-Factory
pip install -e .

# Install other dependencies
pip install fire transformers peft torch torchvision
```

## Step 3: Download Base Model

Download Qwen2.5-VL-3B-Instruct to `./qwen_local/`:

```bash
# Option 1: Using huggingface-cli
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./qwen_local

# Option 2: Using Python
python -c "
from transformers import AutoModel, AutoTokenizer, AutoProcessor
model = AutoModel.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', trust_remote_code=True)
model.save_pretrained('./qwen_local')
tokenizer.save_pretrained('./qwen_local')
processor.save_pretrained('./qwen_local')
"
```

## Step 4: Set Up NuScenes Dataset

Create a symlink to your NuScenes dataset:

```bash
# Replace /path/to/your/nuscenes with your actual NuScenes path
ln -s /path/to/your/nuscenes ./nuscenes
```

Verify the dataset structure:
```bash
ls nuscenes/
# Should see: samples/, maps/, v1.0-trainval/, etc.
```

## Step 5: Download or Copy Training Artifacts

### Option A: Download trained LoRA adapter from shared storage
```bash
# Copy your trained adapter from shared storage
rsync -avz <source-workstation>:/path/to/trainer_output ./trainer_output
```

### Option B: Train from scratch
```bash
conda activate vllm-gpu
llamafactory-cli train train/Qwen2_5-VL/example/3B_full_QA_train_bs8_b2.yaml
```

## Step 6: Verify Setup

Test that everything works:

```bash
conda activate vllm-gpu

# Test inference with a small sample
CUDA_VISIBLE_DEVICES=0 python train/inference_scripts/vllm_infer.py \
  --model_name_or_path ./qwen_local \
  --adapter_name_or_path ./trainer_output \
  --dataset QA_nuscenes_train_b2_exp11 \
  --template qwen2_vl \
  --max_samples 1 \
  --gpu_memory_utilization 0.9 \
  --image_resolution 65536 \
  --image_min_pixels 1024 \
  --max_new_tokens 32 \
  --save_name test_output.jsonl \
  --tensor_parallel_size 1
```

## Directory Structure

After setup, your directory should look like:

```
Impromptu-VLA/
├── data/
│   └── dataset_info.json          # Committed
├── nuscenes/                       # Symlink (not committed)
├── qwen_local/                     # Downloaded separately (not committed)
├── trainer_output/                 # Copied or retrained (not committed)
├── merged_model/                   # Optional, generated (not committed)
├── train/
│   ├── Qwen2_5-VL/
│   │   └── example/
│   │       └── 3B_full_QA_train_bs8_b2.yaml  # Committed
│   └── inference_scripts/
│       └── vllm_infer.py           # Committed (with fixes)
├── utils/
│   └── merge_adapter.py            # Committed (with fixes)
├── nuscenes_train.json             # Committed
├── nuscenes_test.json              # Committed
└── .gitignore                      # Committed
```

## Important Files to Sync Between Workstations

### Configuration Files (Committed to Git)
- ✅ `data/dataset_info.json` - Dataset configuration
- ✅ `nuscenes_train.json` - Training data paths
- ✅ `nuscenes_test.json` - Test data paths
- ✅ `train/Qwen2_5-VL/example/*.yaml` - Training configs
- ✅ `train/inference_scripts/vllm_infer.py` - Fixed inference script
- ✅ `utils/merge_adapter.py` - Fixed merge script

### Large Files (NOT Committed, Transfer Separately)
- ❌ `qwen_local/` - Base model (~7GB) - Download from HuggingFace
- ❌ `trainer_output/` - LoRA adapter (~60MB) - Copy via rsync/scp
- ❌ `nuscenes/` - Dataset symlink - Create on each workstation
- ❌ `merged_model/` - Optional, can regenerate

## Troubleshooting

### Issue: "Processor was not found"
**Solution**: Make sure you downloaded the base model with processor files, or copy them:
```bash
cp qwen_local/preprocessor_config.json merged_model/
cp qwen_local/generation_config.json merged_model/
```

### Issue: "ValueError: The data should contain the fields: {'image_embeds', 'image_grid_thw'}"
**Solution**: This is fixed in the committed `vllm_infer.py`. Make sure you pulled the latest version.

### Issue: Tensor parallel size error
**Solution**: Always set `--tensor_parallel_size 1` if you have only 1 GPU.

## Notes

- The LoRA adapter in `trainer_output/` is only ~60MB and can be easily copied between workstations
- The base model `qwen_local/` is ~7GB and should be downloaded from HuggingFace on each workstation
- Training checkpoints in `trainer_output/checkpoint-*/` are large and optional - only the final adapter is needed
