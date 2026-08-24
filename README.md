# MandelbrotV1-Distributed-Pretraining

This repository contains the distributed pretraining setup used for our
four-GPU experiments. The default cluster runs four processes on one Linux
machine: one server, two middle servers, and one client.

## 1. Environment Setup (Linux)

### 1.1 Requirements

- Linux x86_64
- CPython 3.12
- 4 RTX 5090 GPUs
- A CUDA driver/toolkit compatible with the selected PyTorch build
- `tmux`

Install `tmux` before launching the training cluster:

```bash
sudo apt-get update
sudo apt-get install -y tmux
```

### 1.2 Create and Activate the Virtual Environment

Run the following commands from the repository root:

```bash
cd ../MandelbrotV1
python3.12 -m venv .venv
source .venv/bin/activate
```

### 1.3 Install Dependencies

```bash
# 1) Install PyTorch first (CUDA versions are not on the default PyPI index)
# Example below uses cu128; replace cu128 with your CUDA version if different
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1

# 2) Install the build tools required by FlashAttention
python -m pip install -U pip setuptools wheel packaging ninja

# 3) Install FlashAttention separately without pip build isolation
# MAX_JOBS limits parallel compilation to reduce peak RAM usage
MAX_JOBS=4 python -m pip install --no-build-isolation "flash-attn==2.8.3.post1"

# 4) Install the remaining dependencies. The installed FlashAttention version
# satisfies flash-attn>=2.8.0 in requirements.txt, so pip will skip rebuilding it.
python -m pip install -r requirements.txt
```

Notes:
- `torch==2.7.1` / `torchvision==0.22.1` are version examples. If your hardware/driver does not match, switch to the appropriate CUDA version and installation source. This exact combination is not mandatory.

- Building FlashAttention from source requires a complete CUDA development toolkit, including `nvcc`. Run `nvcc --version` before installation if you are unsure whether it is available.

- You also need to replace a file in the transformers package: copy `modeling_outputs.py` from the `transformers` directory in this folder, overwriting the one at `.venv/lib/python3.12/site-packages/transformers/modeling_outputs.py`.

- Then install `tokenizers-0.19.1-cp312-cp312-manylinux_2_34_x86_64.whl`.

```bash
# Install the local wheel from the repository root
pip install -U --force-reinstall --no-deps ./tokenizers-0.19.1-cp312-cp312-manylinux_2_34_x86_64.whl
```

## 2. Data Preparation

The training script supports two data source formats:

1) `.txt` — reads text line by line. `--train_folder` can point to a single `.txt` file or a directory containing multiple `.txt` files (searched recursively).

2) `.jsonl` — one JSON object per line; the script reads only the `text` field:

```json
{"text": "Question: ... Answer: ..."}
```

If your data uses a `question`/`answer` structure, convert it to the `text` format above first (or convert directly to `.txt` with one sample per line).

The script uses our internally processed WanJuan-CC dataset (~10 GB). The dataset was not uploaded to the repository because it is too large.
To test with a different dataset, modify the `--train_folder` argument in the script.

---

## 3. Distributed Training

### 3.1 Commonly Used Parameters

The most frequently used arguments are listed below (training essentials only):

- `--train_folder` — path to training data (`.txt` file/directory or `.jsonl` file/directory)
- `--tokenizer_name` — tokenizer directory (used by `MandelbrotV1TokenizerManager.from_pretrained(...)`)
- `--block_size` — context length (recommended starting point: 256)
- `--per_device_train_batch_size` / `--gradient_accumulation_steps`
- `--max_steps` / `--save_steps` / `--logging_steps`
- `--output_dir` — output directory for saving checkpoints

To list all available arguments:

```bash
python scripts/train_pretrain_france_middle_server1.py -h
```

### 3.2 Training Example

The experiments described in this repository were conducted on four GPUs.
Option `1` in `start.sh` launches the four-process cluster and assigns one
process to each GPU:

- GPU 0: Server
- GPU 1: Middle Server 1
- GPU 2: Middle Server 2
- GPU 3: Client

```bash
# 1) Activate the virtual environment
cd /path/to/MandelbrotV1
source .venv/bin/activate

# 2) Launch distributed training and enter 1 at the prompt
#    to select the four-process (four-GPU) mode
bash start.sh

# 3) Monitor GPU training status with tmux
#    Ctrl+B then press the corresponding number key to switch between GPUs
#    Ctrl+B then D to detach from tmux
tmux attach -t train_cluster

# 4) Stop distributed training
tmux kill-session -t train_cluster 2>/dev/null
```
