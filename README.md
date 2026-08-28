# MandelbrotV1-Distributed-Pretraining

This repository contains the distributed pretraining setup used for four-RTX-5090 GPU experiments. The default cluster runs four processes on one Linux
machine: one server, two middle servers, and one client.

> Using this distributed setup, a 5B-parameter model can be pretrained from scratch for approximately USD 1,500.with a throughput exceeding 10,000 tokens/s.

## Global Initiative

Learn more about the [Mandelbrot Global Initiative](http://www.mandelbrot.cn:8080/JSPWiki/Wiki.jsp?page=Global%20Initiative2).

## Contact

For project inquiries, collaboration, or consultation, please contact us at
[shengwei@mandelbrot.cn](shengwei@mandelbrot.cn).

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

# 2) Install the remaining dependencies
# Upgrade the packaging/build tools first because some dependencies need them.
python -m pip install -U pip setuptools wheel packaging ninja
python -m pip install -r requirements.txt

# 3) Install FlashAttention separately from source without pip build isolation
# This uses the PyTorch/CUDA installation in the active virtual environment.
# First compare the CUDA version used by PyTorch with the nvcc compiler version.
python -c "import torch; print('PyTorch CUDA:', torch.version.cuda)"
nvcc --version

# If nvcc is available and its CUDA version matches PyTorch, skip the following
# find and export commands. Otherwise, locate nvcc and set the correct toolkit.
find /usr/local /opt -type f -name nvcc 2>/dev/null | head

# Set CUDA_HOME to the actual toolkit root that contains bin/nvcc.
# /usr/local/cuda-12.8 is only an example and must match your environment.
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
"$CUDA_HOME/bin/nvcc" --version

# Force a local source build instead of downloading a prebuilt wheel from GitHub.
# MAX_JOBS limits parallel compilation to reduce peak RAM usage.
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \
python -m pip install -v --no-build-isolation "flash-attn==2.8.3.post1"

# 4) Upgrade NCCL separately after all other dependencies are installed
# --no-deps prevents pip from reinstalling or changing the PyTorch stack.
python -m pip install --upgrade --no-deps "nvidia-nccl-cu12==2.30.7"

# Verify the installed NCCL package version
python -m pip show nvidia-nccl-cu12
```

Notes:
- `torch==2.7.1` / `torchvision==0.22.1` are version examples. If your hardware/driver does not match, switch to the appropriate CUDA version and installation source. This exact combination is not mandatory.

- FlashAttention uses `nvcc` to compile its CUDA kernels, which is why the compiler path and version may need to be checked. Compare `nvcc --version` with `torch.version.cuda` first. If `nvcc` is already available and the versions match, you do not need to locate it or set the CUDA environment variables again. If `nvcc` is missing or points to a different toolkit, locate it, set `CUDA_HOME` to the toolkit root containing `bin/nvcc`, and verify it with `"$CUDA_HOME/bin/nvcc" --version`. The `/usr/local/cuda-12.8` value above is only an example.

- Install the NCCL upgrade only after PyTorch and all packages in `requirements.txt` have been installed. Installing or repairing the PyTorch dependencies later may restore PyTorch's pinned NCCL package, in which case the standalone NCCL upgrade command must be run again. The command above is for a CUDA 12 (`cu12`) PyTorch build such as `cu128`; use the matching NCCL package for a different CUDA major version.

- With a prebuilt PyTorch wheel, `torch.cuda.nccl.version()` can continue to show the NCCL version used when PyTorch was compiled (for example, `2.26.2`) even after the separately installed `nvidia-nccl-cu12` package has been upgraded to `2.30.7`. Use `python -m pip show nvidia-nccl-cu12` to confirm the installed package version. Because PyTorch 2.7.1 pins an older NCCL package in its dependency metadata, `python -m pip check` may report a version conflict after this intentional override.

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

The following diagram illustrates the distributed training architecture across
the Server, Middle Servers, and Clients.

[![Distributed training architecture](docs/images/distributed-training-architecture.png)](docs/images/distributed-training-architecture.png)

*Figure 1. Distributed training architecture.*

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

#### Startup Timing Note

`start.sh` launches the four role scripts sequentially and uses `sleep` between
the launches. On some machines, model/tokenizer loading and gRPC/NCCL
initialization take longer than these default delays. If the next role starts
before the previous role is ready, the cluster may appear to hang without an
immediate error.

If this happens, manually increase the `sleep` values in `start.sh`, especially
the delays after the server and middle-server launches. There is no single
correct delay for every environment; allow enough time for each role to finish
its initialization before starting the next one. Alternatively, start the four
scripts manually in separate terminals and wait for each preceding role to be
ready before launching the next role.

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

## 4. Training During Distributed Inference

In this mode, the Server and Middle Servers perform distributed inference,
while the Client continues training the final model blocks.

[![Training during distributed inference architecture](docs/images/training-during-inference-architecture.png)](docs/images/training-during-inference-architecture.png)

*Figure 2. Training during distributed inference.*

For startup instructions and checkpoint configuration, see the
[scripts usage guide](scripts/README.md#2-training-during-distributed-inference).

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
