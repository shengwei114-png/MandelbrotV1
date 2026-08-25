# Scripts Usage Guide

This directory provides scripts for two four-process workflows:

1. Distributed inference
2. Training during distributed inference

Both workflows use four GPUs and four terminals. Run every command from the
repository root, and activate the same Python virtual environment in every
terminal.

## Before You Start

- Make sure four GPUs are available.
- Use a unique rank for each process: `0`, `1`, `2`, and `3`.
- Keep `WORLD_SIZE=4` in all four terminals.
- Start the processes in order: Server, Middle Server 1, Middle Server 2, and
  finally Client.
- Wait until each process has completed initialization before starting the next
  one. Model loading and gRPC/NCCL initialization can take some time.
- Use the checkpoint produced for the corresponding model shard. Do not load a
  Server checkpoint into a Middle Server or Client process.

The examples below pass checkpoint paths on the command line. This overrides
the default paths defined in the scripts, so editing the Python files is not
required.

## 1. Distributed Inference

For distributed inference, all four roles load their trained weights through
`--checkpoint_dir`.

| Terminal | Role | Rank | Script | Required checkpoint |
| --- | --- | ---: | --- | --- |
| 1 | Server | 0 | `test_server_5B.py` | Server checkpoint |
| 2 | Middle Server 1 | 1 | `test_middle_server_5B_A.py` | Middle Server 1 checkpoint |
| 3 | Middle Server 2 | 2 | `test_middle_server_5B_B.py` | Middle Server 2 checkpoint |
| 4 | Client | 3 | `test_client_5B.py` | Client checkpoint |

Open four terminals in the repository root and run the following commands.
Replace every example checkpoint path with the actual path produced by
training.

### Terminal 1: Server

```bash
source .venv/bin/activate && \
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=4 && \
python scripts/test_server_5B.py \
  --checkpoint_dir /path/to/server/checkpoint
```

Wait until the Server is ready before continuing.

### Terminal 2: Middle Server 1

```bash
source .venv/bin/activate && \
export RANK=1 LOCAL_RANK=1 WORLD_SIZE=4 && \
python scripts/test_middle_server_5B_A.py \
  --checkpoint_dir /path/to/middle1/checkpoint
```

Wait until Middle Server 1 is ready before continuing.

### Terminal 3: Middle Server 2

```bash
source .venv/bin/activate && \
export RANK=2 LOCAL_RANK=2 WORLD_SIZE=4 && \
python scripts/test_middle_server_5B_B.py \
  --checkpoint_dir /path/to/middle2/checkpoint
```

Wait until Middle Server 2 is ready before starting the Client.

### Terminal 4: Inference Client

```bash
source .venv/bin/activate && \
export RANK=3 LOCAL_RANK=3 WORLD_SIZE=4 && \
python scripts/test_client_5B.py \
  --checkpoint_dir /path/to/client/checkpoint
```

## 2. Training During Distributed Inference

In this workflow, the Server and both Middle Servers run the same distributed
inference services, while the fourth process resumes the Client training
script.

The three service scripts load their model shards through `--checkpoint_dir`.
The training Client must use `--resume_from_checkpoint` instead.

| Terminal | Role | Rank | Script | Checkpoint argument |
| --- | --- | ---: | --- | --- |
| 1 | Server | 0 | `test_server_5B.py` | `--checkpoint_dir` |
| 2 | Middle Server 1 | 1 | `test_middle_server_5B_A.py` | `--checkpoint_dir` |
| 3 | Middle Server 2 | 2 | `test_middle_server_5B_B.py` | `--checkpoint_dir` |
| 4 | Training Client | 3 | `train_pretrain_france_client1.py` | `--resume_from_checkpoint` |

### Terminal 1: Server

```bash
source .venv/bin/activate && \
export RANK=0 LOCAL_RANK=0 WORLD_SIZE=4 && \
python scripts/test_server_5B.py \
  --checkpoint_dir /path/to/server/checkpoint
```

### Terminal 2: Middle Server 1

```bash
source .venv/bin/activate && \
export RANK=1 LOCAL_RANK=1 WORLD_SIZE=4 && \
python scripts/test_middle_server_5B_A.py \
  --checkpoint_dir /path/to/middle1/checkpoint
```

### Terminal 3: Middle Server 2

```bash
source .venv/bin/activate && \
export RANK=2 LOCAL_RANK=2 WORLD_SIZE=4 && \
python scripts/test_middle_server_5B_B.py \
  --checkpoint_dir /path/to/middle2/checkpoint
```

### Terminal 4: Training Client

```bash
source .venv/bin/activate && \
export RANK=3 LOCAL_RANK=3 WORLD_SIZE=4 && \
python scripts/train_pretrain_france_client1.py \
  --resume_from_checkpoint /path/to/client/checkpoint
```

Any additional training arguments can be appended to the final command as
needed.

## Checkpoint Notes

- `--checkpoint_dir` specifies the model-shard checkpoint loaded by each
  distributed inference script.
- `--resume_from_checkpoint` resumes the Client training state from an existing
  Client checkpoint.
- The checkpoint directories must contain the files expected by the relevant
  script's checkpoint loader.
- If a path contains spaces, wrap it in quotes.
- If you prefer persistent defaults, update the corresponding argument default
  inside each script. Passing the path on the command line is recommended
  because it keeps the scripts reusable.

## Common Startup Problems

- `Invalid rank requested`: verify that the four ranks are exactly `0`, `1`,
  `2`, and `3`, and that every process uses `WORLD_SIZE=4`.
- `Connection refused`: the next service was started before the preceding one
  was ready, or the expected gRPC service is not running.
- `TCPStore ... failed to recv` or `Broken pipe`: another rank usually exited
  first. Inspect the earliest error in the Server and Middle Server terminals.
