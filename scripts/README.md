# Scripts Usage Guide

This directory provides scripts for two four-process workflows managed by
`start.sh` and `tmux`:

1. Distributed inference (`start.sh` option `2`)
2. Training during distributed inference (`start.sh` option `3`)

Both workflows use four GPUs. `start.sh` creates one tmux window for each
process and assigns ranks `0`, `1`, `2`, and `3` with `WORLD_SIZE=4`.

## Before You Start

- Run `start.sh` from the repository root.
- Make sure the `.venv` virtual environment and `tmux` are available.
- Make sure four GPUs are available.
- Set every checkpoint path before starting the cluster.
- Use the checkpoint produced for the corresponding model shard. Do not load a
  Server checkpoint into a Middle Server or Client process.
- `start.sh` starts the processes in this order: Server, Middle Server 1,
  Middle Server 2, and Client. If a process starts before the preceding process
  is ready, increase the corresponding `sleep` value in `start.sh`.

Start the launcher with:

```bash
cd /path/to/MandelbrotV1
source .venv/bin/activate
bash start.sh
```

## 1. Distributed Inference

Enter `2` when `start.sh` displays the startup menu. The launcher starts the
following four processes automatically:

| tmux window | Role | Rank | Script | Required checkpoint |
| --- | --- | ---: | --- | --- |
| 0 | Server | 0 | `test_server_5B.py` | Server checkpoint |
| 1 | Middle Server 1 | 1 | `test_middle_server_5B_A.py` | Middle Server 1 checkpoint |
| 2 | Middle Server 2 | 2 | `test_middle_server_5B_B.py` | Middle Server 2 checkpoint |
| 3 | Inference Client | 3 | `test_client_5B.py` | Client checkpoint |

### Configure the Checkpoints

Before selecting option `2`, update the `--checkpoint_dir` default in each
inference script so that it points to the checkpoint produced for that role:

```python
p.add_argument("--checkpoint_dir", default="/absolute/path/to/checkpoint")
```

Update the following files:

- `test_server_5B.py`: Server checkpoint
- `test_middle_server_5B_A.py`: Middle Server 1 checkpoint
- `test_middle_server_5B_B.py`: Middle Server 2 checkpoint
- `test_client_5B.py`: Client checkpoint

The four paths may be different because every process loads a different model
shard.

### Configure the Inference Prompt

The `--prompt` argument in `test_client_5B.py` controls the sentence supplied
to the distributed model for generation:

```python
p.add_argument(
    "--prompt",
    default="I'd like to see more computer language instructions like the",
)
```

Replace the default value with the sentence that you want the model to
continue. Because `start.sh` currently starts `test_client_5B.py` without
command-line arguments, the value configured as the script default is used.

For example:

```python
p.add_argument(
    "--prompt",
    default="Explain how distributed inference works",
)
```

After configuring the four checkpoints and the prompt, run `bash start.sh` and
enter `2`.

## 2. Training During Distributed Inference

Enter `3` when `start.sh` displays the startup menu. The Server and both Middle
Servers provide distributed inference services, while the fourth process
resumes Client training.

| tmux window | Role | Rank | Script | Checkpoint setting |
| --- | --- | ---: | --- | --- |
| 0 | Server | 0 | `test_server_5B.py` | `--checkpoint_dir` |
| 1 | Middle Server 1 | 1 | `test_middle_server_5B_A.py` | `--checkpoint_dir` |
| 2 | Middle Server 2 | 2 | `test_middle_server_5B_B.py` | `--checkpoint_dir` |
| 3 | Training Client | 3 | `train_pretrain_france_client1.py` | `--resume_from_checkpoint` |

Before selecting option `3`:

1. Set `--checkpoint_dir` in `test_server_5B.py` to the trained Server
   checkpoint.
2. Set `--checkpoint_dir` in `test_middle_server_5B_A.py` to the trained Middle
   Server 1 checkpoint.
3. Set `--checkpoint_dir` in `test_middle_server_5B_B.py` to the trained Middle
   Server 2 checkpoint.
4. Set `--resume_from_checkpoint` in `train_pretrain_france_client1.py` to the
   Client checkpoint from which training should resume.

For the training Client, change the argument default from `None` to the actual
checkpoint path:

```python
p.add_argument(
    "--resume_from_checkpoint",
    type=str,
    default="/absolute/path/to/client/checkpoint",
    help="Path to a checkpoint to resume training from",
)
```

After configuring these paths, run `bash start.sh` and enter `3`.

## Monitoring and Stopping the Cluster

Attach to the tmux session:

```bash
tmux attach -t train_cluster
```

Inside tmux:

- Press `Ctrl+B`, then a window number (`0`-`3`) to view a process.
- Press `Ctrl+B`, then `D` to detach without stopping the processes.

Stop all four processes by terminating the tmux session:

```bash
tmux kill-session -t train_cluster
```

## Common Startup Problems

- `Invalid rank requested`: verify that the four ranks are exactly `0`, `1`,
  `2`, and `3`, and that every process uses `WORLD_SIZE=4`.
- `Connection refused`: a service may not have completed initialization before
  the next process started. Increase the relevant `sleep` value in `start.sh`.
- `TCPStore ... failed to recv` or `Broken pipe`: another rank usually exited
  first. Inspect the earliest error in the Server and Middle Server terminals.
