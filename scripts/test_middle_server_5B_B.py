#!/usr/bin/env python3
# coding: utf-8
import os, sys, time, argparse, threading
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import torch
from configuration_MandelbrotV1 import MandelbrotV1Config
from modeling_MandelbrotV1 import MandelbrotV1ForCausalLM
from tokenization_MandelbrotV1_fast import MandelbrotV1TokenizerManager,global_vars
global_vars["is_Loss_Log"] = True
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','ALL_PROXY','all_proxy'):
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'
from BaseTextDataset import StreamingTextDataset, worker_init_fn, AsyncPrefetchDataLoader

p = argparse.ArgumentParser()
p.add_argument("--model_root", default="../MandelbrotV1/configs/middle2")
p.add_argument("--checkpoint_dir", default="../MandelbrotV1/outputs_incremental_pretraining_middle2/checkpoint-epoch-0")
p.add_argument("--tokenizer_dir", default="../MandelbrotV1/tok_26layers4/26_tokenizers")
p.add_argument("--local_rank", type=int, default=-1,
                   help="Local rank for distributed training. Set by torchrun/launch utility automatically.")
p.add_argument("--max_steps", type=int, default=-1)
args = p.parse_args()
local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
if local_rank >= 0 and torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = MandelbrotV1Config.from_pretrained(args.model_root)
tm =MandelbrotV1TokenizerManager.tokenizer_manager =  MandelbrotV1TokenizerManager.from_pretrained(
    pretrained_path=args.tokenizer_dir, layer_subfolder_prefix="layer", num_layers=None)
model = MandelbrotV1ForCausalLM(
    config=config, start_block_idx=22, end_block_idx=25,
    is_Distribution=True, distribution_Type="middle_Server")
model = model.to(device); model.eval()
from modeling_MandelbrotV1 import MandelbrotV1ShardManager
sd = MandelbrotV1ShardManager.load_state_dict_from_dir(args.checkpoint_dir)
model.load_state_dict(sd, strict=False)
use_dr = getattr(config, "use_dimensionality_reduction", True)
if not isinstance(model, MandelbrotV1ForCausalLM):
    model.module.last_layer_idx = None if use_dr else 0
    model.module.bMax_frequency = use_dr
ds = StreamingTextDataset(config, folder="./data_france/france_en.txt",
    tokenizer_name=args.tokenizer_dir, tokenizer_layer_idx=0,
    block_size=20, batch_size=1, is_Distribution=True,
    distribution_Type="middle_Server", worker_id="middle_0", model=model,
    use_dimensionality_reduction=True)
ds.tm = MandelbrotV1TokenizerManager.tokenizer_manager
ds.tokenizer = tm.tokenizers[0]
worker_init_fn(-1, ds)
dl = AsyncPrefetchDataLoader(ds, batch_size=None, num_workers=0, collate_fn=lambda b: b, prefetch_size=2)
print("[Middle] Ready. Entering loop...")



past_key_values = None
autocast_kwargs = {
    "device_type": "cuda", 
    "dtype": torch.bfloat16  # 如果是老显卡请改为 torch.float16
}
print("[Middle] Entering loop...")
for step, batch in enumerate(dl):
    if args.max_steps>0 and step >= args.max_steps: break
    model.last_layer_idx = None if use_dr else ds.last_layer_idx
    if hasattr(batch, "to"): batch = batch.to(device)
    else: batch = torch.tensor(batch, device=device)
    if batch.dim() == 1: batch = batch.unsqueeze(0)
    attn = torch.ones_like(batch, dtype=torch.long)
    print("[Middle] Step", step, "shape:", batch.shape, "cache:", past_key_values is not None)
    with torch.no_grad():
        with torch.amp.autocast(**autocast_kwargs):
            out = model(input_ids=batch, attention_mask=attn, labels=None,
                        layer_idx=model.last_layer_idx, use_cache=True,
                        past_key_values=past_key_values)
    past_key_values = out.get("past_key_values") if isinstance(out, dict) else getattr(out, "past_key_values", None)
    loss = out.get("loss") if isinstance(out, dict) else getattr(out, "loss", None)
    print("[Middle] Step", step, "loss =", loss.item() if loss is not None else "none")
print("[Middle] Done.")
