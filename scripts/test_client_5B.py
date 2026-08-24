#!/usr/bin/env python3
# coding: utf-8
import os, sys, time, argparse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import torch, torch.nn.functional as F
from configuration_MandelbrotV1 import MandelbrotV1Config
from modeling_MandelbrotV1 import MandelbrotV1ForCausalLM
from tokenization_MandelbrotV1_fast import MandelbrotV1TokenizerManager,global_vars
global_vars["is_Loss_Log"] = True
for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy','ALL_PROXY','all_proxy'):
    os.environ.pop(k, None)
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'
from BaseTextDataset import StreamingTextDataset, worker_init_fn, MandelbrotV1ForCausalInferenceModel
from train_pretrain_france_client1 import IntrinsicDimensionAnalyzer
from train_pretrain_france_client1 import IntrinsicDimensionAnalyzer, collect_and_reset_losses
from transformers import LogitsProcessor
from langdetect import detect, DetectorFactory
import unicodedata


p = argparse.ArgumentParser()
p.add_argument("--model_root", default="/root/autodl-tmp/MandelbrotV1/configs/client")
p.add_argument("--checkpoint_dir", default="/root/autodl-tmp/MandelbrotV1/outputs_incremental_pretraining_client/checkpoint-epoch-0")
p.add_argument("--tokenizer_dir", default="/root/autodl-tmp/MandelbrotV1/gemma3_tok_26layers4/gemma3_26layers_v2")
p.add_argument("--local_rank", type=int, default=-1,
                   help="Local rank for distributed training. Set by torchrun/launch utility automatically.")
p.add_argument("--prompt", default="I'd like to see more computer language instructions like the")
p.add_argument("--max_new_tokens", type=int, default=20)
args = p.parse_args()
torch.cuda.empty_cache() 
local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
if local_rank >= 0 and torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DetectorFactory.seed = 0

class StrictEnglishLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, mask_cache_path="english_logits_mask.pt"):
        self.mask_cache_path = mask_cache_path
        if os.path.exists(mask_cache_path):
            print(f"⚡ [INFO] 发现缓存的 Mask，正在从 {mask_cache_path} 加载...")
            self.mask = self._load_mask()
        else:
            print("🚀 [INFO] 未找到缓存，正在构建严格英文白名单，请耐心等待...")
            self.mask = self._build_strict_mask(tokenizer)
            self._save_mask()  # 构建完成后，自动保存到本地
            print("✅ [INFO] 英文白名单构建完成并已缓存！")
    def _save_mask(self):
        """将 mask 张量序列化保存到本地磁盘"""
        torch.save(self.mask, self.mask_cache_path)
        print(f"💾 [INFO] Mask 已成功保存至: {self.mask_cache_path}")

    def _load_mask(self):
        """从本地磁盘反序列化加载 mask 张量"""
        return torch.load(self.mask_cache_path, map_location="cpu")
    def _build_strict_mask(self, tokenizer):
        vocab = tokenizer.get_old_vocab()
        vocab_size = len(vocab)
        
        # 1. 初始化一个全为 -inf 的掩码（默认全部屏蔽）
        mask = torch.full((vocab_size,), float('-inf'), dtype=torch.float32)
        
        allowed_count = 0
        for token, token_id in vocab.items():
            # 2. 基础放行规则：
            # - 包含非拉丁字符（如中文、日文）的 Token 绝对不放行
            # - 长度小于等于 1 的字符（如标点、空格、单个字母）直接放行
            if any(not unicodedata.name(char, '').startswith('LATIN') for char in token if char.strip()):
                continue
                
            if len(token.strip()) <= 1:
                mask[token_id] = 0.0
                allowed_count += 1
                continue
                
            # 3. 核心检测：使用 langdetect 判断是否为英文
            try:
                # 如果检测到是英文，则将其对应的 mask 设为 0（允许生成）
                if detect(token) == 'en':
                    mask[token_id] = 0.0
                    allowed_count += 1
            except Exception:
                # langdetect 对太短的字符串可能会报错，这里直接忽略（不放行）
                continue
                
        print(f"📊 [STATS] 词表总大小: {vocab_size}, 合法英文 Token 数量: {allowed_count}")
        return mask

    def __call__(self, input_ids, scores):
        # 将白名单 mask 加到 scores 上
        # 0 + logit = logit (合法英文)
        # -inf + logit = -inf (乱码/非英文)
        return scores + self.mask.to(scores.device)
# 1. 定义你自己的 LogitsProcessor
class EnglishOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        # 在初始化时，找出所有非英文的 token_id
        self.mask = self._build_mask(tokenizer)

    def _build_mask(self, tokenizer):
        import unicodedata
        vocab = tokenizer.get_old_vocab()
        vocab_size = len(vocab) 
        mask = torch.zeros(vocab_size, dtype=torch.float32)
        
        # 遍历词表，把非拉丁字母的 token 标记为 -inf
        for token, token_id in vocab.items():
            # 如果 token 中包含任何非拉丁字母字符，就屏蔽它
            if any(not unicodedata.name(char, '').startswith('LATIN') for char in token if char.strip()):
                mask[token_id] = float('-inf')
        return mask

    def __call__(self, input_ids, scores):

        masked_scores = scores + self.mask.to(scores.device)

        print(f"[DEBUG] Max Logit: {masked_scores.max().item():.2f}, Min Logit: {masked_scores.min().item()}")
        return masked_scores



config = MandelbrotV1Config.from_pretrained(args.model_root)
tm=MandelbrotV1TokenizerManager.tokenizer_manager = MandelbrotV1TokenizerManager.from_pretrained(
    pretrained_path=args.tokenizer_dir, layer_subfolder_prefix="layer", num_layers=None)
model = MandelbrotV1ForCausalInferenceModel(
    config=config, start_block_idx=25, end_block_idx=0,
    is_Distribution=True, distribution_Type="client")
from modeling_MandelbrotV1 import MandelbrotV1ShardManager
sd = MandelbrotV1ShardManager.load_state_dict_from_dir(args.checkpoint_dir)
model.load_state_dict(sd, strict=False)
del sd
model = model.to(device); 
model.eval()
use_dr = getattr(config, "use_dimensionality_reduction", True)
if not isinstance(model, MandelbrotV1ForCausalLM):
    model.module.last_layer_idx = None if use_dr else 0
    model.module.bMax_frequency = use_dr

ds = StreamingTextDataset(config, folder="./_empty_data",
    tokenizer_name=args.tokenizer_dir, tokenizer_layer_idx=0,
    block_size=20, batch_size=1, is_Distribution=True,
    distribution_Type="client", worker_id="client_0", model=model,
    use_dimensionality_reduction=True)
ds.tm = MandelbrotV1TokenizerManager.tokenizer_manager

tokenizer=ds.tokenizer = ds.tm.tokenizers[0]
worker_init_fn(-1, ds)
print("[Client] Ready.")

use_dimensionality_reduction= getattr(model, "use_dimensionality_reduction", True)

eos_token_id=tokenizer.eos_token_id
bos_token_id=tokenizer.bos_token_id
pad_token_id=tokenizer.pad_token_id

enc = tm.encode(args.prompt, selection_strategy="highest_with_content",
                add_special_tokens=True, bMax_frequency=use_dimensionality_reduction)
chosen_layer = enc["selected_layer"]

if not (use_dimensionality_reduction):
    ds.last_layer_idx=model.last_layer_idx=chosen_layer
    tokenizer=ds.tm.tokenizers[chosen_layer]
    eos_token_id=tokenizer.eos_token_old_id
    bos_token_id=tokenizer.bos_token_old_id
    pad_token_id=tokenizer.pad_token_old_id

global_ids = enc["selected"]["global_ids"] if use_dr else enc["selected"]["local_ids"]
input_ids = torch.tensor([global_ids], dtype=torch.long).to(device)
attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
forward_layer_idx = None if use_dr else 0
ds.tokenizer = tm.tokenizers[chosen_layer]
generated_ids = []
past_key_values = None
print("[Client] Prompt:", repr(args.prompt), "seq_len:", input_ids.shape[1])
autocast_kwargs = {
    "device_type": "cuda", 
    "dtype": torch.bfloat16  # 如果是老显卡请改为 torch.float16
}

model.custom_dataset = ds

#processor = StrictEnglishLogitsProcessor(tokenizer)

with torch.amp.autocast(**autocast_kwargs):
    outputs = model.generate(
        input_ids, 
        attention_mask=attention_mask,  # 显式指定关键字，避免被误认为 generation_config
        max_new_tokens=128, 
        cache_implementation="legacy",
        do_sample=False ,
        #temperature=1,
        top_p=1,
        repetition_penalty=1.1,
        #logits_processor=[processor], 
        pad_token_id=pad_token_id,
        use_cache=True,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
    )


full_seq = outputs[0].tolist()
new_part = full_seq[len(global_ids):]
print(f"[Decode] generated {len(new_part)} new tokens")
# 验证所有新 token 是否仍在同一层

try:
    result_text=tm.decode(full_seq,layer=None if use_dimensionality_reduction else chosen_layer,skip_special_tokens=True,bMax_frequency=use_dimensionality_reduction)
except Exception as e:
    result_text = f"(分层解码失败: {e})"

print("\n=== 生成结果(多层解码) ===")
print(result_text.get('text'))
print("\n✅ 推理完成 (多层)")
