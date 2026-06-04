import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path('/home/s471802/nn-gpt')
RUN = ROOT / 'parallel_runs/20260513_1004_struct1_a3_fixed_failure_reward_l40s_rl'
BASE = ROOT / 'out/llm/deepseek-ai/deepseek-coder-6.7b-instruct'
TOKENIZER = ROOT / 'parallel_runs/20260426_1905_main_resume_quality_diversity_std3/grpo_backbone_outputs_trainer/checkpoint-130'
ADAPTER = ROOT / 'out/nngpt/llm/epoch_archive/epoch_before_full_sft_20260510_150909/A3/deepseek-ai/deepseek-coder-6.7b-instruct'
OUT = RUN / 'probe_stop_strings_20260513.json'

max_prompt_length = 3500
max_new_tokens = 1536
num_prompts = int(os.environ.get('PROBE_NUM_PROMPTS', '4'))
rows = []
with (RUN / 'rl_output/generation_samples.jsonl').open() as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
prompts = []
seen = set()
for row in reversed(rows):
    prompt = row.get('prompt') or ''
    if prompt and prompt not in seen:
        prompts.append(prompt)
        seen.add(prompt)
    if len(prompts) >= num_prompts:
        break
prompts = list(reversed(prompts))
if not prompts:
    raise RuntimeError('no prompts found')

print('loading tokenizer', TOKENIZER, flush=True)
tok = AutoTokenizer.from_pretrained(str(TOKENIZER), trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print('tokenizer eos', repr(tok.eos_token), tok.eos_token_id, 'pad', repr(tok.pad_token), tok.pad_token_id, flush=True)

print('loading model', BASE, flush=True)
model = AutoModelForCausalLM.from_pretrained(
    str(BASE),
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    ),
    torch_dtype=torch.float16,
    device_map={'': 'cuda:0'},
)
print('loading adapter', ADAPTER, flush=True)
model = PeftModel.from_pretrained(model, str(ADAPTER), is_trainable=False)
model.eval()
try:
    model.config.use_cache = True
    model.generation_config.use_cache = True
except Exception:
    pass

variants = [
    ('current_like', {}),
    ('explicit_eos_cache', {
        'eos_token_id': tok.eos_token_id,
        'pad_token_id': tok.eos_token_id,
        'use_cache': True,
    }),
    ('stop_forward', {
        'eos_token_id': tok.eos_token_id,
        'pad_token_id': tok.eos_token_id,
        'use_cache': True,
        'stop_strings': ['</forward>'],
    }),
]
results = []
for name, extra in variants:
    print('variant', name, extra, flush=True)
    inputs = tok(
        prompts,
        return_tensors='pt',
        padding=True,
        padding_side='left',
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False,
    ).to('cuda:0')
    t0 = time.time()
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
    )
    gen_kwargs.update(extra)
    if extra.get('stop_strings'):
        gen_kwargs['tokenizer'] = tok
    with torch.inference_mode():
        out = model.generate(**gen_kwargs)
    dt = time.time() - t0
    prompt_len = inputs['input_ids'].shape[-1]
    new_ids = out[:, prompt_len:]
    eos_id = tok.eos_token_id
    lengths = []
    has_eos = []
    tails = []
    jupyter = 0
    stop_seen = 0
    for row in new_ids:
        pos = (row == eos_id).nonzero(as_tuple=False)
        ended = len(pos) > 0
        length = int(pos[0].item()) + 1 if ended else int(row.shape[0])
        text = tok.decode(row[:length], skip_special_tokens=False)
        lengths.append(length)
        has_eos.append(bool(ended))
        tails.append(text[-400:])
        if '<jupyter_' in text:
            jupyter += 1
        if '</forward>' in text:
            stop_seen += 1
    result = {
        'variant': name,
        'seconds': dt,
        'num_prompts': len(prompts),
        'tokens_total': sum(lengths),
        'tokens_per_second': sum(lengths) / dt if dt > 0 else None,
        'lengths': lengths,
        'has_eos': has_eos,
        'has_eos_rate': sum(has_eos) / len(has_eos),
        'clipped_rate': sum(1 for x in lengths if x >= max_new_tokens) / len(lengths),
        'jupyter_count': jupyter,
        'stop_forward_count': stop_seen,
        'tails': tails,
    }
    print(json.dumps({k:v for k,v in result.items() if k != 'tails'}, ensure_ascii=False), flush=True)
    results.append(result)
OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print('wrote', OUT, flush=True)
