import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np

from hqq_utils import AutoHQQHFModel, get_size_of_model
from hqq.utils.patching import recommended_inductor_config_setter

from quant_cfg import get_quant_config


#####################################################################
# === SPEC NOTICE ===
# Only "load model" and "generate" function selection can be modified.
# DO NOT change PPL calculation, timing, or throughput logic.
#####################################################################

# === (Optional) Define your own custom generate function. ===
# This is useful if you want full control over KV cache and generation steps.
# You can modify this function to suit your needs.
# By default, we use model.generate() for simplicity and general use.

def generate(model, input_ids, past_key_values, max_new_tokens):
    input_ids = input_ids.clone()
    tput = None
    # Run an initial forward pass to compute and store the static KV cache
    with torch.no_grad():
        outputs = model.prefill_forward(input_ids, past_key_values=past_key_values, position_ids=None, attention_mask=None, cache_position=None, logits_to_keep=1)
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    # Generate tokens one by one using a for loop and update the KV cache
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Compute position_ids using the current sequence length
            pos = input_ids.shape[1]
            cache_position = torch.arange(pos, pos+1, device=input_ids.device, dtype=torch.long)

            # Run the model on the last token using the cached key-value pairs
            outputs = model(
                next_token,
                past_key_values=past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position
            )
            logits = outputs.logits

            # Greedily select the token with the highest probability
            next_token = torch.argmax(logits, dim=-1)

            # Append the predicted token to the generated sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Update the KV cache for the next iteration
            past_key_values = outputs.past_key_values

    return input_ids

def apply_mlp_pruning(model, prune_ratio=0.3):
    """
    在每个 transformer layer 的 MLP 中间维度上，剪掉前 prune_ratio 比例的通道。
    - prune_ratio: 保留比例之外要剪掉的比例，例如 0.3 表示剪掉 30%。
    """
    from transformers.modeling_utils import prune_linear_layer
    # 1) unwrap PEFT/DeepSpeed/DataParallel
    transformer = model
    if hasattr(transformer, "module"):
        transformer = transformer.module
    if hasattr(transformer, "base_model"):
        transformer = transformer.base_model
    if hasattr(transformer, "model") and hasattr(transformer.model, "layers"):
        transformer = transformer.model

    # 2) 对每一层 MLP 做 prune
    for layer in transformer.layers:
        # 三个投影：gate_proj (dim D_ff→D)，up_proj (dim D_ff→D)，down_proj (dim D→D_ff)
        gate = layer.mlp.gate_proj
        up   = layer.mlp.up_proj
        down = layer.mlp.down_proj

        # 以 up_proj 的输出维度 D_ff 作为剪枝基准
        D_ff = up.weight.shape[0]
        k = int(D_ff * prune_ratio)             # 要剪的通道数
        if k <= 0:
            continue

        idxs = torch.arange(k, device=up.weight.device)

        # gate_proj 和 up_proj 在 dim=0（输出维度）上 prune
        prune_linear_layer(gate, idxs, dim=0)
        prune_linear_layer(up,   idxs, dim=0)

        # down_proj 在 dim=1（输入维度）上 prune
        prune_linear_layer(down, idxs, dim=1)

    # 3) （可选）如果有 ffn_hidden_size 在 config 里，也同步减小
    cfg = transformer.config
    if hasattr(cfg, "ffn_hidden_size"):
        cfg.ffn_hidden_size = int(cfg.ffn_hidden_size * (1 - prune_ratio))

    print(f"[MLP Prune] 剪掉每层 {prune_ratio*100:.0f}% MLP 隐藏通道")

def apply_head_pruning(model, heads_per_layer=2):
    """
    在每个 attention layer 里，剪掉前 heads_per_layer 个 heads。
    自动 unwrap PEFT/DeepSpeed/DataParallel，然后对真正的 transformer.layers 生效。
    """
    from transformers.modeling_utils import prune_linear_layer
    # 1) Unwrap 各种包装器，得到底层有 .layers 的对象
    transformer = model
    if hasattr(transformer, "module"):        # DataParallel / DeepSpeed Engine
        transformer = transformer.module
    if hasattr(transformer, "base_model"):    # PEFT 包装器
        transformer = transformer.base_model
    if hasattr(transformer, "model") and hasattr(transformer.model, "layers"):
        transformer = transformer.model

    if not hasattr(transformer, "layers"):
        raise AttributeError(f"[HeadPrune] 无法在 {type(transformer)} 上找到 .layers")

    # 2) 读取当前 head 数与 head_dim
    config    = transformer.config
    num_heads = config.num_attention_heads
    head_dim  = config.hidden_size // num_heads

    # 3) 计算要剪掉的 feature index
    heads = list(range(heads_per_layer))
    prune_idxs = [h * head_dim + i for h in heads for i in range(head_dim)]
    prune_idx_tensor = torch.tensor(sorted(prune_idxs), dtype=torch.long, device=transformer.layers[0].self_attn.q_proj.weight.device)

    # 4) 对每层的 Q/K/V/O 投影做 prune
    for layer in transformer.layers:
        attn = layer.self_attn
        for proj in ("q_proj","k_proj","v_proj"):
            prune_linear_layer(getattr(attn, proj), prune_idx_tensor, dim=0)
        # O 投影要在 dim=1 上 prune
        prune_linear_layer(attn.o_proj, prune_idx_tensor, dim=1)

    # 5) 更新 config
    config.num_attention_heads = num_heads - heads_per_layer
    print(f"[HeadPrune] 每层剪掉 {heads_per_layer} 个 heads；新 num_attention_heads = {config.num_attention_heads}")

def apply_flash_attn_patch(model):
    from flash_attn import flash_attn_func
    config = model.config
    num_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads

    for layer in model.model.layers:
        attn = layer.self_attn

        Wq = attn.q_proj
        Wk = attn.k_proj
        Wv = attn.v_proj
        Wo = attn.o_proj

        def new_forward(
            self_attn,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False
        ):
            B, L, D = hidden_states.size()

            # Project to Q, K, V
            q = Wq(hidden_states).view(B, L, num_heads, head_dim).transpose(1, 2)
            k = Wk(hidden_states).view(B, L, num_heads, head_dim).transpose(1, 2)
            v = Wv(hidden_states).view(B, L, num_heads, head_dim).transpose(1, 2)

            # FlashAttention
            out = flash_attn_func(q, k, v, causal=True)

            # Merge heads
            out = out.transpose(1, 2).reshape(B, L, D)
            output = Wo(out)

            return output, None, None if use_cache else None

        # Replace the forward method of the attention block
        attn.forward = new_forward.__get__(attn, attn.__class__)

def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)
    
    nsamples = test_enc.numel() // model.seqlen
    nlls = []  
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        
        with torch.no_grad():
            lm_logits = model(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    return ppl.item()

def main():
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)
    recommended_inductor_config_setter()
    
    max_new_tokens = 256    # Number of new tokens to generate
    device = 'cuda:0'
    backend = 'gemlite'
    
    ### === TODO: Load your model (you may change this part) ===
    model_name = "meta-llama/Llama-3.2-3B-Instruct"   
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    #from flash_attn.flash_attn_interface import flash_attn_unpadded_func
    #model.config._attn_implementation = "flash_attention"
    #####################################
    

    model.eval() 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # === (Optional) Uncomment the following lines if using the custom generate() function. ===

    model.prefill_forward = model.forward
    model.forward = torch.compile(model.forward, mode='max-autotune', dynamic=False, fullgraph=True)

    # Purning
    apply_head_pruning(model, heads_per_layer=2)
    apply_mlp_pruning(model, prune_ratio=0.2)
    model.model.blocks = model.model.layers

    # Quant
    quant_config = get_quant_config(model)
    AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device=device)

    # Lora
    from peft import get_peft_model, LoraConfig, TaskType
    lora_config = LoraConfig(
        r=8,                      # Low-rank dim
        lora_alpha=16,            # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Tweak depending on model architecture
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.eval()
    model.merge_and_unload()  # Merge LoRA weights for inference only


    from hqq.utils.patching import prepare_for_inference
    prepare_for_inference(model, backend=backend) 
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    
    warmup_prompt = "Explain what AI is."
    inputs = tokenizer(warmup_prompt, return_tensors="pt", return_attention_mask=False).to(device)
    input_ids = inputs["input_ids"]
    # attention_mask = inputs["attention_mask"]
    
    # === (Optional) Set up StaticCache for manual KV cache management ===
    past_key_values = StaticCache(
        config=model.config, 
        max_batch_size=1, 
        max_cache_len=max_new_tokens + 16, 
        device=model.device, 
        dtype=torch.float16
    )
    ####################################################################
    
    # from torch.cuda import amp

    for i in tqdm(range(5), desc="Warm Up..."):
        #  === Default: use model.generate() for end-to-end warm-up === 
        # _ = model.generate(
        #     input_ids=input_ids,
        #      attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )
        
        # === (Optional) Use custom generate() if uncommented ===
        # with amp.autocast(enabled=True):
        generated = generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()
        
    prompt = "How to learn a new language?"
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(device)
    input_ids = inputs["input_ids"]
    # attention_mask = inputs["attention_mask"]
    tputs = []
    time_record = []
    for _ in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # === Default: Use model.generate() for end-to-end timing === 
        # generated = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )
        
        # === Optional: Use custom generate() if uncommented ===
        # with amp.autocast(enabled=True):
        generated = generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()

        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        # tput = max_new_tokens / (elapsed_ms / 1000)
        tput = generated[0][input_ids.shape[1]:].shape[0]/(elapsed_ms / 1000)
        time_record.append(elapsed_ms / 1000)
        tputs.append(tput)
        
    response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)
    print(f'Prompt: {prompt}\nResponse: {response}\n')
    
    print(f'Time Record: {time_record}')
    print(f'Throughput Record: {tputs} toks/s\n')

    ### Your final throughput result ###
    print(f'Throughput: {org_tput} toks/s')
    ppl = evaluate_ppl(model, tokenizer, device)
    print(f"Perplexity (PPL): {ppl}")
    
    # Save results to CSV
    import csv
    rounded_tput = round(org_tput, 1)
    ppl = round(ppl, 2)

    with open("result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "value"])
        writer.writerow([0, ppl])
        writer.writerow([1, rounded_tput])
        
if __name__ == '__main__':
    main()