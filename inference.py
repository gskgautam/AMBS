import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, logging
from huggingface_hub import login

@torch.no_grad()
def inference_single(model, tokenizer, device, prompt, steering_vector, layer):
    hook_handle = model.model.layers[layer].register_forward_hook(
        lambda module, inp, out: (out[0] + steering_vector,)
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )

    res = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):]
    hook_handle.remove()
    return res

@torch.no_grad()
def infer_triple(model, tokenizer, device, steering_vectors, layer, q1, q2, q3):
    prompts = [f'\n\n[INST]Q{i+1}: {q}\nAnswer: [/INST]' for i,q in enumerate([q1, q2, q3])]
    toks0 = tokenizer(prompts[0][2:], return_tensors="pt").input_ids
    toks1 = tokenizer(prompts[1], return_tensors="pt").input_ids
    toks2 = tokenizer(prompts[2], return_tensors="pt").input_ids

    continue0 = True
    continue1 = True
    continue2 = True
    num_toks = 0
    while num_toks < 512 and (continue0 or continue1 or continue2):
        cat = []
        lens = []
        if continue0:
            cat.append(toks0)
            lens.append(toks0.shape[1])

        if continue1:
            cat.append(toks1)
            lens.append(toks1.shape[1])

        if continue2:
            cat.append(toks2)
            lens.append(toks2.shape[1])

        combined_input_ids = torch.cat(cat, dim=1).to(device)
        combined_attention_mask = torch.ones_like(combined_input_ids)

        outputs = model(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
            output_hidden_states=True
        )
    
        hidden_state_at_split = outputs.hidden_states[layer]
        chunks = torch.split(hidden_state_at_split, lens, dim=1)
        final_logits = []

        for i, chunk in enumerate(chunks):
            current_hidden_state = chunk + steering_vectors[i]
            chunk_len = chunk.shape[1]
    
            chunk_mask = torch.ones(1, chunk_len, device=device)
            chunk_pos_ids = torch.arange(0, chunk_len, device=device).unsqueeze(0)

            for layer_idx in range(layer, model.config.num_hidden_layers):
                layer_module = model.model.layers[layer_idx]
                current_hidden_state = layer_module(
                    current_hidden_state,
                    attention_mask=chunk_mask,
                    position_ids=chunk_pos_ids
                )[0]
    
            final_state = model.model.norm(current_hidden_state)
            logits = model.lm_head(final_state)
            predicted_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            [toks0, toks1, toks2][i] = torch.cat([[toks0, toks1, toks2][i], [predicted_token_id]], dim=1)
            num_toks += 1
            if predicted_token_id == tokenizer.eos_token_id:
                [continue0, continue1, continue2][i] = False

    return tokenizer.decode(torch.cat([toks0, toks1, toks2], dim=1),
                            skip_special_tokens=True)