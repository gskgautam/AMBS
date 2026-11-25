import torch, copy
from transformers import LlamaConfig, LlamaForCausalLM, GemmaConfig, GemmaForCausalLM
from ambs.model import Offsets, install_hook
torch.manual_seed(0)
for name, C, M, extra in [("llama", LlamaConfig, LlamaForCausalLM, {}), ("gemma", GemmaConfig, GemmaForCausalLM, {"head_dim": 16})]:
    cfg = C(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4, **extra)
    m = M(cfg).eval(); ids = torch.randint(3, 300, (2, 7))
    W = m.get_output_embeddings().weight
    with torch.no_grad():
        h = m.model(input_ids=ids).last_hidden_state
        for var in ["constant", "ic", "no_shared_ref", "direct_vector", "logit_bias", "direct_update"]:
            for n in (3, 1):
                mod = Offsets(64, 300, r=4, n=n, variant=var, seed=1)
                for p in mod.parameters(): p.normal_(0, 0.05)
                w = [1.0] if n == 1 else [0.2, 0.3, 0.5]
                ref = sum(wi * mod.logits(i, h, W) for i, wi in enumerate(w))  # pooled = weighted logits (Eq. 9)
                hd = install_hook(m, mod, w); got = m(input_ids=ids).logits; hd.remove()
                assert torch.allclose(ref, got, atol=1e-4), (name, var, n, (ref-got).abs().max())
                pen = mod.penalty(h.reshape(-1, 64)); assert torch.isfinite(pen)
        # layer intervention: hook output equals manual insertion at block l
        mod = Offsets(64, 300, r=4, n=3, variant="constant", seed=1)
        for p in mod.parameters(): p.normal_(0, 0.05)
        w = [0, 1, 0]
        hd = install_hook(m, mod, w, layer=2); got = m(input_ids=ids).logits; hd.remove()
        m2 = copy.deepcopy(m); c = mod.offset(1, torch.zeros(1, 64))[0]
        def fwd_hook(module, inp, out):
            return (out[0] + c,) + tuple(out[1:]) if isinstance(out, tuple) else out + c
        hd2 = m2.model.layers[1].register_forward_hook(fwd_hook); ref = m2(input_ids=ids).logits; hd2.remove()
        assert torch.allclose(ref, got, atol=1e-4)
    print(f"{name}: all 6 variants (N=3 and N=1) — generation hook == pooled training logits (Eq. 9); layer hook OK")
