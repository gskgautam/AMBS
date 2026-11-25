"""AMBS offsets and every control / ablation in the paper, as modules acting on the frozen backbone readout.

Notation (paper): h = shared readout (after final norm, before W_o), W_o = frozen output head, N pathways.

variant          offset / logit change for pathway i                         paper
---------------  -----------------------------------------------------------  ------------------------------
constant         c_i = b_i (I + A_i B_i)                                       Eq. 5 (AMBS, constant special case)
ic               c_i(h) = h A_i B_i + b_i                                      Input-Conditioned Offsets (AMBS-IC)
no_shared_ref    c_i(h) = (h + b_i) A_i B_i + b_i   (deviation not referenced) Table 6 "No Shared Reference"
direct_vector    c_i = v_i                                                     Eq. 13 (Table 3 control D)
logit_bias       logits += q_i,  q_i in R^|V|                                  Eq. 18 (Table 8)
direct_update    logits += (h A_i) B'_i,  B'_i in R^{r x |V|}                  Table 6 "Direct Update"
Init: B = 0 (B' = 0), A ~ N(0, std 1/sqrt(d)), b ~ N(0, std 0.01) (or 0: "zero-start"), v_i = b_i init, q_i = 0.
N = 1 gives the single-path ablation (factorized) and, with direct_vector, the pooled single-offset control (Eq. 15).
"""
import math

import torch
import torch.nn as nn

HIDDEN_VARIANTS = ("constant", "ic", "no_shared_ref", "direct_vector")
LOGIT_VARIANTS = ("logit_bias", "direct_update")
INPUT_DEPENDENT = ("ic", "no_shared_ref", "direct_update")


class Offsets(nn.Module):
    def __init__(self, d, vocab, r=8, n=3, variant="ic", seed=42, b_init="normal", freeze_b=False):
        super().__init__()
        assert variant in HIDDEN_VARIANTS + LOGIT_VARIANTS, variant
        self.d, self.vocab, self.r, self.n, self.variant = d, vocab, r, n, variant
        g = torch.Generator().manual_seed(seed)
        b0 = torch.randn(n, d, generator=g) * 0.01 if b_init == "normal" else torch.zeros(n, d)
        if variant in ("constant", "ic", "no_shared_ref"):
            self.A = nn.Parameter(torch.randn(n, d, r, generator=g) / math.sqrt(d))
            self.B = nn.Parameter(torch.zeros(n, r, d))
            self.b = nn.Parameter(b0, requires_grad=not freeze_b)
        elif variant == "direct_vector":
            self.v = nn.Parameter(b0)  # same effective initial offset as AMBS (B = 0 => c = b)
        elif variant == "logit_bias":
            self.q = nn.Parameter(torch.zeros(n, vocab))
        elif variant == "direct_update":
            self.A = nn.Parameter(torch.randn(n, d, r, generator=g) / math.sqrt(d))
            self.Bv = nn.Parameter(torch.zeros(n, r, vocab))

    @property
    def is_logit(self):
        return self.variant in LOGIT_VARIANTS

    # ---- hidden-space offsets ----
    def offset(self, i, h):
        """Offset of pathway i for readouts h (..., d)."""
        v = self.variant
        if v == "constant":
            c = self.b[i] + (self.b[i] @ self.A[i]) @ self.B[i]
            return c.expand_as(h)
        if v == "ic":
            return (h @ self.A[i]) @ self.B[i] + self.b[i]
        if v == "no_shared_ref":
            return ((h + self.b[i]) @ self.A[i]) @ self.B[i] + self.b[i]
        if v == "direct_vector":
            return self.v[i].expand_as(h)
        raise ValueError(v)

    # ---- logit-space changes ----
    def logit_delta(self, i, h):
        if self.variant == "logit_bias":
            return self.q[i].expand(*h.shape[:-1], self.vocab)
        if self.variant == "direct_update":
            return (h @ self.A[i]) @ self.Bv[i]
        raise ValueError(self.variant)

    def logits(self, i, h, W):
        """Training logits of pathway i: W_o (h + c_i(h)) or W_o h + delta_i(h)."""
        if self.is_logit:
            return h @ W.T + self.logit_delta(i, h)
        return (h + self.offset(i, h)) @ W.T

    # ---- pooled (fixed-weight geometric pooling, Eq. 9) ----
    def mixed_offset(self, h, w):
        out = torch.zeros_like(h)
        for i in range(self.n):
            if w[i] != 0:
                out = out + w[i] * self.offset(i, h)
        return out

    def mixed_logit_delta(self, h, w):
        out = None
        for i in range(self.n):
            if w[i] != 0:
                t = w[i] * self.logit_delta(i, h)
                out = t if out is None else out + t
        return out if out is not None else torch.zeros(*h.shape[:-1], self.vocab, device=h.device, dtype=h.dtype)

    # ---- pairwise consistency penalty (Eq. 10 / input-conditioned form) ----
    def penalty(self, h):
        if self.n < 2:
            return torch.zeros((), device=h.device)
        f = self.logit_delta if self.is_logit else self.offset
        if self.variant in INPUT_DEPENDENT:
            cs = [f(i, h) for i in range(self.n)]
            reduce = lambda x: (x ** 2).sum(-1).mean()  # averaged over tokens
        else:
            h1 = h[:1]
            cs = [f(i, h1)[0] for i in range(self.n)]
            reduce = lambda x: (x ** 2).sum()
        tot = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                tot = tot + reduce(cs[i] - cs[j])
        return tot

    def pathway_params(self, i):
        """Parameter slices belonging to pathway i (for sequential training)."""
        return [getattr(self, k) for k in ("A", "B", "b", "v", "q", "Bv") if hasattr(self, k)]


class NormHook:
    """Forward hook on the final norm: adds the pooled hidden-space offset (generation)."""

    def __init__(self, mod, w):
        self.mod, self.w = mod, w

    def __call__(self, module, inputs, output):
        return output + self.mod.mixed_offset(output.float(), self.w).to(output.dtype)


class HeadHook:
    """Forward hook on lm_head: adds the pooled logit-space change (generation; logit_bias, direct_update)."""

    def __init__(self, mod, w):
        self.mod, self.w = mod, w

    def __call__(self, module, inputs, output):
        h = inputs[0].float()
        if output.shape[1] != h.shape[1]:  # only the last positions were projected
            h = h[:, -output.shape[1]:]
        return output + self.mod.mixed_logit_delta(h, self.w).to(output.dtype)


class LayerHook:
    """Forward hook on decoder block (layer-l intervention): adds a constant offset to the block's hidden output."""

    def __init__(self, mod, w):
        self.mod, self.w = mod, w

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs = hs + self.mod.mixed_offset(hs.float(), self.w).to(hs.dtype)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs


def final_norm(model):
    return model.model.norm


def install_hook(model, mod, w, layer=None):
    """Attach the right generation hook; layer=None means the paper's default (l = L, after final norm)."""
    if layer is not None:
        return model.model.layers[layer - 1].register_forward_hook(LayerHook(mod, w))
    if mod.is_logit:
        return model.lm_head.register_forward_hook(HeadHook(mod, w))
    return final_norm(model).register_forward_hook(NormHook(mod, w))


# Backwards-compatible alias used by earlier code/tests
AMBSOffsets = Offsets
