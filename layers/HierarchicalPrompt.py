import torch
import torch.nn as nn


class NumericalTokenEncoder(nn.Module):

    def __init__(self, d_stat, d_llm, num_tokens=6):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_encoders = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d_stat, 128), nn.GELU(), nn.Linear(128, d_llm))
                for _ in range(num_tokens)
            ]
        )
        self.position_bias = nn.Parameter(torch.randn(num_tokens, d_llm) * 0.01)

    def forward(self, s):
        tokens = []
        for i, encoder in enumerate(self.token_encoders):
            t = encoder(s) + self.position_bias[i]
            tokens.append(t.unsqueeze(1))
        return torch.cat(tokens, dim=1)


class HierarchicalDynamicPrompt(nn.Module):

    def __init__(
        self,
        d_stat=12,
        d_llm=768,
        num_meta=4,
        num_pattern=6,
        num_numerical=6,
        num_pattern_types=8,
    ):
        super().__init__()
        self.num_meta = num_meta
        self.num_pattern = num_pattern
        self.num_numerical = num_numerical
        self.d_llm = d_llm
        self.meta_base = nn.Parameter(torch.randn(num_meta, d_llm) * 0.02)
        self.meta_modulator = nn.Sequential(nn.Linear(d_stat, num_meta * 2))
        self.pattern_prototypes = nn.Parameter(
            torch.randn(num_pattern_types, num_pattern, d_llm) * 0.02
        )
        self.pattern_router = nn.Sequential(
            nn.Linear(d_stat, 64), nn.GELU(), nn.Linear(64, num_pattern_types)
        )
        self.numerical_encoder = NumericalTokenEncoder(
            d_stat=d_stat, d_llm=d_llm, num_tokens=num_numerical
        )
        self.norm = nn.LayerNorm(d_llm)

    def forward(self, s):
        B = s.shape[0]
        meta_base = self.meta_base.unsqueeze(0).expand(B, -1, -1)
        with torch.autocast(device_type=s.device.type, enabled=False):
            mod = self.meta_modulator(s.float())
            scale, shift = mod.chunk(2, dim=-1)
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
            meta_tokens = meta_base * (1 + scale) + shift
        router_logits = self.pattern_router(s)
        router_weights = self._top3_softmax(router_logits)
        pattern_tokens = torch.einsum(
            "bn,npd->bpd", router_weights, self.pattern_prototypes
        )
        numerical_tokens = self.numerical_encoder(s)
        prompt_tokens = torch.cat(
            [meta_tokens, pattern_tokens, numerical_tokens], dim=1
        )
        return (
            self.norm(prompt_tokens),
            {
                "router_weights": router_weights,
                "meta_tokens": meta_tokens,
                "pattern_tokens": pattern_tokens,
                "numerical_tokens": numerical_tokens,
            },
        )

    def _top3_softmax(self, logits):
        topk_val, topk_idx = logits.topk(3, dim=-1)
        topk_w = torch.softmax(topk_val, dim=-1)
        weights = torch.zeros(logits.shape, dtype=topk_w.dtype, device=logits.device)
        weights.scatter_(-1, topk_idx, topk_w)
        return weights
