import os
from pathlib import Path
import torch.nn.functional as F
import torch
import torch.nn as nn
from modelscope import snapshot_download
from transformers import MambaModel
from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize
from layers.HierarchicalPrompt import HierarchicalDynamicPrompt
import transformers

transformers.logging.set_verbosity_error()


def resolve_modelscope_model_dir(model_id):
    owner, name = model_id.split("/", 1)
    cache_roots = []
    env_cache = os.getenv("MODELSCOPE_CACHE")
    if env_cache:
        cache_roots.append(Path(env_cache))
    cache_roots.append(Path.home() / ".cache" / "modelscope")
    cache_roots.append(Path.home() / ".cache" / "modelscope" / "hub")
    candidate_suffixes = [
        Path("models") / owner / name,
        Path(owner) / name,
        Path("hub") / "models" / owner / name,
    ]

    def is_valid_model_dir(candidate):
        return (
            candidate.is_dir()
            and (candidate / "config.json").exists()
            and (
                (candidate / "model.safetensors").exists()
                or (candidate / "pytorch_model.bin").exists()
            )
        )

    for cache_root in cache_roots:
        for suffix in candidate_suffixes:
            candidate = cache_root / suffix
            if is_valid_model_dir(candidate):
                return str(candidate)
    return None


class FlattenHead(nn.Module):

    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class MultiScaleTimeFreqFusion(nn.Module):

    def __init__(
        self,
        seq_len,
        d_model,
        d_llm,
        small_patch,
        small_stride,
        large_patch,
        large_stride,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embedding_small = PatchEmbedding(
            d_model, small_patch, small_stride, dropout
        )
        patch_nums_ref = (seq_len + small_stride - small_patch) // small_stride + 1
        self.patch_embedding_large = PatchEmbedding(
            d_model, large_patch, large_stride, dropout
        )
        patch_nums_large = (seq_len + large_stride - large_patch) // large_stride + 1
        if patch_nums_ref != patch_nums_large:
            self.need_token_align = True
            self.large_token_align = nn.Linear(patch_nums_large, patch_nums_ref)
        else:
            self.need_token_align = False
        self.adapter_conv = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
        self.adapter_conv_norm = nn.LayerNorm(d_model)
        self.adapter_up = nn.Linear(d_model, d_llm)
        self.adapter_gate = nn.Linear(d_model, d_llm)
        self.adapter_norm = nn.LayerNorm(d_llm)
        self.adapter_drop = nn.Dropout(dropout)
        freq_bins = seq_len // 2 + 1
        self.freq_token_align = nn.Linear(freq_bins, patch_nums_ref)
        self.freq_token_norm = nn.LayerNorm(patch_nums_ref)
        self.freq_feat_expand = nn.Linear(1, d_model)
        self.freq_conv = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model)
        self.freq_conv_norm = nn.LayerNorm(d_model)
        self.freq_up = nn.Linear(d_model, d_llm)
        self.freq_gate = nn.Linear(d_model, d_llm)
        self.freq_out_norm = nn.LayerNorm(d_llm)
        self.freq_drop = nn.Dropout(dropout)
        self.fusion_proj = nn.Linear(3 * d_llm, d_llm)
        self.patch_nums = patch_nums_ref

    def _gated_conv_adapt(self, x):
        residual = x
        x = self.adapter_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.adapter_conv_norm(F.silu(x) + residual)
        return self.adapter_drop(
            self.adapter_norm(self.adapter_up(x) * F.silu(self.adapter_gate(x)))
        )

    def _freq_encode(self, x):
        B, N, T = x.shape
        x_amp = torch.fft.rfft(x.reshape(B * N, T), dim=-1).abs()
        x_tok = F.silu(self.freq_token_norm(self.freq_token_align(x_amp)))
        x_feat = self.freq_feat_expand(x_tok.unsqueeze(-1))
        residual = x_feat
        x_feat = self.freq_conv(x_feat.transpose(1, 2)).transpose(1, 2)
        x_feat = self.freq_conv_norm(F.silu(x_feat) + residual)
        out = self.freq_up(x_feat) * F.silu(self.freq_gate(x_feat))
        return self.freq_drop(self.freq_out_norm(out))

    def forward(self, x):
        branches = []
        n_vars = None
        enc_out_small, n_vars = self.patch_embedding_small(x.to(torch.bfloat16))
        enc_out_small = self._gated_conv_adapt(enc_out_small)
        branches.append(enc_out_small)
        enc_out_large, n_vars_large = self.patch_embedding_large(x.to(torch.bfloat16))
        enc_out_large = self._gated_conv_adapt(enc_out_large)
        if n_vars is None:
            n_vars = n_vars_large
        if self.need_token_align:
            enc_out_large = self.large_token_align(
                enc_out_large.transpose(1, 2)
            ).transpose(1, 2)
        branches.append(enc_out_large)
        enc_out_freq = self._freq_encode(x.float()).to(branches[0].dtype)
        branches.append(enc_out_freq)
        enc_out = torch.cat(branches, dim=-1)
        enc_out = self.fusion_proj(enc_out)
        return (enc_out, n_vars)


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        model_id = "AI-ModelScope/mamba-130m-hf"
        model_dir = resolve_modelscope_model_dir(model_id)
        if model_dir is None:
            snapshot_kwargs = {}
            modelscope_cache = os.getenv("MODELSCOPE_CACHE")
            if modelscope_cache:
                snapshot_kwargs["cache_dir"] = modelscope_cache
            model_dir = snapshot_download(model_id, **snapshot_kwargs)
        self.llm_model = MambaModel.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        self.d_llm = self.llm_model.config.hidden_size
        if hasattr(configs, "llm_layers") and configs.llm_layers < len(
            self.llm_model.layers
        ):
            self.llm_model.layers = self.llm_model.layers[: configs.llm_layers]
        for param in self.llm_model.parameters():
            param.requires_grad = False
        self.d_stat = 12
        _total = getattr(configs, "prompt_tokens", 16)
        _num_meta = min(4, _total // 4)
        _remaining = _total - _num_meta
        _num_pattern = _remaining // 2
        _num_numerical = _remaining - _num_pattern
        self.hierarchical_prompt = HierarchicalDynamicPrompt(
            d_stat=self.d_stat,
            d_llm=self.d_llm,
            num_meta=_num_meta,
            num_pattern=_num_pattern,
            num_numerical=_num_numerical,
            num_pattern_types=getattr(configs, "num_pattern_types", 8),
        )
        self.mstf = MultiScaleTimeFreqFusion(
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            d_llm=self.d_llm,
            small_patch=configs.small_patch,
            small_stride=configs.small_stride,
            large_patch=configs.large_patch,
            large_stride=configs.large_stride,
            dropout=configs.dropout,
        )
        self.patch_nums = self.mstf.patch_nums
        self.head_nf = self.d_ff * self.patch_nums
        self.output_projection = FlattenHead(
            n_vars=configs.enc_in,
            nf=self.head_nf,
            target_window=self.pred_len,
            head_dropout=configs.dropout,
        )
        self.normalize_layers = Normalize(1, affine=True)

    def readout_tokens(self, hidden):
        hidden = hidden[:, -self.patch_nums :, :]
        return hidden[:, :, : self.d_ff]

    def forward(
        self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_prompt_info=False
    ):
        if return_prompt_info:
            dec_out, prompt_info = self.forecast(
                x_enc, x_mark_enc, x_dec, x_mark_dec, return_prompt_info=True
            )
            return (dec_out[:, -self.pred_len :, :], prompt_info)
        else:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_prompt_info=False):
        B, T, N = x_enc.size()
        x_enc = self.normalize_layers(x_enc, "norm")
        x_flat = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T)
        stats = self._compute_statistics(x_flat)
        prompt_embeddings, prompt_info = self.hierarchical_prompt(
            stats.to(torch.float32)
        )
        x_enc_p = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.mstf(x_enc_p)
        prompt_embeddings = prompt_embeddings.to(enc_out.dtype)
        combined_embeddings = torch.cat([prompt_embeddings, enc_out], dim=1)
        mamba_out = self.llm_model(inputs_embeds=combined_embeddings).last_hidden_state
        dec_out = self.readout_tokens(mamba_out)
        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1])
        )
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums :])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        out = self.normalize_layers(dec_out, "denorm")
        if return_prompt_info:
            return (out, prompt_info)
        return out

    def _compute_statistics(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True) + 1e-08
        x_norm = (x - x_mean) / x_std
        stats = [
            x_norm.min(dim=-1).values,
            x_norm.max(dim=-1).values,
            x_norm.median(dim=-1).values,
            x_norm.mean(dim=-1),
            x_norm.std(dim=-1),
            self._linear_slope(x_norm),
            self._skewness(x_norm),
            self._kurtosis(x_norm),
            self._autocorr_lag(x_norm, lag=1),
            self._autocorr_lag(x_norm, lag=2),
            x_norm.diff(dim=-1).std(dim=-1),
            x_norm.max(dim=-1).values - x_norm.min(dim=-1).values,
        ]
        return torch.stack(stats, dim=-1)

    def _linear_slope(self, x):
        T = x.shape[-1]
        t = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype).unsqueeze(0)
        t_mean = 0.5
        x_mean = x.mean(dim=-1, keepdim=True)
        slope = ((t - t_mean) * (x - x_mean)).sum(dim=-1) / (
            (t - t_mean).pow(2).sum() + 1e-08
        )
        return slope

    def _skewness(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-08
        return ((x - mean) / std).pow(3).mean(dim=-1)

    def _kurtosis(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-08
        return ((x - mean) / std).pow(4).mean(dim=-1) - 3.0

    def _autocorr_lag(self, x, lag=1):
        mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - mean
        var = (x_centered**2).sum(dim=-1) + 1e-08
        cov = (x_centered[:, :-lag] * x_centered[:, lag:]).sum(dim=-1)
        return cov / var
