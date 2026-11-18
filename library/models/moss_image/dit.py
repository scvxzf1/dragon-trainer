# SPDX-License-Identifier: Apache-2.0
# Adapted from MOSS-Image-SFT/pipeline_moss_image.py, HF revision
# daf706d2e1c7a6369c17b6d1658671aee6db1ebe (Apache-2.0).
"""MOSS DiT: official state-dict names and per-sample true-length attention."""
import torch
from torch import nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from .math import (
    _RMSNorm,
    _TextProjection,
    _Block,
    _RoPE,
    _validate_skip_layers,
    run_moss_block_sequence,
    time_sinusoid,
)
from .block_swap import MOSSBlockSwapMixin

class MOSSImageTransformer(MOSSBlockSwapMixin, ModelMixin, ConfigMixin):
    """MOSSImage DiT with training-compatible state-dict names and SDPA attention.

    Each sample is evaluated at its true text length, including text refinement;
    padded tokens cannot influence either text or joint attention.
    """
    @register_to_config
    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42,
                 text_dim=4096, freq_dim=256, latent_channels=128, rope_base=10000.0):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError('Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.')
        self.blocks_to_swap = 0
        self.offloader = None
        self._paused_blocks_to_swap = None
        self.gradient_checkpointing = False
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([_Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True
        for block in (*self.layers, *self.text_embedder.blocks):
            block.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        for block in (*self.layers, *self.text_embedder.blocks):
            block.gradient_checkpointing = False

    def set_attn_mode(self, mode: str) -> None:
        if mode not in ('torch', 'sdpa'):
            raise ValueError('MOSS supports only torch/sdpa attention.')

    def load_training_state_dict(self, state_dict, *, prefix=''):
        """Load a consolidated raw/EMA model mapping, strictly; no silent key drops."""
        if prefix:
            if any(not key.startswith(prefix) for key in state_dict):
                raise ValueError('All keys must start with the explicit prefix.')
            state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}
        return self.load_state_dict(state_dict, strict=True)

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask=None, return_dict=True, skip_layers=None):
        skip_layers = _validate_skip_layers(skip_layers, len(self.layers))
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.config.latent_channels:
            raise ValueError('hidden_states must be [B, latent_channels, H, W].')
        batch, _, height, width = hidden_states.shape
        if self.blocks_to_swap and skip_layers and not self.offloader.forward_only:
            raise ValueError('MOSS skipped layers require inference mode when block swap is active.')
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError('encoder_hidden_states must be [B, S, text_dim].')
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2] or (
            mask.dtype != torch.bool and not torch.all((mask == 0) | (mask == 1))
        ):
            raise ValueError('encoder_attention_mask must be a binary [B, S] mask.')
        mask_bool = mask if mask.dtype == torch.bool else mask.bool()
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1)
        times = times.expand(batch)
        # Training computes the sinusoid in float64 and feeds float32 into its MLP.
        embedding = time_sinusoid(times, self.config.freq_dim, times.device)
        time = self.time_embedding(embedding.to(self.time_embedding[0].weight.dtype)).float()
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype)).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype)).view(batch, 2, -1)
        image_tokens = height * width
        samples: list[torch.Tensor] = []
        ropes: list[torch.Tensor] = []
        rope_cache: dict[int, torch.Tensor] = {}
        if batch > 1 and hidden_states.device.type == 'cuda':
            videos_all = self.video_embedder(
                hidden_states.flatten(2).transpose(1, 2).to(self.video_embedder.weight.dtype).contiguous()
            )
            videos = list(videos_all.split(1, dim=0))
        else:
            videos = [
                self.video_embedder(
                    hidden_states[index:index + 1].flatten(2).transpose(1, 2).to(self.video_embedder.weight.dtype).contiguous()
                )
                for index in range(batch)
            ]
        for index in range(batch):
            context = encoder_hidden_states[index:index + 1, mask_bool[index]]
            text_len = int(context.shape[1])
            if text_len == 0:
                raise ValueError('Each sample needs at least one valid text token.')
            video = videos[index]
            rope = rope_cache.get(text_len)
            if rope is None:
                rope = self.rope(height, width, text_len, hidden_states.device).to(video.dtype)
                rope_cache[text_len] = rope
            text = self.text_embedder(context, rope[:, image_tokens:])
            samples.append(torch.cat((video, text), 1))
            ropes.append(rope)

        samples = run_moss_block_sequence(
            self.layers,
            samples,
            ropes,
            mod,
            skip_layers=skip_layers,
            offloader=self.offloader,
            blocks_to_swap=self.blocks_to_swap,
        )

        if batch > 1 and hidden_states.device.type == 'cuda':
            video_out = torch.cat([s[:, :image_tokens] for s in samples], dim=0)
            video_out = self.final_norm_video(video_out.to(self.final_norm_video.weight.dtype))
            video_out = video_out * (1 + head[:, 1, None]) + head[:, 0, None]
            video_out = self.final_linear_video(video_out.to(self.final_linear_video.weight.dtype))
            sample = video_out.transpose(1, 2).reshape(batch, -1, height, width)
        else:
            outputs = []
            for index, x in enumerate(samples):
                x = self.final_norm_video(x[:, :image_tokens].to(self.final_norm_video.weight.dtype))
                x = x * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
                x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
                outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
            sample = torch.cat(outputs)
        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)
