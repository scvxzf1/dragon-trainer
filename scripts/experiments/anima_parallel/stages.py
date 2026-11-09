"""Owned Anima stages for the fixed-shape, plain-MLP-LoRA PP experiment."""

from __future__ import annotations

import torch
from torch import nn

from .common import lora_block_index
from .pipeline_parallel import finish_model, prepare_block_inputs, run_local_blocks


class AnimaOwnedStage(nn.Module):
    """Consume a model/network and retain only this stage's registered modules."""

    def __init__(self, model, network, *, rank: int, split: int, rope):
        super().__init__()
        from networks.lora_modules.lora import LoRAModule

        count = len(model.blocks)
        if rank not in (0, 1) or not 0 < split < count:
            raise ValueError("PP requires rank 0/1 and a nonempty two-stage split")
        if model.blocks_to_swap or model.enable_pooled_text_modulation:
            raise ValueError("PP probe does not support swap or pooled modulation")
        if bool(model._mod_guidance_delta.count_nonzero()):
            raise ValueError("PP probe requires disabled modulation guidance")
        if not model.use_adaln_lora:
            raise ValueError("PP probe requires AdaLN LoRA conditioning")
        if model._native_flatten or model.attn_mode not in {"flash", "torch"}:
            raise ValueError(
                "PP probe requires uncompiled native-grid flash/torch attention"
            )
        expected_names = {
            f"lora_unet_blocks_{index}_mlp_layer{layer}"
            for index in range(count)
            for layer in (1, 2)
        }
        if {lora.lora_name for lora in network.unet_loras} != expected_names:
            raise ValueError(
                "PP probe requires exactly the complete plain MLP adapter set"
            )
        for lora in network.unet_loras:
            if type(lora) is not LoRAModule or any(
                getattr(lora, name, None)
                for name in ("dropout", "rank_dropout", "module_dropout")
            ):
                raise ValueError(
                    "PP probe supports only plain MLP LoRA without dropout"
                )
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("PP probe requires frozen base weights")
        self.input_shapes = None
        self.rank = rank
        self.split = split
        self.total_blocks = count
        self.start, self.end = (0, split) if rank == 0 else (split, count)
        owned = set(range(self.start, self.end))
        self.adapters = nn.ModuleDict(
            {
                lora.lora_name: lora
                for lora in network.unet_loras
                if lora_block_index(lora) in owned
            }
        )
        if len(self.adapters) != 2 * len(owned):
            raise ValueError(
                "PP probe requires both plain MLP LoRAs in every owned block"
            )
        model.blocks = nn.ModuleList(list(model.blocks)[self.start : self.end])
        keep = {"blocks"}
        keep.update(
            {"x_embedder", "pos_embedder", "t_embedder", "t_embedding_norm"}
            if rank == 0
            else {"final_layer"}
        )
        for name in list(model._modules):
            if name not in keep:
                delattr(model, name)
        self.model = model
        self.register_buffer("rope_cos", rope[0].detach().clone(), persistent=False)
        self.register_buffer("rope_sin", rope[1].detach().clone(), persistent=False)

    def _blocks(self, hidden, embedding, context, adaln):
        from networks import attention_dispatch

        params = attention_dispatch.AttentionParams.create_attention_params(
            self.model.attn_mode,
            self.model.attn_softmax_scale,
            v100_flash_stability=self.model.v100_flash_stability,
            debug_finite_checks=self.model.debug_finite_checks,
        )
        return run_local_blocks(
            self.model,
            hidden,
            embedding,
            context,
            params,
            {
                "rope_cos_sin": (self.rope_cos, self.rope_sin),
                "adaln_lora_B_T_3D": adaln,
                "use_fp32": False,
            },
        )

    def forward(self, first, second, context, fourth):
        if self.input_shapes is not None:
            shapes = tuple(
                tuple(value.shape) for value in (first, second, context, fourth)
            )
            if shapes != self.input_shapes:
                raise ValueError("PP probe received a different fixed-shape microbatch")
        if self.rank == 0:
            hidden, embedding, _, kwargs = prepare_block_inputs(
                self.model, first, second, context, fourth
            )
            adaln = kwargs["adaln_lora_B_T_3D"]
            hidden = self._blocks(hidden, embedding, context, adaln)
            return hidden, embedding, context, adaln
        # Receiving-side drop for the frozen conditioning (see
        # ``library.training.pp.stages.FrozenBoundaryPipelineStage``): torch forces
        # every receive buffer to ``requires_grad_(True)``, so the cached
        # conditioning has to be detached here or ``get_bwd_send_ops`` would try to
        # send a gradient back for a tensor the earlier stage does not accept one
        # for. ``hidden`` stays connected: it is the only slot that carries a
        # useful gradient.
        embedding, frozen_context, adaln = (
            second.detach(),
            context.detach(),
            fourth.detach(),
        )
        hidden = self._blocks(first, embedding, frozen_context, adaln)
        return finish_model(self.model, hidden, embedding, adaln)

    def adapter_state(self):
        return {
            key: value.detach().cpu().clone()
            for key, value in self.adapters.state_dict().items()
        }

    def topology(self):
        return {
            "version": 1,
            "family": "anima",
            "rank": self.rank,
            "world_size": 2,
            "total_blocks": self.total_blocks,
            "split": self.split,
            "range": [self.start, self.end],
        }


def stage_examples(model, inputs):
    """Derive static wire metadata before transferring module ownership."""
    with torch.no_grad():
        hidden, embedding, _, kwargs = prepare_block_inputs(model, *inputs)
        adaln = kwargs["adaln_lora_B_T_3D"]
        output = finish_model(model, hidden, embedding, adaln)
    boundary = (
        hidden.detach().requires_grad_(),
        embedding.detach(),
        inputs[2].detach(),
        adaln.detach(),
    )
    return boundary, output.detach().requires_grad_(), kwargs["rope_cos_sin"]


def build_schedule(
    module, inputs, boundary, output, *, microbatches: int, schedule: str = "1f1b"
):
    from torch.distributed.pipelining import Schedule1F1B, ScheduleGPipe
    from torch.nn import functional as F

    from library.training.pp.stages import (
        WIRE_FROZEN_INDICES,
        FrozenBoundaryPipelineStage,
    )

    if microbatches < 2 or schedule not in {"1f1b", "gpipe"}:
        # torch 2.10 enforces ``n_microbatches >= num_stages`` in
        # ``PipelineScheduleSingle.__init__``, the shared base of Schedule1F1B and
        # ScheduleGPipe, so microbatches=1 cannot be run on a 2-stage pipeline by
        # switching schedules.
        raise ValueError(
            "PP probe needs microbatches >= 2 for a 2-stage pipeline in torch 2.10 "
            f"(got {microbatches}) or an unknown schedule {schedule!r}"
        )
    rank = module.rank
    module.input_shapes = tuple(
        tuple(value.shape) for value in (inputs if rank == 0 else boundary)
    )

    class RecordingPipelineStage(FrozenBoundaryPipelineStage):
        """Count the p2p traffic the benchmark reports (measurement only)."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.communication = {"p2p_calls": 0, "p2p_tensor_bytes": 0}

        def _record_ops(self, ops):
            self.communication["p2p_calls"] += len(ops)
            self.communication["p2p_tensor_bytes"] += sum(
                op.tensor.numel() * op.tensor.element_size() for op in ops
            )
            return ops

        def get_fwd_send_ops(self, chunk_id):
            return self._record_ops(super().get_fwd_send_ops(chunk_id))

        def get_fwd_recv_ops(self, chunk_id):
            return self._record_ops(super().get_fwd_recv_ops(chunk_id))

        def get_bwd_send_ops(self, chunk_id):
            return self._record_ops(super().get_bwd_send_ops(chunk_id))

        def get_bwd_recv_ops(self, chunk_id):
            return self._record_ops(super().get_bwd_recv_ops(chunk_id))

    stage = RecordingPipelineStage(
        module,
        rank,
        2,
        next(module.parameters()).device,
        input_args=inputs if rank == 0 else boundary,
        output_args=boundary if rank == 0 else output,
        frozen_input_indices=() if rank == 0 else WIRE_FROZEN_INDICES,
        frozen_output_indices=WIRE_FROZEN_INDICES if rank == 0 else (),
    )
    schedule_type = (
        ScheduleGPipe if microbatches < 3 or schedule == "gpipe" else Schedule1F1B
    )
    return schedule_type(
        stage,
        n_microbatches=microbatches,
        loss_fn=lambda prediction, target: F.mse_loss(
            prediction.float(), target.float()
        ),
        scale_grads=True,
    )
