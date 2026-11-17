"""Routing-state helpers for the LoRA-family network facade."""

from typing import Dict, List, Optional

import torch

from networks.lora_modules import _sigma_sinusoidal_features


def wire_shared_sigma_buffers(network) -> None:
    sigma_loras: List[torch.nn.Module] = []
    by_dim: Dict[int, List[torch.nn.Module]] = {}
    for lora in network.unet_loras + network.text_encoder_loras:
        if "_sigma" not in lora._buffers:
            continue
        sigma_loras.append(lora)
        d = int(getattr(lora, "sigma_feature_dim", 0))
        if d > 0 and "_sigma_features" in lora._buffers:
            by_dim.setdefault(d, []).append(lora)
    network._sigma_aware_loras = sigma_loras
    network._sigma_aware_loras_by_dim = by_dim
    if not sigma_loras:
        network._shared_sigma = None
        network._shared_sigma_features: Dict[int, torch.Tensor] = {}
        return

    shared_sigma = sigma_loras[0]._buffers["_sigma"]
    for lora in sigma_loras:
        lora._buffers["_sigma"] = shared_sigma
    network._shared_sigma = shared_sigma

    network._shared_sigma_features = {}
    for dim, loras in by_dim.items():
        shared_feat = loras[0]._buffers["_sigma_features"]
        for lora in loras:
            lora._buffers["_sigma_features"] = shared_feat
        network._shared_sigma_features[dim] = shared_feat


def wire_shared_fei_buffers(network) -> None:
    fei_loras: List[torch.nn.Module] = []
    by_dim: Dict[int, List[torch.nn.Module]] = {}
    for lora in network.unet_loras + network.text_encoder_loras:
        d = int(getattr(lora, "fei_feature_dim", 0))
        if d <= 0:
            continue
        if "_fei" not in lora._buffers:
            continue
        fei_loras.append(lora)
        by_dim.setdefault(d, []).append(lora)
    network._fei_aware_loras = fei_loras
    network._fei_aware_loras_by_dim = by_dim
    if not fei_loras:
        network._shared_fei: Dict[int, torch.Tensor] = {}
        return

    network._shared_fei = {}
    for dim, loras in by_dim.items():
        shared_feat = loras[0]._buffers["_fei"]
        for lora in loras:
            lora._buffers["_fei"] = shared_feat
        network._shared_fei[dim] = shared_feat


def wire_shared_routing_buffers(network) -> None:
    routing_loras: List[torch.nn.Module] = []
    for lora in network.unet_loras + network.text_encoder_loras:
        if "_routing_weights" not in lora._buffers:
            continue
        routing_loras.append(lora)
    network._routing_aware_loras = routing_loras
    if not routing_loras:
        network._shared_routing_weights: Optional[torch.Tensor] = None
        return

    canonical = routing_loras[0]._buffers["_routing_weights"]
    for lora in routing_loras:
        lora._buffers["_routing_weights"] = canonical
    network._shared_routing_weights = canonical


def wire_shared_content_routing_buffers(network) -> None:
    content_loras: List[torch.nn.Module] = []
    for lora in network.unet_loras + network.text_encoder_loras:
        if "_content_routing_weights" not in lora._buffers:
            continue
        content_loras.append(lora)
    network._content_aware_loras = content_loras
    if not content_loras:
        network._shared_content_routing_weights: Optional[torch.Tensor] = None
        return

    canonical = content_loras[0]._buffers["_content_routing_weights"]
    for lora in content_loras:
        lora._buffers["_content_routing_weights"] = canonical
    network._shared_content_routing_weights = canonical


def wire_shared_freq_routing_buffers(network) -> None:
    freq_loras: List[torch.nn.Module] = []
    for lora in network.unet_loras + network.text_encoder_loras:
        if "_freq_routing_weights" not in lora._buffers:
            continue
        freq_loras.append(lora)
    network._chimera_aware_loras = freq_loras
    if not freq_loras:
        network._shared_freq_routing_weights: Optional[torch.Tensor] = None
        return

    canonical = freq_loras[0]._buffers["_freq_routing_weights"]
    for lora in freq_loras:
        lora._buffers["_freq_routing_weights"] = canonical
    network._shared_freq_routing_weights = canonical


def set_timestep_mask(
    network, timesteps: torch.Tensor, max_timestep: float = 1.0
) -> None:
    if not network.cfg.use_timestep_mask:
        return

    max_rank = network.cfg.lora_dim
    mask = getattr(network, "_shared_timestep_mask", None)
    if mask is None or mask.device != timesteps.device:
        mask = torch.zeros(1, max_rank, device=timesteps.device)
        network._shared_timestep_mask = mask
        network._timestep_mask_arange = torch.arange(
            max_rank, device=timesteps.device
        )
        for lora in network.text_encoder_loras + network.unet_loras:
            lora._timestep_mask = mask

    t = timesteps.float().mean()
    frac = ((max_timestep - t) / max_timestep).clamp(min=0.0, max=1.0)
    r = (
        frac.pow(network.cfg.alpha_rank_scale)
        * (max_rank - network.cfg.min_rank)
        + network.cfg.min_rank
    )
    r = r.clamp(max=float(max_rank))
    mask.copy_((network._timestep_mask_arange < r).to(mask.dtype).unsqueeze(0))


def set_reft_timestep_mask(
    network, timesteps: torch.Tensor, max_timestep: float = 1.0
) -> None:
    if not network.cfg.use_timestep_mask:
        return
    refts = network.text_encoder_refts + network.unet_refts
    if not refts:
        return
    reft_dim = network.cfg.reft_dim

    mask = getattr(network, "_shared_reft_mask", None)
    if mask is None or mask.device != timesteps.device:
        mask = torch.zeros(1, reft_dim, device=timesteps.device)
        network._shared_reft_mask = mask
        network._reft_mask_arange = torch.arange(reft_dim, device=timesteps.device)
        for reft in refts:
            reft._timestep_mask = mask

    t = timesteps.float().mean()
    frac = ((max_timestep - t) / max_timestep).clamp(min=0.0, max=1.0)
    r = frac.pow(network.cfg.alpha_rank_scale) * (reft_dim - 1) + 1
    r = r.clamp(max=float(reft_dim))
    mask.copy_((network._reft_mask_arange < r).to(mask.dtype).unsqueeze(0))


def clear_timestep_mask(network) -> None:
    shared = getattr(network, "_shared_timestep_mask", None)
    if shared is not None:
        shared.fill_(1.0)
    shared_reft = getattr(network, "_shared_reft_mask", None)
    if shared_reft is not None:
        shared_reft.fill_(1.0)


def set_sigma(network, sigmas: torch.Tensor) -> None:
    sigmas = sigmas.detach()
    network._last_sigma = sigmas
    if not (
        network.cfg.router_source == "sigma"
        or network.cfg.specialize_experts_by_sigma_buckets
    ):
        return
    sigma_loras = network._sigma_aware_loras
    if not sigma_loras:
        return

    canonical = sigma_loras[0]._buffers["_sigma"]
    cast = sigmas.to(dtype=canonical.dtype, device=canonical.device)
    needs_rebind = network._shared_sigma is not canonical or canonical.shape != cast.shape
    if needs_rebind:
        new_sigma = cast.detach().clone()
        for lora in sigma_loras:
            lora._buffers["_sigma"] = new_sigma
        network._shared_sigma = new_sigma
        shared_sigma = new_sigma
    else:
        canonical.copy_(cast)
        shared_sigma = canonical

    for dim, loras in network._sigma_aware_loras_by_dim.items():
        canonical_feat = loras[0]._buffers["_sigma_features"]
        feat = _sigma_sinusoidal_features(shared_sigma, dim).detach()
        cast_feat = feat.to(dtype=canonical_feat.dtype, device=canonical_feat.device)
        feat_needs_rebind = (
            network._shared_sigma_features.get(dim) is not canonical_feat
            or canonical_feat.shape != cast_feat.shape
        )
        if feat_needs_rebind:
            new_feat = cast_feat.clone()
            for lora in loras:
                lora._buffers["_sigma_features"] = new_feat
            network._shared_sigma_features[dim] = new_feat
        else:
            canonical_feat.copy_(cast_feat)


def clear_sigma(network) -> None:
    network._last_sigma = None
    if not network._sigma_aware_loras:
        return
    sigma_loras = network._sigma_aware_loras
    canonical = sigma_loras[0]._buffers["_sigma"]
    if canonical.shape != (1,):
        canonical = torch.zeros(1, dtype=canonical.dtype, device=canonical.device)
        for lora in sigma_loras:
            lora._buffers["_sigma"] = canonical
        network._shared_sigma = canonical
    else:
        if network._shared_sigma is not canonical:
            for lora in sigma_loras:
                lora._buffers["_sigma"] = canonical
            network._shared_sigma = canonical
        canonical.zero_()
    for dim, loras in network._sigma_aware_loras_by_dim.items():
        canonical_feat = loras[0]._buffers["_sigma_features"]
        zero_feat = _sigma_sinusoidal_features(canonical, dim)
        cast_feat = zero_feat.to(dtype=canonical_feat.dtype, device=canonical_feat.device)
        if canonical_feat.shape == cast_feat.shape:
            if network._shared_sigma_features.get(dim) is not canonical_feat:
                for lora in loras:
                    lora._buffers["_sigma_features"] = canonical_feat
                network._shared_sigma_features[dim] = canonical_feat
            canonical_feat.copy_(cast_feat)
        else:
            new_feat = cast_feat.detach().clone()
            for lora in loras:
                lora._buffers["_sigma_features"] = new_feat
            network._shared_sigma_features[dim] = new_feat


def set_fei(network, fei: torch.Tensor) -> None:
    fei = fei.detach()
    has_per_layer_fei = bool(getattr(network, "_fei_aware_loras", None))
    global_fei_router = (
        network.global_router
        if (
            network.global_router is not None
            and network.cfg.router_source == "fei"
            and not network.cfg.route_per_layer
        )
        else None
    )
    chimera_freq_router = (
        network.freq_router
        if (
            getattr(network, "freq_router", None) is not None
            and getattr(network, "_chimera_aware_loras", None)
        )
        else None
    )
    if not (
        has_per_layer_fei
        or global_fei_router is not None
        or chimera_freq_router is not None
    ):
        return
    if not (
        network.use_fei_router
        or global_fei_router is not None
        or chimera_freq_router is not None
    ):
        return

    if has_per_layer_fei:
        for dim, loras in network._fei_aware_loras_by_dim.items():
            canonical = loras[0]._buffers["_fei"]
            cast = fei.to(dtype=canonical.dtype, device=canonical.device)
            if cast.dim() == 1:
                cast = cast.unsqueeze(0)
            if cast.shape[-1] != dim:
                raise ValueError(
                    f"set_fei: fei.shape[-1]={cast.shape[-1]} != "
                    f"fei_feature_dim={dim}"
                )
            current_shared = network._shared_fei.get(dim)
            needs_rebind = (
                current_shared is not canonical or canonical.shape != cast.shape
            )
            if needs_rebind:
                new_fei = cast.detach().clone()
                for lora in loras:
                    lora._buffers["_fei"] = new_fei
                network._shared_fei[dim] = new_fei
            else:
                canonical.copy_(cast)

    if global_fei_router is not None:
        gates = global_fei_router(fei)
        network.set_routing_weights(gates)

    if chimera_freq_router is not None:
        sigma = network._last_sigma
        if sigma is None:
            raise RuntimeError(
                "ChimeraHydra FreqRouter requires set_sigma to fire before "
                "set_fei within the same step (apply_router_conditioning "
                "preserves this order -- check custom call sites)."
            )
        sigma_dim = int(network.cfg.sigma_feature_dim)
        sigma_feat = _sigma_sinusoidal_features(sigma, sigma_dim)
        fei_cast = fei.to(device=sigma_feat.device, dtype=sigma_feat.dtype)
        if fei_cast.dim() == 1:
            fei_cast = fei_cast.unsqueeze(0)
        router_in = torch.cat([fei_cast, sigma_feat], dim=-1)
        freq_gates = chimera_freq_router(router_in)
        network.set_freq_routing_weights(freq_gates)


def clear_fei(network) -> None:
    if not getattr(network, "_fei_aware_loras", None):
        return
    for dim, loras in network._fei_aware_loras_by_dim.items():
        canonical = loras[0]._buffers["_fei"]
        if canonical.shape != (1, dim):
            canonical = torch.zeros(
                (1, dim), dtype=canonical.dtype, device=canonical.device
            )
            for lora in loras:
                lora._buffers["_fei"] = canonical
            network._shared_fei[dim] = canonical
        else:
            current_shared = network._shared_fei.get(dim)
            if current_shared is not canonical:
                for lora in loras:
                    lora._buffers["_fei"] = canonical
                network._shared_fei[dim] = canonical
            canonical.zero_()


def set_routing_weights(network, weights: torch.Tensor) -> None:
    if not getattr(network, "_routing_aware_loras", None):
        return
    routing_loras = network._routing_aware_loras
    canonical_buf = routing_loras[0]._buffers["_routing_weights"]
    w = weights.to(dtype=canonical_buf.dtype, device=canonical_buf.device)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    for lora in routing_loras:
        lora._routing_weights = w
    network._shared_routing_weights = w


def clear_routing_weights(network) -> None:
    if not getattr(network, "_routing_aware_loras", None):
        return
    routing_loras = network._routing_aware_loras
    canonical = routing_loras[0]._buffers["_routing_weights"]
    E = int(canonical.shape[-1])
    if canonical.shape != (1, E):
        canonical = torch.full(
            (1, E), 1.0 / max(E, 1), dtype=canonical.dtype, device=canonical.device
        )
        for lora in routing_loras:
            lora._routing_weights = canonical
        network._shared_routing_weights = canonical
    else:
        if network._shared_routing_weights is not canonical:
            for lora in routing_loras:
                lora._routing_weights = canonical
            network._shared_routing_weights = canonical
        canonical.fill_(1.0 / max(E, 1))


def set_crossattn_routing(network, crossattn_emb: torch.Tensor) -> None:
    if network.global_router is None or not getattr(
        network, "use_crossattn_router", False
    ):
        return
    gates = network.global_router(crossattn_emb)
    network.set_routing_weights(gates)


def set_freq_routing_weights(network, weights: torch.Tensor) -> None:
    if not getattr(network, "_chimera_aware_loras", None):
        return
    freq_loras = network._chimera_aware_loras
    canonical_buf = freq_loras[0]._buffers["_freq_routing_weights"]
    w = weights.to(dtype=canonical_buf.dtype, device=canonical_buf.device)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    for lora in freq_loras:
        lora._freq_routing_weights = w
    network._shared_freq_routing_weights = w


def clear_freq_routing_weights(network) -> None:
    if not getattr(network, "_chimera_aware_loras", None):
        return
    freq_loras = network._chimera_aware_loras
    canonical = freq_loras[0]._buffers["_freq_routing_weights"]
    K_f = int(canonical.shape[-1])
    if canonical.shape != (1, K_f):
        canonical = torch.full(
            (1, K_f), 1.0 / max(K_f, 1), dtype=canonical.dtype, device=canonical.device
        )
        for lora in freq_loras:
            lora._freq_routing_weights = canonical
        network._shared_freq_routing_weights = canonical
    else:
        if network._shared_freq_routing_weights is not canonical:
            for lora in freq_loras:
                lora._buffers["_freq_routing_weights"] = canonical
            network._shared_freq_routing_weights = canonical
        canonical.fill_(1.0 / max(K_f, 1))


def set_content(network, crossattn_emb: torch.Tensor) -> None:
    if network.content_router is None:
        return
    if not getattr(network, "_content_aware_loras", None):
        return
    gates = network.content_router(crossattn_emb)
    network.set_content_routing_weights(gates)


def set_content_routing_weights(network, weights: torch.Tensor) -> None:
    if not getattr(network, "_content_aware_loras", None):
        return
    content_loras = network._content_aware_loras
    canonical_buf = content_loras[0]._buffers["_content_routing_weights"]
    w = weights.to(dtype=canonical_buf.dtype, device=canonical_buf.device)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    for lora in content_loras:
        lora._content_routing_weights = w
    network._shared_content_routing_weights = w


def clear_content_routing_weights(network) -> None:
    if not getattr(network, "_content_aware_loras", None):
        return
    content_loras = network._content_aware_loras
    canonical = content_loras[0]._buffers["_content_routing_weights"]
    K_c = int(canonical.shape[-1])
    if canonical.shape != (1, K_c):
        canonical = torch.full(
            (1, K_c), 1.0 / max(K_c, 1), dtype=canonical.dtype, device=canonical.device
        )
        for lora in content_loras:
            lora._content_routing_weights = canonical
        network._shared_content_routing_weights = canonical
    else:
        if network._shared_content_routing_weights is not canonical:
            for lora in content_loras:
                lora._buffers["_content_routing_weights"] = canonical
            network._shared_content_routing_weights = canonical
        canonical.fill_(1.0 / max(K_c, 1))


def clear_step_caches(network) -> None:
    network._last_sigma = None
    network._router_stats_cache = None
    network._chimera_router_stats_cache = None
    for lora in network.unet_loras + network.text_encoder_loras:
        if hasattr(lora, "_last_gate"):
            lora._last_gate = None
    if network.global_router is not None:
        network.global_router._last_gates = None
        network.global_router._last_input = None
        network.global_router._last_fei = None
    if getattr(network, "freq_router", None) is not None:
        network.freq_router._last_gates = None
        network.freq_router._last_input = None
    if getattr(network, "content_router", None) is not None:
        network.content_router._last_gates = None
        network.content_router._last_input = None
