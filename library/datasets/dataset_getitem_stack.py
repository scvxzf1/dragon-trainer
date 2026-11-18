"""Batch stacking helpers for BaseDataset.__getitem__."""

from __future__ import annotations

from typing import Any

import torch


def none_or_stack_elements(tensors_list, converter):
    if (
        len(tensors_list) == 0
        or tensors_list[0] is None
        or len(tensors_list[0]) == 0
        or tensors_list[0][0] is None
    ):
        return None

    result = []
    for i in range(len(tensors_list[0])):
        tensors = [x[i] for x in tensors_list]
        if tensors[0] is None:
            result.append(None)
            continue
        if tensors[0].ndim == 0:
            result.append(torch.stack([converter(x[i]) for x in tensors_list]))
            continue

        min_len = min([len(x) for x in tensors])
        max_len = max([len(x) for x in tensors])

        if min_len == max_len:
            result.append(torch.stack([converter(x) for x in tensors]))
        else:
            tensors = [converter(x) for x in tensors]
            out = torch.zeros(
                (len(tensors), max_len, *tensors[0].shape[1:]),
                dtype=tensors[0].dtype,
                device=tensors[0].device,
            )
            for row_idx, x in enumerate(tensors):
                out[row_idx, : x.shape[0]] = x
            result.append(out)
    return result


def assemble_training_example(
    *,
    dataset,
    custom_attributes,
    loss_weights,
    text_encoder_outputs_list,
    input_ids_list,
    alpha_mask_list,
    images,
    latents_list,
    cond_latents_list,
    qwen_edit_reference_latents,
    captions,
    original_sizes_hw,
    crop_top_lefts,
    target_sizes_hw,
    flippeds,
    inversion_runs_list,
    ip_features_list,
    ip_features_shuffled_list,
    neg_crossattn_list,
    neg_jaccard_list,
    byg_tuple_list,
    bucket,
    image_index,
) -> dict[str, Any]:

    example = {}
    example["custom_attributes"] = custom_attributes
    example["loss_weights"] = torch.tensor(loss_weights, dtype=torch.float32)
    example["text_encoder_outputs_list"] = none_or_stack_elements(
        text_encoder_outputs_list,
        lambda x: (
            x
            if isinstance(x, torch.Tensor)
            else torch.tensor(x, dtype=torch.float32)
        ),
    )
    example["input_ids_list"] = none_or_stack_elements(input_ids_list, lambda x: x)

    none_or_not = [x is None for x in alpha_mask_list]
    if all(none_or_not):
        example["alpha_masks"] = None
    elif any(none_or_not):
        existing_hw = next(
            (tuple(int(d) for d in m.shape[-2:]) for m in alpha_mask_list if m is not None),
            None,
        )
        for i in range(len(alpha_mask_list)):
            if alpha_mask_list[i] is None:
                if existing_hw is not None:
                    mask_hw = existing_hw
                elif images[i] is not None:
                    mask_hw = (images[i].shape[1], images[i].shape[2])
                else:
                    mask_hw = tuple(int(d) for d in latents_list[i].shape[-2:])
                alpha_mask_list[i] = torch.ones(mask_hw, dtype=torch.float32)
        example["alpha_masks"] = torch.stack(alpha_mask_list)
    else:
        example["alpha_masks"] = torch.stack(alpha_mask_list)

    if images[0] is not None:
        images = torch.stack(images)
        images = images.to(memory_format=torch.contiguous_format).float()
    else:
        images = None
    example["images"] = images

    example["latents"] = (
        torch.stack(latents_list) if latents_list[0] is not None else None
    )
    if cond_latents_list and any(t is not None for t in cond_latents_list):
        if not all(t is not None for t in cond_latents_list):
            raise ValueError(
                "Mixed cond_cache_dir batch: some samples have condition "
                "latents and some do not. Split condition-control data into "
                "its own dataset."
            )
        example["cond_latents"] = torch.stack(cond_latents_list)
    else:
        example["cond_latents"] = None
    if qwen_edit_reference_latents and any(t is not None for t in qwen_edit_reference_latents):
        if not all(t is not None for t in qwen_edit_reference_latents):
            raise ValueError("Mixed Qwen edit and non-edit samples in one batch are not supported")
        example["qwen_edit_reference_latents"] = torch.stack(qwen_edit_reference_latents)
    else:
        example["qwen_edit_reference_latents"] = None
    example["captions"] = captions

    example["original_sizes_hw"] = torch.tensor(original_sizes_hw, dtype=torch.long)
    example["crop_top_lefts"] = torch.tensor(crop_top_lefts, dtype=torch.long)
    example["target_sizes_hw"] = torch.tensor(target_sizes_hw, dtype=torch.long)
    example["flippeds"] = flippeds

    example["network_multipliers"] = torch.full(
        (len(captions),), float(dataset.network_multiplier), dtype=torch.float32
    )

    # Inversion runs for functional-loss supervision (postfix-func).
    # If any sample in the batch has inversions loaded, stack them; samples
    # without matching inversions get zero-tensor placeholders and mask=False.
    valid_inversions = [t for t in inversion_runs_list if t is not None]
    if valid_inversions:
        ref_shape = valid_inversions[0].shape  # [N_runs, S, D]
        stacked = torch.stack(
            [
                t if t is not None else torch.zeros(ref_shape, dtype=torch.float32)
                for t in inversion_runs_list
            ],
            dim=0,
        )
        mask = torch.tensor(
            [t is not None for t in inversion_runs_list], dtype=torch.bool
        )
        example["inversion_runs"] = stacked  # [B, N_runs, S, D]
        example["inversion_mask"] = mask  # [B]
    else:
        example["inversion_runs"] = None
        example["inversion_mask"] = None

    # IP-Adapter cached PE features. All samples in a bucket share the
    # training resolution and therefore the same PE bucket -> same T_pe,
    # so a plain stack works.
    if ip_features_list and ip_features_list[0] is not None:
        example["ip_features"] = torch.stack(ip_features_list, dim=0)
    else:
        example["ip_features"] = None
    # Validation-only shuffled (unrelated) reference for the
    # IPAdapterMethodAdapter shuffled_ref baseline. None outside validation.
    if ip_features_shuffled_list and ip_features_shuffled_list[0] is not None:
        example["ip_features_shuffled"] = torch.stack(
            ip_features_shuffled_list, dim=0
        )
    else:
        example["ip_features_shuffled"] = None

    # Soft-tokens contrastive negatives: (B, k, S, D) cached text embeddings.
    # All cached crossattn_emb share the padded sequence length, so a plain
    # stack works. None when no sampler is attached (or any target in the
    # bucket couldn't reach k distinct negatives).
    if neg_crossattn_list and all(t is not None for t in neg_crossattn_list):
        example["neg_crossattn_emb"] = torch.stack(neg_crossattn_list, dim=0)
    else:
        example["neg_crossattn_emb"] = None
    # Per-negative tag-overlap weights (B, k) for jaccard mode; None for
    # shuffled / hard (the loss then runs plain InfoNCE).
    if neg_jaccard_list and all(t is not None for t in neg_jaccard_list):
        example["neg_jaccard"] = torch.stack(neg_jaccard_list, dim=0)
    else:
        example["neg_jaccard"] = None

    # BYG edit-tuple text conditionings. All cached embeddings use padded
    # sequence length, so per-role stacking is shape-stable.
    if byg_tuple_list and all(t is not None for t in byg_tuple_list):
        for role in dataset._byg_roles:
            example[f"byg_{role}_emb"] = torch.stack(
                [t[f"{role}_emb"] for t in byg_tuple_list], dim=0
            )
            if all(f"{role}_mask" in t for t in byg_tuple_list):
                example[f"byg_{role}_mask"] = torch.stack(
                    [t[f"{role}_mask"] for t in byg_tuple_list], dim=0
                )

    if dataset.debug_dataset:
        example["image_keys"] = bucket[image_index : image_index + dataset.batch_size]
    return example
