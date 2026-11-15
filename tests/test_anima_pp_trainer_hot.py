"""Anima PP trainer-level hot test (frozen contract §6/§7; GPU acceptance).

Fixed nodeid: ``test_two_rank_cuda_trainer_resume`` — the same shape as the
MOSS hot test: the real production entry (``train_session`` →
``run_pp_trainer_session`` → ``build_anima_pipeline_runtime``), no mocks,
straight + resume in one session pair.

GPU boundary (frozen contract §6):

* only ``GPU-b4b014b2-953b-28e8-b8c1-425c4a401165`` and
  ``GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061`` (the two V100s) are allowed;
  the GTX 1050 must never be selected (it hangs);
* single-round budget ``timeout 60``; failed rounds are re-run only with
  recorded evidence, never in an unbounded loop;
* a CPU/Gloo cold pass (``tests/test_anima_pp_trainer_checkpoint.py``) is a
  precondition before any CUDA round.

This module stays **collect-only on a CPU machine**: the hot test self-skips
unless both V100 UUIDs are visible, so the CPU CI collection never allocates
a GPU or runs the two-rank CUDA path.
"""

from __future__ import annotations

import os

import pytest
import torch

V100_UUIDS = (
    "GPU-b4b014b2-953b-28e8-b8c1-425c4a401165",
    "GPU-379b8c19-d965-f6fa-d70d-8a9b876ff061",
)


def _v100_pair_visible() -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False
    # torch exposes uuid as torch._C._CUuuid whose str() is the canonical
    # dashed form; comparing objects against uuid.UUID would never match.
    visible = {str(torch.cuda.get_device_properties(i).uuid) for i in range(2)}
    return visible == {value.split("GPU-")[1] for value in V100_UUIDS}


pytestmark = pytest.mark.skipif(
    not _v100_pair_visible(),
    reason="the anima PP trainer hot test runs only on the two authorized "
    "V100s (GPU-b4b014b2…/GPU-379b8c19…); CPU/Gloo coverage lives in "
    "tests/test_anima_pp_trainer_checkpoint.py",
)


def test_two_rank_cuda_trainer_resume(tmp_path):
    """Straight + resume on the real entry over two CUDA ranks (§6 budget).

    The GPU acceptance task launches this nodeid with
    ``CUDA_VISIBLE_DEVICES=<the two V100 UUIDs>`` and ``timeout 60`` after
    ``gpu-gate.sh --count 2 --json`` reports both cards free; a GTX 1050 in
    the visibility set is a hard stop, not a fallback.
    """

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    for uuid in V100_UUIDS:
        assert uuid in cuda_visible or not cuda_visible, (
            "the hot test must be launched with the two authorized V100 UUIDs"
        )
    from tests.anima_pp_trainer_hot_support import run_cuda_session_pair

    run_cuda_session_pair(tmp_path)
