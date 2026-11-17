"""Exception-safe state transitions for a batch of training previews."""

from contextlib import contextmanager

import torch


def reset_network_preview_routing(network, *, include_timestep_mask: bool = False) -> None:
    """Clear batch-bound training router/mask buffers back to broadcastable (1, ...) state."""
    if network is None:
        return
    methods = (
        ("clear_timestep_mask",) if include_timestep_mask else ()
    ) + (
        "clear_sigma",
        "clear_fei",
        "clear_routing_weights",
        "clear_freq_routing_weights",
        "clear_content_routing_weights",
        "clear_step_caches",
    )
    for method_name in methods:
        fn = getattr(network, method_name, None)
        if callable(fn):
            fn()


@contextmanager
def preview_lifecycle(dit, network, *, disable_block_swap=False):
    """Enter inference once and restore RNG, swap layout and adapter mode.

    The model's switch methods own preparation. Sequential predictions and
    prompts reuse the forward-only layout until the batch leaves this scope.
    Cleanup failures are reported without replacing the original exception.
    """
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    was_training = getattr(network, "training", True)
    had_preview_net = hasattr(dit, "_preview_network") if dit is not None else False
    prev_preview_net = getattr(dit, "_preview_network", None) if dit is not None else None
    swap_attempted = False
    pause_attempted = False
    pause_completed = False
    paused = False
    error = None
    try:
        if network is not None:
            network.eval()
            reset_network_preview_routing(network, include_timestep_mask=True)
        if dit is not None and network is not None:
            try:
                object.__setattr__(dit, "_preview_network", network)
            except Exception:
                pass
        if disable_block_swap and hasattr(dit, "pause_block_swap"):
            swap_attempted = pause_attempted = True
            paused = dit.pause_block_swap()
            pause_completed = True
        elif hasattr(dit, "switch_block_swap_for_inference"):
            swap_attempted = True
            dit.switch_block_swap_for_inference()
        yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        failures = []

        def attempt(action):
            try:
                return action()
            except Exception as exc:
                failures.append(exc)
                return None

        if dit is not None and network is not None:
            if had_preview_net:
                attempt(lambda: object.__setattr__(dit, "_preview_network", prev_preview_net))
            elif hasattr(dit, "_preview_network"):
                attempt(lambda: delattr(dit, "_preview_network"))
        if swap_attempted:
            # A failed pause can have drained/moved blocks before recording its
            # paused marker. Resume if possible, otherwise rebuild training.
            if pause_attempted and (paused or not pause_completed):
                resumed = (
                    attempt(dit.resume_block_swap)
                    if hasattr(dit, "resume_block_swap") else False
                )
                if not resumed and hasattr(dit, "switch_block_swap_for_training"):
                    attempt(dit.switch_block_swap_for_training)
            elif hasattr(dit, "switch_block_swap_for_training"):
                attempt(dit.switch_block_swap_for_training)
        if network is not None:
            attempt(lambda: reset_network_preview_routing(network))
            attempt(network.train if was_training else network.eval)
        # Restore randomness last, including any consumed by recovery hooks.
        attempt(lambda: torch.set_rng_state(rng_state))
        if cuda_rng_state is not None:
            attempt(lambda: torch.cuda.set_rng_state(cuda_rng_state))
        if failures:
            if error is None:
                error = failures.pop(0)
                for failure in failures:
                    error.add_note(f"Preview cleanup also failed: {failure!r}")
                raise error
            for failure in failures:
                error.add_note(f"Preview cleanup failed: {failure!r}")
