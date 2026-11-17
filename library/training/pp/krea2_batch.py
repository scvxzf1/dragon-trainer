"""Krea-2 fixed microbatch geometry messages shared by WebUI and runtime."""


def batch_divisibility_message(batch: int, microbatches: int) -> str | None:
    if microbatches > 0 and batch % microbatches == 0:
        return None
    message = (
        f"train batch size {batch} is not divisible by "
        f"pipeline_parallel_microbatches={microbatches}; PP v0 freezes the "
        "wire buffers so every microbatch must be the same size. "
    )
    if batch < 2:
        return message + (
            "Two-stage PP requires train_batch_size / dataset batch_size >= 2. "
            "Set batch_size=2 and --pipeline_parallel_microbatches=2, or use "
            "single-GPU training; microbatches=1 cannot serve two stages."
        )
    divisor = next((n for n in range(2, min(batch, 32) + 1) if batch % n == 0), None)
    if divisor is not None:
        return message + (
            f"Set --pipeline_parallel_microbatches={divisor}; it divides batch "
            f"{batch} and is >= pipeline_parallel_stages=2."
        )
    return message + (
        f"Set train_batch_size / dataset batch_size={batch + batch % 2} and "
        "--pipeline_parallel_microbatches=2, or choose another batch size "
        "with a divisor between 2 and 32."
    )
