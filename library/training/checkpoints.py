import argparse
from dataclasses import dataclass
import json
import logging
import math
import os
import shutil
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# checkpoint filename templates
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
DEFAULT_EPOCH_NAME = "epoch"
DEFAULT_LAST_OUTPUT_NAME = "last"

DEFAULT_STEP_NAME = "at"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"
STEP_DIFFUSERS_DIR_NAME = "{}-step{:08d}"

CHECKPOINT_STATE_NAME = "{}-checkpoint-state"
CHECKPOINT_FILE_NAME = "{}-checkpoint"
CHECKPOINT_EPOCH_STATE_NAME = "{}-checkpoint-{:06d}-state"
CHECKPOINT_EPOCH_FILE_NAME = "{}-checkpoint-{:06d}"
CHECKPOINT_LATEST_STATE_FILE_NAME = "{}-checkpoint-latest.json"


@dataclass(frozen=True)
class ResumeStartPlan:
    initial_step: int
    epoch_to_start: int
    steps_from_state: Optional[int] = None
    global_step: int = 0


def default_if_none(value, default):
    return default if value is None else value


def _save_last_keep_count(value) -> Optional[int]:
    if value is None:
        return None
    try:
        keep_count = int(value)
    except (TypeError, ValueError):
        return None
    if keep_count < 0:
        return None
    if keep_count == 0:
        return 1
    return keep_count


def get_epoch_ckpt_name(args: argparse.Namespace, ext: str, epoch_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
    return EPOCH_FILE_NAME.format(model_name, epoch_no) + ext


def get_step_ckpt_name(args: argparse.Namespace, ext: str, step_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)
    return STEP_FILE_NAME.format(model_name, step_no) + ext


def get_last_ckpt_name(args: argparse.Namespace, ext: str):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return model_name + ext


def get_remove_epoch_no(args: argparse.Namespace, epoch_no: int):
    keep_count = _save_last_keep_count(getattr(args, "save_last_n_epochs", None))
    if keep_count is None:
        return None

    save_every_n_epochs = getattr(args, "save_every_n_epochs", None)
    if not save_every_n_epochs:
        return None
    remove_epoch_no = epoch_no - save_every_n_epochs * keep_count
    if remove_epoch_no <= 0:
        return None
    return remove_epoch_no


def get_remove_step_no(args: argparse.Namespace, step_no: int):
    last_n_steps = getattr(args, "save_last_n_steps", None)
    if last_n_steps is None:
        return None
    try:
        last_n_steps = int(last_n_steps)
    except (TypeError, ValueError):
        return None
    if last_n_steps < 0:
        return None

    remove_step_no = step_no - last_n_steps - 1
    remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)
    if remove_step_no < 0:
        return None
    return remove_step_no


def save_sd_model_on_epoch_end_or_stepwise_common(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    sd_saver,
    diffusers_saver,
):
    if on_epoch_end:
        epoch_no = epoch + 1
        saving = epoch_no % args.save_every_n_epochs == 0
        if not saving:
            return

        model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
        remove_no = get_remove_epoch_no(args, epoch_no)
    else:
        model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)
        epoch_no = epoch
        remove_no = get_remove_step_no(args, global_step)

    os.makedirs(args.output_dir, exist_ok=True)
    if save_stable_diffusion_format:
        ext = ".safetensors" if use_safetensors else ".ckpt"

        if on_epoch_end:
            ckpt_name = get_epoch_ckpt_name(args, ext, epoch_no)
        else:
            ckpt_name = get_step_ckpt_name(args, ext, global_step)

        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        logger.info("")
        logger.info(f"saving checkpoint: {ckpt_file}")
        sd_saver(ckpt_file, epoch_no, global_step)

        # remove older checkpoints
        if remove_no is not None:
            if on_epoch_end:
                remove_ckpt_name = get_epoch_ckpt_name(args, ext, remove_no)
            else:
                remove_ckpt_name = get_step_ckpt_name(args, ext, remove_no)

            remove_ckpt_file = os.path.join(args.output_dir, remove_ckpt_name)
            if os.path.exists(remove_ckpt_file):
                logger.info(f"removing old checkpoint: {remove_ckpt_file}")
                os.remove(remove_ckpt_file)

    else:
        if on_epoch_end:
            out_dir = os.path.join(
                args.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, epoch_no)
            )
        else:
            out_dir = os.path.join(
                args.output_dir, STEP_DIFFUSERS_DIR_NAME.format(model_name, global_step)
            )

        logger.info("")
        logger.info(f"saving model: {out_dir}")
        diffusers_saver(out_dir)

        # remove older checkpoints
        if remove_no is not None:
            if on_epoch_end:
                remove_out_dir = os.path.join(
                    args.output_dir,
                    EPOCH_DIFFUSERS_DIR_NAME.format(model_name, remove_no),
                )
            else:
                remove_out_dir = os.path.join(
                    args.output_dir,
                    STEP_DIFFUSERS_DIR_NAME.format(model_name, remove_no),
                )

            if os.path.exists(remove_out_dir):
                logger.info(f"removing old model: {remove_out_dir}")
                shutil.rmtree(remove_out_dir)

    if args.save_state:
        if on_epoch_end:
            save_and_remove_state_on_epoch_end(args, accelerator, epoch_no)
        else:
            save_and_remove_state_stepwise(args, accelerator, global_step)


def save_and_remove_state_on_epoch_end(
    args: argparse.Namespace,
    accelerator,
    epoch_no,
    created_state_dirs: set[str] | None = None,
    global_step: Optional[int] = None,
    loader_cursor: Optional[dict] = None,
    rank_digest: Optional[dict] = None,
):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)

    logger.info("")
    logger.info(f"saving state at epoch {epoch_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(
        args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no)
    )
    # This path used to call ``accelerator.save_state`` directly, bypassing the
    # rank-0 publish agreement and the versioned bundle. It is a resumable state,
    # so it goes through the same protocol as every other state save.
    publish_state_with_bundle(
        args,
        accelerator,
        state_dir,
        epoch_no=epoch_no,
        global_step=global_step,
        loader_cursor=loader_cursor,
        created_state_dirs=created_state_dirs,
        rank_digest=rank_digest,
    )

    last_n_epochs = (
        args.save_last_n_epochs_state
        if args.save_last_n_epochs_state is not None
        else args.save_last_n_epochs
    )
    keep_count = _save_last_keep_count(last_n_epochs)
    # Every rank calls this helper now, but only rank 0 may remove the shared
    # directory (a peer deleting it mid-save is the race this protocol removes).
    if keep_count is not None and accelerator.is_main_process:
        remove_epoch_no = epoch_no - args.save_every_n_epochs * keep_count
        if remove_epoch_no <= 0:
            return
        state_dir_old = os.path.join(
            args.output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no)
        )
        if os.path.exists(state_dir_old):
            if not _path_created_in_current_run(created_state_dirs, state_dir_old):
                logger.info(
                    f"skip removing pre-existing state not created by this run: {state_dir_old}"
                )
                return
            logger.info(f"removing old state: {state_dir_old}")
            shutil.rmtree(state_dir_old)


def save_and_remove_state_stepwise(
    args: argparse.Namespace,
    accelerator,
    step_no,
    created_state_dirs: set[str] | None = None,
    loader_cursor: Optional[dict] = None,
    rank_digest: Optional[dict] = None,
):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)

    logger.info("")
    logger.info(f"saving state at step {step_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(
        args.output_dir, STEP_STATE_NAME.format(model_name, step_no)
    )
    # A step-cadence state is a resume point too: same all-rank publish protocol,
    # same versioned bundle and same per-rank cursor artifacts.
    publish_state_with_bundle(
        args,
        accelerator,
        state_dir,
        epoch_no=None,
        global_step=step_no,
        loader_cursor=loader_cursor,
        created_state_dirs=created_state_dirs,
        rank_digest=rank_digest,
    )

    last_n_steps = (
        args.save_last_n_steps_state
        if args.save_last_n_steps_state
        else args.save_last_n_steps
    )
    if last_n_steps is not None and accelerator.is_main_process:
        # Rank 0 only: peers are in this call too, and a shared directory must
        # have a single owner.
        remove_step_no = step_no - last_n_steps - 1
        remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)

        if remove_step_no > 0:
            state_dir_old = os.path.join(
                args.output_dir, STEP_STATE_NAME.format(model_name, remove_step_no)
            )
            if os.path.exists(state_dir_old):
                if not _path_created_in_current_run(created_state_dirs, state_dir_old):
                    logger.info(
                        f"skip removing pre-existing state not created by this run: {state_dir_old}"
                    )
                    return
                logger.info(f"removing old state: {state_dir_old}")
                shutil.rmtree(state_dir_old)


def get_checkpoint_state_dir(args: argparse.Namespace, epoch_no: Optional[int] = None):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    if epoch_no is not None:
        return os.path.join(
            args.output_dir, CHECKPOINT_EPOCH_STATE_NAME.format(model_name, epoch_no)
        )
    return os.path.join(args.output_dir, CHECKPOINT_STATE_NAME.format(model_name))


def get_checkpoint_ckpt_name(
    args: argparse.Namespace, ext: str, epoch_no: Optional[int] = None
):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    if epoch_no is not None:
        return CHECKPOINT_EPOCH_FILE_NAME.format(model_name, epoch_no) + ext
    return CHECKPOINT_FILE_NAME.format(model_name) + ext


def get_checkpoint_latest_state_file(args: argparse.Namespace):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return os.path.join(
        args.output_dir, CHECKPOINT_LATEST_STATE_FILE_NAME.format(model_name)
    )


def _checkpoint_epoch_no_from_state_name(
    args: argparse.Namespace, name: str
) -> Optional[int]:
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    prefix = f"{model_name}-checkpoint-"
    suffix = "-state"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None
    raw = name[len(prefix) : -len(suffix)]
    if len(raw) != 6 or not raw.isdigit():
        return None
    return int(raw)


def _checkpointing_last_n_epochs(args: argparse.Namespace) -> int:
    value = getattr(args, "checkpointing_last_n_epochs", 1)
    try:
        keep_last = int(value)
    except (TypeError, ValueError):
        return 1
    if keep_last == -1:
        return -1
    return keep_last if keep_last > 0 else 1


def _checkpoint_epoch_state_entries(args: argparse.Namespace) -> list[tuple[int, str]]:
    output_dir = getattr(args, "output_dir", "")
    if not output_dir or not os.path.isdir(output_dir):
        return []
    entries: list[tuple[int, str]] = []
    for name in os.listdir(output_dir):
        epoch_no = _checkpoint_epoch_no_from_state_name(args, name)
        if epoch_no is None:
            continue
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            entries.append((epoch_no, path))
    entries.sort(key=lambda item: (item[0], item[1]))
    return entries


def _read_checkpoint_train_state(state_dir: str) -> dict[str, Any]:
    train_state_file = os.path.join(state_dir, "train_state.json")
    if not os.path.exists(train_state_file):
        return {}
    try:
        with open(train_state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001 - bad state means this candidate is skipped
        logger.info(f"skip checkpoint state because train_state.json is unreadable: {e}")
        return {}
    return data if isinstance(data, dict) else {}


def _write_checkpoint_latest_state_marker(
    args: argparse.Namespace, state_dir: str, epoch_no: int
) -> None:
    marker_file = get_checkpoint_latest_state_file(args)
    tmp_file = marker_file + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "state_dir": os.path.abspath(state_dir),
                "epoch": epoch_no,
            },
            f,
        )
    os.replace(tmp_file, marker_file)


def _checkpoint_latest_marker_state_dir(args: argparse.Namespace) -> Optional[str]:
    marker_file = get_checkpoint_latest_state_file(args)
    if not os.path.exists(marker_file):
        return None
    try:
        with open(marker_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001 - stale marker should not block fresh runs
        logger.info(f"skip checkpoint marker because it is unreadable: {e}")
        return None
    state_dir = str(data.get("state_dir") or "").strip()
    if not state_dir:
        return None
    state_dir = os.path.abspath(state_dir)
    output_dir = os.path.abspath(args.output_dir)
    try:
        if os.path.commonpath([output_dir, state_dir]) != output_dir:
            return None
    except ValueError:
        return None
    return state_dir if os.path.isdir(state_dir) else None


def _checkpoint_state_candidates(args: argparse.Namespace) -> list[tuple[int, float, str]]:
    candidates: list[tuple[int, float, str]] = []
    marker_state_dir = _checkpoint_latest_marker_state_dir(args)
    legacy_state_dir = get_checkpoint_state_dir(args)
    state_dirs = [marker_state_dir] if marker_state_dir else []
    if os.path.isdir(legacy_state_dir):
        state_dirs.append(legacy_state_dir)

    seen: set[str] = set()
    for state_dir in state_dirs:
        state_dir = os.path.abspath(state_dir)
        if state_dir in seen:
            continue
        seen.add(state_dir)
        data = _read_checkpoint_train_state(state_dir)
        try:
            step = int(data.get("current_step"))
        except (TypeError, ValueError):
            continue
        try:
            mtime = os.path.getmtime(os.path.join(state_dir, "train_state.json"))
        except OSError:
            mtime = 0.0
        candidates.append((step, mtime, state_dir))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates


def resume_skip_plan(
    resume_step: int, num_batches: int, gradient_accumulation_steps: int
) -> tuple[int, int]:
    """Split completed optimizer steps into whole epochs and residual batches."""
    steps_per_epoch = max(1, math.ceil(num_batches / gradient_accumulation_steps))
    epoch_to_start, residual_steps = divmod(resume_step, steps_per_epoch)
    return epoch_to_start, residual_steps * gradient_accumulation_steps


def plan_resume_start(
    args: argparse.Namespace,
    *,
    steps_from_state: Optional[int],
    batches_per_epoch: int,
    num_processes: int,
    updates_per_epoch: Optional[int] = None,
) -> ResumeStartPlan:
    """Resolve resume/initial-step counters without mutating checkpoint state."""
    steps_from_state_out = steps_from_state
    staged_updates_per_epoch = max(0, int(updates_per_epoch or 0))
    # The caller passes the accelerator-prepared dataloader length, already
    # sharded per process. Keep num_processes for existing callers.
    initial_steps_per_epoch = staged_updates_per_epoch or max(
        1, math.ceil(batches_per_epoch / args.gradient_accumulation_steps)
    )

    initial_step = 0
    if args.initial_epoch is not None or args.initial_step is not None:
        if steps_from_state is not None:
            logger.warning(
                "steps from the state is ignored because initial_step is specified"
            )
        if args.initial_step is not None:
            initial_step = args.initial_step
        else:
            initial_step = (args.initial_epoch - 1) * initial_steps_per_epoch
    else:
        if steps_from_state is not None:
            initial_step = steps_from_state
            steps_from_state_out = None

    if (
        args.resume
        and steps_from_state is not None
        and args.initial_epoch is None
        and args.initial_step is None
        and not args.skip_until_initial_step
    ):
        logger.warning(
            "--resume loaded train_state.json but --skip_until_initial_step was not set; "
            "enabling it so global_step, dataloader skip, and scheduler progress stay aligned"
        )
        args.skip_until_initial_step = True

    if initial_step > 0 and args.max_train_steps <= initial_step:
        raise ValueError(
            f"恢复点已训练到 step {initial_step}，当前配置目标是 {args.max_train_steps}，"
            "继续训练不会产生新步数。请增加 max_train_steps / max_train_epochs 后再续训。"
        )

    epoch_to_start = 0
    resume_global_step = 0
    if initial_step > 0:
        resume_global_step = initial_step
        if staged_updates_per_epoch:
            epoch_to_start = initial_step // staged_updates_per_epoch
            # Stage loaders are rebuilt from current global progress. Replaying
            # a stage0-sized batch skip here can exhaust the loader forever.
            initial_step = 0
        else:
            epoch_to_start, skip_batches = resume_skip_plan(
                initial_step, batches_per_epoch, args.gradient_accumulation_steps
            )
            if args.skip_until_initial_step and not args.resume:
                logger.info(
                    "initial_step is specified but not resuming. lr scheduler will be started from the beginning"
                )
            if args.skip_until_initial_step:
                logger.info(
                    f"skipping {initial_step} steps: {epoch_to_start} full epochs"
                    f" + {skip_batches} batches of epoch {epoch_to_start + 1}"
                )
            initial_step = skip_batches if args.skip_until_initial_step else 0

    return ResumeStartPlan(
        initial_step=initial_step,
        epoch_to_start=epoch_to_start,
        steps_from_state=steps_from_state_out,
        global_step=resume_global_step,
    )


def _remove_path(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def _abs_path_key(path: str) -> str:
    return os.path.abspath(str(path))


def _record_created_path(paths: set[str] | None, path: str) -> None:
    if paths is not None:
        paths.add(_abs_path_key(path))


def _path_created_in_current_run(paths: set[str] | None, path: str) -> bool:
    # None preserves the legacy behavior for old helper callers.  CheckpointSaver
    # passes a set so cleanup only touches files/directories written by this run.
    return paths is None or _abs_path_key(path) in paths


def _recover_checkpoint_state_dirs(state_dir: str, tmp_dir: str, backup_dir: str) -> None:
    """Compat shim: the recovery logic now lives in the publish protocol.

    Kept as a thin delegate (not a second implementation) so old callers keep
    working while there is exactly one definition of "clear stale dirs"
    (``checkpoint_protocol.recover_state_dirs``).
    """
    from library.training.multigpu import checkpoint_protocol as ckpt_proto

    ckpt_proto.recover_state_dirs(state_dir, tmp_dir, backup_dir)


def _checkpoint_network_state_compatible(state_dir: str, network: Any) -> bool:
    model_state_file = os.path.join(state_dir, "model.safetensors")
    if not os.path.exists(model_state_file):
        logger.info(
            f"skip auto-resume because checkpoint model state is missing: {model_state_file}"
        )
        return False
    try:
        from safetensors.torch import load_file

        checkpoint_keys = set(load_file(model_state_file, device="cpu").keys())
    except Exception as e:  # noqa: BLE001 - corrupted state should not block a fresh run
        logger.info(f"skip auto-resume because checkpoint model state is unreadable: {e}")
        return False

    network_keys = set(network.state_dict().keys())
    normalized_network_keys = {
        key.removeprefix("module.") for key in network_keys
    } | network_keys
    normalized_checkpoint_keys = {
        key.removeprefix("module.") for key in checkpoint_keys
    } | checkpoint_keys
    if not normalized_checkpoint_keys:
        logger.info("skip auto-resume because checkpoint model state is empty")
        return False
    if normalized_checkpoint_keys.issubset(normalized_network_keys):
        return True

    missing = sorted(normalized_checkpoint_keys - normalized_network_keys)[:5]
    logger.info(
        "skip auto-resume because checkpoint model state is incompatible; "
        f"unexpected keys: {missing}"
    )
    return False


def save_checkpoint_state(
    args: argparse.Namespace,
    accelerator,
    epoch_no: Optional[int] = None,
    global_step: Optional[int] = None,
    loader_cursor: Optional[dict] = None,
    rank_digest: Optional[dict] = None,
):
    """Save one state directory through the all-rank bundle protocol.

    The ``pipeline_parallel`` refusal for this path lives **only** in
    :func:`publish_state_with_bundle` (the choke point in front of
    ``accelerator.save_state``), which this function calls unconditionally. A
    second guard here used to duplicate it; it could never fire first in a way
    that mattered, and a mutation test showed the suite stays green without it.
    Do not add the check back here -- extend the one in
    ``publish_state_with_bundle`` instead, so there is a single refusal point.

    Consequence of delegating: for a refused run this function has already
    logged and run ``os.makedirs(args.output_dir, exist_ok=True)`` (the run root,
    which the run owns anyway). The state directory itself is only created after
    the delegated guard, so a refused save still leaves no resumable artifact.
    """
    state_dir = get_checkpoint_state_dir(args, epoch_no)

    logger.info("")
    logger.info(f"saving checkpoint state to {state_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    return publish_state_with_bundle(
        args,
        accelerator,
        state_dir,
        epoch_no=epoch_no,
        global_step=global_step,
        loader_cursor=loader_cursor,
        rank_digest=rank_digest,
    )


#: Per-rank loader-cursor artifact written inside the state directory. A per-rank
#: file name makes it a ``rank_local`` key: exactly one worker owns each name.
RANK_CURSOR_FILE = "multigpu_cursor_rank{rank}.json"


def _write_rank_cursor(state_dir: str, payload: dict) -> str:
    name = RANK_CURSOR_FILE.format(rank=int(payload["rank"]))
    path = os.path.join(state_dir, name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)
    return name


def _read_rank_cursors(state_dir: str, world_size: int) -> dict:
    out = {}
    for rank in range(max(1, int(world_size))):
        path = os.path.join(state_dir, RANK_CURSOR_FILE.format(rank=rank))
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: cursor payload is not an object")
        out[str(rank)] = data
    return out


def publish_state_with_bundle(
    args: argparse.Namespace,
    accelerator,
    state_dir: str,
    *,
    epoch_no: Optional[int],
    global_step: Optional[int],
    loader_cursor: Optional[dict],
    created_state_dirs: set[str] | None = None,
    rank_digest: Optional[dict] = None,
):
    """All-rank save + rank-0 atomic publish + versioned resume bundle.

    ``accelerator.save_state`` is what makes the save all-rank; the publish itself
    is rank-0 only, and the outcome is agreed before any rank continues (contract
    §8 P1-1). Every rank writes **its own** cursor artifact into the state
    directory before rank 0 renames it, so the published directory describes every
    worker's real position and the bundle's RNG flags come from the files that
    actually landed on disk.
    """
    from library.training.multigpu import checkpoint_protocol as ckpt_proto
    from library.training.multigpu import cursor as cursor_mod
    from library.training.multigpu import rank_digest as rank_digest_mod
    from library.training.multigpu import resume_protocol

    # Choke point for ``accelerator.save_state``: it runs on every rank and the
    # publish renames the result into one directory. Under pipeline_parallel each
    # rank owns a different stage, so that directory is not resumable -- fail
    # closed before any rank writes a shard.
    from library.training.pp.gate import require_accelerate_checkpoint

    require_accelerate_checkpoint(args, where="checkpoints.publish_state_with_bundle")

    tmp_dir = state_dir + ".tmp"
    backup_dir = state_dir + ".backup"

    # Every rank records the checkpoint position **before** any publish, in the
    # same order, and the agreed value is written into the resume bundle. A rank
    # that disagrees about step/epoch cannot produce a resumable state, so this
    # aborts on all ranks instead of publishing (contract §8 P1-1).
    # A step-cadence state has no epoch in its directory name, but the position it
    # records is (step, epoch, cursor): defaulting that epoch to 0 made every step
    # save refuse its own cursor ("per-rank cursors disagree about the epoch
    # (agreed 0, got {'0': 1, ...})"), which is the RUN-C save path.
    cursor_epoch = loader_cursor.get("epoch") if loader_cursor else None
    if epoch_no is not None:
        agreed_epoch_hint = int(epoch_no)
    elif isinstance(cursor_epoch, int) and not isinstance(cursor_epoch, bool):
        agreed_epoch_hint = int(cursor_epoch)
    else:
        agreed_epoch_hint = 0
    records = ckpt_proto.checkpoint_records(
        accelerator,
        global_step=-1 if global_step is None else int(global_step),
        epoch=agreed_epoch_hint,
    )
    recorded_step = int(records["global_step"])
    agreed_epoch = int(records["epoch"])
    world_size = int(getattr(accelerator, "num_processes", 1) or 1)
    rank = int(getattr(accelerator, "process_index", 0) or 0)

    local_cursor = None
    if loader_cursor:
        local_cursor = cursor_mod.make_cursor(
            position=int(loader_cursor["sampler_position"]),
            epoch=int(loader_cursor.get("epoch", agreed_epoch)),
            rank=rank,
            batches_per_epoch=loader_cursor.get("batches_per_epoch"),
        )

    def _before_publish(target_dir: str) -> None:
        device = getattr(accelerator, "device", None)
        payload = dict(local_cursor) if local_cursor else {
            "schema": cursor_mod.CURSOR_SCHEMA,
            "sampler_position": "not_reported",
            "epoch": agreed_epoch,
            "rank": rank,
        }
        # The device mapping is recorded per rank for the same reason the cursor
        # is: only that worker knows which device it really ran on, and a bundle
        # must be able to place every rank before a resume is called exact.
        payload["device_type"] = str(getattr(device, "type", "unknown"))
        payload["device_index"] = getattr(device, "index", None)
        _write_rank_cursor(target_dir, payload)
        if rank_digest is not None:
            # Same mechanism, same moment, one file per rank: a per-rank digest of
            # what this worker really holds at the commit boundary (group R).
            digest_payload = dict(rank_digest)
            digest_payload["rank"] = rank
            rank_digest_mod.write(digest_payload, target_dir)

    def _after_publish(published_dir: str) -> None:
        if epoch_no is not None and getattr(accelerator, "is_main_process", True):
            _write_checkpoint_latest_state_marker(args, published_dir, epoch_no)
        dist_type = str(
            getattr(getattr(accelerator, "state", None), "distributed_type", "")
        ).lower()
        backend = "nccl" if "cuda" in dist_type or "gpu" in dist_type else ("gloo" if dist_type else "none")
        rank_cursors = _read_rank_cursors(published_dir, world_size)
        mismatched = {
            key: value.get("epoch")
            for key, value in rank_cursors.items()
            if value.get("epoch") != agreed_epoch
        }
        if mismatched:
            # Rollback runs inside publish_state_dir; this refuses on every rank.
            raise ckpt_proto.CheckpointPublishError(
                f"{ckpt_proto.CODE_RECORD_FAILED}: per-rank cursors disagree about the "
                f"epoch (agreed {agreed_epoch}, got {mismatched})"
            )
        bundle = resume_protocol.build_bundle(
            accelerator,
            backend=backend,
            # Rank 0's cursor keeps the bundle's top-level field meaningful; the
            # per-rank truth is ``extra.rank_cursors``.
            loader_cursor=rank_cursors.get("0") or rank_cursors.get(str(rank)) or {},
            # RNG flags are derived by *loading* the per-rank random_states_*.pkl
            # that accelerate just wrote - never asserted (contract §8 P1-2).
            evidence_dir=published_dir,
            rank_local={key: [RANK_CURSOR_FILE.format(rank=int(key))] for key in rank_cursors},
            global_step=recorded_step if recorded_step >= 0 else None,
            extra={
                "rank_identity": {
                    "local_rank": rank,
                    "world_size": world_size,
                    "device": str(getattr(getattr(accelerator, "device", None), "type", "unknown")),
                },
                "rank_cursors": rank_cursors,
                "epoch": agreed_epoch,
            },
        )
        # ``build_bundle`` only knows this rank's device; the published state must
        # describe every worker, and each rank just reported its own in its cursor
        # artifact, so the mapping is taken from those files.
        bundle["rank_devices"] = {
            key: {
                "type": str(value.get("device_type", "unknown")),
                "index": value.get("device_index"),
            }
            for key, value in rank_cursors.items()
        }
        if rank_digest is not None:
            # Every rank just wrote its own digest into the published directory, so
            # the comparison is a local read of files that already landed - no
            # collective, no extra IO on the data path. A disagreement is reported
            # as a named WARNING: an existing run keeps its verdict, and the
            # verifier decides what a rank difference means (R-a/R-b/R-c).
            digest_summary = rank_digest_mod.summarize_state_dir(
                published_dir, world_size
            )
            rank_digest_mod.log_outcome(digest_summary, published_dir)
            bundle.setdefault("extra", {})["rank_digests"] = digest_summary
            bundle.setdefault("rank_local_keys", {})
            for key in digest_summary["files"]:
                bundle["rank_local_keys"].setdefault(str(key), []).append(
                    rank_digest_mod.FILE.format(rank=int(key))
                )
        resume_protocol.write_bundle(bundle, published_dir, is_main=True)

    ckpt_proto.save_state_published(
        accelerator,
        tmp_dir,
        state_dir,
        backup_dir,
        before_publish=_before_publish,
        after_publish=_after_publish,
    )
    if created_state_dirs is not None:
        _record_created_path(created_state_dirs, state_dir)
    return state_dir


def save_state_on_train_end(
    args: argparse.Namespace,
    accelerator,
    global_step: Optional[int] = None,
    loader_cursor: Optional[dict] = None,
):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)

    logger.info("")
    logger.info("saving last state.")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name))
    # Every rank must take part: a rank-0-only dump leaves a state directory that
    # no worker can resume, and its bundle would describe only one process.
    publish_state_with_bundle(
        args,
        accelerator,
        state_dir,
        epoch_no=None,
        global_step=global_step,
        loader_cursor=loader_cursor,
    )


def save_sd_model_on_train_end_common(
    args: argparse.Namespace,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    epoch: int,
    global_step: int,
    sd_saver,
    diffusers_saver,
):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)

    if save_stable_diffusion_format:
        os.makedirs(args.output_dir, exist_ok=True)

        ckpt_name = model_name + (".safetensors" if use_safetensors else ".ckpt")
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        logger.info(f"save trained model as StableDiffusion checkpoint to {ckpt_file}")
        sd_saver(ckpt_file, epoch, global_step)
    else:
        out_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        logger.info(f"save trained model as Diffusers to {out_dir}")
        diffusers_saver(out_dir)


def _rank_digest_payload(
    digest_context, network: Any, global_step: Optional[int], epoch: Optional[int]
) -> Optional[dict]:
    """This rank's commit-boundary digest, or ``None`` when it is switched off.

    Built **after** the cadence gate in every ``maybe_save_*``: hashing the adapter
    tensors is commit-boundary work, never per-step work.
    """
    if digest_context is None:
        return None
    from library.training.multigpu import rank_digest as rank_digest_mod

    return rank_digest_mod.build_from_loop_state(
        digest_context,
        global_step=global_step,
        epoch=epoch,
        network=network,
    )


class CheckpointSaver:
    """Owns every save / remove operation across a training run.

    Replaces the cluster of save_model / remove_model / save_model_hook /
    load_model_hook closures and the inline save-tick blocks scattered through
    train(). State that used to live in closures (metadata refs, save_dtype,
    sai-spec callable, mp.Value handles, ``steps_from_state``) becomes
    instance attributes.

    ``metadata`` is a shared mutable dict — the trainer also writes
    ``ss_epoch`` between saves; the saver only writes during a save.
    """

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        accelerator,
        save_dtype,
        metadata: dict,
        minimum_metadata: dict,
        get_sai_model_spec_fn: Callable[[argparse.Namespace], dict],
        current_epoch,
        current_step,
        progress_sink=None,
    ):
        self.args = args
        self.accelerator = accelerator
        self.save_dtype = save_dtype
        self.metadata = metadata
        self.minimum_metadata = minimum_metadata
        self.get_sai_model_spec_fn = get_sai_model_spec_fn
        self.current_epoch = current_epoch
        self.current_step = current_step
        # Optional structured-progress sink (Phase 0). When set, every
        # checkpoint write emits a ``ckpt`` event.
        self.progress_sink = progress_sink
        self._created_checkpoint_paths: set[str] = set()
        self._created_state_dirs: set[str] = set()
        self._created_marker_paths: set[str] = set()
        # Set by the load_state pre-hook when resuming. Read by train() to
        # decide initial_step.
        self.steps_from_state: Optional[int] = None

    def register_hooks(self, network: Any) -> None:
        """Install accelerator save/load pre-hooks that persist epoch/step
        state to ``train_state.json`` and strip non-network models from the
        save list (we only want the adapter weights, not the frozen DiT)."""
        accelerator = self.accelerator
        unwrap_type = type(accelerator.unwrap_model(network))

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, unwrap_type):
                        remove_indices.append(i)
                for i in reversed(remove_indices):
                    if len(weights) > i:
                        weights.pop(i)

            train_state_file = os.path.join(output_dir, "train_state.json")
            if not accelerator.is_main_process:
                # One writer per state dir: every rank used to truncate-and-write
                # the same sidecar, so a peer could read a half-written file (N-2).
                return
            logger.info(
                f"save train state to {train_state_file} at epoch "
                f"{self.current_epoch.value} step {self.current_step.value + 1}"
            )
            with open(train_state_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "current_epoch": self.current_epoch.value,
                        "current_step": self.current_step.value + 1,
                    },
                    f,
                )

        def load_model_hook(models, input_dir):
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, unwrap_type):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                models.pop(i)

            # Versioned resume validation (contract §8 P1-2): decide *before*
            # accepting the state whether this is an exact resume, a named
            # degraded one, legacy (not exact) or a hard rejection.
            from library.training.multigpu import resume_protocol

            try:
                decision = resume_protocol.guard_resume(
                    input_dir,
                    world_size=int(getattr(accelerator, "num_processes", 1) or 1),
                )
            except resume_protocol.ResumeBundleError:
                raise
            self.resume_decision = decision
            decision.raise_if_rejected()
            # Machine-checkable resume evidence next to the run identity
            # (``resume_exact`` / ``resume_codes`` / ``resume_bundle_version``), so a
            # verification ledger can read the verdict without replaying the run.
            # Best-effort: a manifest that cannot be updated must not abort a resume
            # that the protocol just accepted.
            try:
                from library.training.multigpu import identity as identity_mod

                identity_mod.record_resume_outcome(
                    self.args.output_dir,
                    decision=decision,
                    accelerator=accelerator,
                    bundle_version=(decision.bundle or {}).get("version"),
                )
            except Exception as exc:  # pragma: no cover - best-effort by design
                logger.warning(
                    "could not record the resume outcome in the identity manifest: %s",
                    exc,
                )

            train_state_file = os.path.join(input_dir, "train_state.json")
            if os.path.exists(train_state_file):
                with open(train_state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.steps_from_state = data["current_step"]
                logger.info(f"load train state from {train_state_file}: {data}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    def auto_resume(self, network: Any | None = None) -> None:
        """If ``checkpointing_epochs`` is enabled and a resumable checkpoint
        exists below ``max_train_steps``, point ``args.resume`` at it and
        force ``skip_until_initial_step``. No-op when ``args.resume`` is
        already set or no checkpoint exists.

        When ``network`` is provided, require the resumable checkpoint's
        ``model.safetensors`` keys to be compatible before resuming.
        """
        args = self.args
        if not getattr(args, "checkpointing_epochs", None) or args.resume:
            return
        candidates = _checkpoint_state_candidates(args)
        if not candidates:
            return
        ckpt_step, _, checkpoint_state_dir = candidates[0]
        if network is not None and not _checkpoint_network_state_compatible(
            checkpoint_state_dir, network
        ):
            return
        if ckpt_step < args.max_train_steps:
            args.resume = checkpoint_state_dir
            args.skip_until_initial_step = True
            logger.info(
                f"auto-resuming from checkpoint at step {ckpt_step}: {checkpoint_state_dir}"
            )
        else:
            logger.info(
                f"checkpoint already reached max_train_steps "
                f"({ckpt_step} >= {args.max_train_steps}), starting fresh"
            )

    def save(self, ckpt_name: str, network: Any, steps: int, epoch_no: int) -> None:
        """Write a network checkpoint with up-to-date training metadata."""
        args = self.args
        accelerator = self.accelerator
        unwrapped_nw = accelerator.unwrap_model(network)

        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        ckpt_preexisting = (
            os.path.exists(ckpt_file)
            and not _path_created_in_current_run(
                self._created_checkpoint_paths, ckpt_file
            )
        )
        moe_file = os.path.splitext(ckpt_file)[0] + "_moe.safetensors"
        moe_preexisting = (
            os.path.exists(moe_file)
            and not _path_created_in_current_run(self._created_checkpoint_paths, moe_file)
        )

        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        self.metadata["ss_training_finished_at"] = str(time.time())
        self.metadata["ss_steps"] = str(steps)
        self.metadata["ss_epoch"] = str(epoch_no)

        metadata_to_save = self.minimum_metadata if args.no_metadata else self.metadata
        sai_metadata = self.get_sai_model_spec_fn(args)
        metadata_to_save.update(sai_metadata)

        unwrapped_nw.save_weights(ckpt_file, self.save_dtype, metadata_to_save)
        if not ckpt_preexisting:
            _record_created_path(self._created_checkpoint_paths, ckpt_file)
        if os.path.exists(moe_file):
            if not moe_preexisting:
                _record_created_path(self._created_checkpoint_paths, moe_file)

        if self.progress_sink is not None:
            self.progress_sink.ckpt(global_step=steps, path=ckpt_file)

    def remove(self, old_ckpt_name: str) -> None:
        """Delete an old checkpoint plus its HydraLoRA ``_moe`` sibling if present."""
        args = self.args
        accelerator = self.accelerator
        old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
        if os.path.exists(old_ckpt_file):
            if not _path_created_in_current_run(
                self._created_checkpoint_paths, old_ckpt_file
            ):
                accelerator.print(
                    f"skip removing pre-existing checkpoint not created by this run: {old_ckpt_file}"
                )
            else:
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)
        moe_file = os.path.splitext(old_ckpt_file)[0] + "_moe.safetensors"
        if os.path.exists(moe_file):
            if not _path_created_in_current_run(self._created_checkpoint_paths, moe_file):
                accelerator.print(
                    f"skip removing pre-existing checkpoint not created by this run: {moe_file}"
                )
                return
            accelerator.print(f"removing old checkpoint: {moe_file}")
            os.remove(moe_file)

    def maybe_save_step(
        self,
        network: Any,
        global_step: int,
        epoch: int,
        loader_cursor: Optional[dict] = None,
        digest_context: Any = None,
    ) -> None:
        """Step-cadence save. ``global_step`` must already be incremented."""
        args = self.args
        accelerator = self.accelerator
        if (
            args.save_every_n_steps is None
            or global_step % args.save_every_n_steps != 0
        ):
            return
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            ckpt_name = get_step_ckpt_name(args, "." + args.save_model_as, global_step)
            self.save(ckpt_name, network, global_step, epoch)
        if args.save_state:
            # Every rank takes part: accelerate writes one shard per rank, so a
            # rank-0-only call produces a state directory nobody can resume from
            # — and its first collective (the publish protocol's recovery barrier)
            # would wait forever for peers that already returned (contract §8
            # P1-1; review t25).
            save_and_remove_state_stepwise(
                args,
                accelerator,
                global_step,
                self._created_state_dirs,
                loader_cursor=loader_cursor,
                rank_digest=_rank_digest_payload(
                    digest_context, network, global_step, epoch
                ),
            )
        if accelerator.is_main_process:
            remove_step_no = get_remove_step_no(args, global_step)
            if remove_step_no is not None:
                remove_ckpt_name = get_step_ckpt_name(
                    args, "." + args.save_model_as, remove_step_no
                )
                self.remove(remove_ckpt_name)

    def maybe_save_epoch(
        self,
        network: Any,
        global_step: int,
        epoch: int,
        num_train_epochs: int,
        loader_cursor: Optional[dict] = None,
        digest_context: Any = None,
    ) -> None:
        """Epoch-cadence save. ``epoch`` is 0-indexed; saver writes ``epoch+1``."""
        args = self.args
        accelerator = self.accelerator
        if args.save_every_n_epochs is None:
            return
        epoch_no = epoch + 1
        if epoch_no % args.save_every_n_epochs != 0:
            return
        if accelerator.is_main_process:
            ckpt_name = get_epoch_ckpt_name(args, "." + args.save_model_as, epoch_no)
            self.save(ckpt_name, network, global_step, epoch_no)
            remove_epoch_no = get_remove_epoch_no(args, epoch_no)
            if remove_epoch_no is not None:
                remove_ckpt_name = get_epoch_ckpt_name(
                    args, "." + args.save_model_as, remove_epoch_no
                )
                self.remove(remove_ckpt_name)
        if args.save_state:
            # Same reason as ``maybe_save_step``: the epoch state directory is
            # written by every rank (contract §8 P1-1).
            save_and_remove_state_on_epoch_end(
                args,
                accelerator,
                epoch_no,
                self._created_state_dirs,
                global_step,
                loader_cursor=loader_cursor,
                rank_digest=_rank_digest_payload(
                    digest_context, network, global_step, epoch_no
                ),
            )

    def maybe_save_resumable(
        self,
        network: Any,
        global_step: int,
        epoch: int,
        num_train_epochs: int,
        loader_cursor: Optional[dict] = None,
        digest_context: Any = None,
    ) -> None:
        """``checkpointing_epochs``-cadence resumable save. ``epoch`` is 0-indexed."""
        args = self.args
        accelerator = self.accelerator
        if not (
            args.checkpointing_epochs is not None and args.checkpointing_epochs > 0
        ):
            return
        epoch_no = epoch + 1
        if epoch_no % args.checkpointing_epochs != 0:
            return
        if accelerator.is_main_process:
            ckpt_name = get_checkpoint_ckpt_name(
                args, "." + args.save_model_as, epoch_no
            )
            self.save(ckpt_name, network, global_step, epoch_no)
        expected_state_dir = get_checkpoint_state_dir(args, epoch_no)
        state_preexisting = (
            os.path.exists(expected_state_dir)
            and not _path_created_in_current_run(
                self._created_state_dirs, expected_state_dir
            )
        )
        latest_marker = get_checkpoint_latest_state_file(args)
        marker_preexisting = (
            accelerator.is_main_process
            and os.path.exists(latest_marker)
            and not _path_created_in_current_run(
                self._created_marker_paths, latest_marker
            )
        )
        state_dir = save_checkpoint_state(
            args,
            accelerator,
            epoch_no,
            global_step=global_step,
            loader_cursor=loader_cursor,
            rank_digest=_rank_digest_payload(
                digest_context, network, global_step, epoch_no
            ),
        )
        if state_dir and not state_preexisting:
            _record_created_path(self._created_state_dirs, state_dir)
        if accelerator.is_main_process:
            if os.path.exists(latest_marker) and not marker_preexisting:
                _record_created_path(self._created_marker_paths, latest_marker)
        if accelerator.is_main_process:
            self._cleanup_old_resumable_checkpoints()

    def _cleanup_old_resumable_checkpoints(self) -> None:
        """Keep only the configured number of numbered resumable checkpoints."""
        args = self.args
        keep_last = _checkpointing_last_n_epochs(args)
        if keep_last == -1:
            return
        entries = _checkpoint_epoch_state_entries(args)
        if len(entries) <= keep_last:
            return
        for epoch_no, state_dir in entries[: len(entries) - keep_last]:
            if os.path.exists(state_dir):
                if not _path_created_in_current_run(self._created_state_dirs, state_dir):
                    logger.info(
                        f"skip removing pre-existing checkpoint state not created by this run: {state_dir}"
                    )
                else:
                    logger.info(f"removing old checkpoint state: {state_dir}")
                    shutil.rmtree(state_dir)
            ckpt_name = get_checkpoint_ckpt_name(
                args, "." + args.save_model_as, epoch_no
            )
            self.remove(ckpt_name)

    def cleanup_resumable(self) -> None:
        """At training end, remove the legacy single resumable checkpoint.

        Numbered checkpoint states are intentionally kept according to
        ``checkpointing_last_n_epochs`` so history can resume any retained point.
        """
        args = self.args
        if not getattr(args, "checkpointing_epochs", None):
            return
        if not self.accelerator.is_main_process:
            return
        checkpoint_state_dir = get_checkpoint_state_dir(args)
        explicit_resume = os.path.abspath(str(getattr(args, "resume", "") or ""))
        keep_explicit_resume_state = (
            explicit_resume
            and explicit_resume == os.path.abspath(checkpoint_state_dir)
        )
        if keep_explicit_resume_state:
            logger.info(
                f"training complete, keeping explicit resume checkpoint state: {checkpoint_state_dir}"
            )
        elif os.path.exists(checkpoint_state_dir):
            if _path_created_in_current_run(
                self._created_state_dirs, checkpoint_state_dir
            ):
                logger.info(
                    f"training complete, removing checkpoint state: {checkpoint_state_dir}"
                )
                shutil.rmtree(checkpoint_state_dir)
            else:
                logger.info(
                    f"training complete, keeping pre-existing checkpoint state: {checkpoint_state_dir}"
                )
        latest_marker = get_checkpoint_latest_state_file(args)
        if os.path.exists(latest_marker) and not keep_explicit_resume_state:
            if _path_created_in_current_run(self._created_marker_paths, latest_marker):
                logger.info(f"training complete, removing checkpoint marker: {latest_marker}")
                os.remove(latest_marker)
            else:
                logger.info(
                    f"training complete, keeping pre-existing checkpoint marker: {latest_marker}"
                )
        checkpoint_ckpt = os.path.join(
            args.output_dir,
            get_checkpoint_ckpt_name(args, "." + args.save_model_as),
        )
        if os.path.exists(checkpoint_ckpt) and not keep_explicit_resume_state:
            if _path_created_in_current_run(
                self._created_checkpoint_paths, checkpoint_ckpt
            ):
                logger.info(f"removing checkpoint weights: {checkpoint_ckpt}")
                os.remove(checkpoint_ckpt)
            else:
                logger.info(
                    f"training complete, keeping pre-existing checkpoint weights: {checkpoint_ckpt}"
                )

    def save_final(self, network: Any, global_step: int, num_train_epochs: int) -> None:
        """Write the final ``<output_name>.<ext>`` checkpoint. Main-process only."""
        if not self.accelerator.is_main_process:
            return
        args = self.args
        ckpt_name = get_last_ckpt_name(args, "." + args.save_model_as)
        self.save(ckpt_name, network, global_step, num_train_epochs)
        logger.info("model saved.")
