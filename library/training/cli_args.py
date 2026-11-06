"""Argparse argument adders for the Anima training CLI.

Each ``add_*_arguments`` call plugs a related group of flags into the training
parser. The groups split the flag surface into coherent chunks so individual
entry points (training, preprocessing, distillation, ...) can opt into only
what they need.

No real logic lives here — these are pure argparse declarations.
"""

from __future__ import annotations

import argparse
import logging
import os

from library.datasets import base as _datasets_base
from library.env import KNOWN_MODEL_FAMILIES
from library.log import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def add_sd_models_arguments(parser: argparse.ArgumentParser):
    from library.models.qwen_image_2_1.preview_memory import add_arguments

    add_arguments(parser)
    parser.add_argument(
        "--v2",
        action="store_true",
        help="load Stable Diffusion v2.0 model",
    )
    parser.add_argument(
        "--v_parameterization",
        action="store_true",
        help="enable v-parameterization training",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        default=None,
        choices=list(KNOWN_MODEL_FAMILIES),
        help="model family switch: anima (default), krea2_raw, z_image, qwen_image_2_1, or moss_image. "
        "None → fall back to resolve_model_family() (env/base.toml, stage 6 truth source).",
    )
    parser.add_argument(
        "--moss_te_quantization", type=str, choices=("nf4", "bf16"), default="nf4",
        help="MOSS-Image frozen text encoder: NF4 with BF16 compute (default), or BF16 reference; no fallback.",
    )
    parser.add_argument(
        "--qwen_image_2_1_task",
        type=str,
        choices=("t2i", "edit"),
        default="t2i",
        help="Qwen Image 2.1 task mode: text-to-image (default) or paired-reference edit LoRA.",
    )
    parser.add_argument(
        "--qwen_text_encoder_cache_policy",
        type=str,
        choices=("auto", "cpu_offload", "gpu", "cpu"),
        default="auto",
        help="Qwen3-VL cache execution policy only; does not control VAE or DiT placement.",
    )
    parser.add_argument(
        "--qwen_fused_projections",
        type=str,
        choices=("off", "mlp", "qkv", "all"),
        default="off",
        help="[EXPERIMENTAL] Qwen Image 2.1 BF16 frozen-base projection packing; default off.",
    )
    parser.add_argument(
        "--qwen_saved_projection_blocks", type=int, default=0,
        help="[EXPERIMENTAL] Retain packed projection outputs in the final N Qwen blocks; default off.",
    )
    parser.add_argument(
        "--qwen_projection_budget_mib", type=int, default=0,
        help="[EXPERIMENTAL] Maximum raw QKV plus gate/up payload in MiB; required when blocks are selected.",
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="directory for caching Tokenizer (for offline training)",
    )


def add_optimizer_arguments(parser: argparse.ArgumentParser):
    def int_or_float(value):
        if value.endswith("%"):
            try:
                return float(value[:-1]) / 100.0
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Value '{value}' is not a valid percentage"
                )
        try:
            float_value = float(value)
            if float_value >= 1:
                return int(value)
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{value}' is not an int or float")

    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="",
        help="Optimizer to use: "
        "CAME, Lion8bit, PagedLion8bit, Lion, SGDNesterov, SGDNesterov8bit, "
        "DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD, "
        "ProdigyPlusScheduleFree, Automagic, AdaFactor.",
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="use 8bit AdamW optimizer (requires bitsandbytes)",
    )
    parser.add_argument(
        "--use_lion_optimizer",
        action="store_true",
        help="use Lion optimizer (requires lion-pytorch)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2.0e-6, help="learning rate"
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm, 0 for no clipping",
    )
    parser.add_argument(
        "--max_nonfinite_window_discards",
        type=int,
        default=3,
        help=(
            "Consecutive accumulation windows a non-adaptive run may discard after one "
            "replay each (E3) before the run fails with nonfinite_skip. Adaptive trials "
            "use --adaptive_max_nonfinite_skips instead (default 0)."
        ),
    )

    parser.add_argument(
        "--optimizer_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...")',
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="",
        help="custom scheduler module",
    )
    parser.add_argument(
        "--lr_scheduler_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for scheduler (like "T_max=100")',
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="scheduler to use for learning rate",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int_or_float,
        default=0,
        help="Int number of steps for the warmup in the lr scheduler (default is 0) or float with ratio of train steps",
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int_or_float,
        default=0,
        help="Int number of steps for the decay in the lr scheduler (default is 0) or float (<1) with ratio of train steps",
    )
    parser.add_argument(
        "--lr_scheduler_num_cycles",
        type=int,
        default=1,
        help="Number of restarts for cosine scheduler with restarts",
    )
    parser.add_argument(
        "--lr_scheduler_power",
        type=float,
        default=1,
        help="Polynomial power for polynomial scheduler",
    )
    parser.add_argument(
        "--fused_backward_pass",
        action="store_true",
        help="Combines backward pass and optimizer step to reduce VRAM usage.",
    )
    parser.add_argument(
        "--lr_scheduler_timescale",
        type=int,
        default=None,
        help="Inverse sqrt timescale for inverse sqrt scheduler,defaults to `num_warmup_steps`",
    )
    parser.add_argument(
        "--lr_scheduler_min_lr_ratio",
        type=float,
        default=None,
        help="The minimum learning rate as a ratio of the initial learning rate for cosine with min lr scheduler and warmup decay scheduler",
    )


def add_training_arguments(parser: argparse.ArgumentParser, support_dreambooth: bool):
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="directory to output trained model",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="base name of trained model file",
    )
    parser.add_argument(
        "--progress_jsonl",
        type=str,
        default=None,
        help="path to the structured progress JSONL event stream. Unset → derive "
        "<output_dir>/<output_name>.progress.jsonl (default on). Pass an empty string "
        "/ 'none' / 'off' to disable.",
    )
    parser.add_argument(
        "--memory_probe_jsonl",
        type=str,
        default=None,
        help=(
            "Write opt-in CUDA memory / adapter / optimizer diagnostics to JSONL. "
            "Use off/none to disable, auto for <logs>/<output_name>.memory_probe.jsonl, "
            "or an explicit path."
        ),
    )
    parser.add_argument(
        "--memory_probe_max_steps",
        type=int,
        default=2,
        help=(
            "Maximum number of training steps for detailed memory probe snapshots. "
            "0 means record every step. Setup summaries are always recorded when "
            "--memory_probe_jsonl is enabled."
        ),
    )
    parser.add_argument(
        "--peak_probe_jsonl",
        type=str,
        default=None,
        help=(
            "Write fine-grained DiT/LoKr CUDA peak diagnostics to JSONL. "
            "Use off/none to disable, auto for <logs>/<output_name>.peak_probe.jsonl, "
            "or an explicit path. Intended for short tight-VRAM profiling runs."
        ),
    )
    parser.add_argument(
        "--peak_probe_max_steps",
        type=int,
        default=2,
        help=(
            "Maximum number of training steps for fine-grained peak probe events. "
            "0 means record every step."
        ),
    )
    parser.add_argument(
        "--peak_probe_level",
        type=str,
        default="block",
        choices=["block", "ops", "lokr", "full"],
        help=(
            "Granularity for --peak_probe_jsonl. block records per-DiT-block "
            "boundaries outside compiled block graphs; ops adds in-block "
            "attention/MLP markers; lokr adds LoKr delta markers; full records all. "
            "Use block for 50-step baselines, full only for short focused runs."
        ),
    )
    parser.add_argument(
        "--gradient_flow_probe_jsonl",
        type=str,
        default=None,
        help=(
            "Write opt-in block-level gradient and real optimizer-update telemetry "
            "to JSONL. Use off/none to disable, auto for a logs path, or an "
            "explicit path."
        ),
    )
    parser.add_argument(
        "--gradient_flow_probe_every_n_steps",
        type=int,
        default=10,
        help="Record gradient-flow telemetry every N optimizer steps after the dense window.",
    )
    parser.add_argument(
        "--gradient_flow_probe_dense_steps",
        type=int,
        default=20,
        help="Record every optimizer step through this 1-based step number.",
    )
    parser.add_argument(
        "--anima_freeze_blocks",
        type=str,
        default=None,
        help=(
            "Experimental resume-only Anima A/B control: comma-separated DiT block "
            "indices whose LoRA parameters stop updating after state restore."
        ),
    )
    parser.add_argument(
        "--anima_freeze_fuse",
        action="store_true",
        help=(
            "Fold --anima_freeze_blocks LoRA deltas into resident BF16 base weights "
            "and disable those adapter branches. Requires torch_compile=false and "
            "blocks_to_swap=0."
        ),
    )
    parser.add_argument(
        "--huggingface_repo_id",
        type=str,
        default=None,
        help="huggingface repo name to upload",
    )
    parser.add_argument(
        "--huggingface_repo_type",
        type=str,
        default=None,
        help="huggingface repo type to upload",
    )
    parser.add_argument(
        "--huggingface_path_in_repo",
        type=str,
        default=None,
        help="huggingface model path to upload files",
    )
    parser.add_argument(
        "--huggingface_token",
        type=str,
        default=None,
        help="huggingface token",
    )
    parser.add_argument(
        "--huggingface_repo_visibility",
        type=str,
        default=None,
        help="huggingface repository visibility ('public' for public, 'private' or None for private)",
    )
    parser.add_argument(
        "--save_state_to_huggingface",
        action="store_true",
        help="save state to huggingface",
    )
    parser.add_argument(
        "--resume_from_huggingface", action="store_true", help="resume from huggingface"
    )
    parser.add_argument(
        "--async_upload",
        action="store_true",
        help="upload to huggingface asynchronously",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving",
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=None,
        help="save checkpoint every N epochs",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=None,
        help="save checkpoint every N steps",
    )
    parser.add_argument(
        "--save_n_epoch_ratio",
        type=int,
        default=None,
        help="save checkpoint N epoch ratio",
    )
    parser.add_argument(
        "--save_last_n_epochs",
        type=int,
        default=None,
        help="save last N checkpoints when saving every N epochs (remove older checkpoints)",
    )
    parser.add_argument(
        "--save_last_n_epochs_state",
        type=int,
        default=None,
        help="save last N checkpoints of state (overrides the value of --save_last_n_epochs)",
    )
    parser.add_argument(
        "--save_last_n_steps",
        type=int,
        default=None,
        help="save checkpoints until N steps elapsed (remove older checkpoints if N steps elapsed)",
    )
    parser.add_argument(
        "--save_last_n_steps_state",
        type=int,
        default=None,
        help="save states until N steps elapsed (remove older states if N steps elapsed, overrides --save_last_n_steps)",
    )
    parser.add_argument(
        "--save_state",
        action="store_true",
        help="save training state additionally (including optimizer states etc.) when saving model",
    )
    parser.add_argument(
        "--save_state_on_train_end",
        action="store_true",
        help="save training state (including optimizer states etc.) on train end",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="saved state to resume training",
    )
    parser.add_argument(
        "--checkpointing_epochs",
        type=int,
        default=None,
        help="save resumable checkpoint every N epochs (auto-resumes on next run)",
    )
    parser.add_argument(
        "--checkpointing_last_n_epochs",
        type=int,
        default=1,
        help="keep last N resumable checkpoint states saved by --checkpointing_epochs; -1 keeps all",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="batch size for training",
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=None,
        choices=[None, 150, 225, 4096],
        help="max token length of text encoder (legacy 150/225; MOSS-Image supports 4096)",
    )
    parser.add_argument(
        "--mem_eff_attn",
        action="store_true",
        help="use memory efficient attention for CrossAttention",
    )
    parser.add_argument(
        "--profile_steps",
        type=str,
        default=None,
        help=(
            "toggle the CUDA profiler for the given step range, e.g. '3-5'. "
            "Pair with: nsys profile --capture-range=cudaProfilerApi "
            "--capture-range-end=stop ... so nsys only records that window. "
            "NVTX ranges (step / forward / backward / optimizer) label the timeline."
        ),
    )
    parser.add_argument(
        "--qwen_nf4_compile_experiment", action="store_true", default=False,
        help="experimental strict Qwen Edit mixed NF4 compile policy; default disabled",
    )
    parser.add_argument(
        "--allow_unverified_nf4_compile",
        action="store_true",
        help="explicitly accept unverified mixed NF4 compile numerical risk for this run",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="use torch.compile (requires PyTorch 2.0)",
    )
    parser.add_argument(
        "--dynamo_backend",
        type=str,
        default="inductor",
        choices=[
            "eager",
            "aot_eager",
            "inductor",
            "aot_ts_nvfuser",
            "nvprims_nvfuser",
            "cudagraphs",
            "ofi",
            "fx2trt",
            "onnxrt",
            "tensort",
            "ipex",
            "tvm",
        ],
        help="dynamo backend type (default is inductor)",
    )
    parser.add_argument(
        "--compile_inductor_mode",
        type=str,
        default=None,
        choices=[
            None,
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help="Inductor preset forwarded as torch.compile(..., mode=...). "
        "'reduce-overhead' enables CUDAGraphs — requires stable tensor addresses "
        "across steps and is incompatible with block swap.",
    )
    parser.add_argument(
        "--compile_block_scope",
        type=str,
        default="resident",
        choices=["resident", "all"],
        help=(
            "Which DiT blocks torch.compile should wrap when block swap is enabled. "
            "resident compiles only GPU-resident head blocks and leaves swapped "
            "tail blocks eager for speed stability; all also compiles swapped "
            "blocks, which can lower activation memory on tight VRAM runs but "
            "may recompile more often."
        ),
    )
    parser.add_argument(
        "--compile_dynamic_seq",
        action="store_true",
        help=(
            "Mark only the native-flattened sequence axis dynamic before the "
            "compiled DiT block. The mark is applied inside the checkpointed "
            "callable so gradient-checkpoint recompute sees the same symbolic "
            "bounds as the original forward."
        ),
    )
    parser.add_argument(
        "--compile_seq_bands",
        action="store_true",
        help=(
            "With --compile_dynamic_seq, dispatch each native token-count "
            "cluster through its own tight dynamic-sequence band instead of "
            "one union range. Bands are derived from the active dataset and "
            "startup sample resolutions; register-token tails widen their "
            "upper bounds and refuse silent band merging. Anima-only and "
            "off by default; Krea-2 keeps its fixed padded sequence graphs."
        ),
    )
    parser.add_argument(
        "--activation_memory_budget",
        type=float,
        default=1.0,
        help=(
            "torch.compile AOT partitioner activation-memory budget. Values "
            "<1.0 ask the partitioner to recompute cheap intermediates, but the "
            "setting is ignored when gradient_checkpointing is enabled because "
            "repartitioning can make checkpoint recompute pick a different graph."
        ),
    )
    parser.add_argument(
        "--partitioner_recompute_views",
        action="store_true",
        help=(
            "torch._functorch.config.recompute_views: let the AOT partitioner "
            "recompute view ops in backward instead of saving them. Ignored "
            "when gradient_checkpointing is enabled for the same repartitioning "
            "hazard as activation_memory_budget."
        ),
    )
    parser.add_argument(
        "--partitioner_aggressive_recomputation",
        action="store_true",
        help=(
            "torch._functorch.config.aggressive_recomputation: allow the AOT "
            "partitioner to recompute more op classes when the min-cut is "
            "cheap. Ignored when gradient_checkpointing is enabled."
        ),
    )
    parser.add_argument(
        "--xformers", action="store_true", help="use xformers for CrossAttention"
    )
    parser.add_argument(
        "--sdpa",
        action="store_true",
        help="use sdpa for CrossAttention (requires PyTorch 2.0)",
    )
    parser.add_argument(
        "--vae", type=str, default=None, help="path to checkpoint of vae to replace"
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1600,
        help="training steps",
    )
    parser.add_argument(
        "--max_train_epochs",
        type=int,
        default=None,
        help="training epochs (overrides max_train_steps)",
    )
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=1,
        help="max num workers for DataLoader",
    )
    parser.add_argument(
        "--persistent_data_loader_workers",
        action="store_true",
        help="persistent DataLoader workers",
    )
    parser.add_argument(
        "--dataloader_pin_memory", action="store_true", help="pin DataLoader memory"
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=1,
        help="prefetch_factor for DataLoader workers (only valid when num_workers>0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for training",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="enable gradient checkpointing",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="use mixed precision",
    )
    parser.add_argument(
        "--full_fp16", action="store_true", help="fp16 training including gradients"
    )
    from library.training.adaptive_runtime.training_config import add_arguments as add_precision_arguments
    add_precision_arguments(parser)
    parser.add_argument(
        "--full_bf16", action="store_true", help="bf16 training including gradients"
    )

    parser.add_argument(
        "--clip_skip",
        type=int,
        default=None,
        help="use output of nth layer from back of text encoder (n>=1)",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="enable logging and output TensorBoard log to this directory",
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", "all"],
        help="what logging tool(s) to use",
    )
    parser.add_argument(
        "--log_prefix", type=str, default=None, help="add prefix for each log directory"
    )
    parser.add_argument(
        "--log_tracker_name",
        type=str,
        default=None,
        help="name of tracker to use for logging",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="The name of the specific wandb session",
    )
    parser.add_argument(
        "--log_tracker_config",
        type=str,
        default=None,
        help="path to tracker config file to use for logging",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="specify WandB API key to log in before starting training (optional).",
    )
    parser.add_argument(
        "--log_config", action="store_true", help="log training configuration"
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=1,
        help="only emit step-level metrics every N global steps (default 1 = every step). "
        "Validation and epoch logs are unaffected. Useful for long W&B runs.",
    )

    parser.add_argument(
        "--noise_offset",
        type=float,
        default=None,
        help="enable noise offset with this value (if enabled, around 0.1 is recommended)",
    )
    parser.add_argument(
        "--noise_offset_random_strength",
        action="store_true",
        help="use random strength between 0~noise_offset for noise offset.",
    )
    parser.add_argument(
        "--multires_noise_iterations",
        type=int,
        default=None,
        help="enable multires noise with this number of iterations (if enabled, around 6-10 is recommended)",
    )
    parser.add_argument(
        "--ip_noise_gamma",
        type=float,
        default=None,
        help="enable input perturbation noise. recommended value: around 0.1",
    )
    parser.add_argument(
        "--ip_noise_gamma_random_strength",
        action="store_true",
        help="Use random strength between 0~ip_noise_gamma for input perturbation noise.",
    )
    parser.add_argument(
        "--multires_noise_discount",
        type=float,
        default=0.3,
        help="set discount value for multires noise",
    )
    parser.add_argument(
        "--adaptive_noise_scale",
        type=float,
        default=None,
        help="add `latent mean absolute value * this value` to noise_offset (disabled if None, default)",
    )
    parser.add_argument(
        "--zero_terminal_snr",
        action="store_true",
        help="fix noise scheduler betas to enforce zero terminal SNR",
    )
    parser.add_argument(
        "--min_timestep",
        type=int,
        default=None,
        help="set minimum time step for U-Net training (0~999, default is 0)",
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=None,
        help="set maximum time step for U-Net training (1~1000, default is 1000)",
    )
    parser.add_argument(
        "--t_min",
        type=float,
        default=None,
        help="Restrict training sigma range: minimum sigma (0.0~1.0).",
    )
    parser.add_argument(
        "--t_max",
        type=float,
        default=None,
        help="Restrict training sigma range: maximum sigma (0.0~1.0). Default 1.0.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l1", "l2", "huber", "smooth_l1", "pseudo_huber"],
        help="The type of loss function to use (L1, L2, Huber, smooth L1, or pseudo-Huber), default is L2",
    )
    parser.add_argument(
        "--huber_schedule",
        type=str,
        default="snr",
        choices=["constant", "exponential", "snr"],
        help="The scheduling method for Huber loss. default is snr",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.1,
        help="The Huber loss decay parameter. default is 0.1",
    )
    parser.add_argument(
        "--huber_scale",
        type=float,
        default=1.0,
        help="The Huber loss scale parameter. default is 1.0",
    )
    parser.add_argument(
        "--pseudo_huber_c",
        type=float,
        default=0.03,
        help="Pseudo-Huber loss parameter c. Small c -> L1-like, large c -> MSE-like. default is 0.03",
    )
    parser.add_argument(
        "--multiscale_loss_weight",
        type=float,
        default=0,
        help="Weight for 2x-downsampled multiscale loss term. 0 = disabled. default is 0",
    )
    parser.add_argument(
        "--lowram", action="store_true", help="enable low RAM optimization."
    )
    parser.add_argument(
        "--highvram", action="store_true", help="disable low VRAM optimization."
    )

    parser.add_argument(
        "--sample_every_n_steps",
        type=int,
        default=None,
        help="generate sample images every N steps",
    )
    parser.add_argument(
        "--sample_at_first",
        action="store_true",
        help="generate sample images before training",
    )
    parser.add_argument(
        "--sample_every_n_epochs",
        type=int,
        default=None,
        help="generate sample images every N epochs",
    )
    parser.add_argument(
        "--sample_prompts",
        type=str,
        default=None,
        help="file for prompts to generate sample images",
    )
    parser.add_argument(
        "--sample_sampler",
        type=str,
        default="euler",
        choices=[
            "euler",
            "er_sde",
            "lcm",
            "ddim",
            "pndm",
            "lms",
            "euler_a",
            "heun",
            "dpm_2",
            "dpm_2_a",
            "dpmsolver",
            "dpmsolver++",
            "dpmsingle",
            "k_lms",
            "k_euler",
            "k_euler_a",
            "k_dpm_2",
            "k_dpm_2_a",
        ],
        help=(
            "sampler for Anima training preview images. Legacy Diffusers "
            "scheduler names are accepted and fall back to euler."
        ),
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="using .toml instead of args to pass hyperparameter",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="method name under configs/methods/ (e.g. 'tlora', 'hydralora', 'chimera'). Merged after preset so method settings win on overlap.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        help="hardware preset section name in configs/presets.toml (e.g. 'default', 'fast_16gb', 'low_vram').",
    )
    parser.add_argument(
        "--methods_subdir",
        type=str,
        default="methods",
        help="subfolder under configs/ that holds the method file (default 'methods'). Use 'gui-methods' for the GUI-friendly per-variant configs (lora, ortholora, tlora, reft, hydralora, …).",
    )
    parser.add_argument(
        "--output_config",
        action="store_true",
        help="output command line args to given .toml file",
    )
    if support_dreambooth:
        parser.add_argument(
            "--prior_loss_weight",
            type=float,
            default=1.0,
            help="loss weight for regularization images",
        )


def add_masked_loss_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--conditioning_data_dir",
        type=str,
        default=None,
        help="conditioning data directory",
    )
    parser.add_argument(
        "--masked_loss", action="store_true", help="apply mask for calculating loss."
    )
    parser.add_argument(
        "--mask_loss_normalize",
        choices=("none", "foreground_mean"),
        default="none",
        help="Spatial mask reduction: legacy full-latent mean or foreground-area mean.",
    )
    parser.add_argument(
        "--foreground_loss_weight",
        type=float,
        default=1.0,
        help="Multiplier applied after foreground_mean reduction.",
    )


def add_dit_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--use_text_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache text-encoder outputs to disk and read them during training "
        "(load TE → cache → free → load DiT). Set false for live encoding "
        "(e.g. IP-Adapter live mode). Subsumes the legacy "
        "cache_text_encoder_outputs{,_to_disk} pair.",
    )
    parser.add_argument(
        "--text_encoder_batch_size",
        type=int,
        default=None,
        help="text encoder batch size (default: None, use dataset's batch size)",
    )
    parser.add_argument(
        "--disable_mmap_load_safetensors",
        action="store_true",
        help="disable mmap load for safetensors. Speed up model loading in WSL environment",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="uniform",
        choices=[
            "sigma_sqrt",
            "logit_normal",
            "mode",
            "cosmap",
            "min_snr",
            "p2",
            "none",
            "uniform",
        ],
        help="weighting scheme for timestep distribution. Default is uniform",
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean for logit_normal weighting scheme",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std for logit_normal weighting scheme",
    )
    parser.add_argument(
        "--mode_scale", type=float, default=1.29, help="Scale of mode weighting scheme"
    )
    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=None,
        help="[EXPERIMENTAL] Sets the number of blocks to swap during the forward and backward passes.",
    )
    parser.add_argument(
        "--block_swap_auto",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "[EXPERIMENTAL] In-process water-level tuner: after the first optimizer step, "
            "promote swapped blocks to resident while keeping block_swap_vram_margin_mib "
            "free. Only lowers blocks_to_swap; differs from the separate auto_block_swap "
            "startup calibration."
        ),
    )
    parser.add_argument(
        "--block_swap_vram_margin_mib",
        type=int,
        default=400,
        help="[EXPERIMENTAL] Free-VRAM margin kept by --block_swap_auto (default 400 MiB).",
    )
    parser.add_argument(
        "--block_swap_auto_stop_ratio",
        type=float,
        default=0.0,
        help=(
            "[EXPERIMENTAL] 0 disables (default): promote as far as VRAM allows, which is "
            "speed-optimal. A value like 0.5 stops promoting once the projected DMA time "
            "would fall below that fraction of the measured step, keeping more blocks "
            "swapped at a small speed cost. Env: ANIMA_BLOCK_SWAP_AUTO_DMA_RATIO_FLOOR."
        ),
    )
    from library.training.auto_block_swap.config import add_arguments as add_auto_swap_arguments

    add_auto_swap_arguments(parser)
    parser.add_argument(
        "--pipeline_parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "[EXPERIMENTAL] Enable model-family pipeline parallelism. The "
            "current release validates the topology and refuses to fall back "
            "to DDP until the 1F1B trainer schedule is enabled."
        ),
    )
    parser.add_argument(
        "--pipeline_parallel_stages",
        type=int,
        default=2,
        choices=[2],
        help="Number of pipeline stages; must equal the distributed world size when enabled.",
    )
    parser.add_argument(
        "--pipeline_parallel_microbatches",
        type=int,
        default=4,
        help=(
            "Number of microbatches per pipeline step. Use 4-8 to reduce the "
            "two-stage pipeline bubble; each microbatch remains batch size one."
        ),
    )
    parser.add_argument(
        "--pipeline_parallel_schedule",
        type=str,
        default="1f1b",
        choices=["1f1b"],
        help="Pipeline schedule. Only the 1F1B contract is accepted in the experimental path.",
    )
    parser.add_argument(
        "--pipeline_parallel_split",
        type=str,
        default="balanced",
        choices=["balanced"],
        help="How model-family transformer blocks are split across pipeline stages.",
    )
    parser.add_argument(
        "--block_swap_profile_jsonl",
        type=str,
        default=None,
        help=(
            "Write block-swap transfer/wait timings to JSONL. Use off/none to "
            "disable, auto for the caller-resolved default path, or an explicit path."
        ),
    )
    parser.add_argument(
        "--block_swap_transfer_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp8_e4m3", "int8"],
        help=(
            "Transfer/storage dtype for frozen DiT block-swap CPU masters. "
            "bf16 keeps the current path; fp8_e4m3 is experimental and only "
            "quantizes frozen base weights while restoring them to the execution dtype. "
            "int8 is a narrower experiment for selected frozen MLP/attention Linear weights. "
            "Mutually exclusive with --base_compute w8a16_convrot/w8a8_convrot."
        ),
    )
    parser.add_argument(
        "--base_compute",
        type=str,
        default="bf16",
        choices=["bf16", "w8a16_convrot", "w8a8_convrot", "nf4"],
        help=(
            "[EXPERIMENTAL] Frozen DiT base Linear compute path. "
            "bf16 is the default high-precision path. "
            "w8a16_convrot applies group Regular Hadamard (ConvRot) + int8 "
            "weights with bf16/fp16 activations on selected scope (default mlp). "
            "w8a8_convrot additionally fake-quants activations (default off). "
            "Not the same as --block_swap_transfer_dtype int8. "
            "nf4 (Krea-2 only) quantizes frozen base to 4-bit NormalFloat via bnb, "
            "13B->6.6GB, enables 24GB training; verified with block_swap (offloader "
            "NF4 path via Params4bit integral transport). Mutually exclusive with "
            "ConvRot."
        ),
    )
    parser.add_argument(
        "--nf4_prequantized_path",
        type=str,
        default=None,
        help=(
            "[EXPERIMENTAL] Path to a pre-quantized NF4 safetensors (save_nf4_dit "
            "product). Legacy v1 files overlay the BF16 base; self-contained v2 files "
            "do not read the BF16 payload. A v2 file may also be supplied directly as "
            "--pretrained_model_name_or_path. Only used with --base_compute nf4. "
            "When no NF4 file is supplied, falls back to in-line quantize_dit_to_nf4."
        ),
    )
    parser.add_argument(
        "--convrot_group_size",
        type=int,
        default=256,
        choices=[64, 256, 1024],
        help="[EXPERIMENTAL] ConvRot group size for RHT (must divide in_features).",
    )
    parser.add_argument(
        "--convrot_hadamard",
        type=str,
        default="sylvester",
        choices=["sylvester", "regular"],
        help=(
            "[EXPERIMENTAL] Hadamard construction for group RHT. "
            "sylvester (default) is power-of-two FWHT-compatible; "
            "regular is paper-aligned (group must be 4^k, e.g. 64/256/1024). "
            "regular@64 measured 3/3 checkpoint gate on 3080."
        ),
    )
    parser.add_argument(
        "--convrot_scope",
        type=str,
        default="mlp",
        help=(
            "[EXPERIMENTAL] ConvRot module scope, same syntax as int8 linear scope: "
            "mlp / attention / all / self_attn_qkv / comma combinations."
        ),
    )
    parser.add_argument(
        "--convrot_weight_source",
        type=str,
        default="online_from_bf16",
        choices=["online_from_bf16", "prequant_checkpoint"],
        help=(
            "[EXPERIMENTAL] ConvRot weight source. online_from_bf16 rotates and "
            "quantizes the loaded bf16 frozen base online. prequant_checkpoint "
            "loads rotated int8 weights + per-out scales from "
            "--convrot_prequant_path (native anima_lora_convrot_prequant_v1 "
            "or {name}.weight + {name}.weight_scale pairs)."
        ),
    )
    parser.add_argument(
        "--convrot_prequant_path",
        type=str,
        default=None,
        help=(
            "[EXPERIMENTAL] Safetensors/pt path for "
            "convrot_weight_source=prequant_checkpoint. Export with "
            "scripts/experiments/convrot_export_prequant.py. Removes apply-time "
            "online weight RHT+quant only; activation RHT still runs each step."
        ),
    )
    parser.add_argument(
        "--convrot_min_in_features",
        type=int,
        default=0,
        help=(
            "[EXPERIMENTAL][P1-G] Skip ConvRot on Linears with in_features below "
            "this threshold (0=off). Useful to drop small layers where RHT fixed "
            "cost dominates (e.g. 2048 keeps Anima mlp.layer1 only when combined "
            "with largest-only, or 4096 for layer2-in)."
        ),
    )
    parser.add_argument(
        "--convrot_largest_in_features_only",
        action="store_true",
        help=(
            "[EXPERIMENTAL][P1-G] Among scope-matched modules, only patch those "
            "with the maximum in_features (Anima mlp: layer2 in=8192). Default off."
        ),
    )
    parser.add_argument(
        "--convrot_large_layer_mode",
        type=str,
        default=None,
        choices=["w8a16", "w8a8", "w8a16_convrot", "w8a8_convrot"],
        help=(
            "[EXPERIMENTAL][P1-F] Override compute mode for layers with "
            "in_features >= --convrot_large_min_in_features. Example: "
            "base_compute=w8a16_convrot + large_layer_mode=w8a8 + large_min=4096 "
            "keeps small MLP on W8A16 and big in-dim on W8A8."
        ),
    )
    parser.add_argument(
        "--convrot_large_min_in_features",
        type=int,
        default=None,
        help=(
            "[EXPERIMENTAL][P1-F] in_features threshold for "
            "--convrot_large_layer_mode. Required when large_layer_mode is set."
        ),
    )
    parser.add_argument(
        "--block_swap_restore_mode",
        type=str,
        default="slab",
        choices=["foreach", "slab"],
        help=(
            "Restore path for cached CUDA block swap. slab (default) collapses the "
            "block's frozen-weight H2D into one contiguous slab copy; on RTX 3080 "
            "this is ~0.7ms/block faster than per-tensor foreach (see "
            "docs/findings/blockswap_baseline_20260806.md). Slab requires uniform "
            "non-int8 dtype and falls back to foreach automatically otherwise."
        ),
    )
    parser.add_argument(
        "--block_swap_int8_restore_mode",
        type=str,
        default=None,
        choices=["copy", "direct_bind", "reuse_storage", "fused"],
        help=(
            "Restore path used only when block_swap_transfer_dtype=int8. copy "
            "(default) dequantizes through fp32 temporaries; reuse_storage writes "
            "chunked into the resident slot; direct_bind rebinds the slot; fused "
            "uses one torch.mul(int8, fp32, out=half) kernel straight into the slot "
            "with a small static staging ring (no dynamic allocation). On a PCIe "
            "Gen1 link fused measured 2.3986 -> 1.9740 s/step and reserved "
            "9.16 -> 7.21 GiB "
            "(docs/findings/moss_image_int8_blockswap_diagnosis_20261018.md)."
        ),
    )
    parser.add_argument(
        "--block_swap_int8_staging_ring",
        type=int,
        default=None,
        help=(
            "Staging buffers used by block_swap_int8_restore_mode=fused, i.e. the "
            "VRAM/speed dial. 0 keeps one 48 MiB buffer per swapped slot (no "
            "cross-slot event wait, fastest: measured 1.1204 -> 1.0818 s/step at "
            "swap=10) at the cost of swap x 48 MiB; N>=1 shares N buffers between "
            "all slots (fixed N x 48 MiB). 1 is the lowest-footprint setting and is "
            "what makes aggressive swap=6/7 fit (9732 MiB with 2 buffers vs ~9932 "
            "MiB per-slot, above the 9788 MiB card limit). Default 2. Env fallback: "
            "ANIMA_BLOCK_SWAP_INT8_STAGING_RING."
        ),
    )
    parser.add_argument(
        "--block_swap_int8_dual_stream",
        default=None,
        action="store_true",
        help=(
            "Split the int8 fused restore across two streams (pure DMA stream plus "
            "a priority=-1 dequant stream). Measured: no throughput change, but the "
            "per-step spread collapses from up to 1.59 s to ~0.015 s, which makes "
            "long-run ETA estimates reliable. Only meaningful with "
            "block_swap_int8_restore_mode=fused. Default off. Env fallback: "
            "ANIMA_BLOCK_SWAP_INT8_DUAL_STREAM."
        ),
    )
    parser.add_argument(
        "--block_swap_int8_scope",
        type=str,
        default=None,
        help="Comma-separated int8 candidate scope; default all. Env fallback ANIMA_BLOCK_SWAP_INT8_SCOPE.",
    )
    parser.add_argument(
        "--block_swap_int8_restore_chunk_rows",
        type=int,
        default=None,
        help=(
            "Row chunk size for block_swap_int8_restore_mode=reuse_storage; lowers "
            "the transient VRAM peak of the non-fused path. 0 (default) uses the "
            "whole tensor. Env fallback: ANIMA_BLOCK_SWAP_INT8_RESTORE_CHUNK_ROWS."
        ),
    )
    parser.add_argument(
        "--selective_checkpoint",
        type=str,
        default="off",
        choices=[
            "off",
            "adapter_aware",
            "every_other",
            "mlp_only",
            "mlp_layer1_only",
            "peak_blocks_adapter_aware",
            "peak_blocks_mlp",
            "peak_blocks_mlp_layer1",
        ],
        help=(
            "Selective DiT activation checkpointing for block-swap training. "
            "off keeps the normal path; every_other checkpoints every other block; "
            "adapter_aware checkpoints full DiT blocks but caches small adapter/router "
            "intermediates; "
            "mlp_only checkpoints each block's full MLP branch; mlp_layer1_only "
            "checkpoints only MLP layer1+GELU; peak_blocks_* applies only to "
            "--selective_checkpoint_blocks."
        ),
    )
    parser.add_argument(
        "--selective_checkpoint_blocks",
        type=str,
        default="",
        help=(
            "Comma/range list for peak_blocks selective checkpoint modes, "
            "e.g. '25-27' or '24,25,26,27'. Empty/auto defaults to the last 3 DiT blocks."
        ),
    )
    parser.add_argument(
        "--disable_block_swap_for_eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Temporarily disable block swapping during validation/sample preview. "
        "Only use when eval fits in VRAM without swapped blocks.",
    )


def add_network_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--network_weights",
        type=str,
        default=None,
        help="pretrained weights for network",
    )
    parser.add_argument(
        "--network_module",
        type=str,
        default=None,
        help="network module to train",
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network)",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version)",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons)",
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value)",
    )
    parser.add_argument(
        "--network_train_unet_only",
        action="store_true",
        help="only training U-Net part",
    )
    parser.add_argument(
        "--network_train_text_encoder_only",
        action="store_true",
        help="only training Text Encoder part",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point)",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="path to a pretrained LoRA checkpoint to merge into DiT weights before training. "
        "Intended for adapter training on top of a fixed LoRA. "
        "The LoRA is baked into the base weights at load time — no runtime hooks.",
    )
    parser.add_argument(
        "--lora_multiplier",
        type=float,
        default=1.0,
        help="multiplier applied to the frozen LoRA merged via --lora_path",
    )


def get_sanitized_config_or_none(args: argparse.Namespace):
    if not args.log_config:
        return None

    sensitive_args = ["wandb_api_key", "huggingface_token"]
    sensitive_path_args = [
        "pretrained_model_name_or_path",
        "vae",
        "tokenizer_cache_dir",
        "train_data_dir",
        "conditioning_data_dir",
        "reg_data_dir",
        "output_dir",
        "logging_dir",
    ]
    filtered_args = {}
    for k, v in vars(args).items():
        if k not in sensitive_args + sensitive_path_args:
            if (
                v is None
                or isinstance(v, bool)
                or isinstance(v, str)
                or isinstance(v, float)
                or isinstance(v, int)
            ):
                filtered_args[k] = v
            elif isinstance(v, list):
                filtered_args[k] = f"{v}"
            elif isinstance(v, object):
                filtered_args[k] = f"{v}"

    return filtered_args


def verify_command_line_training_args(args: argparse.Namespace):
    wandb_enabled = args.log_with is not None and args.log_with != "tensorboard"
    if not wandb_enabled:
        return

    sensitive_args = ["wandb_api_key", "huggingface_token"]
    sensitive_path_args = [
        "pretrained_model_name_or_path",
        "vae",
        "tokenizer_cache_dir",
        "train_data_dir",
        "conditioning_data_dir",
        "reg_data_dir",
        "output_dir",
        "logging_dir",
    ]

    for arg in sensitive_args:
        if getattr(args, arg, None) is not None:
            logger.warning(
                f"wandb is enabled, but option `{arg}` is included in the command line. It is recommended to move it to the `.toml` file."
            )

    for arg in sensitive_path_args:
        if getattr(args, arg, None) is not None and os.path.isabs(getattr(args, arg)):
            logger.info(
                f"wandb is enabled, but option `{arg}` is included in the command line and it is an absolute path. It is recommended to move it to the `.toml` file or use relative path."
            )

    if getattr(args, "config_file", None) is not None:
        logger.info(
            "wandb is enabled, but option `config_file` is included in the command line. Please be careful about the information included in the path."
        )

    if (
        args.huggingface_repo_id is not None
        and args.huggingface_repo_visibility != "public"
    ):
        logger.info(
            "wandb is enabled, but option huggingface_repo_id is included in the command line and huggingface_repo_visibility is not 'public'. It is recommended to move it to the `.toml` file."
        )


def enable_high_vram(args: argparse.Namespace):
    if args.highvram:
        logger.info("highvram is enabled")
        _datasets_base.enable_high_vram()


def verify_training_args(args: argparse.Namespace):
    from library.models.qwen_image_2_1.preview_memory import validate_config

    validate_config(args)
    enable_high_vram(args)

    if getattr(args, "blocks_to_swap", None) is not None and args.blocks_to_swap < 0:
        raise ValueError("blocks_to_swap must be greater than or equal to 0")

    if args.v2 and args.clip_skip is not None:
        logger.warning("v2 with clip_skip will be unexpected")

    # Expand the two semantic cache knobs into the legacy internal flags that
    # the dataset / strategy / metadata code still reads. `use_vae_cache` and
    # `use_text_cache` are the only config-facing surface; disk caching is the
    # only supported mode (RAM-only was never used), so the `_to_disk` siblings
    # always track their base flag. The old keys survive as schema aliases
    # (see library/config/schema.py) so pre-existing configs still resolve.
    args.cache_latents = args.cache_latents_to_disk = bool(args.use_vae_cache)
    args.cache_text_encoder_outputs = args.cache_text_encoder_outputs_to_disk = bool(
        args.use_text_cache
    )

    prior_weight = float(getattr(args, "prior_preservation_weight", 0.0) or 0.0)
    inverted_mask_prior_weight = float(
        getattr(args, "inverted_mask_prior_weight", 0.0) or 0.0
    )
    dop_enabled = bool((getattr(args, "diff_output_preservation_class", None) or "").strip())
    blank_enabled = bool(getattr(args, "blank_prompt_preservation", False))
    if prior_weight > 0.0 and not (blank_enabled or dop_enabled):
        raise ValueError(
            "prior_preservation_weight requires blank_prompt_preservation=true or "
            "diff_output_preservation_class to select a prior-preservation mode."
        )
    if prior_weight > 0.0 and blank_enabled and dop_enabled:
        raise ValueError(
            "blank_prompt_preservation and diff_output_preservation_class are mutually exclusive."
        )
    if prior_weight > 0.0 and not getattr(args, "use_text_cache", False):
        raise ValueError(
            "prior preservation requires use_text_cache=true so prior text "
            "conditions are cached before the text encoder is released."
        )
    if inverted_mask_prior_weight > 0.0 and not getattr(args, "use_text_cache", False):
        raise ValueError(
            "inverted_mask_prior requires use_text_cache=true so the base "
            "reference forward can reuse cached text conditions."
        )
    if (
        prior_weight > 0.0
        and (blank_enabled or dop_enabled)
        and not getattr(args, "cache_llm_adapter_outputs", False)
    ):
        raise ValueError(
            "prior preservation requires cache_llm_adapter_outputs=true so training "
            "batches carry max-padded crossattn_emb without re-running the text encoder."
        )
    if (
        inverted_mask_prior_weight > 0.0
        and not getattr(args, "cache_llm_adapter_outputs", False)
    ):
        raise ValueError(
            "inverted_mask_prior requires cache_llm_adapter_outputs=true so training "
            "batches carry max-padded crossattn_emb for the adapter-disabled base forward."
        )

    if args.adaptive_noise_scale is not None and args.noise_offset is None:
        raise ValueError("adaptive_noise_scale requires noise_offset")

    if args.scale_v_pred_loss_like_noise_pred and not args.v_parameterization:
        raise ValueError(
            "scale_v_pred_loss_like_noise_pred can be enabled only with v_parameterization"
        )

    if args.v_pred_like_loss and args.v_parameterization:
        raise ValueError("v_pred_like_loss cannot be enabled with v_parameterization")

    if args.zero_terminal_snr and not args.v_parameterization:
        logger.warning(
            "zero_terminal_snr is enabled, but v_parameterization is not enabled. training will be unexpected"
        )

    if args.sample_every_n_epochs is not None and args.sample_every_n_epochs <= 0:
        logger.warning(
            "sample_every_n_epochs is less than or equal to 0, so it will be disabled"
        )
        args.sample_every_n_epochs = None

    if args.sample_every_n_steps is not None and args.sample_every_n_steps <= 0:
        logger.warning(
            "sample_every_n_steps is less than or equal to 0, so it will be disabled"
        )
        args.sample_every_n_steps = None


def add_dataset_arguments(
    parser: argparse.ArgumentParser,
    support_dreambooth: bool,
    support_caption: bool,
    support_caption_dropout: bool,
):
    parser.add_argument(
        "--train_data_dir", type=str, default=None, help="directory for train images"
    )
    parser.add_argument(
        "--cache_info",
        action="store_true",
        help="cache meta information for faster dataset loading",
    )
    parser.add_argument(
        "--caption_separator", type=str, default=",", help="separator for caption"
    )
    parser.add_argument(
        "--caption_extension",
        type=str,
        default=".caption",
        help="extension of caption files",
    )
    parser.add_argument(
        "--caption_extention",
        type=str,
        default=None,
        help="extension of caption files (backward compatibility)",
    )
    parser.add_argument(
        "--keep_tokens",
        type=int,
        default=0,
        help="keep heading N tokens when shuffling caption tokens",
    )
    parser.add_argument(
        "--keep_tokens_separator",
        type=str,
        default="",
        help="A custom separator to divide the caption into fixed and flexible parts.",
    )
    parser.add_argument(
        "--secondary_separator",
        type=str,
        default=None,
        help="a secondary separator for caption.",
    )
    parser.add_argument(
        "--enable_wildcard", action="store_true", help="enable wildcard for caption"
    )
    parser.add_argument(
        "--caption_prefix", type=str, default=None, help="prefix for caption text"
    )
    parser.add_argument(
        "--caption_suffix", type=str, default=None, help="suffix for caption text"
    )
    parser.add_argument(
        "--color_aug", action="store_true", help="enable weak color augmentation"
    )
    parser.add_argument(
        "--flip_aug", action="store_true", help="enable horizontal flip augmentation"
    )
    parser.add_argument(
        "--face_crop_aug_range",
        type=str,
        default=None,
        help="enable face-centered crop augmentation and its range (e.g. 2.0,4.0)",
    )
    parser.add_argument("--random_crop", action="store_true", help="enable random crop")
    parser.add_argument(
        "--debug_dataset",
        action="store_true",
        help="show images for debugging (do not train)",
    )
    parser.add_argument(
        "--use_vae_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache VAE-encoded latents to disk and read them during training. "
        "Set false for live VAE encoding (e.g. IP-Adapter live mode, where "
        "batch['images'] carries the raw reference). Subsumes the legacy "
        "cache_latents{,_to_disk} pair.",
    )
    parser.add_argument(
        "--vae_batch_size", type=int, default=1, help="batch size for caching latents"
    )
    parser.add_argument(
        "--skip_cache_check",
        action="store_true",
        help="skip the content validation of cache",
    )
    parser.add_argument(
        "--require_cache_identity",
        action="store_true",
        help=(
            "Treat preprocessed sidecars without a cache-identity stamp as "
            "stale and refuse them, instead of reusing them silently. Sidecars "
            "stamped with different text-encoder / VAE / tokenizer weights are "
            "always rejected. Re-run preprocessing (or pass --overwrite to the "
            "preprocess task) to stamp and rebuild."
        ),
    )
    parser.add_argument(
        "--resize_interpolation",
        type=str,
        default=None,
        choices=[
            "lanczos",
            "nearest",
            "bilinear",
            "linear",
            "bicubic",
            "cubic",
            "area",
        ],
        help="Resize interpolation",
    )
    parser.add_argument(
        "--token_warmup_min", type=int, default=1, help="start learning at N tags"
    )
    parser.add_argument(
        "--token_warmup_step",
        type=float,
        default=0,
        help="tag length reaches maximum on N steps",
    )
    parser.add_argument(
        "--alpha_mask",
        action="store_true",
        help="use alpha channel as mask for training",
    )
    parser.add_argument(
        "--dataset_class",
        type=str,
        default=None,
        help="dataset class for arbitrary dataset (package.module.Class)",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=None,
        help=(
            "Global override applied to every subset's sample_ratio (0<r≤1). "
            "Unset = use each subset's own value. Exposed here so presets like "
            "`[half]` can propagate a single value across the dataset blueprint."
        ),
    )
    parser.add_argument(
        "--path_pattern",
        type=str,
        default=None,
        help=(
            "fnmatch glob applied to each image's path relative to its "
            "subset's image_dir. `|` separates alternatives — e.g. "
            "`char_a/*` keeps only the char_a/ subfolder, "
            "`char_a/*|char_b/*` keeps either. Unset / `*` = use everything. "
            "Validation and image-count thresholds honour the filtered pool."
        ),
    )

    if support_caption_dropout:
        parser.add_argument(
            "--caption_dropout_rate",
            type=float,
            default=0.0,
            help="Rate out dropout caption(0.0~1.0)",
        )
        parser.add_argument(
            "--caption_dropout_every_n_epochs",
            type=int,
            default=0,
            help="Dropout all captions every N epochs",
        )
        parser.add_argument(
            "--caption_tag_dropout_rate",
            type=float,
            default=0.0,
            help="Rate out dropout comma separated tokens(0.0~1.0)",
        )

    if support_dreambooth:
        parser.add_argument(
            "--reg_data_dir",
            type=str,
            default=None,
            help="directory for regularization images",
        )

    if support_caption:
        parser.add_argument(
            "--in_json", type=str, default=None, help="json metadata for dataset"
        )
        parser.add_argument(
            "--dataset_repeats",
            type=int,
            default=1,
            help="repeat dataset when training with captions",
        )


def add_sd_saving_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--save_model_as",
        type=str,
        default=None,
        choices=[None, "ckpt", "safetensors", "diffusers", "diffusers_safetensors"],
        help="format to save the model (default is same to original)",
    )
    parser.add_argument(
        "--use_safetensors", action="store_true", help="use safetensors format to save"
    )


def setup_parser() -> argparse.ArgumentParser:
    """Build the full Anima training ArgumentParser.

    Extracted from train.py so the entrypoint stays a thin wrapper. Flag
    surface and defaults must stay identical.
    """
    from library.anima import training as anima_train_utils
    from library.config import loader as config_util
    from library.log import add_logging_arguments
    from library.models import sai_spec as sai_model_spec
    from library.training.losses import add_custom_train_arguments

    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    add_sd_models_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    add_dataset_arguments(parser, True, True, True)
    add_training_arguments(parser, True)
    add_masked_loss_arguments(parser)
    add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)
    add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)

    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="[EXPERIMENTAL] enable offloading of tensors to CPU during checkpointing for U-Net or DiT, if supported"
        "",
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save metadata in output model",
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors)",
    )

    parser.add_argument(
        "--unet_lr",
        type=float,
        default=None,
        help="learning rate for U-Net",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=None,
        nargs="*",
        help="learning rate for Text Encoder, can be multiple",
    )

    add_network_arguments(parser)
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help=(
            "Run the VAE in fp32 (never half). Forces fp32 unconditionally on "
            "every GPU and precision."
        ),
    )
    parser.add_argument(
        "--half_vae",
        action="store_true",
        help=(
            "Explicitly allow the VAE to run in half precision, overriding the "
            "automatic fp32 protection that kicks in on pre-Ampere GPUs (sm<8, "
            "e.g. V100/T4) under fp16 training. NOT recommended there: fp16 VAE "
            "decode can produce artifacts (花图/糊图). No-op on Ampere+ or under "
            "bf16/fp32."
        ),
    )
    parser.add_argument(
        "--skip_until_initial_step",
        action="store_true",
        help="skip training until initial_step is reached",
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=None,
        help="initial epoch number, 1 means first epoch (same as not specifying). NOTE: initial_epoch/step doesn't affect to lr scheduler. Which means lr scheduler will start from 0 without `--resume`."
        + "",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=None,
        help="initial step number including all epochs, 0 means first step (same as not specifying). overwrites initial_epoch."
        + "",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=None,
        help="Validation seed for shuffling validation dataset, training `--seed` used otherwise",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.0,
        help="Split for validation images out of the training dataset",
    )
    parser.add_argument(
        "--validation_split_num",
        type=int,
        default=0,
        help=(
            "Count-based validation split (number of held-out images). When "
            "set (>0), wins over the fractional `--validation_split`. Also "
            "determines how many samples CMMD evaluation generates per pass."
        ),
    )
    parser.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=None,
        help="Run validation on validation dataset every N steps. By default, validation will only occur every epoch if a validation dataset is available",
    )
    parser.add_argument(
        "--validate_every_n_epochs",
        type=int,
        default=None,
        help="Run validation dataset every N epochs. By default, validation will run every epoch if a validation dataset is available",
    )
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=None,
        help="Max number of validation dataset items processed. By default, validation will run the entire validation dataset",
    )
    parser.add_argument(
        "--validation_sigmas",
        type=float,
        nargs="+",
        default=None,
        help="Sigma values for validation loss (0.0~1.0). Low values = fine detail. Default: 0.1 0.4 0.7. (Legacy FM-val path — unused under the CMMD val replacement.)",
    )
    parser.add_argument(
        "--validation_sample_steps",
        type=int,
        default=20,
        help="Denoising steps used by CMMD validation when sampling each held-out item. Default 20.",
    )
    parser.add_argument(
        "--validation_cfg_scale",
        type=float,
        default=1.0,
        help="CFG scale used by CMMD validation. Default 1.0 (no CFG, fastest). Bump to 4.0 to match production sampling but generation cost ~2×.",
    )
    parser.add_argument(
        "--use_cmmd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CMMD (PE-Core MMD²) as the validation signal. Set "
        "`use_cmmd = false` in the method TOML (or pass `--no-use_cmmd`) to "
        "skip CMMD and run only the legacy per-σ FM-MSE val pass — useful "
        "on tight VRAM where the PE encoder + sampling path doesn't fit.",
    )
    parser.add_argument(
        "--validation_baselines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run each method adapter's validation baselines (e.g. IP-Adapter "
        "no_ip / shuffled_ref) as FM-MSE delta diagnostics during validation. "
        "Set `validation_baselines = false` in the method TOML (or pass "
        "`--no-validation_baselines`) to skip them — each baseline adds a full "
        "extra val forward per (batch, σ), so this roughly halves IP-Adapter "
        "validation time when you don't need the deltas.",
    )
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). "
        "Cannot be used with --cpu_offload_checkpointing or --blocks_to_swap.",
    )
    parser.add_argument(
        "--print-config",
        dest="print_config",
        action="store_true",
        help="Dump the fully merged config (base → preset → method → CLI) as TOML "
        "with provenance comments, then exit 0. Does not start training.",
    )
    parser.add_argument(
        "--config-snapshot",
        dest="config_snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write output/<output_name>.snapshot.toml next to the checkpoint on every real "
        "run (provenance + git SHA). Pass --no-config-snapshot to disable.",
    )
    parser.add_argument(
        "--config-strict",
        dest="config_strict",
        action="store_true",
        help="Treat config-schema warnings (unknown keys, off-list choices) as errors.",
    )
    return parser
