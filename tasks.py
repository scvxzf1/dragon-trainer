#!/usr/bin/env python3
"""Cross-platform task runner -- replaces Makefile for Windows compatibility.

Usage:
    python tasks.py <command> [extra args...]

Examples:
    python tasks.py lora
    python tasks.py lora --network_dim 32 --max_train_epochs 64
    python tasks.py test
    python tasks.py test                     # add SPECTRUM=1 to enable Spectrum
    python tasks.py test                     # add MOD=1 to enable modulation guidance
    python tasks.py test                     # add NOLORA=1 to run against the bare DiT
    python tasks.py download-models
    python tasks.py exp-chimera              # experimental method
    python tasks.py exp-test-ip ref.png      # experimental inference

Command implementations live under ``scripts/tasks/`` (shipped methods) and
``scripts/experimental_tasks/`` (unstable methods exposed under ``exp-*``).
This file is just a name → callable dispatch table.
"""

import os
import sys

from scripts.experimental_tasks import inference as exp_inference
from scripts.experimental_tasks import parallel as exp_parallel
from scripts.experimental_tasks import nm_w4a16 as exp_nm_w4a16
from scripts.experimental_tasks import training as exp_training
from scripts.tasks import (
    daemon,
    dcw,
    downloads,
    inference,
    masking,
    preprocess,
    tagger,
    training,
    utilities,
    v100_flash,
    web,
)

COMMANDS = {
    # ── Training ──────────────────────────────────────────────────────
    "lora": (
        training.cmd_lora,
        "LoRA family (lora|tlora|tlora_rf|hydralora via configs/methods/lora.toml)",
    ),
    "register": (
        training.cmd_register,
        "Register-token adapter on a frozen DiT (configs/methods/register.toml).",
    ),
    "lora-gui": (
        training.cmd_lora_gui,
        "Train from a self-contained configs/gui-methods/<variant>.toml "
        "(variant from GUI_PRESETS env or 1st positional; e.g. tlora, hydralora, reft).",
    ),
    # ── Training daemon ───────────────────────────────────────────────
    "daemon": (
        daemon.cmd_daemon,
        "Start the local training-job daemon (idempotent; detached, waits for /health).",
    ),
    "daemon-attach": (
        daemon.cmd_daemon_attach,
        "Follow the daemon (read-only). JOB=<id> tails that job's stdout; "
        "ctrl-C detaches only — training keeps running.",
    ),
    "daemon-jobs": (
        daemon.cmd_daemon_jobs,
        "List persisted daemon jobs; works while the daemon is stopped.",
    ),
    "daemon-log": (
        daemon.cmd_daemon_log,
        "Read the latest job stdout, or JOB=<id>; -n N tails N lines (default 100).",
    ),
    "daemon-kill": (
        daemon.cmd_daemon_kill,
        "Abort the running job (or JOB=<id>) and free the GPU; daemon stays up "
        "and starts the next queued job.",
    ),
    "daemon-terminate": (
        daemon.cmd_daemon_terminate,
        "Stop the daemon entirely (active job killed, GPU freed, queue discarded).",
    ),
    "v100-flash-install": (
        v100_flash.cmd_install,
        "Build and install the pinned flash-attention-v100 cp313 wheel.",
    ),
    "v100-flash-validate": (
        v100_flash.cmd_validate,
        "Run the strict V100 FlashAttention acceptance suite.",
    ),
    # ── Inference ─────────────────────────────────────────────────────
    "test": (
        inference.cmd_test,
        "Inference with latest LoRA. SPECTRUM=1 enables Spectrum acceleration; "
        "MOD=1 adds the latest distilled pooled_text_proj (modulation guidance); "
        "NOLORA=1 runs against the bare DiT (skips --lora_weight).",
    ),
    "test-mod": (
        inference.cmd_test_mod,
        "Compatibility alias for `test` with modulation guidance enabled.",
    ),
    "test-hydra": (
        inference.cmd_test_hydra,
        "Inference with latest HydraLoRA moe (router-live)",
    ),
    "test-merge": (
        inference.cmd_test_merge,
        "Inference with latest *_merged.safetensors (MODEL_DIR=..., default 'output_temp')",
    ),
    "test-dcw": (
        inference.cmd_test_dcw,
        "Inference with latest LoRA + DCW post-step bias correction. "
        "Honors SPECTRUM=1 / MOD=1 / NOLORA=1.",
    ),
    "test-smc-cfg": (
        inference.cmd_test_smc_cfg,
        "Inference with latest LoRA + SMC-CFG (sliding-mode control CFG, arXiv:2603.03281). "
        "Honors SPECTRUM=1 / MOD=1 / NOLORA=1.",
    ),
    "test-dcw-v4": (
        inference.cmd_test_dcw_v4,
        "Inference with DCW v4 learnable calibrator (auto-resolves fusion_head.safetensors). "
        "No LoRA by default; SPECTRUM=1 / MOD=1 / NOLORA=0 (attach latest LoRA) all compose.",
    ),
    "dcw": (
        dcw.cmd_dcw,
        "Calibrate DCW v4: sample 5 aspect buckets (default 8×2 seeds, "
        "shuffle_seed=0) + train fusion head",
    ),
    "dcw-train": (
        dcw.cmd_dcw_train,
        "Train-only on existing output/dcw/ pool (~30s, no sampling)",
    ),
    "test-spectrum-dcw": (
        inference.cmd_test_spectrum_dcw,
        "Spectrum-accelerated inference + DCW post-step bias correction",
    ),
    "test-dcw-v4-spectrum": (
        inference.cmd_test_dcw_v4_spectrum,
        "Spectrum-accelerated inference + DCW v4 learnable calibrator (auto-resolves fusion_head.safetensors)",
    ),
    # ── Preprocess ────────────────────────────────────────────────────
    "preprocess": (
        preprocess.cmd_preprocess,
        "Full preprocessing (resize + VAE + text embeddings)",
    ),
    "preprocess-config": (
        preprocess.cmd_preprocess_config,
        "Preprocess the dirs named in a --dataset_config TOML (resize --src "
        "→ image_dir, then VAE + TE caches → cache_dir). Used by the trainer node.",
    ),
    "preprocess-resize": (
        preprocess.cmd_preprocess_resize,
        "Resize images to bucket resolutions",
    ),
    "preprocess-vae": (preprocess.cmd_preprocess_vae, "Cache VAE latents"),
    "preprocess-te": (preprocess.cmd_preprocess_te, "Cache text encoder embeddings"),
    "preprocess-pooled": (
        preprocess.cmd_preprocess_pooled,
        "Cache pooled text embeddings sidecar (consumed by distill-mod). No GPU.",
    ),
    "preprocess-pe": (
        preprocess.cmd_preprocess_pe,
        "Cache PE-Core vision-encoder features into the LoRA cache dir. "
        "Consumed by IP-Adapter live-disk mode and the DCW v4 fusion head.",
    ),
    "caption-index": (
        preprocess.cmd_caption_index,
        "Build the typed-tag caption index (character/copyright/artist groups) "
        "at post_image_dataset/captions/caption_index.json. Pure data, no GPU.",
    ),
    # ── Anima Tagger ──────────────────────────────────────────────────
    "preprocess-tagger": (
        tagger.cmd_preprocess_tagger,
        "Build the Anima Tagger vocab/manifest + cache pooled PE features "
        "+ cache resized images (build_vocab + build_features + "
        "build_resized). Needs CAPTION_CORPUS_DIR in .env.",
    ),
    "tagger": (
        tagger.cmd_tagger,
        "Two-stage train: Stage 1 head-only on cached pooled features, "
        "then Stage 2 PE-LoRA warm-started from Stage 1. Requires "
        "`make preprocess-tagger` first.",
    ),
    "test-tagger": (
        tagger.cmd_test_tagger,
        "Predict tags for a single image (--image <path>) or sample a random "
        "val-split stem. Pass --show_scores for rating + top-K kept tags.",
    ),
    # ── Downloads ─────────────────────────────────────────────────────
    "download-models": (downloads.cmd_download_models, "Download all models"),
    "download-anima": (downloads.cmd_download_anima, "Download Anima model"),
    "download-krea2": (
        downloads.cmd_download_krea2,
        "Download Krea-2-Raw base (DiT bf16 + Qwen3-VL + VAE) for the krea2_raw family",
    ),
    "download-anima-variant": (
        downloads.cmd_download_anima_variant,
        "Download an alternate Anima base DiT (aesthetic / turbo / preview); "
        "ARGS=<name>, no args lists them",
    ),
    "download-sam3": (downloads.cmd_download_sam3, "Download SAM3 model"),
    "download-mit": (downloads.cmd_download_mit, "Download MIT model"),
    "download-pe": (
        downloads.cmd_download_pe,
        "Download PE-Core-L14-336 (img2emb encoder)",
    ),
    "download-pe-spatial": (
        downloads.cmd_download_pe_spatial,
        "Download PE-Spatial-B16-512 (Anima Tagger aux encoder)",
    ),
    "download-tagger": (
        downloads.cmd_download_tagger,
        "Download Anima Tagger v2 vocab.json for caption-index/preprocess metadata.",
    ),
    # ── Masking ───────────────────────────────────────────────────────
    "mask": (
        masking.cmd_mask,
        "Run SAM + MIT (via tempdir) and write merged masks under post_image_dataset/masks/",
    ),
    "mask-clean": (
        masking.cmd_mask_clean,
        "Remove post_image_dataset/masks/",
    ),
    "web": (web.cmd_web, "Launch lightweight Web UI on port 20102 (--port N to override)"),
    "web-next-build": (web.cmd_frontend_build, "Build and safely publish the isolated /next frontend"),
    "web-next-check": (web.cmd_frontend_check, "Typecheck and test the /next frontend"),
    "web-next-e2e": (web.cmd_frontend_e2e, "Run isolated /next browser tests with mocked API commands"),
    # ── Utilities ─────────────────────────────────────────────────────
    "merge": (
        utilities.cmd_merge,
        "Bake latest LoRA (ADAPTER_DIR=..., default 'output/ckpt') into base DiT",
    ),
    "comfy-batch": (utilities.cmd_comfy_batch, "Run ComfyUI batch workflow"),
    "distill-prep": (
        utilities.cmd_distill_prep,
        'Pre-stage artifacts for distill-mod: T5("") uncond sidecar + '
        "teacher-synthetic clean latents pool (--skip_synth / --skip_uncond to "
        "stage only one).",
    ),
    "distill-mod": (
        utilities.cmd_distill_mod,
        "Distill pooled_text_proj MLP for modulation guidance",
    ),
    "test-unit": (
        utilities.cmd_test_unit,
        "Run every pytest layer (legacy compatibility entry).",
    ),
    "test-core": (
        utilities.cmd_test_core,
        "Run tests outside explicitly marked integration, hardware, and research layers.",
    ),
    "test-integration": (
        utilities.cmd_test_integration,
        "Run integration tests that do not require accelerator hardware.",
    ),
    "test-hardware": (
        utilities.cmd_test_hardware,
        "Run explicitly enabled accelerator hardware tests.",
    ),
    "test-experimental": (
        utilities.cmd_test_experimental,
        "Run benchmark helper and research probe tests.",
    ),
    "test-all": (utilities.cmd_test_all, "Run every test layer."),
    "test-backend-smoke": (utilities.cmd_test_backend_smoke, "Run backend web/training smoke subset"),
    "test-fast": (
        utilities.cmd_test_fast,
        "Run the fast smoke layer (pytest fast markers over runner/bench guards).",
    ),
    "test-focused": (
        utilities.cmd_test_focused,
        "Run an explicit pytest target or selector; refuses empty target lists.",
    ),
    "type-check": (
        utilities.cmd_type_check,
        "Run the configured pyright pilot gate.",
    ),
    "export-logs": (
        utilities.cmd_export_logs,
        "Dump TB scalar logs to JSON (RUN=<dir>, ALL=1 for every subrun, JSONL=1 for line-delimited)",
    ),
    "run-status": (
        utilities.cmd_run_status,
        "Step/ETA/last-ckpt one-liner for a run from its progress.jsonl "
        "(RUN=<output_name|path>, default newest; ARGS='--list|--json')",
    ),
    "print-config": (
        utilities.cmd_print_config,
        "Dump merged config (METHOD=<name> PRESET=<name>) with provenance",
    ),
    "explain-config": (
        utilities.cmd_explain_config,
        "Explain config value history (METHOD=<name> PRESET=<name>, pass -- --key name)",
    ),
    "config-compat": (
        utilities.cmd_config_compat,
        "Print compile/checkpoint/block-swap compatibility diagnostics",
    ),
    "training-hot": (
        utilities.cmd_training_hot,
        "Run short training hot-test matrix (pass -- --dry-run, --case gui:loha, etc.)",
    ),
    "moss-tier-hot": (
        utilities.cmd_moss_tier_hot,
        "Run the MOSS-Image resolution / text-cap tier hot test (pass -- plan|all|report)",
    ),
    "update": (
        utilities.cmd_update,
        "Update from GitHub release (preserves datasets/output/models, prompts on "
        "config conflicts, runs uv sync). Pass --dry-run / --version v1.0 / --no-sync.",
    ),
    "vendor-sync": (
        utilities.cmd_vendor_sync,
        "Refresh custom_nodes/*/_vendor/ from live library/*; pass --check for read-only drift detection",
    ),
    # ── Experimental ──────────────────────────────────────────────────
    # Unstable methods kept under exp-* so they don't pollute the main command
    # surface. May produce broken output, change without notice, or be removed.
    "exp-nm-w4a16": (
        exp_nm_w4a16.cmd_nm_w4a16,
        "[experimental] 2:4 INT4/BF16 static costs or explicit GPU probe against dense BF16.",
    ),
    "exp-anima-pipeline-bench": (
        exp_parallel.cmd_anima_pipeline_bench,
        "[experimental] Fixed-effective-batch Anima PP/single/TP benchmark and PP resume validation.",
    ),
    "exp-turbo": (
        exp_training.cmd_turbo,
        "[experimental] DP-DMD distillation — bakes CFG=4 / 28-step Anima "
        "into a 2-step LoRA student (configs/methods/turbo.toml). "
        "Single-GPU bespoke loop (bypasses train.py/accelerate, like distill-mod).",
    ),
    "exp-turbo-prep": (
        exp_training.cmd_turbo_prep,
        "[experimental] Turbo dataset curation — score cached stems and emit "
        "post_image_dataset/turbo_prep/keep_list.json.",
    ),
    "exp-spd": (
        exp_training.cmd_spd,
        "[experimental] SPD fine-tuning LoRA — §4.3 trajectory adapter that teaches a "
        "plain LoRA to follow the SPD multi-resolution trajectory (configs/methods/spd.toml). "
        "Single-GPU bespoke loop (bypasses train.py/accelerate, like distill-mod). "
        "Output is a normal LoRA — infer with the SPD sampler at the trained schedule.",
    ),
    "exp-soft-tokens": (
        exp_training.cmd_soft_tokens,
        "[experimental] SoftREPA-style per-layer × per-t soft tokens training (infer via exp-test-soft)",
    ),
    "exp-chimera": (
        exp_training.cmd_chimera,
        "[experimental] ChimeraHydra dual-pool additive routing "
        "(content + freq pools on OrthoHydra; configs/methods/chimera.toml)",
    ),
    "exp-ip-adapter": (
        exp_training.cmd_ip_adapter,
        "[experimental] IP-Adapter training (decoupled image cross-attention)",
    ),
    "exp-ip-adapter-preprocess": (
        exp_training.cmd_ip_adapter_preprocess,
        "[experimental] Alias for `preprocess` + `preprocess-pe` (IP-Adapter "
        "reuses the LoRA pipeline's caches under post_image_dataset/lora/).",
    ),
    "exp-easycontrol": (
        exp_training.cmd_easycontrol,
        "[experimental] EasyControl training (extended self-attn KV with VAE-encoded reference)",
    ),
    "exp-easycontrol-preprocess": (
        exp_training.cmd_easycontrol_preprocess,
        "[experimental] Full EasyControl preprocess: latents + text emb. "
        "Source: easycontrol-dataset/  Cache: post_image_dataset/easycontrol/.",
    ),
    "exp-easycontrol-download": (
        exp_training.cmd_easycontrol_download,
        "[experimental] Download an EasyControl adapter's extra weights. "
        "EASYADAPTER=colorize downloads Sketch2Manga screening weights.",
    ),
    "exp-byg": (
        exp_training.cmd_byg,
        "[experimental] BYG unpaired instruction-editing training "
        "(configs/methods/byg.toml). Run exp-byg-data first.",
    ),
    "exp-byg-data": (
        exp_training.cmd_byg_data,
        "[experimental] Build BYG edit-tuple sidecars into post_image_dataset/byg/.",
    ),
    "exp-test-soft": (
        exp_inference.cmd_test_soft,
        "[experimental] Inference with latest soft_tokens weight "
        "(SoftREPA-style per-layer × per-t bank, spliced into cross-attn via "
        "monkey-patched Block.forward). Composes freely with --spectrum.",
    ),
    "exp-test-turbo": (
        exp_inference.cmd_test_turbo,
        "[experimental] Inference with latest turbo student LoRA at 2 steps, cfg=1.0 "
        "(CFG is baked into the student).",
    ),
    "exp-test-spd": (
        exp_inference.cmd_test_spd,
        "[experimental] Inference with latest SPD fine-tune LoRA on the SPD sampler "
        "at its trained schedule (read from safetensors metadata). cfg=4.0, Euler.",
    ),
    "exp-test-ip": (
        exp_inference.cmd_test_ip,
        "[experimental] Inference with latest IP-Adapter weight. Usage: exp-test-ip <ref_image> [--prompt ... --ip_scale ...]",
    ),
    "exp-test-easycontrol": (
        exp_inference.cmd_test_easycontrol,
        "[experimental] Inference with latest EasyControl weight. Usage: exp-test-easycontrol <ref_image> [--prompt ... --easycontrol_scale ...]",
    ),
    "exp-test-byg": (
        exp_inference.cmd_test_byg,
        "[experimental] BYG inference placeholder. Training is available via exp-byg.",
    ),
    "exp-test-directedit": (
        exp_inference.cmd_test_directedit,
        "[experimental] DirectEdit on a random source image. PROMPT='...' supplies the edit "
        "instruction (appended to the Anima Tagger source caption). REF_IMAGE=path overrides the "
        "random pick. Usage: exp-test-directedit [ref_image] [extra...]",
    ),
    "exp-test-directedit-dry": (
        exp_inference.cmd_test_directedit_dry,
        "[experimental] DirectEdit functional check: random source image + random crossattn "
        "embed (no TE, no captioner); ψ_tar == ψ_src so output should reconstruct the source. "
        "REF_IMAGE=path overrides the random pick. Usage: exp-test-directedit-dry [ref_image] [extra...]",
    ),
    "exp-invert-directedit": (
        exp_inference.cmd_invert_directedit,
        "[experimental] Probe: invert the K-slot orthogonal postfix tail per image, then run "
        "DirectEdit dry mode twice (baseline T5(tags) vs T5(tags)+tail). Outputs side-by-side under "
        "output/tests/invert_directedit/<stem>/. Env: N_IMAGES (default 1), REF_IMAGE, K (default 48), "
        "INVERT_STEPS, LAMBDA_ZERO, SIGMA_MIN, BASIS, SEED.",
    ),
}


def _is_inline_env_assignment(token: str) -> bool:
    if token.startswith("-") or "=" not in token:
        return False
    key, _value = token.split("=", 1)
    return (
        bool(key)
        and not key[0].isdigit()
        and key.upper() == key
        and all(ch.isalnum() or ch == "_" for ch in key)
    )


def _apply_inline_env(extra: list[str]) -> list[str]:
    """Support Make-style trailing env vars: ``tasks.py cmd KEY=value``."""
    forwarded: list[str] = []
    for token in extra:
        if _is_inline_env_assignment(token):
            key, value = token.split("=", 1)
            os.environ[key] = value
        else:
            forwarded.append(token)
    return forwarded


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python tasks.py <command> [extra args...]\n")
        print("Commands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:30s} {desc}")
        print("\nExtra arguments are forwarded to the underlying command.")
        print("Example: python tasks.py lora --network_dim 32 --max_train_epochs 64")
        print("Example: python tasks.py print-config METHOD=lora PRESET=default")
        sys.exit(0)

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run 'python tasks.py --help' for available commands", file=sys.stderr)
        sys.exit(1)

    extra = _apply_inline_env(sys.argv[2:])
    fn, desc = COMMANDS[command]
    if extra and extra[0] in ("-h", "--help"):
        print(f"python tasks.py {command} -- {desc}\n")
        if fn.__doc__:
            print(fn.__doc__.strip())
        else:
            print("(no detailed help available)")
        print(
            "\nUnrecognised flags are forwarded verbatim to the underlying script. "
            "Run the underlying script with --help for its full flag set."
        )
        sys.exit(0)
    fn(extra)


if __name__ == "__main__":
    main()
