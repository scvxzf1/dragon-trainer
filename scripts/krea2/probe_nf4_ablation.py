"""R-verify: Krea-2 NF4 × {完整检查点, 块交换} 消融矩阵统一探针.

围绕三个显存优化正交轴做消融, 每个配置跑 N 步训练, 六维指标全记录到 JSONL:
  1. 显存: GPU peak / allocated (训练 loop 内 reset_peak 后)
  2. 内存: host RSS 峰值 / 系统可用内存 (主战场, bf16 master 22.64GB 曾宕机)
  3. 速度: avg/首/末 step 时间, 吞吐
  4. loss: first5 / last5 / 单调下降 / 量级
  5. 数学实现: flow-matching 公式快照 (σ 采样, target=noise-latent, velocity 预测,
     5D latent (B,C,T=1,H,W) 不变量) — 记录配置 + 公式版本, 证训练图正确
  6. 数学偏移: NF4 vs bf16 forward delta (纯量化误差). 控制变量: producer 和
     consumer 都用**训练前初始 LoRA** (zero-init, B=0 → LoRA delta=0, 等于纯
     frozen DiT forward) 各跑一次 inference_mode forward, delta 才是纯量化对
     DiT forward 的偏移, 不混入训练轨迹分叉. producer 落盘 ref velocity,
     consumer 读 ref 算 max delta / rel L2 / cosine.

三个正交轴 (环境变量开关):
  K2_ABL_NF4=0/1          (NF4 量化冻结 DiT; 0=bf16 基线)
  K2_ABL_SWAP=N           (块交换 blocks_to_swap; 0=off)
  K2_ABL_CKPT=0/1         (完整检查点: 训练中途存 LoRA+opt state, reload 续训,
                            验证 loss 连续性 + round-trip delta)
  K2_ABL_GRAD_CKPT=full|every_other|full_except|off
                           (激活检查点消融; 默认 full)
  K2_ABL_UNCKPT_BLOCKS=26,27
                           (full_except 模式下不 checkpoint 的 block)
  K2_ABL_COMPILE=0/1       (在 LoRA + grad-ckpt 之后编译 block._forward)
  K2_ABL_ATTN_MODE=torch|flash
  K2_ABL_LORA_DIM=N        (LoRA rank; 默认 16)
  K2_ABL_LORA_ALPHA=N      (LoRA alpha; 默认 8)
  K2_ABL_IMG=1024         (分辨率)
  K2_ABL_STEPS=30         (训练步数; CKPT 模式下中途存+续训各跑一半)
  K2_ABL_GPU=1            (PG199)
  K2_ABL_NF4_PATH=path    (磁盘 NF4, 跳过在线量化; 留空则在线量化)
  K2_ABL_TE_CPU=0/1       (TE 留 CPU encode, 小卡用)
  K2_ABL_TAG=name         (消融格标签, 写进 JSONL + 文件名)
  K2_ABL_REF=path         (bf16 reference velocity .pt, 数学偏移基准; bf16 格
                            自身用 K2_ABL_REF_OUT 落盘 reference)
  K2_ABL_OUT=path         (JSONL 输出; 默认 docs/findings/krea2_nf4_ablation.jsonl)

消融矩阵 (每格一个 JSONL 记录):
  bf16基线      NF4=0 SWAP=0 CKPT=0  ← 落盘 reference velocity
  NF4-only      NF4=1 SWAP=0 CKPT=0  ← 读 ref 算偏移
  NF4+swap      NF4=1 SWAP=4 CKPT=0
  NF4+ckpt      NF4=1 SWAP=0 CKPT=1  ← 中途存+续训, loss 连续性
  NF4+swap+ckpt NF4=1 SWAP=4 CKPT=1  ← 三轴全开

PG199 32GB. 一次只跑一个配置 (GPU 串行), 由调度层按矩阵逐格跑.
非目标: 真实数据集 sweep, 跨机复现, FID (留生产路径).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2_ABL_GPU", "1"))

import json
import resource
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from library.models.krea2_raw.family import Krea2TextEmbedding, forward_for_loss  # noqa: E402
from library.models.krea2_raw.lora_targets import krea2_target_kwargs  # noqa: E402
from library.models.krea2_raw.strategy import (  # noqa: E402
    Krea2TextEncodingStrategy,
    Krea2TokenizeStrategy,
    load_krea2_text_encoder,
)
from library.models.krea2_raw.weights import load_krea2_dit  # noqa: E402
from library.models.qwen_vae import load_vae  # noqa: E402
from networks.lora_anima.config import LoRANetworkCfg  # noqa: E402
from networks.lora_anima.network import LoRANetwork  # noqa: E402
from networks.lora_modules.lora import LoRAModule  # noqa: E402

from probe_train import FIXED_SIGMA, PROMPT, gpu_power, make_test_image  # noqa: E402

KREA2_DIT = ROOT / "models" / "diffusion_models" / "krea2_raw_bf16.safetensors"
KREA2_TE = ROOT / "models" / "text_encoders" / "qwen3vl_4b_bf16.safetensors"
KREA2_VAE = ROOT / "models" / "vae" / "qwen_image_vae.safetensors"

LORA_DIM = 16
LORA_ALPHA = 8.0
LR = 2e-3
FINDINGS_DIR = ROOT / "docs" / "findings"
CKPT_DIR = ROOT / "output" / "tests" / "krea2_ablation"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip() in ("1", "true", "yes", "on")


def host_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def meminfo_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return -1.0


def make_network(
    dit, lora_dim: int = LORA_DIM, lora_alpha: float = LORA_ALPHA
) -> LoRANetwork:
    kwargs = {
        **krea2_target_kwargs(),
        "model_family": "krea2_raw",
        "lora_dim": lora_dim,
        "alpha": lora_alpha,
    }
    cfg = LoRANetworkCfg.from_kwargs(
        kwargs,
        network_dim=lora_dim,
        network_alpha=lora_alpha,
        neuron_dropout=None,
        module_class=LoRAModule,
    )
    net = LoRANetwork(text_encoders=[], unet=dit, cfg=cfg, multiplier=1.0)
    net.apply_to(text_encoders=[], unet=dit, apply_text_encoder=False, apply_unet=True)
    return net


def encode_te_vae(img_size, device, dtype, te_device):
    """TE -> hiddens -> free; VAE -> latents -> free. 返回 (hiddens, txtmask, latents_4d)."""
    cache_path = os.environ.get("K2_ABL_CONDITIONS_CACHE", "") or None
    if cache_path and Path(cache_path).is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        return cached["hiddens"], cached["txtmask"], cached["latents_4d"]
    te_model, tokenizer = load_krea2_text_encoder(str(KREA2_TE), dtype=dtype, device=te_device)
    tok = Krea2TokenizeStrategy()
    tokens = tok.tokenize([PROMPT])
    enc = Krea2TextEncodingStrategy()
    with torch.inference_mode():
        [hiddens, txtmask] = enc.encode_tokens(tok, [te_model], tokens)
    hiddens = hiddens.to("cpu")
    txtmask = txtmask.to("cpu")
    del te_model, tokens, enc, tok
    if te_device == "cuda":
        torch.cuda.empty_cache()

    vae = load_vae(str(KREA2_VAE), device=device, dtype=dtype, eval=True)
    pixels = make_test_image(img_size, device, dtype)
    with torch.no_grad():
        latents_4d = vae.encode_pixels_to_latents(pixels)
    latents_4d = latents_4d.to("cpu")
    del vae, pixels
    torch.cuda.empty_cache()
    if cache_path:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"hiddens": hiddens, "txtmask": txtmask, "latents_4d": latents_4d},
            cache_file,
        )
    return hiddens, txtmask, latents_4d


def install_finite_diagnostics(dit):
    """Install opt-in forward/backward hooks for one full-model probe."""
    if not _env_bool("K2_ABL_DIAG", False):
        return []
    handles = []
    seen = {"forward": set(), "backward": set()}

    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value.detach()).all())
        if isinstance(value, (tuple, list)):
            return all(finite(item) for item in value)
        return True

    def forward_hook(name):
        def hook(_module, _inputs, output):
            if not finite(output) and name not in seen["forward"]:
                seen["forward"].add(name)
                print(f"  [diag] first nonfinite forward: {name}", flush=True)
        return hook

    def backward_hook(name):
        def hook(_module, grad_input, grad_output):
            if not finite(grad_output) and name not in seen["backward"]:
                seen["backward"].add(name)
                print(f"  [diag] first nonfinite backward output: {name}", flush=True)
            if not finite(grad_input) and name not in seen["backward"]:
                seen["backward"].add(name)
                print(f"  [diag] first nonfinite backward input: {name}", flush=True)
        return hook

    for index, block in enumerate(getattr(dit, "blocks", ())):
        for suffix, module in (("attn", getattr(block, "attn", None)), ("block", block)):
            if module is None:
                continue
            name = f"blocks.{index}.{suffix}"
            handles.append(module.register_forward_hook(forward_hook(name)))
            handles.append(module.register_full_backward_hook(backward_hook(name)))
    return handles


def load_dit(nf4, nf4_path, device, dtype):
    """加载 DiT, NF4 路径 (磁盘优先 > 在线量化 > bf16). 返回 (dit, nf4_source)."""
    if nf4:
        if nf4_path and Path(nf4_path).exists():
            dit = load_krea2_dit(KREA2_DIT, device="cpu", dtype=dtype, eval=False, nf4_path=Path(nf4_path))
            return dit, "disk_nf4"
        dit = load_krea2_dit(KREA2_DIT, device="cpu", dtype=dtype, eval=False, nf4=True)
        return dit, "inline_nf4"
    dit = load_krea2_dit(KREA2_DIT, device="cpu", dtype=dtype, eval=False)
    return dit, "bf16"


def ref_forward(dit, latents_5d, text_emb, fixed_noise, device, dtype):
    """训练前初始 LoRA (zero-init) forward, 捕获纯 frozen DiT velocity.

    数学偏移的基准: LoRA B 矩阵 zero-init → LoRA delta=0 → 输出=纯 frozen DiT
    forward. producer (bf16) 落盘它, consumer (NF4) 算 delta = 纯量化误差,
    不混入训练轨迹分叉. inference_mode 隔离, 不留计算图.
    """
    b = latents_5d.shape[0]
    sigma = torch.full((b,), FIXED_SIGMA, device=device, dtype=dtype)
    x_t = (1.0 - sigma) * latents_5d + sigma * fixed_noise
    if dit.blocks_to_swap and dit.blocks_to_swap > 0:
        dit.prepare_block_swap_before_forward()
    with torch.inference_mode():
        velocity = forward_for_loss(dit, x_t, text_emb, sigma)
    return velocity.detach().to("cpu", copy=True)


def train_steps(dit, network, opt, latents_5d, text_emb, fixed_noise, fixed_target,
                n_steps, device, dtype):
    """跑 n_steps 训练, 返回指标 dict."""
    b = latents_5d.shape[0]
    losses, grad_norms, step_times = [], [], []

    for step in range(n_steps):
        sigma = torch.full((b,), FIXED_SIGMA, device=device, dtype=dtype)
        x_t = (1.0 - sigma) * latents_5d + sigma * fixed_noise
        target = fixed_target

        opt.zero_grad(set_to_none=True)
        t_sync = time.time()
        if dit.blocks_to_swap and dit.blocks_to_swap > 0:
            dit.prepare_block_swap_before_forward()
        velocity = forward_for_loss(dit, x_t, text_emb, sigma)
        loss = torch.nn.functional.mse_loss(velocity, target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        step_t = time.time() - t_sync

        lv = loss.item()
        losses.append(lv)
        grad_norms.append(float(grad_norm))
        step_times.append(step_t)
        if step % 5 == 0 or step == n_steps - 1:
            print(f"  step {step:3d}: loss={lv:.4f}, grad_norm={grad_norms[-1]:.4f}, step={step_t:.2f}s")
        if not torch.isfinite(torch.tensor(lv)):
            print(f"  loss 非有限, 提前终止")
            break

    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "step_times": step_times,
    }


def main() -> int:
    nf4 = _env_bool("K2_ABL_NF4", False)
    swap = _env_int("K2_ABL_SWAP", 0)
    ckpt = _env_bool("K2_ABL_CKPT", False)
    grad_ckpt = os.environ.get("K2_ABL_GRAD_CKPT", "full").strip().lower()
    if grad_ckpt not in {"full", "every_other", "full_except", "off"}:
        raise ValueError(f"invalid K2_ABL_GRAD_CKPT={grad_ckpt!r}")
    unckpt_blocks = {
        int(value)
        for value in os.environ.get("K2_ABL_UNCKPT_BLOCKS", "").split(",")
        if value.strip()
    }
    compile_blocks = _env_bool("K2_ABL_COMPILE", False)
    attn_mode = os.environ.get("K2_ABL_ATTN_MODE", "torch")
    lora_dim = _env_int("K2_ABL_LORA_DIM", LORA_DIM)
    lora_alpha = float(os.environ.get("K2_ABL_LORA_ALPHA", LORA_ALPHA))
    img_size = _env_int("K2_ABL_IMG", 1024)
    n_steps = _env_int("K2_ABL_STEPS", 30)
    nf4_path = os.environ.get("K2_ABL_NF4_PATH", "") or None
    te_device = "cpu" if _env_bool("K2_ABL_TE_CPU", False) else "cuda"
    tag = os.environ.get("K2_ABL_TAG", "untagged")
    ref_in = os.environ.get("K2_ABL_REF", "") or None
    ref_out = os.environ.get("K2_ABL_REF_OUT", "") or None

    device = torch.device("cuda")
    dtype = torch.bfloat16
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

    config = {
        "tag": tag,
        "nf4": nf4,
        "swap": swap,
        "ckpt": ckpt,
        "img_size": img_size,
        "n_steps": n_steps,
        "nf4_source": None,
        "te_device": te_device,
        "gpu": gpu_id,
        "dtype": str(dtype),
        "lora_dim": lora_dim,
        "lora_alpha": lora_alpha,
        "lr": LR,
        "grad_ckpt": grad_ckpt,
        "unckpt_blocks": sorted(unckpt_blocks),
        "compile_blocks": compile_blocks,
        "attn_mode": attn_mode,
        "fixed_sigma": FIXED_SIGMA,
        "flow_matching_formula": "x_t=(1-σ)*latent+σ*noise; target=noise-latent; loss=mse(velocity,target); 5D latent (B,C,T=1,H,W)",
        "seed": 123,
    }
    print(f"=== NF4 消融探针 [tag={tag}] NF4={nf4} SWAP={swap} CKPT={ckpt} "
          f"{img_size}×{img_size} steps={n_steps} GPU={gpu_id} ===")

    # === A. TE + VAE encode ===
    print(f"\n--- A. TE (device={te_device}) + VAE encode ---")
    t0 = time.time()
    hiddens, txtmask, latents_4d = encode_te_vae(img_size, device, dtype, te_device)
    print(f"  encode: {time.time()-t0:.1f}s, hiddens {tuple(hiddens.shape)}, latents {tuple(latents_4d.shape)}")

    # === B. DiT + LoRA + (block_swap) + grad-ckpt ===
    print(f"\n--- B. DiT + LoRA + block_swap={swap} + grad-ckpt ---")
    t1 = time.time()
    dit, nf4_source = load_dit(nf4, nf4_path, device, dtype)
    from library.models.krea2_raw.attention_backend import prepare_krea2_attention

    prepare_krea2_attention(
        dit, attn_mode, dtype=dtype, compile_enabled=compile_blocks
    )
    config["nf4_source"] = nf4_source
    print(f"  DiT 加载 ({nf4_source}): {time.time()-t1:.1f}s")

    for p in dit.parameters():
        p.requires_grad_(False)

    network = make_network(dit, lora_dim=lora_dim, lora_alpha=lora_alpha)
    if swap > 0:
        dit.enable_block_swap(swap, device)
        dit.move_to_device_except_swap_blocks(device)
        network = network.to(device).to(dtype)
        dit.switch_block_swap_for_training()
    else:
        # swap=0: DiT 全量上 GPU (load_dit 统一用 cpu 加载, 这里显式搬).
        dit = dit.to(device)
        network = network.to(device).to(dtype)
    dit.disable_gradient_checkpointing()
    if grad_ckpt == "full":
        dit.enable_gradient_checkpointing()
    elif grad_ckpt == "every_other":
        for block_idx, block in enumerate(dit.blocks):
            if block_idx % 2 == 0:
                block.enable_gradient_checkpointing()
    elif grad_ckpt == "full_except":
        invalid = unckpt_blocks.difference(range(len(dit.blocks)))
        if invalid:
            raise ValueError(f"invalid K2_ABL_UNCKPT_BLOCKS={sorted(invalid)}")
        for block_idx, block in enumerate(dit.blocks):
            if block_idx not in unckpt_blocks:
                block.enable_gradient_checkpointing()
    if compile_blocks:
        dit.compile_blocks(
            backend=os.environ.get("K2_ABL_COMPILE_BACKEND", "inductor"),
            mode=os.environ.get("K2_ABL_COMPILE_MODE") or None,
            compile_block_scope=os.environ.get("K2_ABL_COMPILE_SCOPE", "resident"),
        )

    n_lora = len(network.unet_loras)
    n_train = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"  LoRA: {n_lora} 模块, {n_train/1e6:.2f}M 可训; DiT blocks={dit.num_blocks} swap={getattr(dit,'blocks_to_swap',0)}")
    rss_after_setup = host_rss_gb()
    avail_after_setup = meminfo_available_gb()
    print(f"  setup 后 host RSS 峰值 {rss_after_setup:.2f}GB, 系统可用 {avail_after_setup:.2f}GB")

    opt = torch.optim.AdamW(network.parameters(), lr=LR, weight_decay=0.0)
    diagnostic_handles = install_finite_diagnostics(dit)

    latents_5d = latents_4d.to(device).unsqueeze(2).requires_grad_(False)
    text_emb = Krea2TextEmbedding(hiddens.to(device), txtmask.to(device))
    b = latents_5d.shape[0]
    torch.manual_seed(123)
    fixed_noise = torch.randn_like(latents_5d)
    fixed_target = fixed_noise - latents_5d

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    power_start = gpu_power()
    rss_before_train = host_rss_gb()
    t_train_start = time.time()

    # === 训练前: 数学偏移 ref forward (初始 LoRA zero-init, 纯 frozen DiT) ===
    # producer (bf16): 落盘 ref velocity; consumer (NF4): 算 delta = 纯量化误差.
    # 此时 LoRA B 矩阵 zero-init → LoRA delta=0 → forward=纯 frozen DiT, 不混
    # 训练轨迹. 放 reset_peak 之后、训练之前, 不计入显存 peak (inference_mode).
    ref_velocity = None
    if ref_out:
        ref_velocity = ref_forward(dit, latents_5d, text_emb, fixed_noise, device, dtype)
        torch.save(ref_velocity, ref_out)
        print(f"\n--- 训练前: 落盘 bf16 reference velocity -> {ref_out} (shape {tuple(ref_velocity.shape)}) ---")
    elif ref_in and Path(ref_in).exists():
        ref_v_loaded = torch.load(ref_in, map_location="cpu")
        cur_v = ref_forward(dit, latents_5d, text_emb, fixed_noise, device, dtype)
        if cur_v.shape == ref_v_loaded.shape:
            max_delta = (cur_v.float() - ref_v_loaded.float()).abs().max().item()
            rel_l2 = (cur_v.float() - ref_v_loaded.float()).norm().item() / ref_v_loaded.float().norm().item()
            cos = torch.nn.functional.cosine_similarity(
                cur_v.float().flatten().unsqueeze(0), ref_v_loaded.float().flatten().unsqueeze(0)
            ).item()
        else:
            max_delta = rel_l2 = cos = -1.0
        ref_velocity = {"ref_role": "consumer", "ref_path": ref_in,
                        "max_delta": max_delta, "rel_l2": rel_l2, "cosine": cos,
                        "cur_shape": list(cur_v.shape), "ref_shape": list(ref_v_loaded.shape)}
        print(f"\n--- 训练前: 数学偏移 vs bf16 ref: max_delta={max_delta:.2e}, rel_l2={rel_l2:.2e}, cos={cos:.4f} ---")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # === C. 训练 (CKPT 模式: 中途存+续训) ===
    ckpt_info = {}
    if ckpt:
        half = n_steps // 2
        print(f"\n--- C1. 训练前半 {half} 步 (CKPT 模式) ---")
        m1 = train_steps(dit, network, opt, latents_5d, text_emb, fixed_noise, fixed_target,
                         half, device, dtype)

        # 存完整 checkpoint: LoRA state + optimizer state (DiT NF4 frozen 不存)
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = CKPT_DIR / f"{tag}_lora.safetensors"
        opt_path = CKPT_DIR / f"{tag}_opt.pt"
        t_sv = time.time()
        metadata = {"ss_network_spec": "lora", "ss_base_model_version": "krea2_raw",
                    "ss_model_family": "krea2_raw"}
        network.save_weights(str(ckpt_path), dtype=dtype, metadata=metadata)
        torch.save({"opt_state": opt.state_dict(), "step": half}, str(opt_path))
        save_secs = time.time() - t_sv
        ckpt_info["ckpt_lora_mb"] = ckpt_path.stat().st_size / 1e6
        ckpt_info["ckpt_opt_mb"] = opt_path.stat().st_size / 1e6
        ckpt_info["save_secs"] = save_secs
        print(f"  checkpoint 存: LoRA {ckpt_info['ckpt_lora_mb']:.1f}MB + opt {ckpt_info['ckpt_opt_mb']:.1f}MB, {save_secs:.1f}s")

        # 捕获存盘前 LoRA state + forward 基准 (round-trip 验证)
        pre_sd = {k: v.detach().to("cpu", copy=True) for k, v in network.state_dict().items()}
        sigma_f = torch.full((b,), FIXED_SIGMA, device=device, dtype=dtype)
        x_t_f = (1.0 - sigma_f) * latents_5d + sigma_f * fixed_noise
        if dit.blocks_to_swap and dit.blocks_to_swap > 0:
            dit.prepare_block_swap_before_forward()
        with torch.inference_mode():
            pre_ckpt_out = forward_for_loss(dit, x_t_f, text_emb, sigma_f).detach().to("cpu", copy=True)

        # reload LoRA + opt state 续训 (DiT 不重载, NF4 master 保留)
        network.load_weights(str(ckpt_path))
        opt_sd = torch.load(str(opt_path), map_location="cpu")
        opt.load_state_dict(opt_sd["opt_state"])
        print(f"  checkpoint reload 完成, 续训后半 {n_steps - half} 步")

        # round-trip 验证: reload 后 LoRA state + forward delta
        post_sd = network.state_dict()
        sd_deltas = []
        for k in pre_sd:
            if k in post_sd and pre_sd[k].shape == post_sd[k].to("cpu").shape:
                sd_deltas.append((pre_sd[k].float() - post_sd[k].to("cpu").float()).abs().max().item())
        ckpt_info["lora_roundtrip_max_delta"] = max(sd_deltas) if sd_deltas else -1.0
        if dit.blocks_to_swap and dit.blocks_to_swap > 0:
            dit.prepare_block_swap_before_forward()
        with torch.inference_mode():
            post_ckpt_out = forward_for_loss(dit, x_t_f, text_emb, sigma_f).to("cpu")
        ckpt_info["fwd_roundtrip_max_delta"] = (pre_ckpt_out.float() - post_ckpt_out.float()).abs().max().item()
        print(f"  round-trip: LoRA delta={ckpt_info['lora_roundtrip_max_delta']:.2e}, fwd delta={ckpt_info['fwd_roundtrip_max_delta']:.2e}")

        # 续训后半 (验证 loss 连续性: reload 后 loss 不应跳变)
        print(f"\n--- C2. 续训后半 {n_steps - half} 步 ---")
        m2 = train_steps(dit, network, opt, latents_5d, text_emb, fixed_noise, fixed_target,
                         n_steps - half, device, dtype)
        losses = m1["losses"] + m2["losses"]
        grad_norms = m1["grad_norms"] + m2["grad_norms"]
        step_times = m1["step_times"] + m2["step_times"]
        # loss 连续性: reload 前最后一步 vs reload 后第一步
        ckpt_info["loss_before_ckpt"] = m1["losses"][-1]
        ckpt_info["loss_after_ckpt"] = m2["losses"][0]
        ckpt_info["loss_jump_at_ckpt"] = abs(m2["losses"][0] - m1["losses"][-1])
        print(f"  loss 连续性: 前 {m1['losses'][-1]:.4f} -> 后 {m2['losses'][0]:.4f}, 跳变 {ckpt_info['loss_jump_at_ckpt']:.4f}")
    else:
        print(f"\n--- C. 训练 {n_steps} 步 ---")
        m = train_steps(dit, network, opt, latents_5d, text_emb, fixed_noise, fixed_target,
                        n_steps, device, dtype)
        losses = m["losses"]
        grad_norms = m["grad_norms"]
        step_times = m["step_times"]

    train_secs = time.time() - t_train_start
    peak_gpu = torch.cuda.max_memory_allocated() / 1e9
    alloc_gpu = torch.cuda.memory_allocated() / 1e9
    rss_peak = host_rss_gb()
    avail_after = meminfo_available_gb()
    power_end = gpu_power()

    # === D. 数学偏移已在训练前算好 (ref_velocity: producer→tensor, consumer→dict, None→无 ref) ===
    math_offset = {"has_ref": False}
    if ref_out and ref_velocity is not None:
        math_offset = {"has_ref": True, "ref_role": "producer", "ref_path": ref_out,
                       "ref_shape": list(ref_velocity.shape)}
    elif ref_in and isinstance(ref_velocity, dict):
        math_offset = {"has_ref": True, **ref_velocity}

    # === E. 汇总六维指标 + JSONL 落盘 ===
    finite_all = all(torch.isfinite(torch.tensor(x)).item() for x in losses)
    first5 = sum(losses[:5]) / min(5, len(losses))
    last5 = sum(losses[-5:]) / min(5, len(losses))
    loss_down = last5 < first5
    grad_nonzero = all(g > 0 for g in grad_norms)
    avg_step = sum(step_times) / len(step_times) if step_times else 0.0

    record = {
        "config": config,
        "metrics": {
            "gpu_peak_gb": round(peak_gpu, 3),
            "gpu_allocated_gb": round(alloc_gpu, 3),
            "host_rss_peak_gb": round(rss_peak, 3),
            "host_rss_before_train_gb": round(rss_before_train, 3),
            "sys_avail_before_gb": round(avail_after_setup, 3),
            "sys_avail_after_gb": round(avail_after, 3),
            "avg_step_s": round(avg_step, 3),
            "first_step_s": round(step_times[0], 3) if step_times else 0,
            "last_step_s": round(step_times[-1], 3) if step_times else 0,
            "train_secs": round(train_secs, 2),
            "loss_first5": round(first5, 6),
            "loss_last5": round(last5, 6),
            "loss_down": loss_down,
            "loss_finite": finite_all,
            "grad_nonzero": grad_nonzero,
            "grad_norm_min": round(min(grad_norms), 6) if grad_norms else 0,
            "grad_norm_max": round(max(grad_norms), 6) if grad_norms else 0,
            "gpu_power_start_w": round(power_start, 1),
            "gpu_power_end_w": round(power_end, 1),
            "n_lora_modules": n_lora,
            "n_train_params_m": round(n_train / 1e6, 3),
        },
        "math_impl": {
            "formula": config["flow_matching_formula"],
            "sigma": FIXED_SIGMA,
            "seed": 123,
            "latent_5d_shape": list(latents_5d.shape),
        },
        "math_offset": math_offset,
        "ckpt": ckpt_info,
        "losses": [round(x, 6) for x in losses],
        "step_times": [round(x, 3) for x in step_times],
    }

    for handle in diagnostic_handles:
        handle.remove()

    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(
        os.environ.get(
            "K2_ABL_OUT", str(FINDINGS_DIR / "krea2_nf4_ablation.jsonl")
        )
    )
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n=== JSONL 落盘: {jsonl_path} (tag={tag}) ===")

    # === F. 判定 ===
    # host RAM 阈值只对 NF4+swap 格生效: 该格主战场是 pinned NF4 master (~5.7GB),
    # host RAM 应远低于 bf16 路径 22.64GB master. bf16 基线格 host RSS ~26GB 是
    # bf16 权重副本的预期值 (不限), NF4-only 格 master 也在 ~6.7GB (不限, 无 swap).
    ram_constrained = nf4 and swap > 0
    ram_ok = (rss_peak < 20.0) if ram_constrained else True
    ok = finite_all and loss_down and grad_nonzero and ram_ok
    if ckpt:
        ok = ok and ckpt_info.get("fwd_roundtrip_max_delta", 1) < 1e-3
        ok = ok and ckpt_info.get("loss_jump_at_ckpt", 1) < first5 * 0.5  # 跳变不超过首5一半
    print(f"\n=== 判定 [tag={tag}]: 通过={ok} ===")
    ram_label = f"{rss_peak:.2f}GB (<20: {ram_ok})" if ram_constrained else f"{rss_peak:.2f}GB (不限)"
    print(f"  GPU peak {peak_gpu:.2f}GB | host RSS {ram_label}")
    print(f"  loss {first5:.4f}->{last5:.4f} (↓:{loss_down}) | grad≠0:{grad_nonzero} | finite:{finite_all}")
    print(f"  avg step {avg_step:.2f}s | train {train_secs:.1f}s")
    if math_offset.get("has_ref") and math_offset.get("ref_role") == "consumer":
        print(f"  数学偏移 vs bf16: max_delta={math_offset['max_delta']:.2e} rel_l2={math_offset['rel_l2']:.2e} cos={math_offset['cosine']:.4f}")
    if ckpt:
        print(f"  CKPT round-trip: LoRA delta={ckpt_info['lora_roundtrip_max_delta']:.2e} fwd delta={ckpt_info['fwd_roundtrip_max_delta']:.2e} loss跳变={ckpt_info['loss_jump_at_ckpt']:.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
