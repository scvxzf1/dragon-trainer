"""WebUI form metadata retained independently of the removed desktop GUI."""

from __future__ import annotations

FIELD_HELP: dict[str, dict[str, str]] = {
    # Architecture
    "network_dim": {
        "en": "LoRA rank (dimension of low-rank matrices). Higher = more expressive but more VRAM. Typical: 8–64.",
        "ko": "LoRA 랭크 (저랭크 행렬의 차원). 높을수록 표현력이 좋지만 VRAM 사용량 증가. 일반적: 8–64.",
    },
    "network_alpha": {
        "en": "LoRA scaling factor. Effective scale = alpha / dim. When alpha == dim, scale is 1.0. Lower alpha = more conservative updates.",
        "ko": "LoRA 스케일링 계수. 실효 스케일 = alpha / dim. alpha == dim이면 1.0. 낮을수록 보수적 업데이트.",
    },
    "network_module": {
        "en": "Python module path for the LoRA network implementation.",
        "ko": "LoRA 네트워크 구현의 Python 모듈 경로.",
    },
    "use_timestep_mask": {
        "en": "Enable T-LoRA: effective rank varies with denoising timestep via power-law schedule. Default (anima mode) is full rank at high noise, reduced at low noise; paper mode inverts that schedule.",
        "ko": "T-LoRA 활성화: 디노이징 타임스텝에 따라 유효 랭크 변동. 기본(anima)은 높은 노이즈에서 전체 랭크, 낮은 노이즈에서 축소; paper 모드는 반대.",
    },
    "timestep_mask_mode": {
        "en": "T-LoRA rank schedule direction under Anima t∈[0,1] (t=0 noise, t=1 clean). 'anima' = full rank at noise end (default). 'paper' = inverted (Soboleva et al.; min_rank at high noise).",
        "ko": "T-LoRA 랭크 스케줄 방향(Anima t∈[0,1], t=0 노이즈, t=1 클린). 'anima' = 노이즈 끝 전체 랭크(기본). 'paper' = 반전(Soboleva 등; 고노이즈에서 min_rank).",
    },
    "timestep_mask_at_inference": {
        "en": "Reapply the T-LoRA rank mask at eval/inference every denoise step (paper-faithful). Default off keeps train-only full-rank inference and static merge. When on, checkpoints refuse static merge and load as dynamic hooks.",
        "ko": "평가/추론 시 매 디노이즈 스텝에 T-LoRA 랭크 마스크 재적용(논문 충실). 기본 off는 학습 전용·전체 랭크 추론·정적 병합 유지. on이면 정적 병합 거부 및 동적 훅 로드.",
    },
    "use_ortho": {
        "en": "Enable OrthoLoRA: SVD-based orthogonal parameterization of the update matrix (linear layers only). Regularizes toward structured updates; saved as plain LoRA via thin SVD at checkpoint time.",
        "ko": "OrthoLoRA 활성화: 업데이트 행렬의 SVD 기반 직교 파라미터화 (선형 레이어 전용). 구조화된 업데이트로 정규화되며, 저장 시 thin SVD로 일반 LoRA로 변환.",
    },
    "use_lokr": {
        "en": "Enable LoKr (Low-Rank Kronecker Product): decomposes the weight delta as kron(W1, W2) and saves LyCORIS-compatible lokr_w1/lokr_w2 tensors.",
        "ko": "LoKr (Low-Rank Kronecker Product) 활성화: 가중치 델타를 kron(W1, W2)로 분해하고 LyCORIS 호환 lokr_w1/lokr_w2 텐서로 저장합니다.",
    },
    "lokr_factor": {
        "en": "LoKr Kronecker factor. W1 is factor×factor and W2 is (out/factor)×(in/factor). Default 8 works for common Anima DiT dimensions.",
        "ko": "LoKr Kronecker 인자. W1은 factor×factor, W2는 (out/factor)×(in/factor)입니다. 기본값 8은 일반적인 Anima DiT 차원에 맞습니다.",
    },
    "use_moe_style": {
        "en": "MoE expert layout: 'shared_A' (HydraLoRA — one shared lora_down + N per-expert lora_up heads), 'independent_A' (FeRA — N fully-independent down/up pairs), or false (no MoE). Produces a *_moe.safetensors sibling for router-live inference; requires cache_llm_adapter_outputs=true.",
        "ko": "MoE 전문가 레이아웃: 'shared_A' (HydraLoRA — 공유 lora_down + N개 전문가별 lora_up), 'independent_A' (FeRA — 독립적인 N쌍의 down/up), 또는 false (MoE 비활성화). 라우터-라이브 추론용 *_moe.safetensors 동반 파일 생성. cache_llm_adapter_outputs=true 필요.",
    },
    "route_per_layer": {
        "en": "If true, each layer owns its own router (per-layer routing). If false, a single network-level GlobalRouter broadcasts gate weights to every routed module.",
        "ko": "true이면 레이어별 라우터 사용 (per-layer routing). false이면 네트워크 전역 GlobalRouter 하나가 모든 라우팅 모듈에 게이트 가중치 브로드캐스트.",
    },
    "router_source": {
        "en": "Routing signal: 'sigma' (sinusoidal embedding of the denoising timestep), 'fei' (mean-pooled rank features from preceding LoRA modules), or 'pooled_text' (pooled T5 caption embedding).",
        "ko": "라우팅 신호: 'sigma' (디노이징 타임스텝의 sinusoidal 임베딩), 'fei' (선행 LoRA 모듈의 평균 풀링 랭크 특징), 또는 'pooled_text' (T5 캡션 풀링 임베딩).",
    },
    "num_experts": {
        "en": "HydraLoRA expert count. More experts = more capacity but more VRAM and slower training. Typical: 2–8.",
        "ko": "HydraLoRA 전문가 수. 많을수록 표현력 증가하지만 VRAM 사용량 증가 및 학습 속도 감소. 일반적: 2–8.",
    },
    "balance_loss_weight": {
        "en": "HydraLoRA load-balancing loss weight. Discourages router collapse onto a single expert. Typical: 0.01.",
        "ko": "HydraLoRA 부하 균형 손실 가중치. 라우터가 단일 전문가로 붕괴되는 것을 방지. 일반적: 0.01.",
    },
    "balance_loss_warmup_ratio": {
        "en": "Fraction of training steps to hold the balance loss at 0 before activating it. Lets the router specialize first, then switches the penalty on to stop further collapse of a diverged router. 0.0 disables the warmup. Typical: 0.3–0.5.",
        "ko": "밸런스 손실을 0으로 유지하는 학습 스텝 비율. 먼저 라우터가 전문화되도록 한 뒤 페널티를 활성화해 분화된 라우터의 추가 붕괴를 방지. 0.0 = 비활성화. 일반적: 0.3–0.5.",
    },
    "add_reft": {
        "en": "Enable ReFT: block-level residual-stream intervention (Wu et al. 2024). Adds R^T·(ΔW·h + b)·scale to each selected DiT block's output. Composes with any LoRA variant.",
        "ko": "ReFT 활성화: 블록 수준 잔차 스트림 개입 (Wu et al. 2024). 선택된 DiT 블록 출력에 R^T·(ΔW·h + b)·scale 추가. 모든 LoRA 변형과 함께 사용 가능.",
    },
    "reft_dim": {
        "en": "ReFT intervention rank — dimension of R and ΔW in each ReFTModule. Typical: 32–64.",
        "ko": "ReFT 개입 랭크 — 각 ReFTModule의 R 및 ΔW 차원. 일반적: 32–64.",
    },
    "reft_alpha": {
        "en": "ReFT scaling factor (effective scale = alpha / dim). Typical: same as reft_dim.",
        "ko": "ReFT 스케일링 계수 (실효 스케일 = alpha / dim). 일반적: reft_dim과 동일.",
    },
    "reft_layers": {
        "en": "Which DiT blocks receive ReFT modules. 'all', 'last_8', 'first_4', 'stride_2', or comma-separated indices like '3,7,11,15'.",
        "ko": "ReFT 모듈이 적용될 DiT 블록. 'all', 'last_8', 'first_4', 'stride_2', 또는 '3,7,11,15'와 같은 쉼표 구분 인덱스.",
    },
    "sigma_feature_dim": {
        "en": "Sinusoidal σ feature dimension fed into the σ-router bias MLP. Typical: 128.",
        "ko": "σ 라우터 바이어스 MLP에 입력되는 sinusoidal σ 특징 차원. 일반적: 128.",
    },
    "router_targets": {
        "en": "Regex over layer names — only matching Linears participate in routed adaptation (Hydra MoE leaves + σ / FEI feature concatenation share the same scope). Typical: '.*(mlp\\.layer[12])$' to confine MoE to the FFN sublayers.",
        "ko": "레이어 이름에 대한 정규식 — 일치하는 Linear만 라우팅된 적응에 참여 (Hydra MoE leaves + σ / FEI 특징 연결이 동일한 범위를 공유). 일반적: '.*(mlp\\.layer[12])$' — FFN 서브레이어로 MoE 제한.",
    },
    "per_bucket_balance_weight": {
        "en": "Extra per-σ-bucket load-balance penalty, scaled by balance_loss_weight. Encourages routing diversity within each timestep bucket. Typical: 0.3.",
        "ko": "σ 버킷별 추가 부하 균형 페널티, balance_loss_weight로 스케일. 각 타임스텝 버킷 내 라우팅 다양성 유도. 일반적: 0.3.",
    },
    "num_sigma_buckets": {
        "en": "Number of timestep buckets used for per-bucket balance accounting. Typical: 3 (low / mid / high noise).",
        "ko": "버킷별 균형 계산에 사용되는 타임스텝 버킷 수. 일반적: 3 (저/중/고 노이즈).",
    },
    "specialize_experts_by_sigma_buckets": {
        "en": "Hard-partition the expert pool into σ-bands: each timestep bucket only routes to its assigned experts. Forces specialization on top of the soft σ-router bias. Pairs with sigma_bucket_boundaries.",
        "ko": "전문가 풀을 σ-밴드로 하드 분할: 각 타임스텝 버킷은 할당된 전문가만 사용. 소프트 σ-라우터 바이어스 위에 강제 특화 부여. sigma_bucket_boundaries와 함께 사용.",
    },
    "sigma_bucket_boundaries": {
        "en": "Custom σ-bucket edges, length = num_sigma_buckets + 1, monotone 0.0 → 1.0. Defaults to uniform linspace(0, 1, N+1) when omitted. Example: [0.0, 0.5, 0.8, 1.0].",
        "ko": "사용자 지정 σ-버킷 경계, 길이 = num_sigma_buckets + 1, 0.0 → 1.0 단조 증가. 생략 시 uniform linspace(0, 1, N+1) 사용. 예: [0.0, 0.5, 0.8, 1.0].",
    },
    "network_args": {
        "en": "Extra kwargs passed to the network module. Pick a Variant to auto-fill.",
        "ko": "네트워크 모듈에 전달되는 추가 kwargs. Variant 선택으로 자동 채우기 가능.",
    },
    "min_rank": {
        "en": "Minimum active rank when T-LoRA timestep masking is enabled. At the lowest-noise timesteps, rank drops to this value.",
        "ko": "T-LoRA 타임스텝 마스킹 사용 시 최소 활성 랭크. 가장 낮은 노이즈에서 이 값까지 감소.",
    },
    "alpha_rank_scale": {
        "en": "Power-law exponent for the T-LoRA rank schedule. 1.0 is linear; >1 steeper (more capacity near high noise); <1 flatter.",
        "ko": "T-LoRA 랭크 스케줄의 멱함수 지수. 1.0은 선형, >1은 더 가파름(고노이즈 쪽 용량 증가), <1은 더 평탄.",
    },
    "network_train_unet_only": {
        "en": "Train only the DiT (U-Net). Text encoder weights are frozen. Recommended for most LoRA training.",
        "ko": "DiT(U-Net)만 학습. 텍스트 인코더 가중치는 동결. 대부분의 LoRA 학습에 권장.",
    },
    "network_weights": {
        "en": "Path to a pre-trained adapter checkpoint to warm-start from. Leave empty for plain LoRA training.",
        "ko": "워밍업으로 사용할 사전 학습 어댑터 체크포인트 경로. 일반 LoRA 학습 시에는 비워두세요.",
    },
    "dim_from_weights": {
        "en": "Read network_dim from the warm-start checkpoint instead of the form value. Set together with network_weights so rank matches the warm-start LoRA.",
        "ko": "network_dim을 폼 값 대신 워밍업 체크포인트에서 읽기. network_weights와 함께 설정하여 랭크를 워밍업 LoRA와 일치시킵니다.",
    },
    # Training
    "learning_rate": {
        "en": "Base learning rate for the optimizer. Typical: 1e-5 to 1e-4.",
        "ko": "옵티마이저 기본 학습률. 일반적: 1e-5 ~ 1e-4.",
    },
    "max_train_epochs": {
        "en": "Total training epochs. One epoch = one full pass through the dataset.",
        "ko": "총 학습 에폭 수. 1 에폭 = 데이터셋 전체를 1회 순회.",
    },
    "save_every_n_epochs": {
        "en": "Save a checkpoint every N epochs. Set equal to max_train_epochs to save only the final model.",
        "ko": "N 에폭마다 체크포인트 저장. max_train_epochs와 같게 설정하면 최종 모델만 저장.",
    },
    "checkpointing_epochs": {
        "en": "Save resumable training state every N epochs. State files are large; use a larger interval than save_every_n_epochs.",
        "ko": "N 에폭마다 학습 재개 상태 저장. 상태 파일이 크므로 save_every_n_epochs보다 큰 간격 권장.",
    },
    "gradient_accumulation_steps": {
        "en": "Accumulate gradients over N steps before updating. Effective batch size = batch_size × accumulation_steps.",
        "ko": "N 스텝 동안 그레이디언트 누적 후 업데이트. 실효 배치 크기 = batch_size × accumulation_steps.",
    },
    "use_shuffled_caption_variants": {
        "en": "Consume preprocessed caption-shuffle variants from the text-encoder cache. When the cache holds multiple variants, a random one is drawn per sample. Falls back silently to single-variant if no variants were preprocessed.",
        "ko": "전처리된 캡션 셔플 변형을 텍스트 인코더 캐시에서 사용. 캐시에 여러 변형이 있으면 샘플당 무작위 선택. 변형이 전처리되지 않았다면 단일 캡션으로 자동 대체.",
    },
    "caption_dropout_rate": {
        "en": "Probability per sample of dropping the caption (replaced with empty text embedding). Pushes the LoRA toward an unconditional bias — useful for style training where you want the look to apply regardless of prompt. Typical: 0.0–0.05 for character/concept LoRAs, 0.1–0.25 for style LoRAs (그림체 학습). Too high can blur prompt-driven diversity (pose/composition).",
        "ko": "샘플별로 캡션을 비울(빈 텍스트 임베딩으로 대체) 확률. LoRA를 무조건부(unconditional) 방향으로 학습시켜, 프롬프트와 무관하게 항상 적용되는 \"스타일\"을 학습할 때 유리. 일반적: 캐릭터/컨셉 LoRA는 0.0–0.05, 그림체 학습은 0.1–0.25. 너무 높이면 캡션이 담당하던 다양성(포즈/구도)까지 함께 약해짐.",
    },
    "optimizer_type": {
        "en": "Optimizer algorithm. CAME is available via pytorch-optimizer; non-default optimizers may need LR retuning. ProdigyPlusScheduleFree is experimental and is best started with learning_rate=1.0, lr_scheduler=constant, max_grad_norm=0.",
        "ko": "옵티마이저 알고리즘. CAME는 pytorch-optimizer를 통해 사용할 수 있으며, 기본값이 아닌 옵티마이저는 학습률 재조정이 필요할 수 있습니다. ProdigyPlusScheduleFree는 실험적이며 learning_rate=1.0, lr_scheduler=constant, max_grad_norm=0으로 시작하는 것이 좋습니다.",
    },
    "lr_scheduler": {
        "en": "Learning rate schedule. constant: fixed LR. Others: cosine, cosine_with_restarts, polynomial, lulu_loss_gated_cosine. Fine-tune lulu via lr_scheduler_args.",
        "ko": "학습률 스케줄. constant: 고정 LR. 기타: cosine, cosine_with_restarts, polynomial, lulu_loss_gated_cosine. lulu 세부 조정은 lr_scheduler_args로 설정합니다.",
    },
    "timestep_sampling": {
        "en": "How denoising timesteps are sampled during training. sigmoid: biased toward middle timesteps (recommended for flow matching).",
        "ko": "학습 중 디노이징 타임스텝 샘플링 방법. sigmoid: 중간 타임스텝 편향 (flow matching 권장).",
    },
    "discrete_flow_shift": {
        "en": "Flow-matching shift parameter controlling the noise schedule distribution. Default: 1.0.",
        "ko": "노이즈 스케줄 분포를 제어하는 flow-matching 시프트 매개변수. 기본값: 1.0.",
    },
    # Performance
    "attn_mode": {
        "en": "Attention backend. flash4: FlashAttention-4 (Linux, fastest). flash: FlashAttention-2. flex: PyTorch flex attention (cross-platform).",
        "ko": "어텐션 백엔드. flash4: FlashAttention-4 (Linux, 최속). flash: FlashAttention-2. flex: PyTorch flex attention (크로스 플랫폼).",
    },
    "v100_flash_stability": {
        "en": "V100 Flash diagnostic mode. off keeps full Flash; hybrid routes cross-attention through Torch SDPA; safe enables fail-fast finite checks. Flash remains diagnostic-only until strict validation passes.",
        "ko": "V100 Flash 진단 모드. off는 전체 Flash, hybrid는 크로스 어텐션을 Torch SDPA로 라우팅, safe는 유한성 검사를 즉시 수행합니다.",
    },
    "nf4_dir1": {
        "en": "[Experimental] Run the NF4 Linear inner GEMM in fp16 with per-layer frexp/ldexp rescaling. Only meaningful with base_compute=nf4, and only beneficial on GPUs without native bf16 tensor cores (sm70/sm75); enabling it on sm80+ raises. compute_dtype stays bfloat16.",
        "ko": "[실험적] NF4 Linear 내부 GEMM을 fp16으로 실행하고 레이어별 frexp/ldexp 스케일링을 적용합니다. base_compute=nf4에서만 의미가 있고, bf16 텐서 코어가 없는 GPU(sm70/sm75)에서만 이득이 있습니다. sm80+에서 켜면 오류가 발생합니다. compute_dtype은 bfloat16으로 유지됩니다.",
    },
    "krea2_attn_fp16": {
        "en": "[Experimental] Krea-2 attention fp16 inner path: expand GQA to MHA so fp16 takes the EFFICIENT SDPA kernel, plus frexp/ldexp rescaling of the backward dO. Only meaningful on sm<80; enabling it on sm80+ raises. compute_dtype stays bfloat16.",
        "ko": "[실험적] Krea-2 어텐션 fp16 내부 경로입니다. GQA를 MHA로 확장해 fp16이 EFFICIENT SDPA 커널을 사용하도록 하고, 역방향 dO에 frexp/ldexp 스케일링을 적용합니다. sm<80에서만 의미가 있으며 sm80+에서 켜면 오류가 발생합니다. compute_dtype은 bfloat16으로 유지됩니다.",
    },
    "debug_finite_checks": {
        "en": "Abort when attention tensors, residuals, loss, or gradients contain NaN/Inf. Diagnostic only; invalid values are never replaced or hidden.",
        "ko": "어텐션 텐서, 잔차, 손실 또는 그라디언트에 NaN/Inf가 있으면 즉시 중단합니다. 진단 전용으로 값을 숨기지 않습니다.",
    },
    "compile_dynamic_seq": {
        "en": "Mark the native token-sequence axis dynamic inside compiled DiT blocks so compatible buckets reuse one graph.",
        "ko": "컴파일된 DiT 블록의 토큰 시퀀스 축을 동적으로 표시해 호환 버킷이 하나의 그래프를 재사용합니다.",
    },
    "compile_seq_bands": {
        "en": "Split Anima dynamic sequence lengths into tight data-derived bands. Off by default; requires compile_dynamic_seq.",
        "ko": "Anima 동적 시퀀스 길이를 데이터 기반의 좁은 밴드로 나눉니다. 기본값은 비활성화이며 compile_dynamic_seq가 필요합니다.",
    },
    "gradient_checkpointing": {
        "en": "Recompute activations during backward pass instead of storing them. Trades compute for VRAM. Essential for low-VRAM setups.",
        "ko": "역전파 시 활성값을 저장 대신 재계산. 연산으로 VRAM 절약. 저사양 필수.",
    },
    "unsloth_offload_checkpointing": {
        "en": "Offload gradient checkpoints to CPU RAM. Further VRAM reduction at cost of speed. Requires gradient_checkpointing=true.",
        "ko": "그레이디언트 체크포인트를 CPU RAM으로 오프로드. 속도 감소 대신 VRAM 추가 절약. gradient_checkpointing=true 필요.",
    },
    "max_token_length": {
        "en": "Caption token cap for MOSS-Image. MOSS stores real-length text states, so this value is part of the text cache identity: changing it requires rebuilding the text cache.",
        "ko": "MOSS-Image의 캡션 토큰 상한입니다. MOSS는 실제 길이 상태를 저장하므로 이 값이 텍스트 캐시 정체성에 포함되며, 변경 시 텍스트 캐시를 다시 만들어야 합니다.",
    },
    "blocks_to_swap": {
        "en": "Number of DiT blocks to swap between GPU and CPU. 0: all on GPU. Higher values = more CPU offloading for low VRAM.",
        "ko": "GPU와 CPU 간 스왑할 DiT 블록 수. 0: 전부 GPU. 높을수록 더 많이 CPU로 오프로드.",
    },
    "block_swap_transfer_dtype": {
        "en": "CPU-master and transfer precision for frozen block-swap weights. MOSS int8 is experimental and may shift per-step loss by roughly 0.3%-2.0%.",
        "ko": "블록 스왑 고정 가중치의 CPU 마스터/전송 정밀도입니다. MOSS int8은 실험적이며 스텝 손실이 약 0.3%-2.0% 이동할 수 있습니다.",
    },
    "block_swap_int8_restore_mode": {
        "en": "INT8 restore implementation. fused is the default one-kernel direct-to-slot path; copy, reuse_storage, and direct_bind remain diagnostic alternatives.",
        "ko": "INT8 복원 구현입니다. fused는 기본 단일 커널 슬롯 직접 쓰기 경로이며 나머지는 진단용 대안입니다.",
    },
    "block_swap_int8_staging_ring": {
        "en": "Fused INT8 staging-buffer count: 0 allocates per swapped slot for speed; 1 is minimal VRAM; 2 is the default two-buffer ring.",
        "ko": "fused INT8 staging 버퍼 수입니다. 0은 속도를 위한 슬롯별 버퍼, 1은 최소 VRAM, 2는 기본 이중 버퍼 링입니다.",
    },
    "block_swap_int8_dual_stream": {
        "en": "Use a DMA stream plus priority dequant stream for stable step times. The dual path owns two buffers and ignores staging_ring.",
        "ko": "안정적인 스텝 시간을 위해 DMA 스트림과 고우선순위 역양자화 스트림을 사용합니다. 이 경로는 자체 두 버퍼를 사용하며 staging_ring을 무시합니다.",
    },
    "block_swap_int8_restore_chunk_rows": {
        "en": "Rows per reuse_storage restore chunk. 0 restores whole tensors; smaller values lower temporary peaks but add kernels.",
        "ko": "reuse_storage 복원 청크당 행 수입니다. 0은 전체 텐서를 복원하며, 작은 값은 피크를 낮추지만 커널을 추가합니다.",
    },
    "pipeline_parallel": {
        "en": "Enable the experimental two-GPU pipeline planner for the selected model family. Launch remains blocked until the 1F1B trainer schedule is connected.",
        "ko": "선택한 모델 패밀리의 실험적 2-GPU 파이프라인 계획을 활성화합니다. 1F1B 학습 스케줄 연결 전까지 시작은 차단됩니다.",
    },
    "pipeline_parallel_stages": {
        "en": "Pipeline stage count. The current topology requires exactly two stages and two worker processes.",
        "ko": "파이프라인 스테이지 수입니다. 현재 토폴로지는 정확히 2개 스테이지와 2개 워커 프로세스가 필요합니다.",
    },
    "pipeline_parallel_microbatches": {
        "en": "Microbatches per pipeline step. Four to eight reduces the two-stage bubble while each microbatch remains batch size one.",
        "ko": "파이프라인 스텝당 마이크로배치 수입니다. 각 마이크로배치가 배치 1일 때 4~8개가 2단계 버블을 줄입니다.",
    },
    "pipeline_parallel_schedule": {
        "en": "Pipeline execution schedule. The first production target accepts only 1F1B.",
        "ko": "파이프라인 실행 스케줄입니다. 첫 구현 목표는 1F1B만 허용합니다.",
    },
    "pipeline_parallel_split": {
        "en": "Model-family block partition policy. Balanced uses contiguous ranges; Krea-2 keeps its initial 13/15 heuristic while Anima and Z-Image start evenly split.",
        "ko": "모델 패밀리 블록 분할 정책입니다. balanced는 연속 범위를 사용하며 Krea-2는 13/15, Anima와 Z-Image는 균등 분할로 시작합니다.",
    },
    "torch_compile": {
        "en": "Enable torch.compile for the forward pass. Faster training after initial compilation. Best with static_token_count=true.",
        "ko": "torch.compile 활성화. 초기 컴파일 후 학습 속도 향상. static_token_count=true와 함께 사용 권장.",
    },
    "compile_mode": {
        "en": "'blocks': compile each DiT block individually (default). 'full': compile entire model as one graph for cross-block memory optimization. Full mode is incompatible with gradient checkpointing and block swap.",
        "ko": "'blocks': 각 DiT 블록을 개별 컴파일 (기본값). 'full': 전체 모델을 하나의 그래프로 컴파일하여 블록 간 메모리 최적화. full 모드는 gradient checkpointing 및 block swap과 호환 불가.",
    },
    "trim_crossattn_kv": {
        "en": "Remove zero-padding from cross-attention KV for efficiency. Flash4 applies LSE correction to maintain correct softmax.",
        "ko": "효율을 위해 크로스 어텐션 KV에서 제로 패딩 제거. Flash4는 정확한 softmax를 위해 LSE 보정 적용.",
    },
    "cache_llm_adapter_outputs": {
        "en": "Cache the LLM adapter layer outputs to disk. Avoids recomputing text encoder projections each epoch.",
        "ko": "LLM 어댑터 레이어 출력을 디스크에 캐싱. 매 에폭 텍스트 인코더 투영 재계산 회피.",
    },
    "masked_loss": {
        "en": "Apply loss only to non-masked regions (e.g., exclude text bubbles). Requires mask files in masks/ directory.",
        "ko": "마스크되지 않은 영역에만 손실 적용 (예: 말풍선 제외). masks/ 디렉토리에 마스크 파일 필요.",
    },
    "mixed_precision": {
        "en": "Mixed precision mode. bf16: recommended for modern GPUs. fp16: for older GPUs without bf16 support.",
        "ko": "혼합 정밀도 모드. bf16: 최신 GPU 권장. fp16: bf16 미지원 구형 GPU용.",
    },
    "static_token_count": {
        "en": "Fixed 4096 token count for all batches. Gives torch.compile a single static shape — no recompilation across aspect ratios.",
        "ko": "모든 배치에 4096 토큰 고정. torch.compile에 단일 정적 셰이프 제공 — 화면비별 재컴파일 없음.",
    },
    "vae_chunk_size": {
        "en": "VAE decoding chunk size. Larger = faster but more VRAM. 64 is a good balance.",
        "ko": "VAE 디코딩 청크 크기. 클수록 빠르지만 VRAM 더 사용. 64가 적절.",
    },
    "vae_disable_cache": {
        "en": "Disable VAE's internal KV cache. Reduces VRAM during VAE encoding/decoding.",
        "ko": "VAE 내부 KV 캐시 비활성화. VAE 인코딩/디코딩 시 VRAM 감소.",
    },
    "cache_latents": {
        "en": "Cache VAE-encoded latents in memory. Avoids re-encoding images every epoch.",
        "ko": "VAE 인코딩된 레이턴트를 메모리에 캐싱. 매 에폭 이미지 재인코딩 회피.",
    },
    "cache_latents_to_disk": {
        "en": "Save cached latents to disk instead of RAM. Frees system memory at cost of disk I/O.",
        "ko": "캐시된 레이턴트를 RAM 대신 디스크에 저장. 디스크 I/O 대신 시스템 메모리 절약.",
    },
    "cache_text_encoder_outputs": {
        "en": "Cache text encoder outputs. Essential for lazy loading: encode → cache → free encoder → load DiT.",
        "ko": "텍스트 인코더 출력 캐싱. 지연 로딩 필수: 인코딩 → 캐시 → 인코더 해제 → DiT 로드.",
    },
    "cache_text_encoder_outputs_to_disk": {
        "en": "Save cached text encoder outputs to disk. Required for the lazy loading sequence to free VRAM before loading DiT.",
        "ko": "캐시된 텍스트 인코더 출력을 디스크에 저장. DiT 로드 전 VRAM 해제를 위한 지연 로딩 필수.",
    },
    "skip_cache_check": {
        "en": "Skip validation of cached files on startup. Faster startup when caches are known to be valid.",
        "ko": "시작 시 캐시 파일 검증 건너뛰기. 캐시가 유효함을 알 때 빠른 시작.",
    },
    "use_cmmd": {
        "en": "Use CMMD (PE-Core MMD²) as the validation signal. Off by default in the WebUI — CMMD adds the PE encoder + a sampling pass per held-out item, which costs extra VRAM and time. Off → falls back to the cheaper per-σ FM-MSE val pass (uninformative on Anima but free).",
        "ko": "CMMD (PE-Core MMD²)를 검증 신호로 사용. WebUI 기본값은 OFF — CMMD는 검증 항목마다 PE 인코더와 샘플링 패스를 추가해 VRAM과 시간 비용이 큼. OFF면 더 저렴한 σ별 FM-MSE 검증으로 대체 (Anima에서 유의미한 신호는 아니지만 무료).",
    },
    # Paths
    "pretrained_model_name_or_path": {
        "en": "Path to the base DiT model weights (.safetensors).",
        "ko": "기본 DiT 모델 가중치 경로 (.safetensors).",
    },
    "qwen3": {
        "en": "Path to the Qwen3 text encoder weights for text-to-image conditioning.",
        "ko": "텍스트-투-이미지 컨디셔닝용 Qwen3 텍스트 인코더 가중치 경로.",
    },
    "vae": {
        "en": "Path to the VAE model for image encoding/decoding.",
        "ko": "이미지 인코딩/디코딩용 VAE 모델 경로.",
    },
    "output_dir": {
        "en": "Directory for saving trained LoRA checkpoints.",
        "ko": "학습된 LoRA 체크포인트 저장 디렉토리.",
    },
    "output_name": {
        "en": "Base filename for saved checkpoints (epoch number is appended automatically).",
        "ko": "저장되는 체크포인트의 기본 파일명 (에폭 번호 자동 추가).",
    },
    "save_model_as": {
        "en": "Checkpoint format. safetensors: recommended (fast, safe).",
        "ko": "체크포인트 형식. safetensors: 권장 (빠르고 안전).",
    },
    "source_image_dir": {
        "en": (
            "Where raw images and .txt captions live. The Preprocess button feeds "
            "this to resize_images.py (writes resized PNGs) and "
            "cache_text_embeddings.py (caches captions). Override per preset/method "
            "if you keep multiple datasets side by side."
        ),
        "ko": (
            "원본 이미지와 .txt 캡션이 있는 디렉토리. 전처리 버튼이 이 경로를 "
            "resize_images.py(리사이즈된 PNG 저장)와 cache_text_embeddings.py"
            "(캡션 캐시)에 전달합니다. 여러 데이터셋을 병행할 때 프리셋/메소드별로 "
            "오버라이드하세요."
        ),
    },
    "resized_image_dir": {
        "en": (
            "Where preprocess writes VAE-aligned PNGs. Also resolved into the dataset "
            "subset's image_dir at training time (via {resized_image_dir} template "
            "in base.toml), so editing this propagates to both preprocess and training."
        ),
        "ko": (
            "전처리가 VAE에 맞춰 리사이즈한 PNG를 저장하는 디렉토리. 학습 시 "
            "데이터셋 서브셋의 image_dir로도 사용됩니다(base.toml의 "
            "{resized_image_dir} 템플릿 치환). 이 값을 바꾸면 전처리와 학습 양쪽에 "
            "반영됩니다."
        ),
    },
    "lora_cache_dir": {
        "en": (
            "Where preprocess writes VAE latent (.npz) and text-encoder "
            "(_anima_te.safetensors) caches. Also resolved into the dataset subset's "
            "cache_dir at training time."
        ),
        "ko": (
            "전처리가 VAE 잠재 변수(.npz)와 텍스트 인코더 출력"
            "(_anima_te.safetensors) 캐시를 저장하는 디렉토리. 학습 시 데이터셋 "
            "서브셋의 cache_dir로도 사용됩니다."
        ),
    },
}

FORM_GROUPS = {
    "Architecture": {
        "network_dim",
        "network_alpha",
        "network_module",
        "max_token_length",
        "network_args",
        "use_ortho",
        "use_timestep_mask",
        "use_moe_style",
        "route_per_layer",
        "router_source",
        "add_reft",
        "min_rank",
        "alpha_rank_scale",
        "timestep_mask_mode",
        "timestep_mask_at_inference",
        "num_experts",
        "balance_loss_weight",
        "balance_loss_warmup_ratio",
        "reft_dim",
        "reft_alpha",
        "reft_layers",
        "sigma_feature_dim",
        "router_targets",
        "per_bucket_balance_weight",
        "num_sigma_buckets",
        "specialize_experts_by_sigma_buckets",
        "sigma_bucket_boundaries",
        "network_train_unet_only",
    },
    "Training": {
        "learning_rate",
        "max_train_epochs",
        "save_every_n_epochs",
        "checkpointing_epochs",
        "gradient_accumulation_steps",
        "use_shuffled_caption_variants",
        "caption_dropout_rate",
        "optimizer_type",
        "lr_scheduler",
        "timestep_sampling",
        "discrete_flow_shift",
        "use_valid",
        "validation_split_num",
    },
    "Performance": {
        "attn_mode",
        "v100_flash_stability",
        "debug_finite_checks",
        "compile_dynamic_seq",
        "compile_seq_bands",
        "gradient_checkpointing",
        "unsloth_offload_checkpointing",
        "blocks_to_swap",
        "block_swap_transfer_dtype",
        "block_swap_int8_restore_mode",
        "block_swap_int8_staging_ring",
        "block_swap_int8_dual_stream",
        "block_swap_int8_restore_chunk_rows",
        "pipeline_parallel",
        "pipeline_parallel_stages",
        "pipeline_parallel_microbatches",
        "pipeline_parallel_schedule",
        "pipeline_parallel_split",
        "base_compute",
        "convrot_group_size",
        "convrot_scope",
        "convrot_hadamard",
        "convrot_min_in_features",
        "convrot_largest_in_features_only",
        "convrot_large_layer_mode",
        "convrot_large_min_in_features",
        "nf4_dir1",
        "krea2_attn_fp16",
        "torch_compile",
        "cache_llm_adapter_outputs",
        "masked_loss",
        "mixed_precision",
        "vae_chunk_size",
        "vae_disable_cache",
        "use_vae_cache",
        "use_text_cache",
        "skip_cache_check",
        "layer_start",
        "use_cmmd",
    },
    "Paths": {
        "pretrained_model_name_or_path",
        "qwen3",
        "vae",
        "output_dir",
        "output_name",
        "save_model_as",
        "source_image_dir",
        "resized_image_dir",
        "lora_cache_dir",
        "path_pattern",
        "drop_lowres_images",
        "min_pixels",
    },
}

BASIC_FIELDS = {
    "learning_rate",
    "max_train_epochs",
    "save_every_n_epochs",
    "network_dim",
    "network_alpha",
    "network_weights",
    "num_experts",
    "output_name",
    "use_shuffled_caption_variants",
    "caption_dropout_rate",
    "gradient_checkpointing",
    "blocks_to_swap",
    "block_swap_transfer_dtype",
    "block_swap_int8_restore_mode",
    "block_swap_int8_staging_ring",
    "block_swap_int8_dual_stream",
    "block_swap_int8_restore_chunk_rows",
    "source_image_dir",
    "lora_cache_dir",
    "output_dir",
    "path_pattern",
    "drop_lowres_images",
    "min_pixels",
    "use_valid",
    "validation_split_num",
}

__all__ = ["FIELD_HELP", "FORM_GROUPS", "BASIC_FIELDS"]
