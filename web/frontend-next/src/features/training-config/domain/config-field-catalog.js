export const CONFIG_STAGE_META = Object.freeze([
    {
        id: 'input',
        label: '输入准备',
        shortLabel: '输入',
        color: 'amber',
        accent: '#d99114',
        clusters: [
            ['models', '模型路径'],
            ['dataset', '数据集'],
            ['captions', 'Caption 与遮罩'],
            ['filters', '输入筛选'],
            ['cache', '缓存策略'],
        ],
    },
    {
        id: 'method',
        label: '方法配置',
        shortLabel: '方法',
        color: 'indigo',
        accent: '#6366b5',
        clusters: [
            ['contract', '方法契约'],
            ['warm-start', '热启动'],
            ['capacity', '容量与目标层'],
            ['adapter', 'Adapter 分支'],
            ['orthogonal', '正交与时间步'],
            ['reft', 'ReFT'],
            ['routing', 'MoE 与路由'],
            ['fei', 'FEI 特征'],
            ['chimera', 'ChimeraHydra'],
            ['ip-adapter', 'IP-Adapter'],
            ['easycontrol', 'EasyControl'],
            ['soft-tokens', 'Soft Tokens'],
            ['raw', '原始参数'],
            ['spd-audit', 'SPD 审计'],
        ],
    },
    {
        id: 'training',
        label: '训练计划',
        shortLabel: '训练',
        color: 'blue',
        accent: '#3182ce',
        clusters: [
            ['identity', '训练身份'],
            ['volume', '训练量'],
            ['optimizer', '优化器与学习率'],
            ['loss', '时间步与损失'],
            ['regularization', '正则化'],
            ['preview', '训练预览'],
            ['output', '保存与续训点'],
            ['observability', '日志与验证'],
        ],
    },
    {
        id: 'resources',
        label: '资源与预检',
        shortLabel: '资源',
        color: 'green',
        accent: '#3a9b58',
        clusters: [
            ['topology', '设备拓扑'],
            ['compute', '计算路径'],
            ['activation', '激活显存'],
            ['residency', '模型驻留'],
            ['compile', 'Compile'],
            ['preprocess', '预处理执行资源'],
            ['diagnostics', '诊断与兼容'],
            ['unclassified', '待分类审计'],
        ],
    },
].map((stage) => Object.freeze({
    ...stage,
    clusters: Object.freeze(stage.clusters.map(([id, label]) => Object.freeze({ id, label }))),
})));

const FIELD_GROUPS = Object.freeze([
    group('input', 'models', [
        'model_family',
        'qwen_image_2_1_task',
        'pretrained_model_name_or_path',
        'qwen3',
        'vocab_pack_path',
        'moss_te_quantization',
        'vae',
    ]),
    group('input', 'dataset', ['dataset_config']),
    group('input', 'captions', [
        'use_shuffled_caption_variants',
        'caption_dropout_rate',
        'masked_loss',
    ]),
    group('input', 'filters', [
        'path_pattern',
        'drop_lowres_images',
        'min_pixels',
    ]),
    group('input', 'cache', [
        'use_vae_cache',
        'use_text_cache',
        'cache_llm_adapter_outputs',
        'ip_features_cache_to_disk',
        'skip_cache_check',
        'reuse_dataset_cache_copy',
        'reuse_vae_latents',
        'reuse_text_encoder_cache',
        'cache_fingerprint_mode',
        'force_rebuild_preprocess_cache',
    ]),

    group('method', 'contract', [
        'network_module',
        'lora_adapter_kind',
        'max_token_length',
    ]),
    group('method', 'warm-start', [
        'network_weights',
        'dim_from_weights',
    ]),
    group('method', 'capacity', [
        'network_dim',
        'network_alpha',
        'network_train_unet_only',
        'train_adaln',
    ]),
    group('method', 'adapter', [
        'dora_wd',
        'lokr_factor',
        'lokr_use_einsum',
        'lokr_decompose_w2',
        'lokr_full_factor',
        'lokr_allow_legacy_dim',
        'lokr_factor_group_size',
        'lokr_project_chunk_bytes',
        'lokr_grouped_delta_backend',
        'lokr_grouped_delta_backward_backend',
        'vera_projection_prng_key',
        'vera_d_initial',
        'vera_save_projection',
    ]),
    group('method', 'orthogonal', [
        'use_ortho',
        'min_rank',
        'alpha_rank_scale',
        'channel_scaling_alpha',
        'layer_start',
        'use_timestep_mask',
        'timestep_mask_mode',
        'timestep_mask_at_inference',
    ]),
    group('method', 'reft', [
        'add_reft',
        'reft_dim',
        'reft_alpha',
        'reft_layers',
    ]),
    group('method', 'routing', [
        'use_moe_style',
        'route_per_layer',
        'router_source',
        'num_experts',
        'router_targets',
        'network_router_lr_scale',
        'balance_loss_weight',
        'balance_loss_warmup_ratio',
        'sigma_feature_dim',
        'num_sigma_buckets',
        'sigma_bucket_boundaries',
        'specialize_experts_by_sigma_buckets',
        'per_bucket_balance_weight',
        'router_hidden_dim',
        'router_tau',
    ]),
    group('method', 'fei', [
        'fei_feature_dim',
        'fei_sigma_low_div',
        'fera_num_bands',
        'fera_fecl_weight',
    ]),
    group('method', 'chimera', [
        'use_chimera_hydra',
        'num_experts_content',
        'num_experts_freq',
        'content_router_source',
        'content_router_init_std',
        'content_router_layer_norm',
        'freq_router_init_std',
        'freq_router_layer_norm',
        'network_content_router_lr_scale',
        'network_freq_router_lr_scale',
        'balance_w_content',
        'balance_w_freq',
    ]),
    group('method', 'ip-adapter', [
        'use_ip_adapter',
        'ip_image_drop_p',
        'encoder',
        'encoder_dim',
        'resampler_layers',
        'resampler_heads',
        'ip_scale',
        'gate_lr',
        'pe_lora_enabled',
        'pe_lora_rank',
        'pe_lora_alpha',
        'pe_lora_layer_from',
        'ip_pair_mode',
        'ip_pair_prob',
        'ip_pair_min_level',
        'ip_pair_caption_strip_p',
        'validation_baselines',
    ]),
    group('method', 'easycontrol', [
        'use_easycontrol',
        'easycontrol_drop_p',
        'easycontrol_cond_noise_max',
        'b_cond_init',
        'cond_scale',
        'apply_ffn_lora',
        'cond_token_count',
    ]),
    group('method', 'soft-tokens', [
        'n_layers',
        'n_t_buckets',
        'init_std',
        'splice_position',
        'dual_bank',
        'contrastive_weight',
        'contrastive_k',
        'contrastive_every_n',
        'contrastive_negative_mode',
        'contrastive_objective',
        'contrastive_jaccard_alpha',
        'contrastive_tau',
        'contrastive_warmup_ratio',
        'softrank_softness',
        'softrank_method',
    ]),
    group('method', 'raw', ['network_args']),
    group('method', 'spd-audit', [
        'dit_path',
        'data_dir',
        'iterations',
    ]),

    group('training', 'identity', ['output_name']),
    group('training', 'volume', [
        'max_train_epochs',
        'max_train_steps',
        'train_batch_size',
        'gradient_accumulation_steps',
        'sample_ratio',
    ]),
    group('training', 'optimizer', [
        'optimizer_type',
        'learning_rate',
        'optimizer_args',
        'weight_decay',
        'lr_scheduler',
        'lr_warmup_steps',
    ]),
    group('training', 'loss', [
        'timestep_sampling',
        'discrete_flow_shift',
        'weighting_scheme',
        'min_snr_gamma',
        'p2_gamma',
        'p2_k',
        'sigmoid_scale',
        'sigmoid_bias',
        'velocity_direction_loss_weight',
    ]),
    group('training', 'regularization', [
        'prior_loss_weight',
        'prior_preservation_weight',
        'blank_prompt_preservation',
        'diff_output_preservation_trigger',
        'diff_output_preservation_class',
        'inverted_mask_prior_weight',
    ]),
    group('training', 'preview', [
        'sample_prompts',
        'sample_every_n_epochs',
        'sample_every_n_steps',
        'sample_at_first',
        'sample_sampler',
        'qwen_sample_swap_mode',
        'qwen_sample_blocks_to_swap',
        'sample_vae_decode',
        'seed',
    ]),
    group('training', 'output', [
        'save_model_as',
        'save_precision',
        'save_every_n_epochs',
        'save_last_n_epochs',
        'checkpointing_epochs',
        'checkpointing_last_n_epochs',
    ]),
    group('training', 'observability', [
        'log_every_n_steps',
        'logging_dir',
        'log_with',
        'use_cmmd',
        'ip_diagnostics_epochs',
    ]),

    group('resources', 'topology', [
        'pipeline_parallel',
        'pipeline_parallel_stages',
        'pipeline_parallel_microbatches',
        'pipeline_parallel_schedule',
        'pipeline_parallel_split',
    ]),
    group('resources', 'compute', [
        'precision_preference',
        'base_compute',
        'attn_mode',
        'v100_flash_stability',
        'convrot_group_size',
        'convrot_scope',
        'convrot_hadamard',
        'convrot_min_in_features',
        'convrot_largest_in_features_only',
        'convrot_large_layer_mode',
        'convrot_large_min_in_features',
        'nf4_dir1',
        'krea2_attn_fp16',
    ]),
    group('resources', 'activation', [
        'gradient_checkpointing',
        'cpu_offload_checkpointing',
        'unsloth_offload_checkpointing',
        'selective_checkpoint',
        'selective_checkpoint_blocks',
    ]),
    group('resources', 'residency', [
        'auto_block_swap',
        'auto_block_swap_mode',
        'auto_block_swap_interval',
        'auto_block_swap_vram_reserve_percent',
        'auto_block_swap_preference',
        'auto_block_swap_max_trials',
        'auto_block_swap_timeout',
        'auto_block_swap_swap_io_limit_mb',
        'blocks_to_swap',
        'block_swap_transfer_dtype',
        'block_swap_restore_mode',
        'block_swap_int8_restore_mode',
        'block_swap_int8_staging_ring',
        'block_swap_int8_dual_stream',
        'block_swap_int8_restore_chunk_rows',
        'disable_block_swap_for_eval',
    ]),
    group('resources', 'compile', [
        'torch_compile',
        'compile_block_scope',
        'compile_inductor_mode',
        'compile_dynamic_seq',
        'compile_seq_bands',
        'qwen_nf4_compile_experiment',
        'activation_memory_budget',
        'use_custom_down_autograd',
        'debug_finite_checks',
    ]),
    group('resources', 'preprocess', [
        'qwen_text_encoder_cache_policy',
        'preprocess_vae_cache_batch_size',
        'preprocess_text_cache_batch_size',
        'preprocess_memory_profile',
        'preprocess_precision_preference',
        'max_data_loader_n_workers',
        'dataloader_pin_memory',
        'persistent_data_loader_workers',
        'vae_chunk_size',
        'vae_disable_cache',
    ]),
    group('resources', 'diagnostics', [
        'block_swap_profile_jsonl',
        'memory_probe_jsonl',
        'memory_probe_max_steps',
        'peak_probe_jsonl',
        'peak_probe_max_steps',
        'peak_probe_level',
        'gradient_flow_probe_jsonl',
        'gradient_flow_probe_every_n_steps',
        'gradient_flow_probe_dense_steps',
    ]),
]);

const CONTEXT_FIELD_KEYS = new Set(['model_family', 'dataset_config', 'lora_adapter_kind']);
const AUDIT_ONLY_FIELD_KEYS = new Set(['dit_path', 'data_dir', 'iterations']);
const METHOD_CONDITIONAL_CLUSTERS = new Set([
    'capacity', 'adapter', 'orthogonal', 'reft', 'routing', 'fei', 'chimera',
    'ip-adapter', 'easycontrol', 'soft-tokens', 'raw', 'spd-audit',
]);

const FIELD_RELATIONS = Object.freeze({
    dim_from_weights: relation(['network_weights'], ['network_weights']),
    network_dim: relation(['dim_from_weights'], ['dim_from_weights']),
    network_alpha: relation(['dim_from_weights'], ['dim_from_weights']),
    dora_wd: relation(['lora_adapter_kind'], ['lora_adapter_kind']),
    lokr_factor: relation(['lora_adapter_kind'], ['lora_adapter_kind']),
    lokr_use_einsum: relation(['lora_adapter_kind'], ['lokr_factor']),
    lokr_decompose_w2: relation(['lora_adapter_kind'], ['lokr_factor']),
    lokr_full_factor: relation(['lora_adapter_kind'], ['lokr_factor']),
    lokr_allow_legacy_dim: relation(['lora_adapter_kind'], ['lokr_factor']),
    lokr_factor_group_size: relation(['lora_adapter_kind'], ['lokr_factor']),
    lokr_project_chunk_bytes: relation(['lora_adapter_kind'], ['lokr_factor']),
    lokr_grouped_delta_backend: relation(['lora_adapter_kind'], ['lokr_factor']),
    lokr_grouped_delta_backward_backend: relation(['lora_adapter_kind'], ['lokr_grouped_delta_backend']),
    vera_projection_prng_key: relation(['lora_adapter_kind'], ['lora_adapter_kind']),
    vera_d_initial: relation(['lora_adapter_kind'], ['lora_adapter_kind']),
    vera_save_projection: relation(['lora_adapter_kind'], ['lora_adapter_kind']),
    min_rank: relation(['use_ortho'], ['use_ortho']),
    alpha_rank_scale: relation(['use_ortho'], ['use_ortho']),
    channel_scaling_alpha: relation(['use_ortho'], ['use_ortho']),
    layer_start: relation(['use_ortho'], ['use_ortho']),
    timestep_mask_mode: relation(['use_timestep_mask'], ['use_timestep_mask']),
    timestep_mask_at_inference: relation(['use_timestep_mask'], ['use_timestep_mask']),
    reft_dim: relation(['add_reft'], ['add_reft']),
    reft_alpha: relation(['add_reft'], ['add_reft']),
    reft_layers: relation(['add_reft'], ['add_reft']),
    route_per_layer: relation(['use_moe_style'], ['use_moe_style']),
    router_source: relation(['use_moe_style'], ['use_moe_style']),
    num_experts: relation(['use_moe_style'], ['use_moe_style']),
    fei_feature_dim: relation(['router_source'], ['router_source']),
    fei_sigma_low_div: relation(['router_source'], ['router_source']),
    fera_num_bands: relation(['router_source'], ['router_source']),
    fera_fecl_weight: relation(['router_source'], ['router_source']),
    num_experts_content: relation(['use_chimera_hydra'], ['use_chimera_hydra']),
    num_experts_freq: relation(['use_chimera_hydra'], ['use_chimera_hydra']),
    encoder: relation(['use_ip_adapter'], ['use_ip_adapter']),
    encoder_dim: relation(['use_ip_adapter'], ['use_ip_adapter']),
    resampler_layers: relation(['use_ip_adapter'], ['use_ip_adapter']),
    resampler_heads: relation(['use_ip_adapter'], ['use_ip_adapter']),
    ip_scale: relation(['use_ip_adapter'], ['use_ip_adapter']),
    gate_lr: relation(['use_ip_adapter'], ['use_ip_adapter']),
    pe_lora_enabled: relation(['use_ip_adapter'], ['use_ip_adapter']),
    pe_lora_rank: relation(['pe_lora_enabled'], ['pe_lora_enabled']),
    pe_lora_alpha: relation(['pe_lora_enabled'], ['pe_lora_enabled']),
    pe_lora_layer_from: relation(['pe_lora_enabled'], ['pe_lora_enabled']),
    use_easycontrol: relation(['$method']),
    easycontrol_drop_p: relation(['use_easycontrol'], ['use_easycontrol']),
    easycontrol_cond_noise_max: relation(['use_easycontrol'], ['use_easycontrol']),
    b_cond_init: relation(['use_easycontrol'], ['use_easycontrol']),
    cond_scale: relation(['use_easycontrol'], ['use_easycontrol']),
    apply_ffn_lora: relation(['use_easycontrol'], ['use_easycontrol']),
    cond_token_count: relation(['use_easycontrol'], ['use_easycontrol']),
    max_train_steps: relation(['max_train_epochs'], ['max_train_epochs']),
    discrete_flow_shift: relation(['timestep_sampling'], ['timestep_sampling']),
    weighting_scheme: relation(['timestep_sampling'], ['timestep_sampling']),
    min_snr_gamma: relation(['weighting_scheme'], ['weighting_scheme']),
    p2_gamma: relation(['weighting_scheme'], ['weighting_scheme']),
    p2_k: relation(['weighting_scheme'], ['p2_gamma']),
    sigmoid_scale: relation(['weighting_scheme'], ['weighting_scheme']),
    sigmoid_bias: relation(['weighting_scheme'], ['sigmoid_scale']),
    pipeline_parallel_stages: relation(['pipeline_parallel'], ['pipeline_parallel']),
    pipeline_parallel_microbatches: relation(['pipeline_parallel'], ['pipeline_parallel']),
    pipeline_parallel_schedule: relation(['pipeline_parallel'], ['pipeline_parallel']),
    pipeline_parallel_split: relation(['pipeline_parallel'], ['pipeline_parallel']),
    convrot_group_size: relation(['base_compute'], ['base_compute']),
    convrot_scope: relation(['base_compute'], ['base_compute']),
    convrot_hadamard: relation(['base_compute'], ['base_compute']),
    convrot_min_in_features: relation(['base_compute'], ['base_compute']),
    convrot_largest_in_features_only: relation(['base_compute'], ['base_compute']),
    convrot_large_layer_mode: relation(['base_compute'], ['base_compute']),
    convrot_large_min_in_features: relation(['base_compute'], ['base_compute']),
    selective_checkpoint_blocks: relation(['selective_checkpoint'], ['selective_checkpoint']),
    auto_block_swap_max_trials: relation(['auto_block_swap'], ['auto_block_swap']),
    auto_block_swap_mode: relation(['auto_block_swap', 'model_family'], ['auto_block_swap']),
    auto_block_swap_interval: relation(['auto_block_swap', 'auto_block_swap_mode'], ['auto_block_swap_mode']),
    auto_block_swap_timeout: relation(['auto_block_swap'], ['auto_block_swap']),
    auto_block_swap_swap_io_limit_mb: relation(['auto_block_swap'], ['auto_block_swap']),
    auto_block_swap_vram_reserve_percent: relation(['auto_block_swap'], ['auto_block_swap']),
    auto_block_swap_preference: relation(['auto_block_swap', 'model_family'], ['auto_block_swap']),
    block_swap_transfer_dtype: relation(['blocks_to_swap'], ['blocks_to_swap']),
    block_swap_restore_mode: relation(['blocks_to_swap'], ['blocks_to_swap']),
    block_swap_int8_restore_mode: relation(['block_swap_transfer_dtype'], ['block_swap_transfer_dtype']),
    block_swap_int8_staging_ring: relation(['block_swap_int8_restore_mode'], ['block_swap_int8_restore_mode']),
    block_swap_int8_dual_stream: relation(['block_swap_int8_restore_mode'], ['block_swap_int8_restore_mode']),
    block_swap_int8_restore_chunk_rows: relation(['block_swap_int8_restore_mode'], ['block_swap_int8_restore_mode']),
    disable_block_swap_for_eval: relation(['blocks_to_swap'], ['blocks_to_swap']),
    compile_block_scope: relation(['torch_compile'], ['torch_compile']),
    compile_inductor_mode: relation(['torch_compile'], ['torch_compile']),
    compile_dynamic_seq: relation(['torch_compile'], ['torch_compile']),
    compile_seq_bands: relation(['compile_dynamic_seq'], ['compile_dynamic_seq']),
    activation_memory_budget: relation(['torch_compile'], ['torch_compile']),
    use_custom_down_autograd: relation(['torch_compile'], ['torch_compile']),
    persistent_data_loader_workers: relation(['max_data_loader_n_workers'], ['max_data_loader_n_workers']),
    block_swap_profile_jsonl: relation(['blocks_to_swap'], ['blocks_to_swap']),
    qwen_sample_blocks_to_swap: relation(['qwen_sample_swap_mode'], ['qwen_sample_swap_mode']),
    memory_probe_max_steps: relation(['memory_probe_jsonl'], ['memory_probe_jsonl']),
    peak_probe_max_steps: relation(['peak_probe_jsonl'], ['peak_probe_jsonl']),
    peak_probe_level: relation(['peak_probe_jsonl'], ['peak_probe_jsonl']),
    gradient_flow_probe_every_n_steps: relation(['gradient_flow_probe_jsonl'], ['gradient_flow_probe_jsonl']),
    gradient_flow_probe_dense_steps: relation(['gradient_flow_probe_jsonl'], ['gradient_flow_probe_jsonl']),
    reuse_vae_latents: relation(['use_vae_cache'], ['use_vae_cache']),
    reuse_text_encoder_cache: relation(['use_text_cache'], ['use_text_cache']),
});

function group(stage, cluster, keys) {
    return Object.freeze({ stage, cluster, keys: Object.freeze(keys) });
}

function relation(controlledBy = [], dependsOn = []) {
    return Object.freeze({
        controlledBy: Object.freeze(controlledBy),
        dependsOn: Object.freeze(dependsOn),
    });
}

function buildCatalog() {
    const catalog = {};
    FIELD_GROUPS.forEach(({ stage, cluster, keys }) => keys.forEach((key, index) => {
        if (catalog[key]) throw new Error(`Duplicate config field owner: ${key}`);
        const relationMeta = FIELD_RELATIONS[key] || relation(
            METHOD_CONDITIONAL_CLUSTERS.has(cluster) ? ['$method'] : [],
        );
        const location = AUDIT_ONLY_FIELD_KEYS.has(key)
            ? 'audit_only'
            : (CONTEXT_FIELD_KEYS.has(key) ? 'context' : 'body');
        catalog[key] = Object.freeze({
            key,
            location,
            stage,
            cluster,
            policy: Object.freeze({
                kind: location === 'audit_only'
                    ? 'audit_only'
                    : (relationMeta.controlledBy.length ? 'conditional' : 'always'),
            }),
            controlledBy: relationMeta.controlledBy,
            dependsOn: relationMeta.dependsOn,
            siblingOrder: (index + 1) * 10,
        });
    }));
    return Object.freeze(catalog);
}

export const CONFIG_FIELD_CATALOG = buildCatalog();
export const CONFIG_FIELD_CATALOG_KEYS = Object.freeze(Object.keys(CONFIG_FIELD_CATALOG));

const STAGE_INDEX = new Map(CONFIG_STAGE_META.map((stage, index) => [stage.id, index]));
const CLUSTER_INDEX = new Map(CONFIG_STAGE_META.flatMap((stage) =>
    stage.clusters.map((cluster, index) => [`${stage.id}/${cluster.id}`, index])
));

export function configFieldCatalogEntry(key) {
    const safeKey = String(key || '');
    return CONFIG_FIELD_CATALOG[safeKey] || Object.freeze({
        key: safeKey,
        location: 'audit_only',
        stage: 'resources',
        cluster: 'unclassified',
        policy: Object.freeze({ kind: 'audit_only' }),
        controlledBy: Object.freeze([]),
        dependsOn: Object.freeze([]),
        siblingOrder: 9990,
    });
}

function orderTuple(entry) {
    return [
        STAGE_INDEX.get(entry.stage) ?? Number.MAX_SAFE_INTEGER,
        CLUSTER_INDEX.get(`${entry.stage}/${entry.cluster}`) ?? Number.MAX_SAFE_INTEGER,
        Number(entry.siblingOrder) || 0,
        entry.key,
    ];
}

function compareEntries(left, right) {
    const a = orderTuple(left);
    const b = orderTuple(right);
    for (let index = 0; index < a.length - 1; index += 1) {
        if (a[index] !== b[index]) return a[index] - b[index];
    }
    if (a[3] === b[3]) return 0;
    return String(a[3]) < String(b[3]) ? -1 : 1;
}

export function sortConfigCatalogItems(items, entryFor = (item) => item) {
    const records = items.map((item) => ({ item, entry: entryFor(item) }));
    const byKey = new Map(records.map((record) => [record.entry.key, record]));
    if (byKey.size !== records.length) throw new Error('Config catalog sort received duplicate field keys');
    const outgoing = new Map(records.map((record) => [record.entry.key, []]));
    const indegree = new Map(records.map((record) => [record.entry.key, 0]));
    records.forEach(({ entry }) => new Set([...entry.dependsOn, ...entry.controlledBy]).forEach((dependency) => {
        if (dependency.startsWith('$') || !byKey.has(dependency)) return;
        outgoing.get(dependency).push(entry.key);
        indegree.set(entry.key, indegree.get(entry.key) + 1);
    }));
    const ready = records.filter(({ entry }) => indegree.get(entry.key) === 0)
        .sort((a, b) => compareEntries(a.entry, b.entry));
    const sorted = [];
    while (ready.length) {
        const next = ready.shift();
        sorted.push(next.item);
        outgoing.get(next.entry.key).forEach((key) => {
            indegree.set(key, indegree.get(key) - 1);
            if (indegree.get(key) === 0) {
                ready.push(byKey.get(key));
                ready.sort((a, b) => compareEntries(a.entry, b.entry));
            }
        });
    }
    if (sorted.length !== items.length) throw new Error('Config field catalog contains a dependency cycle');
    return sorted;
}

export function validateConfigFieldCatalog(records = Object.values(CONFIG_FIELD_CATALOG)) {
    const errors = [];
    const byKey = new Map();
    records.forEach((entry) => {
        if (byKey.has(entry.key)) errors.push({ code: 'duplicate-owner', key: entry.key });
        else byKey.set(entry.key, entry);
        if (!STAGE_INDEX.has(entry.stage)) errors.push({ code: 'unknown-stage', key: entry.key });
        else if (!CLUSTER_INDEX.has(`${entry.stage}/${entry.cluster}`)) errors.push({ code: 'unknown-cluster', key: entry.key });
    });
    records.forEach((entry) => new Set([...entry.dependsOn, ...entry.controlledBy]).forEach((dependency) => {
        if (dependency.startsWith('$')) return;
        const parent = byKey.get(dependency);
        if (!parent) {
            errors.push({ code: 'missing-reference', key: entry.key, dependency });
            return;
        }
        if (compareEntries(parent, entry) >= 0) {
            errors.push({ code: 'reverse-dependency', key: entry.key, dependency });
        }
    }));
    if (!errors.some((error) => error.code === 'duplicate-owner')) {
        try {
            sortConfigCatalogItems(records);
        } catch (error) {
            errors.push({ code: 'dependency-cycle', message: error.message });
        }
    }
    return errors;
}

export function assertValidConfigFieldCatalog(records = Object.values(CONFIG_FIELD_CATALOG)) {
    const errors = validateConfigFieldCatalog(records);
    if (errors.length) throw new Error(`Invalid config field catalog: ${JSON.stringify(errors)}`);
    return true;
}
