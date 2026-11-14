import {
    CHIMERA_UI_DEFAULT_FIELDS,
    IP_ADAPTER_UI_DEFAULT_FIELDS,
    LOKR_SCOPED_FIELD_KEYS,
    METHOD_SCOPED_CONFIG_FORM_FIELDS,
    NETWORK_ARG_FIELD_MAP,
    SPD_UI_DEFAULT_FIELDS,
    VERA_SCOPED_FIELD_KEYS,
} from './defaults.js';
import { normalizeBooleanConfigValue } from './config-field-types.js';
import {
    modelFamilyCapability,
    modelFamilyPipelineCapability,
    modelFamilySupportsPipelineParallel,
    normalizeModelFamily,
} from './model-family.js';

const CONVROT_FIELD_KEYS = new Set([
    'convrot_group_size',
    'convrot_scope',
    'convrot_hadamard',
    'convrot_min_in_features',
    'convrot_largest_in_features_only',
    'convrot_large_layer_mode',
    'convrot_large_min_in_features',
]);

const PIPELINE_PARALLEL_FIELD_KEYS = new Set([
    'pipeline_parallel',
    'pipeline_parallel_stages',
    'pipeline_parallel_microbatches',
    'pipeline_parallel_schedule',
    'pipeline_parallel_split',
]);

const COMPILE_CHILD_KEYS = new Set([
    'compile_block_scope',
    'compile_inductor_mode',
    'compile_dynamic_seq',
    'compile_seq_bands',
    'activation_memory_budget',
    'use_custom_down_autograd',
    'debug_finite_checks',
]);

const BLOCK_SWAP_CHILD_KEYS = new Set([
    'block_swap_transfer_dtype',
    'block_swap_restore_mode',
    'block_swap_int8_restore_mode',
    'block_swap_int8_staging_ring',
    'block_swap_int8_dual_stream',
    'block_swap_int8_restore_chunk_rows',
    'disable_block_swap_for_eval',
    'block_swap_profile_jsonl',
]);

// int8-only knobs: gated on transfer_dtype=int8, then narrowed by restore mode
// (staging_ring/dual_stream belong to fused, chunk_rows to reuse_storage).
const BLOCK_SWAP_INT8_FUSED_KEYS = new Set([
    'block_swap_int8_staging_ring',
    'block_swap_int8_dual_stream',
]);

const PROBE_PARENT_BY_KEY = new Map([
    ['memory_probe_max_steps', 'memory_probe_jsonl'],
    ['peak_probe_max_steps', 'peak_probe_jsonl'],
    ['peak_probe_level', 'peak_probe_jsonl'],
    ['gradient_flow_probe_every_n_steps', 'gradient_flow_probe_jsonl'],
    ['gradient_flow_probe_dense_steps', 'gradient_flow_probe_jsonl'],
]);

const CACHE_PARENT_BY_KEY = new Map([
    ['reuse_vae_latents', ['use_vae_cache', 'VAE 缓存', 'vae-cache-disabled']],
    ['reuse_text_encoder_cache', ['use_text_cache', '文本缓存', 'text-cache-disabled']],
]);

const FAMILY_LABELS = Object.freeze({
    lokr: 'LoKr',
    vera: 'VeRA',
    soft_tokens: 'Soft Tokens',
    ip_adapter: 'IP-Adapter',
    easycontrol: 'EasyControl',
});

const METHOD_LABELS = Object.freeze({
    spd: 'SPD',
    chimera: 'ChimeraHydra',
    ip_adapter: 'IP-Adapter',
    easycontrol: 'EasyControl',
    soft_tokens: 'Soft Tokens',
});

const LORA_ADAPTER_KINDS = new Set(['lora', 'loha', 'lokr', 'glora', 'vera']);
const LORA_ADAPTER_METHODS = new Set([
    'lora', 'glora', 'loha', 'lokr', 'ortholora', 'tlora', 'hydralora', 'reft', 'chimera',
]);
const SHARED_METHOD_FIELD_KEYS = new Set(['seed', 'channel_scaling_alpha']);
const FAMILY_CAPABILITY_FIELD_KEYS = new Set([
    'network_module',
    'lora_adapter_kind',
    'attn_mode',
]);

function unavailable(reason, code) {
    return { enabled: false, reason, code };
}

function contextValue(context, key, fallback = undefined) {
    if (Object.prototype.hasOwnProperty.call(context?.values || {}, key)) return context.values[key];
    return fallback;
}

function toggleEnabled(context, key, fallback = false) {
    return normalizeBooleanConfigValue(key, contextValue(context, key, fallback), fallback);
}

function choiceEnabled(value) {
    if (value === true || value === 1) return true;
    return !['', '0', 'false', 'none', 'off', 'disabled'].includes(String(value ?? '').trim().toLowerCase());
}

function familyEnabled(family, adapter, method) {
    return {
        lokr: adapter === 'lokr' || method === 'lokr',
        vera: adapter === 'vera' || method === 'vera',
        soft_tokens: method === 'soft_tokens',
        ip_adapter: method === 'ip_adapter',
        easycontrol: method === 'easycontrol',
    }[family] === true;
}

export function resolveConfigAdapterKind(values = {}) {
    const selected = String(values.lora_adapter_kind ?? '').trim().toLowerCase();
    if (LORA_ADAPTER_KINDS.has(selected)) return selected;
    if (normalizeBooleanConfigValue('use_glora', values.use_glora)) return 'glora';
    if (normalizeBooleanConfigValue('use_vera', values.use_vera)) return 'vera';
    if (normalizeBooleanConfigValue('use_lokr', values.use_lokr)) return 'lokr';
    if (normalizeBooleanConfigValue('use_loha', values.use_loha)) return 'loha';
    return 'lora';
}

export function configFieldAvailability(key, context = {}) {
    const method = String(context.method || 'lora').trim().toLowerCase();
    const adapter = String(context.adapter || 'lora').trim().toLowerCase();
    const baseCompute = String(context.baseCompute || 'bf16').trim().toLowerCase();
    const modelFamily = normalizeModelFamily(context.modelFamily || 'anima');
    const familyCapability = modelFamilyCapability(modelFamily);
    const pipelineParallel = context.pipelineParallel === true
        || toggleEnabled(context, 'pipeline_parallel');

    const pipelineCapability = modelFamilyPipelineCapability(modelFamily);
    if (PIPELINE_PARALLEL_FIELD_KEYS.has(key) && pipelineCapability?.configurable !== true) {
        return unavailable(
            `流水线并行尚未为当前模型族 ${modelFamily} 声明分层能力。`,
            'pipeline-parallel-model-family',
        );
    }
    if (PIPELINE_PARALLEL_FIELD_KEYS.has(key) && !modelFamilySupportsPipelineParallel(modelFamily)) {
        if (key === 'pipeline_parallel' && pipelineParallel) {
            return { enabled: true, reason: '', code: null };
        }
        return unavailable(
            `当前模型族 ${modelFamily} 只有流水线分层配置，主训练 runtime 尚未接入，不能启动该模式。`,
            'pipeline-parallel-runtime-unavailable',
        );
    }
    if (key !== 'pipeline_parallel' && PIPELINE_PARALLEL_FIELD_KEYS.has(key) && !pipelineParallel) {
        return unavailable(
            '请先开启流水线并行。',
            'pipeline-parallel-disabled',
        );
    }

    if (!familyCapability && FAMILY_CAPABILITY_FIELD_KEYS.has(key)) {
        return unavailable(
            `未识别模型族 ${modelFamily}；请选择已注册模型族后再编辑该能力字段。`,
            'model-family-unknown',
        );
    }

    if (key === 'moss_te_quantization' && modelFamily !== 'moss_image') {
        return {
            ...unavailable('仅用于 MOSS-Image 的冻结文本编码器。', 'moss-te-family'),
            visible: false,
        };
    }
    // This token-cap field controls MOSS real-length states and cache identity;
    // other families use their own tokenizer length settings.
    if (key === 'max_token_length' && modelFamily !== 'moss_image') {
        return {
            ...unavailable('文本编码上限仅用于 MOSS-Image。', 'moss-token-cap-family'),
            visible: false,
        };
    }
    if (modelFamily === 'moss_image') {
        const fixed = {
            torch_compile: false, compile_dynamic_seq: false, compile_seq_bands: false,
            gradient_checkpointing: true, use_vae_cache: true, use_text_cache: true,
            network_train_unet_only: true, weighted_captions: false, masked_loss: false,
            use_shuffled_caption_variants: false, cache_llm_adapter_outputs: false,
            auto_block_swap: false, selective_checkpoint: 'off', base_compute: 'bf16',
        };
        if (Object.prototype.hasOwnProperty.call(fixed, key)) {
            const required = fixed[key];
            const current = contextValue(context, key, required);
            const matches = typeof required === 'boolean'
                ? normalizeBooleanConfigValue(key, current, required) === required
                : String(current).toLowerCase() === required;
            if (matches) return unavailable(`MOSS-Image 首版要求 ${key}=${required}。`, 'moss-fixed-contract');
        }
    }

    if (modelFamily === 'krea2_raw') {
        const fixedKrea2Values = {
            compile_dynamic_seq: [false, 'Krea-2 使用固定 token-family 编译图，不支持动态序列编译。', 'krea2-compile-dynamic-seq'],
            compile_seq_bands: [false, 'Krea-2 使用固定 token-family 编译图，不支持分带动态序列编译。', 'krea2-compile-seq-bands'],
            compile_inductor_mode: ['default', 'Krea-2 仅支持 default Inductor mode。', 'krea2-compile-inductor-mode'],
            v100_flash_stability: ['off', 'V100 Flash 稳定性诊断仅适用于 Anima；Krea-2 必须保持 off。', 'krea2-v100-flash-stability'],
        }[key];
        if (fixedKrea2Values) {
            const [requiredValue, reason, code] = fixedKrea2Values;
            const currentValue = contextValue(context, key, key === 'compile_dynamic_seq' || key === 'compile_seq_bands' ? false : requiredValue);
            const isCurrentValueFixed = typeof requiredValue === 'boolean'
                ? normalizeBooleanConfigValue(key, currentValue, false) === requiredValue
                : String(currentValue ?? '').trim().toLowerCase() === requiredValue;
            return isCurrentValueFixed
                ? unavailable(reason, code)
                : { enabled: true, reason: '', code: null };
        }
        if (key === 'selective_checkpoint') {
            const mode = String(contextValue(context, key, 'off') ?? 'off').trim().toLowerCase();
            if (!['off', 'every_other'].includes(mode)) {
                return { enabled: true, reason: '', code: null };
            }
        }
    }

    if (key === 'lora_adapter_kind') {
        if (!LORA_ADAPTER_METHODS.has(method)) {
            return unavailable(
                `当前方法为 ${METHOD_LABELS[method] || method}，不使用 LoRA Adapter 变体。`,
                'method-context',
            );
        }
    }

    if (key === 'max_train_steps' && context.maxTrainEpochsConfigured === true) {
        return unavailable(
            'max_train_epochs 已设置；训练端会按当前数据集重新计算并覆盖 max_train_steps。',
            'epochs-override-steps',
        );
    }

    if (
        ['network_dim', 'network_alpha'].includes(key)
        && context.dimFromWeights === true
        && String(context.networkWeights || '').trim()
    ) {
        return unavailable(
            '已开启从热启动权重读取维度；训练端会使用检查点中的 rank/alpha。',
            'weights-override-dimensions',
        );
    }

    if (COMPILE_CHILD_KEYS.has(key) && !toggleEnabled(context, 'torch_compile', true)) {
        return unavailable('请先启用 torch.compile。', 'torch-compile-disabled');
    }
    if (key === 'compile_seq_bands' && !toggleEnabled(context, 'compile_dynamic_seq')) {
        return unavailable('请先启用动态序列编译。', 'compile-dynamic-seq-disabled');
    }

    if (key === 'selective_checkpoint_blocks'
        && !choiceEnabled(contextValue(context, 'selective_checkpoint', 'off'))) {
        return unavailable('请先选择一种选择性重算模式。', 'selective-checkpoint-disabled');
    }

    if (key === 'blocks_to_swap' && toggleEnabled(context, 'auto_block_swap')) {
        return unavailable('本次运行由 AUTO 校准；保留此手动值供关闭 AUTO 后使用。', 'auto-block-swap-enabled');
    }
    if (key === 'qwen_sample_blocks_to_swap'
        && String(contextValue(context, 'qwen_sample_swap_mode', 'inherit')).trim().toLowerCase() !== 'custom') {
        return unavailable('请先选择“自定义”预览块交换。', 'qwen-preview-swap-inherit');
    }
    if (['auto_block_swap_mode', 'auto_block_swap_interval', 'auto_block_swap_max_trials', 'auto_block_swap_timeout', 'auto_block_swap_swap_io_limit_mb', 'auto_block_swap_vram_reserve_percent', 'auto_block_swap_preference'].includes(key)
        && !toggleEnabled(context, 'auto_block_swap')) {
        return unavailable('请先启用 AUTO 块交换。', 'auto-block-swap-disabled');
    }
    const dynamicSwap = contextValue(context, 'auto_block_swap_mode', 'startup') === 'dynamic';
    if (key === 'auto_block_swap_interval' && !dynamicSwap) {
        return unavailable('仅用于 dynamic 模式。', 'dynamic-swap-disabled');
    }
    if (key === 'auto_block_swap_max_trials' && dynamicSwap) {
        return unavailable('dynamic 不运行启动候选搜索。', 'startup-swap-disabled');
    }
    if (['auto_block_swap_mode', 'auto_block_swap_interval'].includes(key)
        && modelFamily !== 'krea2_raw' && !dynamicSwap) {
        return unavailable('动态调整当前仅支持 Krea-2。', 'dynamic-swap-family');
    }
    if (BLOCK_SWAP_CHILD_KEYS.has(key)
        && !toggleEnabled(context, 'auto_block_swap')
        && !(Number(contextValue(context, 'blocks_to_swap', 0)) > 0)) {
        return unavailable('请先设置大于 0 的 block swap 数量。', 'block-swap-disabled');
    }

    const int8ChildKey = BLOCK_SWAP_INT8_FUSED_KEYS.has(key)
        || key === 'block_swap_int8_restore_chunk_rows';
    if (key === 'block_swap_int8_restore_mode' || int8ChildKey) {
        const transferDtype = String(contextValue(context, 'block_swap_transfer_dtype', 'bf16') ?? 'bf16')
            .trim()
            .toLowerCase();
        if (transferDtype !== 'int8') {
            return unavailable(
                '请先把传输精度切换为 int8（实验性：流量减半，约 1% loss 偏移）。',
                'block-swap-int8-disabled',
            );
        }
        const int8RestoreMode = String(contextValue(context, 'block_swap_int8_restore_mode', 'fused') ?? 'fused')
            .trim()
            .toLowerCase();
        if (BLOCK_SWAP_INT8_FUSED_KEYS.has(key) && int8RestoreMode !== 'fused') {
            return unavailable('仅 INT8 恢复模式 = fused 使用该参数。', 'block-swap-int8-not-fused');
        }
        if (key === 'block_swap_int8_restore_chunk_rows' && int8RestoreMode !== 'reuse_storage') {
            return unavailable('仅 INT8 恢复模式 = reuse_storage 使用该参数。', 'block-swap-int8-not-reuse');
        }
    }

    const probeParent = PROBE_PARENT_BY_KEY.get(key);
    if (probeParent && !choiceEnabled(contextValue(context, probeParent, 'off'))) {
        return unavailable(`请先启用 ${probeParent}。`, 'probe-disabled');
    }

    const cacheParent = CACHE_PARENT_BY_KEY.get(key);
    if (cacheParent && !toggleEnabled(context, cacheParent[0])) {
        return unavailable(`请先启用${cacheParent[1]}。`, cacheParent[2]);
    }

    if (CONVROT_FIELD_KEYS.has(key) && !['w8a16_convrot', 'w8a8_convrot'].includes(baseCompute)) {
        return unavailable(
            `当前基础计算类型为 ${baseCompute}；ConvRot 仅适用于 W8A16 或 W8A8。请先切换基础计算类型。`,
            'convrot-base-compute',
        );
    }

    const methodScope = METHOD_SCOPED_CONFIG_FORM_FIELDS.get(key);
    if (methodScope && !methodScope.has(method)) {
        const expected = [...methodScope].map((name) => METHOD_LABELS[name] || name).join('、');
        return unavailable(`该参数仅适用于 ${expected}，当前方法为 ${METHOD_LABELS[method] || method}。`, 'method-scope');
    }

    if (SPD_UI_DEFAULT_FIELDS.has(key) && !SHARED_METHOD_FIELD_KEYS.has(key) && method !== 'spd') {
        return unavailable(`该参数属于 SPD 实验，当前方法为 ${METHOD_LABELS[method] || method}。切换到 SPD 后可编辑。`, 'spd-method');
    }

    if (
        CHIMERA_UI_DEFAULT_FIELDS.has(key)
        && !SHARED_METHOD_FIELD_KEYS.has(key)
        && key !== 'use_chimera_hydra'
        && method !== 'chimera'
    ) {
        return unavailable(`该参数属于 ChimeraHydra，当前方法为 ${METHOD_LABELS[method] || method}。切换方法后可编辑。`, 'chimera-method');
    }

    if (IP_ADAPTER_UI_DEFAULT_FIELDS.has(key) && method !== 'ip_adapter') {
        return unavailable(`该参数属于 IP-Adapter，当前方法为 ${METHOD_LABELS[method] || method}。切换方法后可编辑。`, 'ip-adapter-method');
    }

    const spec = NETWORK_ARG_FIELD_MAP.get(key);
    if (spec && !familyEnabled(spec.family, adapter, method)) {
        return unavailable(`该参数属于 ${FAMILY_LABELS[spec.family] || spec.family}，当前适配器为 ${FAMILY_LABELS[adapter] || adapter}。`, 'adapter-family');
    }

    if (LOKR_SCOPED_FIELD_KEYS.has(key) && adapter !== 'lokr') {
        return unavailable(`该参数仅适用于 LoKr，当前适配器为 ${FAMILY_LABELS[adapter] || adapter}。请先启用 LoKr。`, 'lokr-adapter');
    }

    if (VERA_SCOPED_FIELD_KEYS.has(key) && adapter !== 'vera') {
        return unavailable(`该参数仅适用于 VeRA，当前适配器为 ${FAMILY_LABELS[adapter] || adapter}。请先启用 VeRA。`, 'vera-adapter');
    }

    if (key === 'dora_wd' && adapter !== 'lora') {
        return unavailable(`DoRA 权重衰减仅适用于普通 LoRA，当前适配器为 ${FAMILY_LABELS[adapter] || adapter}。`, 'dora-adapter');
    }

    return { enabled: true, reason: '', code: null };
}
