export const FALLBACK_MODEL_FAMILIES = Object.freeze([
    {
        name: 'moss_image',
        display_name: 'MOSS-Image',
        aliases: ['moss', 'moss_image'],
        supported_network_specs: ['lora'],
        supports_method_adapters: false,
        plain_lora_only: true,
        supported_tasks: ['t2i'],
        supported_attention_modes: ['torch', 'sdpa'],
        ui_constraints: {
            torch_compile: false,
            compile_dynamic_seq: false,
            compile_seq_bands: false,
            compile_inductor_modes: [],
            allows_cudagraphs: false,
            runtime_base_compute: ['bf16'],
            mixed_precisions: ['bf16', 'fp16', 'no'],
            block_swap_transfer_dtypes: ['bf16', 'int8'],
            selective_checkpoint_modes: ['off'],
            supports_checkpoint_offload: false,
            supports_v100_flash_stability: false,
            supports_adaptive_swap_capacity: false,
            accepted_attention_modes: ['sdpa', 'torch'],
            supports_stage_block_swap: false,
        },
        pipeline_parallel: {
            configurable: true,
            runtime_available: true,
            runtime_backend: 'torch_pipelining_1f1b',
            trainer_wired: true,
            supports_stage_block_swap: false,
            requires_batch_divisible_by_microbatches: true,
            block_container: 'layers',
            default_num_blocks: 42,
            known_num_blocks: [42],
            stages: [2],
            schedules: ['1f1b'],
            splits: ['balanced'],
        },
    },
    {
        name: 'anima',
        display_name: 'Anima',
        aliases: ['anima'],
        supported_network_specs: null,
        supports_method_adapters: true,
        plain_lora_only: false,
        supported_attention_modes: ['flash', 'torch', 'mem_efficient', 'sageattn', 'flex', 'xformers', 'sdpa'],
        ui_constraints: {
            torch_compile: true,
            compile_dynamic_seq: true,
            compile_seq_bands: true,
            compile_inductor_modes: ['default', 'max-autotune', 'max-autotune-no-cudagraphs', 'reduce-overhead'],
            allows_cudagraphs: true,
            runtime_base_compute: ['bf16', 'w8a16_convrot', 'w8a8_convrot'],
            mixed_precisions: ['bf16', 'fp16', 'no'],
            block_swap_transfer_dtypes: ['bf16', 'fp8_e4m3', 'int8'],
            selective_checkpoint_modes: [
                'adapter_aware', 'every_other', 'mlp_layer1_only', 'mlp_only', 'off',
                'peak_blocks_adapter_aware', 'peak_blocks_mlp', 'peak_blocks_mlp_layer1',
            ],
            supports_checkpoint_offload: true,
            supports_v100_flash_stability: true,
            supports_adaptive_swap_capacity: true,
            accepted_attention_modes: ['flash', 'flex', 'mem_efficient', 'sageattn', 'sdpa', 'torch', 'xformers'],
            supports_stage_block_swap: false,
        },
        pipeline_parallel: {
            configurable: true,
            runtime_available: true,
            runtime_backend: 'torch_pipelining_1f1b',
            trainer_wired: true,
            supports_stage_block_swap: false,
            requires_batch_divisible_by_microbatches: false,
            block_container: 'blocks',
            default_num_blocks: 28,
            known_num_blocks: [28, 40],
            stages: [2],
            schedules: ['1f1b'],
            splits: ['balanced'],
        },
    },
    {
        name: 'krea2_raw',
        display_name: 'Krea-2',
        aliases: ['krea2', 'krea2_raw'],
        supported_network_specs: null,
        supports_method_adapters: false,
        plain_lora_only: false,
        supported_attention_modes: ['torch', 'flash', 'sdpa'],
        ui_constraints: {
            torch_compile: true,
            compile_dynamic_seq: false,
            compile_seq_bands: false,
            compile_inductor_modes: ['default'],
            allows_cudagraphs: false,
            runtime_base_compute: ['bf16', 'nf4', 'w8a16_convrot', 'w8a8_convrot'],
            mixed_precisions: ['bf16', 'fp16', 'no'],
            block_swap_transfer_dtypes: ['bf16', 'fp8_e4m3', 'int8'],
            selective_checkpoint_modes: ['every_other', 'off'],
            supports_checkpoint_offload: false,
            supports_v100_flash_stability: false,
            supports_adaptive_swap_capacity: true,
            accepted_attention_modes: ['cudnn', 'cudnn_sdpa', 'flash', 'sdpa', 'torch'],
            supports_stage_block_swap: true,
        },
        pipeline_parallel: {
            configurable: true,
            runtime_available: true,
            runtime_backend: 'torch_pipelining_1f1b',
            trainer_wired: true,
            supports_stage_block_swap: true,
            requires_batch_divisible_by_microbatches: true,
            block_container: 'blocks',
            default_num_blocks: 28,
            known_num_blocks: [28],
            stages: [2],
            schedules: ['1f1b'],
            splits: ['balanced'],
        },
    },
    {
        name: 'z_image',
        display_name: 'Z-Image',
        aliases: ['zimage', 'z_image'],
        supported_network_specs: null,
        supports_method_adapters: false,
        plain_lora_only: false,
        supported_attention_modes: ['flash', 'torch', 'sdpa'],
        ui_constraints: {
            torch_compile: false,
            compile_dynamic_seq: false,
            compile_seq_bands: false,
            compile_inductor_modes: [],
            allows_cudagraphs: false,
            runtime_base_compute: ['bf16'],
            mixed_precisions: ['bf16', 'fp16', 'no'],
            block_swap_transfer_dtypes: ['bf16', 'fp8_e4m3', 'int8'],
            selective_checkpoint_modes: ['off'],
            supports_checkpoint_offload: false,
            supports_v100_flash_stability: false,
            supports_adaptive_swap_capacity: true,
            accepted_attention_modes: ['flash', 'native', 'sdpa', 'torch'],
            supports_stage_block_swap: false,
        },
        pipeline_parallel: {
            configurable: true,
            runtime_available: false,
            runtime_backend: null,
            trainer_wired: false,
            supports_stage_block_swap: false,
            requires_batch_divisible_by_microbatches: false,
            block_container: 'layers',
            default_num_blocks: 30,
            known_num_blocks: [30],
            stages: [2],
            schedules: ['1f1b'],
            splits: ['balanced'],
        },
    },
    {
        name: 'qwen_image_2_1',
        display_name: 'Qwen Image 2.1',
        aliases: ['qwen_image_2_1', 'qwen_image_21', 'qwen21'],
        supported_network_specs: ['lora', 'loha', 'lokr'],
        supports_method_adapters: false,
        plain_lora_only: true,
        supported_tasks: ['t2i', 'edit'],
        supported_attention_modes: ['torch', 'sdpa', 'flash'],
        ui_constraints: {
            torch_compile: true,
            compile_dynamic_seq: true,
            compile_seq_bands: false,
            compile_inductor_modes: ['default'],
            allows_cudagraphs: false,
            runtime_base_compute: ['bf16'],
            mixed_precisions: ['bf16', 'fp16', 'no'],
            block_swap_transfer_dtypes: ['bf16', 'fp8_e4m3', 'int8'],
            selective_checkpoint_modes: ['off'],
            supports_checkpoint_offload: false,
            supports_v100_flash_stability: false,
            supports_adaptive_swap_capacity: false,
            accepted_attention_modes: ['flash', 'native', 'sdpa', 'torch'],
            supports_stage_block_swap: false,
        },
        pipeline_parallel: null,
    },
]);

let capabilityByFamily = new Map();
let canonicalByAlias = new Map();
let capabilityLoadPromise = null;

function itemsFromPayload(payload) {
    if (Array.isArray(payload)) return payload;
    if (payload && Array.isArray(payload.items)) return payload.items;
    return [];
}

export function configureModelFamilyCapabilities(payload) {
    const items = itemsFromPayload(payload);
    if (!items.length) throw new Error('model-family capability catalog is empty');

    const nextCapabilities = new Map();
    const nextAliases = new Map();
    items.forEach((item) => {
        const name = String(item?.name || '').trim().toLowerCase().replaceAll('-', '_');
        if (!name) return;
        const normalized = { ...item, name };
        nextCapabilities.set(name, normalized);
        [name, ...(Array.isArray(item.aliases) ? item.aliases : [])].forEach((alias) => {
            const key = String(alias || '').trim().toLowerCase().replaceAll('-', '_');
            if (key) nextAliases.set(key, name);
        });
    });
    if (!nextCapabilities.size) {
        throw new Error('model-family capability catalog has no valid entries');
    }
    capabilityByFamily = nextCapabilities;
    canonicalByAlias = nextAliases;
    return [...capabilityByFamily.values()];
}

configureModelFamilyCapabilities(FALLBACK_MODEL_FAMILIES);

export async function loadModelFamilyCapabilities(api) {
    if (typeof api !== 'function') throw new TypeError('api must be a function');
    if (!capabilityLoadPromise) {
        capabilityLoadPromise = Promise.resolve()
            .then(() => api('/api/config/model-families'))
            .then((payload) => configureModelFamilyCapabilities(payload))
            .catch((error) => {
                capabilityLoadPromise = null;
                console.warn('[model-family] capability catalog unavailable; using fallback', error);
                return [...capabilityByFamily.values()];
            });
    }
    return capabilityLoadPromise;
}

export function normalizeModelFamily(value) {
    const normalized = String(value ?? '').trim().toLowerCase().replaceAll('-', '_');
    return canonicalByAlias.get(normalized) || normalized;
}

export function modelFamilyCapability(value) {
    return capabilityByFamily.get(normalizeModelFamily(value)) || null;
}

export function modelFamilyPipelineCapability(value) {
    return modelFamilyCapability(value)?.pipeline_parallel || null;
}

export function modelFamilySupportsPipelineParallel(value) {
    const capability = modelFamilyPipelineCapability(value);
    return capability?.configurable === true && capability?.runtime_available === true;
}

function taskFromContext(context) {
    if (typeof context === 'string') return context.trim().toLowerCase();
    const values = context?.values && typeof context.values === 'object' ? context.values : context;
    const task = context?.task ?? values?.qwen_image_2_1_task ?? values?.qwen_task;
    return String(task ?? 't2i').trim().toLowerCase();
}

function qwenSupportedNetworkSpecs(capability) {
    const specs = capability?.supported_network_specs;
    if (Array.isArray(specs)) {
        return specs.map((spec) => String(spec).trim().toLowerCase()).filter(Boolean);
    }
    // Older capability payloads only exposed plain_lora_only. Keep that
    // contract conservative until the backend publishes explicit variants.
    if (capability?.plain_lora_only === true) return ['lora'];
    return ['lora', 'loha', 'lokr'];
}

export function modelFamilyOptionSupported(fieldKey, family, option, context) {
    if (fieldKey === 'auto_block_swap_preference' && option === 'ram') {
        return isKrea2ModelFamily(family);
    }
    const capability = modelFamilyCapability(family);
    if (!capability) return true;
    const value = String(option ?? '').trim();
    const lower = value.toLowerCase();
    const ui = capability.ui_constraints || null;
    if (ui) {
        if (fieldKey === 'attn_mode' && Array.isArray(ui.accepted_attention_modes)) {
            return ui.accepted_attention_modes.includes(lower);
        }
        if (fieldKey === 'compile_inductor_mode' && Array.isArray(ui.compile_inductor_modes) && ui.compile_inductor_modes.length) {
            return ui.compile_inductor_modes.includes(lower);
        }
        if (fieldKey === 'block_swap_transfer_dtype' && Array.isArray(ui.block_swap_transfer_dtypes)) {
            return ui.block_swap_transfer_dtypes.includes(lower);
        }
        if ((capability.name === 'krea2_raw' || capability.name === 'moss_image')
            && fieldKey === 'selective_checkpoint'
            && Array.isArray(ui.selective_checkpoint_modes)) {
            return ui.selective_checkpoint_modes.includes(lower);
        }
        if (capability.name === 'krea2_raw' && fieldKey === 'v100_flash_stability' && ui.supports_v100_flash_stability === false) {
            return lower === 'off';
        }
        if (capability.name === 'moss_image' && fieldKey === 'base_compute' && Array.isArray(ui.runtime_base_compute)) {
            return ui.runtime_base_compute.includes(lower);
        }
    }
    if (capability.name === 'moss_image') {
        const fixedOptions = {
            lora_adapter_kind: ['lora'], network_module: ['networks.lora_anima'],
            mixed_precision: ['bf16', 'fp16', 'no'], precision_preference: ['bf16', 'fp16', 'fp32'],
            moss_te_quantization: ['nf4', 'bf16'],
            timestep_sampling: ['uniform'], weighting_scheme: ['none'], sample_sampler: ['euler'],
            router_source: ['none'],
        }[fieldKey];
        if (fixedOptions) return fixedOptions.includes(lower);
        if (['use_moe_style', 'route_per_layer', 'dora_wd', 'use_ortho', 'use_loha', 'use_lokr'].includes(fieldKey)) {
            return ['false', 'off', 'none', '0'].includes(lower);
        }
    }
    if (fieldKey === 'attn_mode') {
        return (capability.supported_attention_modes || []).includes(lower);
    }
    if (fieldKey === 'qwen_image_2_1_task') {
        const supportedTasks = Array.isArray(capability.supported_tasks)
            ? capability.supported_tasks
            : capability.name === 'qwen_image_2_1' ? ['t2i', 'edit'] : [];
        return supportedTasks.includes(value.toLowerCase());
    }
    if (capability.name === 'qwen_image_2_1') {
        const task = taskFromContext(context);
        if (fieldKey === 'lora_adapter_kind') {
            const supportedSpecs = qwenSupportedNetworkSpecs(capability);
            return (task === 'edit' ? ['lora'] : supportedSpecs).includes(value.toLowerCase());
        }
        if (fieldKey === 'network_module') return value === 'networks.lora_anima';
        if (fieldKey === 'use_moe_style') return ['false', 'none', 'off', '0'].includes(value.toLowerCase());
        if (fieldKey === 'router_source') return value.toLowerCase() === 'none';
        if (fieldKey === 'route_per_layer') return ['false', 'off', '0'].includes(value.toLowerCase());
        if (fieldKey === 'dora_wd' || fieldKey === 'use_ortho') return ['false', 'off', '0'].includes(value.toLowerCase());
        if (fieldKey === 'use_loha') return ['false', 'off', '0'].includes(value.toLowerCase()) || (task === 't2i' && ['true', 'on', '1'].includes(value.toLowerCase()));
        if (fieldKey === 'use_lokr') return ['false', 'off', '0'].includes(value.toLowerCase()) || (task === 't2i' && ['true', 'on', '1'].includes(value.toLowerCase()));
    }
    // Variant selection is open; concrete network/forward contracts validate execution.
    return true;
}

export function modelFamilySelectOptions(fieldKey, family, options, currentValue, context) {
    const supported = (Array.isArray(options) ? options : []).filter((option) => (
        modelFamilyOptionSupported(fieldKey, family, option, context)
    ));
    if (
        currentValue !== null
        && currentValue !== undefined
        && !supported.some((option) => String(option) === String(currentValue))
    ) {
        supported.push(currentValue);
    }
    return supported.map((value) => ({
        value,
        supported: modelFamilyOptionSupported(fieldKey, family, value, context),
    }));
}

export function isKrea2ModelFamily(value) {
    return normalizeModelFamily(value) === 'krea2_raw';
}
