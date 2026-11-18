import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NEXT_DOMAIN = ROOT / "web" / "frontend-next" / "src" / "features" / "training-config" / "domain"
LOKR_KEYS = [
    "lokr_factor",
    "lokr_use_einsum",
    "lokr_decompose_w2",
    "lokr_full_factor",
    "lokr_allow_legacy_dim",
    "lokr_factor_group_size",
    "lokr_project_chunk_bytes",
    "lokr_grouped_delta_backend",
    "lokr_grouped_delta_backward_backend",
]
VERA_KEYS = ["vera_projection_prng_key", "vera_d_initial", "vera_save_projection"]
def _node_json(script: str) -> dict:
    if not shutil.which("node"):
        pytest.skip("node is required for Dragon disclosure checks")
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return json.loads(result.stdout)


def test_disclosure_rules_follow_adapter_method_and_family_context() -> None:
    module_uri = (
        NEXT_DOMAIN / "config-field-disclosure-rules.js"
    ).resolve().as_uri()
    payload = _node_json(f"""
const mod = await import({json.dumps(module_uri + '?matrix')});
const visible = (key, context) => mod.configFieldDisclosure(key, context).visible;
const common = {{ values: {{ use_vae_cache: true, use_text_cache: true }} }};
const contexts = {{
  lora: {{ ...common, method: 'lora', adapter: 'lora', modelFamily: 'anima' }},
  lokr: {{ ...common, method: 'lora', adapter: 'lokr', modelFamily: 'anima' }},
  vera: {{ ...common, method: 'lora', adapter: 'vera', modelFamily: 'anima', values: {{ ...common.values, use_timestep_mask: true }} }},
  krea: {{ ...common, method: 'lora', adapter: 'lora', modelFamily: 'krea2_raw' }},
  kreaLegacyOrtho: {{ ...common, method: 'lora', adapter: 'lora', modelFamily: 'krea2_raw', values: {{ ...common.values, use_ortho: true }} }},
  kreaLegacyMoe: {{ ...common, method: 'hydralora', adapter: 'lora', modelFamily: 'krea2_raw', values: {{ ...common.values, use_moe_style: 'shared_A' }} }},
  kreaLegacyLokr: {{ ...common, method: 'lora', adapter: 'lokr', modelFamily: 'krea2_raw' }},
  zImageLegacyReft: {{ ...common, method: 'reft', adapter: 'lora', modelFamily: 'z_image', values: {{ ...common.values, add_reft: true }} }},
  switchedLokr: {{ ...common, method: 'lokr', adapter: 'lora', modelFamily: 'anima' }},
  ip: {{ ...common, method: 'ip_adapter', adapter: 'lora', modelFamily: 'anima', values: {{ ...common.values, use_ip_adapter: true, pe_lora_enabled: false }} }},
}};
console.log(JSON.stringify({{
  lora: {{
    general: visible('learning_rate', contexts.lora),
    dora: visible('dora_wd', contexts.lora),
    lokr: {json.dumps(LOKR_KEYS)}.map((key) => visible(key, contexts.lora)),
    vera: {json.dumps(VERA_KEYS)}.map((key) => visible(key, contexts.lora)),
    ip: visible('use_ip_adapter', contexts.lora),
    ipCache: visible('ip_features_cache_to_disk', contexts.lora),
    weightDecay: visible('weight_decay', contexts.lora),
  }},
  lokr: {{
    fields: {json.dumps(LOKR_KEYS)}.map((key) => visible(key, contexts.lokr)),
    dora: visible('dora_wd', contexts.lokr),
    vera: visible('vera_d_initial', contexts.lokr),
  }},
  switchedLokr: visible('lokr_factor', contexts.switchedLokr),
  vera: {{
    fields: {json.dumps(VERA_KEYS)}.map((key) => visible(key, contexts.vera)),
    lokr: visible('lokr_factor', contexts.vera),
    timestep: visible('use_timestep_mask', contexts.vera),
  }},
  krea: {{
    general: visible('learning_rate', contexts.krea),
    adapter: mod.configFieldDisclosure('lora_adapter_kind', contexts.krea),
    dora: visible('dora_wd', contexts.krea),
    ortho: visible('use_ortho', contexts.krea),
  }},
  familyGuards: {{
    kreaNetworkModule: mod.configFieldDisclosure('network_module', contexts.krea),
    kreaNetworkArgs: mod.configFieldDisclosure('network_args', contexts.krea),
    kreaOrtho: mod.configFieldDisclosure('use_ortho', contexts.kreaLegacyOrtho),
    kreaRoute: mod.configFieldDisclosure('route_per_layer', contexts.kreaLegacyMoe),
    kreaLokr: mod.configFieldDisclosure('lokr_factor', contexts.kreaLegacyLokr),
    zImageReft: mod.configFieldDisclosure('reft_dim', contexts.zImageLegacyReft),
  }},
  ip: {{
    parent: visible('use_ip_adapter', contexts.ip),
    encoder: visible('encoder', contexts.ip),
    peRank: visible('pe_lora_rank', contexts.ip),
    easy: visible('use_easycontrol', contexts.ip),
  }},
}}));
""")

    assert payload["lora"] == {
        "general": True,
        "dora": True,
        "lokr": [False] * len(LOKR_KEYS),
        "vera": [False] * len(VERA_KEYS),
        "ip": False,
        "ipCache": False,
        "weightDecay": False,
    }
    assert payload["lokr"]["fields"] == [True] * len(LOKR_KEYS)
    assert payload["lokr"]["dora"] is False
    assert payload["lokr"]["vera"] is False
    assert payload["switchedLokr"] is False
    assert payload["vera"]["fields"] == [True] * len(VERA_KEYS)
    assert payload["vera"]["lokr"] is False
    assert payload["vera"]["timestep"] is True
    assert payload["krea"]["general"] is True
    assert payload["krea"]["adapter"]["visible"] is True
    assert payload["krea"]["adapter"]["code"] is None
    assert payload["krea"]["dora"] is True
    assert payload["krea"]["ortho"] is False
    for result in payload["familyGuards"].values():
        assert result["visible"] is True
        assert result["code"] is None
    assert payload["ip"] == {
        "parent": True,
        "encoder": True,
        "peRank": False,
        "easy": False,
    }


def test_unknown_model_family_fails_closed_for_method_and_capability_fields() -> None:
    disclosure_uri = (
        NEXT_DOMAIN / "config-field-disclosure-rules.js"
    ).resolve().as_uri()
    availability_uri = (
        NEXT_DOMAIN / "config-field-availability.js"
    ).resolve().as_uri()
    payload = _node_json(f"""
const disclosure = await import({json.dumps(disclosure_uri + '?unknown-family')});
const availability = await import({json.dumps(availability_uri + '?unknown-family')});
const context = {{
  method: 'hydralora',
  adapter: 'lokr',
  modelFamily: 'future_family',
  values: {{ use_ortho: true, use_moe_style: 'shared_A' }},
}};
console.log(JSON.stringify({{
  general: disclosure.configFieldDisclosure('learning_rate', context),
  adapter: disclosure.configFieldDisclosure('lora_adapter_kind', context),
  adapterAvailability: availability.configFieldAvailability('lora_adapter_kind', context),
  networkModuleDisclosure: disclosure.configFieldDisclosure('network_module', context),
  networkArgs: disclosure.configFieldDisclosure('network_args', context),
  lokr: disclosure.configFieldDisclosure('lokr_factor', context),
  ortho: disclosure.configFieldDisclosure('use_ortho', context),
  route: disclosure.configFieldDisclosure('route_per_layer', context),
  networkModule: availability.configFieldAvailability('network_module', context),
  attention: availability.configFieldAvailability('attn_mode', context),
  generalAvailability: availability.configFieldAvailability('learning_rate', context),
}}));
""")

    assert payload["general"]["visible"] is True
    assert payload["adapter"]["visible"] is True
    assert payload["adapter"]["code"] is None
    for key in ("networkModuleDisclosure", "networkArgs", "lokr", "ortho", "route"):
        assert payload[key]["visible"] is False
        assert payload[key]["code"] == "model-family-unknown"
    for key in ("adapterAvailability", "networkModule", "attention"):
        assert payload[key]["enabled"] is False
        assert payload[key]["code"] == "model-family-unknown"
    assert payload["generalAvailability"]["enabled"] is True


def test_adapter_selector_is_persistent_and_capability_aware() -> None:
    disclosure_uri = (
        NEXT_DOMAIN / "config-field-disclosure-rules.js"
    ).resolve().as_uri()
    availability_uri = (
        NEXT_DOMAIN / "config-field-availability.js"
    ).resolve().as_uri()
    payload = _node_json(f"""
const disclosure = await import({json.dumps(disclosure_uri + '?persistent-adapter')});
const availability = await import({json.dumps(availability_uri + '?persistent-adapter')});
const state = (context) => ({{
  presentation: disclosure.configFieldDisclosure('lora_adapter_kind', context),
  availability: availability.configFieldAvailability('lora_adapter_kind', context),
}});
console.log(JSON.stringify({{
  anima: state({{ method: 'lora', adapter: 'lora', modelFamily: 'anima' }}),
  krea: state({{ method: 'lora', adapter: 'lora', modelFamily: 'krea2_raw' }}),
  zImage: state({{ method: 'lora', adapter: 'lora', modelFamily: 'z_image' }}),
  independentMethod: state({{ method: 'ip_adapter', adapter: 'lora', modelFamily: 'anima' }}),
  unknownFamily: state({{ method: 'lora', adapter: 'lora', modelFamily: 'future_family' }}),
  kreaLegacyLokr: state({{ method: 'lora', adapter: 'lokr', modelFamily: 'krea2_raw' }}),
}}));
""")

    for result in payload.values():
        assert result["presentation"]["visible"] is True
        assert result["presentation"]["code"] is None

    assert payload["anima"]["availability"]["enabled"] is True
    for key in ("krea", "zImage"):
        assert payload[key]["availability"]["enabled"] is True
        assert payload[key]["availability"]["code"] is None
    assert payload["independentMethod"]["availability"]["enabled"] is False
    assert payload["independentMethod"]["availability"]["code"] == "method-context"
    assert payload["unknownFamily"]["availability"]["enabled"] is False
    assert payload["unknownFamily"]["availability"]["code"] == "model-family-unknown"
    assert payload["kreaLegacyLokr"]["availability"]["enabled"] is True


def test_general_parent_children_stay_visible_and_disable_until_enabled() -> None:
    disclosure_uri = (
        NEXT_DOMAIN / "config-field-disclosure-rules.js"
    ).resolve().as_uri()
    availability_uri = (
        NEXT_DOMAIN / "config-field-availability.js"
    ).resolve().as_uri()
    payload = _node_json(f"""
const disclosure = await import({json.dumps(disclosure_uri + '?persistent-parents')});
const availability = await import({json.dumps(availability_uri + '?persistent-parents')});
const state = (key, values, extra = {{}}) => {{
  const context = {{ method: 'lora', adapter: 'lora', modelFamily: 'anima', values, ...extra }};
  return {{
    presentation: disclosure.configFieldDisclosure(key, context),
    availability: availability.configFieldAvailability(key, context),
  }};
}};
console.log(JSON.stringify({{
  pipelineOff: state('pipeline_parallel_stages', {{ pipeline_parallel: false }}),
  pipelineOn: state('pipeline_parallel_stages', {{ pipeline_parallel: true }}, {{ pipelineParallel: true }}),
  pipelineUnwired: state('pipeline_parallel_stages', {{ pipeline_parallel: true }}, {{ modelFamily: 'z_image', pipelineParallel: true }}),
  convrotOff: state('convrot_scope', {{ base_compute: 'bf16' }}, {{ baseCompute: 'bf16' }}),
  convrotOn: state('convrot_scope', {{ base_compute: 'w8a16_convrot' }}, {{ baseCompute: 'w8a16_convrot' }}),
  compileOff: state('compile_block_scope', {{ torch_compile: false }}),
  compileOn: state('compile_block_scope', {{ torch_compile: true }}),
  bandsOff: state('compile_seq_bands', {{ torch_compile: true, compile_dynamic_seq: false }}),
  bandsOn: state('compile_seq_bands', {{ torch_compile: true, compile_dynamic_seq: true }}),
  selectiveOff: state('selective_checkpoint_blocks', {{ selective_checkpoint: 'off' }}),
  selectiveOn: state('selective_checkpoint_blocks', {{ selective_checkpoint: 'every_other' }}),
  swapOff: state('block_swap_restore_mode', {{ blocks_to_swap: 0 }}),
  swapOn: state('block_swap_restore_mode', {{ blocks_to_swap: 12 }}),
  probeOff: state('gradient_flow_probe_dense_steps', {{ gradient_flow_probe_jsonl: 'off' }}),
  probeOn: state('gradient_flow_probe_dense_steps', {{ gradient_flow_probe_jsonl: 'auto' }}),
  vaeReuseOff: state('reuse_vae_latents', {{ use_vae_cache: false }}),
  vaeReuseOn: state('reuse_vae_latents', {{ use_vae_cache: true }}),
  textReuseOff: state('reuse_text_encoder_cache', {{ use_text_cache: false }}),
  textReuseOn: state('reuse_text_encoder_cache', {{ use_text_cache: true }}),
  unknown: state('__future_runtime_field__', {{}}),
}}));
""")

    expected_disabled_codes = {
        "pipelineOff": "pipeline-parallel-disabled",
        "pipelineUnwired": "pipeline-parallel-runtime-unavailable",
        "compileOff": "torch-compile-disabled",
        "bandsOff": "compile-dynamic-seq-disabled",
        "selectiveOff": "selective-checkpoint-disabled",
        "swapOff": "block-swap-disabled",
        "probeOff": "probe-disabled",
        "vaeReuseOff": "vae-cache-disabled",
        "textReuseOff": "text-cache-disabled",
    }
    for key, code in expected_disabled_codes.items():
        assert payload[key]["presentation"]["visible"] is True
        assert payload[key]["availability"]["enabled"] is False
        assert payload[key]["availability"]["code"] == code

    for key in (
        "pipelineOn",
        "compileOn",
        "bandsOn",
        "selectiveOn",
        "swapOn",
        "probeOn",
        "vaeReuseOn",
        "textReuseOn",
    ):
        assert payload[key]["presentation"]["visible"] is True
        assert payload[key]["availability"]["enabled"] is True

    assert payload["convrotOff"]["presentation"]["visible"] is False
    assert payload["convrotOff"]["presentation"]["code"] == "feature-disabled"
    assert payload["convrotOn"]["presentation"]["visible"] is True
    assert payload["unknown"]["presentation"]["visible"] is False
    assert payload["unknown"]["presentation"]["code"] == "audit-only"


def test_hidden_adapter_drafts_remain_in_patch_and_preserve_network_args() -> None:
    module_uri = (NEXT_DOMAIN / "config-values.js").resolve().as_uri()
    payload = _node_json(f"""
const mod = await import({json.dumps(module_uri + '?hidden-draft')});
const baselineValues = {{
  lokr_factor: 8,
  learning_rate: 0.0001,
  network_args: ['lokr_use_einsum=true', 'unrelated_flag=keep'],
}};
const draftValues = {{
  lokr_factor: 16,
  lokr_use_einsum: false,
  learning_rate: 0.0001,
}};
const raw = mod.collectConfigDraftChanges({{
  scopeKeys: ['lokr_factor', 'lokr_use_einsum', 'learning_rate'],
  baselineValues,
  draftValues,
}});
console.log(JSON.stringify({{ raw, patch: mod.prepareConfigPatch(raw, baselineValues) }}));
""")

    assert payload["raw"] == {"lokr_factor": 16, "lokr_use_einsum": False}
    assert payload["patch"]["lokr_factor"] == 16
    assert payload["patch"]["network_args"] == [
        "lokr_use_einsum=false",
        "unrelated_flag=keep",
    ]


def test_shared_preview_and_method_fields_are_not_disabled_by_overlapping_sets() -> None:
    module_uri = (NEXT_DOMAIN / "config-field-availability.js").resolve().as_uri()
    payload = _node_json(f"""
const mod = await import({json.dumps(module_uri + '?shared-method-fields')});
const check = (key, method) => mod.configFieldAvailability(key, {{
  method, adapter: 'lora', modelFamily: 'anima', baseCompute: 'bf16',
}});
console.log(JSON.stringify({{
  seed: check('seed', 'lora'),
  channelOrtho: check('channel_scaling_alpha', 'ortholora'),
  channelChimera: check('channel_scaling_alpha', 'chimera'),
  channelSpd: check('channel_scaling_alpha', 'spd'),
}}));
""")

    assert all(item["enabled"] is True for item in payload.values())
