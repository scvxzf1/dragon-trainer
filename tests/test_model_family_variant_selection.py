"""Network variants are selectable independently of model-family allowlists."""

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from library.models.family_registry import MODEL_FAMILY_REGISTRY


ROOT = Path(__file__).resolve().parents[1]
NEXT_DOMAIN = ROOT / "web/frontend-next/src/features/training-config/domain"
VARIANT_TOKEN = "auto-block-swap-20260908-v3"


@pytest.mark.parametrize("family", [name for name, spec in MODEL_FAMILY_REGISTRY.items() if not spec.plain_lora_only])
def test_registered_models_have_no_blanket_network_variant_gate(family):
    spec = MODEL_FAMILY_REGISTRY[family]
    assert spec.supported_network_specs is None
    assert spec.plain_lora_only is False


@pytest.mark.parametrize("stale_catalog", [False, True])
def test_variant_options_and_selected_parameters_are_open_for_every_model(stale_catalog):
    if not shutil.which("node"):
        pytest.skip("node is required for frontend checks")
    family_uri = (NEXT_DOMAIN / "model-family.js").as_uri()
    availability_uri = (NEXT_DOMAIN / "config-field-availability.js").as_uri()
    disclosure_uri = (NEXT_DOMAIN / "config-field-disclosure-rules.js").as_uri()
    script = f"""
const family = await import({json.dumps(family_uri + '?v=' + VARIANT_TOKEN)});
const availability = await import({json.dumps(availability_uri)});
const disclosure = await import({json.dumps(disclosure_uri)});
const names = ['anima', 'krea2_raw', 'z_image'];
if ({json.dumps(stale_catalog)}) {{
    family.configureModelFamilyCapabilities(names.map(name => ({{
        ...family.modelFamilyCapability(name),
        plain_lora_only: true,
        supported_network_specs: ['lora'],
        supports_method_adapters: false,
    }})));
}}
const choices = {{
    lora_adapter_kind: ['lora', 'loha', 'lokr', 'glora', 'vera'],
    network_module: ['networks.lora_anima', 'networks.ortho_lora', 'networks.reft'],
    use_moe_style: [false, 'shared_A', 'independent_A'],
    router_source: ['none', 'input', 'sigma', 'fei', 'crossattn_emb'],
}};
const branches = [
    ['lora_adapter_kind', 'lora', 'lora', {{}}],
    ['network_module', 'lora', 'lora', {{}}],
    ['network_args', 'lora', 'lora', {{}}],
    ['dora_wd', 'lora', 'lora', {{}}],
    ['lokr_factor', 'lora', 'lokr', {{}}],
    ['vera_d_initial', 'lora', 'vera', {{}}],
    ['use_ortho', 'ortholora', 'lora', {{ use_ortho: true }}],
    ['reft_dim', 'reft', 'lora', {{ add_reft: true }}],
    ['route_per_layer', 'hydralora', 'lora', {{ use_moe_style: 'shared_A' }}],
];
console.log(JSON.stringify(names.map(modelFamily => ({{
    modelFamily,
    choices: Object.fromEntries(Object.entries(choices).map(([key, options]) => [
        key, family.modelFamilySelectOptions(key, modelFamily, options, options[0]),
    ])),
    branches: branches.map(([key, method, adapter, values]) => {{
        const context = {{ modelFamily, method, adapter, values }};
        return {{ key, ...availability.configFieldAvailability(key, context),
            visible: disclosure.configFieldDisclosure(key, context).visible }};
    }}),
    flash: family.modelFamilyOptionSupported('attn_mode', modelFamily, 'flash'),
    invalidAttention: family.modelFamilyOptionSupported('attn_mode', modelFamily, 'not_a_backend'),
}}))));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=20,
    )
    for row in json.loads(result.stdout):
        assert [item["value"] for item in row["choices"]["lora_adapter_kind"]] == [
            "lora", "loha", "lokr", "glora", "vera",
        ]
        assert len(row["choices"]["network_module"]) == 3
        assert len(row["choices"]["use_moe_style"]) == 3
        assert len(row["choices"]["router_source"]) == 5
        assert all(item["supported"] for options in row["choices"].values() for item in options)
        for branch in row["branches"]:
            assert branch["enabled"], (row["modelFamily"], branch)
            assert branch["visible"], (row["modelFamily"], branch)
        assert row["flash"] is True
        assert row["invalidAttention"] is False


@pytest.mark.parametrize("stale_catalog", [False, True])
def test_qwen_image_2_1_task_gates_network_variants(stale_catalog):
    if not shutil.which("node"):
        pytest.skip("node is required for frontend checks")
    family_uri = (NEXT_DOMAIN / "model-family.js").as_uri()
    script = f"""
const family = await import({json.dumps(family_uri + '?v=qwen-image-21-v2')});
if ({json.dumps(stale_catalog)}) {{
    family.configureModelFamilyCapabilities([{{
        ...family.modelFamilyCapability('qwen_image_2_1'),
        supported_network_specs: ['lora', 'loha', 'lokr'],
        plain_lora_only: true,
    }}]);
}}
const choices = ['lora', 'loha', 'lokr', 'glora', 'vera'];
const select = (task) => family.modelFamilySelectOptions(
    'lora_adapter_kind', 'qwen_image_2_1', choices, 'lora',
    {{ values: {{ qwen_image_2_1_task: task }} }},
);
const context = (task) => ({{ values: {{ qwen_image_2_1_task: task }} }});
const t2i = select('t2i');
const edit = select('edit');
family.configureModelFamilyCapabilities([{{
    ...family.modelFamilyCapability('qwen_image_2_1'),
    supported_network_specs: undefined,
    plain_lora_only: true,
}}]);
const legacyT2I = select('t2i');
console.log(JSON.stringify({{
    t2i,
    edit,
    legacyT2I,
    moduleEdit: family.modelFamilySelectOptions('network_module', 'qwen_image_2_1', ['networks.lora_anima', 'networks.reft'], 'networks.lora_anima', context('edit')),
    attention: family.modelFamilySelectOptions('attn_mode', 'qwen_image_2_1', ['torch', 'sdpa', 'flash'], 'torch', context('edit')),
    doraEdit: family.modelFamilyOptionSupported('dora_wd', 'qwen_image_2_1', true, context('edit')),
    routeEdit: family.modelFamilyOptionSupported('router_source', 'qwen_image_2_1', 'fei', context('edit')),
}}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=20,
    )
    selected = json.loads(result.stdout)
    assert [item["value"] for item in selected["t2i"]] == ["lora", "loha", "lokr"]
    assert [item["value"] for item in selected["edit"]] == ["lora"]
    assert [item["value"] for item in selected["legacyT2I"]] == ["lora"]
    assert [item["value"] for item in selected["moduleEdit"]] == ["networks.lora_anima"]
    assert [item["value"] for item in selected["attention"]] == ["torch", "sdpa", "flash"]
    assert selected["doraEdit"] is False
    assert selected["routeEdit"] is False


def test_ui_constraints_gate_hard_compile_and_alias_options() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for frontend checks")
    family_uri = (NEXT_DOMAIN / "model-family.js").as_uri()
    availability_uri = (NEXT_DOMAIN / "config-field-availability.js").as_uri()
    script = f"""
const family = await import({json.dumps(family_uri + '?v=ui-constraints')});
const availability = await import({json.dumps(availability_uri)});
console.log(JSON.stringify({{
    zCompileOff: availability.configFieldAvailability('torch_compile', {{ modelFamily: 'z_image', values: {{ torch_compile: false }} }}),
    zCompileOnEditable: availability.configFieldAvailability('torch_compile', {{ modelFamily: 'z_image', values: {{ torch_compile: true }} }}),
    zResidualDynamicSeqEditable: availability.configFieldAvailability('compile_dynamic_seq', {{ modelFamily: 'z_image', values: {{ torch_compile: false, compile_dynamic_seq: true }} }}),
    qwenResidualBandsEditable: availability.configFieldAvailability('compile_seq_bands', {{ modelFamily: 'qwen_image_2_1', values: {{ torch_compile: true, compile_dynamic_seq: false, compile_seq_bands: true }} }}),
    qwenResidualInductorEditable: availability.configFieldAvailability('compile_inductor_mode', {{ modelFamily: 'qwen_image_2_1', values: {{ torch_compile: false, compile_inductor_mode: 'reduce-overhead' }} }}),
    zNativeAttn: family.modelFamilyOptionSupported('attn_mode', 'z_image', 'native'),
    qwenNativeAttn: family.modelFamilyOptionSupported('attn_mode', 'qwen_image_2_1', 'native'),
    kreaCudnnAttn: family.modelFamilyOptionSupported('attn_mode', 'krea2_raw', 'cudnn'),
    qwenAutoSwapOff: availability.configFieldAvailability('auto_block_swap', {{ modelFamily: 'qwen_image_2_1', values: {{ auto_block_swap: false }} }}),
}}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=20,
    )
    payload = json.loads(result.stdout)
    assert payload["zCompileOff"]["enabled"] is False
    assert payload["zCompileOff"]["code"] == "family-torch-compile-unsupported"
    assert payload["zCompileOnEditable"]["enabled"] is True
    assert payload["zResidualDynamicSeqEditable"]["enabled"] is True
    assert payload["qwenResidualBandsEditable"]["enabled"] is True
    assert payload["qwenResidualInductorEditable"]["enabled"] is True
    assert payload["zNativeAttn"] is True
    assert payload["qwenNativeAttn"] is True
    assert payload["kreaCudnnAttn"] is True
    assert payload["qwenAutoSwapOff"]["enabled"] is False
