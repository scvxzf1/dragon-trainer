"""CPU-only format/lifecycle checks; no real model weights or GPU required."""
import json
from types import SimpleNamespace
import pytest
import torch
from safetensors.torch import save_file, load_file
from bitsandbytes.functional import quantize_4bit
from library.anima.mixed_nf4 import conversion, format as fmt, runtime


def packed_fixture():
    torch.manual_seed(42)
    packed, q = quantize_4bit(torch.randn(64,64,dtype=torch.bfloat16), blocksize=64,
                              compress_statistics=True, quant_type='nf4')
    return packed, q.as_dict(packed=True)


def test_fixed_policy():
    paths = fmt.quantized_paths()
    assert len(paths) == 49 and len(set(paths)) == 49
    assert all(f'blocks.{i}.mlp.layer2' in paths for i in range(28))
    assert [i for i in range(28) if f'blocks.{i}.mlp.layer1' not in paths] == list(range(14,21))


def test_quantstate_exact_roundtrip_and_input_grad_with_plain_lora(tmp_path):
    from networks.lora_modules.lora import LoRAModule
    packed, qs = packed_fixture()
    state = {'w.' + k:v for k,v in qs.items()}
    _, q = fmt.quant_state(state, 'w', (64,64))
    for k,v in q.as_dict(packed=True).items():
        assert torch.equal(v, qs[k])
    path = tmp_path/'tiny.safetensors'
    save_file({'weight': packed, **state}, str(path))
    saved = load_file(str(path))
    restored_qs = {k: saved['w.'+k] for k in qs}
    before = runtime.restore_linear(packed, qs, in_features=64, out_features=64)
    after = runtime.restore_linear(saved['weight'], restored_qs, in_features=64,out_features=64)
    x = torch.randn(3,64,dtype=torch.bfloat16,requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    y, yr = before(x), after(xr)
    torch.testing.assert_close(y,yr,rtol=0,atol=0)
    y.float().square().mean().backward();yr.float().square().mean().backward()
    torch.testing.assert_close(x.grad,xr.grad,rtol=0,atol=0)
    adapter = LoRAModule('tiny',after,lora_dim=4,alpha=4).bfloat16()
    adapter.apply_to()
    torch.testing.assert_close(after(xr.detach()),yr.detach(),rtol=0,atol=0)
    with torch.no_grad(): adapter.lora_up.weight.normal_(std=.01)
    after(xr.detach()).float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
    after.to('cpu', dtype=torch.bfloat16)
    assert after.weight.dtype == torch.uint8 and after.compute_dtype == torch.bfloat16


@pytest.mark.parametrize('mutation', ['missing','fp4','blocksize','dtype','shape'])
def test_bad_quantstate_rejected(mutation):
    _, qs = packed_fixture()
    if mutation == 'missing': del qs['nested_absmax']
    else:
        from bitsandbytes.functional import unpack_tensor_to_dict, pack_dict_to_tensor
        meta = unpack_tensor_to_dict(qs['quant_state.bitsandbytes__nf4'])
        meta[{'fp4':'quant_type','blocksize':'blocksize','dtype':'dtype','shape':'shape'}[mutation]] = {
            'fp4':'fp4','blocksize':128,'dtype':'float16','shape':[32,128]}[mutation]
        qs['quant_state.bitsandbytes__nf4'] = pack_dict_to_tensor(meta)
    with pytest.raises(ValueError): fmt.quant_state({'w.'+k:v for k,v in qs.items()},'w',(64,64))


@pytest.mark.parametrize('mutation', [
    'quant_map_zero', 'nested_map_zero', 'negative_scale', 'quant_map_dtype',
    'nested_map_dtype', 'scale_dtype', 'quant_map_shape', 'nested_map_shape',
])
def test_corrupted_quantstate_contents_rejected(mutation):
    _, qs = packed_fixture()
    if mutation == 'quant_map_zero':
        qs['quant_map'].zero_()
    elif mutation == 'nested_map_zero':
        qs['nested_quant_map'].zero_()
    elif mutation == 'negative_scale':
        qs['nested_absmax'].fill_(-1)
    elif mutation == 'quant_map_dtype':
        qs['quant_map'] = qs['quant_map'].bfloat16()
    elif mutation == 'nested_map_dtype':
        qs['nested_quant_map'] = qs['nested_quant_map'].bfloat16()
    elif mutation == 'scale_dtype':
        qs['nested_absmax'] = qs['nested_absmax'].bfloat16()
    elif mutation == 'quant_map_shape':
        qs['quant_map'] = qs['quant_map'].reshape(4, 4)
    elif mutation == 'nested_map_shape':
        qs['nested_quant_map'] = qs['nested_quant_map'].reshape(16, 16)
    with pytest.raises(ValueError):
        fmt.quant_state({'w.' + k: v for k, v in qs.items()}, 'w', (64, 64))


def test_normal_and_malformed_format_fail_closed(tmp_path):
    p = tmp_path/'a.safetensors'
    save_file({'net.x':torch.ones(1,dtype=torch.bfloat16)},str(p))
    assert fmt.inspect_mixed_nf4(p) is None
    for raw in ['{','{}',json.dumps({'version':99})]:
        save_file({'net.x':torch.ones(1)},str(p),metadata={fmt.MANIFEST_KEY:raw})
        with pytest.raises(ValueError):fmt.inspect_mixed_nf4(p)
    save_file({'net.x.weight.absmax':torch.ones(1)},str(p))
    with pytest.raises(ValueError):fmt.inspect_mixed_nf4(p)


@pytest.mark.parametrize('dtype,merge',[(torch.float16,None),(torch.float32,None),(torch.bfloat16,[{}])])
def test_runtime_rejects_dtype_and_merge_before_read(dtype,merge):
    with pytest.raises(ValueError): runtime.load_mixed_state(None,'absent',dtype=dtype,
        lora_weights_list=merge,rename_hook=None,concat_hook=None,device='cpu')


def test_conversion_no_overwrite(tmp_path):
    p = tmp_path/'exists.safetensors';p.write_bytes(b'preserve')
    with pytest.raises(FileExistsError):conversion.convert_checkpoint('absent',p)
    assert p.read_bytes() == b'preserve'


def test_source_header_is_standard_bf16_without_allocations():
    class Slice:
        def __init__(self,shape):self.shape=shape
        def get_dtype(self):return 'BF16'
        def get_shape(self):return self.shape
    shapes = {f'net.blocks.{i}.mlp.layer{l}.weight':([8192,2048] if l==1 else [2048,8192])
              for i in range(28) for l in (1,2)}
    shapes.update({f'net.extra.{i}':[1] for i in range(629)})
    fake=SimpleNamespace(keys=lambda:shapes.keys(),get_slice=lambda k:Slice(shapes[k]))
    prefix, result=conversion.validate_source(fake)
    assert prefix=='net.' and result==shapes
    shapes['net.blocks.0.mlp.layer1.weight']=[4096,2048]
    with pytest.raises(ValueError):conversion.validate_source(fake)


def test_conversion_small_fixture_preserves_metadata_and_non_targets(tmp_path,monkeypatch):
    # Production structural validation stays strict; only this fixture substitutes tiny headers.
    from library.anima import checkpoint
    source,tmp = tmp_path/'source.safetensors',tmp_path/'output.safetensors'
    original={'net.blocks.0.mlp.layer1.weight':torch.randn(64,64,dtype=torch.bfloat16),
              'net.non_target':torch.randn(16,dtype=torch.bfloat16)}
    save_file(original,str(source),metadata={'original':'kept'})
    monkeypatch.setattr(checkpoint,'inspect_anima_checkpoint',lambda _:SimpleNamespace(num_blocks=28,model_channels=2048))
    monkeypatch.setattr(conversion,'validate_source',lambda _:('net.',{k:list(v.shape) for k,v in original.items()}))
    monkeypatch.setattr(conversion,'validate_model_shapes',lambda _:None)
    monkeypatch.setattr(conversion,'quantized_paths',lambda:['blocks.0.mlp.layer1'])
    def verify(path,base):
        from safetensors import safe_open
        state=load_file(path)
        assert torch.equal(state['net.non_target'].view(torch.uint8),original['net.non_target'].view(torch.uint8))
        with safe_open(path,framework='pt') as f:
            metadata = f.metadata()
            assert metadata['original']=='kept'
            assert fmt.MANIFEST_KEY not in metadata
            assert 'diffusion.mixed_nf4.manifest' in metadata
            manifest = json.loads(metadata['diffusion.mixed_nf4.manifest'])
            assert manifest['policy'] == 'anima-selective-mlp-nf4-v1'
            assert manifest['prefix'] == 'net.'
        return {'bytes':__import__('pathlib').Path(path).stat().st_size}
    monkeypatch.setattr(conversion,'verify_checkpoint',verify)
    result=conversion.convert_checkpoint(source,tmp)
    assert tmp.exists() and result['source_sha256']==fmt.sha256(source)
    assert result['output_sha256']==fmt.sha256(tmp)


def test_runtime_integrates_packed_module_and_existing_fusion_hooks(tmp_path,monkeypatch):
    from library.anima.weights import _dit_rename_hook,_dit_concat_hook
    p,qs=packed_fixture();path=tmp_path/'runtime.safetensors'
    block=torch.nn.Module();block.mlp=torch.nn.Module()
    block.mlp.layer1=torch.nn.Linear(64,64,bias=False,device='meta')
    block.self_attn=torch.nn.Module()
    block.self_attn.qkv_proj=torch.nn.Linear(64,192,bias=False,device='meta')
    model=torch.nn.Module();model.blocks=torch.nn.ModuleList([block])
    w='net.blocks.0.mlp.layer1.weight'
    state={w:p,**{w+'.'+k:v for k,v in qs.items()}}
    projections={f'net.blocks.0.self_attn.{k}_proj.weight':torch.randn(64,64,dtype=torch.bfloat16)
                 for k in ('q','k','v')}
    state.update(projections);save_file(state,str(path))
    monkeypatch.setattr(runtime,'inspect_mixed_nf4',lambda _: {'modules':['blocks.0.mlp.layer1'],
        'prefix':'net.','tensor_shapes':{w:[64,64]}})
    assert runtime.load_mixed_state(model,path,dtype=torch.bfloat16,lora_weights_list=None,
        rename_hook=_dit_rename_hook,concat_hook=_dit_concat_hook,device='cpu') == ([],[])
    assert model.blocks[0].mlp.layer1.weight.dtype==torch.uint8
    torch.testing.assert_close(block.self_attn.qkv_proj.weight,torch.cat(list(projections.values())),rtol=0,atol=0)
    x=torch.randn(2,64,dtype=torch.bfloat16,requires_grad=True)
    model.blocks[0].mlp.layer1(x).float().sum().backward()
    assert torch.isfinite(x.grad).all()
