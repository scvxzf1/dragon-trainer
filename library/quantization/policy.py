"""Exact main-block MLP policy; never select arbitrary matching linears."""
from __future__ import annotations

from collections.abc import Iterator, Mapping

from library.models.family_registry import MODEL_FAMILY_REGISTRY

POLICY = 'main-mlp-nf4-v1'
ALL_LINEAR_POLICY = 'all-linear-nf4-v1'
ANIMA_SELECTIVE_POLICY = 'anima-selective-mlp-nf4-v1'


class _RegistryMixedNf4Families(Mapping[str, tuple[str, str, tuple[str, ...]]]):
    def _items(self) -> dict[str, tuple[str, str, tuple[str, ...]]]:
        return {
            name: (
                spec.topology.block_container,
                spec.topology.mixed_nf4_mlp_spec[0],
                spec.topology.mixed_nf4_mlp_spec[1],
            )
            for name, spec in MODEL_FAMILY_REGISTRY.items()
            if spec.topology.mixed_nf4_mlp_spec is not None
        }

    def __getitem__(self, key: str) -> tuple[str, str, tuple[str, ...]]:
        return self._items()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items())

    def __len__(self) -> int:
        return len(self._items())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self._items() == dict(other)
        return NotImplemented


FAMILIES: Mapping[str, tuple[str, str, tuple[str, ...]]] = _RegistryMixedNf4Families()


def policy_paths(family, count, policy=POLICY):
    if family not in FAMILIES or type(count) is not int or count <= 0:
        raise ValueError('Unsupported mixed NF4 family or empty main blocks')
    root, mlp, names = FAMILIES[family]
    if policy not in (POLICY, ANIMA_SELECTIVE_POLICY):
        raise ValueError('Unsupported mixed NF4 policy')
    if policy == ANIMA_SELECTIVE_POLICY and family != 'anima':
        raise ValueError('Anima selective NF4 policy is supported only for anima')
    paths = [f'{root}.{i}.{mlp}.{name}' for i in range(count) for name in names]
    if policy == ANIMA_SELECTIVE_POLICY and count == 28:
        paths = [path for path in paths if not (
            path.endswith('.layer1') and 14 <= int(path.split('.')[1]) <= 20
        )]
    return paths


def model_linear_paths(model):
    """Return every ordinary Linear path for the Qwen Image 2.1 transformer."""
    import torch

    paths = []
    for name, module in model.named_modules():
        if not name or type(module) is not torch.nn.Linear:
            continue
        if 'forward' in module.__dict__ or hasattr(module, 'quant_state'):
            raise ValueError(f'Model already has packed weights or adapter: {name}')
        if module.in_features % 64 or module.out_features % 64:
            raise ValueError(f'NF4 dimensions must be divisible by 64: {name}')
        paths.append(name)
    if not paths:
        raise ValueError('Qwen Image 2.1 model has no ordinary Linear modules')
    return sorted(paths)


def model_paths(model, family, policy=POLICY):
    import torch

    if family not in FAMILIES:
        raise ValueError('Unsupported mixed NF4 family')
    if policy == ALL_LINEAR_POLICY:
        spec = MODEL_FAMILY_REGISTRY.get(family)
        if spec is None or not spec.topology.supports_all_linear_nf4:
            raise ValueError('All-linear NF4 policy is supported only for qwen_image_2_1')
        paths = model_linear_paths(model)
        from bitsandbytes.nn import Linear4bit, Params4bit
        for name, module in model.named_modules():
            if (isinstance(module, Linear4bit) or isinstance(getattr(module, 'weight', None), Params4bit)
                    or 'forward' in module.__dict__ or hasattr(module, 'lora_down')
                    or hasattr(module, 'lora_up')):
                raise ValueError(f'Model already has packed weights or adapter: {name}')
        return paths
    if policy == ANIMA_SELECTIVE_POLICY:
        if family != 'anima':
            raise ValueError('Anima selective NF4 policy is supported only for anima')
    elif policy != POLICY:
        raise ValueError('Unsupported mixed NF4 policy')
    root = FAMILIES[family][0]
    blocks = getattr(model, root, None)
    if not isinstance(blocks, (torch.nn.ModuleList, torch.nn.Sequential)):
        raise ValueError('Expected complete main block collection')
    paths = policy_paths(family, len(blocks), policy)
    for path in paths:
        try:
            linear = model.get_submodule(path)
        except AttributeError as exc:
            raise ValueError(f'Missing policy module: {path}') from exc
        if type(linear) is not torch.nn.Linear or 'forward' in linear.__dict__:
            raise ValueError(f'Policy target must be ordinary unpatched Linear: {path}')
        if linear.in_features % 64 or linear.out_features % 64:
            raise ValueError(f'NF4 dimensions must be divisible by 64: {path}')
    from bitsandbytes.nn import Linear4bit, Params4bit
    for name, module in model.named_modules():
        if (isinstance(module, Linear4bit) or isinstance(getattr(module, 'weight', None), Params4bit)
                or 'forward' in module.__dict__ or hasattr(module, 'quant_state')
                or hasattr(module, 'lora_down') or hasattr(module, 'lora_up')):
            raise ValueError(f'Model already has packed weights or adapter: {name}')
    return paths
