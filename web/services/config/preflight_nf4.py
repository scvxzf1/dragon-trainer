"""Header-only mixed NF4 preflight using the canonical loader contract."""
from library.quantization import inspect_mixed_nf4
from library.quantization.policy import FAMILIES
from library.quantization.training import (
    COMPILE_RISK_FAMILIES, COMPILE_RISK_MESSAGE, mixed_nf4_training_errors,
)


def check_mixed_nf4(cfg, path, add, *, skip_experiment_policy_errors: bool = False):
    if cfg.get('model_family') not in FAMILIES or not path.is_file():
        return
    try:
        manifest = inspect_mixed_nf4(path, family=cfg['model_family'])
        if manifest is None:
            if cfg.get('qwen_nf4_compile_experiment') is True:
                add('error', 'qwen_nf4_compile_experiment',
                    'Qwen NF4 编译实验要求基础模型为预量化 mixed-NF4 权重文件', path)
            return
        family = manifest['family']
        errors = mixed_nf4_training_errors(family, cfg, allow_compile=True)
    except Exception as exc:
        add('error', 'mixed_nf4', f'Mixed NF4 格式或训练配置不兼容: {exc}', path)
        return
    if errors:
        add('error', 'mixed_nf4', 'Mixed NF4 training requires ' + '; '.join(errors), path)
    if cfg.get('qwen_nf4_compile_experiment') is True:
        from library.quantization.qwen_nf4_compile_policy import qwen_nf4_compile_experiment_errors
        experiment_errors = qwen_nf4_compile_experiment_errors(family, cfg)
        if experiment_errors:
            if not skip_experiment_policy_errors:
                add('error', 'qwen_nf4_compile_experiment', '; '.join(experiment_errors), path)
        elif not errors:
            add('warning', 'qwen_nf4_compile_experiment',
                'Experimental Qwen NF4 compile: two-step AdamW, preview and full-state resume smoke passed; continuous/resume trajectory equivalence, other optimizers, long runs, speed, image quality and other hardware remain unverified', path)
        return
    if cfg.get('torch_compile') and family in COMPILE_RISK_FAMILIES:
        add('error', 'mixed_nf4_compile', COMPILE_RISK_MESSAGE, path,
            overridable=True, risk_code='mixed_nf4_compile')
