from __future__ import annotations


def test_krea2_nf4_probe_stamps_krea2_family(monkeypatch):
    from scripts.krea2.probe_nf4_ablation import make_network

    captured = {}

    class _Cfg:
        @classmethod
        def from_kwargs(cls, kwargs, **_options):
            captured.update(kwargs)
            return object()

    class _Network:
        def __init__(self, *_args, **_kwargs):
            pass

        def apply_to(self, **_kwargs):
            return None

    import scripts.krea2.probe_nf4_ablation as probe

    monkeypatch.setattr(probe, "LoRANetworkCfg", _Cfg)
    monkeypatch.setattr(probe, "LoRANetwork", _Network)
    make_network(object())

    assert captured["model_family"] == "krea2_raw"
