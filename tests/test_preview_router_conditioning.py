from types import SimpleNamespace

import torch

from library.training.preview_lifecycle import apply_preview_router_conditioning


class Network:
    cfg = SimpleNamespace(router_source="fei", fei_sigma_low_div=8.0)
    use_fei_router = True
    freq_router = None
    use_content_router = True
    use_crossattn_router = True

    def __init__(self):
        self.events = []

    def set_step_index(self, value):
        self.events.append(("step", value))

    def set_sigma(self, value):
        self.events.append(("sigma", value.detach().clone()))

    def set_fei(self, value):
        self.events.append(("fei", value.detach().clone()))

    def set_content(self, value):
        self.events.append(("content", value))

    def set_crossattn_routing(self, value):
        self.events.append(("crossattn", value))


def test_preview_router_conditioning_updates_all_sources_for_5d_latent():
    network = Network()
    text = torch.ones(1, 3, 4)
    apply_preview_router_conditioning(
        network,
        torch.randn(2, 4, 1, 8, 8),
        torch.tensor([0.25]),
        step_index=3,
        crossattn_emb=text,
    )

    assert [event[0] for event in network.events] == [
        "step", "sigma", "fei", "content", "crossattn"
    ]
    assert network.events[0][1] == 3
    assert network.events[2][1].shape == (2, 2)
    torch.testing.assert_close(network.events[2][1].sum(dim=-1), torch.ones(2))


def test_preview_router_conditioning_is_noop_without_network():
    apply_preview_router_conditioning(None, torch.zeros(1), torch.zeros(1), step_index=0)
