"""Model construction and checkpoint compatibility (requires PyTorch and timm)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from mammo_routing.config import CONDITIONS  # noqa: E402
from mammo_routing.models import build_condition, build_single  # noqa: E402

AFFINITY = np.full((4, 2), 0.75)


@pytest.mark.parametrize("key", [k for k in CONDITIONS if k != "C3_vit5bb"])
def test_conditions_forward(key):
    model = build_condition(key, AFFINITY, pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    out = model(x, true_category_idx=torch.tensor([0, 1])) if key == "C3_oracle" else model(x)
    assert out["main_logits"].shape == (2, 2)


def test_state_dict_keys_are_stable():
    keys = set(build_condition("C3_beta0", AFFINITY, pretrained=False).state_dict())
    for k in ("affinity_matrix", "temperature", "gate.0.weight", "projectors.0.0.weight",
              "finding_predictor.0.weight", "classifier.8.weight"):
        assert k in keys


def test_single_backbone():
    model = build_single("resnet50", pretrained=False).eval()
    assert model(torch.randn(2, 3, 224, 224))["main_logits"].shape == (2, 2)
