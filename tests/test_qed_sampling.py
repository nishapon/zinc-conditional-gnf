import pytest
import torch

from zinc_gnf.data.sampling import (
    capped_qed_sample_weights,
    make_capped_qed_sampler,
)


def test_rare_qed_bins_receive_more_weight():
    qed = torch.tensor(
        [0.82] * 100
        + [0.52] * 20
        + [0.22] * 5
    )

    weights = capped_qed_sample_weights(
        qed,
        bins=20,
        power=0.5,
        max_weight=4.0,
    )

    common = weights[:100].mean()
    middle = weights[100:120].mean()
    rare = weights[120:].mean()

    assert rare > middle > common


def test_weights_are_capped():
    qed = torch.tensor(
        [0.82] * 1000 + [0.12]
    )

    weights = capped_qed_sample_weights(
        qed,
        max_weight=4.0,
    )

    assert float(weights.min()) >= 1.0
    assert float(weights.max()) <= 4.0


def test_sampler_is_reproducible():
    qed = [0.2, 0.4, 0.6, 0.8] * 10

    first_generator = torch.Generator()
    first_generator.manual_seed(42)

    second_generator = torch.Generator()
    second_generator.manual_seed(42)

    first = list(
        make_capped_qed_sampler(
            qed,
            generator=first_generator,
        )
    )
    second = list(
        make_capped_qed_sampler(
            qed,
            generator=second_generator,
        )
    )

    assert first == second


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bins": 1},
        {"power": 0.0},
        {"power": 1.1},
        {"max_weight": 0.5},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        capped_qed_sample_weights(
            [0.4, 0.8],
            **kwargs,
        )
