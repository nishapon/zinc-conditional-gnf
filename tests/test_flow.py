"""Correctness tests for the conditional node-level GNF."""

from __future__ import annotations

import torch

from zinc_gnf.models.flow import (
    ConditionalNodeGNF,
    build_flow_from_config,
)


def _inputs(
    *,
    dtype=torch.float32,
):
    torch.manual_seed(10)

    embeddings = torch.randn(
        3,
        5,
        8,
        dtype=dtype,
    )
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )
    condition = torch.tensor(
        [
            [-1.0, -0.5],
            [0.0, 0.0],
            [1.0, 0.5],
        ],
        dtype=dtype,
    )

    embeddings = (
        embeddings
        * mask.unsqueeze(-1).to(dtype)
    )

    return embeddings, condition, mask


def _small_flow(
    *,
    dtype=torch.float32,
):
    torch.manual_seed(42)

    return ConditionalNodeGNF(
        embedding_dim=8,
        condition_dim=2,
        hidden_dim=32,
        context_dim=16,
        num_heads=4,
        num_blocks=3,
        max_log_scale=0.25,
    ).to(dtype=dtype)


def _make_non_identity(
    model,
):
    torch.manual_seed(123)

    with torch.no_grad():
        model.prior_head.weight.normal_(
            mean=0.0,
            std=0.03,
        )
        model.prior_head.bias.normal_(
            mean=0.0,
            std=0.03,
        )

        for block in model.blocks:
            block.actnorm.log_scale.normal_(
                mean=0.0,
                std=0.03,
            )
            block.actnorm.shift.normal_(
                mean=0.0,
                std=0.03,
            )

            for coupling in (
                block.first_coupling,
                block.second_coupling,
            ):
                coupling.film.weight.normal_(
                    mean=0.0,
                    std=0.03,
                )
                coupling.output_projection.weight.normal_(
                    mean=0.0,
                    std=0.03,
                )
                coupling.output_projection.bias.normal_(
                    mean=0.0,
                    std=0.03,
                )


def test_initial_flow_is_identity_with_standard_normal_prior():
    model = _small_flow()
    embeddings, condition, mask = _inputs()

    latent, logdet = model(
        embeddings,
        condition,
        mask,
    )
    recovered, inverse_logdet = model.inverse(
        latent,
        condition,
        mask,
    )
    mean, sigma = model.prior_parameters(
        condition
    )

    torch.testing.assert_close(
        latent,
        embeddings,
    )
    torch.testing.assert_close(
        recovered,
        embeddings,
    )
    torch.testing.assert_close(
        logdet,
        torch.zeros_like(logdet),
    )
    torch.testing.assert_close(
        inverse_logdet,
        torch.zeros_like(
            inverse_logdet
        ),
    )
    torch.testing.assert_close(
        mean,
        torch.zeros_like(mean),
    )
    torch.testing.assert_close(
        sigma,
        torch.ones_like(sigma),
    )


def test_nonidentity_flow_is_exactly_invertible():
    model = _small_flow()
    _make_non_identity(model)

    embeddings, condition, mask = _inputs()

    latent, forward_logdet = model(
        embeddings,
        condition,
        mask,
    )
    recovered, inverse_logdet = model.inverse(
        latent,
        condition,
        mask,
    )

    torch.testing.assert_close(
        recovered,
        embeddings,
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        forward_logdet + inverse_logdet,
        torch.zeros_like(forward_logdet),
        atol=2e-5,
        rtol=2e-5,
    )


def test_analytical_logdet_matches_numerical_jacobian():
    torch.manual_seed(50)

    model = ConditionalNodeGNF(
        embedding_dim=4,
        condition_dim=2,
        hidden_dim=16,
        context_dim=8,
        num_heads=2,
        num_blocks=2,
        max_log_scale=0.2,
    ).double()

    _make_non_identity(model)

    embeddings = torch.randn(
        1,
        2,
        4,
        dtype=torch.float64,
    )
    condition = torch.tensor(
        [[0.3, -0.4]],
        dtype=torch.float64,
    )
    mask = torch.ones(
        1,
        2,
        dtype=torch.bool,
    )

    _, analytical_logdet = model(
        embeddings,
        condition,
        mask,
    )

    def flattened_forward(flattened):
        output, _ = model(
            flattened.reshape(1, 2, 4),
            condition,
            mask,
        )
        return output.reshape(-1)

    jacobian = torch.autograd.functional.jacobian(
        flattened_forward,
        embeddings.reshape(-1),
    )

    sign, numerical_logdet = (
        torch.linalg.slogdet(jacobian)
    )

    assert sign.abs().item() == 1.0

    torch.testing.assert_close(
        analytical_logdet[0],
        numerical_logdet,
        atol=1e-8,
        rtol=1e-8,
    )


def test_node_permutation_equivariance():
    model = _small_flow()
    _make_non_identity(model)

    embeddings, condition, mask = _inputs()

    original_latent, original_logdet = model(
        embeddings,
        condition,
        mask,
    )

    permutation = torch.tensor(
        [2, 0, 4, 1, 3]
    )

    permuted_latent, permuted_logdet = model(
        embeddings[:, permutation],
        condition,
        mask[:, permutation],
    )

    torch.testing.assert_close(
        permuted_latent,
        original_latent[:, permutation],
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        permuted_logdet,
        original_logdet,
        atol=2e-5,
        rtol=2e-5,
    )


def test_padding_does_not_change_real_nodes_or_likelihood():
    model = _small_flow()
    _make_non_identity(model)

    embeddings = torch.randn(2, 3, 8)
    condition = torch.tensor(
        [
            [-0.5, -0.2],
            [0.5, 0.2],
        ]
    )
    mask = torch.ones(
        2,
        3,
        dtype=torch.bool,
    )

    original = model.log_prob(
        embeddings,
        condition,
        mask,
    )

    padded_embeddings = torch.cat(
        [
            embeddings,
            torch.randn(2, 2, 8),
        ],
        dim=1,
    )
    padded_mask = torch.cat(
        [
            mask,
            torch.zeros(
                2,
                2,
                dtype=torch.bool,
            ),
        ],
        dim=1,
    )

    padded = model.log_prob(
        padded_embeddings,
        condition,
        padded_mask,
    )

    torch.testing.assert_close(
        padded["latent"][:, :3],
        original["latent"],
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        padded["log_prob"],
        original["log_prob"],
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        padded["nll_per_dimension"],
        original["nll_per_dimension"],
        atol=2e-5,
        rtol=2e-5,
    )

    assert (
        padded["latent"][:, 3:] == 0
    ).all()


def test_both_condition_fields_affect_nonidentity_flow():
    model = _small_flow()
    _make_non_identity(model)

    embeddings, condition, mask = _inputs()

    original_latent, _ = model(
        embeddings,
        condition,
        mask,
    )
    original_mean, original_sigma = (
        model.prior_parameters(condition)
    )

    for column in range(2):
        changed_condition = condition.clone()
        changed_condition[:, column] += 0.7

        changed_latent, _ = model(
            embeddings,
            changed_condition,
            mask,
        )
        changed_mean, changed_sigma = (
            model.prior_parameters(
                changed_condition
            )
        )

        total_difference = (
            (
                changed_latent
                - original_latent
            ).abs().sum()
            + (
                changed_mean
                - original_mean
            ).abs().sum()
            + (
                changed_sigma
                - original_sigma
            ).abs().sum()
        )

        assert total_difference.item() > 1e-7


def test_sampling_is_differentiable_and_masked():
    model = _small_flow()

    condition = torch.tensor(
        [
            [-0.5, -0.2],
            [0.5, 0.3],
        ]
    )
    n_node = torch.tensor([2, 4])

    generator = torch.Generator().manual_seed(
        999
    )

    generated, mask = model.sample(
        condition,
        n_node,
        generator=generator,
    )

    assert generated.shape == (2, 4, 8)
    assert mask.shape == (2, 4)
    assert mask.sum(dim=1).tolist() == [
        2,
        4,
    ]
    assert (
        generated[0, 2:] == 0
    ).all()

    loss = generated.square().mean()
    loss.backward()

    parameters_with_gradient = sum(
        parameter.grad is not None
        and torch.isfinite(
            parameter.grad
        ).all()
        and parameter.grad.abs().sum().item() > 0
        for parameter in model.parameters()
    )

    assert parameters_with_gradient > 0


def test_build_flow_from_configuration():
    model = build_flow_from_config(
        {
            "embedding_dim": 32,
            "condition_dim": 2,
            "hidden_dim": 64,
            "context_dim": 24,
            "attention_heads": 4,
            "coupling_blocks": 5,
            "max_log_scale": 0.3,
            "minimum_sigma": 0.15,
        }
    )

    assert model.embedding_dim == 32
    assert model.condition_dim == 2
    assert model.hidden_dim == 64
    assert model.context_dim == 24
    assert model.num_heads == 4
    assert model.num_blocks == 5
    assert model.max_log_scale == 0.3
    assert model.minimum_sigma == 0.15
