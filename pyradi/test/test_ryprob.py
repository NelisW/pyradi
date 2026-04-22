"""Regression tests for ryprob.py — probability utility functions."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import ryprob as rp


# ── checkParamsNum ────────────────────────────────────────────────────────────

class TestCheckParamsNum:
    def test_correct_single_allowed_count_returns_true(self):
        assert rp.checkParamsNum('fn', 'Dist', 'dist', [1.0], [1]) is True

    def test_correct_one_of_several_allowed_counts(self):
        assert rp.checkParamsNum('fn', 'Dist', 'dist', [1.0, 2.0], [0, 2]) is True

    def test_zero_params_allowed(self):
        assert rp.checkParamsNum('fn', 'Dist', 'dist', [], [0]) is True

    def test_wrong_count_returns_false(self, capsys):
        result = rp.checkParamsNum('fn', 'Dist', 'dist', [1.0, 2.0, 3.0], [1, 2])
        assert result is False

    def test_wrong_count_prints_message(self, capsys):
        rp.checkParamsNum('fn', 'Dist', 'dist', [1.0, 2.0, 3.0], [1])
        captured = capsys.readouterr()
        assert 'Wrong number of parameters' in captured.out


# ── validateParam ─────────────────────────────────────────────────────────────

class TestValidateParam:
    def test_greater_than_passes(self):
        assert rp.validateParam('fn', 'D', 'rd', 'p', 'x', 5.0, ['> 0']) is True

    def test_greater_than_fails(self):
        assert rp.validateParam('fn', 'D', 'rd', 'p', 'x', -1.0, ['> 0']) is False

    def test_less_than_passes(self):
        assert rp.validateParam('fn', 'D', 'rd', 'p', 'x', 3.0, ['< 10']) is True

    def test_less_than_fails(self):
        assert rp.validateParam('fn', 'D', 'rd', 'p', 'x', 15.0, ['< 10']) is False

    def test_greater_or_equal_boundary(self):
        assert rp.validateParam('fn', 'D', 'rd', 'p', 'x', 0.0, ['>= 0']) is True

    def test_less_or_equal_boundary(self):
        assert rp.validateParam('fn', 'D', 'rd', 'p', 'x', 5.0, ['<= 5']) is True

    def test_not_equal_passes(self):
        assert rp.validateParam('fn', 'D', 'rd', 'p', 'x', 3.0, ['!= 0']) is True

    def test_not_equal_fails(self):
        assert rp.validateParam('fn', 'D', 'rd', 'p', 'x', 0.0, ['!= 0']) is False

    def test_multiple_conditions_both_pass(self):
        assert rp.validateParam('fn', 'D', 'rd', 'p', 'x', 5.0, ['> 0', '<= 10']) is True

    def test_multiple_conditions_first_fails(self):
        assert rp.validateParam('fn', 'D', 'rd', 'p', 'x', -1.0, ['> 0', '<= 10']) is False

    def test_multiple_conditions_second_fails(self):
        assert rp.validateParam('fn', 'D', 'rd', 'p', 'x', 20.0, ['> 0', '<= 10']) is False

    def test_failure_prints_message(self, capsys):
        rp.validateParam('fn', 'D', 'rd', 'p', 'x', -1.0, ['> 0'])
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# ── unifomOnSphere ────────────────────────────────────────────────────────────

class TestUnifomOnSphere:
    def test_default_dimension_shape(self):
        result = rp.unifomOnSphere(10)
        assert result.shape == (10, 3)

    def test_custom_dimension_shape(self):
        result = rp.unifomOnSphere(5, dimension=4)
        assert result.shape == (5, 4)

    def test_unit_norms_3d(self):
        np.random.seed(0)
        result = rp.unifomOnSphere(50)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_unit_norms_2d(self):
        np.random.seed(1)
        result = rp.unifomOnSphere(20, dimension=2)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    def test_single_point(self):
        np.random.seed(2)
        result = rp.unifomOnSphere(1)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-12)

    def test_high_dimensional(self):
        np.random.seed(3)
        result = rp.unifomOnSphere(10, dimension=10)
        assert result.shape == (10, 10)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)


# ── distribution_exp (direct call) ───────────────────────────────────────────

class TestDistributionExp:
    def test_returns_correct_shape_1d(self):
        np.random.seed(0)
        out = np.zeros((50,))
        result = rp.distribution_exp([1.0], out, 'test')
        assert result.shape == (50,)

    def test_returns_correct_shape_2d(self):
        np.random.seed(0)
        out = np.zeros((4, 5))
        result = rp.distribution_exp([1.0], out, 'test')
        assert result.shape == (4, 5)

    def test_all_values_positive(self):
        np.random.seed(0)
        out = np.zeros((200,))
        result = rp.distribution_exp([1.0], out, 'test')
        assert (result > 0).all()

    def test_none_params_uses_default_lambda_1(self):
        np.random.seed(0)
        out = np.zeros((100,))
        result = rp.distribution_exp(None, out, 'test')
        assert result.shape == (100,)
        assert (result > 0).all()

    def test_empty_params_uses_default(self):
        np.random.seed(0)
        out = np.zeros((50,))
        result = rp.distribution_exp([], out, 'test')
        assert result.shape == (50,)
        assert (result > 0).all()

    def test_invalid_lambda_returns_zeros(self, capsys):
        """Lambda <= 0 fails validateParam and returns the unchanged zero array."""
        out = np.zeros((10,))
        result = rp.distribution_exp([-1.0], out, 'test')
        np.testing.assert_array_equal(result, np.zeros((10,)))


# ── distribution_lognormal (direct call) ─────────────────────────────────────

class TestDistributionLognormal:
    def test_shape_1d(self):
        np.random.seed(0)
        out = np.zeros((100,))
        result = rp.distribution_lognormal([0.0, 1.0], out, 'test')
        assert result.shape == (100,)

    def test_all_values_positive(self):
        np.random.seed(0)
        out = np.zeros((200,))
        result = rp.distribution_lognormal([0.0, 1.0], out, 'test')
        assert (result > 0).all()

    def test_none_params_uses_default(self):
        np.random.seed(0)
        out = np.zeros((50,))
        result = rp.distribution_lognormal(None, out, 'test')
        assert result.shape == (50,)
        assert (result > 0).all()

    def test_invalid_sigma_returns_zeros(self, capsys):
        """Sigma <= 0 fails validateParam and returns unchanged zero array."""
        out = np.zeros((10,))
        result = rp.distribution_lognormal([0.0, -1.0], out, 'test')
        np.testing.assert_array_equal(result, np.zeros((10,)))


# ── distribution_logistic (direct call) ──────────────────────────────────────

class TestDistributionLogistic:
    def test_shape_1d(self):
        np.random.seed(0)
        out = np.zeros((100,))
        result = rp.distribution_logistic([0.0, 1.0], out, 'test')
        assert result.shape == (100,)

    def test_all_finite(self):
        np.random.seed(0)
        out = np.zeros((200,))
        result = rp.distribution_logistic([0.0, 1.0], out, 'test')
        assert np.isfinite(result).all()

    def test_none_params_uses_default(self):
        np.random.seed(0)
        out = np.zeros((50,))
        result = rp.distribution_logistic(None, out, 'test')
        assert result.shape == (50,)

    def test_shape_2d(self):
        np.random.seed(0)
        out = np.zeros((4, 5))
        result = rp.distribution_logistic([0.0, 1.0], out, 'test')
        assert result.shape == (4, 5)


# ── distribution_inversegauss (direct call) ───────────────────────────────────

class TestDistributionInversegauss:
    def test_valid_params_shape(self):
        np.random.seed(0)
        out = np.zeros((100,))
        result = rp.distribution_inversegauss([1.0, 1.0], out, 'test')
        assert result.shape == (100,)

    def test_valid_params_all_positive(self):
        np.random.seed(0)
        out = np.zeros((200,))
        result = rp.distribution_inversegauss([1.0, 1.0], out, 'test')
        assert (result > 0).all()

    def test_none_params_mu_zero_fails_validation(self, capsys):
        """Default mu=0 fails > 0 check, out stays all-zeros."""
        out = np.zeros((20,))
        result = rp.distribution_inversegauss(None, out, 'test')
        np.testing.assert_array_equal(result, np.zeros((20,)))


# ── distribution_wald (direct call) ──────────────────────────────────────────

class TestDistributionWald:
    def test_valid_chi_shape(self):
        np.random.seed(0)
        out = np.zeros((30,))
        result = rp.distribution_wald([2.0], out, 'test')
        assert result.shape == (30,)

    def test_none_params_chi_zero_fails_validation(self, capsys):
        """Default chi=0 fails > 0 check, out stays all-zeros."""
        out = np.zeros((10,))
        result = rp.distribution_wald(None, out, 'test')
        np.testing.assert_array_equal(result, np.zeros((10,)))


# ── distributions_generator ───────────────────────────────────────────────────

class TestDistributionsGenerator:
    def test_none_samplesize_returns_scalar(self):
        np.random.seed(42)
        result = rp.distributions_generator('exp', [1.0], sampleSize=None)
        # sampleSize=None → out shape (1,) → out = out[0] → scalar
        assert np.isscalar(result)

    def test_int_samplesize_returns_square_array(self):
        np.random.seed(42)
        result = rp.distributions_generator('exp', [1.0], sampleSize=4)
        assert result.shape == (4, 4)

    def test_tuple_samplesize_1d(self):
        np.random.seed(42)
        result = rp.distributions_generator('exp', [1.0], sampleSize=(10,))
        assert result.shape == (10,)

    def test_tuple_samplesize_2d(self):
        np.random.seed(42)
        result = rp.distributions_generator('exp', [1.0], sampleSize=(3, 5))
        assert result.shape == (3, 5)

    def test_list_samplesize(self):
        np.random.seed(42)
        result = rp.distributions_generator('exp', [1.0], sampleSize=[2, 3])
        assert result.shape == (2, 3)

    def test_exp_name_accepted(self):
        result = rp.distributions_generator('exp', [1.0], sampleSize=(20,))
        assert result.shape == (20,)
        assert (result > 0).all()

    def test_exponential_name_accepted(self):
        result = rp.distributions_generator('exponential', [1.0], sampleSize=(20,))
        assert result.shape == (20,)

    def test_lognorm_name_accepted(self):
        result = rp.distributions_generator('lognorm', [0.0, 1.0], sampleSize=(20,))
        assert result.shape == (20,)
        assert (result > 0).all()

    def test_lognormal_name_variant(self):
        for name in ['lognormal', 'cobbdouglas', 'antilognormal']:
            result = rp.distributions_generator(name, [0.0, 1.0], sampleSize=(10,))
            assert result.shape == (10,)

    def test_ig_name_accepted(self):
        # NOTE: due to a pre-existing bug, distribParams is always passed as None
        # internally, so 'ig' defaults to mu=0 which fails > 0 → returns zeros
        result = rp.distributions_generator('ig', [1.0, 1.0], sampleSize=(10,))
        assert result.shape == (10,)

    def test_inversegauss_name_variants(self):
        for name in ['inversegauss', 'invgauss']:
            result = rp.distributions_generator(name, [1.0, 1.0], sampleSize=(5,))
            assert result.shape == (5,)

    def test_logistic_name_accepted(self):
        result = rp.distributions_generator('logistic', [0.0, 1.0], sampleSize=(20,))
        assert result.shape == (20,)
        assert np.isfinite(result).all()

    def test_wald_name_accepted(self):
        # NOTE: pre-existing bug — distribParams is not forwarded, chi defaults to 0,
        # which fails validation, so returns zeros
        result = rp.distributions_generator('wald', [2.0], sampleSize=(10,))
        assert result.shape == (10,)

    def test_unknown_name_returns_zeros(self, capsys):
        result = rp.distributions_generator('notadist', None, sampleSize=(5,))
        np.testing.assert_array_equal(result, np.zeros((5,)))

    def test_whitespace_in_name_stripped(self):
        result = rp.distributions_generator('  exp  ', [1.0], sampleSize=(5,))
        assert result.shape == (5,)

    def test_mixed_case_name_normalised(self):
        result = rp.distributions_generator('EXP', [1.0], sampleSize=(5,))
        assert result.shape == (5,)


# ── KDEbounded (requires sklearn) ─────────────────────────────────────────────

class TestKDEbounded:
    @pytest.fixture(autouse=True)
    def require_sklearn(self):
        pytest.importorskip('sklearn')

    def test_returns_four_tuple(self):
        np.random.seed(0)
        x = np.random.normal(5, 1, 50)
        x_d = np.linspace(0, 10, 100)
        result = rp.KDEbounded(x_d, x, bandwidth=0.5)
        assert len(result) == 4

    def test_output_domain_matches_input(self):
        np.random.seed(0)
        x = np.random.normal(5, 1, 50)
        x_d = np.linspace(0, 10, 100)
        out_xd, _, _, _ = rp.KDEbounded(x_d, x, bandwidth=0.5)
        np.testing.assert_array_equal(out_xd, x_d)

    def test_prob_shape_matches_domain(self):
        np.random.seed(0)
        x = np.random.normal(5, 1, 50)
        x_d = np.linspace(0, 10, 100)
        _, prob, _, _ = rp.KDEbounded(x_d, x, bandwidth=0.5)
        assert prob.shape == x_d.shape

    def test_bandwidth_returned_unchanged(self):
        np.random.seed(0)
        x = np.random.normal(5, 1, 50)
        x_d = np.linspace(0, 10, 100)
        _, _, bw, _ = rp.KDEbounded(x_d, x, bandwidth=0.5)
        assert bw == 0.5

    def test_kernel_returned_unchanged(self):
        np.random.seed(0)
        x = np.random.normal(5, 1, 50)
        x_d = np.linspace(0, 10, 100)
        _, _, _, kern = rp.KDEbounded(x_d, x, bandwidth=0.5, kernel='tophat')
        assert kern == 'tophat'

    def test_lower_bound_zeroes_at_boundary(self):
        np.random.seed(0)
        x = np.abs(np.random.normal(0, 1, 100))
        x_d = np.linspace(-1, 5, 200)
        _, prob, _, _ = rp.KDEbounded(x_d, x, bandwidth=0.3, lowbnd=0.0)
        below = prob[x_d <= 0.0]
        assert (below == 0).all()

    def test_upper_bound_zeroes_at_boundary(self):
        np.random.seed(0)
        x = np.random.uniform(0, 3, 100)
        x_d = np.linspace(0, 5, 200)
        _, prob, _, _ = rp.KDEbounded(x_d, x, bandwidth=0.3, uppbnd=3.0)
        above = prob[x_d >= 3.0]
        assert (above == 0).all()
