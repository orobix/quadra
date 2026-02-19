import numpy as np
import pytest
import torch

from quadra.utils.anomaly import EvalThreshold, ensure_scores_consistency, normalize_anomaly_score
from quadra.utils.tests.helpers import get_quadra_test_device


class TestEvalThreshold:
    def test_valid(self):
        et = EvalThreshold(raw=10.0, normalized=100.0)
        assert et.raw == 10.0
        assert et.normalized == 100.0


class TestEnsureScoresConsistency:
    """The invariant: (result >= eval_threshold.normalized) == (raw_score >= eval_threshold.raw).

    All inputs are deliberately inconsistent (normalized on the *wrong* side
    of the boundary) so that the function is forced to correct them.
    """

    @pytest.mark.parametrize(
        "raw_score, wrong_normalized, eval_raw, eval_norm, expected_pred",
        [
            # IS anomaly, normalized placed one step BELOW boundary
            (9.0, float(np.nextafter(np.float32(80.0), np.float32(-np.inf))), 8.0, 80.0, 1),
            # IS anomaly, raw score exactly AT eval_raw (>= is inclusive)
            (8.0, 79.9, 8.0, 80.0, 1),
            # NOT anomaly, normalized placed exactly AT boundary (not strictly below)
            (7.0, 80.0, 8.0, 80.0, 0),
            # NOT anomaly, normalized placed well above boundary
            (7.0, 95.0, 8.0, 80.0, 0),
        ],
    )
    def test_scalar_np_fp32(self, raw_score, wrong_normalized, eval_raw, eval_norm, expected_pred):
        et = EvalThreshold(raw=eval_raw, normalized=eval_norm)
        result = ensure_scores_consistency(
            np.array(wrong_normalized, dtype=np.float32),
            np.array(raw_score, dtype=np.float32),
            et,
        )
        assert int(result >= eval_norm) == expected_pred

    @pytest.mark.parametrize(
        "raw_score, wrong_normalized, eval_raw, eval_norm, expected_pred",
        [
            (9.0, 79.0, 8.0, 80.0, 1),
            (7.0, 85.0, 8.0, 80.0, 0),
        ],
    )
    def test_scalar_np_fp16(self, raw_score, wrong_normalized, eval_raw, eval_norm, expected_pred):
        et = EvalThreshold(raw=eval_raw, normalized=eval_norm)
        result = ensure_scores_consistency(
            np.array(wrong_normalized, dtype=np.float16),
            np.array(raw_score, dtype=np.float16),
            et,
        )
        assert int(result >= eval_norm) == expected_pred

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_array_np_wrong_side(self, dtype):
        """Every score is on the wrong side of the boundary so the function
        must correct all of them."""
        eval_raw, eval_norm = 8.0, 80.0
        et = EvalThreshold(raw=eval_raw, normalized=eval_norm)

        raw_scores = np.array([4.0, 7.0, 8.0, 9.0, 12.0], dtype=dtype)
        # anomaly scores (8,9,12) placed BELOW boundary; non-anomaly (4,7) placed ABOVE
        wrong_normalized = np.array([85.0, 85.0, 75.0, 75.0, 75.0], dtype=dtype)

        result = ensure_scores_consistency(wrong_normalized.copy(), raw_scores, et)

        raw_preds = (raw_scores >= eval_raw).astype(int)
        norm_preds = (result >= eval_norm).astype(int)
        np.testing.assert_array_equal(norm_preds, raw_preds)

    @pytest.mark.parametrize(
        "raw_score, wrong_normalized, eval_raw, eval_norm, expected_pred",
        [
            (9.0, float(np.nextafter(np.float32(80.0), np.float32(-np.inf))), 8.0, 80.0, 1),
            (8.0, 79.9, 8.0, 80.0, 1),
            (7.0, 80.0, 8.0, 80.0, 0),
            (7.0, 95.0, 8.0, 80.0, 0),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_scalar_torch(self, raw_score, wrong_normalized, eval_raw, eval_norm, expected_pred, dtype):
        device = get_quadra_test_device()

        et = EvalThreshold(raw=eval_raw, normalized=eval_norm)
        result = ensure_scores_consistency(
            torch.tensor(wrong_normalized, dtype=dtype, device=device),
            torch.tensor(raw_score, dtype=dtype, device=device),
            et,
        )
        assert int(result >= eval_norm) == expected_pred

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_array_torch_wrong_side(self, dtype):
        device = get_quadra_test_device()

        eval_raw, eval_norm = 8.0, 80.0
        et = EvalThreshold(raw=eval_raw, normalized=eval_norm)

        raw_scores = torch.tensor([4.0, 7.0, 8.0, 9.0, 12.0], dtype=dtype, device=device)
        wrong_normalized = torch.tensor([85.0, 85.0, 75.0, 75.0, 75.0], dtype=dtype, device=device)

        result = ensure_scores_consistency(wrong_normalized.clone(), raw_scores, et)

        raw_preds = (raw_scores >= eval_raw).int()
        norm_preds = (result >= eval_norm).int()
        assert torch.equal(norm_preds, raw_preds)

    @pytest.mark.parametrize(
        "boundary",
        [
            80.0,  # exactly representable in fp16
            80.03,  # rounds DOWN in fp16 (fp16(80.03) = 80.0 < 80.03) → ceiling needed
            99.995,  # rounds DOWN in fp16 → ceiling needed
        ],
    )
    def test_fp16_np_anomaly_clipped_to_ceiling(self, boundary):
        """IS anomaly score placed at fp16(boundary)-10 must be clipped to a value
        that is still >= boundary (float64) after the ceiling rounding."""
        et = EvalThreshold(raw=2.0, normalized=boundary)
        raw = np.array(2.0, dtype=np.float16)
        # Place normalized score well below boundary so clipping must fire
        wrong_norm = np.array(float(np.float16(boundary)) - 10.0, dtype=np.float16)
        result = ensure_scores_consistency(wrong_norm, raw, et)
        assert result >= boundary

    @pytest.mark.parametrize(
        "boundary",
        [80.0, 80.03, 99.995],
    )
    def test_fp16_np_non_anomaly_clipped_below_boundary(self, boundary):
        """NOT anomaly score placed well above boundary must be clipped to a value
        that is strictly < boundary (float64)."""
        et = EvalThreshold(raw=2.0, normalized=boundary)
        raw = np.array(0.5, dtype=np.float16)
        wrong_norm = np.array(float(np.float16(boundary)) + 10.0, dtype=np.float16)
        result = ensure_scores_consistency(wrong_norm, raw, et)
        assert result < boundary

    @pytest.mark.parametrize(
        "boundary",
        [80.0, 80.03, 99.995],
    )
    def test_fp16_torch_anomaly_clipped_to_ceiling(self, boundary):
        device = get_quadra_test_device()

        et = EvalThreshold(raw=2.0, normalized=boundary)
        raw = torch.tensor(2.0, dtype=torch.float16, device=device)
        wrong_norm = torch.tensor(float(torch.tensor(boundary, dtype=torch.float16)) - 10.0, dtype=torch.float16)
        wrong_norm = wrong_norm.to(device)
        result = ensure_scores_consistency(wrong_norm, raw, et)
        assert result >= boundary

    @pytest.mark.parametrize(
        "boundary",
        [80.0, 80.03, 99.995],
    )
    def test_fp16_torch_non_anomaly_clipped_below_boundary(self, boundary):
        device = get_quadra_test_device()

        et = EvalThreshold(raw=2.0, normalized=boundary)
        raw = torch.tensor(0.5, dtype=torch.float16, device=device)
        wrong_norm = torch.tensor(float(torch.tensor(boundary, dtype=torch.float16)) + 10.0, dtype=torch.float16)
        wrong_norm = wrong_norm.to(device)
        result = ensure_scores_consistency(wrong_norm, raw, et)
        assert result < boundary


@pytest.mark.parametrize("raw_score, threshold", [(1.345, 1.24), (1.24, 1.345)])
def test_anomaly_score_normalization_torch(raw_score: float, threshold: float):
    device = get_quadra_test_device()

    score = torch.tensor(raw_score, dtype=torch.float32, device=device)
    normalized_score = normalize_anomaly_score(score, threshold)
    np.testing.assert_allclose(normalized_score.cpu().numpy(), raw_score / threshold * 100.0)


@pytest.mark.parametrize("raw_score, threshold", [(1.345, 1.24), (1.24, 1.345)])
def test_anomaly_score_normalization_torch_with_dim(raw_score: float, threshold: float):
    device = get_quadra_test_device()
    score = torch.tensor([raw_score], dtype=torch.float32, device=device)
    normalized_score = normalize_anomaly_score(score, threshold)
    np.testing.assert_allclose(
        normalized_score.cpu().numpy(), np.array([raw_score], dtype=np.float32) / threshold * 100.0
    )


@pytest.mark.parametrize("raw_score, threshold", [(1.345, 1.24), (1.24, 1.345)])
def test_anomaly_score_normalization_torch_fp16(raw_score: float, threshold: float):
    device = get_quadra_test_device()
    score = torch.tensor(raw_score, dtype=torch.float16, device=device)
    normalized_score = normalize_anomaly_score(score, threshold)
    np.testing.assert_allclose(normalized_score.cpu().numpy(), raw_score / threshold * 100.0, rtol=1e-3)


@pytest.mark.parametrize("raw_score, threshold", [(1.345, 1.24), (1.24, 1.345)])
def test_anomaly_score_normalization_torch_fp16_with_dim(raw_score: float, threshold: float):
    device = get_quadra_test_device()
    score = torch.tensor([raw_score], dtype=torch.float16, device=device)
    normalized_score = normalize_anomaly_score(score, threshold)
    np.testing.assert_allclose(normalized_score.cpu().numpy(), raw_score / threshold * 100.0, rtol=1e-3)


@pytest.mark.parametrize("raw_score, threshold", [(1.345, 1.24), (1.24, 1.345)])
def test_anomaly_score_normalization_np(raw_score: float, threshold: float):
    score = np.array(raw_score, dtype=np.float32)
    normalized_score = normalize_anomaly_score(score, threshold)
    np.testing.assert_allclose(normalized_score, raw_score / threshold * 100.0)


@pytest.mark.parametrize("raw_score, threshold", [(1.345, 1.24), (1.24, 1.345)])
def test_anomaly_score_normalization_np_with_dim(raw_score: float, threshold: float):
    score = np.array([raw_score], dtype=np.float32)
    normalized_score = normalize_anomaly_score(score, threshold)
    np.testing.assert_allclose(normalized_score, np.array([raw_score], dtype=np.float32) / threshold * 100.0)


@pytest.mark.parametrize("raw_score, threshold", [(1.345, 1.24), (1.24, 1.345)])
def test_anomaly_score_normalization_np_fp16(raw_score: float, threshold: float):
    score = np.array(raw_score, dtype=np.float16)
    normalized_score = normalize_anomaly_score(score, threshold)
    np.testing.assert_allclose(normalized_score, np.array(raw_score, dtype=np.float16) / threshold * 100.0)


@pytest.mark.parametrize("raw_score, threshold", [(1.345, 1.24), (1.24, 1.345)])
def test_anomaly_score_normalization_np_fp16_with_dim(raw_score: float, threshold: float):
    score = np.array([raw_score], dtype=np.float16)
    normalized_score = normalize_anomaly_score(score, threshold)
    np.testing.assert_allclose(normalized_score, np.array([raw_score], dtype=np.float16) / threshold * 100.0)


@pytest.mark.parametrize("raw_score, threshold", [(1.345, 1.24), (1.24, 1.345)])
def test_anomaly_score_normalization_float(raw_score: float, threshold: float):
    normalized_score = normalize_anomaly_score(raw_score, threshold)
    np.testing.assert_allclose(normalized_score, raw_score / threshold * 100.0)


class TestNormalizeAnomalyScoreWithEvalThreshold:
    """Verify (normalized >= eval_norm) == (raw >= eval_raw) for every sample."""

    @pytest.mark.parametrize(
        "scores, training, eval_raw, eval_norm",
        [
            # eval < training: scores between eval and training boundaries are IS anomaly
            # in raw space but cross the training boundary → would be misclassified without
            # eval_threshold enforcing consistency at the right boundary
            ([4.0, 7.0, 8.0, 9.5, 12.0], 10.0, 8.0, 80.0),
            # eval > training: scores between training and eval boundaries cross the training
            # boundary but are still NOT anomaly relative to the eval threshold
            ([8.0, 9.0, 10.0, 11.0, 13.0], 10.0, 12.0, 120.0),
            # eval == training: default path, kept for non-regression
            ([8.0, 9.0, 10.0, 11.0, 12.0], 10.0, 10.0, 100.0),
            # training threshold == 0, eval == training: normalized = (raw + 1) * 100
            ([-2.0, -0.5, 0.0, 1.0, 3.0], 0.0, 0.0, 100.0),
            # training threshold == 0, eval > training: normalized = (raw + 1) * 100
            ([0.0, 0.2, 0.5, 1.0, 2.0], 0.0, 0.5, 150.0),
            # training threshold == 0, eval < training: eval_raw is negative
            ([-1.0, -0.5, 0.0, 0.5, 1.0], 0.0, -0.5, 50.0),
        ],
    )
    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_consistency_np(self, scores, training, eval_raw, eval_norm, dtype):
        et = EvalThreshold(raw=eval_raw, normalized=eval_norm)
        raw = np.array(scores, dtype=dtype)
        result = normalize_anomaly_score(raw.copy(), training, eval_threshold=et)

        raw_preds = (raw >= eval_raw).astype(int)
        norm_preds = (result >= eval_norm).astype(int)
        np.testing.assert_array_equal(norm_preds, raw_preds)

    @pytest.mark.parametrize(
        "scores, training, eval_raw, eval_norm",
        [
            ([4.0, 7.0, 8.0, 9.5, 12.0], 10.0, 8.0, 80.0),
            ([8.0, 9.0, 10.0, 11.0, 13.0], 10.0, 12.0, 120.0),
            ([8.0, 9.0, 10.0, 11.0, 12.0], 10.0, 10.0, 100.0),
            # training threshold == 0, eval == training: normalized = (raw + 1) * 100
            ([-2.0, -0.5, 0.0, 1.0, 3.0], 0.0, 0.0, 100.0),
            # training threshold == 0, eval > training: normalized = (raw + 1) * 100
            ([0.0, 0.2, 0.5, 1.0, 2.0], 0.0, 0.5, 150.0),
            # training threshold == 0, eval < training: eval_raw is negative
            ([-1.0, -0.5, 0.0, 0.5, 1.0], 0.0, -0.5, 50.0),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_consistency_torch(self, scores, training, eval_raw, eval_norm, dtype):
        device = get_quadra_test_device()

        et = EvalThreshold(raw=eval_raw, normalized=eval_norm)
        raw = torch.tensor(scores, dtype=dtype, device=device)
        result = normalize_anomaly_score(raw.clone(), training, eval_threshold=et)

        raw_preds = (raw >= eval_raw).int()
        norm_preds = (result >= eval_norm).int()
        assert torch.equal(norm_preds, raw_preds), (
            f"dtype={dtype}: raw_preds={raw_preds.tolist()}, norm_preds={norm_preds.tolist()}"
        )

    def test_regression_fp32_score_at_training_boundary(self):
        """Regression for the ULP-gap bug.

        For a fp32 score at nextafter(training_threshold, -inf), fp32 arithmetic
        normalises it to a value below the float64 eval_norm.  Without eval_threshold
        the prediction is therefore False (NOT anomaly) even though the raw score
        IS anomaly relative to eval_raw.

        The assertion on `result_default` verifies the input actually creates an
        inconsistency; if it does not, the test should be revised.
        """
        training = 10.0
        # Largest fp32 value strictly below training threshold
        eval_raw = float(np.nextafter(np.float32(training), np.float32(-np.inf)))
        # eval_norm in float64 lands in the ULP gap above nextafter(fp32(100), -inf)
        eval_norm = eval_raw / training * 100.0

        score = np.array([eval_raw], dtype=np.float32)

        result_default = normalize_anomaly_score(score.copy(), training)
        # Precondition: without eval_threshold the prediction IS wrong
        assert result_default[0] < eval_norm, (
            "Test precondition failed: expected the default path to produce an inconsistent "
            f"result ({result_default[0]:.10f} should be < eval_norm={eval_norm:.10f})"
        )

        et = EvalThreshold(raw=eval_raw, normalized=eval_norm)
        result_fix = normalize_anomaly_score(score.copy(), training, eval_threshold=et)
        assert result_fix[0] >= eval_norm, f"Fix failed: {result_fix[0]:.10f} < eval_norm={eval_norm:.10f}"
