import pytest
from quadra.utils.anomaly import normalize_anomaly_score
import torch
import numpy as np


@pytest.mark.parametrize("raw_score, threshold", [(1.345, 1.24), (1.24, 1.345)])
def test_anomaly_score_normalization_torch(raw_score: float, threshold: float):
    score = torch.tensor(raw_score, dtype=torch.float32)
    normalized_score = normalize_anomaly_score(score, threshold)
    np.testing.assert_allclose(normalized_score.numpy(), raw_score / threshold * 100.0)


@pytest.mark.parametrize("raw_score, threshold", [(1.345, 1.24), (1.24, 1.345)])
def test_anomaly_score_normalization_torch_with_dim(raw_score: float, threshold: float):
    score = torch.tensor([raw_score], dtype=torch.float32)
    normalized_score = normalize_anomaly_score(score, threshold)
    np.testing.assert_allclose(normalized_score.numpy(), np.array([raw_score], dtype=np.float32) / threshold * 100.0)


@pytest.mark.parametrize("raw_score, threshold", [(1.345, 1.24), (1.24, 1.345)])
def test_anomaly_score_normalization_torch_fp16(raw_score: float, threshold: float):
    score = torch.tensor(raw_score, dtype=torch.float16)
    normalized_score = normalize_anomaly_score(score, threshold)
    np.testing.assert_allclose(normalized_score.numpy(), raw_score / threshold * 100.0, rtol=1e-3)


@pytest.mark.parametrize("raw_score, threshold", [(1.345, 1.24), (1.24, 1.345)])
def test_anomaly_score_normalization_torch_fp16_with_dim(raw_score: float, threshold: float):
    score = torch.tensor([raw_score], dtype=torch.float16)
    normalized_score = normalize_anomaly_score(score, threshold)
    np.testing.assert_allclose(normalized_score.numpy(), raw_score / threshold * 100.0, rtol=1e-3)


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
