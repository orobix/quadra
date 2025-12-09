"""Test MLflow integration for sklearn classification tasks."""
# pylint: disable=redefined-outer-name
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from quadra.utils.tests.fixtures import base_classification_dataset, base_patch_classification_dataset
from quadra.utils.tests.helpers import execute_quadra_experiment, get_quadra_test_device


@pytest.mark.parametrize("task_type", ["sklearn_classification", "sklearn_classification_patch"])
def test_sklearn_mlflow_integration(
    tmp_path: Path,
    task_type: str,
    base_classification_dataset: base_classification_dataset,
    base_patch_classification_dataset: base_patch_classification_dataset,
):
    """Test that MLflow logging works for sklearn classification tasks."""
    if task_type == "sklearn_classification":
        data_path, _ = base_classification_dataset
        experiment = "base/classification/sklearn_classification"
    else:
        data_path, _, class_to_idx = base_patch_classification_dataset
        experiment = "base/classification/sklearn_classification_patch"

    device = get_quadra_test_device()
    
    # Create a temporary MLflow tracking directory
    mlflow_tracking_uri = f"file://{tmp_path}/mlruns"

    train_overrides = [
        f"experiment={experiment}",
        f"datamodule.data_path={data_path}",
        "backbone=resnet18",
        f"task.device={device}",
        "export.types=[pytorch,torchscript]",
        "datamodule.num_workers=1",
        "logger=mlflow",
        f"logger.mlflow.tracking_uri={mlflow_tracking_uri}",
        "logger.mlflow.experiment_name=test_sklearn_mlflow",
        "core.upload_artifacts=true",
        "task.output.report=true",
    ]
    
    if task_type == "sklearn_classification_patch":
        class_to_idx_parameter = str(class_to_idx).replace("'", "")
        train_overrides.append(f"datamodule.class_to_idx={class_to_idx_parameter}")

    train_path = tmp_path / "train"
    train_path.mkdir()

    # Execute the experiment
    execute_quadra_experiment(overrides=train_overrides, experiment_path=train_path)

    # Verify that MLflow artifacts were created
    mlruns_path = tmp_path / "mlruns"
    assert mlruns_path.exists(), f"MLflow tracking directory not found at {mlruns_path}"
    
    # Find the experiment directory
    experiment_dirs = [d for d in mlruns_path.iterdir() if d.is_dir() and d.name != "0"]
    assert len(experiment_dirs) > 0, "No MLflow experiment directories found"
    
    # Find the run directory
    experiment_dir = experiment_dirs[0]
    run_dirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) > 0, "No MLflow run directories found"
    
    run_dir = run_dirs[0]
    
    # Check that artifacts were logged
    artifacts_dir = run_dir / "artifacts"
    assert artifacts_dir.exists(), f"MLflow artifacts directory not found at {artifacts_dir}"
    
    # Check for metadata artifacts
    metadata_dir = artifacts_dir / "metadata"
    assert metadata_dir.exists(), "Metadata directory not found in MLflow artifacts"
    assert (metadata_dir / "config_resolved.yaml").exists(), "config_resolved.yaml not found in artifacts"
    
    # Check for task-specific artifacts
    if task_type == "sklearn_classification":
        output_dir = artifacts_dir / "classification_output"
        assert output_dir.exists(), "Classification output directory not found in MLflow artifacts"
    else:
        output_dir = artifacts_dir / "patch_output"
        assert output_dir.exists(), "Patch output directory not found in MLflow artifacts"
    
    # Check that metrics were logged
    metrics_dir = run_dir / "metrics"
    assert metrics_dir.exists(), f"MLflow metrics directory not found at {metrics_dir}"
    
    # Check for validation accuracy metric
    val_accuracy_files = list(metrics_dir.glob("*val_accuracy*"))
    assert len(val_accuracy_files) > 0, "Validation accuracy metric not found in MLflow"
    
    # Check that hyperparameters were logged
    params_dir = run_dir / "params"
    assert params_dir.exists(), f"MLflow params directory not found at {params_dir}"
    
    # Verify some expected hyperparameters (MLflow may convert / to - in file names)
    # Check for either format since MLflow file backend converts / to -
    param_files = [f.name for f in params_dir.iterdir()]
    
    # Check for library version parameter (stored as library-version in files)
    assert any("library" in f and "version" in f for f in param_files), \
        "Library version parameter not found in MLflow"
    
    # Check for experiment_path parameter
    assert "experiment_path" in param_files, "experiment_path parameter not found in MLflow"

    print(f"✓ MLflow integration test passed for {task_type}")


if __name__ == "__main__":
    # Allow running the test directly for debugging
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
