import glob
import json
import logging
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from lightning_fabric.utilities.device_parser import _parse_gpu_ids
from mlflow.models import infer_signature
from mlflow.models.signature import ModelSignature
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.base import ClassifierMixin

import quadra
import quadra.utils.export as quadra_export
from quadra.models.base import ModelSignatureWrapper
from quadra.models.evaluation import BaseEvaluationModel
from quadra.utils.utils import get_tensorboard_logger, model_type_from_path, upload_file_tensorboard

try:
    import onnx

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


log = logging.getLogger(__name__)


def _collect_config_file_paths(file_names: list[str]) -> list[str]:
    """Collect existing config file paths from current directory.

    Args:
        file_names: List of relative file paths to check.

    Returns:
        List of existing file paths.
    """
    config_paths = []
    for f in file_names:
        full_path = os.path.join(os.getcwd(), f)
        if os.path.isfile(full_path):
            config_paths.append(full_path)
    return config_paths


def _load_model_json(export_folder: str) -> dict[str, Any] | None:
    """Load model.json from export folder if it exists.

    Args:
        export_folder: Directory containing model.json.

    Returns:
        Parsed JSON dict or None if file doesn't exist.
    """
    model_json_path = os.path.join(export_folder, "model.json")
    if os.path.exists(model_json_path):
        with open(model_json_path) as f:
            return json.load(f)
    return None


def _create_model_zip_archive(model_path: str, model_type: str, export_folder: str, temp_dir: str) -> None:
    """Create a zip archive for a model.

    Args:
        model_path: Path to the model file.
        model_type: Type of the model (e.g., "pytorch", "onnx").
        export_folder: Directory containing model files.
        temp_dir: Temporary directory for zip creation.
    """
    model_name = os.path.basename(model_path)

    if model_type == "pytorch" and os.path.isfile(os.path.join(export_folder, "model_config.yaml")):
        shutil.copy(model_path, temp_dir)
        shutil.copy(os.path.join(export_folder, "model_config.yaml"), temp_dir)
        shutil.make_archive("assets", "zip", root_dir=temp_dir)
    else:
        shutil.make_archive(
            "assets",
            "zip",
            root_dir=os.path.dirname(model_path),
            base_dir=model_name,
        )
    shutil.move("assets.zip", temp_dir)


def _log_single_model(
    model_path: str,
    model_type: str,
    inputs: list[Any],
    config: DictConfig,
    export_folder: str,
    mlflow_zip_models: bool,
    device: str,
) -> bool:
    """Log a single model to MLflow.

    Args:
        model_path: Path to the model file.
        model_type: Type of model (pytorch, torchscript, onnx, etc.).
        inputs: List of sample inputs for signature inference.
        config: Configuration with inference settings.
        export_folder: Export folder containing models.
        mlflow_zip_models: Whether to zip models before logging.
        device: Device for model loading.

    Returns:
        True if model was uploaded successfully, False otherwise.
    """
    if model_type == "pytorch" and not mlflow_zip_models:
        log.warning("Pytorch format still not supported for mlflow upload")
        return False

    if mlflow_zip_models:
        with TemporaryDirectory() as temp_dir:
            _create_model_zip_archive(model_path, model_type, export_folder, temp_dir)
            mlflow.pyfunc.log_model(
                artifact_path=model_path,
                loader_module="not.used",
                data_path=os.path.join(temp_dir, "assets.zip"),
                pip_requirements=[""],
            )
        return True

    # Non-zipped model logging
    loaded_model = quadra_export.import_deployment_model(
        model_path,
        device=device,
        inference_config=config.inference,
    )

    if model_type in ["torchscript", "pytorch"]:
        signature = infer_signature_model(loaded_model.model, inputs)
        mlflow.pytorch.log_model(
            loaded_model.model,
            artifact_path=model_path,
            signature=signature,
        )
        return True

    if model_type in ["onnx", "simplified_onnx"] and ONNX_AVAILABLE:
        if loaded_model.model_path is None:
            log.warning("Cannot log onnx model on mlflow, BaseEvaluationModel 'model_path' attribute is None")
            return False

        signature = infer_signature_model(loaded_model, inputs)
        model_proto = onnx.load(loaded_model.model_path)
        mlflow.onnx.log_model(
            model_proto,
            artifact_path=model_path,
            signature=signature,
        )
        return True

    return False


def _log_torch_trained_models_to_mlflow(
    export_folder: str,
    config: DictConfig,
    device: str,
    half_precision: bool,
) -> None:
    """Log torch trained deployment models to MLflow.

    Args:
        export_folder: Directory containing exported models.
        config: Configuration with upload settings.
        device: Device for model loading.
        half_precision: Whether models use half precision.
    """
    types_to_upload = config.core.get("upload_models")
    if not types_to_upload:
        return

    deployed_models = glob.glob(os.path.join(export_folder, "*"))
    model_json = _load_model_json(export_folder)

    if model_json is None:
        return

    input_size = model_json["input_size"]
    if not isinstance(input_size[0], list):
        input_size = [input_size]

    inputs = list(quadra_export.generate_torch_inputs(input_size, device=device, half_precision=half_precision))
    mlflow_zip_models = config.core.get("mlflow_zip_models", False)
    model_uploaded = False

    for model_path in deployed_models:
        model_type = model_type_from_path(model_path)

        if model_type is None or model_type not in types_to_upload:
            continue

        if _log_single_model(model_path, model_type, inputs, config, export_folder, mlflow_zip_models, device):
            model_uploaded = True

    if model_uploaded:
        model_json_path = os.path.join(export_folder, "model.json")
        mlflow.log_artifact(model_json_path, export_folder)


def _log_folder_artifacts(
    folder: str,
    mlflow_client: Any,
    artifact_base_path: str,
    recursive: bool = True,
) -> None:
    """Log all files from a folder to MLflow.

    Args:
        folder: Local folder to upload.
        mlflow_client: MLflow client or logger with log_artifact method.
        artifact_base_path: Base path in artifact store.
        recursive: Whether to search recursively.
    """
    if not os.path.isdir(folder):
        return

    pattern = os.path.join(folder, "**/*") if recursive else os.path.join(folder, "*")
    artifacts = glob.glob(pattern, recursive=recursive)

    for artifact_path in artifacts:
        if os.path.isdir(artifact_path):
            continue

        dirname = Path(artifact_path).parent.name
        mlflow_client.log_artifact(
            artifact_path,
            artifact_path=os.path.join(artifact_base_path, dirname),
        )


@torch.inference_mode()
def infer_signature_model(model: BaseEvaluationModel, data: list[Any]) -> ModelSignature | None:
    """Infer input and output signature for a PyTorch/Torchscript model."""
    model = model.eval()
    model_output = model(*data)

    try:
        output_signature = infer_signature_input(model_output)

        if len(data) == 1:
            signature_input = infer_signature_input(data[0])
        else:
            signature_input = infer_signature_input(data)
    except ValueError:
        # TODO: Solve circular import as it is not possible to import get_logger right now
        # log.warning("Unable to infer signature for model output type %s", type(model_output))
        return None

    return infer_signature(signature_input, output_signature)


def infer_signature_input(input_tensor: Any) -> Any:
    """Recursively infer the signature input format to pass to mlflow.models.infer_signature.

    Raises:
        ValueError: If the input type is not supported or when nested dicts or sequences are encountered.
    """
    signature: dict[str, Any] | np.ndarray
    if isinstance(input_tensor, Sequence):
        # Mlflow currently does not support sequence outputs, so we use a dict instead
        signature = {}
        for i, x in enumerate(input_tensor):
            if isinstance(x, Sequence):
                # Nested signature is currently not supported by mlflow
                raise ValueError("Nested sequences are not supported")
                # TODO: Enable this once mlflow supports nested signatures
                # signature[f"output_{i}"] = {f"output_{j}": infer_signature_torch(y) for j, y in enumerate(x)}
            if isinstance(x, dict):
                # Nested dicts are not supported
                raise ValueError("Nested dicts are not supported")

            signature[f"output_{i}"] = infer_signature_input(x)
    elif isinstance(input_tensor, torch.Tensor):
        signature = input_tensor.cpu().numpy()
    elif isinstance(input_tensor, dict):
        signature = {}
        for k, v in input_tensor.items():
            if isinstance(v, dict):
                # Nested dicts are not supported
                raise ValueError("Nested dicts are not supported")
            if isinstance(v, Sequence):
                # Nested signature is currently not supported by mlflow
                raise ValueError("Nested sequences are not supported")

            signature[k] = infer_signature_input(v)
    else:
        raise ValueError(f"Unable to infer signature for model output type {type(input_tensor)}")

    return signature


def get_mlflow_logger(trainer: Trainer) -> MLFlowLogger | None:
    """Safely get Mlflow logger from Trainer loggers.

    Args:
        trainer: Pytorch Lightning trainer.

    Returns:
        An mlflow logger if available, else None.
    """
    if isinstance(trainer.logger, MLFlowLogger):
        return trainer.logger

    if isinstance(trainer.logger, list):
        for logger in trainer.logger:
            if isinstance(logger, MLFlowLogger):
                return logger

    return None


def upload_lightning_artifacts(
    config: DictConfig,
    trainer: pl.Trainer,
    logger: list[pl.loggers.Logger],
    export_folder: str,
) -> None:
    """Upload config files to MLFlow server.

    Args:
        config: Configuration composed by Hydra..
        trainer: LightningTrainer.
        logger: List of LightningLoggers.
        export_folder: Folder where the deployment models are exported.
    """
    # pylint: disable=unused-argument
    if len(logger) > 0 and config.core.get("upload_artifacts"):
        mlflow_logger = get_mlflow_logger(trainer=trainer)
        tensorboard_logger = get_tensorboard_logger(trainer=trainer)
        file_names = ["config.yaml", "config_resolved.yaml", "config_tree.txt", "data/dataset.csv"]
        if "16" in str(trainer.precision):
            index = _parse_gpu_ids(config.trainer.devices, include_cuda=True)[0]
            device = "cuda:" + str(index)
            half_precision = True
        else:
            device = "cpu"
            half_precision = False

        if mlflow_logger is not None:
            config_paths = _collect_config_file_paths(file_names)

            for path in config_paths:
                mlflow_logger.experiment.log_artifact(
                    run_id=mlflow_logger.run_id, local_path=path, artifact_path="metadata"
                )

            with mlflow.start_run(run_id=mlflow_logger.run_id) as _:
                _log_torch_trained_models_to_mlflow(export_folder, config, device, half_precision)

        # TODO: Why tensorboard here?
        if tensorboard_logger is not None:
            config_paths = _collect_config_file_paths(file_names)

            for path in config_paths:
                upload_file_tensorboard(file_path=path, tensorboard_logger=tensorboard_logger)

            tensorboard_logger.experiment.flush()


class SklearnMLflowClient:
    """MLflow client for non-Lightning tasks.

    Wraps the native ``mlflow`` API to manage a single run lifecycle and provide
    logging helpers. Reads configuration from ``config.logger.mlflow`` which uses the
    same keys as PyTorch Lightning's ``MLFlowLogger`` (``experiment_name``,
    ``tracking_uri``, ``tags``, ``run_id``).

    All public methods are safe no-ops when MLflow is not installed, not configured,
    or when the current run is disabled (e.g. ``core.unit_test`` is True).

    Args:
        config: The full Hydra DictConfig for the experiment.
    """

    def __init__(self, config: DictConfig):
        self._config = config
        self._run_id: str
        self._experiment_name: str = "default"
        self._tracking_uri: str
        self._enabled: bool = False
        self._setup()

    @property
    def enabled(self) -> bool:
        """Whether MLflow is configured and available."""
        return self._enabled

    @property
    def run_id(self) -> str:
        """The active run ID, or None."""
        return self._run_id

    @property
    def experiment_id(self) -> str:
        """The active experiment ID, or None."""
        return self._experiment_id

    @property
    def experiment_name(self) -> str:
        """The active experiment name, or None."""
        return self._experiment_name

    @property
    def tracking_uri(self) -> str:
        """The MLflow tracking URI."""
        return self._tracking_uri

    def _setup(self) -> None:
        """Determine whether MLflow integration should be active."""
        logger_config = self._config.get("logger")
        if logger_config is None:
            log.info("No logger config found, sklearn MLflow integration disabled")
            return

        mlflow_config = logger_config.get("mlflow")
        if mlflow_config is None:
            log.info("No mlflow logger config found, sklearn MLflow integration disabled")
            return

        tracking_uri = mlflow_config.get("tracking_uri")
        if tracking_uri is None:
            log.info("No MLflow tracking URI configured, sklearn MLflow integration disabled")
            return

        self._experiment_name = mlflow_config.get("experiment_name", self._config.core.get("name", "default"))
        self._tracking_uri = tracking_uri
        self._enabled = True

    def start_run(self) -> None:
        """Start a new MLflow run (or resume one if ``run_id`` is set in config)."""
        if not self._enabled:
            return

        try:
            mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._experiment_name)

            run = mlflow.start_run()
            self._run_id = run.info.run_id
            self._experiment_id = run.info.experiment_id
            log.info(
                "MLflow run started at %s/#/experiments/%s/runs/%s",
                self._tracking_uri,
                self._experiment_id,
                self._run_id,
            )
        except Exception as e:
            log.warning("Failed to start MLflow run: %s. MLflow integration will be disabled.", e)
            self._enabled = False

    def end_run(self) -> None:
        """End the active MLflow run."""
        if not self._enabled or self._run_id is None:
            return

        try:
            mlflow.end_run()
            log.info("MLflow run ended: run_id=%s", self._run_id)
        except Exception as e:
            log.warning("Failed to end MLflow run: %s", e)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters, truncating values that exceed MLflow's limit.

        Args:
            params: Dictionary of parameter name-value pairs.
        """
        if not self._enabled or self._run_id is None:
            return

        try:
            # Truncate long values and convert to strings
            safe_params: dict[str, str] = {}
            for k, v in params.items():
                str_v = str(v)
                if len(str_v) > 500:
                    str_v = str_v[:500]
                safe_params[k] = str_v

            # mlflow.log_params has a 100-param batch limit
            param_items = list(safe_params.items())
            for i in range(0, len(param_items), 100):
                chunk = dict(param_items[i : i + 100])
                mlflow.log_params(chunk)
        except Exception as e:
            log.warning("Failed to log params to MLflow: %s", e)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to the active run.

        Args:
            metrics: Dictionary of metric name-value pairs.
            step: Optional step number for the metrics.
        """
        if not self._enabled or self._run_id is None:
            return

        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            log.warning("Failed to log metrics to MLflow: %s", e)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a single artifact file.

        Args:
            local_path: Path to the file to upload.
            artifact_path: Destination directory in the artifact store.
        """
        if not self._enabled or self._run_id is None:
            return

        if not os.path.isfile(local_path):
            return

        try:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        except Exception as e:
            log.warning("Failed to log artifact %s to MLflow: %s", local_path, e)

    def log_sklearn_model(self, model: Any, artifact_path: str) -> None:
        """Log an sklearn model using ``mlflow.sklearn.log_model``.

        Args:
            model: A fitted sklearn classifier.
            artifact_path: Destination path in the artifact store.
        """
        if not self._enabled or self._run_id is None:
            return

        try:
            mlflow.sklearn.log_model(model, artifact_path=artifact_path)
            log.info("Sklearn model logged to MLflow at artifact_path=%s", artifact_path)
        except Exception as e:
            log.warning("Failed to log sklearn model to MLflow: %s", e)

    def log_backbone_models(self, export_folder: str, half_precision: bool = False, device: str = "cpu") -> None:
        """Log backbone deployment models.

        Supports torchscript, pytorch and onnx models.

        Args:
            export_folder: Directory containing the exported deployment models.
            half_precision: Whether models were exported with half precision.
            device: Device to use for model loading during signature inference.
        """
        if not self._enabled or self._run_id is None:
            return

        try:
            _log_torch_trained_models_to_mlflow(export_folder, self._config, device, half_precision)
        except Exception as e:
            log.warning("Failed to log backbone models to MLflow: %s", e)


def _build_sklearn_hyperparameters(config: DictConfig, backbone: torch.nn.Module) -> dict[str, Any]:
    """Build hyperparameters dict for sklearn tasks.

    Mirrors the logic in ``quadra.utils.utils.log_hyperparameters`` but does not
    require a Lightning Trainer or Module.

    Args:
        config: The full Hydra DictConfig.
        backbone: The PyTorch backbone model.

    Returns:
        Dictionary of hyperparameter name-value pairs.
    """
    hparams: dict[str, Any] = {}

    # Hydra choices and overrides
    if HydraConfig.initialized():
        hydra_cfg = HydraConfig.get()
        hydra_choices = OmegaConf.to_container(hydra_cfg.runtime.choices)
        if isinstance(hydra_choices, dict):
            for item in hydra_cfg.overrides.task:
                if "." in item:
                    continue

                override, value = item.split("=")
                hydra_choices[override] = value

            for k, v in hydra_choices.items():
                if isinstance(k, str):
                    k_replaced = k.replace("@", "_at_")
                    hparams[k_replaced] = v
                    if v is not None and isinstance(v, str) and "@" in v:
                        hparams[k_replaced] = v.replace("@", "_at_")

    # Backbone parameters
    hparams["model/params_total"] = sum(p.numel() for p in backbone.parameters())
    hparams["model/params_trainable"] = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    hparams["model/params_not_trainable"] = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)

    # Experiment metadata
    hparams["experiment_path"] = config.core.get("experiment_path", "")
    hparams["command"] = config.core.get("command", "")
    hparams["library/version"] = str(quadra.__version__)

    # Sklearn-specific params
    hparams["sklearn_model"] = config.model.get("_target_", "unknown")
    hparams["n_splits"] = config.datamodule.get("n_splits", 1)

    # Git info
    try:
        with open(os.devnull, "w") as fnull:
            if (
                subprocess.call(["git", "-C", get_original_cwd(), "status"], stderr=subprocess.STDOUT, stdout=fnull)
                == 0
            ):
                hparams["git/commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
                hparams["git/branch"] = (
                    subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("ascii").strip()
                )
                hparams["git/remote"] = (
                    subprocess.check_output(["git", "remote", "get-url", "origin"]).decode("ascii").strip()
                )
    except Exception:
        pass

    return hparams


class SklearnMLflowMixin:
    """Base mixin that adds MLflow tracking to sklearn-based tasks.

    Provides protected methods to be called from task lifecycle methods
    (``prepare``, ``train``, ``generate_report``, ``finalize``).

    The mixin stores a :class:`SklearnMLflowClient` instance on ``self._mlflow_client``.
    All methods are safe no-ops when MLflow is not configured.
    """

    mlflow_client: SklearnMLflowClient | None = None

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.config: DictConfig
        self.output: DictConfig
        self.backbone: ModelSignatureWrapper
        self.model: ClassifierMixin
        self.metadata: dict[str, Any]
        self.export_folder: str

    def _init_mlflow(self) -> None:
        """Initialize MLflow client, start a run, and log hyperparameters.

        Should be called at the end of ``prepare()`` after backbone and config are ready.
        """
        self.mlflow_client = SklearnMLflowClient(self.config)
        self.mlflow_client.start_run()

        if self.mlflow_client.enabled:
            hparams = _build_sklearn_hyperparameters(self.config, self.backbone)
            self.mlflow_client.log_params(hparams)

        # Upload config metadata
        self._log_config_metadata()

    def _log_config_metadata(self) -> None:
        """Upload config files to MLflow as metadata artifacts."""
        if self.mlflow_client is None or not self.mlflow_client.enabled:
            return

        file_names = ["config_resolved.yaml", "config_tree.txt", "data/dataset.csv"]

        for f in file_names:
            full_path = os.path.join(os.getcwd(), f)
            if os.path.isfile(full_path):
                self.mlflow_client.log_artifact(full_path, artifact_path="metadata")

    def _log_train_metrics(self) -> None:
        """Log train metrics from ``self.metadata``.

        Should be overridden by subclasses to implement task-specific metric logging.
        """
        raise NotImplementedError("Subclasses must implement _log_train_metrics")

    def _upload_report_artifacts(self) -> None:
        """Upload report artifacts to MLflow.

        Should be overridden by subclasses to implement task-specific artifact uploading.
        """
        raise NotImplementedError("Subclasses must implement _upload_report_artifacts")

    def _finalize_mlflow(self) -> None:
        """Upload config metadata, models, and end the MLflow run.

        Should be called at the start of ``finalize()`` before ``super().finalize()``.
        """
        if self.mlflow_client is None or not self.mlflow_client.enabled:
            return

        try:
            if self.config.core.get("upload_artifacts"):
                # Upload sklearn classifier
                self.mlflow_client.log_sklearn_model(
                    self.model, artifact_path=os.path.join(self.export_folder, "sklearn_classifier")
                )

                # Upload backbone models (torchscript, pytorch, onnx)
                half_precision = getattr(self, "half_precision", False)
                device = getattr(self, "device", "cpu")
                self.mlflow_client.log_backbone_models(self.export_folder, half_precision=half_precision, device=device)
        except Exception as e:
            log.warning("Failed to log models/metadata to MLflow: %s", e)
        finally:
            self.mlflow_client.end_run()


class SklearnClassificationMLflowMixin(SklearnMLflowMixin):
    """Mixin for sklearn classification tasks with cross-validation support.

    Extends the base SklearnMLflowMixin to log cross-validation metrics and
    per-fold report artifacts.
    """

    def _log_train_metrics(self) -> None:
        """Log cross-validation metrics from ``self.metadata``."""
        if self.mlflow_client is None or not self.mlflow_client.enabled:
            return

        try:
            accuracies = self.metadata.get("test_accuracy", [])
            for fold_idx, accuracy in enumerate(accuracies):
                self.mlflow_client.log_metrics({"cv_fold_accuracy": accuracy}, step=fold_idx)

            if len(accuracies) > 0:
                self.mlflow_client.log_metrics(
                    {
                        "cv_mean_accuracy": float(np.mean(accuracies)),
                        "cv_std_accuracy": float(np.std(accuracies)),
                        "cv_num_folds": len(accuracies),
                    }
                )
        except Exception as e:
            log.warning("Failed to log CV metrics to MLflow: %s", e)

    def _upload_report_artifacts(self) -> None:
        """Upload per-fold report folders and final confusion matrix."""
        if self.mlflow_client is None or not self.mlflow_client.enabled:
            return

        if not self.config.core.get("upload_artifacts"):
            return

        try:
            # Upload per-fold report folders
            for count in range(len(self.metadata.get("test_accuracy", []))):
                folder = f"{self.output.folder}_{count}"
                _log_folder_artifacts(folder, self.mlflow_client, "classification_output", recursive=True)

            # Upload final combined confusion matrix
            final_folder = f"{self.output.folder}"
            final_cm_path = os.path.join(final_folder, "test_confusion_matrix.png")
            if os.path.isfile(final_cm_path):
                self.mlflow_client.log_artifact(final_cm_path, artifact_path="classification_output")
        except Exception as e:
            log.warning("Failed to log report artifacts to MLflow: %s", e)

    def _upload_test_artifacts(self, output_folder: str) -> None:
        """Upload test output artifacts to MLflow.

        Args:
            output_folder: Local folder containing test results to upload.
        """
        if self.mlflow_client is None or not self.mlflow_client.enabled:
            return

        if not self.config.core.get("upload_artifacts"):
            return

        try:
            _log_folder_artifacts(output_folder, self.mlflow_client, "test_output", recursive=True)
        except Exception as e:
            log.warning("Failed to log test artifacts to MLflow: %s", e)


class SklearnPatchMLflowMixin(SklearnMLflowMixin):
    """Mixin for sklearn patch classification tasks.

    Extends the base SklearnMLflowMixin to log patch-specific artifacts like
    reconstruction results.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.reconstruction_results: dict[str, Any]

    def _log_train_metrics(self) -> None:
        """Log validation accuracy from ``self.metadata``."""
        if self.mlflow_client is None or not self.mlflow_client.enabled:
            return

        try:
            accuracy = self.metadata.get("test_accuracy")
            if accuracy is not None:
                self.mlflow_client.log_metrics({"val_accuracy": accuracy})
        except Exception as e:
            log.warning("Failed to log validation metrics to MLflow: %s", e)

    def _upload_report_artifacts(self) -> None:
        """Upload reconstruction results and patch report folder."""
        if self.mlflow_client is None or not self.mlflow_client.enabled:
            return

        if not self.config.core.get("upload_artifacts"):
            return

        try:
            # Upload reconstruction results JSON
            reconstruction_json_path = "reconstruction_results.json"
            if os.path.isfile(reconstruction_json_path):
                self.mlflow_client.log_artifact(reconstruction_json_path, artifact_path="patch_output")

            # Upload report folder with patch visualizations
            if os.path.isdir(self.output.folder):
                _log_folder_artifacts(self.output.folder, self.mlflow_client, "patch_output", recursive=True)
        except Exception as e:
            log.warning("Failed to log report artifacts to MLflow: %s", e)
