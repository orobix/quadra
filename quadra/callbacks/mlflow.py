from __future__ import annotations

import glob
import os
from typing import Any, Literal

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.types import STEP_OUTPUT

from quadra.utils.mlflow import get_mlflow_logger


def check_minio_credentials() -> None:
    """Check minio credentials for aws based storage such as minio.

    Returns:
        None
    """
    check = os.environ.get("AWS_ACCESS_KEY_ID") is not None and os.environ.get("AWS_SECRET_ACCESS_KEY") is not None
    if not check:
        raise ValueError(
            "You are trying to upload mlflow artifacts, but minio credentials are not set. Please set them in your"
            " environment variables."
        )


def check_file_server_dependencies() -> None:
    """Check file dependencies as boto3.

    Returns:
        None
    """
    try:
        # pylint: disable=unused-import,import-outside-toplevel
        import boto3  # noqa
        import minio  # noqa
    except ImportError as e:
        raise ImportError(
            "You are trying to upload mlflow artifacts, but boto3 and minio are not installed. Please install them by"
            " calling pip install minio boto3."
        ) from e


def validate_artifact_storage(logger: MLFlowLogger):
    """Validate artifact storage.

    Args:
        logger: Mlflow logger from pytorch lightning.

    """
    from quadra.utils.utils import get_logger  # pylint: disable=[import-outside-toplevel]

    log = get_logger(__name__)

    client = logger.experiment
    # TODO: we have to access the internal api to get the artifact uri, however there could be a better way
    artifact_uri = client._tracking_client._get_artifact_repo(  # pylint: disable=protected-access
        logger.run_id
    ).artifact_uri
    if artifact_uri.startswith("s3://"):
        check_minio_credentials()
        check_file_server_dependencies()
        log.info("Mlflow artifact storage is AWS/S3 basedand credentials and dependencies are satisfied.")
    else:
        log.info("Mlflow artifact storage uri is %s. Validation checks are not implemented.", artifact_uri)


class UploadCodeAsArtifact(Callback):
    """Callback used to upload Code as artifact.

    Uploads all *.py files to mlflow as an artifact, at the beginning of the run but
        after initializing the trainer. It creates project-source folder under mlflow
        artifacts and other necessary subfolders.

    Args:
        source_dir: Folder where all the source files are stored.
    """

    def __init__(self, source_dir: str):
        self.source_dir = source_dir

    @rank_zero_only
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule):
        """Triggered at the end of test. Uploads all *.py files to mlflow as an artifact.

        Args:
            trainer: Pytorch Lightning trainer.
            pl_module: Pytorch Lightning module.
        """
        logger = get_mlflow_logger(trainer=trainer)

        if logger is None:
            return

        experiment = logger.experiment

        for path in glob.glob(os.path.join(self.source_dir, "**/*.py"), recursive=True):
            stripped_path = path.replace(self.source_dir, "")
            if len(stripped_path.split("/")) > 1:
                file_path_tree = "/" + "/".join(stripped_path.split("/")[:-1])
            else:
                file_path_tree = ""
            experiment.log_artifact(
                run_id=logger.run_id,
                local_path=path,
                artifact_path=f"project-source{file_path_tree}",
            )


class LogGradients(Callback):
    """Callback used to logs of the model at the end of the of each training step.

    Args:
        norm: Norm to use for the gradient. Default is L2 norm.
        tag: Tag to add to the gradients. If None, no tag will be added.
        sep: Separator to use in the log.
        round_to: Number of decimals to round the gradients to.
        log_all_grads: If True, log all gradients, not just the total norm.
    """

    def __init__(
        self,
        norm: int = 2,
        tag: str | None = None,
        sep: str = "/",
        round_to: int = 3,
        log_all_grads: bool = False,
    ):
        self.norm = norm
        self.tag = tag
        self.sep = sep
        self.round_to = round_to
        self.log_all_grads = log_all_grads

    def _grad_norm(self, named_params) -> dict:
        """Compute the gradient norm and return it in a dictionary."""
        grad_tag = "" if self.tag is None else "_" + self.tag
        results = {}
        for name, p in named_params:
            if p.requires_grad and p.grad is not None:
                norm = float(p.grad.data.norm(self.norm))
                key = f"grad_norm_{self.norm}{grad_tag}{self.sep}{name}"
                results[key] = round(norm, 3)
        total_norm = float(torch.tensor(list(results.values())).norm(self.norm))
        if not self.log_all_grads:
            # clear dictionary
            results = {}
        results[f"grad_norm_{self.norm}_total{grad_tag}"] = round(total_norm, self.round_to)
        return results

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: int | None = 0,
    ) -> None:
        """Method called at the end of the train batch
        Args:
            trainer: pl.trainer
            pl_module: lightning module
            outputs: outputs
            batch: batch
            batch_idx: index
            unused: dl index.


        Returns:
            None
        """
        # pylint: disable=unused-argument
        logger = get_mlflow_logger(trainer=trainer)

        if logger is None:
            return

        named_params = pl_module.named_parameters()
        grads = self._grad_norm(named_params)
        logger.log_metrics(grads)


class UploadCheckpointsAsArtifact(Callback):
    """Callback used to upload checkpoints as artifacts.

    Args:
        ckpt_dir: Folder where all the checkpoints are stored in artifact folder.
        ckpt_ext: Extension of checkpoint files (default: ckpt).
        upload_best_only: Only upload best checkpoint (default: False)
        delete_after_upload: Delete the checkpoint from local storage after uploading (default: True)
        upload: If True, upload the checkpoints. If False, only save them on local machine.
    """

    def __init__(
        self,
        ckpt_dir: str = "checkpoints/",
        ckpt_ext: str = "ckpt",
        upload_best_only: bool = False,
        delete_after_upload: bool = True,
        upload: bool = True,
    ):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only
        self.ckpt_ext = ckpt_ext
        self.delete_after_upload = delete_after_upload
        self.upload = upload

    @rank_zero_only
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule):
        """Triggered at the end of test. Uploads all model checkpoints to mlflow as an artifact.

        Args:
            trainer: Pytorch Lightning trainer.
            pl_module: Pytorch Lightning module.
        """
        logger = get_mlflow_logger(trainer=trainer)

        if logger is None:
            return

        experiment = logger.experiment

        if (
            trainer.checkpoint_callback
            and self.upload_best_only
            and hasattr(trainer.checkpoint_callback, "best_model_path")
        ):
            if self.upload:
                experiment.log_artifact(
                    run_id=logger.run_id,
                    local_path=trainer.checkpoint_callback.best_model_path,
                    artifact_path="checkpoints",
                )
        else:
            for path in glob.glob(os.path.join(self.ckpt_dir, f"**/*.{self.ckpt_ext}"), recursive=True):
                if self.upload:
                    experiment.log_artifact(
                        run_id=logger.run_id,
                        local_path=path,
                        artifact_path="checkpoints",
                    )
        if self.delete_after_upload:
            for path in glob.glob(os.path.join(self.ckpt_dir, f"**/*.{self.ckpt_ext}"), recursive=True):
                os.remove(path)


class LogLearningRate(LearningRateMonitor):
    """Learning rate logger at the end of the training step/epoch.

    Args:
        logging_interval: Logging interval.
        log_momentum: If True, log momentum as well.
    """

    def __init__(self, logging_interval: Literal["step", "epoch"] | None = None, log_momentum: bool = False):
        super().__init__(logging_interval=logging_interval, log_momentum=log_momentum)

    def on_train_batch_start(self, trainer, *args, **kwargs):
        """Log learning rate at the beginning of the training step if logging interval is set to step."""
        if not trainer.logger_connector.should_update_logs:
            return
        if self.logging_interval != "epoch":
            logger = get_mlflow_logger(trainer=trainer)

            if logger is None:
                return

            interval = "step" if self.logging_interval is None else "any"
            latest_stat = self._extract_stats(trainer, interval)

            if latest_stat:
                logger.log_metrics(latest_stat, step=trainer.global_step)

    def on_train_epoch_start(self, trainer, *args, **kwargs):
        """Log learning rate at the beginning of the epoch if logging interval is set to epoch."""
        if self.logging_interval != "step":
            interval = "epoch" if self.logging_interval is None else "any"
            latest_stat = self._extract_stats(trainer, interval)
            logger = get_mlflow_logger(trainer=trainer)

            if logger is None:
                return

            if latest_stat:
                logger.log_metrics(latest_stat, step=trainer.global_step)
