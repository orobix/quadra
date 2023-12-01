import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.batch_size_finder import BatchSizeFinder as LightningBatchSizeFinder
from pytorch_lightning.utilities import rank_zero_only
from torch import nn

from quadra.utils.utils import get_logger

log = get_logger(__name__)


class LightningTrainerBaseSetup(Callback):
    """Custom callback used to setup a lightning trainer with default options.

    Args:
        log_every_n_steps: Default value for trainer.log_every_n_steps if the dataloader is too small.
    """

    def __init__(self, log_every_n_steps: int = 1) -> None:
        self.log_every_n_steps = log_every_n_steps

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called on every stage."""
        if not hasattr(trainer, "datamodule") or not hasattr(trainer, "log_every_n_steps"):
            raise ValueError("Trainer must have a datamodule and log_every_n_steps attribute.")

        len_train_dataloader = len(trainer.datamodule.train_dataloader())
        if len_train_dataloader <= trainer.log_every_n_steps:
            if len_train_dataloader > self.log_every_n_steps:
                trainer.log_every_n_steps = self.log_every_n_steps
                log.info("`trainer.log_every_n_steps` is too high, setting it to %d", self.log_every_n_steps)
            else:
                trainer.log_every_n_steps = 1
                log.warning(
                    "The default log_every_n_steps %d is too high given the datamodule lenght %d, fallback to 1",
                    self.log_every_n_steps,
                    len_train_dataloader,
                )


class BatchSizeFinder(LightningBatchSizeFinder):
    """Batch size finder setting the proper model training status as the current one from lightning seems bugged.
    It also allows to skip some batch size finding steps.

    Args:
        find_train_batch_size: Whether to find the training batch size.
        find_validation_batch_size: Whether to find the validation batch size.
        find_test_batch_size: Whether to find the test batch size.
        find_predict_batch_size: Whether to find the predict batch size.
        mode: The mode to use for batch size finding. See `pytorch_lightning.callbacks.BatchSizeFinder` for more
            details.
        steps_per_trial: The number of steps per trial. See `pytorch_lightning.callbacks.BatchSizeFinder` for more
            details.
        init_val: The initial value for batch size. See `pytorch_lightning.callbacks.BatchSizeFinder` for more details.
        max_trials: The maximum number of trials. See `pytorch_lightning.callbacks.BatchSizeFinder` for more details.
        batch_arg_name: The name of the batch size argument. See `pytorch_lightning.callbacks.BatchSizeFinder` for more
            details.
    """

    def __init__(
        self,
        find_train_batch_size: bool = True,
        find_validation_batch_size: bool = False,
        find_test_batch_size: bool = False,
        find_predict_batch_size: bool = False,
        mode: str = "power",
        steps_per_trial: int = 3,
        init_val: int = 2,
        max_trials: int = 25,
        batch_arg_name: str = "batch_size",
    ) -> None:
        super().__init__(mode, steps_per_trial, init_val, max_trials, batch_arg_name)
        self.find_train_batch_size = find_train_batch_size
        self.find_validation_batch_size = find_validation_batch_size
        self.find_test_batch_size = find_test_batch_size
        self.find_predict_batch_size = find_predict_batch_size

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.find_train_batch_size or trainer.state.stage is None:
            # If called during validation skip it as it will be triggered during on_validation_start
            return None

        if trainer.state.stage.value != "train":
            return None

        if not isinstance(pl_module.model, nn.Module):
            raise ValueError("The model must be a nn.Module")
        pl_module.model.train()

        return super().on_fit_start(trainer, pl_module)

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.find_validation_batch_size:
            return None

        if not isinstance(pl_module.model, nn.Module):
            raise ValueError("The model must be a nn.Module")
        pl_module.model.eval()

        return super().on_validation_start(trainer, pl_module)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.find_test_batch_size:
            return None

        if not isinstance(pl_module.model, nn.Module):
            raise ValueError("The model must be a nn.Module")
        pl_module.model.eval()

        return super().on_test_start(trainer, pl_module)

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.find_predict_batch_size:
            return None

        if not isinstance(pl_module.model, nn.Module):
            raise ValueError("The model must be a nn.Module")
        pl_module.model.eval()

        return super().on_predict_start(trainer, pl_module)
