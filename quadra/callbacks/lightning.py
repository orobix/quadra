import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

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
