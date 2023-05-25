import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only

from quadra.schedulers.warmup import CosineAnnealingWithLinearWarmUp
from quadra.utils.utils import get_logger

log = get_logger(__name__)


class WarmupInit(Callback):
    """Custom callback used to setup a warmup scheduler.

    Args:
        scheduler_config: scheduler configuration.
    """

    def __init__(
        self,
        scheduler_config: DictConfig,
    ) -> None:
        self.scheduler_config = scheduler_config

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when fit begins."""
        if not hasattr(trainer, "datamodule"):
            raise ValueError("Trainer must have a datamodule attribute.")

        if not any(isinstance(s.scheduler, CosineAnnealingWithLinearWarmUp) for s in trainer.lr_scheduler_configs):
            return

        log.info("Using warmup scheduler, forcing optimizer learning rate to zero.")
        for i, _ in enumerate(trainer.optimizers):
            for param_group in trainer.optimizers[i].param_groups:
                param_group["lr"] = 0.0
            trainer.optimizers[i].defaults["lr"] = 0.0

        batch_size = trainer.datamodule.batch_size
        train_dataloader = trainer.datamodule.train_dataloader()
        len_train_dataloader = len(train_dataloader)
        if isinstance(trainer.device_ids, list) and pl_module.device.type == "cuda":
            num_gpus = len(trainer.device_ids)
            len_train_dataloader = len_train_dataloader // num_gpus
            if not train_dataloader.drop_last:
                len_train_dataloader += int((len_train_dataloader % num_gpus) != 0)

        if len_train_dataloader == 1:
            log.warning(
                "From this dataset size, we can only generate single batch. The batch size will be set as lenght of"
                " the dataset "
            )
            batch_size = len(train_dataloader.dataset)

        if isinstance(trainer.device_ids, list) and pl_module.device.type == "cuda":
            batch_size = batch_size * len(trainer.device_ids)

        scheduler = hydra.utils.instantiate(
            self.scheduler_config,
            optimizer=pl_module.optimizer,
            batch_size=batch_size,
            len_loader=len_train_dataloader,
        )

        for i, s in enumerate(trainer.lr_scheduler_configs):
            if isinstance(s.scheduler, CosineAnnealingWithLinearWarmUp):
                trainer.lr_scheduler_configs[i].scheduler = scheduler
