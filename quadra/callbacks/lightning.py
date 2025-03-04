from __future__ import annotations

import os
import uuid
from copy import deepcopy
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.batch_size_finder import BatchSizeFinder as LightningBatchSizeFinder
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.exceptions import _TunerExitException
from pytorch_lightning.utilities.memory import garbage_collection_cuda, is_oom_error
from pytorch_lightning.utilities.parsing import lightning_getattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
from torch import nn

from quadra.utils.utils import get_logger

log = get_logger(__name__)

# pylint: disable=protected-access


def _scale_batch_size(
    trainer: pl.Trainer,
    mode: str = "power",
    steps_per_trial: int = 3,
    init_val: int = 2,
    max_trials: int = 25,
    batch_arg_name: str = "batch_size",
) -> int | None:
    """Iteratively try to find the largest batch size for a given model that does not give an out of memory (OOM)
    error.

    Args:
        trainer: A Trainer instance.
        mode: Search strategy to update the batch size:

            - ``'power'``: Keep multiplying the batch size by 2, until we get an OOM error.
            - ``'binsearch'``: Initially keep multiplying by 2 and after encountering an OOM error
                do a binary search between the last successful batch size and the batch size that failed.

        steps_per_trial: number of steps to run with a given batch size.
            Ideally 1 should be enough to test if an OOM error occurs,
            however in practise a few are needed
        init_val: initial batch size to start the search with
        max_trials: max number of increases in batch size done before
            algorithm is terminated
        batch_arg_name: name of the attribute that stores the batch size.
            It is expected that the user has provided a model or datamodule that has a hyperparameter
            with that name. We will look for this attribute name in the following places

            - ``model``
            - ``model.hparams``
            - ``trainer.datamodule`` (the datamodule passed to the tune method)

    """
    if trainer.fast_dev_run:  # type: ignore[attr-defined]
        rank_zero_warn("Skipping batch size scaler since `fast_dev_run` is enabled.")
        return None

    # Save initial model, that is loaded after batch size is found
    ckpt_path = os.path.join(trainer.default_root_dir, f".scale_batch_size_{uuid.uuid4()}.ckpt")
    trainer.save_checkpoint(ckpt_path)

    # Arguments we adjust during the batch size finder, save for restoring
    params = __scale_batch_dump_params(trainer)

    # Set to values that are required by the algorithm
    __scale_batch_reset_params(trainer, steps_per_trial)

    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.disable()

    lightning_setattr(trainer.lightning_module, batch_arg_name, init_val)

    if mode == "power":
        new_size = _run_power_scaling(trainer, init_val, batch_arg_name, max_trials, params)
    elif mode == "binsearch":
        new_size = _run_binary_scaling(trainer, init_val, batch_arg_name, max_trials, params)
    else:
        raise ValueError(f"Unknown mode {mode}")

    garbage_collection_cuda()

    log.info("Finished batch size finder, will continue with full run using batch size %d", new_size)

    __scale_batch_restore_params(trainer, params)

    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.enable()

    trainer._checkpoint_connector.restore(ckpt_path)
    trainer.strategy.remove_checkpoint(ckpt_path)

    return new_size


def __scale_batch_dump_params(trainer: pl.Trainer) -> dict[str, Any]:
    """Dump the parameters that need to be reset after the batch size finder.."""
    dumped_params = {
        "loggers": trainer.loggers,
        "callbacks": trainer.callbacks,  # type: ignore[attr-defined]
    }
    loop = trainer._active_loop
    assert loop is not None
    if isinstance(loop, pl.loops._FitLoop):
        dumped_params["max_steps"] = trainer.max_steps
        dumped_params["limit_train_batches"] = trainer.limit_train_batches
        dumped_params["limit_val_batches"] = trainer.limit_val_batches
    elif isinstance(loop, pl.loops._EvaluationLoop):
        stage = trainer.state.stage
        assert stage is not None
        dumped_params["limit_eval_batches"] = getattr(trainer, f"limit_{stage.dataloader_prefix}_batches")
        dumped_params["loop_verbose"] = loop.verbose

    dumped_params["loop_state_dict"] = deepcopy(loop.state_dict())
    return dumped_params


def __scale_batch_reset_params(trainer: pl.Trainer, steps_per_trial: int) -> None:
    """Reset the parameters that need to be reset after the batch size finder."""
    from pytorch_lightning.loggers.logger import DummyLogger  # pylint: disable=import-outside-toplevel

    trainer.logger = DummyLogger() if trainer.logger is not None else None
    trainer.callbacks = []  # type: ignore[attr-defined]

    loop = trainer._active_loop
    assert loop is not None
    if isinstance(loop, pl.loops._FitLoop):
        trainer.limit_train_batches = 1.0
        trainer.limit_val_batches = steps_per_trial
        trainer.fit_loop.epoch_loop.max_steps = steps_per_trial
    elif isinstance(loop, pl.loops._EvaluationLoop):
        stage = trainer.state.stage
        assert stage is not None
        setattr(trainer, f"limit_{stage.dataloader_prefix}_batches", steps_per_trial)
        loop.verbose = False


def __scale_batch_restore_params(trainer: pl.Trainer, params: dict[str, Any]) -> None:
    """Restore the parameters that need to be reset after the batch size finder."""
    # TODO: There are more states that needs to be reset (#4512 and #4870)
    trainer.loggers = params["loggers"]
    trainer.callbacks = params["callbacks"]  # type: ignore[attr-defined]

    loop = trainer._active_loop
    assert loop is not None
    if isinstance(loop, pl.loops._FitLoop):
        loop.epoch_loop.max_steps = params["max_steps"]
        trainer.limit_train_batches = params["limit_train_batches"]
        trainer.limit_val_batches = params["limit_val_batches"]
    elif isinstance(loop, pl.loops._EvaluationLoop):
        stage = trainer.state.stage
        assert stage is not None
        setattr(trainer, f"limit_{stage.dataloader_prefix}_batches", params["limit_eval_batches"])

    loop.load_state_dict(deepcopy(params["loop_state_dict"]))
    loop.restarting = False
    if isinstance(loop, pl.loops._EvaluationLoop) and "loop_verbose" in params:
        loop.verbose = params["loop_verbose"]

    # make sure the loop's state is reset
    _reset_dataloaders(trainer)
    loop.reset()


def _run_power_scaling(
    trainer: pl.Trainer,
    new_size: int,
    batch_arg_name: str,
    max_trials: int,
    params: dict[str, Any],
) -> int:
    """Batch scaling mode where the size is doubled at each iteration until an OOM error is encountered."""
    # this flag is used to determine whether the previously scaled batch size, right before OOM, was a success or not
    # if it was we exit, else we continue downscaling in case we haven't encountered a single optimal batch size
    any_success = False
    # In the original
    for i in range(max_trials):
        garbage_collection_cuda()

        # reset after each try
        _reset_progress(trainer)

        try:
            if i == 0:
                rank_zero_info(f"Starting batch size finder with batch size {new_size}")
                new_size, _ = _adjust_batch_size(trainer, batch_arg_name, factor=1.0, desc=None)
                changed = True
            else:
                new_size, changed = _adjust_batch_size(trainer, batch_arg_name, factor=2.0, desc="succeeded")
            # Force the train dataloader to reset as the batch size has changed
            _reset_dataloaders(trainer)
            _try_loop_run(trainer, params)

            any_success = True

            # In the original lightning implementation this is done before _reset_dataloaders
            # As such the batch size is not checked for the last iteration!!!
            if not changed:
                break
        except RuntimeError as exception:
            if is_oom_error(exception):
                # If we fail in power mode, half the size and return
                garbage_collection_cuda()
                if any_success:
                    # In the original lightning code there's a line that doesn't halve the size properly if batch_size
                    # is bigger than the dataset length
                    rank_zero_info(f"Batch size {new_size} failed, using batch size {new_size // 2}")
                    new_size = new_size // 2
                    lightning_setattr(trainer.lightning_module, batch_arg_name, new_size)
                else:
                    # In this case it means the first iteration will fail already, probably due to a way to big
                    # initial batch size, since the next iteration will start from (new_size // 2) * 2, which is the
                    # same divide by 4 instead and retry
                    rank_zero_info(f"Batch size {new_size} failed at first iteration, using batch size {new_size // 4}")
                    new_size = new_size // 4
                    lightning_setattr(trainer.lightning_module, batch_arg_name, new_size)

                # Force the train dataloader to reset as the batch size has changed
                _reset_dataloaders(trainer)
                if any_success:
                    break
            else:
                raise  # some other error not memory related

    return new_size


def _run_binary_scaling(
    trainer: pl.Trainer,
    new_size: int,
    batch_arg_name: str,
    max_trials: int,
    params: dict[str, Any],
) -> int:
    """Batch scaling mode where the size is initially is doubled at each iteration until an OOM error is encountered.

    Hereafter, the batch size is further refined using a binary search

    """
    low = 1
    high = None
    count = 0
    while True:
        garbage_collection_cuda()

        # reset after each try
        _reset_progress(trainer)

        try:
            # run loop
            _try_loop_run(trainer, params)
            count += 1
            if count > max_trials:
                break
            # Double in size
            low = new_size
            if high:
                if high - low <= 1:
                    break
                midval = (high + low) // 2
                new_size, changed = _adjust_batch_size(trainer, batch_arg_name, value=midval, desc="succeeded")
            else:
                new_size, changed = _adjust_batch_size(trainer, batch_arg_name, factor=2.0, desc="succeeded")

            if not changed:
                break

            # Force the train dataloader to reset as the batch size has changed
            _reset_dataloaders(trainer)

        except RuntimeError as exception:
            # Only these errors should trigger an adjustment
            if is_oom_error(exception):
                # If we fail in power mode, half the size and return
                garbage_collection_cuda()

                high = new_size
                midval = (high + low) // 2
                new_size, _ = _adjust_batch_size(trainer, batch_arg_name, value=midval, desc="failed")

                # Force the train dataloader to reset as the batch size has changed
                _reset_dataloaders(trainer)

                if high - low <= 1:
                    break
            else:
                raise  # some other error not memory related

    return new_size


def _adjust_batch_size(
    trainer: pl.Trainer,
    batch_arg_name: str = "batch_size",
    factor: float = 1.0,
    value: int | None = None,
    desc: str | None = None,
) -> tuple[int, bool]:
    """Helper function for adjusting the batch size.

    Args:
        trainer: instance of pytorch_lightning.Trainer
        batch_arg_name: name of the attribute that stores the batch size
        factor: value which the old batch size is multiplied by to get the
            new batch size
        value: if a value is given, will override the batch size with this value.
            Note that the value of `factor` will not have an effect in this case
        desc: either ``"succeeded"`` or ``"failed"``. Used purely for logging

    Returns:
        The new batch size for the next trial and a bool that signals whether the
        new value is different than the previous batch size.

    """
    model = trainer.lightning_module
    batch_size = lightning_getattr(model, batch_arg_name)
    assert batch_size is not None

    loop = trainer._active_loop
    assert loop is not None
    loop.setup_data()
    combined_loader = loop._combined_loader
    assert combined_loader is not None
    try:
        combined_dataset_length = combined_loader._dataset_length()
        if batch_size >= combined_dataset_length:
            rank_zero_info(f"The batch size {batch_size} is greater or equal than the length of your dataset.")
            return batch_size, False
    except NotImplementedError:
        # all datasets are iterable style
        pass

    new_size = value if value is not None else int(batch_size * factor)
    if desc:
        rank_zero_info(f"Batch size {batch_size} {desc}, trying batch size {new_size}")
    changed = new_size != batch_size
    lightning_setattr(model, batch_arg_name, new_size)

    return new_size, changed


def _reset_dataloaders(trainer: pl.Trainer) -> None:
    """Reset the dataloaders to force a reload."""
    loop = trainer._active_loop
    assert loop is not None
    loop._combined_loader = None  # force a reload
    loop.setup_data()
    if isinstance(loop, pl.loops._FitLoop):
        loop.epoch_loop.val_loop._combined_loader = None
        loop.epoch_loop.val_loop.setup_data()


def _try_loop_run(trainer: pl.Trainer, params: dict[str, Any]) -> None:
    """Try to run the loop with the current batch size."""
    loop = trainer._active_loop
    assert loop is not None
    loop.load_state_dict(deepcopy(params["loop_state_dict"]))
    loop.restarting = False
    loop.run()


def _reset_progress(trainer: pl.Trainer) -> None:
    """Reset the progress of the trainer."""
    if trainer.lightning_module.automatic_optimization:
        trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.reset()
    else:
        trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.reset()

    trainer.fit_loop.epoch_progress.reset()


# Most of the code above is copied from the original lightning implementation since almost everything is private


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

    def scale_batch_size(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Scale the batch size."""
        new_size = _scale_batch_size(
            trainer,
            self._mode,
            self._steps_per_trial,
            self._init_val,
            self._max_trials,
            self._batch_arg_name,
        )

        self.optimal_batch_size = new_size
        if self._early_exit:
            raise _TunerExitException()
