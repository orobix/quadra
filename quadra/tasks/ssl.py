from __future__ import annotations

import json
import os
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, open_dict
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn.functional import interpolate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from quadra.callbacks.scheduler import WarmupInit
from quadra.models.base import ModelSignatureWrapper
from quadra.models.evaluation import BaseEvaluationModel
from quadra.tasks.base import LightningTask, Task
from quadra.utils import utils
from quadra.utils.export import export_model, import_deployment_model

log = utils.get_logger(__name__)


class SSL(LightningTask):
    """SSL Task.

    Args:
        config: The experiment configuration
        checkpoint_path: The path to the checkpoint to load the model from Defaults to None
        report: Whether to create the report
        run_test: Whether to run final test
    """

    def __init__(
        self,
        config: DictConfig,
        run_test: bool = False,
        report: bool = False,
        checkpoint_path: str | None = None,
    ):
        super().__init__(
            config=config,
            checkpoint_path=checkpoint_path,
            run_test=run_test,
            report=report,
        )
        self._backbone: nn.Module
        self._optimizer: torch.optim.Optimizer
        self._lr_scheduler: torch.optim.lr_scheduler._LRScheduler
        self.export_folder = "deployment_model"

    def learnable_parameters(self) -> list[nn.Parameter]:
        """Get the learnable parameters."""
        raise NotImplementedError("This method must be implemented by the subclass")

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Get the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_config: DictConfig) -> None:
        """Set the optimizer."""
        log.info("Instantiating optimizer <%s>", self.config.optimizer["_target_"])
        self._optimizer = hydra.utils.instantiate(optimizer_config, self.learnable_parameters())

    @property
    def scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Get the scheduler."""
        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler_config: DictConfig) -> None:
        log.info("Instantiating scheduler <%s>", scheduler_config["_target_"])
        if "CosineAnnealingWithLinearWarmUp" in self.config.scheduler["_target_"]:
            # This scheduler will be overwritten by the SSLCallback
            self._scheduler = hydra.utils.instantiate(
                scheduler_config,
                optimizer=self.optimizer,
                batch_size=1,
                len_loader=1,
            )
            self.add_callback(WarmupInit(scheduler_config=scheduler_config))
        else:
            self._scheduler = hydra.utils.instantiate(scheduler_config, optimizer=self.optimizer)

    def test(self) -> None:
        """Test the model."""
        if self.run_test and not self.config.trainer.get("fast_dev_run"):
            log.info("Starting testing!")
            log.info("Using last epoch's weights for testing.")
            self.trainer.test(datamodule=self.datamodule, model=self.module, ckpt_path=None)

    def export(self) -> None:
        """Deploy a model ready for production."""
        half_precision = "16" in self.trainer.precision

        input_shapes = self.config.export.input_shapes

        model_json, export_paths = export_model(
            config=self.config,
            model=self.module.model,
            export_folder=self.export_folder,
            half_precision=half_precision,
            input_shapes=input_shapes,
            idx_to_class=None,
        )

        if len(export_paths) == 0:
            return

        with open(os.path.join(self.export_folder, "model.json"), "w") as f:
            json.dump(model_json, f, cls=utils.HydraEncoder)


class Simsiam(SSL):
    """Simsiam model as a pytorch_lightning.LightningModule.

    Args:
        config: the main config
        checkpoint_path: if a checkpoint is specified, then it will return a trained model,
            with weights loaded from the checkpoint path specified.
            Defaults to None.
        run_test: Whether to run final test
    """

    def __init__(
        self,
        config: DictConfig,
        checkpoint_path: str | None = None,
        run_test: bool = False,
    ):
        super().__init__(config=config, checkpoint_path=checkpoint_path, run_test=run_test)
        self.backbone: nn.Module
        self.projection_mlp: nn.Module
        self.prediction_mlp: nn.Module

    def learnable_parameters(self) -> list[nn.Parameter]:
        """Get the learnable parameters."""
        return list(
            list(self.backbone.parameters())
            + list(self.projection_mlp.parameters())
            + list(self.prediction_mlp.parameters()),
        )

    def prepare(self) -> None:
        """Prepare the experiment."""
        super().prepare()
        self.backbone = hydra.utils.instantiate(self.config.model.model)
        self.projection_mlp = hydra.utils.instantiate(self.config.model.projection_mlp)
        self.prediction_mlp = hydra.utils.instantiate(self.config.model.prediction_mlp)
        self.optimizer = self.config.optimizer
        self.scheduler = self.config.scheduler
        self.module = self.config.model.module

    @property
    def module(self) -> LightningModule:
        """Get the module of the model."""
        return super().module

    @module.setter
    def module(self, module_config):
        """Set the module of the model."""
        module = hydra.utils.instantiate(
            module_config,
            model=self.backbone,
            projection_mlp=self.projection_mlp,
            prediction_mlp=self.prediction_mlp,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
        )
        if self.checkpoint_path is not None:
            module = module.__class__.load_from_checkpoint(
                self.checkpoint_path,
                model=self.backbone,
                projection_mlp=self.projection_mlp,
                prediction_mlp=self.prediction_mlp,
                criterion=module.criterion,
                optimizer=self.optimizer,
                lr_scheduler=self.scheduler,
            )
        self._module = module


class SimCLR(SSL):
    """SimCLR model as a pytorch_lightning.LightningModule.

    Args:
        config: the main config
        checkpoint_path: if a checkpoint is specified, then it will return a trained model,
            with weights loaded from the checkpoint path specified.
            Defaults to None.
        run_test: Whether to run final test
    """

    def __init__(
        self,
        config: DictConfig,
        checkpoint_path: str | None = None,
        run_test: bool = False,
    ):
        super().__init__(config=config, checkpoint_path=checkpoint_path, run_test=run_test)
        self.backbone: nn.Module
        self.projection_mlp: nn.Module

    def learnable_parameters(self) -> list[nn.Parameter]:
        """Get the learnable parameters."""
        return list(self.backbone.parameters()) + list(self.projection_mlp.parameters())

    def prepare(self) -> None:
        """Prepare the experiment."""
        super().prepare()
        self.backbone = hydra.utils.instantiate(self.config.model.model)
        self.projection_mlp = hydra.utils.instantiate(self.config.model.projection_mlp)
        self.optimizer = self.config.optimizer
        self.scheduler = self.config.scheduler
        self.module = self.config.model.module

    @property
    def module(self) -> LightningModule:
        return super().module

    @module.setter
    def module(self, module_config):
        """Set the module of the model."""
        module = hydra.utils.instantiate(
            module_config,
            model=self.backbone,
            projection_mlp=self.projection_mlp,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
        )
        if self.checkpoint_path is not None:
            module = module.__class__.load_from_checkpoint(
                self.checkpoint_path,
                model=self.backbone,
                projection_mlp=self.projection_mlp,
                criterion=module.criterion,
                optimizer=self.optimizer,
                lr_scheduler=self.scheduler,
            )
        self._module = module
        self._module.model = ModelSignatureWrapper(self._module.model)


class Barlow(SimCLR):
    """Barlow model as a pytorch_lightning.LightningModule.

    Args:
        config: the main config
        checkpoint_path: if a checkpoint is specified, then it will return a trained model,
            with weights loaded from the checkpoint path specified.
            Defaults to None.
        run_test: Whether to run final test
    """

    def __init__(
        self,
        config: DictConfig,
        checkpoint_path: str | None = None,
        run_test: bool = False,
    ):
        super().__init__(config=config, checkpoint_path=checkpoint_path, run_test=run_test)

    def prepare(self) -> None:
        """Prepare the experiment."""
        super(SimCLR, self).prepare()
        self.backbone = hydra.utils.instantiate(self.config.model.model)

        with open_dict(self.config):
            self.config.model.projection_mlp.hidden_dim = (
                self.config.model.projection_mlp.hidden_dim * self.config.model.projection_mlp_mult
            )
            self.config.model.projection_mlp.output_dim = (
                self.config.model.projection_mlp.output_dim * self.config.model.projection_mlp_mult
            )
        self.projection_mlp = hydra.utils.instantiate(self.config.model.projection_mlp)
        self.optimizer = self.config.optimizer
        self.scheduler = self.config.scheduler
        self.module = self.config.model.module


class BYOL(SSL):
    """BYOL model as a pytorch_lightning.LightningModule.

    Args:
        config: the main config
        checkpoint_path: if a checkpoint is specified, then it will return a trained model,
            with weights loaded from the checkpoint path specified.
            Defaults to None.
        run_test: Whether to run final test
        **kwargs: Keyword arguments
    """

    def __init__(
        self,
        config: DictConfig,
        checkpoint_path: str | None = None,
        run_test: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            config=config,
            checkpoint_path=checkpoint_path,
            run_test=run_test,
            **kwargs,
        )
        self.student_model: nn.Module
        self.teacher_model: nn.Module
        self.student_projection_mlp: nn.Module
        self.student_prediction_mlp: nn.Module
        self.teacher_projection_mlp: nn.Module

    def learnable_parameters(self) -> list[nn.Parameter]:
        """Get the learnable parameters."""
        return list(
            list(self.student_model.parameters())
            + list(self.student_projection_mlp.parameters())
            + list(self.student_prediction_mlp.parameters()),
        )

    def prepare(self) -> None:
        """Prepare the experiment."""
        super().prepare()
        self.student_model = hydra.utils.instantiate(self.config.model.student)
        self.teacher_model = hydra.utils.instantiate(self.config.model.student)
        self.student_projection_mlp = hydra.utils.instantiate(self.config.model.projection_mlp)
        self.student_prediction_mlp = hydra.utils.instantiate(self.config.model.prediction_mlp)
        self.teacher_projection_mlp = hydra.utils.instantiate(self.config.model.projection_mlp)
        self.optimizer = self.config.optimizer
        self.scheduler = self.config.scheduler
        self.module = self.config.model.module

    @property
    def module(self) -> LightningModule:
        return super().module

    @module.setter
    def module(self, module_config):
        """Set the module of the model."""
        module = hydra.utils.instantiate(
            module_config,
            student=self.student_model,
            teacher=self.teacher_model,
            student_projection_mlp=self.student_projection_mlp,
            student_prediction_mlp=self.student_prediction_mlp,
            teacher_projection_mlp=self.teacher_projection_mlp,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
        )
        if self.checkpoint_path is not None:
            module = module.__class__.load_from_checkpoint(
                self.checkpoint_path,
                student=self.student_model,
                teacher=self.teacher_model,
                student_projection_mlp=self.student_projection_mlp,
                student_prediction_mlp=self.student_prediction_mlp,
                teacher_projection_mlp=self.teacher_projection_mlp,
                criterion=module.criterion,
                optimizer=self.optimizer,
                lr_scheduler=self.scheduler,
            )
        self._module = module


class DINO(SSL):
    """DINO model as a pytorch_lightning.LightningModule.

    Args:
        config: the main config
        checkpoint_path: if a checkpoint is specified, then it will return a trained model,
            with weights loaded from the checkpoint path specified.
            Defaults to None.
        run_test: Whether to run final test
    """

    def __init__(
        self,
        config: DictConfig,
        checkpoint_path: str | None = None,
        run_test: bool = False,
    ):
        super().__init__(config=config, checkpoint_path=checkpoint_path, run_test=run_test)
        self.student_model: nn.Module
        self.teacher_model: nn.Module
        self.student_projection_mlp: nn.Module
        self.teacher_projection_mlp: nn.Module

    def learnable_parameters(self) -> list[nn.Parameter]:
        """Get the learnable parameters."""
        return list(
            list(self.student_model.parameters()) + list(self.student_projection_mlp.parameters()),
        )

    def prepare(self) -> None:
        """Prepare the experiment."""
        super().prepare()
        self.student_model = cast(nn.Module, hydra.utils.instantiate(self.config.model.student))
        self.teacher_model = cast(nn.Module, hydra.utils.instantiate(self.config.model.student))
        self.student_projection_mlp = cast(nn.Module, hydra.utils.instantiate(self.config.model.student_projection_mlp))
        self.teacher_projection_mlp = cast(nn.Module, hydra.utils.instantiate(self.config.model.teacher_projection_mlp))
        self.optimizer = self.config.optimizer
        self.scheduler = self.config.scheduler
        self.module = self.config.model.module

    @property
    def module(self) -> LightningModule:
        return super().module

    @module.setter
    def module(self, module_config):
        """Set the module of the model."""
        module = hydra.utils.instantiate(
            module_config,
            student=self.student_model,
            teacher=self.teacher_model,
            student_projection_mlp=self.student_projection_mlp,
            teacher_projection_mlp=self.teacher_projection_mlp,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
        )
        if self.checkpoint_path is not None:
            module = module.__class__.load_from_checkpoint(
                self.checkpoint_path,
                student=self.student_model,
                teacher=self.teacher_model,
                student_projection_mlp=self.student_projection_mlp,
                teacher_projection_mlp=self.teacher_projection_mlp,
                criterion=module.criterion,
                optimizer=self.optimizer,
                lr_scheduler=self.scheduler,
            )
        self._module = module


class EmbeddingVisualization(Task):
    """Visualization task for learned embeddings.

    Args:
        config: The loaded experiment config
        model_path: The path to a deployment model
        report_folder: Where to save the embeddings
        embedding_image_size: If not None rescale the images associated with the embeddings, tensorboard will save
            on disk a large sprite containing all the images in a matrix fashion, if the dimension of this sprite is too
            big it's not possible to load it in the browser. Rescaling the output image from the model input size to
            something smaller can solve this issue. The field is an int to always rescale to a squared image.
    """

    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        report_folder: str = "embeddings",
        embedding_image_size: int | None = None,
    ):
        super().__init__(config=config)

        self.config = config
        self.metadata = {
            "report_files": [],
        }
        self.model_path = model_path
        self._device = utils.get_device()
        log.info("Current device: %s", self._device)

        self.report_folder = report_folder
        if self.model_path is None:
            raise ValueError(
                "Model path cannot be found!, please specify it in the config or pass it as an argument for evaluation"
            )
        self.embeddings_path = os.path.join(self.model_path, self.report_folder)
        if not os.path.exists(self.embeddings_path):
            os.makedirs(self.embeddings_path)
        self.embedding_writer = SummaryWriter(self.embeddings_path)
        self.writer_step = 0  # for tensorboard
        self.embedding_image_size = embedding_image_size
        self._deployment_model: BaseEvaluationModel
        self.deployment_model_type: str

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: str):
        self._device = device

        if self.deployment_model is not None:
            # After prepare
            self.deployment_model = self.deployment_model.to(self._device)

    @property
    def deployment_model(self):
        """Get the deployment model."""
        return self._deployment_model

    @deployment_model.setter
    def deployment_model(self, model_path: str):
        """Set the deployment model."""
        self._deployment_model = import_deployment_model(
            model_path=model_path, device=self.device, inference_config=self.config.inference
        )

    def prepare(self) -> None:
        """Prepare the evaluation."""
        super().prepare()
        self.deployment_model = self.model_path

    @torch.no_grad()
    def test(self) -> None:
        """Run embeddings extraction."""
        self.datamodule.setup("fit")
        idx_to_class = self.datamodule.val_dataset.idx_to_class
        self.datamodule.setup("test")
        dataloader = self.datamodule.test_dataloader()
        images = []
        metadata: list[tuple[int, str, str]] = []
        embeddings = []
        std = torch.tensor(self.config.transforms.std).view(1, -1, 1, 1)
        mean = torch.tensor(self.config.transforms.mean).view(1, -1, 1, 1)
        dl = self.datamodule.test_dataloader()
        counter = 0

        is_half_precision = False
        for param in self.deployment_model.parameters():
            if param.dtype == torch.half:
                is_half_precision = True
            break

        for batch in tqdm(dataloader):
            im, target = batch
            if is_half_precision:
                im = im.half()

            x = self.deployment_model(im.to(self.device))
            targets = [int(t.item()) for t in target]
            class_names = [idx_to_class[t.item()] for t in target]
            file_paths = [s[0] for s in dl.dataset.samples[counter : counter + len(im)]]
            embeddings.append(x.cpu())
            im = im * std
            im += mean

            if self.embedding_image_size is not None:
                im = interpolate(im, self.embedding_image_size)

            images.append(im.cpu())
            metadata.extend(zip(targets, class_names, file_paths, strict=False))
            counter += len(im)
        images = torch.cat(images, dim=0)
        embeddings = torch.cat(embeddings, dim=0)
        self.embedding_writer.add_embedding(
            embeddings,
            metadata=metadata,
            label_img=images,
            global_step=self.writer_step,
            metadata_header=["class", "class_name", "path"],
        )
