import getpass
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

from aigo_research.utils.utils import get_logger

log = get_logger(__name__)

try:
    import mlflow  # noqa
    from mlflow.entities import Run  # noqa
    from mlflow.entities.model_registry import ModelVersion  # noqa
    from mlflow.exceptions import RestException  # noqa
    from mlflow.tracking import MlflowClient  # noqa

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


VERSION_MD_TEMPLATE = "## **Version {}**\n"
DESCRIPTION_MD_TEMPLATE = "### Description: \n{}\n"


class AbstractModelManager(ABC):
    """Abstract class for model managers."""

    @abstractmethod
    def register_model(
        self, model_location: str, model_name: str, description: str, tags: Optional[Dict[str, Any]]
    ) -> Any:
        """Register a model in the model registry."""

    @abstractmethod
    def get_latest_version(self, model_name: str) -> Any:
        """Get the latest version of a model for all the possible stages or filtered by stage."""

    @abstractmethod
    def transistion_model(self, model_name: str, version: int, stage: str) -> Any:
        """Transition the model with the given version to a new stage."""

    @abstractmethod
    def delete_model(self, model_name: str, version: int) -> None:
        """Delete a model with the given version."""

    @abstractmethod
    def register_best_model(
        self,
        experiment_name: str,
        metric: str,
        model_name: str,
        tags: Optional[Dict[str, Any]] = None,
        description: str = "",
        mode: str = "max",
        model_path: str = "model",
    ) -> Any:
        """Register the best model from an experiment."""


class MlflowModelManager(AbstractModelManager):
    """Model manager for Mlflow."""

    def __init__(self):
        if not MLFLOW_AVAILABLE:
            raise ImportError("Mlflow is not available, please install it with pip install mlflow")

        if os.getenv("MLFLOW_TRACKING_URI") is None:
            raise ValueError("MLFLOW_TRACKING_URI environment variable is not set")

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        self.client = MlflowClient()

    def register_model(
        self, model_location: str, model_name: str, description: str, tags: Optional[Dict[str, Any]] = None
    ) -> ModelVersion:
        """Register a model in the model registry.

        Args:
            model_location: The model uri
            model_name: The name of the model after it is registered
            description: A description of the model, this will be added to the model changelog
            tags: A dictionary of tags to add to the model

        Returns:
            The model version
        """
        model_version = mlflow.register_model(model_uri=model_location, name=model_name, tags=tags)
        log.info("Registered model %s with version %s", model_name, model_version.version)
        registered_model_description = self.client.get_registered_model(model_name).description

        if model_version.version == "1":
            header = "# MODEL CHANGELOG\n"
        else:
            header = ""

        new_model_description = VERSION_MD_TEMPLATE.format(model_version.version)
        new_model_description += self._get_author_and_date()
        new_model_description += DESCRIPTION_MD_TEMPLATE.format(description)

        self.client.update_registered_model(model_name, header + registered_model_description + new_model_description)

        self.client.update_model_version(
            model_name, model_version.version, "# MODEL CHANGELOG\n" + new_model_description
        )

        return model_version

    def get_latest_version(self, model_name: str) -> ModelVersion:
        """Get the latest version of a model.

        Args:
            model_name: The name of the model

        Returns:
            The model version
        """
        latest_version = max(int(x.version) for x in self.client.get_latest_versions("manager_model"))
        model_version = self.client.get_model_version(model_name, latest_version)

        return model_version

    def transistion_model(self, model_name: str, version: int, stage: str) -> Optional[ModelVersion]:
        """Transition a model to a new stage.

        Args:
            model_name: The name of the model
            version: The version of the model
            stage: The stage of the model
        """
        previous_stage = self._safe_get_stage(model_name, version)

        if previous_stage is None:
            return None

        if previous_stage.lower() == stage.lower():
            log.warning("Model %s version %s is already in stage %s", model_name, version, stage)
            return self.client.get_model_version(model_name, version)

        log.info("Transitioning model %s version %s from %s to %s", model_name, version, previous_stage, stage)
        model_version = self.client.transition_model_version_stage(name=model_name, version=version, stage=stage)
        new_stage = model_version.current_stage
        registered_model_description = self.client.get_registered_model(model_name).description
        single_model_description = self.client.get_model_version(model_name, version).description

        new_model_description = "## **Transition:**\n"
        new_model_description += f"### Version {model_version.version} from {previous_stage} to {new_stage}\n"
        new_model_description += self._get_author_and_date()

        self.client.update_registered_model(model_name, registered_model_description + new_model_description)
        self.client.update_model_version(
            model_name, model_version.version, single_model_description + new_model_description
        )

        return model_version

    def delete_model(self, model_name: str, version: int, description: str = "") -> None:
        """Delete a model.

        Args:
            model_name: The name of the model
            version: The version of the model
            description: Why the model was deleted, this will be added to the model changelog
        """
        model_stage = self._safe_get_stage(model_name, version)

        if model_stage is None:
            return

        if (
            input(
                f"Model named `{model_name}`, version {version} is in stage {model_stage}, "
                "type the model name to continue deletion:"
            )
            != model_name
        ):
            log.warning("Model name did not match, aborting deletion")
            return

        log.info("Deleting model %s version %s", model_name, version)
        self.client.delete_model_version(model_name, version)

        registered_model_description = self.client.get_registered_model(model_name).description
        single_model_description = self.client.get_model_version(model_name, version).description

        new_model_description = "## **Deletion:**\n"
        new_model_description += VERSION_MD_TEMPLATE.format(version)
        new_model_description += self._get_author_and_date()

        if len(description) > 0:
            new_model_description += DESCRIPTION_MD_TEMPLATE.format(description)
        else:
            new_model_description += DESCRIPTION_MD_TEMPLATE.format("N/A")

        self.client.update_registered_model(model_name, registered_model_description + new_model_description)
        self.client.update_model_version(model_name, version, single_model_description + new_model_description)

    def register_best_model(
        self,
        experiment_name: str,
        metric: str,
        model_name: str,
        tags: Optional[Dict[str, Any]] = None,
        description: str = "",
        mode: str = "max",
        model_path: str = "model",
    ) -> Optional[ModelVersion]:
        """Register the best model from an experiment.

        Args:
            experiment_name: The name of the experiment
            metric: The metric to use to determine the best model
            model_name: The name of the model after it is registered
            tags: A dictionary of tags to add to the model
            description: A description of the model, this will be added to the model changelog
            mode: The mode to use to determine the best model, either "max" or "min"
            model_path: The path to the model within the experiment run

        Returns:
            The registered model version if successful, otherwise None
        """
        if mode not in ["max", "min"]:
            raise ValueError(f"Mode must be either 'max' or 'min', got {mode}")

        experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id
        runs = self.client.search_runs(experiment_ids=[experiment_id])

        if len(runs) == 0:
            log.error("No runs found for experiment %s", experiment_name)
            return None

        best_run: Optional[Run] = None

        # We can only make comparisons if the model is on the top folder, otherwise just check if the folder exists
        # TODO: Is there a better way to do this?
        base_model_path = model_path.split("/")[0]

        for run in runs:
            run_artifacts = [x.path for x in self.client.list_artifacts(run.info.run_id) if x.path == base_model_path]

            if len(run_artifacts) == 0:
                # If we don't find the given model path, skip this run
                continue

            if best_run is None:
                # If we find a run with the model it must also have the metric
                if run.data.metrics.get(metric) is not None:
                    best_run = run
                continue

            if mode == "max":
                if run.data.metrics[metric] > best_run.data.metrics[metric]:
                    best_run = run
            else:
                if run.data.metrics[metric] < best_run.data.metrics[metric]:
                    best_run = run

        if best_run is None:
            log.error("No runs found for experiment %s with the given metric", experiment_name)
            return None

        best_model_uri = f"runs:/{best_run.info.run_id}/{model_path}"

        model_version = self.register_model(
            model_location=best_model_uri, model_name=model_name, tags=tags, description=description
        )

        return model_version

    @staticmethod
    def _get_author_and_date() -> str:
        """Get the author and date markdown template."""
        author_and_date = f"### Author: {getpass.getuser()}\n"
        author_and_date += f"### Date: {datetime.now().astimezone().strftime('%d/%m/%Y %H:%M:%S %Z')}\n"

        return author_and_date

    def _safe_get_stage(self, model_name: str, version: int) -> Optional[str]:
        """Get the stage of a model version.

        Args:
            model_name: The name of the model
            version: The version of the model

        Returns:
            The stage of the model version if it exists, otherwise None
        """
        try:
            model_stage = self.client.get_model_version(model_name, version).current_stage
            return model_stage
        except RestException:
            log.error("Model named %s with version %s does not exist", model_name, version)
            return None
