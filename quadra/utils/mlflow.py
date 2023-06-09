try:
    from mlflow.models import infer_signature  # noqa
    from mlflow.models.signature import ModelSignature  # noqa

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from typing import List, Optional, TypeVar

import torch
from torch import nn

NnModuleT = TypeVar("NnModuleT", bound=nn.Module)


@torch.inference_mode()
def infer_signature_torch(model: NnModuleT, data: List[torch.Tensor]) -> Optional[ModelSignature]:
    """Infer signature for a PyTorch/Torchscript model."""
    from quadra.utils.utils import get_logger  # pylint: disable=[import-outside-toplevel]

    log = get_logger(__name__)

    model = model.eval()
    model_output = model(*data)

    if isinstance(model_output, tuple):
        # Mlflow currently does not support tuple outputs, so we use a dict instead
        model_output = {f"output_{i}": x.cpu().numpy() for i, x in enumerate(model_output)}
    elif isinstance(model_output, torch.Tensor):
        model_output = model_output.cpu().numpy()
    elif isinstance(model_output, dict):
        model_output = {k: v.cpu().numpy() for k, v in model_output.items()}
    else:
        log.warning("Unable to infer signature for model output type %s", type(model_output))
        return None

    if len(data) == 1:
        signature_input = data[0].cpu().numpy()
    else:
        signature_input = {f"input_{i}": x.cpu().numpy() for i, x in enumerate(data)}

    return infer_signature(signature_input, model_output)
