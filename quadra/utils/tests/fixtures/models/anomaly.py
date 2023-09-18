import pytest
import torch
from anomalib.models.draem.torch_model import DraemModel
from anomalib.models.efficient_ad.torch_model import EfficientAdModel
from anomalib.models.padim.torch_model import PadimModel
from anomalib.models.patchcore.torch_model import PatchcoreModel


@pytest.fixture
def padim_resnet18():
    """Yield a padim model with resnet18 encoder."""
    yield PadimModel(
        input_size=[224, 224],  # TODO: This is hardcoded may be not a good idea
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pretrained_weights=None,
        tied_covariance=False,
        pre_trained=False,
    )


@torch.inference_mode()
def _initialize_patchcore_model(patchcore_model: PatchcoreModel, coreset_sampling_ratio: float = 0.1) -> PatchcoreModel:
    """Initialize a Patchcore model by simulating a training step.

    Args:
        patchcore_model: Patchcore model to initialize
        coreset_sampling_ratio: Coreset sampling ratio to use for the initialization

    Returns:
        Patchcore model with initialized memory bank
    """
    with torch.no_grad():
        training_features = None
        random_input = torch.rand([1, 3, *patchcore_model.input_size])

        if training_features is None:
            training_features = patchcore_model(random_input)
        else:
            training_features = torch.cat([training_features, patchcore_model(random_input)], dim=0)

        patchcore_model.eval()
        patchcore_model.subsample_embedding(training_features, sampling_ratio=coreset_sampling_ratio)

        # Simulate a memory bank with 5 images, at the current stage patchcore onnx export is not handling
        # large memory banks well, so we are using a small one for the benchmark
        memory_bank_number, memory_bank_n_features = patchcore_model.memory_bank.shape
        patchcore_model.memory_bank = torch.rand([5 * memory_bank_number, memory_bank_n_features])
        patchcore_model.train()

    return patchcore_model


@pytest.fixture
def patchcore_resnet18():
    """Yield a patchcore model with resnet18 encoder."""
    model = PatchcoreModel(
        input_size=[224, 224],  # TODO: This is hardcoded may be not a good idea
        backbone="resnet18",
        layers=["layer2", "layer3"],
        pre_trained=False,
    )

    yield _initialize_patchcore_model(model)


@pytest.fixture
def draem():
    """Yield a draem model."""
    yield DraemModel()


@pytest.fixture
def efficient_ad_small():
    """Yield a draem model."""

    class EfficientAdForwardWrapper(EfficientAdModel):
        """Wrap the forward method to avoid passing optional parameters."""

        def forward(self, x):
            return super().forward(x, None)

    model = EfficientAdForwardWrapper(
        teacher_out_channels=384,
        input_size=[256, 256],  # TODO: This is hardcoded may be not a good idea
        pretrained_teacher_type="nelson",
    )

    yield model
