import numpy as np
import segmentation_models_pytorch as smp
import skimage
import torch
from segmentation_models_pytorch.encoders.mix_transformer import MixVisionTransformerEncoder, mix_transformer_encoders
from skimage.morphology import medial_axis

from quadra.utils import utils

log = utils.get_logger(__name__)


# TODO: Waiting for the next release of segmentation_models_pytorch to fix dummy tensor creation.
# In v0.5.0 the dummy tensor device and dtype are recorded as constants in the TorchScript graph; this causes an error
# at inference on GPU when training was run in fp32, because the tensor device is fixed to CPU at trace time.
class QuadraMixVisionTransformerEncoder(MixVisionTransformerEncoder):
    """MixVisionTransformerEncoder variant with TorchScript-compatible dummy tensor creation."""

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # create dummy output for the first block
        batch_size, _, height, width = x.shape
        # In this way the dummy tensor has the same device and dtype as the input tensor,
        # but is not recorded as a constant in the graph when exporting to TorchScript
        dummy = x.new_empty([batch_size, 0, height // 2, width // 2])

        features = [x, dummy]

        if self._depth >= 2:
            x = self.patch_embed1(x)
            x = self.block1(x)
            x = self.norm1(x)
            x = x.contiguous()
            features.append(x)

        if self._depth >= 3:
            x = self.patch_embed2(x)
            x = self.block2(x)
            x = self.norm2(x)
            x = x.contiguous()
            features.append(x)

        if self._depth >= 4:
            x = self.patch_embed3(x)
            x = self.block3(x)
            x = self.norm3(x)
            x = x.contiguous()
            features.append(x)

        if self._depth >= 5:
            x = self.patch_embed4(x)
            x = self.block4(x)
            x = self.norm4(x)
            x = x.contiguous()
            features.append(x)

        return features


def smooth_mask(mask: np.ndarray) -> np.ndarray:
    """Smooths for segmentation.

    Args:
        mask: Input mask

    Returns:
        Smoothed mask
    """
    labeled_mask = skimage.measure.label(mask)
    labels = np.arange(0, np.max(labeled_mask) + 1)
    output_mask = np.zeros_like(mask).astype(np.float32)
    for l in labels:
        component_mask = labeled_mask == l
        _, distance = medial_axis(component_mask, return_distance=True)
        component_mask_norm = distance ** (1 / 2.2)
        component_mask_norm = (component_mask_norm - np.min(component_mask_norm)) / (
            np.max(component_mask_norm) - np.min(component_mask_norm)
        )
        output_mask += component_mask_norm
    output_mask = output_mask * mask
    return output_mask


def patch_mix_transformer_encoder():
    """Patch the smp encoder registry to replace MixVisionTransformerEncoder with QuadraMixVisionTransformerEncoder."""
    log.info("Patching MixVisionTransformerEncoder with QuadraMixVisionTransformerEncoder")
    for model_name in mix_transformer_encoders:
        smp.encoders.encoders[model_name]["encoder"] = QuadraMixVisionTransformerEncoder
        log.debug(
            "Patched %s encoder with QuadraMixVisionTransformerEncoder: %s",
            model_name,
            smp.encoders.encoders[model_name]["encoder"],
        )
