import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders.mix_transformer import mix_transformer_encoders

from quadra.utils.segmentation import QuadraMixVisionTransformerEncoder, patch_mix_transformer_encoder
from quadra.utils.tests.helpers import get_quadra_test_device


class TestPatchMixTransformerEncoder:
    def test_all_encoders_are_patched(self):
        patch_mix_transformer_encoder()
        for name in mix_transformer_encoders:
            assert smp.encoders.encoders[name]["encoder"] is QuadraMixVisionTransformerEncoder

    def test_patch_is_idempotent(self):
        patch_mix_transformer_encoder()
        patch_mix_transformer_encoder()
        for name in mix_transformer_encoders:
            assert smp.encoders.encoders[name]["encoder"] is QuadraMixVisionTransformerEncoder


class TestQuadraMixVisionTransformerEncoderForward:
    def _make_encoder(self, encoder_name: str = "mit_b0"):
        patch_mix_transformer_encoder()
        return smp.encoders.get_encoder(encoder_name, in_channels=3, depth=5, weights=None)

    def test_dummy_tensor_matches_input_device(self):
        device = get_quadra_test_device()
        encoder = self._make_encoder().to(device)
        x = torch.zeros(1, 3, 64, 64, device=device)
        features = encoder(x)
        # features[1] is the dummy placeholder for stage 0
        assert features[1].device.type == torch.device(device).type

    def test_dummy_tensor_matches_input_dtype(self):
        device = get_quadra_test_device()
        encoder = self._make_encoder().half().to(device)
        x = torch.zeros(1, 3, 64, 64, device=device, dtype=torch.float16)
        features = encoder(x)
        assert features[1].dtype == torch.float16

    def test_output_feature_count(self):
        device = get_quadra_test_device()
        encoder = self._make_encoder().to(device)
        x = torch.zeros(1, 3, 64, 64, device=device)
        features = encoder(x)
        # depth=5 → [input, dummy, stage1, stage2, stage3, stage4]
        assert len(features) == 6
