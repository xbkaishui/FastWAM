import torch
import pytest
import sys
# sys.path.insert(0, "/root/foresee/FastWAM/src")

from fastwam.datasets.lerobot.utils.normalizer import CustomActionFieldNormalizer

STATS_FILE = "/root/autodl-fs/datasets/pangceban_left_right_data_20260602_658_delta_action/meta/stats.json"


class TestCustomActionFieldNormalizer:
    """Tests for CustomActionFieldNormalizer."""

    def test_forward_backward_roundtrip_zscore(self):
        """Test that forward + backward is identity (z-score mode)."""
        norm = CustomActionFieldNormalizer()
        x = torch.randn(4, 32, 28)  # batch of 4
        normalized = norm.forward(x)
        recovered = norm.backward(normalized)
        assert torch.allclose(x, recovered, atol=1e-5), \
            f"Max diff: {(x - recovered).abs().max().item()}"

    def test_forward_backward_roundtrip_minmax(self):
        """Test that forward + backward is identity (min/max mode)."""
        norm = CustomActionFieldNormalizer(mode="min/max")
        x = torch.randn(4, 32, 28)
        normalized = norm.forward(x)
        recovered = norm.backward(normalized)
        assert torch.allclose(x, recovered, atol=1e-5), \
            f"Max diff: {(x - recovered).abs().max().item()}"

    def test_zscore_normalizes_mean(self):
        """Test that z-score normalization of the mean gives ~0."""
        norm = CustomActionFieldNormalizer(mode="z-score")
        # Feeding the mean should produce values close to 0
        x = norm.mean.unsqueeze(0)  # [1, 32, 28]
        normalized = norm.forward(x)
        assert normalized.abs().max().item() < 1e-5

    def test_output_shape_preserved(self):
        """Test that output shape matches input shape."""
        norm = CustomActionFieldNormalizer(STATS_FILE, mode="z-score")
        for shape in [(32, 28), (1, 32, 28), (8, 32, 28)]:
            x = torch.randn(*shape)
            out = norm.forward(x)
            assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

    def test_invalid_mode_raises(self):
        """Test that unsupported mode raises ValueError."""
        with pytest.raises(ValueError):
            CustomActionFieldNormalizer(STATS_FILE, mode="invalid_mode")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
