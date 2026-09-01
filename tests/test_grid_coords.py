import pytest
import torch

from ttsim3d.grid_coords import (
    fourier_rescale_3d_force_size,
    get_upsampling,
)


class TestGetUpsampling:
    """Tests for get_upsampling function."""

    def test_upsampling_4x(self):
        """Test 4x upsampling for large pixel size."""
        assert get_upsampling(wanted_pixel_size=2.0, wanted_output_size=300) == 4

    def test_upsampling_2x(self):
        """Test 2x upsampling for medium pixel size."""
        assert get_upsampling(wanted_pixel_size=1.0, wanted_output_size=400) == 2

    def test_upsampling_1x(self):
        """Test 1x upsampling for small pixel size."""
        assert get_upsampling(wanted_pixel_size=0.5, wanted_output_size=500) == 1

    def test_upsampling_respects_max_size(self):
        """Test that upsampling respects max_size constraint."""
        # Should not return 4x if it exceeds max_size
        result = get_upsampling(
            wanted_pixel_size=2.0, wanted_output_size=500, max_size=1536
        )
        assert result < 4


class TestFourierRescale3dForceSize:
    """Tests for fourier_rescale_3d_force_size function."""

    def test_rfft_mode_cropping(self):
        """Test rfft=True mode crops correctly from the end of last dimension."""
        # Create a simple real FFT volume (shape: (10, 10, 6) for rfft)
        volume_fft = torch.randn(10, 10, 6, dtype=torch.complex64)
        original_shape = (10, 10, 10)
        target_size = 4

        result = fourier_rescale_3d_force_size(
            volume_fft,
            original_shape,
            target_size,
            rfft=True,
            fftshift=True,
        )

        # Check that result has the expected shape
        assert result.ndim == 3
        # First two dimensions should be target_size
        assert result.shape[0] == target_size
        assert result.shape[1] == target_size
        # Last dimension should be smaller (rfft dimension)
        assert result.shape[2] <= target_size

    def test_non_rfft_mode_cropping(self):
        """Test rfft=False mode crops all dimensions equally."""
        # Create a simple FFT volume
        volume_fft = torch.randn(10, 10, 10, dtype=torch.complex64)
        original_shape = (10, 10, 10)
        target_size = 4

        result = fourier_rescale_3d_force_size(
            volume_fft,
            original_shape,
            target_size,
            rfft=False,
            fftshift=True,
        )

        # Check that result has the expected shape (all dimensions equal)
        assert result.shape == (target_size, target_size, target_size)

    def test_rfft_vs_non_rfft_shapes_differ(self):
        """Test that rfft and non-rfft modes produce different output shapes."""
        volume_fft_rfft = torch.randn(10, 10, 6, dtype=torch.complex64)
        volume_fft_non_rfft = torch.randn(10, 10, 10, dtype=torch.complex64)
        original_shape = (10, 10, 10)
        target_size = 4

        result_rfft = fourier_rescale_3d_force_size(
            volume_fft_rfft,
            original_shape,
            target_size,
            rfft=True,
            fftshift=True,
        )

        result_non_rfft = fourier_rescale_3d_force_size(
            volume_fft_non_rfft,
            original_shape,
            target_size,
            rfft=False,
            fftshift=True,
        )

        # Shapes should differ (especially in the last dimension)
        assert result_rfft.shape != result_non_rfft.shape

    def test_non_cubic_volume_raises_error(self):
        """Test that non-cubic volumes raise AssertionError."""
        volume_fft = torch.randn(10, 12, 10, dtype=torch.complex64)
        with pytest.raises(AssertionError, match="Volume must be cubic"):
            fourier_rescale_3d_force_size(
                volume_fft, (10, 12, 10), target_size=4, rfft=False
            )

    def test_negative_target_size_raises_error(self):
        """Test that negative target size raises AssertionError."""
        volume_fft = torch.randn(10, 10, 10, dtype=torch.complex64)
        with pytest.raises(AssertionError, match="Target size must be positive"):
            fourier_rescale_3d_force_size(
                volume_fft, (10, 10, 10), target_size=-4, rfft=False
            )

    def test_zero_target_size_raises_error(self):
        """Test that zero target size raises AssertionError."""
        volume_fft = torch.randn(10, 10, 10, dtype=torch.complex64)
        with pytest.raises(AssertionError, match="Target size must be positive"):
            fourier_rescale_3d_force_size(
                volume_fft, (10, 10, 10), target_size=0, rfft=False
            )
