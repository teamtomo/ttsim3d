import pytest
import torch

from ttsim3d.device_handler import get_device


class TestGetDevice:
    """Tests for get_device function."""

    def test_cpu_device(self):
        """Test requesting CPU device."""
        devices = get_device("cpu")
        assert len(devices) == 1
        assert devices[0].type == "cpu"

    def test_cuda_all_devices(self):
        """Test requesting all available CUDA devices."""
        if torch.cuda.device_count() == 0:
            pytest.skip("CUDA not available")

        devices = get_device("all")
        assert len(devices) == torch.cuda.device_count()
        for i, device in enumerate(devices):
            assert device.type == "cuda"
            assert device.index == i

    def test_cuda_single_device_by_int(self):
        """Test requesting a single CUDA device by integer."""
        if torch.cuda.device_count() == 0:
            pytest.skip("CUDA not available")

        devices = get_device(0)
        assert len(devices) == 1
        assert devices[0].type == "cuda"
        assert devices[0].index == 0

    def test_cuda_single_device_by_string(self):
        """Test requesting a single CUDA device by string."""
        if torch.cuda.device_count() == 0:
            pytest.skip("CUDA not available")

        devices = get_device("cuda:0")
        assert len(devices) == 1
        assert devices[0].type == "cuda"
        assert devices[0].index == 0

    def test_cuda_multiple_devices_by_int_list(self):
        """Test requesting multiple CUDA devices by integer list."""
        if torch.cuda.device_count() < 2:
            pytest.skip("At least 2 CUDA devices required")

        devices = get_device([0, 1])
        assert len(devices) == 2
        assert devices[0].type == "cuda"
        assert devices[0].index == 0
        assert devices[1].type == "cuda"
        assert devices[1].index == 1

    def test_cuda_multiple_devices_by_string_list(self):
        """Test requesting multiple CUDA devices by string list."""
        if torch.cuda.device_count() < 2:
            pytest.skip("At least 2 CUDA devices required")

        devices = get_device(["cuda:0", "cuda:1"])
        assert len(devices) == 2
        assert devices[0].type == "cuda"
        assert devices[0].index == 0
        assert devices[1].type == "cuda"
        assert devices[1].index == 1

    def test_cuda_mixed_device_list(self):
        """Test requesting CUDA devices with mixed int and string formats."""
        if torch.cuda.device_count() < 2:
            pytest.skip("At least 2 CUDA devices required")

        devices = get_device([0, "cuda:1"])
        assert len(devices) == 2
        assert devices[0].type == "cuda"
        assert devices[0].index == 0
        assert devices[1].type == "cuda"
        assert devices[1].index == 1

    def test_invalid_device_type(self):
        """Test that invalid device type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid device selection"):
            get_device(3.14)  # float is invalid

    def test_invalid_device_dict(self):
        """Test that dict device type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid device selection"):
            get_device({"device": "cuda:0"})

    def test_empty_device_list(self):
        """Test that empty list raises AssertionError."""
        with pytest.raises(AssertionError, match="No valid devices found"):
            get_device([])

    def test_returns_list(self):
        """Test that function always returns a list."""
        devices = get_device("cpu")
        assert isinstance(devices, list)
        assert all(isinstance(d, torch.device) for d in devices)
