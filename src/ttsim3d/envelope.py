"""Matched template envelopes from per-frame B-factor dose weights.

A template searched against a frame-weighted sum should carry the sum's signal
envelope. The envelope comes from :mod:`torch_fourier_filter.frame_weights`: the
matched damage envelope of a frame weight table (the default curve, or a table refit
per dataset), times an optional smooth correction stored in a template envelope file.

Exposures here are physical, in e-/A^2: the default curve was fitted in physical
exposure at both 200 and 300 kV, so there is no voltage factor.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import mrcfile
import numpy as np
import torch

from ttsim3d.device_handler import get_device
from ttsim3d.mrc_handler import tensor_to_mrc

if TYPE_CHECKING:
    import os

    from torch_fourier_filter.frame_weights import FrameWeightTable, TemplateEnvelope

DEFAULT_FRAME_WEIGHTS = "default"
MATCHED_DOSE_PER_FRAME = 1.0  # e-/A^2, the frames the envelope is computed over


def _require_frame_weights() -> None:
    """Check that :mod:`torch_fourier_filter.frame_weights` imports.

    The functions below import from it locally, so the rest of ttsim3d works with a
    torch-fourier-filter that does not have it yet.
    """
    # TODO: import at module level once torch-fourier-filter is pinned to the
    # release that contains teamtomo/teamtomo#203 (torch_fourier_filter.frame_weights).
    try:
        import torch_fourier_filter.frame_weights  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "The 'matched' dose filter needs torch_fourier_filter.frame_weights, "
            "which the installed torch-fourier-filter does not have. Install a "
            "torch-fourier-filter that includes teamtomo/teamtomo#203."
        ) from err


def matched_frame_exposures(
    dose_start: float, dose_end: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frames of about 1 e-/A^2 covering ``[dose_start, dose_end]``.

    The same frames as ``TemplateEnvelope.default``: ``round(dose_end - dose_start)``
    equal fractions (at least one).

    Parameters
    ----------
    dose_start : float
        Exposure at the start of the first frame, in e-/A^2.
    dose_end : float
        Exposure at the end of the last frame, in e-/A^2.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(start, end)`` exposure of each frame.
    """
    if dose_end <= dose_start:
        raise ValueError(
            f"The matched envelope needs dose_end > dose_start; got dose_start="
            f"{dose_start} and dose_end={dose_end}."
        )
    _require_frame_weights()
    from torch_fourier_filter.frame_weights import frame_exposures

    n_frames = max(1, round((dose_end - dose_start) / MATCHED_DOSE_PER_FRAME))
    start, end = frame_exposures(
        n_frames, dose_start, (dose_end - dose_start) / n_frames
    )
    return start, end


def _warn_if_outside_table(
    table: FrameWeightTable, dose_start: float, dose_end: float
) -> None:
    """Warn when the dose range runs past the exposures a table was measured at."""
    lo = float(table.exposure_start.min())
    hi = float(table.exposure_end.max())
    tol = 1e-6
    if dose_start < lo - tol or dose_end > hi + tol:
        warnings.warn(
            f"The dose range {dose_start:g}-{dose_end:g} e-/A^2 runs past the frame "
            f"weight table's {lo:g}-{hi:g} e-/A^2; the table's first and last frame "
            "values are held constant outside it.",
            stacklevel=3,
        )


def load_template_envelope(
    dose_start: float,
    dose_end: float,
    frame_weights: str | os.PathLike | FrameWeightTable | None = None,
    template_envelope: str | os.PathLike | TemplateEnvelope | None = None,
) -> TemplateEnvelope:
    """The matched template envelope for a frame-weighted sum.

    Parameters
    ----------
    dose_start : float
        Exposure at the start of the sum, in e-/A^2. Not used with
        ``template_envelope``.
    dose_end : float
        Exposure at the end of the sum, in e-/A^2. Not used with
        ``template_envelope``.
    frame_weights : str | os.PathLike | FrameWeightTable | None
        ``"default"`` (or None) for the default per-frame B-factor curve, or a frame
        weight table (or the path to one saved as JSON), which is resampled to
        frames of 1 e-/A^2 over ``[dose_start, dose_end]``.
    template_envelope : str | os.PathLike | TemplateEnvelope | None
        A complete template envelope (or the path to one saved as JSON), with its
        own frame weights and optional correction. Takes precedence over
        ``frame_weights`` and the dose range.

    Returns
    -------
    TemplateEnvelope
        Call it on frequencies in 1/A, or use ``.grid(...)`` for a DFT grid.
    """
    _require_frame_weights()
    from torch_fourier_filter.frame_weights import (
        FrameWeightTable,
        TemplateEnvelope,
        default_frame_bfactors,
    )

    if template_envelope is not None:
        if isinstance(template_envelope, TemplateEnvelope):
            return template_envelope
        return TemplateEnvelope.from_json(Path(template_envelope))

    start, end = matched_frame_exposures(dose_start, dose_end)
    if frame_weights is None or (
        isinstance(frame_weights, str) and frame_weights == DEFAULT_FRAME_WEIGHTS
    ):
        table = default_frame_bfactors(start, end)
    else:
        measured = (
            frame_weights
            if isinstance(frame_weights, FrameWeightTable)
            else FrameWeightTable.from_json(Path(frame_weights))
        )
        _warn_if_outside_table(measured, dose_start, dose_end)
        table = measured.resample(start, end)

    return TemplateEnvelope(frame_weights=table)


def apply_template_envelope(
    volume: torch.Tensor,
    envelope: TemplateEnvelope,
    pixel_spacing: float,
) -> torch.Tensor:
    """Multiply a real-space volume's spectrum by a template envelope.

    Parameters
    ----------
    volume : torch.Tensor
        Real-space volume, shape ``(d, h, w)``.
    envelope : TemplateEnvelope
        The envelope to apply.
    pixel_spacing : float
        Voxel size of the volume in Angstroms.

    Returns
    -------
    torch.Tensor
        The filtered volume, same shape as ``volume``.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume; got shape {tuple(volume.shape)}.")
    shape = (int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2]))
    # .to(): the grid must sit on the volume's device even if a torch-grid-utils /
    # torch-fourier-filter build returns it on the CPU
    envelope_grid = envelope.grid(
        shape, pixel_spacing, rfft=True, fftshift=False, device=volume.device
    ).to(volume.device)
    volume_rfft = torch.fft.rfftn(volume, dim=(-3, -2, -1))
    volume_rfft *= envelope_grid
    filtered: torch.Tensor = torch.fft.irfftn(volume_rfft, s=shape, dim=(-3, -2, -1))
    return filtered


def _read_voxel_size(mrc: mrcfile.mrcfile.MrcFile, input_mrc: str) -> float:
    voxel_size = np.array(
        [mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=float
    )
    if np.any(voxel_size <= 0):
        raise ValueError(
            f"{input_mrc} has no voxel size in its header; pass pixel_spacing."
        )
    if not np.allclose(voxel_size, voxel_size[0], rtol=1e-4):
        raise ValueError(
            f"{input_mrc} has anisotropic voxels {tuple(voxel_size)}; only cubic "
            "voxels are supported."
        )
    return float(voxel_size[0])


def refilter_mrc(
    input_mrc: str | os.PathLike,
    output_mrc: str | os.PathLike,
    envelope: TemplateEnvelope,
    pixel_spacing: float | None = None,
    device: int | str = "cpu",
) -> torch.Tensor:
    """Apply a template envelope to a map on disk and write the result.

    The input map should be simulated without dose weighting (for example
    ``ttsim3d-cli --apply-dose-weighting False``), so that the envelope is the only
    dose filter on the output.

    Parameters
    ----------
    input_mrc : str | os.PathLike
        The map to filter.
    output_mrc : str | os.PathLike
        Where to write the filtered map; it gets the input's voxel size.
    envelope : TemplateEnvelope
        The envelope to apply, e.g. from :func:`load_template_envelope`.
    pixel_spacing : float | None
        Voxel size in Angstroms. Default: read from the input header.
    device : int | str
        Device for the FFTs. Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        The filtered volume.
    """
    with mrcfile.open(input_mrc, permissive=True) as mrc:
        if mrc.data is None or mrc.data.ndim != 3:
            raise ValueError(f"{input_mrc} does not hold a 3D volume.")
        if pixel_spacing is None:
            pixel_spacing = _read_voxel_size(mrc, str(input_mrc))
        data = np.array(mrc.data, dtype=np.float32)

    if pixel_spacing <= 0:
        raise ValueError(f"pixel_spacing must be positive; got {pixel_spacing}.")

    torch_device = get_device(device)[0]
    volume = torch.from_numpy(data).to(torch_device)
    filtered = apply_template_envelope(volume, envelope, pixel_spacing)
    tensor_to_mrc(
        output_filename=str(output_mrc),
        final_volume=filtered,
        sim_pixel_spacing=pixel_spacing,
    )
    return filtered
