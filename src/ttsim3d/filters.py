"""Fourier-space filters applied to simulated volumes.

Each ``*_filter_3d`` function returns a real-valued filter on the
non-fftshifted rfft grid of a volume. Pass one or more of them to
:func:`apply_fourier_filters`, which performs a single forward/inverse FFT.
"""

from __future__ import annotations

import os
from typing import Literal

import torch
from torch_fourier_filter.dose_weight import cumulative_dose_filter_3d
from torch_fourier_filter.mtf import make_mtf_grid, read_mtf

DoseSignalModification = Literal["none", "sqrt", "rel_diff"]
MTFSpec = str | os.PathLike[str] | tuple[torch.Tensor, torch.Tensor]


def dose_filter_3d(
    shape: tuple[int, int, int],
    pixel_spacing: float,
    *,
    dose_start: float = 0.0,
    dose_end: float = 30.0,
    critical_bfactor: float | None = None,
    modify_signal: DoseSignalModification = "none",
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Cumulative radiation-damage filter integrated over an exposure range.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Real-space shape of the volume to filter.
    pixel_spacing : float
        Pixel spacing in Angstroms.
    dose_start, dose_end : float
        Start and end of the exposure range in electrons per square Angstrom.
    critical_bfactor : float | None
        B-factor (Angstroms squared) defining the critical exposure. None
        (default) uses the fitted critical exposure of Grant and Grigorieff
        (2015).
    modify_signal : {"none", "sqrt", "rel_diff"}
        Transformation of the filter values ``f``:

        - ``"none"``: ``f``
        - ``"sqrt"``: ``sqrt(f)``
        - ``"rel_diff"``: ``1 - (1 - f) / (1 + f)``
    device : torch.device | str | None
        Device of the returned filter.

    Returns
    -------
    torch.Tensor
        Filter on the rfft grid, shape ``(d, h, w // 2 + 1)``.

    Notes
    -----
    ``torch_fourier_filter.dose_weight.cumulative_dose_filter_3d`` (v0.6) is 0
    at zero frequency, so a dose-filtered volume has zero mean (as in previous
    ttsim3d releases). The physical limit there is
    ``(dose_end - dose_start) / dose_end``.
    """
    if dose_start < 0:
        raise ValueError(f"dose_start must be non-negative, got {dose_start}")
    if dose_end <= dose_start:
        raise ValueError(
            f"dose_end ({dose_end}) must be greater than dose_start ({dose_start})"
        )
    if critical_bfactor is not None and critical_bfactor <= 0:
        raise ValueError(
            f"critical_bfactor must be positive or None, got {critical_bfactor}"
        )

    dose_filter = cumulative_dose_filter_3d(
        volume_shape=shape,
        pixel_size=pixel_spacing,
        start_exposure=dose_start,
        end_exposure=dose_end,
        crit_exposure_bfactor=-1 if critical_bfactor is None else critical_bfactor,
        rfft=True,
        fftshift=False,
        device=_as_device(device),
    )

    if modify_signal == "none":
        return dose_filter
    if modify_signal == "sqrt":
        return torch.sqrt(dose_filter)
    if modify_signal == "rel_diff":
        return 1 - (1 - dose_filter) / (1 + dose_filter)
    raise ValueError(
        f"modify_signal must be 'none', 'sqrt', or 'rel_diff', got {modify_signal!r}"
    )


def mtf_filter_3d(
    shape: tuple[int, int, int],
    mtf: MTFSpec,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Detector modulation transfer function (MTF) filter.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Real-space shape of the volume to filter.
    mtf : str | os.PathLike | tuple[torch.Tensor, torch.Tensor]
        Name of a bundled MTF reference (case-insensitive, see
        ``torch_fourier_filter.mtf.list_mtf_references``), a path to a RELION
        MTF star file, or a ``(frequencies, amplitudes)`` pair with frequencies
        in cycles per pixel (0 to 0.5).
    device : torch.device | str | None
        Device of the returned filter.

    Returns
    -------
    torch.Tensor
        Filter on the rfft grid, shape ``(d, h, w // 2 + 1)``.
    """
    if isinstance(mtf, tuple):
        frequencies, amplitudes = mtf
    else:
        frequencies, amplitudes = read_mtf(mtf)

    if frequencies.shape != amplitudes.shape or frequencies.ndim != 1:
        raise ValueError(
            "MTF frequencies and amplitudes must be 1D tensors of the same shape, "
            f"got {tuple(frequencies.shape)} and {tuple(amplitudes.shape)}"
        )

    resolved_device = _as_device(device)
    return make_mtf_grid(
        image_shape=shape,
        mtf_frequencies=frequencies.to(resolved_device),
        mtf_amplitudes=amplitudes.to(resolved_device),
        rfft=True,
        fftshift=False,
        device=resolved_device,
    )


def apply_fourier_filters(volume: torch.Tensor, *filters: torch.Tensor) -> torch.Tensor:
    """Multiply the rfft of ``volume`` by each filter and transform back.

    Parameters
    ----------
    volume : torch.Tensor
        Real-space volume(s), shape ``(..., d, h, w)``.
    *filters : torch.Tensor
        Filters on the non-fftshifted rfft grid, each broadcastable to
        ``(..., d, h, w // 2 + 1)``.

    Returns
    -------
    torch.Tensor
        The filtered volume, same shape as ``volume``. Returned unchanged when
        no filters are given.
    """
    if not filters:
        return volume

    dims = (-3, -2, -1)
    dft = torch.fft.rfftn(volume, dim=dims)
    for fourier_filter in filters:
        dft = dft * fourier_filter.to(dft.device)

    filtered: torch.Tensor = torch.fft.irfftn(dft, s=volume.shape[-3:], dim=dims)
    return filtered


def _as_device(device: torch.device | str | None) -> torch.device | None:
    return None if device is None else torch.device(device)
