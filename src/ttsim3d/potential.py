"""Simulate electrostatic-potential volumes from atomic structures."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch
from torch_calculate_electrostatic_potential import (
    GridConfig,
    calculate_scattering_potential_3d,
    resolve_scattering_parameters,
)
from torch_fourier_rescale import fourier_rescale_3d  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from torch_structure_manipulation import AtomicStructure

ScatteringFactors = Literal["peng_elemental", "peng_bonded"]
BondedFallback = Literal["elemental", "error"]

#: Largest upsampled volume edge length chosen by :func:`auto_upsampling`.
MAX_UPSAMPLED_SIZE = 1536


def auto_upsampling(
    pixel_spacing: float,
    volume_shape: tuple[int, int, int],
    max_size: int = MAX_UPSAMPLED_SIZE,
) -> int:
    """Choose an upsampling factor for simulating at ``pixel_spacing``.

    Coarse pixel spacings are simulated on a finer grid and Fourier cropped back
    to reduce aliasing. The target factor is 4 above 1.5 Angstroms, 2 above
    0.75 Angstroms, and 1 otherwise. It is halved until the upsampled volume's
    largest edge is below ``max_size`` voxels.

    Parameters
    ----------
    pixel_spacing : float
        Final pixel spacing in Angstroms.
    volume_shape : tuple[int, int, int]
        Final volume shape in voxels.
    max_size : int
        Upper bound (exclusive) on the upsampled volume's largest edge.

    Returns
    -------
    int
        The upsampling factor: 1, 2, or 4.
    """
    if pixel_spacing > 1.5:
        upsampling = 4
    elif pixel_spacing > 0.75:
        upsampling = 2
    else:
        upsampling = 1

    largest_edge = max(volume_shape)
    while upsampling > 1 and largest_edge * upsampling >= max_size:
        upsampling //= 2

    return upsampling


def auto_sublattice_radius(
    params_b: torch.Tensor,
    b_factors: torch.Tensor,
    rel_tol: float = 1e-3,
) -> float:
    """Per-atom stencil radius capturing the widest Gaussian to ``rel_tol``.

    Each Peng term is, in real space, proportional to
    ``exp(-4 pi^2 r^2 / (b_i + B))``. The returned radius is where the widest
    term present (largest ``b_i + B``) has fallen to ``rel_tol`` of its peak::

        r = sqrt((max b_i + max B) * ln(1 / rel_tol)) / (2 pi)

    Parameters
    ----------
    params_b : torch.Tensor
        Peng ``b`` parameters of the atoms, in Angstroms squared, shape
        ``(..., n_atoms, n_terms)``.
    b_factors : torch.Tensor
        Atomic B-factors in Angstroms squared.
    rel_tol : float
        Relative value of the widest Gaussian at the stencil edge, in (0, 1).

    Returns
    -------
    float
        Stencil radius in Angstroms.
    """
    if not 0.0 < rel_tol < 1.0:
        raise ValueError(f"rel_tol must be in (0, 1), got {rel_tol}")
    width = float(params_b.max()) + float(b_factors.max())
    if width <= 0:
        raise ValueError(
            "The sum of the largest Peng b parameter and largest B-factor must be "
            f"positive, got {width}."
        )
    return math.sqrt(width * math.log(1.0 / rel_tol)) / (2.0 * math.pi)


def simulate_potential(
    structure: AtomicStructure,
    pixel_spacing: float,
    volume_shape: tuple[int, int, int],
    *,
    upsampling: int | None = None,
    scattering_factors: ScatteringFactors = "peng_elemental",
    bonded_fallback: BondedFallback = "elemental",
    sublattice_radius: float | None = None,
    batch_size: int = 4096,
    verbose: bool = False,
) -> torch.Tensor:
    """Simulate the electrostatic potential of ``structure`` on a 3D grid.

    The potential is evaluated with
    ``torch_calculate_electrostatic_potential`` (Peng 1996 scattering factors,
    averaged over each voxel) on a grid upsampled by ``upsampling``, then
    Fourier cropped back to ``volume_shape`` with the mean preserved. The
    physical origin is at voxel index ``N // 2`` along every axis.

    Parameters
    ----------
    structure : AtomicStructure
        Atoms to simulate, positions in Angstroms (ZYX). The output is computed
        on ``structure.device``.
    pixel_spacing : float
        Final pixel spacing in Angstroms.
    volume_shape : tuple[int, int, int]
        Final volume shape in voxels (ZYX).
    upsampling : int | None
        Integer upsampling factor for the simulation grid. None (default)
        chooses one with :func:`auto_upsampling`.
    scattering_factors : {"peng_elemental", "peng_bonded"}
        Scattering-factor model. ``"peng_bonded"`` requires a structure loaded
        with ``annotate_bonding=True``.
    bonded_fallback : {"elemental", "error"}
        Behavior for atoms without bonded scattering factors.
    sublattice_radius : float | None
        Per-atom stencil radius in Angstroms. None (default) uses
        :func:`auto_sublattice_radius` plus half an upsampled voxel.
    batch_size : int
        Number of atoms evaluated per chunk (memory/speed trade-off only).
    verbose : bool
        If True, show a progress bar over atom chunks.

    Returns
    -------
    torch.Tensor
        Potential in volts with shape ``(..., *volume_shape)``, where ``...``
        are any batch dimensions of ``structure``.
    """
    if pixel_spacing <= 0:
        raise ValueError(f"pixel_spacing must be positive, got {pixel_spacing}")
    if len(volume_shape) != 3 or any(int(n) < 1 for n in volume_shape):
        raise ValueError(
            f"volume_shape must be three positive integers, got {volume_shape}"
        )
    volume_shape = (int(volume_shape[0]), int(volume_shape[1]), int(volume_shape[2]))
    if upsampling is None:
        upsampling = auto_upsampling(pixel_spacing, volume_shape)
    elif upsampling < 1:
        raise ValueError(f"upsampling must be >= 1, got {upsampling}")
    if scattering_factors == "peng_bonded" and (
        structure.bonded_environments is None or structure.molecule_types is None
    ):
        raise ValueError(
            "scattering_factors='peng_bonded' requires bonding annotations; load "
            "the structure with load_structure(..., annotate_bonding=True)."
        )

    device = structure.device
    upsampled_spacing = pixel_spacing / upsampling
    upsampled_shape = (
        volume_shape[0] * upsampling,
        volume_shape[1] * upsampling,
        volume_shape[2] * upsampling,
    )

    params_a, params_b = resolve_scattering_parameters(
        structure.atomic_numbers,
        scattering_factors=scattering_factors,
        bonded_environments=structure.bonded_environments,
        molecule_types=structure.molecule_types,
        bonded_fallback=bonded_fallback,
        device=device,
        dtype=torch.float32,
    )

    if sublattice_radius is None:
        # Pad by half a voxel: the stencil is anchored on the voxel nearest to
        # each atom, which can be up to half a voxel away.
        sublattice_radius = (
            auto_sublattice_radius(params_b, structure.b_factors)
            + upsampled_spacing / 2
        )
    elif sublattice_radius <= 0:
        raise ValueError(f"sublattice_radius must be positive, got {sublattice_radius}")

    grid = GridConfig.from_fft_centered(
        upsampled_shape,
        (upsampled_spacing,) * 3,
        sublattice_radius=sublattice_radius,
        device=device,
    )
    volume = calculate_scattering_potential_3d(
        structure.positions_zyx,
        structure.b_factors,
        params_a,
        params_b,
        grid,
        atom_occupancies=structure.occupancies,
        per_voxel_averaging=True,
        batch_size=batch_size,
        verbose=verbose,
    )

    if upsampling > 1:
        volume, _ = fourier_rescale_3d(
            volume, target_shape=volume_shape, preserve_mean=True
        )

    return volume
