"""Serializable configuration for structure simulations."""

from __future__ import annotations

import os
from pathlib import Path  # noqa: TC003 (pydantic resolves field types at runtime)
from typing import TYPE_CHECKING, Annotated, Literal

import torch
from pydantic import Field, field_serializer, field_validator, model_validator
from teamtomo_basemodel import BaseModelTeamTomo
from torch_fourier_filter.mtf import MTF_REFERENCES

from ttsim3d.filters import apply_fourier_filters, dose_filter_3d, mtf_filter_3d
from ttsim3d.io import write_mrc
from ttsim3d.potential import simulate_potential
from ttsim3d.structure import adjust_b_factors, load_structure

if TYPE_CHECKING:
    from torch_structure_manipulation import AtomicStructure

Device = str | torch.device


class StructureConfig(BaseModelTeamTomo):
    """Which atomic model to simulate and how to prepare it.

    Attributes
    ----------
    path : Path
        PDB or mmCIF file.
    center : bool
        Translate the atoms so their mean position is at the origin (the
        volume center). Set to False for pre-aligned models.
    remove_hydrogens : bool
        Drop hydrogen atoms before simulating.
    b_factor_scale : float
        Multiplicative factor applied to every atomic B-factor.
    b_factor_offset : float
        B-factor (Angstroms squared) added after scaling.
    """

    path: Path
    center: bool = True
    remove_hydrogens: bool = True
    b_factor_scale: float = 1.0
    b_factor_offset: float = 0.0

    @field_serializer("path")
    def _serialize_path(self, value: Path) -> str:
        return str(value)


class PotentialConfig(BaseModelTeamTomo):
    """Options for evaluating the electrostatic potential.

    Field names follow ``torch_fit_in_map.PotentialSimulatorConfig``.

    Attributes
    ----------
    scattering_factors : {"peng_elemental", "peng_bonded"}
        Scattering-factor model. ``"peng_bonded"`` annotates bonding
        environments when the structure is loaded.
    bonded_fallback : {"elemental", "error"}
        Behavior for atoms without bonded scattering factors.
    sublattice_radius : float | None
        Per-atom stencil radius in Angstroms; None picks one from the
        B-factors (see ``ttsim3d.auto_sublattice_radius``).
    upsampling : int | None
        Simulation-grid upsampling factor; None picks one from the pixel
        spacing (see ``ttsim3d.auto_upsampling``).
    batch_size : int
        Atoms evaluated per chunk (memory/speed trade-off only).
    """

    scattering_factors: Literal["peng_elemental", "peng_bonded"] = "peng_elemental"
    bonded_fallback: Literal["elemental", "error"] = "elemental"
    sublattice_radius: Annotated[float, Field(gt=0.0)] | None = None
    upsampling: Annotated[int, Field(ge=1)] | None = None
    batch_size: Annotated[int, Field(ge=1)] = 4096


class DoseFilterConfig(BaseModelTeamTomo):
    """Cumulative radiation-damage filter (see ``ttsim3d.dose_filter_3d``).

    Attributes
    ----------
    start : float
        Start of the exposure range in electrons per square Angstrom.
    end : float
        End of the exposure range in electrons per square Angstrom.
    critical_bfactor : float | None
        B-factor defining the critical exposure; None uses Grant and
        Grigorieff (2015).
    modify_signal : {"none", "sqrt", "rel_diff"}
        Transformation applied to the filter values.
    """

    start: Annotated[float, Field(ge=0.0)] = 0.0
    end: Annotated[float, Field(gt=0.0)] = 30.0
    critical_bfactor: Annotated[float, Field(gt=0.0)] | None = None
    modify_signal: Literal["none", "sqrt", "rel_diff"] = "none"

    @model_validator(mode="after")
    def _check_exposure_range(self) -> DoseFilterConfig:
        if self.end <= self.start:
            raise ValueError(
                f"end ({self.end}) must be greater than start ({self.start})"
            )
        return self


class Simulator(BaseModelTeamTomo):
    """Simulate a filtered 3D potential volume from an atomic model.

    The model holds only serializable settings; every call re-reads the
    structure, so settings (e.g. ``pixel_spacing``) can be changed between
    runs.

    Attributes
    ----------
    structure : StructureConfig
        Atomic model and its preparation.
    pixel_spacing : float
        Pixel spacing of the output volume in Angstroms.
    volume_shape : tuple[int, int, int]
        Shape of the output volume in voxels (ZYX).
    potential : PotentialConfig
        Potential evaluation options.
    dose_filter : DoseFilterConfig | None
        Radiation-damage filter; None disables it.
    mtf : str | None
        Bundled detector MTF name (case-insensitive, e.g. ``"k3_standard_300kv_fl2"``)
        or path to a RELION MTF star file; None (default) disables it.

    Examples
    --------
    >>> sim = Simulator(
    ...     structure={"path": "model.pdb"},
    ...     pixel_spacing=1.0,
    ...     volume_shape=(256, 256, 256),
    ...     mtf="k3_standard_300kv_fl2",
    ... )
    >>> volume = sim.run(device="cuda")  # doctest: +SKIP
    """

    structure: StructureConfig
    pixel_spacing: Annotated[float, Field(gt=0.0)]
    volume_shape: tuple[
        Annotated[int, Field(gt=0)],
        Annotated[int, Field(gt=0)],
        Annotated[int, Field(gt=0)],
    ]
    potential: PotentialConfig = Field(default_factory=PotentialConfig)
    dose_filter: DoseFilterConfig | None = Field(default_factory=DoseFilterConfig)
    mtf: str | None = None

    @field_validator("mtf")
    @classmethod
    def _validate_mtf(cls, value: str | None) -> str | None:
        if value is None or os.path.exists(value):
            return value
        if value.lower() in MTF_REFERENCES:
            return value.lower()
        raise ValueError(
            f"mtf must be an existing star file or a bundled MTF reference "
            f"{sorted(MTF_REFERENCES)}, got {value!r}"
        )

    @field_serializer("volume_shape")
    def _serialize_volume_shape(self, value: tuple[int, int, int]) -> list[int]:
        return list(value)

    def load_structure(self, device: Device = "cpu") -> AtomicStructure:
        """Load and prepare the atomic model described by ``structure``."""
        structure = load_structure(
            self.structure.path,
            center=self.structure.center,
            remove_hydrogens=self.structure.remove_hydrogens,
            annotate_bonding=self.potential.scattering_factors == "peng_bonded",
            device=device,
        )
        return adjust_b_factors(
            structure,
            scale=self.structure.b_factor_scale,
            offset=self.structure.b_factor_offset,
        )

    def simulate_potential(self, device: Device = "cpu") -> torch.Tensor:
        """Unfiltered potential in volts, e.g. as input to ``torch_scattering``."""
        return simulate_potential(
            self.load_structure(device),
            self.pixel_spacing,
            self.volume_shape,
            upsampling=self.potential.upsampling,
            scattering_factors=self.potential.scattering_factors,
            bonded_fallback=self.potential.bonded_fallback,
            sublattice_radius=self.potential.sublattice_radius,
            batch_size=self.potential.batch_size,
        )

    def fourier_filters(self, device: Device = "cpu") -> list[torch.Tensor]:
        """The configured dose and MTF filters on the output volume's rfft grid."""
        filters = []
        if self.dose_filter is not None:
            filters.append(
                dose_filter_3d(
                    self.volume_shape,
                    self.pixel_spacing,
                    dose_start=self.dose_filter.start,
                    dose_end=self.dose_filter.end,
                    critical_bfactor=self.dose_filter.critical_bfactor,
                    modify_signal=self.dose_filter.modify_signal,
                    device=device,
                )
            )
        if self.mtf is not None:
            filters.append(mtf_filter_3d(self.volume_shape, self.mtf, device=device))
        return filters

    def run(self, device: Device = "cpu") -> torch.Tensor:
        """Simulate the potential and apply the configured filters.

        Parameters
        ----------
        device : str | torch.device
            Device to simulate on, e.g. ``"cpu"`` or ``"cuda:0"``.

        Returns
        -------
        torch.Tensor
            The filtered volume with shape ``volume_shape``.
        """
        volume = self.simulate_potential(device)
        return apply_fourier_filters(volume, *self.fourier_filters(device))

    def export_to_mrc(
        self, path: str | os.PathLike[str], device: Device = "cpu"
    ) -> None:
        """Run the simulation and write the result to an MRC file."""
        write_mrc(path, self.run(device), self.pixel_spacing)
