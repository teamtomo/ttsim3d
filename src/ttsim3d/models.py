"""Pydantic models for input parameters."""

import os
import pathlib
from typing import Annotated, Any, Optional

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
from pydantic.json_schema import SkipJsonSchema
from torch_fourier_filter.mtf import read_mtf

from ttsim3d.mrc_handler import tensor_to_mrc
from ttsim3d.pdb_handler import load_model, remove_hydrogens
from ttsim3d.simulate2d import simulate2d
from ttsim3d.simulate3d import ALLOWED_DOSE_FILTER_MODIFICATIONS, simulate3d

DEFAULT_MTF_REFERENCES = {
    "de20_300kv": "src/data/mtf_de20_300kV.star",
    "falcon2_300kv": "src/data/mtf_falcon2_300kV.star",
    "falcon3EC_200kv": "src/data/mtf_falcon3EC_200kV.star",
    "falcon3EC_300kv": "src/data/mtf_falcon3EC_300kV.star",
    "falcon4EC_200kv": "src/data/mtf_falcon4EC_200kV.star",
    "falcon4EC_300kv": "src/data/mtf_k2_300kV_FL2.star",
    "k2_300kv": "src/data/mtf_k2_300kV.star",
    "k2_200kV_FL2": "src/data/mtf_k2_200kV_FL2.star",
    "k2_300kV_FL2": "src/data/mtf_k2_300kV_FL2.star",
    "k3_200kV_FL2": "src/data/mtf_standard_k3_200kV_FL2.star",
    "k3_300kV_FL2": "src/data/mtf_standard_k3_300kV_FL2.star",
}

# Pydantic type annotation for large tensor excluded from JSON schema and dump
ExcludedTensor = SkipJsonSchema[
    Annotated[torch.Tensor, Field(default=None, exclude=True)]
]


class SimulatorConfig(BaseModel):
    """Configuration for simulating a 3D volume.

    These simulation parameters are intended to be model agnostic, that is,
    the configuration can be used between multiple structure simulations.

    Most model parameters are related to simulation filters to apply with a
    handful also included for storing intermediate results during the
    calculation.

    Attributes
    ----------
    voltage : float
        The voltage of the microscope in kV. Default is 300 kV.
    apply_dose_weighting : bool
        If True, apply dose weighting to the simulation.
    crit_exposure_bfactor : float
        B-factor to use in critical exposure calculations. The default is -1
        and corresponds to the fitted critical exposure function in Grant and
        Grigorieff, 2015.
    dose_filter_modify_signal : Literal["None", "sqrt", "rel_diff"]
        Signal modification to apply to the dose filter. Currently supports
        - 'None': No modification
        - 'sqrt': x' = sqrt(x)
        - 'rel_diff': x' = 1 - (1 - x) / (1 - x)
    dose_start : float
        The starting dose in e/A^2. The default is 0.0 e/A^2.
    dose_end : float
        The ending dose in e/A^2. The default is 30.0 e/A^2.
    apply_dqe : bool
        If True, apply a DQE filter to the simulation.
    mtf_reference : str
        Path to the modulation transfer function (MTF) reference star file, or
        one of the known MTF reference files in:
        - 'k2_300kV': The MTF reference for a K2 camera at 300 kV.
    upsampling : int
        The upsampling factor to apply to the simulation. The default is -1 and
        corresponds to automatic calculation of the upsampling factor.
    store_volume : bool
        If True, store the final simulated volume in real space after requested
        simulation filters are applied under the attribute `Simulator.volume`.
        Default is True.

    Methods
    -------
    model_dump -> dict
    """

    model_config = ConfigDict(validate_default=True)

    # Serializable attributes
    voltage: Annotated[float, Field(ge=0.0)] = 300.0
    apply_dose_weighting: Annotated[bool, Field(default=True)] = True
    crit_exposure_bfactor: float | int = -1
    dose_filter_modify_signal: str = "None"
    dose_start: Annotated[float, Field(ge=0.0)] = 0.0
    dose_end: Annotated[float, Field(ge=0.0)] = 30.0
    apply_dqe: bool = True
    mtf_reference: str = "k2_300kv"
    upsampling: int = -1
    store_volume: bool = True

    @field_validator("dose_filter_modify_signal")  # type: ignore
    def validate_dose_filter_modify_signal(cls, v):
        """Validate model input `dose_filter_modify_signal`."""
        if v not in ALLOWED_DOSE_FILTER_MODIFICATIONS:
            e = f"Invalid dose filter signal modification: {v}. "
            e += f"Allowed values are: {ALLOWED_DOSE_FILTER_MODIFICATIONS}"
            raise ValueError(e)
        return v

    @field_validator("mtf_reference")  # type: ignore
    def validate_mtf_reference(cls, v):
        """Validate model input `mtf_reference`."""
        _path_exists = os.path.exists(v)
        _is_default = v in DEFAULT_MTF_REFERENCES

        if not _path_exists and not _is_default:
            e = f"Invalid MTF reference file: {v}. "
            e += "Please provide a valid path to an MTF reference file."
            raise ValueError(e)

        if _is_default:
            return DEFAULT_MTF_REFERENCES[v]

        return v

    @property
    def mtf_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the MTF tensors from the reference file."""
        frequencies, amplitudes = read_mtf(file_path=self.mtf_reference)

        return frequencies, amplitudes


class Simulator(BaseModel):
    """Class for simulating a 3D volume from a atomistic structure.

    Attributes
    ----------
    pixel_spacing : float
        The pixel spacing of the simulated volume in units of Angstroms. Must
        be greater than 0, and defaults to 1.0 Angstroms.
    volume_shape : tuple[int, int, int]
        The shape of the simulated volume in pixels. The default is
        (400, 400, 400).
    pdb_filepath : pathlib.Path
        The path to the PDB file containing the atomic structure to simulate.
    b_factor_scaling : float
        The scaling factor to apply to the B-factors of the atoms in the pdb
        file. The default is 1.0.
    additional_b_factor : float
        Additional B-factor to apply to the atoms in the pdb file. The default
        is 0.0.
    simulator_config : SimulatorConfig
        Simulation configuration.
    atom_positions_zyx : torch.Tensor
        The positions (float tensor) of the atoms in the structure in units of
        Angstroms. Non-serializable attribute.
    atom_identities : torch.Tensor
        The atomic identities (str tensor) of the atoms in the structure.
        Non-serializable attribute.
    atom_b_factors : torch.Tensor
        The B-factors (float tensor) of the atoms in the structure in units of
        A^2. Non-serializable attribute.
    volume : torch.Tensor
        The simulated volume in real space after requested simulation filters
        are applied. Non-serializable attribute. Only stored if requested in
        the SimulatorConfig.

    Methods
    -------
    __init__ -> None
    load_atoms_from_pdb_model -> None
        Loads the structure atoms from the held pdb file into the attributes
        `atom_positions_zyx`, `atom_identities`, and `atom_b_factors`.
    get_scale_atom_b_factors -> torch.Tensor
        Returns the scaled b-factors of the atoms in the structure based on
        the attributes `b_factor_scaling` and `additional_b_factor`.
    run -> torch.Tensor
        Runs the simulation and returns the simulated volume.
    export_to_mrc -> None
        Exports the simulated volume to an MRC file.
    """

    # Pydantic model configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Serializable attributes
    pixel_spacing: Annotated[float, Field(ge=0.0)] = 1.0
    volume_shape: Annotated[tuple[int, int, int], Field(default=(400, 400, 400))] = (
        400,
        400,
        400,
    )
    pdb_filepath: Annotated[pathlib.Path, Field(...)]
    b_factor_scaling: Annotated[float, Field(default=1.0)] = 1.0
    additional_b_factor: Annotated[float, Field(default=0.0)] = 0.0
    simulator_config: SimulatorConfig

    # Non-serializable and schema-excluded attributes
    atom_positions_zyx: ExcludedTensor
    atom_identities: ExcludedTensor
    atom_b_factors: ExcludedTensor
    volume: ExcludedTensor

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        self.load_atoms_from_pdb_model()

    def load_atoms_from_pdb_model(self) -> None:
        """Loads the structure atoms from held pdb file."""
        atom_positions_zyx, atom_ids, atom_b_factors = load_model(self.pdb_filepath)
        atom_positions_zyx, atom_ids, atom_b_factors = remove_hydrogens(
            atom_positions_zyx, atom_ids, atom_b_factors
        )

        self.atom_positions_zyx = atom_positions_zyx
        self.atom_identities = atom_ids
        self.atom_b_factors = atom_b_factors

    def get_scale_atom_b_factors(self) -> torch.Tensor:
        """Returns b-factors transformed by the scale and additional b-factor.

        Parameters
        ----------
        None

        Returns
        -------
        b_fac: torch.Tensor
            The scaled b-factors.
        """
        if self.atom_b_factors is None:
            raise ValueError("No atom B-factors loaded.")

        b_fac = self.atom_b_factors * self.b_factor_scaling
        b_fac += self.additional_b_factor

        # NOTE (Josh): cisTEM includes a 0.25 factor. Unsure why, but keeping
        # consistent with the original implementation for now.
        b_fac *= 0.25

        return b_fac

    def run(
        self,
        gpu_ids: Optional[int | list[int]] = None,
        atom_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runs the simulation and returns the simulated volume.

        Parameters
        ----------
        gpu_ids: int | list[int]
            A list of GPU IDs to use for the simulation. The default is 'None'
            which will use the CPU. A value of '-1' will use all available
            GPUs, otherwise a list of integers greater than or equal to 0 are
            expected.
        atom_indices: torch.Tensor
            The indices of the atoms to simulate. The default is 'None' which
            will simulate all atoms in the structure.

        Returns
        -------
        volume: torch.Tensor
            The simulated volume.
        """
        assert self.atom_positions_zyx is not None, "No atom positions loaded."
        assert self.atom_identities is not None, "No atom identities loaded."

        if atom_indices is None:
            atom_indices = torch.arange(self.atom_positions_zyx.size(0))

        # Select GPUs to use, or use CPU
        # TODO: Implement GPU selection

        # Get the scaled atom b-factors
        atom_b_factors = self.get_scale_atom_b_factors()

        # Calculate the mtf_frequencies and mtf_amplitudes from reference file
        mtf_frequencies, mtf_amplitudes = self.simulator_config.mtf_tensors

        volume = simulate3d(
            atom_positions_zyx=self.atom_positions_zyx,
            atom_ids=self.atom_identities,
            atom_b_factors=atom_b_factors,
            beam_energy_kev=self.simulator_config.voltage,
            sim_pixel_spacing=self.pixel_spacing,
            sim_volume_shape=self.volume_shape,
            requested_upsampling=self.simulator_config.upsampling,
            apply_dose_weighting=self.simulator_config.apply_dose_weighting,
            dose_start=self.simulator_config.dose_start,
            dose_end=self.simulator_config.dose_end,
            dose_filter_modify_signal=self.simulator_config.dose_filter_modify_signal,  # type: ignore
            dose_filter_critical_bfactor=self.simulator_config.crit_exposure_bfactor,
            apply_dqe=self.simulator_config.apply_dqe,
            mtf_frequencies=mtf_frequencies,
            mtf_amplitudes=mtf_amplitudes,
        )

        if self.simulator_config.store_volume:
            self.volume = volume

        return volume

    def export_to_mrc(
        self,
        mrc_filepath: str | os.PathLike,
        gpu_ids: Optional[int | list[int]] = None,
        atom_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """Exports the simulated volume to an MRC file.

        Parameters
        ----------
        mrc_filepath: str | os.PathLike
            The file path to save the MRC file.
        gpu_ids: int | list[int]
            A list of GPU IDs to use for the simulation. The default is 'None'
            which will use the CPU. A value of '-1' will use all available
            GPUs, otherwise a list of integers greater than or equal to 0 are
            expected. The default is 'None'. This is passed to the `run`
            method.
        atom_indices: torch.Tensor
            The indices of the atoms to simulate. The default is 'None' which
            will simulate all atoms in the structure. This is passed to the
            `run` method.

        Returns
        -------
        None
        """
        volume = self.run(gpu_ids=gpu_ids, atom_indices=atom_indices)

        tensor_to_mrc(
            output_filename=str(mrc_filepath),
            final_volume=volume,
            sim_pixel_spacing=self.pixel_spacing,
        )


class Simulator2D(BaseModel):
    """Class for simulating a 2D image from a atomistic structure.

    Attributes
    ----------
    pixel_spacing : float
        The pixel spacing of the simulated volume in units of Angstroms. Must
        be greater than 0, and defaults to 1.0 Angstroms.
    image_shape : tuple[int, int]
        The shape of the simulated image in pixels. The default is
        (400, 400).
    pdb_filepath : pathlib.Path
        The path to the PDB file containing the atomic structure to simulate.
    b_factor_scaling : float
        The scaling factor to apply to the B-factors of the atoms in the pdb
        file. The default is 1.0.
    additional_b_factor : float
        Additional B-factor to apply to the atoms in the pdb file. The default
        is 0.0.
    simulator_config : SimulatorConfig
        Simulation configuration.
    atom_positions_yx : torch.Tensor
        The positions (float tensor) of the atoms in the structure in units of
        Angstroms. Non-serializable attribute.
    atom_identities : torch.Tensor
        The atomic identities (str tensor) of the atoms in the structure.
        Non-serializable attribute.
    atom_b_factors : torch.Tensor
        The B-factors (float tensor) of the atoms in the structure in units of
        A^2. Non-serializable attribute.
    image : torch.Tensor
        The simulated image in real space after requested simulation filters
        are applied. Non-serializable attribute. Only stored if requested in
        the SimulatorConfig.

    Methods
    -------
    __init__ -> None
    load_atoms_from_pdb_model -> None
        Loads the structure atoms from the held pdb file into the attributes
        `atom_positions_yx`, `atom_identities`, and `atom_b_factors`.
    get_scale_atom_b_factors -> torch.Tensor
        Returns the scaled b-factors of the atoms in the structure based on
        the attributes `b_factor_scaling` and `additional_b_factor`.
    run -> torch.Tensor
        Runs the simulation and returns the simulated volume.
    export_to_mrc -> None
        Exports the simulated volume to an MRC file.
    """

    # Pydantic model configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Serializable attributes
    pixel_spacing: Annotated[float, Field(ge=0.0)] = 1.0
    image_shape: Annotated[tuple[int, int], Field(default=(400, 400))] = (
        400,
        400,
    )
    pdb_filepath: Annotated[pathlib.Path, Field(...)]
    b_factor_scaling: Annotated[float, Field(default=1.0)] = 1.0
    additional_b_factor: Annotated[float, Field(default=0.0)] = 0.0
    simulator_config: SimulatorConfig

    # Non-serializable and schema-excluded attributes
    atom_positions_yx: ExcludedTensor
    atom_identities: ExcludedTensor
    atom_b_factors: ExcludedTensor
    image: ExcludedTensor

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        self.load_atoms_from_pdb_model()

    def load_atoms_from_pdb_model(self) -> None:
        """Loads the structure atoms from held pdb file."""
        atom_positions_zyx, atom_ids, atom_b_factors = load_model(self.pdb_filepath)
        atom_positions_zyx, atom_ids, atom_b_factors = remove_hydrogens(
            atom_positions_zyx, atom_ids, atom_b_factors
        )
        # Collapse z dimension to get 2D coordinates
        atom_positions_yx = atom_positions_zyx[:, 1:]  # Keep only y,x coordinates

        self.atom_positions_yx = atom_positions_yx
        self.atom_identities = atom_ids
        self.atom_b_factors = atom_b_factors

    def get_scale_atom_b_factors(self) -> torch.Tensor:
        """Returns b-factors transformed by the scale and additional b-factor.

        Parameters
        ----------
        None

        Returns
        -------
        b_fac: torch.Tensor
            The scaled b-factors.
        """
        if self.atom_b_factors is None:
            raise ValueError("No atom B-factors loaded.")

        b_fac = self.atom_b_factors * self.b_factor_scaling
        b_fac += self.additional_b_factor

        # NOTE (Josh): cisTEM includes a 0.25 factor. Unsure why, but keeping
        # consistent with the original implementation for now.
        b_fac *= 0.25

        return b_fac

    def run(
        self,
        gpu_ids: Optional[int | list[int]] = None,
        atom_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runs the simulation and returns the simulated volume.

        Parameters
        ----------
        gpu_ids: int | list[int]
            A list of GPU IDs to use for the simulation. The default is 'None'
            which will use the CPU. A value of '-1' will use all available
            GPUs, otherwise a list of integers greater than or equal to 0 are
            expected.
        atom_indices: torch.Tensor
            The indices of the atoms to simulate. The default is 'None' which
            will simulate all atoms in the structure.

        Returns
        -------
        volume: torch.Tensor
            The simulated volume.
        """
        assert self.atom_positions_yx is not None, "No atom positions loaded."
        assert self.atom_identities is not None, "No atom identities loaded."

        if atom_indices is None:
            atom_indices = torch.arange(self.atom_positions_yx.size(0))

        # Select GPUs to use, or use CPU
        # TODO: Implement GPU selection

        # Get the scaled atom b-factors
        atom_b_factors = self.get_scale_atom_b_factors()

        # Calculate the mtf_frequencies and mtf_amplitudes from reference file
        mtf_frequencies, mtf_amplitudes = self.simulator_config.mtf_tensors

        image = simulate2d(
            atom_positions_yx=self.atom_positions_yx,
            atom_ids=self.atom_identities,
            atom_b_factors=atom_b_factors,
            beam_energy_kev=self.simulator_config.voltage,
            sim_pixel_spacing=self.pixel_spacing,
            sim_image_shape=self.image_shape,
            requested_upsampling=self.simulator_config.upsampling,
            apply_dose_weighting=self.simulator_config.apply_dose_weighting,
            dose_start=self.simulator_config.dose_start,
            dose_end=self.simulator_config.dose_end,
            dose_filter_modify_signal=self.simulator_config.dose_filter_modify_signal,  # type: ignore
            dose_filter_critical_bfactor=self.simulator_config.crit_exposure_bfactor,
            apply_dqe=self.simulator_config.apply_dqe,
            mtf_frequencies=mtf_frequencies,
            mtf_amplitudes=mtf_amplitudes,
        )

        if self.simulator_config.store_volume:
            self.image = image

        return image

    def export_to_mrc(
        self,
        mrc_filepath: str | os.PathLike,
        gpu_ids: Optional[int | list[int]] = None,
        atom_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """Exports the simulated volume to an MRC file.

        Parameters
        ----------
        mrc_filepath: str | os.PathLike
            The file path to save the MRC file.
        gpu_ids: int | list[int]
            A list of GPU IDs to use for the simulation. The default is 'None'
            which will use the CPU. A value of '-1' will use all available
            GPUs, otherwise a list of integers greater than or equal to 0 are
            expected. The default is 'None'. This is passed to the `run`
            method.
        atom_indices: torch.Tensor
            The indices of the atoms to simulate. The default is 'None' which
            will simulate all atoms in the structure. This is passed to the
            `run` method.

        Returns
        -------
        None
        """
        image = self.run(gpu_ids=gpu_ids, atom_indices=atom_indices)

        tensor_to_mrc(
            output_filename=str(mrc_filepath),
            final_volume=image,
            sim_pixel_spacing=self.pixel_spacing,
        )
