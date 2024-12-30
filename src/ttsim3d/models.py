"""Pydantic models for input parameters."""

import os
from typing import Optional

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch_fourier_filter.dose_weight import cumulative_dose_filter_3d
from torch_fourier_filter.mtf import make_mtf_grid, read_mtf

from ttsim3d.grid_coords import fourier_rescale_3d_force_size
from ttsim3d.mrc_handler import tensor_to_mrc
from ttsim3d.pdb_handler import load_model, remove_hydrogens
from ttsim3d.simulate3d import (
    _calculate_lead_term,
    place_voxel_neighborhoods_in_volume,
    simulate_atomwise_scattering_potentials,
)

ALLOWED_DOSE_FILTER_MODIFICATIONS = ["None", "sqrt", "rel_diff"]
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


class SimulatorConfig(BaseModel):
    """Configuration for simulating a 3D volume.

    These simulation parameters are intended to be model agnostic, that is,
    the configuration can be used between multiple structure simulations.

    Most model parameters are related to simulation filters to apply with a
    handful also included for storing intermediate results during the
    calculation.

    Attributes
    ----------
    voltage: float
        The voltage of the microscope in kV. Default is 300 kV.
    apply_dose_weighting: bool
        If True, apply dose weighting to the simulation.
    crit_exposure_bfactor: float | int
        B-factor to use in critical exposure calculations. The default is -1
        and corresponds to the fitted critical exposure function in Grant and
        Grigorieff, 2015.
    dose_filter_modify_signal: str
        Signal modification to apply to the dose filter. Currently supports
        - 'None': No modification
        - 'sqrt': x' = sqrt(x)
        - 'rel_diff': x' = 1 - (1 - x) / (1 - x)
    dose_start: float
        The starting dose in e/A^2. The default is 0.0 e/A^2.
    dose_end: float
        The ending dose in e/A^2. The default is 30.0 e/A^2.
    apply_dqe: bool
        If True, apply a DQE filter to the simulation.
    mtf_reference: str
        Path to the modulation transfer function (MTF) reference star file, or
        one of the known MTF reference files in:
        - 'k2_300kV': The MTF reference for a K2 camera at 300 kV.
    upsampling: int
        The upsampling factor to apply to the simulation. The default is -1 and
        corresponds to automatic calculation of the upsampling factor.
    store_dose_filter: bool
        If True, store the dose filter in the simulator class under the
        attribute `Simulator.dose_filter`. Default is False.
    store_dqe_filter: bool
        If True, store the DQE filter in the simulator class under the
        attribute `Simulator.dqe_filter`. Default is False.
    store_neighborhood_atom_potentials: bool
        If True, store the calculated potentials around each of the atoms
        under the attribute `Simulator.neighborhood_atom_potentials` with the
        associated positions in the 3D volume under
        `Simulator.neighborhood_atom_positions`. Default is False.
    store_upsampled_volume_rfft: bool
        If True, store the real fast Fourier transform (rfft) of the upsampled
        volume before any filters are applied under the attribute
        `Simulator.upsampled_volume_rfft`. Default is False.
    store_volume: bool
        If True, store the final simulated volume in real space after requested
        simulation filters are applied under the attribute `Simulator.volume`.
        Default is True.

    Methods
    -------
    model_dump -> dict
    """

    # Serializable attributes
    voltage: float = Field(300.0, ge=0.0)
    apply_dose_weighting: bool = True
    crit_exposure_bfactor: float | int = -1
    dose_filter_modify_signal: str = "None"
    dose_start: float = Field(0.0, ge=0.0)
    dose_end: float = Field(30.0, ge=0.0)
    apply_dqe: bool = True
    mtf_reference: str = "k2_300kV"
    upsampling: int = -1

    store_dose_filter: bool = True
    store_dqe_filter: bool = True
    store_neighborhood_atom_potentials: bool = False
    store_upsampled_volume_rfft: bool = False
    store_volume: bool = True

    @field_validator("crit_exposure_bfactor")  # type: ignore
    def validate_crit_exposure_bfactor(cls, v):
        """Validate model input `crit_exposure_bfactor`."""
        if not isinstance(v, (float, int)):
            e = f"Invalid critical exposure B-factor of type: {v}."
            e += "Expected a float or integer."
            raise ValueError(e)

        return v

    @field_validator("dose_filter_modify_signal")  # type: ignore
    def validate_dose_filter_modify_signal(cls, v):
        """Validate model input `dose_filter_modify_signal`."""
        if v not in ALLOWED_DOSE_FILTER_MODIFICATIONS:
            e = f"Invalid dose filter signal modification: {v}."
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

    # def model_dump(self) -> dict:
    #     """Return the model as a dictionary."""
    #     return self.dict()


class Simulator(BaseModel):
    """Class for simulating a 3D volume from a atomistic structure.

    Attributes
    ----------
    pixel_spacing: float
        The pixel spacing of the simulated volume in units of Angstroms. Must
        be greater than 0, and defaults to 1.0 Angstroms.
    volume_shape: tuple[int, int, int]
        The shape of the simulated volume in pixels. The default is
        (400, 400, 400).
    pdb_filepath: str
        The path to the PDB file containing the atomic structure to simulate.
    b_factor_scaling: float
        The scaling factor to apply to the B-factors of the atoms in the pdb
        file. The default is 1.0.
    additional_b_factor: float
        Additional B-factor to apply to the atoms in the pdb file. The default
        is 0.0.
    simulator_config: SimulatorConfig
        Simulation configuration.
    atom_positions_zyx: torch.Tensor
        The positions (float tensor) of the atoms in the structure in units of
        Angstroms. Non-serializable attribute.
    atom_identities: torch.Tensor
        The atomic identities (str tensor) of the atoms in the structure.
        Non-serializable attribute.
    atom_b_factors: torch.Tensor
        The B-factors (float tensor) of the atoms in the structure in units of
        A^2. Non-serializable attribute.
    upsampled_shape: tuple[int, int, int]
        The shape of the upsampled volume. Non-serializable attribute.
    upsampled_pixel_size: float
        The pixel size of the upsampled volume in units of Angstroms.
        Non-serializable attribute.
    actual_upsampling: int
        The actual upsampling factor applied to the simulation.
        Non-serializable attribute.
    neighborhood_atom_potentials: torch.Tensor
        The calculated potentials around each atom in the structure.
        Non-serializable attribute. Only stored if requested in the
        SimulatorConfig.
    neighborhood_atom_positions: torch.Tensor
        The voxel positions of the calculated potentials around each atom in
        the structure. Non-serializable attribute. Only stored if requested in
        the SimulatorConfig.
    dose_filter: torch.Tensor
        The dose filter applied to the simulation. Non-serializable attribute.
        Only stored if `SimulatorConfig.apply_dose_weighting` is True and
        storage requested in the SimulatorConfig.
    dqe_filter: torch.Tensor
        The DQE filter applied to the simulation. Non-serializable attribute.
        Only stored if `SimulatorConfig.apply_dqe` is True and storage
        requested in the SimulatorConfig.
    upsampled_volume_rfft: torch.Tensor
        The real fast Fourier transform of the upsampled volume before any
        filters are applied. Non-serializable attribute. Only stored if
        requested in the SimulatorConfig.
    volume: torch.Tensor
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
    pixel_spacing: float = Field(1.0, ge=0.0)
    volume_shape: tuple[int, int, int]
    pdb_filepath: str
    b_factor_scaling: float = 1.0
    additional_b_factor: float = 0.0
    simulator_config: SimulatorConfig

    # Non-serializable attributes
    atom_positions_zyx: Optional[torch.Tensor] = Field(None, exclude=True)
    atom_identities: Optional[torch.Tensor] = Field(None, exclude=True)
    atom_b_factors: Optional[torch.Tensor] = Field(None, exclude=True)
    upsampled_shape: Optional[tuple[int, int, int]] = Field(None, exclude=True)
    upsampled_pixel_size: Optional[float] = Field(None, exclude=True)
    actual_upsampling: Optional[int] = Field(None, exclude=True)
    neighborhood_atom_potentials: Optional[torch.Tensor] = Field(None, exclude=True)
    neighborhood_atom_positions: Optional[torch.Tensor] = Field(None, exclude=True)
    dose_filter: Optional[torch.Tensor] = Field(None, exclude=True)
    dqe_filter: Optional[torch.Tensor] = Field(None, exclude=True)
    upsampled_volume_rfft: Optional[torch.Tensor] = Field(None, exclude=True)
    volume: Optional[torch.Tensor] = Field(None, exclude=True)

    def __init__(self) -> None:
        super().__init__()

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
            atom_indices = torch.arange(self.atom_identities.size(0))

        # 1. Select GPUs to use, or use CPU
        # TODO: Implement GPU selection

        # 2. Get the scaled atom b-factors
        b_factors = self.get_scale_atom_b_factors()

        # 3. Calculate the potentials around each atom
        lead_term = _calculate_lead_term(
            beam_energy_kev=self.simulator_config.voltage,
            sim_pixel_spacing=self.pixel_spacing,
        )
        scattering_results = simulate_atomwise_scattering_potentials(
            atom_positions_zyx=self.atom_positions_zyx[atom_indices],
            atom_ids=self.atom_identities[atom_indices],
            atom_b_factors=b_factors[atom_indices],
            sim_pixel_spacing=self.pixel_spacing,
            sim_volume_shape=self.volume_shape,
            lead_term=lead_term,
            upsampling=self.simulator_config.upsampling,  # requested upsampling
        )

        neighborhood_potentials = scattering_results["neighborhood_potentials"]
        voxel_positions = scattering_results["voxel_positions"]
        upsampled_shape = scattering_results["upsampled_shape"]
        upsampled_pixel_size = scattering_results["upsampled_pixel_size"]
        upsampling = scattering_results["upsampling"]

        self.upsampled_shape = upsampled_shape
        self.upsampled_pixel_size = upsampled_pixel_size
        self.actual_upsampling = upsampling

        if self.simulator_config.store_neighborhood_atom_potentials:
            self.neighborhood_atom_potentials = neighborhood_potentials
            self.neighborhood_atom_positions = voxel_positions

        # 4. Place the potentials into the upsampled volume
        upsampled_volume = torch.zeros(upsampled_shape, dtype=torch.float32)
        upsampled_volume = place_voxel_neighborhoods_in_volume(
            neighborhood_potentials=neighborhood_potentials,
            voxel_positions=voxel_positions,
            final_volume=upsampled_volume,
        )

        # 5. Calculate the dose filter
        if self.simulator_config.apply_dose_weighting:
            crit_exp_bf = self.simulator_config.crit_exposure_bfactor
            dose_filter = cumulative_dose_filter_3d(
                volume_shape=upsampled_shape,
                pixel_size=self.upsampled_pixel_size,
                start_exposure=self.dose_start,
                end_exposure=self.dose_end,
                crit_exposure_bfactor=crit_exp_bf,
                rfft=True,
                fftshift=False,
            )

            if self.simulator_config.modify_signal == "sqrt":
                dose_filter = torch.sqrt(dose_filter)
            elif self.simulator_config.modify_signal == "rel_diff":
                eps = 1e-10
                denominator = torch.clamp(1 + dose_filter, min=eps)
                tmp = 1 - (1 - dose_filter) / denominator

                if torch.any(torch.isnan(tmp)):
                    print("Warning: NaN values encountered in dose filter")
                    tmp = torch.nan_to_num(tmp, nan=1.0)

                dose_filter = tmp

            if self.simulator_config.store_dose_filter:
                self.dose_filter = dose_filter

        # 6. Calculate the DQE filter
        if self.simulator_config.apply_dqe:
            mtf_frequencies, mtf_amplitudes = read_mtf(
                file_path=self.simulator_config.mtf_reference
            )
            dqe_filter = make_mtf_grid(
                image_shape=self.volume_shape,
                mtf_frequencies=mtf_frequencies,  # 1D tensor
                mtf_amplitudes=mtf_amplitudes,  # 1D tensor
                rfft=True,
                fftshift=False,
            )

            if self.simulator_config.store_dqe_filter:
                self.dqe_filter = dqe_filter

        # 7. Apply the simulation filters
        upsampled_volume_rfft = torch.fft.rfftn(upsampled_volume, dim=(-3, -2, -1))

        if self.simulator_config.store_upsampled_volume_rfft:
            self.upsampled_volume_rfft = upsampled_volume_rfft

        if self.simulator_config.apply_dose_weighting:
            upsampled_volume_rfft *= dose_filter

        volume_rfft = fourier_rescale_3d_force_size(
            volume_fft=upsampled_volume_rfft,
            volume_shape=upsampled_shape,
            target_size=self.volume_shape[0],  # TODO: pass as tuple
            rfft=True,
            fftshift=False,
        )

        if self.simulator_config.apply_dqe:
            volume_rfft *= dqe_filter

        volume = torch.fft.irfftn(volume_rfft, dim=(-3, -2, -1), s=self.volume_shape)
        volume = torch.fft.ifftshift(volume, dim=(-3, -2, -1))

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
