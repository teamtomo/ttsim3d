"""Comparison of simulation to static simulated .mrc file.

NOTE: If the core algorithm for the simulation changes, then this test will
fail and the reference "good" simulated structure will need to be updated.
"""

import os
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import requests
import torch
from torch_fourier_filter.mtf import read_mtf

from ttsim3d.pdb_handler import load_model, remove_hydrogens
from ttsim3d.simulate3d import simulate3d

# Remote filepaths for testing
PDB_STRUCTURE_FILEPATH = "https://zenodo.org/records/14219436/files/parsed_6Q8Y_whole_LSU_match3.pdb?download=1"
SIMULATED_MRC_FILEPATH = "https://zenodo.org/records/14219436/files/parsed_6Q8Y_whole_LSU_match3_37dff6e22103f08d3453b0163435476c2808d08b.mrc?download=1"
DQE_STARFILE_FILEPATH = (
    "https://zenodo.org/records/14219436/files/mtf_k2_300kV.star?download=1"
)


# Temporary paths
this_path = Path(__file__).resolve()
this_dir = this_path.parent
tmp_dir = this_dir / "tmp"


def download_file(url: str, dest_folder: str) -> str:
    """Download a file from a URL to a destination folder."""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    filename = url.split("/")[-1].split("?")[0]
    file_path = os.path.join(dest_folder, filename)

    # Skip download if file already exists
    if os.path.exists(file_path):
        return file_path

    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad status codes
    # total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    with open(file_path, "wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)

    # print(f"Downloaded {file_path}")
    return file_path


def setup_simulation():
    """Download necessary files and setup temporary directory."""
    pdb_filepath = download_file(PDB_STRUCTURE_FILEPATH, tmp_dir)
    mrc_filepath = download_file(SIMULATED_MRC_FILEPATH, tmp_dir)
    dqe_filepath = download_file(DQE_STARFILE_FILEPATH, tmp_dir)

    return pdb_filepath, mrc_filepath, dqe_filepath


def get_simulation_kwargs(pdb_filepath: str, dqe_filepath: str) -> dict:
    """Get simulation keyword arguments for the test."""
    # Base parameters
    kwargs = {
        "beam_energy_kev": 300,
        "sim_pixel_spacing": 0.95,
        "sim_volume_shape": (400, 400, 400),
        "requested_upsampling": -1,
        "apply_dose_weighting": True,
        "dose_start": 0,
        "dose_end": 50,
        "dose_filter_modify_signal": "None",
        "dose_filter_critical_bfactor": -1.0,
        "apply_dqe": True,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }

    # Load in atom data from pdb file
    atom_positions_zyx, atom_ids, atom_b_factors = load_model(pdb_filepath)
    atom_positions_zyx, atom_ids, atom_b_factors = remove_hydrogens(
        atom_positions_zyx, atom_ids, atom_b_factors
    )
    atom_b_factors *= 0.25  # NOTE: Scaling to match cisTEM b-factor scaling

    kwargs["atom_positions_zyx"] = atom_positions_zyx
    kwargs["atom_ids"] = atom_ids
    kwargs["atom_b_factors"] = atom_b_factors

    # Load in DQE data from star file
    mtf_frequencies, mtf_amplitudes = read_mtf(dqe_filepath)
    kwargs["mtf_frequencies"] = mtf_frequencies
    kwargs["mtf_amplitudes"] = mtf_amplitudes

    return kwargs


def is_ci() -> bool:
    """Check if running in CI environment."""
    return os.environ.get("CI", "false").lower() == "true"


@pytest.mark.skipif(
    is_ci(),
    reason="Skip on CI due to memory constraints. Run locally or on large runner only.",
)
def test_simulate3d():
    """Do the simulation and compare to the reference mrc file."""
    pdb_filepath, mrc_filepath, dqe_filepath = setup_simulation()

    # Ensure files are downloaded
    assert os.path.exists(pdb_filepath), f"File not found: {pdb_filepath}"
    assert os.path.exists(mrc_filepath), f"File not found: {mrc_filepath}"
    assert os.path.exists(dqe_filepath), f"File not found: {dqe_filepath}"

    simulate3d_kwargs = get_simulation_kwargs(pdb_filepath, dqe_filepath)

    # Run the simulation
    simulated_volume = simulate3d(**simulate3d_kwargs)
    simulated_volume = simulated_volume.cpu().numpy()

    # Compare the simulated volume to the reference mrc file
    with (
        mrcfile.open(mrc_filepath) as ref_mrc,
    ):
        assert simulated_volume.shape == ref_mrc.data.shape
        assert np.allclose(simulated_volume, ref_mrc.data, atol=1e-3)
