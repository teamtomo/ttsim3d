"""Comparison of simulation to static simulated .mrc file.

NOTE: If the core algorithm for the simulation changes, then this test will
fail and the reference "good" simulated structure will need to be updated.
"""

import os
from pathlib import Path

import mrcfile
import numpy as np
import requests

from ttsim3d import simulate3d

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


SIMULATION_PARAMS = {
    "pdb_filename": tmp_dir / "parsed_6Q8Y_whole_LSU_match3.pdb",
    "output_filename": tmp_dir / "parsed_6Q8Y_whole_LSU_match3_simulate_test.mrc",
    "sim_volume_shape": (400, 400, 400),  # in voxels
    "sim_pixel_spacing": 0.95,  # in Angstroms
    "num_frames": 50,
    "fluence_per_frame": 1,  # in e-/A^2
    "beam_energy_kev": 300,  # in keV
    "dose_weighting": True,
    "dose_B": -1,
    "apply_dqe": True,
    "mtf_filename": tmp_dir / "mtf_k2_300kV.star",
    "b_scaling": 1.0,
    "added_B": 0.0,
    "upsampling": -1,  # -1 is calculate automatically
    "gpu_id": -999,  # -999 cpu, -1 auto, 0 = gpuid
    "modify_signal": 1,
}


def download_file(url: str, dest_folder: str) -> str:
    """Download a file from a URL to a destination folder."""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    filename = url.split("/")[-1].split("?")[0]
    file_path = os.path.join(dest_folder, filename)

    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad status codes
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    with open(file_path, "wb") as file:
        for i, data in enumerate(response.iter_content(block_size)):
            file.write(data)
            # Print progress for every 1 MB
            if i % (1024 // block_size) == 0:
                print(f"Downloaded {i * block_size // 1024} MB of {total_size // 1024} MB")

    print(f"Downloaded {file_path}")
    return file_path


def setup_simulation():
    """Download necessary files and setup temporary directory."""
    print(this_path)
    print(this_dir)
    print(tmp_dir)

    pdb_filepath = download_file(PDB_STRUCTURE_FILEPATH, tmp_dir)
    mrc_filepath = download_file(SIMULATED_MRC_FILEPATH, tmp_dir)
    dqe_filepath = download_file(DQE_STARFILE_FILEPATH, tmp_dir)

    return pdb_filepath, mrc_filepath, dqe_filepath


def test_simulate3d():
    """Do the simulation and compare to the reference mrc file."""
    pdb_filepath, mrc_filepath, dqe_filepath = setup_simulation()

    # Ensure files are downloaded
    assert os.path.exists(pdb_filepath), f"File not found: {pdb_filepath}"
    assert os.path.exists(mrc_filepath), f"File not found: {mrc_filepath}"
    assert os.path.exists(dqe_filepath), f"File not found: {dqe_filepath}"

    # Run the simulation
    simulate3d.simulate3d(**SIMULATION_PARAMS)

    # Compare the simulated mrc file to the reference
    with (
        mrcfile.open(SIMULATION_PARAMS["output_filename"]) as sim_mrc,
        mrcfile.open(mrc_filepath) as ref_mrc,
    ):
        assert sim_mrc.data.shape == ref_mrc.data.shape
        assert np.allclose(sim_mrc.data, ref_mrc.data, atol=1e-3)
