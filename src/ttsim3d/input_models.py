"""Pydantic models for input parameters."""

from pydantic import BaseModel


class SimulationParams(BaseModel):
    """Parameters for the simulation."""

    pdb_filename: str
    output_filename: str
    sim_volume_shape: tuple[int, int, int] = (400, 400, 400)
    sim_pixel_spacing: float = 0.95
    num_frames: int = 50
    fluence_per_frame: float = 1
    beam_energy_kev: int = 300
    dose_weighting: bool = True
    dose_B: float = -1
    apply_dqe: bool = True
    mtf_filename: str
    b_scaling: float = 0.5
    added_B: float = 0.0
    upsampling: int = -1
    gpu_id: int = -999
    modify_signal: int = 1
