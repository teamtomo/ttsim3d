"""Simple run script."""

from ttsim3d.input_models import SimulationParams
from ttsim3d.simulate3d import simulate3d


def main() -> None:
    """A test function to run the simulate3d function from the ttsim3d package."""
    params = SimulationParams(
        pdb_filename="/Users/josh/git/2dtm_tests/simulator/parsed_6Q8Y_whole_LSU_match3.pdb",
        output_filename="/Users/josh/git/2dtm_tests/simulator/simulated_6Q8Y_whole_LSU_match3.mrc",
        mtf_filename="/Users/josh/git/2dtm_tests/simulator/mtf_k2_300kV.star",
        sim_volume_shape=(400, 400, 400),
        sim_pixel_spacing=0.95,
        num_frames=50,
        fluence_per_frame=1,
        beam_energy_kev=300,
        dose_weighting=True,
        dose_B=-1,
        apply_dqe=True,
        b_scaling=0.5,
        added_B=0.0,
        upsampling=-1,
        gpu_id=-999,
        modify_signal=1,
    )

    simulate3d(**params.model_dump())


if __name__ == "__main__":
    main()
