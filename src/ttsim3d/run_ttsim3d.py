"""Simple run script."""

from ttsim3d.simulate3d import simulate3d


def main() -> None:
    """A test function to run the simulate3d function from the ttsim3d package."""
    simulate3d(
        pdb_filename="/Users/josh/git/2dtm_tests/simulator/parsed_6Q8Y_whole_LSU_match3.pdb",
        output_filename="/Users/josh/git/2dtm_tests/simulator/simulated_6Q8Y_whole_LSU_match3.mrc",
        sim_volume_shape=(400, 400, 400),
        sim_pixel_spacing=0.95,
        num_frames=50,
        fluence_per_frame=1,
        beam_energy_kev=300,
        dose_weighting=True,
        dose_B=-1,
        apply_dqe=True,
        mtf_filename="/Users/josh/git/2dtm_tests/simulator/mtf_k2_300kV.star",
        b_scaling=0.5,
        added_B=0.0,
        upsampling=-1,
        gpu_id=-999,
        modify_signal=1,  # This is how to apply the dose weighting.
    )


if __name__ == "__main__":
    main()
